# SPDX-License-Identifier: Apache-2.0
"""Publish ClinVar split streams and placed-window leakage evidence from a store."""

from __future__ import annotations

import argparse
import hashlib
import heapq
import importlib
import json
import os
import re
import shutil
import stat
import subprocess
import sys
import tempfile
from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final, Protocol, TextIO

from geno_lewm.data._membership_store_publish import _publish_directory_noreplace
from geno_lewm.data.builder import WindowContext
from geno_lewm.data.membership_store import MembershipStore, MembershipStoreHoldoutPolicy
from geno_lewm.data.variant_identity import canonicalize_chromosome
from geno_lewm.encoder.windowing import canonicalize_dna
from geno_lewm.errors import GenoLeWMError, InputError, RuntimeSetupError, exit_code_for
from geno_lewm.provenance import canonical_json_sha256, sha256_file
from geno_lewm.provenance.hashing import canonical_json_bytes, looks_like_sha256

SCHEMA_VERSION: Final = "geno-lewm.membership-split-evidence.v1"
DEFAULT_REPORT_SCHEMA: Final = (
    Path(__file__).resolve().parents[2]
    / "configs"
    / "data_v03"
    / "membership-split-evidence.schema.json"
)
_REPORT_RELATIVE_PATH: Final = Path("evidence/membership-split-evidence.json")
_SCHEMA_RELATIVE_PATH: Final = Path("contract/membership-split-evidence.schema.json")
_CHECKSUMS_RELATIVE_PATH: Final = Path("SHA256SUMS")
_PLACED_WINDOW_FIELDS: Final = frozenset(
    {
        "record_id",
        "source",
        "variant_source",
        "chrom",
        "start_bp",
        "end_bp",
        "sequence",
        "variant_count",
    }
)
_ROLES: Final = ("validation", "evaluation")
_CLASSES: Final = ("B", "LB", "LP", "P")
_COMMIT: Final = re.compile(r"[0-9a-f]{40}")
_ARTIFACT_ID: Final = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*")
_REPOSITORY: Final = re.compile(r"[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+")
_CONTAINER_IMAGE: Final = re.compile(r"[^@\s]+@sha256:[0-9a-f]{64}")
_SAFE_RELATIVE_PATH: Final = re.compile(r"[A-Za-z0-9._-]+(?:/[A-Za-z0-9._-]+)*")
_DATASET_SPLIT: Final = "train_placed_gnomad_common"
_GENERATED_BY: Final = "tools.data.v03_membership_splits"


class _Digest(Protocol):
    def update(self, data: bytes, /) -> object: ...


@dataclass(frozen=True, slots=True)
class _PlacedWindow:
    record_id: str
    source: str
    variant_source: str
    chrom: str
    start_bp: int
    end_bp: int
    sequence: str
    variant_count: int

    @property
    def identity(self) -> dict[str, object]:
        return {
            "record_id": self.record_id,
            "chrom": self.chrom,
            "start_bp": self.start_bp,
            "end_bp": self.end_bp,
            "window_sha256": "sha256:" + hashlib.sha256(self.sequence.encode()).hexdigest(),
        }

    @property
    def context(self) -> WindowContext:
        return WindowContext(
            record_id=self.record_id,
            source=self.source,
            sequence=self.sequence,
            start_bp=self.start_bp,
            chrom=self.chrom,
        )


def build_membership_splits(
    *,
    store_dir: Path,
    placed_windows_jsonl: Path,
    dataset_manifest_json: Path,
    output_dir: Path,
    artifact_id: str,
    membership_repository: str,
    membership_revision: str,
    membership_artifact_path: str,
    training_windows_repository: str,
    training_windows_revision: str,
    training_windows_artifact_path: str,
    expected_store_content_identity: str,
    expected_store_physical_identity: str,
    expected_store_rowset_sha256: str,
    expected_dataset_manifest_sha256: str,
    expected_dataset_snapshot_id: str,
    expected_placed_windows_sha256: str,
    expected_placed_windows_size_bytes: int,
    expected_placed_windows_record_count: int,
    producer_git_commit: str,
    container_image: str,
    sample_seed: int = 0,
    sample_size: int = 128,
    allow_fixture: bool = False,
    report_schema_path: Path | None = None,
) -> dict[str, object]:
    """Build and atomically publish checksum-closed split evidence.

    The membership store is fully verified before use. The placed-window file is
    captured once through a no-follow descriptor, then every derived check reads
    only those captured bytes.
    """
    artifact_id = _require_artifact_id(artifact_id)
    membership_repository = _require_repository(membership_repository, "membership_repository")
    membership_revision = _require_commit(membership_revision, "membership_revision")
    membership_artifact_path = _require_artifact_path(
        membership_artifact_path, "membership_artifact_path"
    )
    training_windows_repository = _require_repository(
        training_windows_repository, "training_windows_repository"
    )
    training_windows_revision = _require_commit(
        training_windows_revision, "training_windows_revision"
    )
    training_windows_artifact_path = _require_artifact_path(
        training_windows_artifact_path, "training_windows_artifact_path"
    )
    producer_git_commit = _require_commit(producer_git_commit, "producer_git_commit")
    container_image = _require_container_image(container_image)
    expected_dataset_snapshot_id = _require_text(
        expected_dataset_snapshot_id, "expected_dataset_snapshot_id"
    )
    for field, value in (
        ("expected_store_content_identity", expected_store_content_identity),
        ("expected_store_physical_identity", expected_store_physical_identity),
        ("expected_store_rowset_sha256", expected_store_rowset_sha256),
        ("expected_dataset_manifest_sha256", expected_dataset_manifest_sha256),
        ("expected_placed_windows_sha256", expected_placed_windows_sha256),
    ):
        if not looks_like_sha256(value):
            raise InputError(f"{field} must be a sha256 digest")
    _require_positive_int(expected_placed_windows_size_bytes, "expected_placed_windows_size_bytes")
    _require_positive_int(
        expected_placed_windows_record_count, "expected_placed_windows_record_count"
    )
    _require_nonnegative_int(sample_seed, "sample_seed")
    _require_positive_int(sample_size, "sample_size")
    if not isinstance(allow_fixture, bool):
        raise InputError("allow_fixture must be a boolean")
    schema_source = (
        DEFAULT_REPORT_SCHEMA if report_schema_path is None else Path(report_schema_path)
    )
    if not allow_fixture and schema_source.resolve() != DEFAULT_REPORT_SCHEMA.resolve():
        raise InputError("official membership split publication requires the tracked report schema")
    invocation_verified = False
    if not allow_fixture:
        _verify_producer_invocation(
            producer_git_commit=producer_git_commit,
            container_image=container_image,
        )
        invocation_verified = True

    output = Path(output_dir).absolute()
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists() or output.is_symlink():
        raise InputError(
            "membership split output already exists",
            details={"path": str(output)},
            remediation="choose a new immutable artifact directory",
        )
    temporary = Path(tempfile.mkdtemp(prefix=f".{output.name}.tmp-", dir=output.parent))
    windows_capture = temporary / ".placed-windows.capture"
    manifest_capture = temporary / ".dataset-manifest.capture"
    try:
        captured_sha256, captured_size = _capture_regular_file(
            Path(placed_windows_jsonl),
            windows_capture,
            "placed-window artifact",
        )
        manifest_sha256, manifest_size = _capture_regular_file(
            Path(dataset_manifest_json),
            manifest_capture,
            "dataset manifest",
        )
        if captured_sha256 != expected_placed_windows_sha256:
            raise InputError(
                "placed-window SHA-256 does not match the expected identity",
                details={
                    "expected": expected_placed_windows_sha256,
                    "observed": captured_sha256,
                },
            )
        if captured_size != expected_placed_windows_size_bytes:
            raise InputError(
                "placed-window size does not match the expected identity",
                details={
                    "expected": expected_placed_windows_size_bytes,
                    "observed": captured_size,
                },
            )
        if manifest_sha256 != expected_dataset_manifest_sha256:
            raise InputError(
                "dataset-manifest SHA-256 does not match the expected identity",
                details={
                    "expected": expected_dataset_manifest_sha256,
                    "observed": manifest_sha256,
                },
            )
        dataset_snapshot_id = _verify_dataset_manifest_binding(
            manifest_capture,
            expected_snapshot_id=expected_dataset_snapshot_id,
            artifact_path=training_windows_artifact_path,
            artifact_sha256=captured_sha256,
            artifact_size_bytes=captured_size,
            artifact_record_count=expected_placed_windows_record_count,
        )

        schema_bytes = _read_regular_file(schema_source, "membership split report schema")
        try:
            schema_payload = json.loads(schema_bytes)
        except json.JSONDecodeError as exc:
            raise InputError("membership split report schema JSON is invalid") from exc
        if not isinstance(schema_payload, Mapping):
            raise InputError("membership split report schema must be a JSON object")

        with MembershipStore.open(Path(store_dir), verify=True) as store:
            manifest = store.manifest
            _require_store_identity(
                manifest.content_identity,
                expected_store_content_identity,
                "content",
            )
            _require_store_identity(
                manifest.physical_identity,
                expected_store_physical_identity,
                "physical",
            )
            _require_store_identity(
                manifest.rowset_sha256,
                expected_store_rowset_sha256,
                "rowset",
            )
            evidence_profile = manifest.snapshot_lineage.evidence_profile
            if allow_fixture and evidence_profile != "synthetic_fixture":
                raise InputError(
                    "fixture opt-in is valid only for synthetic fixture lineage",
                    details={"observed": evidence_profile},
                )
            if not allow_fixture and evidence_profile != "official":
                raise InputError(
                    "membership split publication requires official lineage evidence",
                    details={
                        "observed": evidence_profile,
                    },
                )

            stream_reports = {
                role: _write_split_stream(store, role=role, root=temporary) for role in _ROLES
            }
            _reconcile_stream_counts(stream_reports, manifest.clinvar_class_role_counts)
            audit, record_count, training_chromosomes = _audit_training_windows(
                store,
                windows_capture,
                sample_seed=sample_seed,
                sample_size=sample_size,
            )
            if record_count != expected_placed_windows_record_count:
                raise InputError(
                    "placed-window record count does not match the expected identity",
                    details={
                        "expected": expected_placed_windows_record_count,
                        "observed": record_count,
                    },
                )

            report: dict[str, object] = {
                "$schema": "../contract/membership-split-evidence.schema.json",
                "schema_version": SCHEMA_VERSION,
                "artifact_id": artifact_id,
                "assembly": "GRCh38",
                "ok": True,
                "producer": {
                    "generated_by": _GENERATED_BY,
                    "git_commit": producer_git_commit,
                    "container_image": container_image,
                    "invocation_verified": invocation_verified,
                },
                "membership_store": {
                    "repository": membership_repository,
                    "revision": membership_revision,
                    "artifact_path": membership_artifact_path,
                    "artifact_id": manifest.artifact_id,
                    "content_identity": manifest.content_identity,
                    "physical_identity": manifest.physical_identity,
                    "rowset_sha256": manifest.rowset_sha256,
                    "lineage": {
                        "lineage_id": manifest.snapshot_lineage.lineage_id,
                        "sha256": manifest.snapshot_lineage.sha256,
                        "candidate_snapshot_id": manifest.snapshot_lineage.candidate_snapshot_id,
                        "evidence_profile": manifest.snapshot_lineage.evidence_profile,
                    },
                    "chromosome_roles": manifest.chromosome_roles.to_dict(),
                },
                "training_windows": {
                    "source": {
                        "repository": training_windows_repository,
                        "revision": training_windows_revision,
                        "artifact_path": training_windows_artifact_path,
                    },
                    "sha256": captured_sha256,
                    "size_bytes": captured_size,
                    "record_count": record_count,
                    "assembly": "GRCh38",
                    "role": "train",
                    "split": _DATASET_SPLIT,
                    "chromosomes": list(training_chromosomes),
                    "dataset_manifest": {
                        "path": "dataset_manifest.json",
                        "sha256": manifest_sha256,
                        "size_bytes": manifest_size,
                        "snapshot_id": dataset_snapshot_id,
                    },
                    "record_fields": [
                        "record_id",
                        "source",
                        "variant_source",
                        "chrom",
                        "start_bp",
                        "end_bp",
                        "sequence",
                        "variant_count",
                    ],
                },
                "streams": stream_reports,
                "audits": audit,
                "claim_boundary": {
                    "variant_membership": True,
                    "phased_haplotype_membership": False,
                    "released_v03_snapshot": False,
                    "publication_eligible": (
                        evidence_profile == "official" and invocation_verified
                    ),
                    "limitations": [
                        "This evidence covers deterministic unphased variant memberships and placed-window nonintersection only.",
                        "It does not establish phased-haplotype membership, a released v0.3 snapshot, dataset representativeness, model quality, benchmark performance, or clinical validity.",
                    ],
                },
            }
            _validate_report(report, schema_payload)

        windows_capture.unlink()
        manifest_capture.unlink()
        schema_target = temporary / _SCHEMA_RELATIVE_PATH
        schema_target.parent.mkdir(parents=True, exist_ok=True)
        schema_target.write_bytes(schema_bytes)
        report_target = temporary / _REPORT_RELATIVE_PATH
        report_target.parent.mkdir(parents=True, exist_ok=True)
        _write_json(report_target, report)
        _write_checksums(temporary)
        _fsync_tree(temporary)
        if output.exists() or output.is_symlink():
            raise InputError(
                "membership split output appeared before publication",
                details={"path": str(output)},
            )
        _publish_directory_noreplace(temporary, output)
        _fsync_directory(output.parent)
        return report
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise


def _write_split_stream(
    store: MembershipStore,
    *,
    role: str,
    root: Path,
) -> dict[str, object]:
    chromosomes = tuple(getattr(store.manifest.chromosome_roles, role))
    if len(chromosomes) != 1:
        raise InputError(
            "membership split export requires exactly one chromosome per held role",
            details={"role": role, "chromosomes": list(chromosomes)},
        )
    chrom = chromosomes[0]
    relative_root = Path("splits") / role
    stem = f"clinvar-chr{chrom}"
    labels_relative = relative_root / f"{stem}.labels.jsonl"
    vcf_relative = relative_root / f"{stem}.vcf"
    labels_path = root / labels_relative
    vcf_path = root / vcf_relative
    labels_path.parent.mkdir(parents=True, exist_ok=True)
    class_counts = dict.fromkeys(_CLASSES, 0)
    records = 0
    previous_key: str | None = None
    keyset = hashlib.sha256()
    with (
        labels_path.open("w", encoding="utf-8", newline="\n") as labels,
        vcf_path.open("w", encoding="utf-8", newline="\n") as vcf,
    ):
        _write_vcf_header(vcf)
        for labeled in store.iter_labeled_clinvar(role):
            membership = labeled.membership
            variant = membership.variant
            if membership.role != role or variant.chrom != chrom:
                raise InputError("membership label stream drifted from its chromosome role")
            key = variant.key
            if previous_key is not None and key == previous_key:
                raise InputError("membership label stream contains a duplicate variant")
            previous_key = key
            _update_framed_digest(keyset, key.encode())
            clinical_significance = labeled.clinical_significance
            class_counts[clinical_significance] += 1
            row = {
                "chrom": variant.chrom,
                "pos": variant.pos,
                "ref": variant.ref,
                "alt": variant.alt,
                "clinical_significance": clinical_significance,
            }
            labels.write(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n")
            label = 1 if labeled.is_pathogenic else 0
            vcf.write(
                f"{variant.chrom}\t{variant.pos}\t.\t{variant.ref}\t{variant.alt}"
                f"\t.\tPASS\tCLNSIG={clinical_significance};ROLE={role};LABEL={label}\n"
            )
            records += 1
    if records == 0 or sum(class_counts.values()) != records:
        raise InputError("membership label stream must be non-empty and internally consistent")
    binary_counts = {
        "negative": class_counts["B"] + class_counts["LB"],
        "positive": class_counts["LP"] + class_counts["P"],
    }
    return {
        "role": role,
        "chromosome": chrom,
        "record_count": records,
        "class_counts": class_counts,
        "binary_counts": binary_counts,
        "keyset_sha256": "sha256:" + keyset.hexdigest(),
        "labels_jsonl": _file_identity(labels_relative, labels_path),
        "vcf": _file_identity(vcf_relative, vcf_path),
    }


def _write_vcf_header(handle: TextIO) -> None:
    handle.write("##fileformat=VCFv4.3\n")
    handle.write("##reference=GRCh38\n")
    handle.write("##source=GenoLeWM-v0.3-membership-splits\n")
    handle.write('##INFO=<ID=CLNSIG,Number=1,Type=String,Description="Normalized ClinVar class">\n')
    handle.write('##INFO=<ID=ROLE,Number=1,Type=String,Description="Membership role">\n')
    handle.write('##INFO=<ID=LABEL,Number=1,Type=Integer,Description="Binary target">\n')
    handle.write("#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\n")


def _reconcile_stream_counts(
    reports: Mapping[str, Mapping[str, object]],
    expected: Mapping[str, Mapping[str, int]],
) -> None:
    for role in _ROLES:
        observed = reports[role]["class_counts"]
        if not isinstance(observed, Mapping):
            raise InputError("membership stream class counts are invalid")
        expected_role = {label: expected[label][role] for label in _CLASSES}
        if dict(observed) != expected_role:
            raise InputError(
                "membership label stream class counts drifted from the store manifest",
                details={"role": role, "expected": expected_role, "observed": dict(observed)},
            )


def _audit_training_windows(
    store: MembershipStore,
    captured_path: Path,
    *,
    sample_seed: int,
    sample_size: int,
) -> tuple[dict[str, object], int, tuple[str, ...]]:
    policy = MembershipStoreHoldoutPolicy(store)
    seen_record_ids: set[str] = set()
    seen_intervals: set[tuple[str, int, int]] = set()
    sample_heap: list[tuple[int, str, str, _PlacedWindow]] = []
    chromosomes: set[str] = set()
    record_count = 0
    policy_exclusions = 0
    indexed_overlaps = 0
    for window in _iter_placed_windows(captured_path):
        if window.record_id in seen_record_ids:
            raise InputError(
                "placed-window record_id is duplicated",
                details={"record_id": window.record_id},
            )
        interval = (window.chrom, window.start_bp, window.end_bp)
        if interval in seen_intervals:
            raise InputError(
                "placed-window interval is duplicated",
                details={"chrom": window.chrom, "start_bp": window.start_bp},
            )
        seen_record_ids.add(window.record_id)
        seen_intervals.add(interval)
        chromosomes.add(window.chrom)
        try:
            role = store.manifest.chromosome_roles.role_for(window.chrom)
        except InputError as exc:
            raise InputError(
                "placed training window is outside the membership chromosome roles",
                details={"record_id": window.record_id, "chrom": window.chrom},
            ) from exc
        if role != "train":
            raise InputError(
                "placed training window belongs to a held-out chromosome",
                details={"record_id": window.record_id, "chrom": window.chrom, "role": role},
            )
        if policy.excludes_window(window.context):
            policy_exclusions += 1
        if store.overlaps_interval(
            window.chrom,
            start_bp=window.start_bp,
            end_bp=window.end_bp,
            roles=_ROLES,
        ):
            indexed_overlaps += 1
        _offer_sample(
            sample_heap,
            window,
            sample_seed=sample_seed,
            sample_size=sample_size,
        )
        record_count += 1
    if record_count == 0:
        raise InputError("placed training-window artifact must contain at least one row")
    if policy_exclusions or indexed_overlaps:
        raise InputError(
            "placed training windows intersect the published holdout membership",
            details={
                "policy_exclusions": policy_exclusions,
                "indexed_overlaps": indexed_overlaps,
            },
        )

    sampled = sorted(
        ((digest, window) for _priority, digest, _record_id, window in sample_heap),
        key=lambda item: item[0],
    )
    sample_payload: list[dict[str, object]] = []
    sample_exclusions = 0
    sample_overlaps = 0
    for digest, window in sampled:
        if policy.excludes_window(window.context):
            sample_exclusions += 1
        if store.overlaps_interval(
            window.chrom,
            start_bp=window.start_bp,
            end_bp=window.end_bp,
            roles=_ROLES,
        ):
            sample_overlaps += 1
        sample_payload.append({"priority_sha256": "sha256:" + digest, **window.identity})
    if sample_exclusions or sample_overlaps:
        raise InputError("deterministic placed-window sample intersects holdouts")
    observed_sample_size = len(sample_payload)
    return (
        {
            "exhaustive": {
                "windows_scanned": record_count,
                "policy_exclusions": policy_exclusions,
                "indexed_overlaps": indexed_overlaps,
                "status": "passed",
            },
            "deterministic_sample": {
                "algorithm": "sha256-priority-v1",
                "seed": sample_seed,
                "requested_size": sample_size,
                "observed_size": observed_sample_size,
                "sample_digest": canonical_json_sha256(sample_payload),
                "policy_exclusions": sample_exclusions,
                "indexed_overlaps": sample_overlaps,
                "status": "passed",
            },
        },
        record_count,
        tuple(sorted(chromosomes, key=int)),
    )


def _offer_sample(
    heap: list[tuple[int, str, str, _PlacedWindow]],
    window: _PlacedWindow,
    *,
    sample_seed: int,
    sample_size: int,
) -> None:
    digest = hashlib.sha256(
        str(sample_seed).encode() + b"\x00" + canonical_json_bytes(window.identity)
    ).hexdigest()
    entry = (-int(digest, 16), digest, window.record_id, window)
    if len(heap) < sample_size:
        heapq.heappush(heap, entry)
        return
    if entry[0] > heap[0][0]:
        heapq.heapreplace(heap, entry)


def _iter_placed_windows(path: Path) -> Iterator[_PlacedWindow]:
    with path.open(encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            if not line.strip():
                raise InputError(
                    "placed-window JSONL contains a blank line",
                    details={"line": line_no},
                )
            try:
                raw: object = json.loads(line, object_pairs_hook=_reject_duplicate_pairs)
            except json.JSONDecodeError as exc:
                raise InputError(
                    "placed-window JSONL row is invalid JSON",
                    details={"line": line_no, "column": exc.colno},
                ) from exc
            if not isinstance(raw, Mapping):
                raise InputError(
                    "placed-window JSONL row must be an object",
                    details={"line": line_no},
                )
            if set(raw) != _PLACED_WINDOW_FIELDS:
                raise InputError(
                    "placed-window row keys do not match the closed schema",
                    details={
                        "line": line_no,
                        "missing": sorted(_PLACED_WINDOW_FIELDS - set(raw)),
                        "unexpected": sorted(set(raw) - _PLACED_WINDOW_FIELDS),
                    },
                )
            record_id = _require_text(raw.get("record_id"), "placed-window record_id")
            source = _require_text(raw.get("source"), "placed-window source")
            variant_source = _require_text(
                raw.get("variant_source"), "placed-window variant_source"
            )
            chrom = _require_text(raw.get("chrom"), "placed-window chrom")
            start_bp = _require_nonnegative_int(raw.get("start_bp"), "placed-window start_bp")
            end_bp = _require_positive_int(raw.get("end_bp"), "placed-window end_bp")
            variant_count = _require_positive_int(
                raw.get("variant_count"), "placed-window variant_count"
            )
            sequence = _require_text(raw.get("sequence"), "placed-window sequence")
            if canonicalize_chromosome(chrom) != chrom:
                raise InputError(
                    "placed-window chromosome must use its canonical unprefixed spelling",
                    details={"line": line_no, "chrom": chrom},
                )
            canonical = canonicalize_dna(sequence)
            if canonical != sequence:
                raise InputError("placed-window sequence must already be canonical uppercase DNA")
            if end_bp <= start_bp or end_bp - start_bp != len(sequence):
                raise InputError(
                    "placed-window interval does not match its sequence length",
                    details={
                        "line": line_no,
                        "start_bp": start_bp,
                        "end_bp": end_bp,
                        "sequence_length": len(sequence),
                    },
                )
            yield _PlacedWindow(
                record_id=record_id,
                source=source,
                variant_source=variant_source,
                chrom=chrom,
                start_bp=start_bp,
                end_bp=end_bp,
                sequence=sequence,
                variant_count=variant_count,
            )


def _verify_dataset_manifest_binding(
    path: Path,
    *,
    expected_snapshot_id: str,
    artifact_path: str,
    artifact_sha256: str,
    artifact_size_bytes: int,
    artifact_record_count: int,
) -> str:
    try:
        raw: object = json.loads(
            path.read_text(encoding="utf-8"),
            object_pairs_hook=_reject_duplicate_pairs,
        )
    except OSError as exc:
        raise InputError("captured dataset manifest cannot be read") from exc
    except json.JSONDecodeError as exc:
        raise InputError("captured dataset manifest JSON is invalid") from exc
    if not isinstance(raw, Mapping):
        raise InputError("dataset manifest must be a JSON object")
    snapshot_id = _require_text(raw.get("snapshot_id"), "dataset manifest snapshot_id")
    if snapshot_id != expected_snapshot_id:
        raise InputError(
            "dataset manifest snapshot_id does not match the expected identity",
            details={"expected": expected_snapshot_id, "observed": snapshot_id},
        )
    files = raw.get("files")
    if not isinstance(files, list):
        raise InputError("dataset manifest files must be an array")
    matches: list[Mapping[str, object]] = []
    for item in files:
        if not isinstance(item, Mapping):
            raise InputError("dataset manifest file entries must be objects")
        if item.get("path") == artifact_path:
            matches.append(item)
    if len(matches) != 1:
        raise InputError(
            "dataset manifest must bind the placed-window artifact exactly once",
            details={"artifact_path": artifact_path, "matches": len(matches)},
        )
    binding = matches[0]
    expected = {
        "sha256": artifact_sha256,
        "size_bytes": artifact_size_bytes,
        "records": artifact_record_count,
        "split": _DATASET_SPLIT,
    }
    observed = {
        "sha256": binding.get("sha256"),
        "size_bytes": _require_positive_int(
            binding.get("size_bytes"), "dataset manifest placed-window size_bytes"
        ),
        "records": _require_positive_int(
            binding.get("records"), "dataset manifest placed-window records"
        ),
        "split": binding.get("split"),
    }
    if observed != expected:
        raise InputError(
            "dataset manifest placed-window record count or file identity does not match the captured artifact",
            details={"expected": expected, "observed": observed},
        )
    return snapshot_id


def _capture_regular_file(source: Path, destination: Path, label: str) -> tuple[str, int]:
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_BINARY", 0)
    try:
        descriptor = os.open(source, flags)
    except OSError as exc:
        raise InputError(
            f"{label} cannot be opened as a regular file",
            details={"path": str(source)},
        ) from exc
    digest = hashlib.sha256()
    size = 0
    try:
        metadata = os.fstat(descriptor)
        if not stat.S_ISREG(metadata.st_mode):
            raise InputError(f"{label} must be a regular file")
        with os.fdopen(descriptor, "rb", closefd=False) as reader, destination.open("xb") as writer:
            while chunk := reader.read(1 << 20):
                writer.write(chunk)
                digest.update(chunk)
                size += len(chunk)
            writer.flush()
            os.fsync(writer.fileno())
        if size != metadata.st_size:
            raise InputError(f"{label} changed while being captured")
    finally:
        os.close(descriptor)
    return "sha256:" + digest.hexdigest(), size


def _read_regular_file(path: Path, label: str) -> bytes:
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_BINARY", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError as exc:
        raise InputError(f"{label} cannot be opened", details={"path": str(path)}) from exc
    try:
        if not stat.S_ISREG(os.fstat(descriptor).st_mode):
            raise InputError(f"{label} must be a regular file")
        with os.fdopen(descriptor, "rb", closefd=False) as handle:
            return handle.read()
    finally:
        os.close(descriptor)


def _file_identity(relative: Path, path: Path) -> dict[str, object]:
    return {
        "path": relative.as_posix(),
        "sha256": sha256_file(path),
        "size_bytes": path.stat().st_size,
    }


def _write_checksums(root: Path) -> None:
    files = sorted(
        (
            path
            for path in root.rglob("*")
            if path.is_file() and path.relative_to(root) != _CHECKSUMS_RELATIVE_PATH
        ),
        key=lambda path: path.relative_to(root).as_posix(),
    )
    expected = {
        _REPORT_RELATIVE_PATH.as_posix(),
        _SCHEMA_RELATIVE_PATH.as_posix(),
        "splits/validation/clinvar-chr20.labels.jsonl",
        "splits/validation/clinvar-chr20.vcf",
        "splits/evaluation/clinvar-chr21.labels.jsonl",
        "splits/evaluation/clinvar-chr21.vcf",
    }
    observed = {path.relative_to(root).as_posix() for path in files}
    if observed != expected:
        raise InputError(
            "membership split output files do not match the closed layout",
            details={
                "missing": sorted(expected - observed),
                "unexpected": sorted(observed - expected),
            },
        )
    checksum_path = root / _CHECKSUMS_RELATIVE_PATH
    with checksum_path.open("x", encoding="utf-8", newline="\n") as handle:
        for path in files:
            relative = path.relative_to(root).as_posix()
            handle.write(f"{sha256_file(path).removeprefix('sha256:')}  {relative}\n")


def _write_json(path: Path, payload: Mapping[str, object]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _validate_report(report: Mapping[str, object], schema: Mapping[str, object]) -> None:
    try:
        jsonschema = importlib.import_module("jsonschema")
    except ImportError as exc:
        raise RuntimeSetupError(
            "membership split publication requires jsonschema",
            remediation="run this source tool with the dev dependency group installed",
        ) from exc
    validator_type: Any = jsonschema.Draft202012Validator
    try:
        validator_type.check_schema(schema)
        errors = sorted(
            validator_type(schema).iter_errors(report),
            key=lambda error: tuple(str(item) for item in error.absolute_path),
        )
    except Exception as exc:
        raise InputError("membership split report schema is invalid") from exc
    if errors:
        error = errors[0]
        location = ".".join(str(item) for item in error.absolute_path) or "$"
        raise InputError(
            "membership split report does not satisfy its bundled schema",
            details={"path": location, "error": error.message},
        )


def _fsync_tree(root: Path) -> None:
    binary = getattr(os, "O_BINARY", 0)
    for path in sorted(item for item in root.rglob("*") if item.is_file()):
        descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | binary)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
    directories = sorted(
        (item for item in root.rglob("*") if item.is_dir()),
        key=lambda item: len(item.parts),
        reverse=True,
    )
    for directory in (*directories, root):
        _fsync_directory(directory)


def _fsync_directory(path: Path) -> None:
    directory_flag = getattr(os, "O_DIRECTORY", None)
    if directory_flag is None:
        return
    descriptor = os.open(path, os.O_RDONLY | directory_flag)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _update_framed_digest(digest: _Digest, payload: bytes) -> None:
    digest.update(len(payload).to_bytes(8, "big"))
    digest.update(payload)


def _require_store_identity(observed: str, expected: str, kind: str) -> None:
    if observed != expected:
        raise InputError(
            f"membership store {kind} identity does not match the expected value",
            details={"expected": expected, "observed": observed},
        )


def _require_artifact_id(value: object) -> str:
    text = _require_text(value, "artifact_id")
    if _ARTIFACT_ID.fullmatch(text) is None:
        raise InputError("artifact_id is not canonical")
    return text


def _require_repository(value: object, field: str) -> str:
    text = _require_text(value, field)
    if _REPOSITORY.fullmatch(text) is None:
        raise InputError(f"{field} must be an owner/name repository")
    return text


def _require_commit(value: object, field: str) -> str:
    text = _require_text(value, field)
    if _COMMIT.fullmatch(text) is None:
        raise InputError(f"{field} must be an exact 40-hex commit")
    return text


def _require_container_image(value: object) -> str:
    text = _require_text(value, "container_image")
    if _CONTAINER_IMAGE.fullmatch(text) is None:
        raise InputError("container_image must be digest-pinned")
    return text


def _verify_producer_invocation(
    *,
    producer_git_commit: str,
    container_image: str,
) -> None:
    expected_container = os.environ.get("GENO_LEWM_VERIFIED_SPLIT_CONTAINER_IMAGE")
    if expected_container != container_image:
        raise InputError(
            "container image does not match the trusted launcher binding",
            details={"expected": expected_container, "observed": container_image},
        )
    repository_root = Path(__file__).resolve().parents[2]
    observed_commit = _git_output(repository_root, "rev-parse", "HEAD")
    if observed_commit != producer_git_commit:
        raise InputError(
            "producer Git commit does not match the checked-out source",
            details={"expected": producer_git_commit, "observed": observed_commit},
        )
    if _git_output(repository_root, "status", "--porcelain=v1", "--untracked-files=all"):
        raise InputError("membership split producer checkout must be clean")
    origin = _git_output(repository_root, "remote", "get-url", "origin")
    if origin != "https://github.com/AbdelStark/GenoLeWM.git":
        raise InputError(
            "membership split producer origin is not canonical",
            details={"origin": origin},
        )
    for relative in (
        "tools/data/v03_membership_splits.py",
        "configs/data_v03/membership-split-evidence.schema.json",
    ):
        _git_output(repository_root, "cat-file", "-e", f"{producer_git_commit}:{relative}")


def _git_output(repository_root: Path, *arguments: str) -> str:
    try:
        completed = subprocess.run(
            ("git", "-C", str(repository_root), *arguments),
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError) as exc:
        raise InputError(
            "membership split producer Git state cannot be verified",
            details={"command": ["git", *arguments]},
        ) from exc
    return completed.stdout.strip()


def _require_artifact_path(value: object, field: str) -> str:
    text = _require_text(value, field)
    if _SAFE_RELATIVE_PATH.fullmatch(text) is None or any(
        part in {".", ".."} for part in text.split("/")
    ):
        raise InputError(f"{field} must be a safe relative POSIX path")
    return text


def _require_text(value: object, field: str) -> str:
    if not isinstance(value, str) or not value:
        raise InputError(f"{field} must be a non-empty string")
    return value


def _require_nonnegative_int(value: object, field: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise InputError(f"{field} must be a non-negative integer")
    return value


def _require_positive_int(value: object, field: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise InputError(f"{field} must be a positive integer")
    return value


def _reject_duplicate_pairs(pairs: list[tuple[str, object]]) -> dict[str, object]:
    payload: dict[str, object] = {}
    for key, value in pairs:
        if key in payload:
            raise InputError("duplicate JSON key is not allowed", details={"key": key})
        payload[key] = value
    return payload


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--store-dir", type=Path, required=True)
    parser.add_argument("--placed-windows-jsonl", type=Path, required=True)
    parser.add_argument("--dataset-manifest-json", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--artifact-id", required=True)
    parser.add_argument("--membership-repository", required=True)
    parser.add_argument("--membership-revision", required=True)
    parser.add_argument("--membership-artifact-path", required=True)
    parser.add_argument("--training-windows-repository", required=True)
    parser.add_argument("--training-windows-revision", required=True)
    parser.add_argument("--training-windows-artifact-path", required=True)
    parser.add_argument("--expected-store-content-identity", required=True)
    parser.add_argument("--expected-store-physical-identity", required=True)
    parser.add_argument("--expected-store-rowset-sha256", required=True)
    parser.add_argument("--expected-dataset-manifest-sha256", required=True)
    parser.add_argument("--expected-dataset-snapshot-id", required=True)
    parser.add_argument("--expected-placed-windows-sha256", required=True)
    parser.add_argument("--expected-placed-windows-size-bytes", type=int, required=True)
    parser.add_argument("--expected-placed-windows-record-count", type=int, required=True)
    parser.add_argument("--producer-git-commit", required=True)
    parser.add_argument("--container-image", required=True)
    parser.add_argument("--sample-seed", type=int, default=0)
    parser.add_argument("--sample-size", type=int, default=128)
    parser.add_argument("--report-schema-path", type=Path, default=None)
    parser.add_argument("--allow-fixture", action="store_true", help=argparse.SUPPRESS)
    return parser


def main(argv: list[str] | None = None) -> int:
    """Run the membership split publisher."""
    args = _parser().parse_args(argv)
    try:
        report = build_membership_splits(
            store_dir=args.store_dir,
            placed_windows_jsonl=args.placed_windows_jsonl,
            dataset_manifest_json=args.dataset_manifest_json,
            output_dir=args.output_dir,
            artifact_id=args.artifact_id,
            membership_repository=args.membership_repository,
            membership_revision=args.membership_revision,
            membership_artifact_path=args.membership_artifact_path,
            training_windows_repository=args.training_windows_repository,
            training_windows_revision=args.training_windows_revision,
            training_windows_artifact_path=args.training_windows_artifact_path,
            expected_store_content_identity=args.expected_store_content_identity,
            expected_store_physical_identity=args.expected_store_physical_identity,
            expected_store_rowset_sha256=args.expected_store_rowset_sha256,
            expected_dataset_manifest_sha256=args.expected_dataset_manifest_sha256,
            expected_dataset_snapshot_id=args.expected_dataset_snapshot_id,
            expected_placed_windows_sha256=args.expected_placed_windows_sha256,
            expected_placed_windows_size_bytes=args.expected_placed_windows_size_bytes,
            expected_placed_windows_record_count=args.expected_placed_windows_record_count,
            producer_git_commit=args.producer_git_commit,
            container_image=args.container_image,
            sample_seed=args.sample_seed,
            sample_size=args.sample_size,
            allow_fixture=args.allow_fixture,
            report_schema_path=args.report_schema_path,
        )
    except (GenoLeWMError, OSError) as exc:
        sys.stderr.write(f"error: {exc}\n")
        return exit_code_for(exc) if isinstance(exc, GenoLeWMError) else 2
    sys.stdout.write(json.dumps(report, indent=2, sort_keys=True) + "\n")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
