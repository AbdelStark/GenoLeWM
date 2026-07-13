# SPDX-License-Identifier: Apache-2.0
"""Assemble and verify the checksum-closed, role-bound v0.3 dataset snapshot.

The assembler consumes already-downloaded immutable Hugging Face artifacts.  It
does not resolve branches or tags, and it never uploads.  Publication remains a
separate job step so the resulting Hub commit can be downloaded and verified
again by this module.
"""

from __future__ import annotations

import argparse
import ctypes
import errno
import importlib
import json
import os
import re
import shutil
import stat
import sys
import tempfile
from collections.abc import Mapping, Sequence
from contextlib import suppress
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Final, Literal

from geno_lewm.data import MembershipStore
from geno_lewm.errors import GenoLeWMError, InputError, RuntimeSetupError, exit_code_for
from geno_lewm.provenance import sha256_file
from geno_lewm.provenance.hashing import looks_like_sha256
from tools.release.dataset_integrity import (
    DEFAULT_REPORT_NAME,
    build_dataset_integrity_report,
)
from tools.release.dataset_package import (
    ARTIFACT_ROLE_SCHEMA_VERSION,
    GENERATED_BY as DATASET_PACKAGE_GENERATED_BY,
    MEMBERSHIP_SPLIT_EVIDENCE_SCHEMA_VERSION,
    build_dataset_package,
    load_dataset_package,
    render_data_card,
)

SCHEMA_VERSION: Final = "geno-lewm.v03-dataset-snapshot.v1"
INPUT_CHECK_SCHEMA_VERSION: Final = "geno-lewm.v03-dataset-snapshot-input-check.v1"
GENERATED_BY: Final = "tools.release.v03_dataset_snapshot"
INPUT_CHECK_GENERATED_BY: Final = f"{GENERATED_BY}.check_inputs"
REPORT_NAME: Final = "dataset_snapshot_report.json"
INPUT_CHECK_REPORT_NAME: Final = "dataset_input_check_report.json"

_MEMBERSHIP_FILES: Final = frozenset(
    {
        "SHA256SUMS",
        "contract/membership-build-receipt.schema.json",
        "contract/membership-build-spec.schema.json",
        "contract/membership-build.json",
        "contract/membership-store.schema.json",
        "evidence/download-plan.json",
        "evidence/job-summary.json",
        "evidence/membership-build-report.json",
        "evidence/membership-verify-report.json",
        "evidence/source-download-identities.json",
        "store/build-receipt.json",
        "store/lookup.sqlite",
        "store/manifest.json",
        "store/memberships.parquet",
        "store/snapshot-lineage.json",
    }
)
_SPLIT_FILES: Final = frozenset(
    {
        "SHA256SUMS",
        "contract/membership-split-evidence.schema.json",
        "evidence/membership-split-evidence.json",
        "splits/evaluation/clinvar-chr21.labels.jsonl",
        "splits/evaluation/clinvar-chr21.vcf",
        "splits/validation/clinvar-chr20.labels.jsonl",
        "splits/validation/clinvar-chr20.vcf",
    }
)
_STORE_FILES: Final = (
    "build-receipt.json",
    "lookup.sqlite",
    "manifest.json",
    "memberships.parquet",
    "snapshot-lineage.json",
)
_MEMBERSHIP_PROVENANCE_FILES: Final = tuple(
    sorted(path for path in _MEMBERSHIP_FILES if path != "SHA256SUMS" and not path.startswith("store/"))
)
_TRAIN_CHROMOSOMES: Final = (*map(str, range(1, 20)), "22")
_COMMIT_RE: Final = re.compile(r"[0-9a-f]{40}")
_REPOSITORY_RE: Final = re.compile(r"[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+")
_IMAGE_RE: Final = re.compile(r"[^@\s]+@sha256:[0-9a-f]{64}")
_SAFE_COMPONENT_RE: Final = re.compile(r"[A-Za-z0-9._-]+(?:/[A-Za-z0-9._-]+)*")
_ISO_UTC_RE: Final = re.compile(
    r"[0-9]{4}-[0-9]{2}-[0-9]{2}T[0-9]{2}:[0-9]{2}:[0-9]{2}(?:\.[0-9]+)?Z"
)


@dataclass(frozen=True, slots=True)
class FilteredParquet:
    """Identity and cardinality of one membership-filtered Parquet."""

    source_sha256: str
    source_size_bytes: int
    source_rows: int
    sha256: str
    size_bytes: int
    records: int

    def to_dict(self) -> dict[str, object]:
        return {
            "source_sha256": self.source_sha256,
            "source_size_bytes": self.source_size_bytes,
            "source_rows": self.source_rows,
            "sha256": self.sha256,
            "size_bytes": self.size_bytes,
            "records": self.records,
        }


@dataclass(frozen=True, slots=True)
class V03SnapshotReport:
    """Summary of a locally assembled and independently verified snapshot."""

    dataset_dir: Path
    payload: dict[str, object]

    @property
    def snapshot_id(self) -> str:
        return str(self.payload["snapshot_id"])

    @property
    def report_path(self) -> Path:
        return self.dataset_dir / REPORT_NAME

    def to_dict(self) -> dict[str, object]:
        return dict(self.payload)


def filter_membership_parquet(
    source_path: Path,
    output_path: Path,
    *,
    kind: Literal["gnomad", "clinvar"],
    expected_source_row_ids: set[str] | frozenset[str],
    batch_size: int = 65_536,
) -> FilteredParquet:
    """Copy exactly the membership-selected source rows into a deterministic Parquet.

    Every source column, Arrow type, field annotation, and schema-level metadata
    is retained.  Rows remain in original order.  The call fails closed if an
    expected membership identity is absent or occurs more than once.
    """
    if kind not in {"gnomad", "clinvar"}:
        raise InputError("Parquet membership filter kind must be gnomad or clinvar")
    if not expected_source_row_ids or not all(
        isinstance(value, str) and value for value in expected_source_row_ids
    ):
        raise InputError("expected_source_row_ids must be a non-empty set of strings")
    if isinstance(batch_size, bool) or not isinstance(batch_size, int) or batch_size < 1:
        raise InputError("Parquet filter batch_size must be a positive integer")
    source = _regular_file(Path(source_path), "membership source Parquet")
    output = Path(output_path)
    if output.exists() or output.is_symlink():
        raise InputError("filtered Parquet output already exists", details={"path": str(output)})
    output.parent.mkdir(parents=True, exist_ok=True)
    _reject_symlink_ancestors(output.parent, stop=output.parent.parent)
    pa, pq = _require_pyarrow()
    try:
        parquet = pq.ParquetFile(source)
    except Exception as exc:
        raise InputError(
            "membership source Parquet is invalid", details={"path": str(source)}
        ) from exc
    schema = parquet.schema_arrow
    required = {"chrom", "pos", "ref", "alt"}
    if kind == "clinvar":
        required.add("clinvar_id")
    missing_columns = required - set(schema.names)
    if missing_columns:
        raise InputError(
            "membership source Parquet schema is incomplete",
            details={"missing": sorted(missing_columns), "path": str(source)},
        )
    source_rows = int(parquet.metadata.num_rows)
    expected = frozenset(expected_source_row_ids)
    selected: set[str] = set()
    temporary = output.with_name(f".{output.name}.tmp-{os.getpid()}")
    with suppress(OSError):
        temporary.unlink()
    writer: Any | None = None
    written = 0
    try:
        writer = pq.ParquetWriter(
            temporary,
            schema,
            compression="zstd",
            use_dictionary=True,
            write_statistics=True,
            version="2.6",
            data_page_version="2.0",
        )
        for batch in parquet.iter_batches(batch_size=batch_size):
            columns = batch.to_pydict()
            mask: list[bool] = []
            for index in range(batch.num_rows):
                identity = _source_row_identity(columns, index=index, kind=kind)
                include = identity in expected
                if include:
                    if identity in selected:
                        raise InputError(
                            "membership source Parquet contains a duplicate selected source row",
                            details={"source_row_id": identity, "path": str(source)},
                        )
                    selected.add(identity)
                mask.append(include)
            if any(mask):
                filtered = batch.filter(pa.array(mask, type=pa.bool_()))
                writer.write_batch(filtered)
                written += filtered.num_rows
        writer.close()
        writer = None
        missing = expected - selected
        if missing:
            raise InputError(
                "membership rows were absent from the pinned source Parquet",
                details={"missing_count": len(missing), "examples": sorted(missing)[:10]},
            )
        if written != len(expected):
            raise InputError("filtered Parquet cardinality does not match membership selection")
        observed = pq.ParquetFile(temporary)
        if not observed.schema_arrow.equals(schema, check_metadata=True):
            raise InputError("filtered Parquet did not preserve the exact source Arrow schema")
        if int(observed.metadata.num_rows) != written:
            raise InputError("filtered Parquet footer cardinality drifted")
        temporary.replace(output)
    except Exception:
        if writer is not None:
            with suppress(Exception):
                writer.close()
        with suppress(OSError):
            temporary.unlink()
        raise
    return FilteredParquet(
        source_sha256=sha256_file(source),
        source_size_bytes=source.stat().st_size,
        source_rows=source_rows,
        sha256=sha256_file(output),
        size_bytes=output.stat().st_size,
        records=written,
    )


def _source_row_identity(
    columns: Mapping[str, Sequence[object]],
    *,
    index: int,
    kind: Literal["gnomad", "clinvar"],
) -> str:
    values: dict[str, object] = {}
    for field in ("chrom", "pos", "ref", "alt"):
        value = columns[field][index]
        if value is None or isinstance(value, bool):
            raise InputError("membership source row contains a null or invalid variant field")
        values[field] = value
    try:
        position = int(str(values["pos"]))
    except (TypeError, ValueError) as exc:
        raise InputError("membership source row position is not an integer") from exc
    chrom = str(values["chrom"])
    ref = str(values["ref"])
    alt = str(values["alt"])
    if kind == "gnomad":
        return f"{chrom}:{position}:{ref}:{alt}"
    raw_id = columns["clinvar_id"][index]
    if isinstance(raw_id, bool):
        raise InputError("ClinVar source row clinvar_id is not an integer")
    try:
        clinvar_id = int(str(raw_id))
    except (TypeError, ValueError) as exc:
        raise InputError("ClinVar source row clinvar_id is not an integer") from exc
    return f"{clinvar_id}:{chrom}:{position}:{ref}:{alt}"


def _require_pyarrow() -> tuple[Any, Any]:
    try:
        return importlib.import_module("pyarrow"), importlib.import_module("pyarrow.parquet")
    except ImportError as exc:
        raise RuntimeSetupError(
            "v0.3 dataset snapshot assembly requires PyArrow",
            remediation="install the evidence dependency group",
        ) from exc


def _regular_file(path: Path, label: str) -> Path:
    candidate = Path(path).absolute()
    try:
        metadata = candidate.lstat()
    except OSError as exc:
        raise InputError(f"{label} is missing", details={"path": str(candidate)}) from exc
    if candidate.is_symlink() or not stat.S_ISREG(metadata.st_mode):
        raise InputError(
            f"{label} must be a regular non-symlink file", details={"path": str(candidate)}
        )
    return candidate


def _reject_symlink_ancestors(path: Path, *, stop: Path) -> None:
    current = Path(path).absolute()
    boundary = Path(stop).absolute()
    while True:
        if current.is_symlink():
            raise InputError("snapshot paths must not traverse symbolic links")
        if current in (boundary, current.parent):
            break
        current = current.parent


def main(argv: list[str] | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    try:
        if args.command == "verify":
            report = verify_v03_dataset_snapshot(args.dataset_dir)
        else:
            report = assemble_v03_dataset_snapshot(
                membership_bundle_dir=args.membership_bundle_dir,
                split_bundle_dir=args.split_bundle_dir,
                gnomad_root=args.gnomad_root,
                clinvar_root=args.clinvar_root,
                training_windows_path=args.training_windows,
                dataset_dir=args.dataset_dir,
                split_repository=args.split_repository,
                split_revision=args.split_revision,
                split_artifact_path=args.split_artifact_path,
                snapshot_id=args.snapshot_id,
                generated_at=args.generated_at,
                producer_git_commit=args.producer_git_commit,
                container_image=args.container_image,
            )
    except GenoLeWMError as exc:
        sys.stderr.write(f"error: {exc}\n")
        return exit_code_for(exc)
    sys.stdout.write(json.dumps(report.to_dict(), indent=2, sort_keys=True) + "\n")
    return 0


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    assemble = commands.add_parser("assemble", help="assemble an immutable local snapshot")
    assemble.add_argument("--membership-bundle-dir", type=Path, required=True)
    assemble.add_argument("--split-bundle-dir", type=Path, required=True)
    assemble.add_argument("--gnomad-root", type=Path, required=True)
    assemble.add_argument("--clinvar-root", type=Path, required=True)
    assemble.add_argument("--training-windows", type=Path, required=True)
    assemble.add_argument("--dataset-dir", type=Path, required=True)
    assemble.add_argument("--split-repository", required=True)
    assemble.add_argument("--split-revision", required=True)
    assemble.add_argument("--split-artifact-path", required=True)
    assemble.add_argument("--snapshot-id", required=True)
    assemble.add_argument("--generated-at", required=True)
    assemble.add_argument("--producer-git-commit", required=True)
    assemble.add_argument("--container-image", required=True)
    verify = commands.add_parser("verify", help="independently verify a completed snapshot")
    verify.add_argument("--dataset-dir", type=Path, required=True)
    return parser


# The complete assembler and verifier are defined below the small public
# filtering primitive so tests can exercise the row-selection contract without
# synthesizing a full membership store.


def assemble_v03_dataset_snapshot(
    *,
    membership_bundle_dir: Path,
    split_bundle_dir: Path,
    gnomad_root: Path,
    clinvar_root: Path,
    training_windows_path: Path,
    dataset_dir: Path,
    split_repository: str,
    split_revision: str,
    split_artifact_path: str,
    snapshot_id: str,
    generated_at: str,
    producer_git_commit: str,
    container_image: str,
) -> V03SnapshotReport:
    """Build a schema-1.1 snapshot from exact, already-downloaded Hub inputs."""
    membership_root = _exact_bundle(membership_bundle_dir, _MEMBERSHIP_FILES, "membership")
    split_root = _exact_bundle(split_bundle_dir, _SPLIT_FILES, "membership split")
    split_repository = _repository(split_repository, "split_repository")
    split_revision = _commit(split_revision, "split_revision")
    split_artifact_path = _safe_relative_text(split_artifact_path, "split_artifact_path")
    producer_git_commit = _commit(producer_git_commit, "producer_git_commit")
    if not _IMAGE_RE.fullmatch(container_image):
        raise InputError("container_image must be digest pinned")
    if not isinstance(snapshot_id, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", snapshot_id):
        raise InputError("snapshot_id must be a safe immutable identifier")
    if not isinstance(generated_at, str) or not _ISO_UTC_RE.fullmatch(generated_at):
        raise InputError("generated_at must be an explicit UTC ISO-8601 timestamp ending in Z")

    output = Path(dataset_dir).absolute()
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists() or output.is_symlink():
        raise InputError(
            "v0.3 dataset snapshot output already exists",
            details={"path": str(output)},
            remediation="choose a new immutable output directory",
        )

    split_report = _json_object(
        split_root / "evidence/membership-split-evidence.json", "membership split report"
    )
    source_identities = _json_object(
        membership_root / "evidence/source-download-identities.json",
        "membership source identities",
    )
    membership_spec = _json_object(
        membership_root / "contract/membership-build.json", "membership build spec"
    )
    membership_job = _json_object(
        membership_root / "evidence/job-summary.json", "membership job summary"
    )
    sources = _validate_source_contract(
        source_identities,
        membership_spec=membership_spec,
        membership_job=membership_job,
        gnomad_root=Path(gnomad_root),
        clinvar_root=Path(clinvar_root),
    )
    training_windows = _validate_training_windows(
        split_report, Path(training_windows_path)
    )
    membership_origin = _validate_split_origin(
        split_report,
        split_repository=split_repository,
        split_revision=split_revision,
        split_artifact_path=split_artifact_path,
    )

    temporary = Path(tempfile.mkdtemp(prefix=f".{output.name}.tmp-", dir=output.parent))
    try:
        store_root = membership_root / "store"
        with MembershipStore.open(store_root, verify=True) as store:
            _validate_store_binding(store.manifest, split_report)
            selections = _train_source_row_ids(store, sources)
            staged_files: list[dict[str, object]] = []
            transformations: list[dict[str, object]] = []

            for name in _STORE_FILES:
                relative = f"membership/store/{name}"
                _copy_regular_file(store_root / name, temporary / relative)
                staged_files.append(_evidence_file(relative, "Verified membership store"))

            for source in sources:
                source_id = str(source["source_id"])
                selected = selections.get(source_id, frozenset())
                if source["kind"] == "gnomad" and source["chromosome"] not in _TRAIN_CHROMOSOMES:
                    if selected:
                        raise InputError("held-out gnomAD source unexpectedly has train memberships")
                    continue
                if not selected:
                    raise InputError(
                        "train membership source is empty", details={"source_id": source_id}
                    )
                if source["kind"] == "gnomad":
                    relative = (
                        f"gnomad/v4.1/train/chr{source['chromosome']}.variants.parquet"
                    )
                    split = "train_gnomad_common"
                    description = (
                        f"Exact membership-selected gnomAD v4.1 chr{source['chromosome']} rows"
                    )
                else:
                    relative = "clinvar/2026-04-15/train.variants.parquet"
                    split = "train_clinvar"
                    description = "Exact membership-selected ClinVar training rows"
                filtered = filter_membership_parquet(
                    Path(str(source["local_path"])),
                    temporary / relative,
                    kind=str(source["kind"]),  # type: ignore[arg-type]
                    expected_source_row_ids=set(selected),
                )
                if (
                    filtered.source_sha256 != source["sha256"]
                    or filtered.source_size_bytes != source["size_bytes"]
                ):
                    raise InputError("prepared source changed after its identity preflight")
                staged_files.append(
                    {
                        "path": relative,
                        "artifact_role": "split_data",
                        "split": split,
                        "records": filtered.records,
                        "description": description,
                    }
                )
                transformations.append(
                    {
                        "kind": source["kind"],
                        "source_id": source_id,
                        "source": _public_source_identity(source),
                        "output": {
                            "path": relative,
                            "sha256": filtered.sha256,
                            "size_bytes": filtered.size_bytes,
                            "records": filtered.records,
                        },
                        "selection": "exact_membership_store_train_source_row_id",
                    }
                )

        windows_relative = str(training_windows["artifact_path"])
        _copy_regular_file(
            Path(str(training_windows["local_path"])), temporary / windows_relative
        )
        staged_files.append(
            {
                "path": windows_relative,
                "artifact_role": "split_data",
                "split": str(training_windows["split"]),
                "records": _positive_int(training_windows["records"], "training records"),
                "description": "Placed gnomAD common-variant training windows audited against held roles",
            }
        )

        for role in ("validation", "evaluation"):
            stream = _mapping(split_report.get("streams"), f"split report streams.{role}")[role]
            stream = _mapping(stream, f"split report stream {role}")
            records = _positive_int(stream.get("record_count"), f"{role} record_count")
            labels = _file_identity_mapping(stream.get("labels_jsonl"), f"{role} labels")
            vcf = _file_identity_mapping(stream.get("vcf"), f"{role} vcf")
            labels_relative = str(labels["path"])
            vcf_relative = str(vcf["path"])
            _copy_and_verify_identity(split_root / labels_relative, temporary / labels_relative, labels)
            _copy_and_verify_identity(split_root / vcf_relative, temporary / vcf_relative, vcf)
            staged_files.extend(
                (
                    {
                        "path": labels_relative,
                        "artifact_role": "split_data",
                        "split": role,
                        "records": records,
                        "description": f"Membership-derived ClinVar {role} labels",
                    },
                    {
                        "path": vcf_relative,
                        "artifact_role": "split_companion",
                        "split": role,
                        "records": records,
                        "companion_of": labels_relative,
                        "description": f"VCF companion for the ClinVar {role} labels",
                    },
                )
            )

        split_schema_relative = "contract/membership-split-evidence.schema.json"
        split_report_relative = "evidence/membership-split-evidence.json"
        _copy_regular_file(split_root / split_schema_relative, temporary / split_schema_relative)
        _copy_regular_file(split_root / split_report_relative, temporary / split_report_relative)
        staged_files.extend(
            (
                _evidence_file(split_schema_relative, "Tracked split-evidence schema"),
                _evidence_file(split_report_relative, "Publication-eligible split evidence"),
            )
        )

        membership_checksum_relative = "evidence/membership-provenance/SHA256SUMS"
        _copy_regular_file(membership_root / "SHA256SUMS", temporary / membership_checksum_relative)
        staged_files.append(
            _evidence_file(membership_checksum_relative, "Membership success-bundle checksums")
        )
        for relative_source in _MEMBERSHIP_PROVENANCE_FILES:
            relative_target = f"evidence/membership-provenance/{relative_source}"
            _copy_regular_file(membership_root / relative_source, temporary / relative_target)
            staged_files.append(_evidence_file(relative_target, "Membership build provenance"))
        split_checksum_relative = "evidence/split-provenance/SHA256SUMS"
        _copy_regular_file(split_root / "SHA256SUMS", temporary / split_checksum_relative)
        staged_files.append(
            _evidence_file(split_checksum_relative, "Membership-split success-bundle checksums")
        )

        input_check = _input_check_payload(
            snapshot_id=snapshot_id,
            generated_at=generated_at,
            producer_git_commit=producer_git_commit,
            container_image=container_image,
            membership_origin=membership_origin,
            split_origin={
                "repository": split_repository,
                "revision": split_revision,
                "artifact_path": split_artifact_path,
                **_file_identity(split_root / "SHA256SUMS", "SHA256SUMS"),
            },
            sources=sources,
            training_windows=training_windows,
        )
        _write_json(temporary / INPUT_CHECK_REPORT_NAME, input_check)

        store_manifest = _json_object(membership_root / "store/manifest.json", "membership manifest")
        split_streams = _mapping(split_report.get("streams"), "split report streams")
        train_gnomad_records = sum(
            _positive_int(file["records"], "gnomAD staged records")
            for file in staged_files
            if file.get("split") == "train_gnomad_common"
        )
        train_clinvar_records = sum(
            _positive_int(file["records"], "ClinVar staged records")
            for file in staged_files
            if file.get("split") == "train_clinvar"
        )
        metadata = {
            "schema_version": ARTIFACT_ROLE_SCHEMA_VERSION,
            "snapshot_id": snapshot_id,
            "generated_by": DATASET_PACKAGE_GENERATED_BY,
            "generated_at": generated_at,
            "sources": _metadata_sources(
                membership_origin=membership_origin,
                split_origin={
                    "repository": split_repository,
                    "revision": split_revision,
                    "artifact_path": split_artifact_path,
                },
                source_identities=source_identities,
                training_windows=training_windows,
            ),
            "license": (
                "Apache-2.0 for package metadata and tooling; each upstream data artifact "
                "remains governed by its recorded source terms."
            ),
            "preprocessing": [
                "Verify exact immutable Hub revisions, complete success-bundle inventories, and SHA-256 closure before reading data.",
                "Full-scan the membership store and select prepared gnomAD and ClinVar rows by exact train-role source_row_id.",
                "Preserve source Arrow schemas, metadata, columns, values, and row order in deterministic Parquet outputs.",
                "Bind validation/evaluation JSONL and VCF streams plus placed windows to the publication-eligible split report.",
            ],
            "split_policy": (
                "Chromosomes 1-19 and 22 are training; chromosome 20 is validation; "
                "chromosome 21 is evaluation. Exact variant membership and placed-window "
                "nonintersection are bound to the included verified store and report."
            ),
            "splits": {
                "train_gnomad_common": {
                    "records": train_gnomad_records,
                    "description": "Membership-selected gnomAD training variants",
                },
                "train_clinvar": {
                    "records": train_clinvar_records,
                    "description": "Membership-selected ClinVar training variants",
                },
                str(training_windows["split"]): {
                    "records": _positive_int(training_windows["records"], "training records"),
                    "description": "Placed training windows with exhaustive held-role nonintersection",
                },
                "validation": {
                    "records": int(_mapping(split_streams["validation"], "validation stream")["record_count"]),
                    "description": "Held-out ClinVar chromosome-20 labels",
                },
                "evaluation": {
                    "records": int(_mapping(split_streams["evaluation"], "evaluation stream")["record_count"]),
                    "description": "Held-out ClinVar chromosome-21 labels",
                },
            },
            "leakage_checks": [
                "Every split-contributing artifact is counted and hashed from package bytes.",
                "Comparable variant keys and genomic regions are disjoint between every training and held split.",
                "The included split report records exhaustive and deterministic-sample placed-window nonintersection checks.",
            ],
            "intended_use": (
                "Reproducible GenoLeWM v0.3 training, validation, and evaluation runs that "
                "consume the package-bound membership policy."
            ),
            "limitations": [
                "The package establishes deterministic unphased variant membership, not phased haplotype membership.",
                "It does not establish population representativeness, model quality, benchmark performance, or clinical validity.",
                "Upstream gnomAD and ClinVar scope and curation limitations remain applicable.",
            ],
            "files": staged_files,
            "membership_and_split_evidence": {
                "membership_store": {
                    "path": "membership/store",
                    "artifact_id": store_manifest["artifact_id"],
                    "content_identity": store_manifest["content_identity"],
                    "physical_identity": store_manifest["physical_identity"],
                    "rowset_sha256": store_manifest["rowset_sha256"],
                },
                "report": {
                    "path": split_report_relative,
                    "schema_path": split_schema_relative,
                    "artifact_id": split_report["artifact_id"],
                    "schema_version": split_report["schema_version"],
                },
            },
        }
        metadata_path = temporary / "dataset_package.json"
        _write_json(metadata_path, metadata)
        package_report = build_dataset_package(temporary, metadata_path)
        package = load_dataset_package(temporary, metadata_path)
        integrity_path = temporary / DEFAULT_REPORT_NAME
        integrity_payload = _json_object(integrity_path, "split integrity report")
        integrity_payload["generated_at"] = generated_at
        _write_json(integrity_path, integrity_payload)
        (temporary / "data_card.md").write_text(
            render_data_card(package, integrity_report=integrity_payload), encoding="utf-8"
        )

        snapshot_payload: dict[str, object] = {
            "ok": True,
            "schema_version": SCHEMA_VERSION,
            "generated_by": GENERATED_BY,
            "generated_at": generated_at,
            "snapshot_id": snapshot_id,
            "producer": {
                "git_commit": producer_git_commit,
                "container_image": container_image,
            },
            "input_check": _file_identity(
                temporary / INPUT_CHECK_REPORT_NAME, INPUT_CHECK_REPORT_NAME
            ),
            "package": {
                "schema_version": package_report.schema_version,
                "metadata": _file_identity(metadata_path, "dataset_package.json"),
                "manifest": _file_identity(
                    temporary / "dataset_manifest.json", "dataset_manifest.json"
                ),
                "data_card": _file_identity(temporary / "data_card.md", "data_card.md"),
                "integrity": _file_identity(integrity_path, DEFAULT_REPORT_NAME),
            },
            "transformations": transformations,
            "claim_boundary": {
                "variant_membership": True,
                "phased_haplotype_membership": False,
                "publication_eligible": True,
                "released_v03_snapshot": False,
                "limitations": metadata["limitations"],
            },
        }
        _write_json(temporary / REPORT_NAME, snapshot_payload)
        _write_snapshot_checksums(temporary)
        _fsync_tree(temporary)
        verified = verify_v03_dataset_snapshot(temporary)
        _publish_directory_noreplace(temporary, output)
        _fsync_directory(output.parent)
        return V03SnapshotReport(output, verified.payload)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise


def verify_v03_dataset_snapshot(dataset_dir: Path) -> V03SnapshotReport:
    """Independently verify one local or freshly downloaded v0.3 snapshot."""
    root = Path(dataset_dir).absolute()
    if root.is_symlink() or not root.is_dir():
        raise InputError("v0.3 dataset snapshot must be a non-symlink directory")
    observed = _inventory(root)
    required_generated = {
        "SHA256SUMS",
        "data_card.md",
        "dataset_manifest.json",
        "dataset_package.json",
        DEFAULT_REPORT_NAME,
        INPUT_CHECK_REPORT_NAME,
        REPORT_NAME,
    }
    if not required_generated <= observed:
        raise InputError(
            "v0.3 dataset snapshot generated-file inventory is incomplete",
            details={"missing": sorted(required_generated - observed)},
        )
    manifest_payload = _json_object(root / "dataset_manifest.json", "dataset manifest")
    raw_files = manifest_payload.get("files")
    if not isinstance(raw_files, list) or not raw_files:
        raise InputError("dataset manifest files must be a non-empty list")
    declared_paths = {
        _safe_relative_text(
            str(_mapping(item, "dataset manifest file").get("path")), "dataset manifest path"
        )
        for item in raw_files
    }
    expected_inventory = required_generated | declared_paths
    if observed != expected_inventory:
        raise InputError(
            "v0.3 dataset snapshot exact inventory drifted",
            details={
                "missing": sorted(expected_inventory - observed),
                "unexpected": sorted(observed - expected_inventory),
            },
        )
    _verify_sha256sums(root, expected_inventory - {"SHA256SUMS"})

    package = load_dataset_package(root, root / "dataset_package.json")
    if package.schema_version != ARTIFACT_ROLE_SCHEMA_VERSION:
        raise InputError("v0.3 dataset snapshot requires package schema 1.1.0")
    if package.manifest() != manifest_payload:
        raise InputError("dataset manifest differs from the validated package metadata")
    integrity_payload = _json_object(root / DEFAULT_REPORT_NAME, "split integrity report")
    recomputed_integrity = build_dataset_integrity_report(
        root,
        root / "dataset_manifest.json",
        generated_at=package.generated_at,
    ).to_dict()
    if integrity_payload != recomputed_integrity:
        raise InputError("split integrity report differs from a full recomputation")
    expected_card = render_data_card(package, integrity_report=recomputed_integrity)
    if (root / "data_card.md").read_text(encoding="utf-8") != expected_card:
        raise InputError("data card differs from the validated package and integrity report")

    input_check = _json_object(root / INPUT_CHECK_REPORT_NAME, "snapshot input check")
    _verify_input_check(input_check, package_snapshot_id=package.snapshot_id)
    snapshot = _json_object(root / REPORT_NAME, "dataset snapshot report")
    _verify_snapshot_report(
        snapshot,
        root=root,
        package_snapshot_id=package.snapshot_id,
        generated_at=package.generated_at,
    )
    return V03SnapshotReport(root, snapshot)


def _validate_source_contract(
    payload: dict[str, Any],
    *,
    membership_spec: dict[str, Any],
    membership_job: dict[str, Any],
    gnomad_root: Path,
    clinvar_root: Path,
) -> tuple[dict[str, object], ...]:
    if payload.get("ok") is not True or payload.get("source_count") != 23:
        raise InputError("membership source identity report is not a successful 23-source report")
    repositories = _mapping(payload.get("repositories"), "source repositories")
    files = payload.get("files")
    if not isinstance(files, list) or len(files) != 23:
        raise InputError("membership source identities must contain exactly 23 files")
    spec_sources = membership_spec.get("sources")
    if not isinstance(spec_sources, list) or len(spec_sources) != 23:
        raise InputError("membership build spec must contain exactly 23 sources")
    job_inputs = _mapping(membership_job.get("inputs"), "membership job inputs")
    observed: list[dict[str, object]] = []
    gnomad_chromosomes: set[str] = set()
    clinvar_count = 0
    expected_paths: dict[str, set[str]] = {"gnomad": set(), "clinvar": set()}
    for index, raw in enumerate(files):
        item = _mapping(raw, f"source identities files[{index}]")
        kind = item.get("kind")
        if kind not in {"gnomad", "clinvar"}:
            raise InputError("membership source kind is invalid")
        repository_entry = _mapping(repositories.get(kind), f"{kind} repository")
        repository = _repository(str(repository_entry.get("repo_id")), f"{kind} repository")
        if repository_entry.get("repo_type") != "dataset":
            raise InputError("membership source repository type must be dataset")
        revision = _commit(str(item.get("revision")), f"{kind} revision")
        if repository_entry.get("revision") != revision:
            raise InputError("membership source revision differs from repository binding")
        if job_inputs.get(f"{kind}_revision") != revision:
            raise InputError("membership source revision differs from job summary")
        artifact_path = _safe_relative_text(str(item.get("artifact_path")), "source artifact_path")
        digest = str(item.get("sha256"))
        if not looks_like_sha256(digest):
            raise InputError("membership source sha256 is invalid")
        size_bytes = _positive_int(item.get("size_bytes"), "source size_bytes")
        chromosome: str | None = None
        if kind == "gnomad":
            chromosome = str(item.get("chromosome"))
            if chromosome not in {str(value) for value in range(1, 23)}:
                raise InputError("gnomAD source chromosome is invalid")
            if chromosome in gnomad_chromosomes:
                raise InputError("gnomAD source chromosome is duplicated")
            gnomad_chromosomes.add(chromosome)
            source_id = f"gnomad-v4.1-chr{chromosome}"
            local_path = Path(gnomad_root) / artifact_path
        else:
            if "chromosome" in item:
                raise InputError("ClinVar source identity must not declare a chromosome")
            clinvar_count += 1
            source_id = "clinvar-2026-04-15"
            local_path = Path(clinvar_root) / artifact_path
        spec_item = _mapping(spec_sources[index], f"membership sources[{index}]")
        expected_spec_path = f"{kind}/{artifact_path}"
        if spec_item.get("kind") != kind or spec_item.get("path") != expected_spec_path:
            raise InputError("membership build spec and source identity order/path differ")
        if kind == "gnomad" and str(spec_item.get("chromosome")) != chromosome:
            raise InputError("membership build spec gnomAD chromosome differs")
        if kind == "clinvar" and "chromosome" in spec_item:
            raise InputError("membership build spec ClinVar source declares a chromosome")
        local = _regular_file(local_path, f"prepared {kind} source")
        if sha256_file(local) != digest or local.stat().st_size != size_bytes:
            raise InputError(
                "prepared source identity differs from the membership build evidence",
                details={"artifact_path": artifact_path},
            )
        expected_paths[kind].add(artifact_path)
        observed.append(
            {
                "kind": kind,
                "source_id": source_id,
                "chromosome": chromosome,
                "repository": repository,
                "revision": revision,
                "artifact_path": artifact_path,
                "sha256": digest,
                "size_bytes": size_bytes,
                "local_path": str(local),
            }
        )
    if gnomad_chromosomes != {str(value) for value in range(1, 23)} or clinvar_count != 1:
        raise InputError("membership source set is not exactly gnomAD chr1-22 plus ClinVar")
    _verify_download_root_inventory(Path(gnomad_root), expected_paths["gnomad"], "gnomAD")
    _verify_download_root_inventory(Path(clinvar_root), expected_paths["clinvar"], "ClinVar")
    return tuple(observed)


def _validate_training_windows(
    split_report: dict[str, Any], local_path: Path
) -> dict[str, object]:
    training = _mapping(split_report.get("training_windows"), "split report training_windows")
    source = _mapping(training.get("source"), "training-window source")
    path = _regular_file(local_path, "training-window artifact")
    expected_sha = str(training.get("sha256"))
    expected_size = _positive_int(training.get("size_bytes"), "training-window size_bytes")
    records = _positive_int(training.get("record_count"), "training-window record_count")
    if sha256_file(path) != expected_sha or path.stat().st_size != expected_size:
        raise InputError("training-window identity differs from the split report")
    observed_records = 0
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                raise InputError(
                    "training-window JSONL contains a blank line", details={"line": line_number}
                )
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise InputError(
                    "training-window JSONL row is invalid", details={"line": line_number}
                ) from exc
            if not isinstance(row, dict):
                raise InputError("training-window JSONL rows must be objects")
            observed_records += 1
    if observed_records != records:
        raise InputError("training-window record count differs from the split report")
    return {
        "repository": _repository(str(source.get("repository")), "training repository"),
        "revision": _commit(str(source.get("revision")), "training revision"),
        "artifact_path": _safe_relative_text(
            str(source.get("artifact_path")), "training artifact_path"
        ),
        "sha256": expected_sha,
        "size_bytes": expected_size,
        "records": records,
        "split": str(training.get("split")),
        "local_path": str(path),
    }


def _validate_split_origin(
    split_report: dict[str, Any],
    *,
    split_repository: str,
    split_revision: str,
    split_artifact_path: str,
) -> dict[str, object]:
    store = _mapping(split_report.get("membership_store"), "split report membership_store")
    membership = {
        "repository": _repository(str(store.get("repository")), "membership repository"),
        "revision": _commit(str(store.get("revision")), "membership revision"),
        "artifact_path": _safe_relative_text(
            str(store.get("artifact_path")), "membership artifact_path"
        ),
    }
    if split_report.get("ok") is not True:
        raise InputError("membership split report is not successful")
    if split_report.get("schema_version") != MEMBERSHIP_SPLIT_EVIDENCE_SCHEMA_VERSION:
        raise InputError("membership split report schema version is unsupported")
    return {
        **membership,
        "split_repository": split_repository,
        "split_revision": split_revision,
        "split_artifact_path": split_artifact_path,
    }


def _validate_store_binding(manifest: object, split_report: dict[str, Any]) -> None:
    reported = _mapping(split_report.get("membership_store"), "split report membership_store")
    for field in ("artifact_id", "content_identity", "physical_identity", "rowset_sha256"):
        if getattr(manifest, field, None) != reported.get(field):
            raise InputError(
                "verified membership store differs from the split report",
                details={"field": field},
            )
    roles = getattr(manifest, "chromosome_roles", None)
    if roles is None or roles.to_dict() != reported.get("chromosome_roles"):
        raise InputError("verified membership chromosome roles differ from the split report")


def _train_source_row_ids(
    store: Any, sources: Sequence[Mapping[str, object]]
) -> dict[str, frozenset[str]]:
    allowed = {str(source["source_id"]) for source in sources}
    selected: dict[str, set[str]] = {source_id: set() for source_id in allowed}
    row_count = 0
    for row in store.iter_role("train"):
        if row.role != "train" or row.source not in allowed:
            raise InputError("membership store train iterator emitted an invalid source or role")
        values = selected[row.source]
        if row.source_row_id in values:
            raise InputError("membership store train iterator emitted a duplicate source row")
        values.add(row.source_row_id)
        row_count += 1
    expected_count = getattr(store.manifest, "role_counts", {}).get("train")
    if expected_count is not None and row_count != expected_count:
        raise InputError("membership train iterator cardinality differs from the manifest")
    return {key: frozenset(value) for key, value in selected.items()}


def _input_check_payload(
    *,
    snapshot_id: str,
    generated_at: str,
    producer_git_commit: str,
    container_image: str,
    membership_origin: Mapping[str, object],
    split_origin: Mapping[str, object],
    sources: Sequence[Mapping[str, object]],
    training_windows: Mapping[str, object],
) -> dict[str, object]:
    upstreams: list[dict[str, object]] = [
        {
            "kind": "membership_bundle",
            "repository": membership_origin["repository"],
            "revision": membership_origin["revision"],
            "artifact_path": membership_origin["artifact_path"],
        },
        {"kind": "split_bundle", **dict(split_origin)},
    ]
    upstreams.extend(
        {"kind": f"prepared_{source['kind']}", **_public_source_identity(source)}
        for source in sources
    )
    upstreams.append(
        {
            "kind": "training_windows",
            **{
                key: training_windows[key]
                for key in (
                    "repository",
                    "revision",
                    "artifact_path",
                    "sha256",
                    "size_bytes",
                    "records",
                )
            },
        }
    )
    return {
        "ok": True,
        "schema_version": INPUT_CHECK_SCHEMA_VERSION,
        "generated_by": INPUT_CHECK_GENERATED_BY,
        "generated_at": generated_at,
        "snapshot_id": snapshot_id,
        "producer": {
            "git_commit": producer_git_commit,
            "container_image": container_image,
        },
        "upstreams": upstreams,
        "checks": [
            "exact_immutable_revisions",
            "complete_bundle_inventories",
            "bundle_checksum_closure",
            "prepared_source_identities",
            "verified_membership_store",
            "publication_eligible_split_evidence",
            "training_window_identity_and_cardinality",
        ],
    }


def _metadata_sources(
    *,
    membership_origin: Mapping[str, object],
    split_origin: Mapping[str, object],
    source_identities: Mapping[str, object],
    training_windows: Mapping[str, object],
) -> list[dict[str, str]]:
    repositories = _mapping(source_identities.get("repositories"), "source repositories")
    gnomad = _mapping(repositories.get("gnomad"), "gnomAD repository")
    clinvar = _mapping(repositories.get("clinvar"), "ClinVar repository")
    return [
        _metadata_source("gnomAD prepared shards", gnomad, "Pinned prepared variant shards"),
        _metadata_source("ClinVar prepared shard", clinvar, "Pinned normalized ClinVar shard"),
        _metadata_source(
            "Membership store",
            membership_origin,
            "Verified train/validation/evaluation membership contract",
        ),
        _metadata_source(
            "Membership split evidence",
            split_origin,
            "Publication-eligible stream and nonintersection evidence",
        ),
        _metadata_source(
            "Placed training windows",
            training_windows,
            "Pinned windows referenced by the split report",
        ),
    ]


def _metadata_source(name: str, raw: Mapping[str, object], notes: str) -> dict[str, str]:
    repository = str(raw.get("repo_id", raw.get("repository", raw.get("split_repository", ""))))
    revision = str(raw.get("revision", raw.get("split_revision", "")))
    artifact_path = str(raw.get("artifact_path", raw.get("split_artifact_path", "")))
    return {
        "name": name,
        "revision": revision,
        "url": f"https://huggingface.co/datasets/{repository}/tree/{revision}/{artifact_path}",
        "license": "upstream source terms apply",
        "notes": notes,
    }


def _verify_input_check(payload: dict[str, Any], *, package_snapshot_id: str) -> None:
    expected_top = {
        "ok",
        "schema_version",
        "generated_by",
        "generated_at",
        "snapshot_id",
        "producer",
        "upstreams",
        "checks",
    }
    if set(payload) != expected_top:
        raise InputError("snapshot input-check fields do not match the closed schema")
    if (
        payload.get("ok") is not True
        or payload.get("schema_version") != INPUT_CHECK_SCHEMA_VERSION
        or payload.get("generated_by") != INPUT_CHECK_GENERATED_BY
        or payload.get("snapshot_id") != package_snapshot_id
    ):
        raise InputError("snapshot input-check identity is invalid")
    producer = _mapping(payload.get("producer"), "snapshot input-check producer")
    _commit(str(producer.get("git_commit")), "input-check producer git_commit")
    if not _IMAGE_RE.fullmatch(str(producer.get("container_image"))):
        raise InputError("snapshot input-check container image is not digest pinned")
    upstreams = payload.get("upstreams")
    if not isinstance(upstreams, list) or len(upstreams) != 26:
        raise InputError("snapshot input-check must bind exactly 26 upstream artifacts")
    for upstream in upstreams:
        item = _mapping(upstream, "snapshot input-check upstream")
        repository = item.get("repository", item.get("split_repository"))
        revision = item.get("revision", item.get("split_revision"))
        artifact_path = item.get("artifact_path", item.get("split_artifact_path"))
        _repository(str(repository), "input-check upstream repository")
        _commit(str(revision), "input-check upstream revision")
        _safe_relative_text(str(artifact_path), "input-check upstream artifact_path")
        digest = item.get("sha256")
        if digest is not None and not looks_like_sha256(str(digest)):
            raise InputError("snapshot input-check upstream sha256 is invalid")


def _verify_snapshot_report(
    payload: dict[str, Any],
    *,
    root: Path,
    package_snapshot_id: str,
    generated_at: str,
) -> None:
    expected_top = {
        "ok",
        "schema_version",
        "generated_by",
        "generated_at",
        "snapshot_id",
        "producer",
        "input_check",
        "package",
        "transformations",
        "claim_boundary",
    }
    if set(payload) != expected_top:
        raise InputError("dataset snapshot report fields do not match the closed schema")
    if (
        payload.get("ok") is not True
        or payload.get("schema_version") != SCHEMA_VERSION
        or payload.get("generated_by") != GENERATED_BY
        or payload.get("generated_at") != generated_at
        or payload.get("snapshot_id") != package_snapshot_id
    ):
        raise InputError("dataset snapshot report identity is invalid")
    _match_identity(
        payload.get("input_check"), root / INPUT_CHECK_REPORT_NAME, INPUT_CHECK_REPORT_NAME
    )
    package = _mapping(payload.get("package"), "dataset snapshot report package")
    if package.get("schema_version") != ARTIFACT_ROLE_SCHEMA_VERSION:
        raise InputError("dataset snapshot report package schema is invalid")
    for field, relative in (
        ("metadata", "dataset_package.json"),
        ("manifest", "dataset_manifest.json"),
        ("data_card", "data_card.md"),
        ("integrity", DEFAULT_REPORT_NAME),
    ):
        _match_identity(package.get(field), root / relative, relative)
    transformations = payload.get("transformations")
    if not isinstance(transformations, list) or len(transformations) != 21:
        raise InputError("dataset snapshot report must bind 20 gnomAD and one ClinVar transform")
    for transformation in transformations:
        item = _mapping(transformation, "snapshot transformation")
        if item.get("selection") != "exact_membership_store_train_source_row_id":
            raise InputError("snapshot transformation selection contract drifted")
        output = _mapping(item.get("output"), "snapshot transformation output")
        relative = _safe_relative_text(str(output.get("path")), "transformation output path")
        _match_identity(output, root / relative, relative, include_records=True)
    claim = _mapping(payload.get("claim_boundary"), "snapshot claim boundary")
    expected_claim = {
        "variant_membership": True,
        "phased_haplotype_membership": False,
        "publication_eligible": True,
        "released_v03_snapshot": False,
    }
    if {field: claim.get(field) for field in expected_claim} != expected_claim:
        raise InputError("dataset snapshot report claim boundary is invalid")
    limitations = claim.get("limitations")
    if not isinstance(limitations, list) or not limitations:
        raise InputError("dataset snapshot report limitations must be non-empty")


def _exact_bundle(path: Path, expected: frozenset[str], label: str) -> Path:
    root = Path(path).absolute()
    if root.is_symlink() or not root.is_dir():
        raise InputError(f"{label} bundle must be a non-symlink directory")
    observed = _inventory(root)
    if observed != expected:
        raise InputError(
            f"{label} bundle exact inventory drifted",
            details={"missing": sorted(expected - observed), "unexpected": sorted(observed - expected)},
        )
    _verify_sha256sums(root, expected - {"SHA256SUMS"})
    return root


def _inventory(root: Path) -> set[str]:
    files: set[str] = set()
    for path in root.rglob("*"):
        if path.is_symlink():
            raise InputError("artifact tree contains a symbolic link", details={"path": str(path)})
        if path.is_file():
            files.add(path.relative_to(root).as_posix())
        elif not path.is_dir():
            raise InputError("artifact tree contains a non-regular entry")
    return files


def _verify_download_root_inventory(root: Path, expected: set[str], label: str) -> None:
    directory = Path(root).absolute()
    if directory.is_symlink() or not directory.is_dir():
        raise InputError(f"{label} download root must be a non-symlink directory")
    observed: set[str] = set()
    for path in directory.rglob("*"):
        relative = path.relative_to(directory)
        if relative.parts and relative.parts[0] == ".cache":
            continue
        if path.is_symlink():
            raise InputError(f"{label} download root contains a symbolic link")
        if path.is_file():
            observed.add(relative.as_posix())
        elif not path.is_dir():
            raise InputError(f"{label} download root contains a non-regular entry")
    if observed != expected:
        raise InputError(
            f"{label} download root exact artifact inventory drifted",
            details={"missing": sorted(expected - observed), "unexpected": sorted(observed - expected)},
        )


def _verify_sha256sums(root: Path, expected: set[str] | frozenset[str]) -> None:
    checksum = _regular_file(root / "SHA256SUMS", "SHA256SUMS")
    paths: list[str] = []
    for line_number, line in enumerate(checksum.read_text(encoding="utf-8").splitlines(), start=1):
        if len(line) < 67 or line[64:66] != "  ":
            raise InputError("SHA256SUMS entry is malformed", details={"line": line_number})
        digest, relative = line[:64], line[66:]
        if not re.fullmatch(r"[0-9a-f]{64}", digest):
            raise InputError("SHA256SUMS digest is not canonical", details={"line": line_number})
        _safe_relative_text(relative, "SHA256SUMS path")
        paths.append(relative)
        path = _regular_file(root / relative, "checksummed artifact")
        if sha256_file(path) != "sha256:" + digest:
            raise InputError("SHA256SUMS artifact digest mismatch", details={"path": relative})
    if paths != sorted(expected):
        raise InputError("SHA256SUMS does not close the exact sorted artifact inventory")


def _write_snapshot_checksums(root: Path) -> None:
    inventory = _inventory(root) - {"SHA256SUMS"}
    lines = [
        f"{sha256_file(root / relative).removeprefix('sha256:')}  {relative}"
        for relative in sorted(inventory)
    ]
    (root / "SHA256SUMS").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _copy_regular_file(source: Path, target: Path) -> None:
    src = _regular_file(source, "snapshot input file")
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists() or target.is_symlink():
        raise InputError("snapshot target already exists", details={"path": str(target)})
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    source_fd = os.open(src, flags)
    target_fd: int | None = None
    try:
        metadata = os.fstat(source_fd)
        if not stat.S_ISREG(metadata.st_mode):
            raise InputError("snapshot input changed from a regular file during capture")
        target_fd = os.open(target, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
        while True:
            chunk = os.read(source_fd, 1 << 20)
            if not chunk:
                break
            view = memoryview(chunk)
            while view:
                written = os.write(target_fd, view)
                view = view[written:]
        os.fsync(target_fd)
    finally:
        os.close(source_fd)
        if target_fd is not None:
            os.close(target_fd)


def _copy_and_verify_identity(
    source: Path, target: Path, identity: Mapping[str, object]
) -> None:
    _copy_regular_file(source, target)
    if sha256_file(target) != identity["sha256"] or target.stat().st_size != identity["size_bytes"]:
        raise InputError("copied artifact identity differs from its evidence report")


def _file_identity(path: Path, relative: str) -> dict[str, object]:
    regular = _regular_file(path, "artifact identity input")
    return {
        "path": relative,
        "sha256": sha256_file(regular),
        "size_bytes": regular.stat().st_size,
    }


def _file_identity_mapping(raw: object, label: str) -> dict[str, object]:
    item = _mapping(raw, label)
    if set(item) != {"path", "sha256", "size_bytes"}:
        raise InputError(f"{label} identity fields are invalid")
    path = _safe_relative_text(str(item["path"]), f"{label} path")
    digest = str(item["sha256"])
    if not looks_like_sha256(digest):
        raise InputError(f"{label} sha256 is invalid")
    return {
        "path": path,
        "sha256": digest,
        "size_bytes": _positive_int(item["size_bytes"], f"{label} size_bytes"),
    }


def _match_identity(
    raw: object,
    path: Path,
    relative: str,
    *,
    include_records: bool = False,
) -> None:
    item = _mapping(raw, f"identity for {relative}")
    expected = _file_identity(path, relative)
    for field, value in expected.items():
        if item.get(field) != value:
            raise InputError("artifact identity mismatch", details={"path": relative, "field": field})
    if include_records:
        records = item.get("records")
        if isinstance(records, bool) or not isinstance(records, int) or records < 1:
            raise InputError("transformation output records must be positive")


def _public_source_identity(source: Mapping[str, object]) -> dict[str, object]:
    keys = ("repository", "revision", "artifact_path", "sha256", "size_bytes")
    payload = {key: source[key] for key in keys}
    if source.get("chromosome") is not None:
        payload["chromosome"] = source["chromosome"]
    return payload


def _evidence_file(path: str, description: str) -> dict[str, object]:
    return {"path": path, "artifact_role": "evidence", "description": description}


def _json_object(path: Path, label: str) -> dict[str, Any]:
    regular = _regular_file(path, label)
    try:
        payload = json.loads(regular.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise InputError(f"{label} is not valid UTF-8 JSON") from exc
    if not isinstance(payload, dict):
        raise InputError(f"{label} must be a JSON object")
    return payload


def _write_json(path: Path, payload: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _mapping(raw: object, label: str) -> dict[str, Any]:
    if not isinstance(raw, dict):
        raise InputError(f"{label} must be an object")
    return raw


def _positive_int(raw: object, label: str) -> int:
    if isinstance(raw, bool) or not isinstance(raw, int) or raw < 1:
        raise InputError(f"{label} must be a positive integer")
    return raw


def _repository(value: str, label: str) -> str:
    if not _REPOSITORY_RE.fullmatch(value):
        raise InputError(f"{label} must be an owner/name repository")
    return value


def _commit(value: str, label: str) -> str:
    if not _COMMIT_RE.fullmatch(value):
        raise InputError(f"{label} must be a full lowercase 40-character commit")
    return value


def _safe_relative_text(value: str, label: str) -> str:
    if not isinstance(value, str) or not _SAFE_COMPONENT_RE.fullmatch(value):
        raise InputError(f"{label} must be a canonical safe relative POSIX path")
    pure = PurePosixPath(value)
    if pure.is_absolute() or ".." in pure.parts or "\\" in value:
        raise InputError(f"{label} must be a canonical safe relative POSIX path")
    return value


def _fsync_tree(root: Path) -> None:
    for path in sorted(root.rglob("*"), reverse=True):
        if path.is_file():
            descriptor = os.open(path, os.O_RDONLY)
            try:
                os.fsync(descriptor)
            finally:
                os.close(descriptor)
        elif path.is_dir():
            _fsync_directory(path)
    _fsync_directory(root)


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _publish_directory_noreplace(source: Path, target: Path) -> None:
    if target.exists() or target.is_symlink():
        raise InputError("immutable snapshot output appeared before publication")
    libc = ctypes.CDLL(None, use_errno=True)
    source_bytes = os.fsencode(source)
    target_bytes = os.fsencode(target)
    if sys.platform == "darwin" and hasattr(libc, "renamex_np"):
        result = libc.renamex_np(source_bytes, target_bytes, ctypes.c_uint(0x00000004))
        if result == 0:
            return
    elif hasattr(libc, "renameat2"):
        result = libc.renameat2(
            ctypes.c_int(-100),
            source_bytes,
            ctypes.c_int(-100),
            target_bytes,
            ctypes.c_uint(1),
        )
        if result == 0:
            return
    else:
        source.rename(target)
        return
    error = ctypes.get_errno()
    if error in {errno.EEXIST, errno.ENOTEMPTY}:
        raise InputError("immutable snapshot output already exists")
    raise OSError(error, os.strerror(error), str(target))
