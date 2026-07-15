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
from datetime import datetime, timezone
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
from tools.release.dataset_snapshot import (
    GENERATED_BY as DATASET_SNAPSHOT_GENERATED_BY,
    INPUT_CHECK_GENERATED_BY as DATASET_INPUT_CHECK_GENERATED_BY,
)

REPORT_SCHEMA_VERSION: Final = "1.0.0"
V03_PROVENANCE_SCHEMA_VERSION: Final = "geno-lewm.v03-dataset-snapshot.v1"
GENERATED_BY: Final = "tools.release.v03_dataset_snapshot"
REPORT_NAME: Final = "dataset_snapshot_report.json"
INPUT_CHECK_REPORT_NAME: Final = "dataset_input_check_report.json"
SNAPSHOT_SPEC_NAME: Final = "v03_dataset_snapshot_spec.json"

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
    sorted(
        path for path in _MEMBERSHIP_FILES if path != "SHA256SUMS" and not path.startswith("store/")
    )
)
_TRAIN_CHROMOSOMES: Final = (*map(str, range(1, 20)), "22")
_COMMIT_RE: Final = re.compile(r"[0-9a-f]{40}")
_REPOSITORY_RE: Final = re.compile(r"[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+")
_IMAGE_RE: Final = re.compile(r"[^@\s]+@sha256:[0-9a-f]{64}")
_SAFE_COMPONENT_RE: Final = re.compile(r"[A-Za-z0-9._-]+(?:/[A-Za-z0-9._-]+)*")
_ISO_UTC_RE: Final = re.compile(
    r"[0-9]{4}-[0-9]{2}-[0-9]{2}T[0-9]{2}:[0-9]{2}:[0-9]{2}(?:\.[0-9]{1,6})?Z"
)
_V03_CHECKS: Final = (
    "exact_immutable_revisions",
    "complete_bundle_inventories",
    "bundle_checksum_closure",
    "prepared_source_identities",
    "verified_membership_store",
    "publication_eligible_split_evidence",
    "training_window_identity_and_cardinality",
    "exact_train_membership_source_row_ids",
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
        try:
            if not observed.schema_arrow.equals(schema, check_metadata=True):
                raise InputError("filtered Parquet did not preserve the exact source Arrow schema")
            if int(observed.metadata.num_rows) != written:
                raise InputError("filtered Parquet footer cardinality drifted")
        finally:
            observed.close()
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
            report = verify_v03_dataset_snapshot(
                args.dataset_dir,
                gnomad_root=args.gnomad_root,
                clinvar_root=args.clinvar_root,
            )
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
    verify.add_argument("--gnomad-root", type=Path)
    verify.add_argument("--clinvar-root", type=Path)
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
    if not isinstance(snapshot_id, str) or not re.fullmatch(
        r"[A-Za-z0-9][A-Za-z0-9._-]*", snapshot_id
    ):
        raise InputError("snapshot_id must be a safe immutable identifier")
    generated_at = _utc_timestamp(generated_at, "generated_at")

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
        snapshot_id=snapshot_id,
        gnomad_root=Path(gnomad_root),
        clinvar_root=Path(clinvar_root),
    )
    training_windows = _validate_training_windows(split_report, Path(training_windows_path))
    membership_origin = _validate_split_origin(
        split_report,
        split_repository=split_repository,
        split_revision=split_revision,
        split_artifact_path=split_artifact_path,
        snapshot_id=snapshot_id,
    )
    membership_bundle_origin = {
        "repository": membership_origin["repository"],
        "revision": membership_origin["revision"],
        "artifact_path": membership_origin["artifact_path"],
        "sha256": sha256_file(membership_root / "SHA256SUMS"),
        "size_bytes": (membership_root / "SHA256SUMS").stat().st_size,
    }
    split_bundle_origin = {
        "repository": split_repository,
        "revision": split_revision,
        "artifact_path": split_artifact_path,
        "sha256": sha256_file(split_root / "SHA256SUMS"),
        "size_bytes": (split_root / "SHA256SUMS").stat().st_size,
    }

    temporary = Path(tempfile.mkdtemp(prefix=f".{output.name}.tmp-", dir=output.parent))
    try:
        source_bindings: dict[str, dict[str, object]] = {}
        store_root = membership_root / "store"
        with MembershipStore.open(store_root, verify=True) as store:
            _validate_store_binding(store.manifest, split_report, snapshot_id=snapshot_id)
            selections = _train_source_row_ids(store, sources)
            staged_files: list[dict[str, object]] = []
            transformations: list[dict[str, object]] = []

            for name in _STORE_FILES:
                relative = f"membership/store/{name}"
                source_path = store_root / name
                _copy_regular_file(source_path, temporary / relative)
                staged_files.append(_evidence_file(relative, "Verified membership store"))
                source_bindings[relative] = _source_binding(
                    source_path,
                    _hub_source_path(
                        str(membership_origin["repository"]),
                        str(membership_origin["revision"]),
                        f"{membership_origin['artifact_path']}/store/{name}",
                    ),
                )

            for source in sources:
                source_id = str(source["source_id"])
                selected = selections.get(source_id, frozenset())
                if source["kind"] == "gnomad" and source["chromosome"] not in _TRAIN_CHROMOSOMES:
                    if selected:
                        raise InputError(
                            "held-out gnomAD source unexpectedly has train memberships"
                        )
                    continue
                if not selected:
                    raise InputError(
                        "train membership source is empty", details={"source_id": source_id}
                    )
                if source["kind"] == "gnomad":
                    relative = f"gnomad/v4.1/train/chr{source['chromosome']}.variants.parquet"
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
                source_bindings[relative] = _source_binding(
                    Path(str(source["local_path"])),
                    _hub_source_path(
                        str(source["repository"]),
                        str(source["revision"]),
                        str(source["artifact_path"]),
                    ),
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
        _copy_regular_file(Path(str(training_windows["local_path"])), temporary / windows_relative)
        staged_files.append(
            {
                "path": windows_relative,
                "artifact_role": "split_data",
                "split": str(training_windows["split"]),
                "records": _positive_int(training_windows["records"], "training records"),
                "description": "Placed gnomAD common-variant training windows audited against held roles",
            }
        )
        source_bindings[windows_relative] = _source_binding(
            Path(str(training_windows["local_path"])),
            _hub_source_path(
                str(training_windows["repository"]),
                str(training_windows["revision"]),
                str(training_windows["artifact_path"]),
            ),
        )

        for role in ("validation", "evaluation"):
            stream = _mapping(split_report.get("streams"), f"split report streams.{role}")[role]
            stream = _mapping(stream, f"split report stream {role}")
            records = _positive_int(stream.get("record_count"), f"{role} record_count")
            labels = _file_identity_mapping(stream.get("labels_jsonl"), f"{role} labels")
            vcf = _file_identity_mapping(stream.get("vcf"), f"{role} vcf")
            labels_relative = str(labels["path"])
            vcf_relative = str(vcf["path"])
            _copy_and_verify_identity(
                split_root / labels_relative, temporary / labels_relative, labels
            )
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
            source_bindings[labels_relative] = _source_binding(
                split_root / labels_relative,
                _hub_source_path(
                    split_repository,
                    split_revision,
                    f"{split_artifact_path}/{labels_relative}",
                ),
            )
            source_bindings[vcf_relative] = _source_binding(
                split_root / vcf_relative,
                _hub_source_path(
                    split_repository,
                    split_revision,
                    f"{split_artifact_path}/{vcf_relative}",
                ),
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
        for relative in (split_schema_relative, split_report_relative):
            source_bindings[relative] = _source_binding(
                split_root / relative,
                _hub_source_path(
                    split_repository,
                    split_revision,
                    f"{split_artifact_path}/{relative}",
                ),
            )

        membership_checksum_relative = "evidence/membership-provenance/SHA256SUMS"
        _copy_regular_file(membership_root / "SHA256SUMS", temporary / membership_checksum_relative)
        staged_files.append(
            _evidence_file(
                membership_checksum_relative,
                "Relocated membership checksum record; paths remain relative to the original success-bundle root",
            )
        )
        source_bindings[membership_checksum_relative] = _source_binding(
            membership_root / "SHA256SUMS",
            _hub_source_path(
                str(membership_origin["repository"]),
                str(membership_origin["revision"]),
                f"{membership_origin['artifact_path']}/SHA256SUMS",
            ),
        )
        for relative_source in _MEMBERSHIP_PROVENANCE_FILES:
            relative_target = f"evidence/membership-provenance/{relative_source}"
            _copy_regular_file(membership_root / relative_source, temporary / relative_target)
            staged_files.append(_evidence_file(relative_target, "Membership build provenance"))
            source_bindings[relative_target] = _source_binding(
                membership_root / relative_source,
                _hub_source_path(
                    str(membership_origin["repository"]),
                    str(membership_origin["revision"]),
                    f"{membership_origin['artifact_path']}/{relative_source}",
                ),
            )
        split_checksum_relative = "evidence/split-provenance/SHA256SUMS"
        _copy_regular_file(split_root / "SHA256SUMS", temporary / split_checksum_relative)
        staged_files.append(
            _evidence_file(
                split_checksum_relative,
                "Relocated membership-split checksum record; paths remain relative to the original success-bundle root",
            )
        )
        source_bindings[split_checksum_relative] = _source_binding(
            split_root / "SHA256SUMS",
            _hub_source_path(
                split_repository,
                split_revision,
                f"{split_artifact_path}/SHA256SUMS",
            ),
        )

        snapshot_spec_payload = _snapshot_spec_payload(
            snapshot_id=snapshot_id,
            generated_at=generated_at,
            producer_git_commit=producer_git_commit,
            container_image=container_image,
            membership_origin=membership_bundle_origin,
            split_origin=split_bundle_origin,
            sources=sources,
            training_windows=training_windows,
        )
        snapshot_spec_path = temporary / SNAPSHOT_SPEC_NAME
        _write_json(snapshot_spec_path, snapshot_spec_payload)
        staged_files.append(
            _evidence_file(SNAPSHOT_SPEC_NAME, "Canonical v0.3 snapshot assembly spec")
        )
        source_bindings[SNAPSHOT_SPEC_NAME] = _source_binding(
            snapshot_spec_path, f"generated/{SNAPSHOT_SPEC_NAME}"
        )
        snapshot_files = _snapshot_files(temporary, staged_files, source_bindings)
        snapshot_spec_identity = _file_identity(snapshot_spec_path, SNAPSHOT_SPEC_NAME)
        input_check = _input_check_payload(
            snapshot_id=snapshot_id,
            snapshot_spec=snapshot_spec_identity,
            snapshot_files=snapshot_files,
            v03=snapshot_spec_payload,
        )
        _write_json(temporary / INPUT_CHECK_REPORT_NAME, input_check)

        store_manifest = _json_object(
            membership_root / "store/manifest.json", "membership manifest"
        )
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
                "At assembly time, retain source Arrow schemas, metadata, columns, values, and row order while filtering pinned Parquets by exact train source_row_id.",
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
                    "records": int(
                        _mapping(split_streams["validation"], "validation stream")["record_count"]
                    ),
                    "description": "Held-out ClinVar chromosome-20 labels",
                },
                "evaluation": {
                    "records": int(
                        _mapping(split_streams["evaluation"], "evaluation stream")["record_count"]
                    ),
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
                "Standalone snapshot verification replays exact variant row identities and package integrity; it does not re-download prepared upstream Parquets to independently replay non-identity column values.",
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

        package_membership_evidence = package.membership_and_split_evidence
        if package_membership_evidence is None:
            raise InputError("v0.3 dataset package lost its membership and split binding")
        snapshot_payload: dict[str, object] = {
            "schema_version": REPORT_SCHEMA_VERSION,
            "generated_by": DATASET_SNAPSHOT_GENERATED_BY,
            "generated_at": generated_at,
            "snapshot_id": snapshot_id,
            "report_path": REPORT_NAME,
            "snapshot_spec": snapshot_spec_identity,
            "input_check_path": INPUT_CHECK_REPORT_NAME,
            "input_check": _file_identity(
                temporary / INPUT_CHECK_REPORT_NAME, INPUT_CHECK_REPORT_NAME
            ),
            "metadata_path": "dataset_package.json",
            "package": {
                "schema_version": package_report.schema_version,
                "generated_by": DATASET_PACKAGE_GENERATED_BY,
                "snapshot_id": snapshot_id,
                "metadata": _file_identity(metadata_path, "dataset_package.json"),
                "manifest_path": "dataset_manifest.json",
                "manifest": _file_identity(
                    temporary / "dataset_manifest.json", "dataset_manifest.json"
                ),
                "data_card_path": "data_card.md",
                "data_card": _file_identity(temporary / "data_card.md", "data_card.md"),
                "integrity_path": DEFAULT_REPORT_NAME,
                "integrity": _file_identity(integrity_path, DEFAULT_REPORT_NAME),
                "checksums_path": "SHA256SUMS",
                "files": [file.to_dict() for file in package_report.files],
                "membership_and_split_evidence": package_membership_evidence.to_dict(),
            },
            "files": snapshot_files,
            "v03": {
                **snapshot_spec_payload,
                "transformations": transformations,
                "observed_splits": integrity_payload["splits"],
                "claim_boundary": {
                    "variant_membership": True,
                    "phased_haplotype_membership": False,
                    "standalone_upstream_nonidentity_value_replay": False,
                    "publication_eligible": True,
                    "released_v03_snapshot": False,
                    "limitations": metadata["limitations"],
                },
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


def verify_v03_dataset_snapshot(
    dataset_dir: Path,
    *,
    gnomad_root: Path | None = None,
    clinvar_root: Path | None = None,
) -> V03SnapshotReport:
    """Verify a snapshot, optionally replaying all train outputs from pinned sources."""
    if (gnomad_root is None) != (clinvar_root is None):
        raise InputError("strict upstream replay requires both gnomad_root and clinvar_root")
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
    snapshot = _json_object(root / REPORT_NAME, "dataset snapshot report")
    _verify_input_check(
        input_check,
        root=root,
        package_snapshot_id=package.snapshot_id,
        snapshot_report=snapshot,
    )
    _verify_snapshot_report(
        snapshot,
        root=root,
        package_snapshot_id=package.snapshot_id,
        generated_at=package.generated_at,
        input_check=input_check,
    )
    if gnomad_root is not None and clinvar_root is not None:
        _strict_replay_train_outputs(
            root,
            snapshot=snapshot,
            gnomad_root=gnomad_root,
            clinvar_root=clinvar_root,
            snapshot_id=package.snapshot_id,
        )
    return V03SnapshotReport(root, snapshot)


def _validate_source_contract(
    payload: dict[str, Any],
    *,
    membership_spec: dict[str, Any],
    membership_job: dict[str, Any],
    snapshot_id: str,
    gnomad_root: Path,
    clinvar_root: Path,
) -> tuple[dict[str, object], ...]:
    if payload.get("ok") is not True or payload.get("source_count") != 23:
        raise InputError("membership source identity report is not a successful 23-source report")
    if payload.get("candidate_snapshot_id") != snapshot_id:
        raise InputError(
            "membership source candidate_snapshot_id differs from the requested snapshot_id"
        )
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


def _validate_training_windows(split_report: dict[str, Any], local_path: Path) -> dict[str, object]:
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
    snapshot_id: str,
) -> dict[str, object]:
    store = _mapping(split_report.get("membership_store"), "split report membership_store")
    lineage = _mapping(store.get("lineage"), "split report membership_store.lineage")
    if lineage.get("candidate_snapshot_id") != snapshot_id:
        raise InputError(
            "membership split lineage candidate_snapshot_id differs from the requested snapshot_id"
        )
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


def _validate_store_binding(
    manifest: object, split_report: dict[str, Any], *, snapshot_id: str
) -> None:
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
    lineage = getattr(manifest, "snapshot_lineage", None)
    if lineage is None or getattr(lineage, "candidate_snapshot_id", None) != snapshot_id:
        raise InputError(
            "verified membership store candidate_snapshot_id differs from the requested snapshot_id"
        )


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


def _snapshot_spec_payload(
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
    return {
        "schema_version": V03_PROVENANCE_SCHEMA_VERSION,
        "generated_by": GENERATED_BY,
        "generated_at": generated_at,
        "snapshot_id": snapshot_id,
        "producer": {
            "git_commit": producer_git_commit,
            "container_image": container_image,
        },
        "upstreams": _v03_upstreams(
            membership_origin=membership_origin,
            split_origin=split_origin,
            sources=sources,
            training_windows=training_windows,
        ),
        "checks": list(_V03_CHECKS),
    }


def _v03_upstreams(
    *,
    membership_origin: Mapping[str, object],
    split_origin: Mapping[str, object],
    sources: Sequence[Mapping[str, object]],
    training_windows: Mapping[str, object],
) -> list[dict[str, object]]:
    upstreams: list[dict[str, object]] = [
        {
            "kind": "membership_bundle",
            **{
                key: membership_origin[key]
                for key in ("repository", "revision", "artifact_path", "sha256", "size_bytes")
            },
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
    return upstreams


def _input_check_payload(
    *,
    snapshot_id: str,
    snapshot_spec: Mapping[str, object],
    snapshot_files: Sequence[Mapping[str, object]],
    v03: Mapping[str, object],
) -> dict[str, object]:
    inputs: list[dict[str, object]] = []
    for raw in snapshot_files:
        item = {
            "kind": "dataset_artifact",
            "source_path": raw["source_path"],
            "staged_path": raw["path"],
            "sha256": raw["source_sha256"],
            "size_bytes": raw["source_size_bytes"],
        }
        item.update(
            {
                key: raw[key]
                for key in ("split", "artifact_role", "companion_of", "description")
                if key in raw
            }
        )
        inputs.append(item)
    return {
        "ok": True,
        "schema_version": REPORT_SCHEMA_VERSION,
        "generated_by": DATASET_INPUT_CHECK_GENERATED_BY,
        "snapshot_id": snapshot_id,
        "snapshot_spec": dict(snapshot_spec),
        "source_count": len(inputs),
        "total_size_bytes": sum(
            _positive_int(item["size_bytes"], "snapshot input size_bytes") for item in inputs
        ),
        "inputs": inputs,
        "v03": dict(v03),
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


def _source_binding(path: Path, public_path: str) -> dict[str, object]:
    source = _regular_file(path, "snapshot source-binding input")
    return {
        "source_path": _safe_relative_text(public_path, "snapshot source_path"),
        "source_sha256": sha256_file(source),
        "source_size_bytes": source.stat().st_size,
    }


def _hub_source_path(repository: str, revision: str, artifact_path: str) -> str:
    repository = _repository(repository, "source repository")
    revision = _commit(revision, "source revision")
    artifact_path = _safe_relative_text(artifact_path, "source artifact_path")
    return _safe_relative_text(
        f"hub/datasets/{repository}/{revision}/{artifact_path}", "public Hub source path"
    )


def _snapshot_files(
    root: Path,
    staged_files: Sequence[Mapping[str, object]],
    source_bindings: Mapping[str, Mapping[str, object]],
) -> list[dict[str, object]]:
    paths = [str(item.get("path")) for item in staged_files]
    if len(paths) != len(set(paths)):
        raise InputError("v0.3 snapshot staged paths are duplicated")
    if set(paths) != set(source_bindings):
        raise InputError("v0.3 snapshot source bindings do not close the staged inventory")
    files: list[dict[str, object]] = []
    semantic_fields = ("split", "records", "artifact_role", "companion_of", "description")
    for staged in staged_files:
        relative = _safe_relative_text(str(staged.get("path")), "snapshot staged path")
        binding = _mapping(source_bindings[relative], f"source binding for {relative}")
        source_path = _safe_relative_text(
            str(binding.get("source_path")), f"source path for {relative}"
        )
        source_sha256 = str(binding.get("source_sha256"))
        if not looks_like_sha256(source_sha256):
            raise InputError("snapshot source binding sha256 is invalid")
        source_size_bytes = _positive_int(
            binding.get("source_size_bytes"), "snapshot source binding size_bytes"
        )
        files.append(
            {
                "path": relative,
                "source_path": source_path,
                "source_sha256": source_sha256,
                "source_size_bytes": source_size_bytes,
                **_file_identity(root / relative, relative),
                "already_exists": False,
                **{key: staged[key] for key in semantic_fields if key in staged},
            }
        )
    return files


def _verify_input_check(
    payload: dict[str, Any],
    *,
    root: Path,
    package_snapshot_id: str,
    snapshot_report: Mapping[str, object],
) -> None:
    expected_top = {
        "ok",
        "schema_version",
        "generated_by",
        "snapshot_id",
        "snapshot_spec",
        "source_count",
        "total_size_bytes",
        "inputs",
        "v03",
    }
    if set(payload) != expected_top:
        raise InputError("snapshot input-check fields do not match the closed compatible schema")
    if (
        payload.get("ok") is not True
        or payload.get("schema_version") != REPORT_SCHEMA_VERSION
        or payload.get("generated_by") != DATASET_INPUT_CHECK_GENERATED_BY
        or payload.get("snapshot_id") != package_snapshot_id
    ):
        raise InputError("snapshot input-check identity is invalid")
    snapshot_spec = payload.get("snapshot_spec")
    if snapshot_spec != snapshot_report.get("snapshot_spec"):
        raise InputError("snapshot input-check and snapshot report spec identities differ")
    _match_identity(snapshot_spec, root / SNAPSHOT_SPEC_NAME, SNAPSHOT_SPEC_NAME)
    spec = _json_object(root / SNAPSHOT_SPEC_NAME, "v0.3 dataset snapshot spec")
    _verify_v03_spec(spec, snapshot_id=package_snapshot_id)
    if payload.get("v03") != spec:
        raise InputError("snapshot input-check v0.3 provenance differs from the bound spec")
    raw_files = snapshot_report.get("files")
    if not isinstance(raw_files, list) or not raw_files:
        raise InputError("snapshot input-check requires snapshot report files")
    expected_inputs = [
        _input_entry_from_snapshot_file(_mapping(raw, "snapshot report file")) for raw in raw_files
    ]
    if payload.get("inputs") != expected_inputs:
        raise InputError(
            "snapshot input-check inputs differ from snapshot report source identities"
        )
    if payload.get("source_count") != len(expected_inputs):
        raise InputError("snapshot input-check source_count is stale")
    expected_size = sum(
        _positive_int(item["size_bytes"], "snapshot input size_bytes") for item in expected_inputs
    )
    if payload.get("total_size_bytes") != expected_size:
        raise InputError("snapshot input-check total_size_bytes is stale")


def _input_entry_from_snapshot_file(raw: Mapping[str, object]) -> dict[str, object]:
    item = {
        "kind": "dataset_artifact",
        "source_path": raw.get("source_path"),
        "staged_path": raw.get("path"),
        "sha256": raw.get("source_sha256"),
        "size_bytes": raw.get("source_size_bytes"),
    }
    item.update(
        {
            key: raw[key]
            for key in ("split", "artifact_role", "companion_of", "description")
            if key in raw
        }
    )
    return item


def _verify_v03_spec(payload: Mapping[str, object], *, snapshot_id: str) -> None:
    expected_top = {
        "schema_version",
        "generated_by",
        "generated_at",
        "snapshot_id",
        "producer",
        "upstreams",
        "checks",
    }
    if set(payload) != expected_top:
        raise InputError("v0.3 snapshot spec fields do not match the closed schema")
    if (
        payload.get("schema_version") != V03_PROVENANCE_SCHEMA_VERSION
        or payload.get("generated_by") != GENERATED_BY
        or payload.get("snapshot_id") != snapshot_id
        or payload.get("checks") != list(_V03_CHECKS)
    ):
        raise InputError("v0.3 snapshot spec identity or checks are invalid")
    _utc_timestamp(payload.get("generated_at"), "v0.3 snapshot generated_at")
    producer = _mapping(payload.get("producer"), "v0.3 snapshot producer")
    if set(producer) != {"git_commit", "container_image"}:
        raise InputError("v0.3 snapshot producer fields are invalid")
    _commit(str(producer.get("git_commit")), "v0.3 producer git_commit")
    if not _IMAGE_RE.fullmatch(str(producer.get("container_image"))):
        raise InputError("v0.3 snapshot producer image is not digest pinned")
    upstreams = payload.get("upstreams")
    if not isinstance(upstreams, list) or len(upstreams) != 26:
        raise InputError("v0.3 snapshot spec must bind exactly 26 upstream artifacts")
    kinds: list[str] = []
    gnomad_chromosomes: set[str] = set()
    for raw in upstreams:
        item = _mapping(raw, "v0.3 snapshot upstream")
        kind = str(item.get("kind"))
        kinds.append(kind)
        required = {"kind", "repository", "revision", "artifact_path", "sha256", "size_bytes"}
        optional: set[str] = set()
        if kind == "prepared_gnomad":
            optional = {"chromosome"}
            chromosome = str(item.get("chromosome"))
            if chromosome not in {str(value) for value in range(1, 23)}:
                raise InputError("v0.3 prepared gnomAD upstream chromosome is invalid")
            gnomad_chromosomes.add(chromosome)
        elif kind == "training_windows":
            optional = {"records"}
            _positive_int(item.get("records"), "training-window upstream records")
        elif kind not in {"membership_bundle", "split_bundle", "prepared_clinvar"}:
            raise InputError("v0.3 snapshot upstream kind is invalid")
        if set(item) != required | optional:
            raise InputError("v0.3 snapshot upstream fields are invalid")
        _repository(str(item.get("repository")), "v0.3 upstream repository")
        _commit(str(item.get("revision")), "v0.3 upstream revision")
        _safe_relative_text(str(item.get("artifact_path")), "v0.3 upstream artifact_path")
        if not looks_like_sha256(str(item.get("sha256"))):
            raise InputError("v0.3 snapshot upstream sha256 is invalid")
        _positive_int(item.get("size_bytes"), "v0.3 upstream size_bytes")
    if (
        kinds.count("membership_bundle") != 1
        or kinds.count("split_bundle") != 1
        or kinds.count("prepared_gnomad") != 22
        or kinds.count("prepared_clinvar") != 1
        or kinds.count("training_windows") != 1
        or gnomad_chromosomes != {str(value) for value in range(1, 23)}
    ):
        raise InputError("v0.3 snapshot upstream set is incomplete or duplicated")


def _verify_snapshot_report(
    payload: dict[str, Any],
    *,
    root: Path,
    package_snapshot_id: str,
    generated_at: str,
    input_check: Mapping[str, object],
) -> None:
    expected_top = {
        "schema_version",
        "generated_by",
        "generated_at",
        "snapshot_id",
        "report_path",
        "snapshot_spec",
        "input_check_path",
        "input_check",
        "metadata_path",
        "package",
        "files",
        "v03",
    }
    if set(payload) != expected_top:
        raise InputError("dataset snapshot report fields do not match the closed compatible schema")
    if (
        payload.get("schema_version") != REPORT_SCHEMA_VERSION
        or payload.get("generated_by") != DATASET_SNAPSHOT_GENERATED_BY
        or payload.get("generated_at") != generated_at
        or payload.get("snapshot_id") != package_snapshot_id
        or payload.get("report_path") != REPORT_NAME
        or payload.get("input_check_path") != INPUT_CHECK_REPORT_NAME
        or payload.get("metadata_path") != "dataset_package.json"
    ):
        raise InputError("dataset snapshot report identity is invalid")
    _match_identity(payload.get("snapshot_spec"), root / SNAPSHOT_SPEC_NAME, SNAPSHOT_SPEC_NAME)
    _match_identity(
        payload.get("input_check"), root / INPUT_CHECK_REPORT_NAME, INPUT_CHECK_REPORT_NAME
    )
    if payload.get("snapshot_spec") != input_check.get("snapshot_spec"):
        raise InputError("dataset snapshot report and input-check spec identities differ")

    manifest = _json_object(root / "dataset_manifest.json", "dataset manifest")
    raw_manifest_files = manifest.get("files")
    if not isinstance(raw_manifest_files, list) or not raw_manifest_files:
        raise InputError("dataset manifest files must be a non-empty list")
    snapshot_files = _verified_snapshot_files(
        payload.get("files"), root=root, manifest_files=raw_manifest_files
    )
    expected_package = {
        "schema_version": ARTIFACT_ROLE_SCHEMA_VERSION,
        "generated_by": DATASET_PACKAGE_GENERATED_BY,
        "snapshot_id": package_snapshot_id,
        "metadata": _file_identity(root / "dataset_package.json", "dataset_package.json"),
        "manifest_path": "dataset_manifest.json",
        "manifest": _file_identity(root / "dataset_manifest.json", "dataset_manifest.json"),
        "data_card_path": "data_card.md",
        "data_card": _file_identity(root / "data_card.md", "data_card.md"),
        "integrity_path": DEFAULT_REPORT_NAME,
        "integrity": _file_identity(root / DEFAULT_REPORT_NAME, DEFAULT_REPORT_NAME),
        "checksums_path": "SHA256SUMS",
        "files": raw_manifest_files,
        "membership_and_split_evidence": manifest.get("membership_and_split_evidence"),
    }
    if payload.get("package") != expected_package:
        raise InputError("dataset snapshot report package block is stale or incomplete")

    spec = _json_object(root / SNAPSHOT_SPEC_NAME, "v0.3 dataset snapshot spec")
    extension = _mapping(payload.get("v03"), "dataset snapshot report v03 extension")
    if set(extension) != set(spec) | {"transformations", "observed_splits", "claim_boundary"}:
        raise InputError("dataset snapshot report v03 extension fields are invalid")
    if any(extension.get(key) != value for key, value in spec.items()):
        raise InputError("dataset snapshot report v03 extension differs from the bound spec")
    _verify_snapshot_source_bindings(snapshot_files, spec=spec)
    _verify_relocated_bundle_checksums(root)
    integrity = _json_object(root / DEFAULT_REPORT_NAME, "split integrity report")
    if extension.get("observed_splits") != integrity.get("splits"):
        raise InputError("dataset snapshot report observed splits differ from split integrity")
    claim = _mapping(extension.get("claim_boundary"), "snapshot claim boundary")
    expected_claim = {
        "variant_membership": True,
        "phased_haplotype_membership": False,
        "standalone_upstream_nonidentity_value_replay": False,
        "publication_eligible": True,
        "released_v03_snapshot": False,
    }
    if {field: claim.get(field) for field in expected_claim} != expected_claim:
        raise InputError("dataset snapshot report claim boundary is invalid")
    limitations = claim.get("limitations")
    if (
        set(claim) != set(expected_claim) | {"limitations"}
        or not isinstance(limitations, list)
        or not limitations
    ):
        raise InputError("dataset snapshot report limitations must be non-empty")
    _verify_train_membership_outputs(
        root,
        snapshot_id=package_snapshot_id,
        spec=spec,
        extension=extension,
        snapshot_files=snapshot_files,
    )


def _verified_snapshot_files(
    raw: object,
    *,
    root: Path,
    manifest_files: Sequence[object],
) -> list[dict[str, object]]:
    if not isinstance(raw, list) or len(raw) != len(manifest_files):
        raise InputError("dataset snapshot report files differ from the manifest inventory")
    verified: list[dict[str, object]] = []
    source_fields = {"source_path", "source_sha256", "source_size_bytes", "already_exists"}
    for index, (raw_snapshot, raw_manifest) in enumerate(zip(raw, manifest_files, strict=True)):
        item = _mapping(raw_snapshot, f"snapshot report files[{index}]")
        manifest = _mapping(raw_manifest, f"dataset manifest files[{index}]")
        if set(item) != set(manifest) | source_fields:
            raise InputError("dataset snapshot report file fields are invalid")
        if any(item.get(key) != value for key, value in manifest.items()):
            raise InputError("dataset snapshot report file differs from dataset manifest")
        relative = _safe_relative_text(str(item.get("path")), "snapshot report file path")
        _safe_relative_text(str(item.get("source_path")), "snapshot report source_path")
        if not looks_like_sha256(str(item.get("source_sha256"))):
            raise InputError("dataset snapshot report source sha256 is invalid")
        _positive_int(item.get("source_size_bytes"), "snapshot report source_size_bytes")
        if item.get("already_exists") is not False:
            raise InputError("immutable v0.3 snapshot files must be newly staged")
        _match_identity(item, root / relative, relative)
        verified.append(dict(item))
    spec_files = [item for item in verified if item["path"] == SNAPSHOT_SPEC_NAME]
    if len(spec_files) != 1:
        raise InputError("v0.3 snapshot spec must be one package evidence file")
    spec_identity = _file_identity(root / SNAPSHOT_SPEC_NAME, SNAPSHOT_SPEC_NAME)
    if (
        spec_files[0].get("artifact_role") != "evidence"
        or spec_files[0].get("source_path") != f"generated/{SNAPSHOT_SPEC_NAME}"
        or spec_files[0].get("source_sha256") != spec_identity["sha256"]
        or spec_files[0].get("source_size_bytes") != spec_identity["size_bytes"]
    ):
        raise InputError("v0.3 snapshot spec evidence identity is invalid")
    return verified


def _verify_snapshot_source_bindings(
    snapshot_files: Sequence[Mapping[str, object]], *, spec: Mapping[str, object]
) -> None:
    upstreams = spec.get("upstreams")
    if not isinstance(upstreams, list):
        raise InputError("v0.3 snapshot spec upstreams are invalid")
    by_kind: dict[str, list[dict[str, Any]]] = {}
    for raw in upstreams:
        item = _mapping(raw, "v0.3 snapshot upstream")
        by_kind.setdefault(str(item["kind"]), []).append(item)
    membership = by_kind["membership_bundle"][0]
    split = by_kind["split_bundle"][0]
    windows = by_kind["training_windows"][0]
    prepared_by_output: dict[str, dict[str, Any]] = {}
    for source in (*by_kind["prepared_gnomad"], *by_kind["prepared_clinvar"]):
        if source["kind"] == "prepared_gnomad":
            chromosome = str(source["chromosome"])
            if chromosome not in _TRAIN_CHROMOSOMES:
                continue
            output = f"gnomad/v4.1/train/chr{chromosome}.variants.parquet"
        else:
            output = "clinvar/2026-04-15/train.variants.parquet"
        prepared_by_output[output] = source

    for snapshot_file in snapshot_files:
        relative = str(snapshot_file["path"])
        bound_source: Mapping[str, object] | None = None
        copied = False
        if relative == SNAPSHOT_SPEC_NAME:
            expected = {
                "source_path": f"generated/{SNAPSHOT_SPEC_NAME}",
                "source_sha256": snapshot_file["sha256"],
                "source_size_bytes": snapshot_file["size_bytes"],
            }
        elif relative in prepared_by_output:
            bound_source = prepared_by_output[relative]
            expected = _expected_hub_binding(bound_source)
        elif relative == str(windows["artifact_path"]):
            bound_source = windows
            copied = True
            expected = _expected_hub_binding(bound_source)
        elif relative.startswith("membership/store/"):
            suffix = relative.removeprefix("membership/")
            expected = _expected_bundle_copy_binding(membership, suffix, snapshot_file)
        elif relative == "evidence/membership-provenance/SHA256SUMS":
            expected = _expected_bundle_copy_binding(membership, "SHA256SUMS", snapshot_file)
            if membership.get("sha256") != snapshot_file.get("sha256") or membership.get(
                "size_bytes"
            ) != snapshot_file.get("size_bytes"):
                raise InputError("membership bundle checksum identity differs from its spec")
        elif relative.startswith("evidence/membership-provenance/"):
            suffix = relative.removeprefix("evidence/membership-provenance/")
            expected = _expected_bundle_copy_binding(membership, suffix, snapshot_file)
        elif relative == "evidence/split-provenance/SHA256SUMS":
            expected = _expected_bundle_copy_binding(split, "SHA256SUMS", snapshot_file)
            if split.get("sha256") != snapshot_file.get("sha256") or split.get(
                "size_bytes"
            ) != snapshot_file.get("size_bytes"):
                raise InputError("membership-split bundle checksum identity differs from its spec")
        elif relative in _SPLIT_FILES - {"SHA256SUMS"}:
            expected = _expected_bundle_copy_binding(split, relative, snapshot_file)
        else:
            raise InputError(
                "snapshot file has no exact source-binding rule", details={"path": relative}
            )
        if (
            bound_source is not None
            and copied
            and (
                bound_source.get("sha256") != snapshot_file.get("sha256")
                or bound_source.get("size_bytes") != snapshot_file.get("size_bytes")
            )
        ):
            raise InputError("copied snapshot file differs from its v0.3 upstream identity")
        observed = {
            "source_path": snapshot_file.get("source_path"),
            "source_sha256": snapshot_file.get("source_sha256"),
            "source_size_bytes": snapshot_file.get("source_size_bytes"),
        }
        if observed != expected:
            raise InputError(
                "snapshot file source binding differs from canonical v0.3 provenance",
                details={"path": relative},
            )


def _expected_hub_binding(source: Mapping[str, object]) -> dict[str, object]:
    return {
        "source_path": _hub_source_path(
            str(source["repository"]),
            str(source["revision"]),
            str(source["artifact_path"]),
        ),
        "source_sha256": source["sha256"],
        "source_size_bytes": source["size_bytes"],
    }


def _expected_bundle_copy_binding(
    bundle: Mapping[str, object], relative: str, packaged: Mapping[str, object]
) -> dict[str, object]:
    return {
        "source_path": _hub_source_path(
            str(bundle["repository"]),
            str(bundle["revision"]),
            f"{bundle['artifact_path']}/{relative}",
        ),
        "source_sha256": packaged["sha256"],
        "source_size_bytes": packaged["size_bytes"],
    }


def _verify_relocated_bundle_checksums(root: Path) -> None:
    membership_mapping = {
        relative: (
            root / f"membership/{relative}"
            if relative.startswith("store/")
            else root / f"evidence/membership-provenance/{relative}"
        )
        for relative in _MEMBERSHIP_FILES - {"SHA256SUMS"}
    }
    split_mapping = {relative: root / relative for relative in _SPLIT_FILES - {"SHA256SUMS"}}
    _verify_relocated_checksums(
        root / "evidence/membership-provenance/SHA256SUMS",
        membership_mapping,
        label="membership provenance",
    )
    _verify_relocated_checksums(
        root / "evidence/split-provenance/SHA256SUMS",
        split_mapping,
        label="membership-split provenance",
    )


def _verify_relocated_checksums(
    checksum_path: Path,
    targets: Mapping[str, Path],
    *,
    label: str,
) -> None:
    checksum = _regular_file(checksum_path, f"{label} checksum record")
    observed: list[str] = []
    for line_number, line in enumerate(checksum.read_text(encoding="utf-8").splitlines(), start=1):
        if len(line) < 67 or line[64:66] != "  ":
            raise InputError(f"{label} checksum entry is malformed", details={"line": line_number})
        digest, relative = line[:64], line[66:]
        if not re.fullmatch(r"[0-9a-f]{64}", digest):
            raise InputError(f"{label} checksum digest is invalid", details={"line": line_number})
        _safe_relative_text(relative, f"{label} checksum path")
        observed.append(relative)
        target = targets.get(relative)
        if target is None or sha256_file(_regular_file(target, label)) != f"sha256:{digest}":
            raise InputError(f"{label} relocated checksum target differs")
    if observed != sorted(targets):
        raise InputError(f"{label} checksum record does not close its original inventory")


def _verify_train_membership_outputs(
    root: Path,
    *,
    snapshot_id: str,
    spec: Mapping[str, object],
    extension: Mapping[str, object],
    snapshot_files: Sequence[Mapping[str, object]],
) -> None:
    descriptors = _transformation_descriptors(
        root,
        spec=spec,
        transformations=extension.get("transformations"),
        snapshot_files=snapshot_files,
    )
    split_report = _json_object(
        root / "evidence/membership-split-evidence.json", "membership split report"
    )
    with MembershipStore.open(root / "membership/store", verify=True) as store:
        _validate_store_binding(store.manifest, split_report, snapshot_id=snapshot_id)
        selections = _train_source_row_ids(store, descriptors)
    for descriptor in descriptors:
        source_id = str(descriptor["source_id"])
        relative = str(descriptor["output_path"])
        observed = _parquet_source_row_ids(
            root / relative,
            kind=str(descriptor["kind"]),  # type: ignore[arg-type]
        )
        expected = selections.get(source_id, frozenset())
        if observed != expected:
            raise InputError(
                "packaged train Parquet rows do not exactly match bound store memberships",
                details={
                    "source_id": source_id,
                    "missing_count": len(expected - observed),
                    "unexpected_count": len(observed - expected),
                },
            )


def _strict_replay_train_outputs(
    root: Path,
    *,
    snapshot: Mapping[str, object],
    gnomad_root: Path,
    clinvar_root: Path,
    snapshot_id: str,
) -> None:
    provenance_root = root / "evidence/membership-provenance"
    sources = _validate_source_contract(
        _json_object(
            provenance_root / "evidence/source-download-identities.json",
            "packaged membership source identities",
        ),
        membership_spec=_json_object(
            provenance_root / "contract/membership-build.json",
            "packaged membership build spec",
        ),
        membership_job=_json_object(
            provenance_root / "evidence/job-summary.json",
            "packaged membership job summary",
        ),
        snapshot_id=snapshot_id,
        gnomad_root=gnomad_root,
        clinvar_root=clinvar_root,
    )
    spec = _json_object(root / SNAPSHOT_SPEC_NAME, "v0.3 dataset snapshot spec")
    upstreams = spec.get("upstreams")
    if not isinstance(upstreams, list):
        raise InputError("v0.3 snapshot spec upstreams are invalid")
    observed_prepared = [
        item
        for item in upstreams
        if isinstance(item, dict) and item.get("kind") in {"prepared_gnomad", "prepared_clinvar"}
    ]
    expected_prepared = [
        {"kind": f"prepared_{source['kind']}", **_public_source_identity(source)}
        for source in sources
    ]
    if observed_prepared != expected_prepared:
        raise InputError("strict upstream replay source identities differ from the bound spec")
    raw_snapshot_files = snapshot.get("files")
    if not isinstance(raw_snapshot_files, list):
        raise InputError("strict upstream replay requires snapshot report files")
    snapshot_files = [_mapping(item, "strict replay snapshot file") for item in raw_snapshot_files]
    extension = _mapping(snapshot.get("v03"), "strict replay v03 extension")
    descriptors = _transformation_descriptors(
        root,
        spec=spec,
        transformations=extension.get("transformations"),
        snapshot_files=snapshot_files,
    )
    descriptor_by_source = {str(item["source_id"]): item for item in descriptors}
    split_report = _json_object(
        root / "evidence/membership-split-evidence.json", "membership split report"
    )
    with MembershipStore.open(root / "membership/store", verify=True) as store:
        _validate_store_binding(store.manifest, split_report, snapshot_id=snapshot_id)
        selections = _train_source_row_ids(store, sources)
    snapshot_file_index = {str(item["path"]): item for item in snapshot_files}
    with tempfile.TemporaryDirectory(prefix="geno-lewm-v03-strict-replay-") as temporary:
        replay_root = Path(temporary)
        replayed_sources: set[str] = set()
        for source in sources:
            source_id = str(source["source_id"])
            selected = selections.get(source_id, frozenset())
            descriptor = descriptor_by_source.get(source_id)
            if descriptor is None:
                if selected:
                    raise InputError("strict upstream replay omitted a non-empty train source")
                continue
            if not selected:
                raise InputError("strict upstream replay train source has no bound memberships")
            relative = str(descriptor["output_path"])
            replayed = filter_membership_parquet(
                Path(str(source["local_path"])),
                replay_root / relative,
                kind=str(source["kind"]),  # type: ignore[arg-type]
                expected_source_row_ids=set(selected),
            )
            packaged = snapshot_file_index[relative]
            if (
                replayed.sha256 != packaged.get("sha256")
                or replayed.size_bytes != packaged.get("size_bytes")
                or replayed.records != packaged.get("records")
            ):
                raise InputError(
                    "strict upstream replay output differs from the packaged train Parquet",
                    details={"source_id": source_id, "path": relative},
                )
            replayed_sources.add(source_id)
    if replayed_sources != set(descriptor_by_source):
        raise InputError("strict upstream replay did not close all 20+1 train transformations")


def _transformation_descriptors(
    root: Path,
    *,
    spec: Mapping[str, object],
    transformations: object,
    snapshot_files: Sequence[Mapping[str, object]],
) -> list[dict[str, object]]:
    upstreams = spec.get("upstreams")
    if not isinstance(upstreams, list):
        raise InputError("v0.3 snapshot spec upstreams are invalid")
    expected: list[dict[str, object]] = []
    for raw in upstreams:
        upstream = _mapping(raw, "v0.3 snapshot upstream")
        upstream_kind = upstream.get("kind")
        if upstream_kind == "prepared_gnomad":
            chromosome = str(upstream["chromosome"])
            if chromosome not in _TRAIN_CHROMOSOMES:
                continue
            kind = "gnomad"
            source_id = f"gnomad-v4.1-chr{chromosome}"
            output_path = f"gnomad/v4.1/train/chr{chromosome}.variants.parquet"
        elif upstream_kind == "prepared_clinvar":
            chromosome = None
            kind = "clinvar"
            source_id = "clinvar-2026-04-15"
            output_path = "clinvar/2026-04-15/train.variants.parquet"
        else:
            continue
        source = {key: value for key, value in upstream.items() if key != "kind"}
        expected.append(
            {
                "kind": kind,
                "source_id": source_id,
                "chromosome": chromosome,
                "source": source,
                "output_path": output_path,
            }
        )
    if (
        not isinstance(transformations, list)
        or len(transformations) != len(expected)
        or len(expected) != 21
    ):
        raise InputError("dataset snapshot report must bind 20 gnomAD and one ClinVar transform")
    file_index = {str(item["path"]): item for item in snapshot_files}
    descriptors: list[dict[str, object]] = []
    for raw, expected_item in zip(transformations, expected, strict=True):
        item = _mapping(raw, "snapshot transformation")
        if set(item) != {"kind", "source_id", "source", "output", "selection"}:
            raise InputError("snapshot transformation fields are invalid")
        if (
            item.get("kind") != expected_item["kind"]
            or item.get("source_id") != expected_item["source_id"]
            or item.get("source") != expected_item["source"]
            or item.get("selection") != "exact_membership_store_train_source_row_id"
        ):
            raise InputError("snapshot transformation source binding drifted")
        output = _mapping(item.get("output"), "snapshot transformation output")
        if set(output) != {"path", "sha256", "size_bytes", "records"}:
            raise InputError("snapshot transformation output fields are invalid")
        relative = _safe_relative_text(str(output.get("path")), "transformation output path")
        if relative != expected_item["output_path"]:
            raise InputError("snapshot transformation output path differs from its source")
        _match_identity(output, root / relative, relative, include_records=True)
        snapshot_file = file_index.get(relative)
        if snapshot_file is None or any(
            output.get(key) != snapshot_file.get(key)
            for key in ("path", "sha256", "size_bytes", "records")
        ):
            raise InputError("snapshot transformation output differs from snapshot files")
        descriptors.append(
            {
                "kind": expected_item["kind"],
                "source_id": expected_item["source_id"],
                "chromosome": expected_item["chromosome"],
                "output_path": relative,
            }
        )
    return descriptors


def _parquet_source_row_ids(path: Path, *, kind: Literal["gnomad", "clinvar"]) -> frozenset[str]:
    source = _regular_file(path, "packaged train Parquet")
    _, pq = _require_pyarrow()
    try:
        parquet = pq.ParquetFile(source)
    except Exception as exc:
        raise InputError(
            "packaged train Parquet is invalid", details={"path": str(source)}
        ) from exc
    required = {"chrom", "pos", "ref", "alt"}
    if kind == "clinvar":
        required.add("clinvar_id")
    if not required <= set(parquet.schema_arrow.names):
        raise InputError("packaged train Parquet identity columns are incomplete")
    observed: set[str] = set()
    for batch in parquet.iter_batches(batch_size=65_536):
        columns = batch.to_pydict()
        for index in range(batch.num_rows):
            identity = _source_row_identity(columns, index=index, kind=kind)
            if identity in observed:
                raise InputError(
                    "packaged train Parquet contains a duplicate source_row_id",
                    details={"path": str(source), "source_row_id": identity},
                )
            observed.add(identity)
    return frozenset(observed)


def _exact_bundle(path: Path, expected: frozenset[str], label: str) -> Path:
    root = Path(path).absolute()
    if root.is_symlink() or not root.is_dir():
        raise InputError(f"{label} bundle must be a non-symlink directory")
    observed = _inventory(root)
    if observed != expected:
        raise InputError(
            f"{label} bundle exact inventory drifted",
            details={
                "missing": sorted(expected - observed),
                "unexpected": sorted(observed - expected),
            },
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
            details={
                "missing": sorted(expected - observed),
                "unexpected": sorted(observed - expected),
            },
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
    binary_flag = getattr(os, "O_BINARY", 0)
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | binary_flag
    source_fd = os.open(src, flags)
    target_fd: int | None = None
    try:
        metadata = os.fstat(source_fd)
        if not stat.S_ISREG(metadata.st_mode):
            raise InputError("snapshot input changed from a regular file during capture")
        target_fd = os.open(
            target,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL | binary_flag,
            0o600,
        )
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


def _copy_and_verify_identity(source: Path, target: Path, identity: Mapping[str, object]) -> None:
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
            raise InputError(
                "artifact identity mismatch", details={"path": relative, "field": field}
            )
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
        payload = json.loads(
            regular.read_text(encoding="utf-8"),
            object_pairs_hook=_reject_duplicate_json_pairs,
        )
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise InputError(f"{label} is not valid UTF-8 JSON") from exc
    if not isinstance(payload, dict):
        raise InputError(f"{label} must be a JSON object")
    return payload


def _reject_duplicate_json_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    payload: dict[str, Any] = {}
    for key, value in pairs:
        if key in payload:
            raise InputError("duplicate JSON key is not allowed", details={"key": key})
        payload[key] = value
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


def _utc_timestamp(raw: object, label: str) -> str:
    if not isinstance(raw, str) or not _ISO_UTC_RE.fullmatch(raw):
        raise InputError(f"{label} must be a canonical UTC ISO-8601 timestamp ending in Z")
    base, separator, fraction = raw.removesuffix("Z").partition(".")
    timestamp_format = "%Y-%m-%dT%H:%M:%S" + (".%f" if separator else "") + "Z"
    try:
        parsed = datetime.strptime(raw, timestamp_format).replace(tzinfo=timezone.utc)
    except ValueError as exc:
        raise InputError(f"{label} is not a valid calendar timestamp") from exc
    if parsed.utcoffset() != timezone.utc.utcoffset(parsed):
        raise InputError(f"{label} must resolve to UTC")
    if base != parsed.strftime("%Y-%m-%dT%H:%M:%S"):
        raise InputError(f"{label} is not a canonical calendar timestamp")
    if separator and parsed.microsecond != int(fraction) * 10 ** (6 - len(fraction)):
        raise InputError(f"{label} fractional seconds are not canonical")
    return raw


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
        raise RuntimeSetupError(
            "atomic no-replace directory publication is unavailable",
            remediation="run the snapshot assembler on Linux or macOS with no-replace rename support",
        )
    error = ctypes.get_errno()
    if error in {errno.EEXIST, errno.ENOTEMPTY}:
        raise InputError("immutable snapshot output already exists")
    raise OSError(error, os.strerror(error), str(target))


if __name__ == "__main__":  # pragma: no cover - exercised through subprocess tests
    raise SystemExit(main())
