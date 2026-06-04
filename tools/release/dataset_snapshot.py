# SPDX-License-Identifier: Apache-2.0
"""Build the first-experiment dataset snapshot from explicit local inputs."""

from __future__ import annotations

import argparse
import json
import re
import shutil
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Any, Final

from geno_lewm.data import prepare_clinvar_shard, prepare_gnomad_shard
from geno_lewm.errors import GenoLeWMError, InputError, exit_code_for
from geno_lewm.provenance import sha256_file
from tools.release.dataset_integrity import DEFAULT_REPORT_NAME
from tools.release.dataset_package import (
    GENERATED_BY as DATASET_PACKAGE_GENERATED_BY,
    SCHEMA_VERSION,
    DatasetPackageReport,
    build_dataset_package,
)

GENERATED_BY: Final = "tools.release.dataset_snapshot"
SPEC_CHECK_GENERATED_BY: Final = "tools.release.dataset_snapshot.check_spec"
INPUT_CHECK_GENERATED_BY: Final = "tools.release.dataset_snapshot.check_inputs"
REPORT_NAME: Final = "dataset_snapshot_report.json"
INPUT_CHECK_REPORT_NAME: Final = "dataset_input_check_report.json"
PLACEHOLDER_RE: Final = re.compile(
    r"\b(?:tbd|todo|placeholder|coming soon|fake|dummy|lorem ipsum)\b",
    re.IGNORECASE,
)


@dataclass(frozen=True, slots=True)
class SnapshotFile:
    """One file staged into the dataset snapshot before packaging."""

    path: str
    source_path: str | None
    source_sha256: str | None
    source_size_bytes: int | None
    split: str
    records: int
    description: str
    sha256: str
    size_bytes: int
    already_exists: bool = False

    def to_metadata(self) -> dict[str, object]:
        return {
            "path": self.path,
            "split": self.split,
            "records": self.records,
            "description": self.description,
        }

    def to_dict(self) -> dict[str, object]:
        payload = self.to_metadata()
        payload.update(
            {
                "source_path": self.source_path,
                "source_sha256": self.source_sha256,
                "source_size_bytes": self.source_size_bytes,
                "sha256": self.sha256,
                "size_bytes": self.size_bytes,
                "already_exists": self.already_exists,
            }
        )
        return payload


@dataclass(frozen=True, slots=True)
class DatasetSnapshotReport:
    """Summary emitted by the dataset snapshot builder."""

    schema_version: str
    generated_by: str
    generated_at: str
    snapshot_id: str
    dataset_dir: Path
    report_path: Path
    spec_path: Path
    spec_sha256: str
    spec_size_bytes: int
    input_check_path: Path
    metadata_path: Path
    package_report: DatasetPackageReport
    files: tuple[SnapshotFile, ...]

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "generated_by": self.generated_by,
            "generated_at": self.generated_at,
            "snapshot_id": self.snapshot_id,
            "report_path": _display_path(self.dataset_dir, self.report_path),
            "snapshot_spec": {
                "path": _public_source_reference(str(self.spec_path)),
                "sha256": self.spec_sha256,
                "size_bytes": self.spec_size_bytes,
            },
            "input_check_path": _display_path(self.dataset_dir, self.input_check_path),
            "input_check": _package_artifact_identity(self.dataset_dir, self.input_check_path),
            "metadata_path": _display_path(self.dataset_dir, self.metadata_path),
            "package": _package_report_dict(self.package_report, root=self.dataset_dir),
            "files": [file.to_dict() for file in self.files],
        }


@dataclass(frozen=True, slots=True)
class DatasetSnapshotSpecCheck:
    """Validation report for a public first-experiment dataset snapshot spec."""

    schema_version: str
    generated_by: str
    snapshot_id: str
    spec_path: Path
    spec_sha256: str
    spec_size_bytes: int
    staged_paths: tuple[str, ...]
    source_paths: tuple[str, ...]
    splits: tuple[str, ...]
    sources: tuple[str, ...]

    def to_dict(self) -> dict[str, object]:
        return {
            "ok": True,
            "schema_version": self.schema_version,
            "generated_by": self.generated_by,
            "snapshot_id": self.snapshot_id,
            "snapshot_spec": {
                "path": _public_source_reference(str(self.spec_path)),
                "sha256": self.spec_sha256,
                "size_bytes": self.spec_size_bytes,
            },
            "staged_paths": list(self.staged_paths),
            "source_paths": list(self.source_paths),
            "splits": list(self.splits),
            "sources": list(self.sources),
        }


@dataclass(frozen=True, slots=True)
class DatasetSnapshotInputFile:
    """One local upstream input file resolved from a checked snapshot spec."""

    kind: str
    source_path: str
    staged_path: str
    split: str
    description: str
    sha256: str
    size_bytes: int

    def to_dict(self) -> dict[str, object]:
        return {
            "kind": self.kind,
            "source_path": self.source_path,
            "staged_path": self.staged_path,
            "split": self.split,
            "description": self.description,
            "sha256": self.sha256,
            "size_bytes": self.size_bytes,
        }


@dataclass(frozen=True, slots=True)
class DatasetSnapshotInputCheck:
    """Validation report for local upstream inputs named by a snapshot spec."""

    schema_version: str
    generated_by: str
    snapshot_id: str
    spec_path: Path
    spec_sha256: str
    spec_size_bytes: int
    inputs: tuple[DatasetSnapshotInputFile, ...]

    @property
    def total_size_bytes(self) -> int:
        return sum(file.size_bytes for file in self.inputs)

    def to_dict(self) -> dict[str, object]:
        return {
            "ok": True,
            "schema_version": self.schema_version,
            "generated_by": self.generated_by,
            "snapshot_id": self.snapshot_id,
            "snapshot_spec": {
                "path": _public_source_reference(str(self.spec_path)),
                "sha256": self.spec_sha256,
                "size_bytes": self.spec_size_bytes,
            },
            "source_count": len(self.inputs),
            "total_size_bytes": self.total_size_bytes,
            "inputs": [file.to_dict() for file in self.inputs],
        }


def check_dataset_snapshot_spec(spec_path: Path) -> DatasetSnapshotSpecCheck:
    """Validate a checked snapshot spec without requiring local upstream files."""
    spec = _load_spec(spec_path)
    snapshot_id = _required_spec_text(spec, "snapshot_id")
    _optional_spec_text(spec, "generated_at")
    sources = _check_spec_sources(spec.get("sources"))
    _required_spec_text(spec, "license")
    _check_spec_text_list(spec, "preprocessing")
    _required_spec_text(spec, "split_policy")
    _check_spec_text_list(spec, "leakage_checks")
    _required_spec_text(spec, "intended_use")
    _check_spec_text_list(spec, "limitations")

    staged_files = _check_spec_files(spec)
    splits = _check_spec_splits(spec.get("splits"), staged_files)

    return DatasetSnapshotSpecCheck(
        schema_version=SCHEMA_VERSION,
        generated_by=SPEC_CHECK_GENERATED_BY,
        snapshot_id=snapshot_id,
        spec_path=spec_path,
        spec_sha256=sha256_file(spec_path),
        spec_size_bytes=spec_path.stat().st_size,
        staged_paths=tuple(file["path"] for file in staged_files),
        source_paths=tuple(file["source_path"] for file in staged_files),
        splits=tuple(splits),
        sources=sources,
    )


def check_dataset_snapshot_inputs(spec_path: Path) -> DatasetSnapshotInputCheck:
    """Validate that checked-spec upstream inputs are staged and hashable."""
    spec = _load_spec(spec_path)
    spec_check = check_dataset_snapshot_spec(spec_path)
    files = _check_spec_files(spec)
    inputs: list[DatasetSnapshotInputFile] = []
    for file in files:
        source_path = file["source_path"]
        source = _resolve_source(spec_path.parent, source_path)
        inputs.append(
            DatasetSnapshotInputFile(
                kind=file["kind"],
                source_path=source_path,
                staged_path=file["path"],
                split=file["split"],
                description=file["description"],
                sha256=sha256_file(source),
                size_bytes=source.stat().st_size,
            )
        )
    return DatasetSnapshotInputCheck(
        schema_version=SCHEMA_VERSION,
        generated_by=INPUT_CHECK_GENERATED_BY,
        snapshot_id=spec_check.snapshot_id,
        spec_path=spec_path,
        spec_sha256=spec_check.spec_sha256,
        spec_size_bytes=spec_check.spec_size_bytes,
        inputs=tuple(inputs),
    )


def build_dataset_snapshot(
    spec_path: Path,
    dataset_dir: Path,
    *,
    overwrite: bool = False,
    allow_placeholders: bool = False,
) -> DatasetSnapshotReport:
    """Stage local upstream inputs and build the release dataset package."""
    spec = _load_spec(spec_path)
    spec_dir = spec_path.parent
    dataset_dir.mkdir(parents=True, exist_ok=True)
    snapshot_id = _required_text(spec, "snapshot_id")
    generated_at = _optional_text(spec, "generated_at") or _utc_now()
    files: list[SnapshotFile] = []
    input_check = check_dataset_snapshot_inputs(spec_path)
    input_check_path = dataset_dir / INPUT_CHECK_REPORT_NAME
    input_check_path.write_text(
        json.dumps(input_check.to_dict(), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    for item in _required_list(spec, "carbon_files"):
        if not isinstance(item, dict):
            raise InputError("carbon_files entries must be objects")
        files.append(
            _stage_carbon_file(
                item, spec_dir=spec_dir, dataset_dir=dataset_dir, overwrite=overwrite
            )
        )

    gnomad = spec.get("gnomad")
    if not isinstance(gnomad, dict):
        raise InputError("gnomad must be an object")
    files.append(
        _stage_gnomad_file(gnomad, spec_dir=spec_dir, dataset_dir=dataset_dir, overwrite=overwrite)
    )

    clinvar = spec.get("clinvar")
    if not isinstance(clinvar, dict):
        raise InputError("clinvar must be an object")
    files.append(
        _stage_clinvar_file(
            clinvar,
            spec_dir=spec_dir,
            dataset_dir=dataset_dir,
            overwrite=overwrite,
        )
    )

    metadata = _package_metadata(
        spec,
        snapshot_id=snapshot_id,
        generated_at=generated_at,
        files=tuple(files),
    )
    metadata_path = dataset_dir / "dataset_package.json"
    metadata_path.write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    package_report = build_dataset_package(
        dataset_dir,
        metadata_path,
        allow_placeholders=allow_placeholders,
    )
    report_path = dataset_dir / REPORT_NAME
    report = DatasetSnapshotReport(
        schema_version=SCHEMA_VERSION,
        generated_by=GENERATED_BY,
        generated_at=generated_at,
        snapshot_id=snapshot_id,
        dataset_dir=dataset_dir,
        report_path=report_path,
        spec_path=spec_path,
        spec_sha256=sha256_file(spec_path),
        spec_size_bytes=spec_path.stat().st_size,
        input_check_path=input_check_path,
        metadata_path=metadata_path,
        package_report=package_report,
        files=tuple(files),
    )
    report_path.write_text(
        json.dumps(report.to_dict(), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    _write_snapshot_sha256sums(dataset_dir, package_report)
    return report


def main(argv: list[str] | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    try:
        if args.check_spec:
            payload = check_dataset_snapshot_spec(args.spec_json).to_dict()
        elif args.check_inputs:
            payload = check_dataset_snapshot_inputs(args.spec_json).to_dict()
        else:
            if args.dataset_dir is None:
                parser.error(
                    "--dataset-dir is required unless --check-spec or --check-inputs is supplied"
                )
            payload = build_dataset_snapshot(
                args.spec_json,
                args.dataset_dir,
                overwrite=args.overwrite,
                allow_placeholders=args.allow_placeholders,
            ).to_dict()
    except GenoLeWMError as exc:
        sys.stderr.write(f"error: {exc}\n")
        if exc.details:
            sys.stderr.write(f"  details: {json.dumps(exc.details, sort_keys=True)}\n")
        return exit_code_for(exc)
    sys.stdout.write(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    return 0


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build and package the first-experiment dataset snapshot from local inputs.",
    )
    parser.add_argument("--spec-json", type=Path, required=True)
    parser.add_argument("--dataset-dir", type=Path)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--check-spec",
        action="store_true",
        help="Validate the checked snapshot spec without reading local upstream files.",
    )
    mode.add_argument(
        "--check-inputs",
        action="store_true",
        help="Validate and hash staged local upstream files without building the snapshot.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace staged Carbon, gnomAD, and ClinVar shard files before packaging.",
    )
    parser.add_argument(
        "--allow-placeholders",
        action="store_true",
        help="Allow placeholder wording for local drafts. Do not use for releases.",
    )
    return parser


def _stage_carbon_file(
    item: dict[str, Any],
    *,
    spec_dir: Path,
    dataset_dir: Path,
    overwrite: bool,
) -> SnapshotFile:
    source_value = _required_text(item, "source_path", prefix="carbon_files[].")
    source = _resolve_source(spec_dir, source_value)
    target_relative = _required_text(item, "path", prefix="carbon_files[].")
    target = _safe_relative(dataset_dir, target_relative)
    already_exists = _copy_into_snapshot(source, target, overwrite=overwrite)
    records = _optional_non_negative_int(item, "records", prefix="carbon_files[].")
    if records is None:
        records = _count_records(target)
    return SnapshotFile(
        path=target_relative,
        source_path=_public_source_reference(source_value),
        source_sha256=sha256_file(source),
        source_size_bytes=source.stat().st_size,
        split=_required_text(item, "split", prefix="carbon_files[]."),
        records=records,
        description=_required_text(item, "description", prefix="carbon_files[]."),
        sha256=sha256_file(target),
        size_bytes=target.stat().st_size,
        already_exists=already_exists,
    )


def _stage_gnomad_file(
    item: dict[str, Any],
    *,
    spec_dir: Path,
    dataset_dir: Path,
    overwrite: bool,
) -> SnapshotFile:
    release = _required_text(item, "release", prefix="gnomad.")
    source_value = _required_text(item, "input_vcf", prefix="gnomad.")
    source = _resolve_source(spec_dir, source_value)
    report = prepare_gnomad_shard(
        source,
        dataset_dir,
        release=release,
        min_af=_optional_float(item, "min_af", default=0.01, prefix="gnomad."),
        max_allele_len=_optional_positive_int(
            item,
            "max_allele_len",
            default=16,
            prefix="gnomad.",
        ),
        overwrite=overwrite,
    )
    relative = report.output_path.relative_to(dataset_dir).as_posix()
    return SnapshotFile(
        path=relative,
        source_path=_public_source_reference(source_value),
        source_sha256=sha256_file(source),
        source_size_bytes=source.stat().st_size,
        split=_required_text(item, "split", prefix="gnomad."),
        records=report.records_written,
        description=_required_text(item, "description", prefix="gnomad."),
        sha256=sha256_file(report.output_path),
        size_bytes=report.size_bytes,
        already_exists=report.already_exists,
    )


def _stage_clinvar_file(
    item: dict[str, Any],
    *,
    spec_dir: Path,
    dataset_dir: Path,
    overwrite: bool,
) -> SnapshotFile:
    release = _required_text(item, "release", prefix="clinvar.")
    source_value = _required_text(item, "input_vcf", prefix="clinvar.")
    source = _resolve_source(spec_dir, source_value)
    report = prepare_clinvar_shard(
        source,
        dataset_dir,
        release=release,
        max_allele_len=_optional_positive_int(
            item,
            "max_allele_len",
            default=16,
            prefix="clinvar.",
        ),
        overwrite=overwrite,
    )
    relative = report.output_path.relative_to(dataset_dir).as_posix()
    return SnapshotFile(
        path=relative,
        source_path=_public_source_reference(source_value),
        source_sha256=sha256_file(source),
        source_size_bytes=source.stat().st_size,
        split=_required_text(item, "split", prefix="clinvar."),
        records=report.records_written,
        description=_required_text(item, "description", prefix="clinvar."),
        sha256=sha256_file(report.output_path),
        size_bytes=report.size_bytes,
        already_exists=report.already_exists,
    )


def _package_metadata(
    spec: dict[str, Any],
    *,
    snapshot_id: str,
    generated_at: str,
    files: tuple[SnapshotFile, ...],
) -> dict[str, object]:
    return {
        "schema_version": SCHEMA_VERSION,
        "snapshot_id": snapshot_id,
        "generated_by": DATASET_PACKAGE_GENERATED_BY,
        "generated_at": generated_at,
        "sources": _required_list(spec, "sources"),
        "license": _required_text(spec, "license"),
        "preprocessing": _required_list(spec, "preprocessing"),
        "split_policy": _required_text(spec, "split_policy"),
        "splits": _splits_metadata(spec.get("splits"), files),
        "leakage_checks": _required_list(spec, "leakage_checks"),
        "intended_use": _required_text(spec, "intended_use"),
        "limitations": _required_list(spec, "limitations"),
        "files": [file.to_metadata() for file in files],
    }


def _splits_metadata(raw: Any, files: tuple[SnapshotFile, ...]) -> dict[str, dict[str, object]]:
    if not isinstance(raw, dict) or not raw:
        raise InputError("splits must be a non-empty object")
    totals: dict[str, int] = {}
    for file in files:
        totals[file.split] = totals.get(file.split, 0) + file.records
    splits: dict[str, dict[str, object]] = {}
    for name, entry in raw.items():
        if not isinstance(name, str) or not name.strip():
            raise InputError("split names must be non-empty strings")
        if not isinstance(entry, dict):
            raise InputError("split entries must be objects", details={"split": name})
        records = totals.get(name, 0)
        if records <= 0:
            raise InputError(
                "split has no staged records",
                details={"split": name, "records": records},
            )
        split_payload: dict[str, object] = {"records": records}
        description = _optional_text(entry, "description", prefix=f"splits.{name}.")
        if description is not None:
            split_payload["description"] = description
        declared = _optional_non_negative_int(entry, "records", prefix=f"splits.{name}.")
        if declared is not None and declared != records:
            raise InputError(
                "split declared records do not match staged files",
                details={"split": name, "declared": declared, "observed": records},
            )
        splits[name] = split_payload
    undeclared = sorted(set(totals) - set(splits))
    if undeclared:
        raise InputError("staged file split is not declared", details={"splits": undeclared})
    return splits


def _check_spec_sources(raw: Any) -> tuple[str, ...]:
    sources = _required_list({"sources": raw}, "sources")
    names: list[str] = []
    for index, source in enumerate(sources):
        if not isinstance(source, dict):
            raise InputError("sources entries must be objects", details={"index": index})
        name = _required_spec_text(source, "name", prefix=f"sources[{index}].")
        names.append(name)
        _required_spec_text(source, "revision", prefix=f"sources[{index}].")
        url = _required_spec_text(source, "url", prefix=f"sources[{index}].")
        if not url.startswith(("https://", "http://")):
            raise InputError(
                "sources[].url must be an HTTP URL",
                details={"field": f"sources[{index}].url", "url": url},
            )
        _required_spec_text(source, "license", prefix=f"sources[{index}].")
        _optional_spec_text(source, "notes", prefix=f"sources[{index}].")
    if len(set(names)) != len(names):
        raise InputError("sources must not repeat names", details={"sources": names})
    return tuple(names)


def _check_spec_text_list(spec: dict[str, Any], key: str) -> tuple[str, ...]:
    items = _required_list(spec, key)
    checked: list[str] = []
    for index, item in enumerate(items):
        if not isinstance(item, str) or not item.strip():
            raise InputError(f"{key} entries must be non-empty strings", details={"index": index})
        value = item.strip()
        _reject_spec_placeholder(value, field=f"{key}[{index}]")
        checked.append(value)
    return tuple(checked)


def _check_spec_files(spec: dict[str, Any]) -> tuple[dict[str, str], ...]:
    files: list[dict[str, str]] = []
    seen_paths: set[str] = set()
    for index, item in enumerate(_required_list(spec, "carbon_files")):
        if not isinstance(item, dict):
            raise InputError("carbon_files entries must be objects", details={"index": index})
        source_path = _required_spec_path(item, "source_path", prefix=f"carbon_files[{index}].")
        path = _required_spec_path(item, "path", prefix=f"carbon_files[{index}].")
        _optional_non_negative_int(item, "records", prefix=f"carbon_files[{index}].")
        files.append(
            _checked_file_entry(
                path=path,
                kind="carbon",
                source_path=source_path,
                split=_required_spec_text(item, "split", prefix=f"carbon_files[{index}]."),
                description=_required_spec_text(
                    item,
                    "description",
                    prefix=f"carbon_files[{index}].",
                ),
                seen_paths=seen_paths,
            )
        )

    gnomad = spec.get("gnomad")
    if not isinstance(gnomad, dict):
        raise InputError("gnomad must be an object")
    gnomad_release = _required_spec_release(gnomad, "release", prefix="gnomad.")
    gnomad_min_af = _optional_float(gnomad, "min_af", default=0.01, prefix="gnomad.")
    if not 0.0 <= gnomad_min_af <= 1.0:
        raise InputError("gnomad.min_af must be between 0 and 1")
    _optional_positive_int(gnomad, "max_allele_len", default=16, prefix="gnomad.")
    files.append(
        _checked_file_entry(
            path=f"gnomad/{gnomad_release}/variants.parquet",
            kind="gnomad",
            source_path=_required_spec_path(gnomad, "input_vcf", prefix="gnomad."),
            split=_required_spec_text(gnomad, "split", prefix="gnomad."),
            description=_required_spec_text(gnomad, "description", prefix="gnomad."),
            seen_paths=seen_paths,
        )
    )

    clinvar = spec.get("clinvar")
    if not isinstance(clinvar, dict):
        raise InputError("clinvar must be an object")
    clinvar_release = _required_spec_release(clinvar, "release", prefix="clinvar.")
    _optional_positive_int(clinvar, "max_allele_len", default=16, prefix="clinvar.")
    files.append(
        _checked_file_entry(
            path=f"clinvar/{clinvar_release}/variants.parquet",
            kind="clinvar",
            source_path=_required_spec_path(clinvar, "input_vcf", prefix="clinvar."),
            split=_required_spec_text(clinvar, "split", prefix="clinvar."),
            description=_required_spec_text(clinvar, "description", prefix="clinvar."),
            seen_paths=seen_paths,
        )
    )
    return tuple(files)


def _checked_file_entry(
    *,
    path: str,
    kind: str,
    source_path: str,
    split: str,
    description: str,
    seen_paths: set[str],
) -> dict[str, str]:
    if path in seen_paths:
        raise InputError("dataset snapshot spec has duplicate staged paths", details={"path": path})
    seen_paths.add(path)
    return {
        "kind": kind,
        "path": path,
        "source_path": source_path,
        "split": split,
        "description": description,
    }


def _check_spec_splits(raw: Any, files: tuple[dict[str, str], ...]) -> tuple[str, ...]:
    if not isinstance(raw, dict) or not raw:
        raise InputError("splits must be a non-empty object")
    declared: list[str] = []
    for name, entry in raw.items():
        if not isinstance(name, str) or not name.strip():
            raise InputError("split names must be non-empty strings")
        split_name = name.strip()
        _reject_spec_placeholder(split_name, field="splits")
        if split_name in declared:
            raise InputError("split names must be unique", details={"split": split_name})
        if not isinstance(entry, dict):
            raise InputError("split entries must be objects", details={"split": split_name})
        _required_spec_text(entry, "description", prefix=f"splits.{split_name}.")
        _optional_non_negative_int(entry, "records", prefix=f"splits.{split_name}.")
        declared.append(split_name)

    observed = {file["split"] for file in files}
    undeclared = sorted(observed - set(declared))
    unused = sorted(set(declared) - observed)
    if undeclared:
        raise InputError("staged file split is not declared", details={"splits": undeclared})
    if unused:
        raise InputError("split has no staged files", details={"splits": unused})
    return tuple(declared)


def _required_spec_text(payload: dict[str, Any], key: str, *, prefix: str = "") -> str:
    value = _required_text(payload, key, prefix=prefix)
    _reject_spec_placeholder(value, field=f"{prefix}{key}")
    return value


def _optional_spec_text(payload: dict[str, Any], key: str, *, prefix: str = "") -> str | None:
    value = _optional_text(payload, key, prefix=prefix)
    if value is not None:
        _reject_spec_placeholder(value, field=f"{prefix}{key}")
    return value


def _required_spec_release(payload: dict[str, Any], key: str, *, prefix: str = "") -> str:
    value = _required_spec_text(payload, key, prefix=prefix)
    if "/" in value or "\\" in value or value in {".", ".."} or ".." in Path(value).parts:
        raise InputError(
            f"{prefix}{key} must be a public release path segment",
            details={"release": value},
        )
    return value


def _required_spec_path(payload: dict[str, Any], key: str, *, prefix: str = "") -> str:
    value = _required_spec_text(payload, key, prefix=prefix)
    if "\\" in value or value.startswith("~"):
        raise InputError(
            "dataset snapshot spec paths must be relative POSIX paths",
            details={"field": f"{prefix}{key}", "path": value},
        )
    path = Path(value)
    # Detect absolute paths OS-independently: a POSIX-absolute path (/x) is not
    # flagged by PureWindowsPath.is_absolute and a drive path (C:\x) is not
    # flagged by PurePosixPath, so reject if either treats it as absolute.
    if (
        PurePosixPath(value).is_absolute()
        or PureWindowsPath(value).is_absolute()
        or ".." in path.parts
    ):
        raise InputError(
            "dataset snapshot spec paths must be public relative paths",
            details={"field": f"{prefix}{key}", "path": value},
        )
    return path.as_posix()


def _reject_spec_placeholder(value: str, *, field: str) -> None:
    if PLACEHOLDER_RE.search(value):
        raise InputError(
            "placeholder text is not allowed in dataset snapshot specs",
            details={"field": field},
        )


def _package_report_dict(
    report: DatasetPackageReport,
    *,
    root: Path,
) -> dict[str, object]:
    return {
        "schema_version": SCHEMA_VERSION,
        "generated_by": DATASET_PACKAGE_GENERATED_BY,
        "snapshot_id": report.snapshot_id,
        "metadata": _package_artifact_identity(root, root / "dataset_package.json"),
        "manifest_path": _display_path(root, report.manifest_path),
        "manifest": _package_artifact_identity(root, report.manifest_path),
        "data_card_path": _display_path(root, report.data_card_path),
        "data_card": _package_artifact_identity(root, report.data_card_path),
        "integrity_path": _display_path(root, report.integrity_path),
        "integrity": _package_artifact_identity(root, report.integrity_path),
        "checksums_path": _display_path(root, report.checksums_path),
        "files": [file.to_dict() for file in report.files],
    }


def _package_artifact_identity(root: Path, path: Path) -> dict[str, object]:
    return {
        "path": _display_path(root, path),
        "sha256": sha256_file(path),
        "size_bytes": path.stat().st_size,
    }


def _write_snapshot_sha256sums(
    dataset_dir: Path,
    package_report: DatasetPackageReport,
) -> None:
    files = (
        "data_card.md",
        "dataset_package.json",
        "dataset_manifest.json",
        DEFAULT_REPORT_NAME,
        INPUT_CHECK_REPORT_NAME,
        REPORT_NAME,
        *(file.path for file in package_report.files),
    )
    lines = []
    for relative in dict.fromkeys(files):
        artifact_path = _safe_relative(dataset_dir, relative)
        digest = sha256_file(artifact_path).removeprefix("sha256:")
        lines.append(f"{digest}  {relative}")
    (dataset_dir / "SHA256SUMS").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _load_spec(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise InputError(
            "failed to read dataset snapshot spec", details={"path": str(path)}
        ) from exc
    except json.JSONDecodeError as exc:
        raise InputError(
            "dataset snapshot spec JSON is invalid",
            details={"path": str(path), "line": exc.lineno, "column": exc.colno},
        ) from exc
    if not isinstance(payload, dict):
        raise InputError("dataset snapshot spec must be a JSON object")
    schema_version = _required_text(payload, "schema_version")
    if schema_version != SCHEMA_VERSION:
        raise InputError(
            "unsupported dataset snapshot spec schema version",
            details={"expected": SCHEMA_VERSION, "observed": schema_version},
        )
    return payload


def _resolve_source(spec_dir: Path, value: str) -> Path:
    path = Path(value)
    if not path.is_absolute():
        path = spec_dir / path
    if not path.is_file():
        raise InputError("dataset snapshot input file is missing", details={"path": str(path)})
    return path


def _copy_into_snapshot(source: Path, target: Path, *, overwrite: bool) -> bool:
    if target.exists() and source.resolve() == target.resolve():
        return True
    if target.exists() and not overwrite:
        if sha256_file(source) == sha256_file(target):
            return True
        raise InputError(
            "dataset snapshot target already exists with different content",
            details={"path": str(target)},
        )
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, target)
    return False


def _count_records(path: Path) -> int:
    name = path.name.lower()
    if name.endswith(".jsonl"):
        return _count_nonempty_lines(path)
    if name.endswith((".txt", ".tsv", ".csv")):
        return _count_nonempty_noncomment_lines(path)
    if name.endswith(".vcf"):
        return _count_nonempty_noncomment_lines(path)
    raise InputError(
        "carbon file is not record-countable; supply carbon_files[].records",
        details={"path": str(path)},
    )


def _count_nonempty_lines(path: Path) -> int:
    records = 0
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                records += 1
    return records


def _count_nonempty_noncomment_lines(path: Path) -> int:
    records = 0
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            text = line.strip()
            if text and not text.startswith("#"):
                records += 1
    return records


def _safe_relative(root: Path, relative: str) -> Path:
    candidate = Path(relative)
    if candidate.is_absolute() or ".." in candidate.parts:
        raise InputError(
            "dataset snapshot paths must be relative and stay inside dataset_dir",
            details={"path": relative},
        )
    return root / candidate


def _display_path(root: Path, path: Path) -> str:
    try:
        return path.relative_to(root).as_posix()
    except ValueError:
        return path.name


def _public_source_reference(value: str) -> str:
    path = Path(value)
    if path.is_absolute() or ".." in path.parts:
        return path.name
    return path.as_posix()


def _required_list(payload: dict[str, Any], key: str) -> list[Any]:
    value = payload.get(key)
    if not isinstance(value, list) or not value:
        raise InputError(f"{key} must be a non-empty list")
    return value


def _required_text(payload: dict[str, Any], key: str, *, prefix: str = "") -> str:
    value = payload.get(key)
    if not isinstance(value, str) or not value.strip():
        raise InputError(f"{prefix}{key} must be a non-empty string")
    return value.strip()


def _optional_text(payload: dict[str, Any], key: str, *, prefix: str = "") -> str | None:
    value = payload.get(key)
    if value is None:
        return None
    if not isinstance(value, str) or not value.strip():
        raise InputError(f"{prefix}{key} must be a non-empty string when supplied")
    return value.strip()


def _optional_non_negative_int(
    payload: dict[str, Any],
    key: str,
    *,
    prefix: str = "",
) -> int | None:
    value = payload.get(key)
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise InputError(f"{prefix}{key} must be a non-negative integer when supplied")
    return value


def _optional_positive_int(
    payload: dict[str, Any],
    key: str,
    *,
    default: int,
    prefix: str = "",
) -> int:
    value = payload.get(key, default)
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise InputError(f"{prefix}{key} must be a positive integer")
    return value


def _optional_float(
    payload: dict[str, Any],
    key: str,
    *,
    default: float,
    prefix: str = "",
) -> float:
    value = payload.get(key, default)
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise InputError(f"{prefix}{key} must be a number")
    return float(value)


def _utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
