# SPDX-License-Identifier: Apache-2.0
"""Build a release-ready dataset package from shard files and metadata."""

from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Final, Literal

from geno_lewm.errors import GenoLeWMError, InputError, exit_code_for
from geno_lewm.provenance import sha256_file
from tools.release.dataset_integrity import (
    DEFAULT_REPORT_NAME,
    SplitIntegrityReport,
    write_dataset_integrity_report,
)

SCHEMA_VERSION: Final = "1.0.0"
ARTIFACT_ROLE_SCHEMA_VERSION: Final = "1.1.0"
ArtifactRole = Literal["split_data", "split_companion", "evidence"]
ARTIFACT_ROLES: Final[frozenset[str]] = frozenset({"split_data", "split_companion", "evidence"})
GENERATED_BY: Final = "tools.release.dataset_package"
GENERATED_FILES: Final = frozenset(
    {
        "data_card.md",
        "dataset_manifest.json",
        "dataset_package.json",
        DEFAULT_REPORT_NAME,
        "SHA256SUMS",
    }
)
PLACEHOLDER_RE: Final = re.compile(
    r"\b(?:tbd|todo|placeholder|coming soon|fake|dummy|lorem ipsum)\b",
    re.IGNORECASE,
)


@dataclass(frozen=True, slots=True)
class DatasetArtifact:
    """One data file included in a release dataset package."""

    path: str
    sha256: str
    size_bytes: int
    split: str | None = None
    records: int | None = None
    artifact_role: ArtifactRole | None = None
    companion_of: str | None = None
    description: str | None = None

    def to_manifest(self) -> dict[str, object]:
        payload: dict[str, object] = {
            "path": self.path,
            "sha256": self.sha256,
            "size_bytes": self.size_bytes,
        }
        if self.split is not None:
            payload["split"] = self.split
        if self.records is not None:
            payload["records"] = self.records
        if self.artifact_role is not None:
            payload["artifact_role"] = self.artifact_role
        if self.companion_of is not None:
            payload["companion_of"] = self.companion_of
        if self.description is not None:
            payload["description"] = self.description
        return payload

    def to_dict(self) -> dict[str, object]:
        return self.to_manifest()


@dataclass(frozen=True, slots=True)
class DatasetPackage:
    """Validated dataset package metadata plus computed file identities."""

    schema_version: str
    snapshot_id: str
    generated_by: str
    generated_at: str
    sources: tuple[dict[str, str], ...]
    license: str
    preprocessing: tuple[str, ...]
    split_policy: str
    splits: dict[str, dict[str, object]]
    leakage_checks: tuple[str, ...]
    intended_use: str
    limitations: tuple[str, ...]
    files: tuple[DatasetArtifact, ...]

    def manifest(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "snapshot_id": self.snapshot_id,
            "generated_by": self.generated_by,
            "generated_at": self.generated_at,
            "sources": list(self.sources),
            "license": self.license,
            "preprocessing": list(self.preprocessing),
            "split_policy": self.split_policy,
            "splits": self.splits,
            "leakage_checks": list(self.leakage_checks),
            "intended_use": self.intended_use,
            "limitations": list(self.limitations),
            "files": [file.to_manifest() for file in self.files],
        }

    def metadata(self) -> dict[str, object]:
        return self.manifest()


@dataclass(frozen=True, slots=True)
class DatasetPackageReport:
    """Files written by :func:`build_dataset_package`."""

    schema_version: str
    snapshot_id: str
    manifest_path: Path
    data_card_path: Path
    integrity_path: Path
    checksums_path: Path
    files: tuple[DatasetArtifact, ...]

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "generated_by": GENERATED_BY,
            "snapshot_id": self.snapshot_id,
            "manifest_path": self.manifest_path.name,
            "data_card_path": self.data_card_path.name,
            "integrity_path": self.integrity_path.name,
            "checksums_path": self.checksums_path.name,
            "files": [file.to_dict() for file in self.files],
        }


def build_dataset_package(
    dataset_dir: Path,
    metadata_path: Path,
    *,
    allow_placeholders: bool = False,
) -> DatasetPackageReport:
    """Generate ``dataset_manifest.json``, ``data_card.md``, and ``SHA256SUMS``."""
    package = load_dataset_package(
        dataset_dir,
        metadata_path,
        allow_placeholders=allow_placeholders,
    )
    manifest_path = dataset_dir / "dataset_manifest.json"
    metadata_output_path = dataset_dir / "dataset_package.json"
    data_card_path = dataset_dir / "data_card.md"
    integrity_path = dataset_dir / DEFAULT_REPORT_NAME
    checksums_path = dataset_dir / "SHA256SUMS"

    metadata_output_path.write_text(
        json.dumps(package.metadata(), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    manifest_path.write_text(
        json.dumps(package.manifest(), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    integrity_report = write_dataset_integrity_report(
        dataset_dir, manifest_path=manifest_path, output_path=integrity_path
    )
    data_card_path.write_text(
        render_data_card(package, integrity_report=integrity_report),
        encoding="utf-8",
    )
    _write_sha256sums(
        dataset_dir,
        checksums_path,
        (
            "data_card.md",
            "dataset_package.json",
            "dataset_manifest.json",
            DEFAULT_REPORT_NAME,
            *(file.path for file in package.files),
        ),
    )
    return DatasetPackageReport(
        schema_version=package.schema_version,
        snapshot_id=package.snapshot_id,
        manifest_path=manifest_path,
        data_card_path=data_card_path,
        integrity_path=integrity_path,
        checksums_path=checksums_path,
        files=package.files,
    )


def load_dataset_package(
    dataset_dir: Path,
    metadata_path: Path,
    *,
    allow_placeholders: bool = False,
) -> DatasetPackage:
    """Load, validate, and enrich dataset release metadata."""
    try:
        payload = json.loads(metadata_path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise InputError(
            "failed to read dataset metadata", details={"path": str(metadata_path)}
        ) from exc
    except json.JSONDecodeError as exc:
        raise InputError(
            "dataset metadata JSON is invalid",
            details={"path": str(metadata_path), "line": exc.lineno, "column": exc.colno},
        ) from exc
    return parse_dataset_package(
        payload,
        dataset_dir=dataset_dir,
        allow_placeholders=allow_placeholders,
    )


def parse_dataset_package(
    payload: Any,
    *,
    dataset_dir: Path,
    allow_placeholders: bool = False,
) -> DatasetPackage:
    """Validate decoded dataset metadata and compute artifact hashes."""
    if not isinstance(payload, dict):
        raise InputError("dataset metadata must be a JSON object")
    schema_version = _required_text(payload, "schema_version")
    if schema_version not in {SCHEMA_VERSION, ARTIFACT_ROLE_SCHEMA_VERSION}:
        raise InputError(
            "unsupported dataset-package schema version",
            details={
                "expected": [SCHEMA_VERSION, ARTIFACT_ROLE_SCHEMA_VERSION],
                "observed": schema_version,
            },
        )

    generated_by = _required_text(payload, "generated_by")
    if generated_by != GENERATED_BY:
        raise InputError(
            "dataset-package generated_by is invalid",
            details={"expected": GENERATED_BY, "observed": generated_by},
        )
    generated_at = _optional_text(payload, "generated_at") or _utc_now()
    sources = _parse_sources(payload.get("sources"))
    splits = _parse_splits(payload.get("splits"))
    package = DatasetPackage(
        schema_version=schema_version,
        snapshot_id=_required_text(payload, "snapshot_id"),
        generated_by=generated_by,
        generated_at=generated_at,
        sources=sources,
        license=_required_text(payload, "license"),
        preprocessing=_parse_text_list(payload.get("preprocessing"), field="preprocessing"),
        split_policy=_required_text(payload, "split_policy"),
        splits=splits,
        leakage_checks=_parse_text_list(payload.get("leakage_checks"), field="leakage_checks"),
        intended_use=_required_text(payload, "intended_use"),
        limitations=_parse_text_list(payload.get("limitations"), field="limitations"),
        files=_parse_files(
            payload.get("files"),
            dataset_dir=dataset_dir,
            splits=frozenset(splits),
            schema_version=schema_version,
        ),
    )
    if not allow_placeholders:
        _reject_placeholders(_text_fields(package))
    return package


def render_data_card(
    package: DatasetPackage,
    *,
    integrity_report: SplitIntegrityReport | dict[str, Any] | None = None,
) -> str:
    """Render a Markdown data card for ``package``."""
    lines = [
        f"# Data Card: {package.snapshot_id}",
        "",
        f"Generated by: {package.generated_by}",
        f"Generated: {package.generated_at}",
        "",
        "## Sources",
        "",
        "| Name | Revision | License | URL | Notes |",
        "| --- | --- | --- | --- | --- |",
    ]
    lines.extend(
        (
            "| "
            + " | ".join(
                _md_cell(source.get(key, ""))
                for key in ("name", "revision", "license", "url", "notes")
            )
            + " |"
        )
        for source in package.sources
    )
    lines.extend(["", "## License", "", package.license, "", "## Preprocessing", ""])
    lines.extend(f"- {item}" for item in package.preprocessing)
    lines.extend(["", "## Splits", "", package.split_policy, ""])
    lines.extend(["| Split | Records | Description |", "| --- | ---: | --- |"])
    for split_name, split in package.splits.items():
        lines.append(
            f"| {_md_cell(split_name)} | {split['records']} | {_md_cell(str(split.get('description', '')))} |"
        )
    lines.extend(_class_balance_lines(package, integrity_report))
    lines.extend(
        [
            "",
            "## Leakage Checks",
            "",
            "The generated `split_integrity.json` records split counts, file hashes, comparable-key leakage checks, and genomic-region holdout checks.",
        ]
    )
    lines.extend(f"- {item}" for item in package.leakage_checks)
    lines.extend(["", "## Files", "", "| Path | SHA-256 | Bytes | Split | Records | Description |"])
    lines.append("| --- | --- | ---: | --- | ---: | --- |")
    lines.extend(
        (
            f"| {_md_cell(file.path)} | {_md_cell(file.sha256)} | {file.size_bytes} "
            f"| {_md_cell(file.split or '')} | {'' if file.records is None else file.records} "
            f"| {_md_cell(file.description or '')} |"
        )
        for file in package.files
    )
    lines.extend(["", "## Intended Use", "", package.intended_use, "", "## Limitations", ""])
    lines.extend(f"- {item}" for item in package.limitations)
    lines.append("")
    return "\n".join(lines)


def _class_balance_lines(
    package: DatasetPackage,
    integrity_report: SplitIntegrityReport | dict[str, Any] | None,
) -> list[str]:
    lines = [
        "",
        "## Class Balance",
        "",
        (
            "Class-balance counts are generated from package files by "
            f"`tools.release.dataset_integrity` and stored in `{DEFAULT_REPORT_NAME}`."
        ),
        "",
        "| Split | Records | Labelled records | Unlabelled records | Labels |",
        "| --- | ---: | ---: | ---: | --- |",
    ]
    split_payloads = _integrity_split_payloads(integrity_report)
    for split_name, split in package.splits.items():
        payload = split_payloads.get(split_name, {})
        records_value = split["records"]
        records_int = records_value if isinstance(records_value, int) else 0
        observed = payload.get("observed_records", records_value)
        labelled = payload.get("labelled_records", 0)
        unlabelled = payload.get("unlabelled_records", records_int)
        labels = payload.get("label_counts", {})
        lines.append(
            f"| {_md_cell(split_name)} | {observed} | {labelled} | {unlabelled} "
            f"| {_md_cell(_format_label_counts(labels))} |"
        )
    return lines


def _integrity_split_payloads(
    integrity_report: SplitIntegrityReport | dict[str, Any] | None,
) -> dict[str, dict[str, object]]:
    if isinstance(integrity_report, SplitIntegrityReport):
        return integrity_report.splits
    if isinstance(integrity_report, dict):
        raw = integrity_report.get("splits")
        if isinstance(raw, dict):
            return {key: value for key, value in raw.items() if isinstance(value, dict)}
    return {}


def _format_label_counts(raw: object) -> str:
    if not isinstance(raw, dict) or not raw:
        return "none observed"
    parts = []
    for label, count in sorted(raw.items()):
        if isinstance(label, str) and isinstance(count, int) and not isinstance(count, bool):
            parts.append(f"{label}={count}")
    return ", ".join(parts) if parts else "none observed"


def main(argv: list[str] | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    try:
        report = build_dataset_package(
            args.dataset_dir,
            args.metadata_json,
            allow_placeholders=args.allow_placeholders,
        )
    except GenoLeWMError as exc:
        sys.stderr.write(f"error: {exc}\n")
        return exit_code_for(exc)
    sys.stdout.write(json.dumps(report.to_dict(), indent=2, sort_keys=True) + "\n")
    return 0


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build a release dataset card, manifest, and SHA256SUMS.",
    )
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--metadata-json", type=Path, required=True)
    parser.add_argument(
        "--allow-placeholders",
        action="store_true",
        help="Allow placeholder wording for local drafts. Do not use for releases.",
    )
    return parser


def _parse_sources(raw: Any) -> tuple[dict[str, str], ...]:
    if not isinstance(raw, list) or not raw:
        raise InputError("sources must be a non-empty list")
    sources: list[dict[str, str]] = []
    for index, item in enumerate(raw):
        if not isinstance(item, dict):
            raise InputError("source entries must be objects", details={"index": index})
        source = {
            "name": _required_text(item, "name", prefix=f"sources[{index}]."),
            "revision": _required_text(item, "revision", prefix=f"sources[{index}]."),
        }
        for key in ("url", "license", "notes"):
            value = _optional_text(item, key, prefix=f"sources[{index}].")
            if value is not None:
                source[key] = value
        sources.append(source)
    return tuple(sources)


def _parse_splits(raw: Any) -> dict[str, dict[str, object]]:
    if not isinstance(raw, dict) or not raw:
        raise InputError("splits must be a non-empty object")
    splits: dict[str, dict[str, object]] = {}
    for split_name, value in raw.items():
        if not isinstance(split_name, str) or not split_name.strip():
            raise InputError("split names must be non-empty strings")
        if not isinstance(value, dict):
            raise InputError("split entries must be objects", details={"split": split_name})
        records = value.get("records")
        if isinstance(records, bool) or not isinstance(records, int) or records <= 0:
            raise InputError(
                "split records must be a positive integer",
                details={"split": split_name, "field": "records"},
            )
        split: dict[str, object] = {"records": records}
        description = _optional_text(value, "description", prefix=f"splits.{split_name}.")
        if description is not None:
            split["description"] = description
        splits[split_name.strip()] = split
    return splits


def _parse_files(
    raw: Any,
    *,
    dataset_dir: Path,
    splits: frozenset[str],
    schema_version: str,
) -> tuple[DatasetArtifact, ...]:
    if not isinstance(raw, list) or not raw:
        raise InputError("files must be a non-empty list")
    files: list[DatasetArtifact] = []
    seen: set[str] = set()
    for index, item in enumerate(raw):
        if not isinstance(item, dict):
            raise InputError("file entries must be objects", details={"index": index})
        if schema_version == SCHEMA_VERSION:
            forbidden = [key for key in ("artifact_role", "companion_of") if key in item]
            if forbidden:
                raise InputError(
                    "schema 1.0.0 forbids artifact_role and companion_of",
                    details={"index": index, "fields": forbidden},
                )
        relative = _required_text(item, "path", prefix=f"files[{index}].")
        if relative in GENERATED_FILES:
            raise InputError(
                "generated dataset package files cannot be listed as data files",
                details={"path": relative},
            )
        if relative in seen:
            raise InputError("files list contains duplicate paths", details={"path": relative})
        seen.add(relative)
        path = _safe_relative(dataset_dir, relative)
        if not path.is_file():
            raise InputError("dataset file is missing", details={"path": str(path)})
        artifact_role_raw = _optional_text(item, "artifact_role", prefix=f"files[{index}].")
        if schema_version == ARTIFACT_ROLE_SCHEMA_VERSION and artifact_role_raw is None:
            raise InputError(
                "file artifact_role must be declared for schema 1.1.0",
                details={"path": relative},
            )
        if artifact_role_raw is not None and artifact_role_raw not in ARTIFACT_ROLES:
            raise InputError(
                "file artifact_role is invalid",
                details={"path": relative, "artifact_role": artifact_role_raw},
            )
        artifact_role: ArtifactRole | None = None
        if artifact_role_raw == "split_data":
            artifact_role = "split_data"
        elif artifact_role_raw == "split_companion":
            artifact_role = "split_companion"
        elif artifact_role_raw == "evidence":
            artifact_role = "evidence"
        if artifact_role == "evidence":
            forbidden = [key for key in ("split", "records", "companion_of") if key in item]
            if forbidden:
                raise InputError(
                    "evidence files forbid split, records, and companion_of",
                    details={"path": relative, "fields": forbidden},
                )
        split = _optional_text(item, "split", prefix=f"files[{index}].")
        if artifact_role != "evidence" and split is None:
            raise InputError("file split must be declared", details={"path": relative})
        if artifact_role == "evidence" and split is not None:
            raise InputError(
                "evidence files forbid split",
                details={"path": relative},
            )
        if split is not None and split not in splits:
            raise InputError(
                "file split must be declared in splits",
                details={"path": relative, "split": split},
            )
        records = _optional_non_negative_int(item, "records", prefix=f"files[{index}].")
        if artifact_role in {"split_data", "split_companion"} and records is None:
            raise InputError(
                f"{artifact_role} files require records",
                details={"path": relative},
            )
        if artifact_role == "evidence" and records is not None:
            raise InputError(
                "evidence files forbid records",
                details={"path": relative},
            )
        if artifact_role != "split_companion" and "companion_of" in item:
            raise InputError(
                f"{artifact_role or 'legacy split_data'} files forbid companion_of",
                details={"path": relative},
            )
        companion_of = _optional_text(item, "companion_of", prefix=f"files[{index}].")
        if artifact_role == "split_companion" and companion_of is None:
            raise InputError(
                "split_companion files require companion_of",
                details={"path": relative},
            )
        description = _optional_text(item, "description", prefix=f"files[{index}].")
        files.append(
            DatasetArtifact(
                path=relative,
                sha256=sha256_file(path),
                size_bytes=path.stat().st_size,
                split=split,
                records=records,
                artifact_role=artifact_role,
                companion_of=companion_of,
                description=description,
            )
        )
    _validate_companion_targets(files)
    return tuple(files)


def _validate_companion_targets(files: list[DatasetArtifact]) -> None:
    by_path = {file.path: file for file in files}
    for file in files:
        if file.artifact_role != "split_companion":
            continue
        target = by_path.get(file.companion_of or "")
        if target is None or target.artifact_role != "split_data":
            raise InputError(
                "split_companion companion_of must refer to exactly one split_data file",
                details={"path": file.path, "companion_of": file.companion_of},
            )
        if (file.split, file.records) != (target.split, target.records):
            raise InputError(
                "split_companion must match companion_of split and records",
                details={
                    "path": file.path,
                    "companion_of": target.path,
                    "split": file.split,
                    "companion_split": target.split,
                    "records": file.records,
                    "companion_records": target.records,
                },
            )


def _write_sha256sums(dataset_dir: Path, path: Path, files: tuple[str, ...]) -> None:
    lines = []
    for relative in files:
        artifact_path = _safe_relative(dataset_dir, relative)
        digest = sha256_file(artifact_path).removeprefix("sha256:")
        lines.append(f"{digest}  {relative}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _safe_relative(root: Path, relative: str) -> Path:
    candidate = Path(relative)
    if candidate.is_absolute() or ".." in candidate.parts:
        raise InputError(
            "dataset paths must be relative and stay inside dataset_dir",
            details={"path": relative},
        )
    return root / candidate


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


def _parse_text_list(raw: Any, *, field: str) -> tuple[str, ...]:
    if not isinstance(raw, list) or not raw:
        raise InputError(f"{field} must be a non-empty list")
    values: list[str] = []
    for index, item in enumerate(raw):
        if not isinstance(item, str) or not item.strip():
            raise InputError(
                f"{field} entries must be non-empty strings",
                details={"field": f"{field}[{index}]"},
            )
        values.append(item.strip())
    return tuple(values)


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


def _text_fields(package: DatasetPackage) -> dict[str, str]:
    fields = {
        "snapshot_id": package.snapshot_id,
        "generated_by": package.generated_by,
        "generated_at": package.generated_at,
        "license": package.license,
        "split_policy": package.split_policy,
        "intended_use": package.intended_use,
    }
    for source_index, source in enumerate(package.sources):
        for key, value in source.items():
            fields[f"sources[{source_index}].{key}"] = value
    for index, item in enumerate(package.preprocessing):
        fields[f"preprocessing[{index}]"] = item
    for split_name, split in package.splits.items():
        fields[f"splits.{split_name}.description"] = str(split.get("description", ""))
    for index, item in enumerate(package.leakage_checks):
        fields[f"leakage_checks[{index}]"] = item
    for index, item in enumerate(package.limitations):
        fields[f"limitations[{index}]"] = item
    for index, file in enumerate(package.files):
        fields[f"files[{index}].path"] = file.path
        fields[f"files[{index}].split"] = file.split or ""
        fields[f"files[{index}].records"] = "" if file.records is None else str(file.records)
        fields[f"files[{index}].description"] = file.description or ""
    return fields


def _reject_placeholders(values: dict[str, str]) -> None:
    for key, value in values.items():
        if PLACEHOLDER_RE.search(value):
            raise InputError(
                "placeholder text is not allowed in release dataset packages",
                details={"field": key},
            )


def _md_cell(value: str) -> str:
    return value.replace("\n", " ").replace("|", r"\|").strip()


def _utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
