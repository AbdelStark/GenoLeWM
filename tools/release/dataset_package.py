# SPDX-License-Identifier: Apache-2.0
"""Build a release-ready dataset package from shard files and metadata."""

from __future__ import annotations

import argparse
import importlib
import json
import re
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Final, Literal

from geno_lewm.data import MembershipStore
from geno_lewm.errors import GenoLeWMError, InputError, exit_code_for
from geno_lewm.provenance import sha256_file
from geno_lewm.provenance.hashing import looks_like_sha256
from tools.release.dataset_integrity import (
    DEFAULT_REPORT_NAME,
    SplitIntegrityReport,
    write_dataset_integrity_report,
)

SCHEMA_VERSION: Final = "1.0.0"
ARTIFACT_ROLE_SCHEMA_VERSION: Final = "1.1.0"
ArtifactRole = Literal["split_data", "split_companion", "evidence"]
ARTIFACT_ROLES: Final[frozenset[str]] = frozenset({"split_data", "split_companion", "evidence"})
MEMBERSHIP_STORE_FILES: Final = frozenset(
    {
        "manifest.json",
        "memberships.parquet",
        "lookup.sqlite",
        "snapshot-lineage.json",
        "build-receipt.json",
    }
)
MEMBERSHIP_SPLIT_EVIDENCE_SCHEMA_VERSION: Final = "geno-lewm.membership-split-evidence.v1"
MEMBERSHIP_SPLIT_EVIDENCE_SCHEMA_SHA256: Final = (
    "sha256:a4bf6b1a9c60926878e7de0f67116936d0c1490663eb7de94a732df526116810"
)
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
SAFE_RELATIVE_PATH_RE: Final = re.compile(r"[A-Za-z0-9._-]+(?:/[A-Za-z0-9._-]+)*")


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
class MembershipStoreBinding:
    """Package-local membership store plus its verified semantic identities."""

    path: str
    artifact_id: str
    content_identity: str
    physical_identity: str
    rowset_sha256: str

    def to_dict(self) -> dict[str, str]:
        return {
            "path": self.path,
            "artifact_id": self.artifact_id,
            "content_identity": self.content_identity,
            "physical_identity": self.physical_identity,
            "rowset_sha256": self.rowset_sha256,
        }


@dataclass(frozen=True, slots=True)
class SplitEvidenceBinding:
    """Package-local split report and its bundled validation schema."""

    path: str
    schema_path: str
    artifact_id: str
    schema_version: str

    def to_dict(self) -> dict[str, str]:
        return {
            "path": self.path,
            "schema_path": self.schema_path,
            "artifact_id": self.artifact_id,
            "schema_version": self.schema_version,
        }


@dataclass(frozen=True, slots=True)
class MembershipAndSplitEvidence:
    """Closed binding between split files, their report, and membership store."""

    membership_store: MembershipStoreBinding
    report: SplitEvidenceBinding

    def to_dict(self) -> dict[str, object]:
        return {
            "membership_store": self.membership_store.to_dict(),
            "report": self.report.to_dict(),
        }


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
    membership_and_split_evidence: MembershipAndSplitEvidence | None = None

    def manifest(self) -> dict[str, object]:
        payload: dict[str, object] = {
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
        if self.membership_and_split_evidence is not None:
            payload["membership_and_split_evidence"] = self.membership_and_split_evidence.to_dict()
        return payload

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
    files = _parse_files(
        payload.get("files"),
        dataset_dir=dataset_dir,
        splits=frozenset(splits),
        schema_version=schema_version,
    )
    raw_evidence = payload.get("membership_and_split_evidence")
    if schema_version == SCHEMA_VERSION and "membership_and_split_evidence" in payload:
        raise InputError("schema 1.0.0 forbids membership_and_split_evidence")
    membership_and_split_evidence = (
        None
        if raw_evidence is None
        else _parse_membership_and_split_evidence(
            raw_evidence,
            dataset_dir=dataset_dir,
            files=files,
        )
    )
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
        files=files,
        membership_and_split_evidence=membership_and_split_evidence,
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
    lines.extend(["", "## Files", ""])
    if package.schema_version == ARTIFACT_ROLE_SCHEMA_VERSION:
        lines.append(
            "| Path | Artifact role | Companion of | SHA-256 | Bytes | Split | Records | Description |"
        )
        lines.append("| --- | --- | --- | --- | ---: | --- | ---: | --- |")
        lines.extend(
            (
                f"| {_md_cell(file.path)} | {_md_cell(file.artifact_role or '')} "
                f"| {_md_cell(file.companion_of or '')} | {_md_cell(file.sha256)} "
                f"| {file.size_bytes} | {_md_cell(file.split or '')} "
                f"| {'' if file.records is None else file.records} "
                f"| {_md_cell(file.description or '')} |"
            )
            for file in package.files
        )
    else:
        lines.append("| Path | SHA-256 | Bytes | Split | Records | Description |")
        lines.append("| --- | --- | ---: | --- | ---: | --- |")
        lines.extend(
            (
                f"| {_md_cell(file.path)} | {_md_cell(file.sha256)} | {file.size_bytes} "
                f"| {_md_cell(file.split or '')} | {'' if file.records is None else file.records} "
                f"| {_md_cell(file.description or '')} |"
            )
            for file in package.files
        )
    lines.extend(_membership_and_split_evidence_lines(package.membership_and_split_evidence))
    lines.extend(["", "## Intended Use", "", package.intended_use, "", "## Limitations", ""])
    lines.extend(f"- {item}" for item in package.limitations)
    lines.append("")
    return "\n".join(lines)


def _membership_and_split_evidence_lines(
    evidence: MembershipAndSplitEvidence | None,
) -> list[str]:
    if evidence is None:
        return []
    store = evidence.membership_store
    report = evidence.report
    return [
        "",
        "## Membership and Split Evidence",
        "",
        (
            "Deterministic unphased variant membership is enforced by the package-bound "
            "store and audited split report. This is not phased-haplotype membership."
        ),
        "",
        "| Binding | Value |",
        "| --- | --- |",
        f"| Membership store | `{_md_cell(store.path)}` |",
        f"| Membership artifact | `{_md_cell(store.artifact_id)}` |",
        f"| Membership content identity | `{_md_cell(store.content_identity)}` |",
        f"| Membership physical identity | `{_md_cell(store.physical_identity)}` |",
        f"| Membership rowset | `{_md_cell(store.rowset_sha256)}` |",
        f"| Split evidence report | `{_md_cell(report.path)}` |",
        f"| Split evidence artifact | `{_md_cell(report.artifact_id)}` |",
        f"| Split evidence schema | `{_md_cell(report.schema_path)}` |",
        f"| Split evidence schema version | `{_md_cell(report.schema_version)}` |",
    ]


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
        if schema_version == ARTIFACT_ROLE_SCHEMA_VERSION:
            _require_canonical_relative_posix_path(relative, field=f"files[{index}].path")
        if relative in GENERATED_FILES:
            raise InputError(
                "generated dataset package files cannot be listed as data files",
                details={"path": relative},
            )
        if relative in seen:
            raise InputError("files list contains duplicate paths", details={"path": relative})
        seen.add(relative)
        path = _safe_relative(dataset_dir, relative)
        if schema_version == ARTIFACT_ROLE_SCHEMA_VERSION:
            _reject_symlink_traversal(dataset_dir, relative)
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


def _parse_membership_and_split_evidence(
    raw: Any,
    *,
    dataset_dir: Path,
    files: tuple[DatasetArtifact, ...],
) -> MembershipAndSplitEvidence:
    payload = _required_mapping(raw, "membership_and_split_evidence")
    _require_exact_keys(
        payload,
        {"membership_store", "report"},
        "membership_and_split_evidence",
    )
    raw_store = _required_mapping(
        payload.get("membership_store"),
        "membership_and_split_evidence.membership_store",
    )
    _require_exact_keys(
        raw_store,
        {"path", "artifact_id", "content_identity", "physical_identity", "rowset_sha256"},
        "membership_and_split_evidence.membership_store",
    )
    store = MembershipStoreBinding(
        path=_required_binding_path(raw_store, "path", dataset_dir=dataset_dir),
        artifact_id=_required_text(raw_store, "artifact_id"),
        content_identity=_required_sha256(raw_store, "content_identity"),
        physical_identity=_required_sha256(raw_store, "physical_identity"),
        rowset_sha256=_required_sha256(raw_store, "rowset_sha256"),
    )
    raw_report = _required_mapping(
        payload.get("report"),
        "membership_and_split_evidence.report",
    )
    _require_exact_keys(
        raw_report,
        {"path", "schema_path", "artifact_id", "schema_version"},
        "membership_and_split_evidence.report",
    )
    report = SplitEvidenceBinding(
        path=_required_binding_path(raw_report, "path", dataset_dir=dataset_dir),
        schema_path=_required_binding_path(
            raw_report,
            "schema_path",
            dataset_dir=dataset_dir,
        ),
        artifact_id=_required_text(raw_report, "artifact_id"),
        schema_version=_required_text(raw_report, "schema_version"),
    )
    binding = MembershipAndSplitEvidence(membership_store=store, report=report)
    _validate_membership_and_split_evidence(
        binding,
        dataset_dir=dataset_dir,
        files=files,
    )
    return binding


def _validate_membership_and_split_evidence(
    binding: MembershipAndSplitEvidence,
    *,
    dataset_dir: Path,
    files: tuple[DatasetArtifact, ...],
) -> None:
    by_path = {file.path: file for file in files}
    store_paths = {f"{binding.membership_store.path}/{name}" for name in MEMBERSHIP_STORE_FILES}
    _require_evidence_artifacts(store_paths, by_path, label="membership store")
    _require_evidence_artifacts(
        {binding.report.path, binding.report.schema_path},
        by_path,
        label="split evidence",
    )
    schema_artifact = by_path[binding.report.schema_path]
    if schema_artifact.sha256 != MEMBERSHIP_SPLIT_EVIDENCE_SCHEMA_SHA256:
        raise InputError(
            "membership split evidence must use the tracked schema identity",
            details={
                "expected": MEMBERSHIP_SPLIT_EVIDENCE_SCHEMA_SHA256,
                "observed": schema_artifact.sha256,
            },
        )
    if binding.report.schema_version != MEMBERSHIP_SPLIT_EVIDENCE_SCHEMA_VERSION:
        raise InputError(
            "membership split evidence schema version is unsupported",
            details={
                "expected": MEMBERSHIP_SPLIT_EVIDENCE_SCHEMA_VERSION,
                "observed": binding.report.schema_version,
            },
        )

    store_root = _safe_relative(dataset_dir, binding.membership_store.path)
    if not store_root.is_dir():
        raise InputError(
            "membership store binding path is not a directory",
            details={"path": binding.membership_store.path},
        )
    with MembershipStore.open(store_root, verify=True) as opened:
        observed_store = {
            "artifact_id": opened.manifest.artifact_id,
            "content_identity": opened.manifest.content_identity,
            "physical_identity": opened.manifest.physical_identity,
            "rowset_sha256": opened.manifest.rowset_sha256,
        }
    expected_store = {
        "artifact_id": binding.membership_store.artifact_id,
        "content_identity": binding.membership_store.content_identity,
        "physical_identity": binding.membership_store.physical_identity,
        "rowset_sha256": binding.membership_store.rowset_sha256,
    }
    if observed_store != expected_store:
        raise InputError(
            "membership store binding identity mismatch",
            details={"expected": expected_store, "observed": observed_store},
        )

    schema = _load_json_object(
        _safe_relative(dataset_dir, binding.report.schema_path),
        "membership split evidence schema",
    )
    report = _load_json_object(
        _safe_relative(dataset_dir, binding.report.path),
        "membership split evidence report",
    )
    _validate_json_schema(report, schema)
    _require_publication_eligible_split_evidence(report)
    if (
        report.get("artifact_id") != binding.report.artifact_id
        or report.get("schema_version") != binding.report.schema_version
    ):
        raise InputError(
            "split evidence report binding mismatch",
            details={
                "expected_artifact_id": binding.report.artifact_id,
                "observed_artifact_id": report.get("artifact_id"),
                "expected_schema_version": binding.report.schema_version,
                "observed_schema_version": report.get("schema_version"),
            },
        )
    report_store = _required_mapping(
        report.get("membership_store"),
        "membership split evidence membership_store",
    )
    report_store_identities = {
        key: report_store.get(key)
        for key in ("artifact_id", "content_identity", "physical_identity", "rowset_sha256")
    }
    if report_store_identities != expected_store:
        raise InputError(
            "split evidence membership store identity mismatch",
            details={"expected": expected_store, "observed": report_store_identities},
        )
    _validate_split_evidence_streams(report, by_path)
    _validate_training_window_binding(report, by_path)


def _require_publication_eligible_split_evidence(report: dict[str, Any]) -> None:
    producer = _required_mapping(report.get("producer"), "membership split evidence producer")
    store = _required_mapping(
        report.get("membership_store"),
        "membership split evidence membership_store",
    )
    lineage = _required_mapping(
        store.get("lineage"),
        "membership split evidence membership_store.lineage",
    )
    claim = _required_mapping(
        report.get("claim_boundary"),
        "membership split evidence claim_boundary",
    )
    streams = _required_mapping(report.get("streams"), "membership split evidence streams")
    audits = _required_mapping(report.get("audits"), "membership split evidence audits")
    exhaustive = _required_mapping(
        audits.get("exhaustive"),
        "membership split evidence audits.exhaustive",
    )
    deterministic_sample = _required_mapping(
        audits.get("deterministic_sample"),
        "membership split evidence audits.deterministic_sample",
    )
    eligible = (
        report.get("ok") is True
        and producer.get("invocation_verified") is True
        and lineage.get("evidence_profile") == "official"
        and claim.get("publication_eligible") is True
        and claim.get("variant_membership") is True
        and claim.get("phased_haplotype_membership") is False
        and claim.get("released_v03_snapshot") is False
        and set(streams) == {"validation", "evaluation"}
        and exhaustive.get("status") == "passed"
        and exhaustive.get("policy_exclusions") == 0
        and exhaustive.get("indexed_overlaps") == 0
        and deterministic_sample.get("status") == "passed"
        and deterministic_sample.get("policy_exclusions") == 0
        and deterministic_sample.get("indexed_overlaps") == 0
    )
    if not eligible:
        raise InputError(
            "membership split evidence is not publication eligible",
            details={
                "invocation_verified": producer.get("invocation_verified"),
                "evidence_profile": lineage.get("evidence_profile"),
                "publication_eligible": claim.get("publication_eligible"),
                "streams": sorted(str(key) for key in streams),
            },
        )


def _require_evidence_artifacts(
    paths: set[str],
    by_path: dict[str, DatasetArtifact],
    *,
    label: str,
) -> None:
    invalid = sorted(
        path for path in paths if path not in by_path or by_path[path].artifact_role != "evidence"
    )
    if invalid:
        raise InputError(
            f"{label} binding paths must be declared evidence files",
            details={"paths": invalid},
        )


def _validate_split_evidence_streams(
    report: dict[str, Any],
    by_path: dict[str, DatasetArtifact],
) -> None:
    streams = _required_mapping(report.get("streams"), "membership split evidence streams")
    if not streams:
        raise InputError("membership split evidence streams must be non-empty")
    for stream_name, raw_stream in streams.items():
        stream = _required_mapping(raw_stream, f"membership split evidence streams.{stream_name}")
        role = _required_text(stream, "role", prefix=f"streams.{stream_name}.")
        records = _required_positive_int(
            stream,
            "record_count",
            prefix=f"streams.{stream_name}.",
        )
        labels_identity = _required_file_identity(
            stream.get("labels_jsonl"),
            field=f"streams.{stream_name}.labels_jsonl",
        )
        vcf_identity = _required_file_identity(
            stream.get("vcf"),
            field=f"streams.{stream_name}.vcf",
        )
        labels = by_path.get(str(labels_identity["path"]))
        vcf = by_path.get(str(vcf_identity["path"]))
        roles_match = (
            labels is not None
            and labels.artifact_role == "split_data"
            and labels.split == role
            and labels.records == records
            and vcf is not None
            and vcf.artifact_role == "split_companion"
            and vcf.companion_of == labels.path
            and vcf.split == role
            and vcf.records == records
        )
        if not roles_match:
            raise InputError(
                "split evidence stream artifact roles do not match the package",
                details={"stream": stream_name},
            )
        assert labels is not None
        assert vcf is not None
        _match_reported_file_identity(labels_identity, labels, field="labels_jsonl")
        _match_reported_file_identity(vcf_identity, vcf, field="vcf")


def _validate_training_window_binding(
    report: dict[str, Any],
    by_path: dict[str, DatasetArtifact],
) -> None:
    training = _required_mapping(
        report.get("training_windows"),
        "membership split evidence training_windows",
    )
    source = _required_mapping(
        training.get("source"),
        "membership split evidence training_windows.source",
    )
    source_path = _required_text(source, "artifact_path", prefix="training_windows.source.")
    records = _required_positive_int(training, "record_count", prefix="training_windows.")
    split = _required_text(training, "split", prefix="training_windows.")
    sha256 = _required_sha256(training, "sha256", prefix="training_windows.")
    size_bytes = _required_non_negative_int(
        training,
        "size_bytes",
        prefix="training_windows.",
    )
    candidates = [
        artifact
        for artifact in by_path.values()
        if artifact.artifact_role == "split_data"
        and artifact.split == split
        and artifact.records == records
        and artifact.sha256 == sha256
        and artifact.size_bytes == size_bytes
    ]
    if len(candidates) != 1:
        raise InputError(
            "split evidence training window must bind exactly one split_data file by identity",
            details={
                "source_artifact_path": source_path,
                "split": split,
                "records": records,
                "sha256": sha256,
                "size_bytes": size_bytes,
                "matches": sorted(artifact.path for artifact in candidates),
            },
        )


def _match_reported_file_identity(
    reported: dict[str, object],
    artifact: DatasetArtifact,
    *,
    field: str,
) -> None:
    expected = {
        "path": artifact.path,
        "sha256": artifact.sha256,
        "size_bytes": artifact.size_bytes,
    }
    if reported != expected:
        raise InputError(
            "split evidence file identity mismatch",
            details={"field": field, "expected": expected, "observed": reported},
        )


def _required_file_identity(raw: Any, *, field: str) -> dict[str, object]:
    payload = _required_mapping(raw, field)
    _require_exact_keys(payload, {"path", "sha256", "size_bytes"}, field)
    path = _required_text(payload, "path", prefix=f"{field}.")
    sha256 = _required_sha256(payload, "sha256", prefix=f"{field}.")
    size_bytes = _required_non_negative_int(payload, "size_bytes", prefix=f"{field}.")
    return {"path": path, "sha256": sha256, "size_bytes": size_bytes}


def _load_json_object(path: Path, field: str) -> dict[str, Any]:
    try:
        payload = json.loads(
            path.read_text(encoding="utf-8"),
            object_pairs_hook=_reject_duplicate_json_keys,
        )
    except OSError as exc:
        raise InputError(f"failed to read {field}", details={"path": str(path)}) from exc
    except json.JSONDecodeError as exc:
        raise InputError(
            f"{field} JSON is invalid",
            details={"path": str(path), "line": exc.lineno, "column": exc.colno},
        ) from exc
    if not isinstance(payload, dict):
        raise InputError(f"{field} must be a JSON object")
    return payload


def _validate_json_schema(instance: dict[str, Any], schema: dict[str, Any]) -> None:
    try:
        jsonschema = importlib.import_module("jsonschema")
    except ImportError as exc:
        raise InputError(
            "membership split evidence validation requires jsonschema",
            remediation="install geno-lewm[evidence] or geno-lewm[dev]",
        ) from exc
    validator_type = jsonschema.Draft202012Validator
    try:
        validator_type.check_schema(schema)
    except Exception as exc:
        raise InputError("membership split evidence schema is invalid") from exc
    errors = sorted(
        validator_type(schema).iter_errors(instance),
        key=lambda error: tuple(str(item) for item in error.absolute_path),
    )
    if errors:
        raise InputError(
            "membership split evidence report does not satisfy its bound schema",
            details={"error": errors[0].message},
        )


def _reject_duplicate_json_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    payload: dict[str, Any] = {}
    for key, value in pairs:
        if key in payload:
            raise InputError("membership split evidence JSON contains duplicate keys")
        payload[key] = value
    return payload


def _required_mapping(raw: Any, field: str) -> dict[str, Any]:
    if not isinstance(raw, dict):
        raise InputError(f"{field} must be an object")
    return raw


def _require_exact_keys(payload: dict[str, Any], expected: set[str], field: str) -> None:
    observed = set(payload)
    if observed != expected:
        raise InputError(
            f"{field} fields are invalid",
            details={
                "missing": sorted(expected - observed),
                "unexpected": sorted(observed - expected),
            },
        )


def _required_binding_path(
    payload: dict[str, Any],
    key: str,
    *,
    dataset_dir: Path,
) -> str:
    value = _required_text(payload, key)
    _require_canonical_relative_posix_path(value, field=key)
    _safe_relative(dataset_dir, value)
    return value


def _require_canonical_relative_posix_path(value: str, *, field: str) -> None:
    if SAFE_RELATIVE_PATH_RE.fullmatch(value) is None or any(
        part in {".", ".."} for part in value.split("/")
    ):
        raise InputError(
            "dataset artifact paths must be canonical relative POSIX paths",
            details={"field": field, "path": value},
        )


def _reject_symlink_traversal(root: Path, relative: str) -> None:
    candidate = root
    for part in relative.split("/"):
        candidate /= part
        if candidate.is_symlink():
            raise InputError(
                "schema 1.1.0 dataset artifacts must not traverse symbolic links",
                details={"path": relative},
            )


def _required_sha256(payload: dict[str, Any], key: str, *, prefix: str = "") -> str:
    value = _required_text(payload, key, prefix=prefix)
    if not looks_like_sha256(value):
        raise InputError(f"{prefix}{key} must be a canonical SHA-256 identity")
    return value


def _required_positive_int(
    payload: dict[str, Any],
    key: str,
    *,
    prefix: str = "",
) -> int:
    value = payload.get(key)
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise InputError(f"{prefix}{key} must be a positive integer")
    return value


def _required_non_negative_int(
    payload: dict[str, Any],
    key: str,
    *,
    prefix: str = "",
) -> int:
    value = payload.get(key)
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise InputError(f"{prefix}{key} must be a non-negative integer")
    return value


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
        fields[f"files[{index}].artifact_role"] = file.artifact_role or ""
        fields[f"files[{index}].companion_of"] = file.companion_of or ""
        fields[f"files[{index}].split"] = file.split or ""
        fields[f"files[{index}].records"] = "" if file.records is None else str(file.records)
        fields[f"files[{index}].description"] = file.description or ""
    evidence = package.membership_and_split_evidence
    if evidence is not None:
        store = evidence.membership_store
        report = evidence.report
        fields.update(
            {
                "membership_and_split_evidence.membership_store.path": store.path,
                "membership_and_split_evidence.membership_store.artifact_id": store.artifact_id,
                "membership_and_split_evidence.report.path": report.path,
                "membership_and_split_evidence.report.schema_path": report.schema_path,
                "membership_and_split_evidence.report.artifact_id": report.artifact_id,
                "membership_and_split_evidence.report.schema_version": report.schema_version,
            }
        )
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
