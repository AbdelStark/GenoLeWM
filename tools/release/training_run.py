# SPDX-License-Identifier: Apache-2.0
"""Package and verify evidence for a reproducible training run."""

from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path, PureWindowsPath
from typing import Any, Final

from geno_lewm.errors import GenoLeWMError, InputError, exit_code_for
from geno_lewm.provenance import sha256_file
from geno_lewm.provenance.hashing import looks_like_sha256
from geno_lewm.training.preflight import (
    GENERATED_BY as TRAINING_PREFLIGHT_GENERATED_BY,
    REPORT_NAME as TRAINING_PREFLIGHT_REPORT_NAME,
    SCHEMA_VERSION as TRAINING_PREFLIGHT_SCHEMA_VERSION,
)

SCHEMA_VERSION: Final = "1.0.0"
GENERATED_BY: Final = "tools.release.training_run"
MANIFEST_NAME: Final = "training_run_manifest.json"
CARD_NAME: Final = "training_run_card.md"
CHECKSUMS_NAME: Final = "training_run_SHA256SUMS"
COMMIT_RE: Final = re.compile(r"^[0-9a-fA-F]{7,40}$")
PLACEHOLDER_RE: Final = re.compile(
    r"\b(?:tbd|todo|placeholder|coming soon|fake|dummy|lorem ipsum)\b",
    re.IGNORECASE,
)
GENERATED_FILES: Final = frozenset({MANIFEST_NAME, CARD_NAME, CHECKSUMS_NAME})
ACCEPTED_STATUSES: Final = frozenset({"completed", "completed_negative"})
TRAINING_PREFLIGHT_KIND: Final = "training_preflight"
REQUIRED_PREFLIGHT_DATASET_CORE_FILES: Final = (
    "dataset_package.json",
    "dataset_manifest.json",
    "data_card.md",
    "split_integrity.json",
    "dataset_input_check_report.json",
    "dataset_snapshot_report.json",
    "SHA256SUMS",
)


@dataclass(frozen=True, slots=True)
class TrainingArtifact:
    """One file attached to a training run archive."""

    path: str
    kind: str
    sha256: str
    size_bytes: int
    description: str

    def to_dict(self) -> dict[str, object]:
        return {
            "path": self.path,
            "kind": self.kind,
            "sha256": self.sha256,
            "size_bytes": self.size_bytes,
            "description": self.description,
        }


@dataclass(frozen=True, slots=True)
class TrainingRunManifest:
    """Machine-readable training-run evidence."""

    schema_version: str
    run_id: str
    generated_by: str
    generated_at: str
    command: str
    commit_sha: str
    package_version: str
    dataset_snapshot_id: str
    status: str
    hardware: tuple[str, ...]
    runtime: tuple[str, ...]
    seeds: dict[str, int]
    determinism: str
    monitoring: dict[str, bool]
    result_summary: str
    limitations: tuple[str, ...]
    artifacts: tuple[TrainingArtifact, ...]

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "run_id": self.run_id,
            "generated_by": self.generated_by,
            "generated_at": self.generated_at,
            "command": self.command,
            "commit_sha": self.commit_sha,
            "package_version": self.package_version,
            "dataset_snapshot_id": self.dataset_snapshot_id,
            "status": self.status,
            "hardware": list(self.hardware),
            "runtime": list(self.runtime),
            "seeds": self.seeds,
            "determinism": self.determinism,
            "monitoring": self.monitoring,
            "result_summary": self.result_summary,
            "limitations": list(self.limitations),
            "artifacts": [artifact.to_dict() for artifact in self.artifacts],
        }


@dataclass(frozen=True, slots=True)
class TrainingRunPackageReport:
    """Files written by :func:`build_training_run_package`."""

    run_id: str
    manifest_path: Path
    card_path: Path
    checksums_path: Path
    artifacts: tuple[TrainingArtifact, ...]

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": SCHEMA_VERSION,
            "generated_by": GENERATED_BY,
            "run_id": self.run_id,
            "manifest_path": self.manifest_path.name,
            "card_path": self.card_path.name,
            "checksums_path": self.checksums_path.name,
            "artifacts": [artifact.to_dict() for artifact in self.artifacts],
        }


def build_training_run_package(
    run_dir: Path,
    metadata_path: Path,
    *,
    allow_placeholders: bool = False,
) -> TrainingRunPackageReport:
    """Generate training-run manifest, card, and checksum evidence."""
    payload = _load_metadata(metadata_path)
    manifest = parse_training_run_metadata(
        payload,
        run_dir=run_dir,
        allow_placeholders=allow_placeholders,
    )
    manifest_path = run_dir / MANIFEST_NAME
    card_path = run_dir / CARD_NAME
    checksums_path = run_dir / CHECKSUMS_NAME
    manifest_path.write_text(
        json.dumps(manifest.to_dict(), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    card_path.write_text(render_training_run_card(manifest), encoding="utf-8")
    _write_sha256sums(
        run_dir,
        checksums_path,
        (MANIFEST_NAME, CARD_NAME, *(artifact.path for artifact in manifest.artifacts)),
    )
    return TrainingRunPackageReport(
        run_id=manifest.run_id,
        manifest_path=manifest_path,
        card_path=card_path,
        checksums_path=checksums_path,
        artifacts=manifest.artifacts,
    )


def verify_training_run_manifest(
    run_dir: Path,
    manifest_path: Path | None = None,
    *,
    require_preflight: bool = False,
) -> TrainingRunManifest:
    """Load ``training_run_manifest.json`` and re-check attached artifacts."""
    manifest_path = run_dir / MANIFEST_NAME if manifest_path is None else manifest_path
    try:
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise InputError(
            "failed to read training-run manifest", details={"path": str(manifest_path)}
        ) from exc
    except json.JSONDecodeError as exc:
        raise InputError(
            "training-run manifest JSON is invalid",
            details={"path": str(manifest_path), "line": exc.lineno, "column": exc.colno},
        ) from exc
    if not isinstance(payload, dict):
        raise InputError("training-run manifest must be a JSON object")
    manifest = _manifest_from_payload(payload)
    _verify_training_artifacts(
        run_dir,
        manifest.artifacts,
        require_preflight=require_preflight,
    )
    _verify_metrics_artifact(run_dir, manifest.artifacts)
    _verify_training_preflight_artifact(
        run_dir,
        manifest,
        require_preflight=require_preflight,
    )
    return manifest


def parse_training_run_metadata(
    payload: Any,
    *,
    run_dir: Path,
    allow_placeholders: bool = False,
) -> TrainingRunManifest:
    """Validate decoded training-run metadata and compute artifact hashes."""
    if not isinstance(payload, dict):
        raise InputError("training-run metadata must be a JSON object")
    schema_version = _required_text(payload, "schema_version")
    if schema_version != SCHEMA_VERSION:
        raise InputError(
            "unsupported training-run schema version",
            details={"expected": SCHEMA_VERSION, "observed": schema_version},
        )
    command = _required_text(payload, "command")
    commit_sha = _validate_commit(_required_text(payload, "commit_sha"))
    status = _required_text(payload, "status")
    if status not in ACCEPTED_STATUSES:
        raise InputError(
            "training-run status must be completed or completed_negative",
            details={"status": status},
        )
    monitoring = _parse_monitoring(payload.get("monitoring"))
    seeds = _parse_seeds(payload.get("seeds"))
    artifacts = _collect_artifacts(payload, run_dir=run_dir)
    generated_by = _required_text(payload, "generated_by")
    if generated_by != GENERATED_BY:
        raise InputError(
            "training-run generated_by is invalid",
            details={"expected": GENERATED_BY, "observed": generated_by},
        )
    manifest = TrainingRunManifest(
        schema_version=schema_version,
        run_id=_required_text(payload, "run_id"),
        generated_by=generated_by,
        generated_at=_optional_text(payload, "generated_at") or _utc_now(),
        command=command,
        commit_sha=commit_sha,
        package_version=_required_text(payload, "package_version"),
        dataset_snapshot_id=_required_text(payload, "dataset_snapshot_id"),
        status=status,
        hardware=_parse_text_list(payload.get("hardware"), field="hardware"),
        runtime=_parse_text_list(payload.get("runtime"), field="runtime"),
        seeds=seeds,
        determinism=_required_text(payload, "determinism"),
        monitoring=monitoring,
        result_summary=_required_text(payload, "result_summary"),
        limitations=_parse_text_list(payload.get("limitations"), field="limitations"),
        artifacts=artifacts,
    )
    _verify_metrics_artifact(run_dir, artifacts)
    if not allow_placeholders:
        _reject_placeholders(_text_fields(manifest))
    return manifest


def render_training_run_card(manifest: TrainingRunManifest) -> str:
    """Render a Markdown card for a training run archive."""
    lines = [
        f"# Training Run: {manifest.run_id}",
        "",
        f"Generated by: {manifest.generated_by}",
        f"Generated: {manifest.generated_at}",
        "",
        "## Run Identity",
        "",
        f"- Run id: {manifest.run_id}",
        f"- Status: {manifest.status}",
        f"- Commit SHA: {manifest.commit_sha}",
        f"- Package version: {manifest.package_version}",
        f"- Dataset snapshot: {manifest.dataset_snapshot_id}",
        "",
        "## Command",
        "",
        "```console",
        manifest.command,
        "```",
        "",
        "## Hardware",
        "",
    ]
    lines.extend(f"- {item}" for item in manifest.hardware)
    lines.extend(["", "## Runtime", ""])
    lines.extend(f"- {item}" for item in manifest.runtime)
    lines.extend(["", "## Reproducibility", ""])
    lines.append(f"- Determinism: {manifest.determinism}")
    lines.extend(f"- Seed `{name}`: {value}" for name, value in sorted(manifest.seeds.items()))
    lines.extend(["", "## Monitoring", ""])
    lines.extend(
        f"- {name}: {'enabled' if enabled else 'disabled'}"
        for name, enabled in sorted(manifest.monitoring.items())
    )
    lines.extend(["", "## Artifacts", "", "| Kind | Path | SHA-256 | Bytes | Description |"])
    lines.append("| --- | --- | --- | ---: | --- |")
    lines.extend(
        (
            f"| {_md_cell(artifact.kind)} | {_md_cell(artifact.path)} | "
            f"{_md_cell(artifact.sha256)} | {artifact.size_bytes} | "
            f"{_md_cell(artifact.description)} |"
        )
        for artifact in manifest.artifacts
    )
    lines.extend(["", "## Result Summary", "", manifest.result_summary, "", "## Limitations", ""])
    lines.extend(f"- {item}" for item in manifest.limitations)
    lines.append("")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    try:
        report = build_training_run_package(
            args.run_dir,
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
        description="Build training-run manifest, card, and checksum evidence.",
    )
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--metadata-json", type=Path, required=True)
    parser.add_argument(
        "--allow-placeholders",
        action="store_true",
        help="Allow placeholder wording for local drafts. Do not use for releases.",
    )
    return parser


def _load_metadata(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise InputError(
            "failed to read training-run metadata", details={"path": str(path)}
        ) from exc
    except json.JSONDecodeError as exc:
        raise InputError(
            "training-run metadata JSON is invalid",
            details={"path": str(path), "line": exc.lineno, "column": exc.colno},
        ) from exc
    if not isinstance(payload, dict):
        raise InputError("training-run metadata must be a JSON object")
    return payload


def _collect_artifacts(payload: dict[str, Any], *, run_dir: Path) -> tuple[TrainingArtifact, ...]:
    declared: list[tuple[str, str, str]] = [
        (
            "dataset_manifest",
            _required_text(payload, "dataset_manifest"),
            "dataset manifest used by the run",
        ),
        (
            "training_config",
            _required_text(payload, "training_config"),
            "exact training configuration",
        ),
        ("metrics", _required_text(payload, "metrics"), "machine-readable training metrics"),
    ]
    preflight_report = _optional_text(payload, "training_preflight_report")
    if preflight_report is not None:
        declared.append(
            (
                TRAINING_PREFLIGHT_KIND,
                preflight_report,
                "Carbon training clean-machine preflight report",
            )
        )
    declared.extend(
        ("log", path, "training log")
        for path in _parse_text_list(payload.get("logs"), field="logs")
    )
    declared.extend(
        ("checkpoint", path, "checkpoint artifact")
        for path in _parse_text_list(payload.get("checkpoint_files"), field="checkpoint_files")
    )
    seen: set[str] = set()
    artifacts: list[TrainingArtifact] = []
    for kind, relative, description in declared:
        if relative in GENERATED_FILES:
            raise InputError(
                "generated training-run files cannot be listed as inputs",
                details={"path": relative},
            )
        if relative in seen:
            raise InputError(
                "training-run artifact paths must be unique", details={"path": relative}
            )
        seen.add(relative)
        path = _safe_relative(run_dir, relative)
        if not path.is_file():
            raise InputError("training-run artifact is missing", details={"path": str(path)})
        artifacts.append(
            TrainingArtifact(
                path=relative,
                kind=kind,
                sha256=sha256_file(path),
                size_bytes=path.stat().st_size,
                description=description,
            )
        )
    return tuple(artifacts)


def _manifest_from_payload(payload: dict[str, Any]) -> TrainingRunManifest:
    schema_version = _required_text(payload, "schema_version")
    if schema_version != SCHEMA_VERSION:
        raise InputError(
            "unsupported training-run manifest schema version",
            details={"expected": SCHEMA_VERSION, "observed": schema_version},
        )
    status = _required_text(payload, "status")
    if status not in ACCEPTED_STATUSES:
        raise InputError("training-run status is not releasable", details={"status": status})
    monitoring = _parse_monitoring(payload.get("monitoring"))
    artifacts = _parse_manifest_artifacts(payload.get("artifacts"))
    generated_by = _required_text(payload, "generated_by")
    if generated_by != GENERATED_BY:
        raise InputError(
            "training-run manifest generated_by is invalid",
            details={"expected": GENERATED_BY, "observed": generated_by},
        )
    return TrainingRunManifest(
        schema_version=schema_version,
        run_id=_required_text(payload, "run_id"),
        generated_by=generated_by,
        generated_at=_required_text(payload, "generated_at"),
        command=_required_text(payload, "command"),
        commit_sha=_validate_commit(_required_text(payload, "commit_sha")),
        package_version=_required_text(payload, "package_version"),
        dataset_snapshot_id=_required_text(payload, "dataset_snapshot_id"),
        status=status,
        hardware=_parse_text_list(payload.get("hardware"), field="hardware"),
        runtime=_parse_text_list(payload.get("runtime"), field="runtime"),
        seeds=_parse_seeds(payload.get("seeds")),
        determinism=_required_text(payload, "determinism"),
        monitoring=monitoring,
        result_summary=_required_text(payload, "result_summary"),
        limitations=_parse_text_list(payload.get("limitations"), field="limitations"),
        artifacts=artifacts,
    )


def _parse_manifest_artifacts(raw: Any) -> tuple[TrainingArtifact, ...]:
    if not isinstance(raw, list) or not raw:
        raise InputError("training-run manifest artifacts must be a non-empty list")
    artifacts: list[TrainingArtifact] = []
    for index, item in enumerate(raw):
        if not isinstance(item, dict):
            raise InputError(
                "training-run artifact entries must be objects", details={"index": index}
            )
        path = _required_text(item, "path", prefix=f"artifacts[{index}].")
        kind = _required_text(item, "kind", prefix=f"artifacts[{index}].")
        digest = _required_text(item, "sha256", prefix=f"artifacts[{index}].")
        if not looks_like_sha256(digest):
            raise InputError("training-run artifact sha256 is invalid", details={"path": path})
        size = item.get("size_bytes")
        if isinstance(size, bool) or not isinstance(size, int) or size < 0:
            raise InputError(
                "training-run artifact size_bytes must be non-negative", details={"path": path}
            )
        artifacts.append(
            TrainingArtifact(
                path=path,
                kind=kind,
                sha256=digest,
                size_bytes=size,
                description=_required_text(item, "description", prefix=f"artifacts[{index}]."),
            )
        )
    return tuple(artifacts)


def _verify_training_artifacts(
    run_dir: Path,
    artifacts: tuple[TrainingArtifact, ...],
    *,
    require_preflight: bool,
) -> None:
    required_kinds = {"dataset_manifest", "training_config", "metrics", "checkpoint", "log"}
    if require_preflight:
        required_kinds.add(TRAINING_PREFLIGHT_KIND)
    observed_kinds = {artifact.kind for artifact in artifacts}
    missing = sorted(required_kinds - observed_kinds)
    if missing:
        raise InputError(
            "training-run manifest is missing required artifact kinds", details={"missing": missing}
        )
    for artifact in artifacts:
        path = _safe_relative(run_dir, artifact.path)
        if not path.is_file():
            raise InputError("training-run artifact is missing", details={"path": str(path)})
        observed_hash = sha256_file(path)
        if observed_hash != artifact.sha256:
            raise InputError(
                "training-run artifact hash mismatch",
                details={
                    "path": artifact.path,
                    "expected": artifact.sha256,
                    "observed": observed_hash,
                },
            )
        observed_size = path.stat().st_size
        if observed_size != artifact.size_bytes:
            raise InputError(
                "training-run artifact size mismatch",
                details={
                    "path": artifact.path,
                    "expected": artifact.size_bytes,
                    "observed": observed_size,
                },
            )


def _verify_training_preflight_artifact(
    run_dir: Path,
    manifest: TrainingRunManifest,
    *,
    require_preflight: bool,
) -> None:
    artifacts = tuple(
        artifact for artifact in manifest.artifacts if artifact.kind == TRAINING_PREFLIGHT_KIND
    )
    if not artifacts:
        return
    if len(artifacts) != 1:
        raise InputError("training-run archive must contain exactly one preflight report")
    artifact = artifacts[0]
    if require_preflight and artifact.path != TRAINING_PREFLIGHT_REPORT_NAME:
        raise InputError(
            f"release training-run preflight report must be {TRAINING_PREFLIGHT_REPORT_NAME}",
            details={"path": artifact.path},
        )
    path = _safe_relative(run_dir, artifact.path)
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise InputError(
            "training preflight report JSON is invalid", details={"path": str(path)}
        ) from exc
    if not isinstance(payload, dict):
        raise InputError("training preflight report must be a JSON object")
    _verify_training_preflight_payload(
        run_dir,
        payload,
        manifest,
        require_preflight=require_preflight,
    )


def _verify_training_preflight_payload(
    run_dir: Path,
    payload: dict[str, Any],
    manifest: TrainingRunManifest,
    *,
    require_preflight: bool,
) -> None:
    if payload.get("schema_version") != TRAINING_PREFLIGHT_SCHEMA_VERSION:
        raise InputError(
            "training preflight report schema version is invalid",
            details={
                "expected": TRAINING_PREFLIGHT_SCHEMA_VERSION,
                "observed": payload.get("schema_version"),
            },
        )
    if payload.get("generated_by") != TRAINING_PREFLIGHT_GENERATED_BY:
        raise InputError(
            "training preflight report generator is invalid",
            details={"observed": payload.get("generated_by")},
        )
    if payload.get("ok") is not True:
        raise InputError("training preflight report must have ok=true")
    if payload.get("dataset_snapshot_id") != manifest.dataset_snapshot_id:
        raise InputError(
            "training preflight dataset snapshot does not match training run",
            details={
                "expected": manifest.dataset_snapshot_id,
                "observed": payload.get("dataset_snapshot_id"),
            },
        )
    _verify_no_error_preflight_issues(payload)
    _verify_preflight_training_config(run_dir, payload, manifest.artifacts)
    if require_preflight:
        _verify_preflight_dataset_evidence(payload, manifest)
        _verify_public_safe_preflight_paths(payload)


def _verify_no_error_preflight_issues(payload: dict[str, Any]) -> None:
    issues = payload.get("issues")
    if not isinstance(issues, list):
        raise InputError("training preflight report issues must be a list")
    for index, issue in enumerate(issues):
        if not isinstance(issue, dict):
            raise InputError("training preflight report issues must be objects")
        if issue.get("severity") == "error":
            raise InputError(
                "training preflight report cannot contain error issues",
                details={"index": index, "code": issue.get("code")},
            )


def _verify_preflight_training_config(
    run_dir: Path,
    payload: dict[str, Any],
    artifacts: tuple[TrainingArtifact, ...],
) -> None:
    config_artifacts = tuple(
        artifact for artifact in artifacts if artifact.kind == "training_config"
    )
    if len(config_artifacts) != 1:
        raise InputError("training-run archive must contain exactly one training config artifact")
    training_config = payload.get("training_config")
    if not isinstance(training_config, dict):
        raise InputError("training preflight report training_config must be an object")
    config_artifact = config_artifacts[0]
    config_path = _safe_relative(run_dir, config_artifact.path)
    expected_hash = sha256_file(config_path)
    if training_config.get("sha256") != expected_hash:
        raise InputError(
            "training preflight config hash does not match training run",
            details={
                "path": config_artifact.path,
                "expected": expected_hash,
                "observed": training_config.get("sha256"),
            },
        )
    expected_size = config_path.stat().st_size
    if training_config.get("size_bytes") != expected_size:
        raise InputError(
            "training preflight config size does not match training run",
            details={
                "path": config_artifact.path,
                "expected": expected_size,
                "observed": training_config.get("size_bytes"),
            },
        )
    if not isinstance(training_config.get("resolved"), dict):
        raise InputError("training preflight report must include resolved training config")


def _verify_preflight_dataset_evidence(
    payload: dict[str, Any],
    manifest: TrainingRunManifest,
) -> None:
    dataset = payload.get("dataset")
    if not isinstance(dataset, dict):
        raise InputError("training preflight report dataset must be an object")
    if dataset.get("snapshot_id") != manifest.dataset_snapshot_id:
        raise InputError(
            "training preflight nested dataset snapshot does not match training run",
            details={
                "expected": manifest.dataset_snapshot_id,
                "observed": dataset.get("snapshot_id"),
            },
        )
    core_files = dataset.get("core_files")
    if not isinstance(core_files, dict):
        raise InputError("training preflight dataset core_files must be an object")
    for relative in REQUIRED_PREFLIGHT_DATASET_CORE_FILES:
        raw_identity = core_files.get(relative)
        if not isinstance(raw_identity, dict):
            raise InputError(
                "training preflight dataset core file evidence missing",
                details={"path": relative},
            )
        _verify_preflight_dataset_file_identity(raw_identity, expected_path=relative)
    files = dataset.get("files")
    if not isinstance(files, list) or not files:
        raise InputError("training preflight dataset files must be a non-empty list")


def _verify_preflight_dataset_file_identity(
    raw: dict[str, Any],
    *,
    expected_path: str,
) -> None:
    if raw.get("path") != expected_path:
        raise InputError(
            "training preflight dataset core file path mismatch",
            details={"expected": expected_path, "observed": raw.get("path")},
        )
    digest = raw.get("sha256")
    if not isinstance(digest, str) or not looks_like_sha256(digest):
        raise InputError(
            "training preflight dataset core file sha256 is invalid",
            details={"path": expected_path, "observed": digest},
        )
    size_bytes = raw.get("size_bytes")
    if isinstance(size_bytes, bool) or not isinstance(size_bytes, int) or size_bytes <= 0:
        raise InputError(
            "training preflight dataset core file size is invalid",
            details={"path": expected_path, "observed": size_bytes},
        )


def _verify_public_safe_preflight_paths(payload: dict[str, Any]) -> None:
    for key_path, value in _iter_preflight_path_references(payload):
        if not _is_public_relative_reference(value):
            raise InputError(
                "training preflight report paths must be public-relative",
                details={"field": key_path, "path": value},
            )


def _iter_preflight_path_references(
    value: object, *, prefix: str = ""
) -> tuple[tuple[str, str], ...]:
    references: list[tuple[str, str]] = []
    if isinstance(value, dict):
        for key, child in value.items():
            key_text = str(key)
            child_path = f"{prefix}.{key_text}" if prefix else key_text
            if "path" in key_text.lower() and isinstance(child, str):
                references.append((child_path, child))
            else:
                references.extend(_iter_preflight_path_references(child, prefix=child_path))
    elif isinstance(value, list):
        for index, child in enumerate(value):
            references.extend(_iter_preflight_path_references(child, prefix=f"{prefix}[{index}]"))
    return tuple(references)


def _is_public_relative_reference(value: str) -> bool:
    if not value.strip() or value.startswith("~"):
        return False
    path = Path(value)
    windows_path = PureWindowsPath(value)
    return (
        not path.is_absolute()
        and not windows_path.is_absolute()
        and not windows_path.drive
        and ".." not in path.parts
        and ".." not in windows_path.parts
    )


def _verify_metrics_artifact(run_dir: Path, artifacts: tuple[TrainingArtifact, ...]) -> None:
    metrics_paths = [artifact.path for artifact in artifacts if artifact.kind == "metrics"]
    if len(metrics_paths) != 1:
        raise InputError("training-run archive must contain exactly one metrics artifact")
    path = _safe_relative(run_dir, metrics_paths[0])
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise InputError("training metrics JSON is invalid", details={"path": str(path)}) from exc
    if not isinstance(payload, dict):
        raise InputError("training metrics must be a JSON object", details={"path": str(path)})
    sample_count = payload.get("sample_count")
    if isinstance(sample_count, bool) or not isinstance(sample_count, int) or sample_count <= 0:
        raise InputError("training metrics must record positive sample_count")
    metrics = payload.get("metrics")
    if not isinstance(metrics, dict) or not metrics:
        raise InputError("training metrics must contain a non-empty metrics object")
    if not any(_is_numeric_metric(value) for value in metrics.values()):
        raise InputError("training metrics must contain at least one numeric metric value")


def _is_numeric_metric(value: object) -> bool:
    if isinstance(value, bool):
        return False
    if isinstance(value, int | float):
        return True
    if isinstance(value, dict):
        nested = value.get("value")
        return not isinstance(nested, bool) and isinstance(nested, int | float)
    return False


def _parse_monitoring(raw: Any) -> dict[str, bool]:
    if not isinstance(raw, dict):
        raise InputError("monitoring must be an object")
    required = ("collapse_monitoring", "nan_monitoring")
    monitoring: dict[str, bool] = {}
    for key in required:
        value = raw.get(key)
        if value is not True:
            raise InputError("collapse and NaN monitoring must be enabled", details={"field": key})
        monitoring[key] = True
    return monitoring


def _parse_seeds(raw: Any) -> dict[str, int]:
    if not isinstance(raw, dict) or not raw:
        raise InputError("seeds must be a non-empty object")
    seeds: dict[str, int] = {}
    for key, value in raw.items():
        if not isinstance(key, str) or not key:
            raise InputError("seed names must be non-empty strings")
        if isinstance(value, bool) or not isinstance(value, int):
            raise InputError("seed values must be integers", details={"seed": key})
        seeds[key] = value
    return seeds


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


def _text_fields(manifest: TrainingRunManifest) -> dict[str, str]:
    fields = {
        "run_id": manifest.run_id,
        "generated_by": manifest.generated_by,
        "generated_at": manifest.generated_at,
        "command": manifest.command,
        "commit_sha": manifest.commit_sha,
        "package_version": manifest.package_version,
        "dataset_snapshot_id": manifest.dataset_snapshot_id,
        "status": manifest.status,
        "determinism": manifest.determinism,
        "result_summary": manifest.result_summary,
    }
    for index, item in enumerate(manifest.hardware):
        fields[f"hardware[{index}]"] = item
    for index, item in enumerate(manifest.runtime):
        fields[f"runtime[{index}]"] = item
    for index, item in enumerate(manifest.limitations):
        fields[f"limitations[{index}]"] = item
    for index, artifact in enumerate(manifest.artifacts):
        fields[f"artifacts[{index}].path"] = artifact.path
        fields[f"artifacts[{index}].description"] = artifact.description
    return fields


def _reject_placeholders(values: dict[str, str]) -> None:
    for key, value in values.items():
        if PLACEHOLDER_RE.search(value):
            raise InputError(
                "placeholder text is not allowed in release training-run packages",
                details={"field": key},
            )


def _write_sha256sums(run_dir: Path, path: Path, files: tuple[str, ...]) -> None:
    lines = []
    for relative in files:
        artifact_path = _safe_relative(run_dir, relative)
        digest = sha256_file(artifact_path).removeprefix("sha256:")
        lines.append(f"{digest}  {relative}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _safe_relative(root: Path, relative: str) -> Path:
    candidate = Path(relative)
    if candidate.is_absolute() or ".." in candidate.parts:
        raise InputError(
            "training-run paths must be relative and stay inside run_dir",
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


def _validate_commit(value: str) -> str:
    value = value.strip().lower()
    if not COMMIT_RE.match(value):
        raise InputError("commit_sha must be a 7-40 character hex Git commit")
    return value


def _md_cell(value: str) -> str:
    return value.replace("\n", " ").replace("|", r"\|").strip()


def _utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
