# SPDX-License-Identifier: Apache-2.0
"""Clean-machine preflight for Carbon-backed GenoLeWM training."""

from __future__ import annotations

import importlib.metadata
import importlib.util
import json
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Final, Literal

import yaml

from geno_lewm.config import config_to_dict, load_config
from geno_lewm.errors import GenoLeWMError
from geno_lewm.provenance import sha256_file

REPORT_NAME: Final = "training_preflight_report.json"
SCHEMA_VERSION: Final = "1.0.0"
GENERATED_BY: Final = "geno_lewm.training.preflight"
REQUIRED_TRAINING_MODULES: Final = (
    "torch",
    "transformers",
    "safetensors.torch",
    "pyarrow",
    "pyarrow.parquet",
)
_DATASET_CORE_FILES: Final = (
    "dataset_package.json",
    "dataset_manifest.json",
    "data_card.md",
    "split_integrity.json",
    "dataset_input_check_report.json",
    "dataset_snapshot_report.json",
    "SHA256SUMS",
)
_DATASET_PACKAGE_GENERATED_BY: Final = "tools.release.dataset_package"
_DATASET_SNAPSHOT_GENERATED_BY: Final = "tools.release.dataset_snapshot"
_DATASET_INPUT_CHECK_GENERATED_BY: Final = "tools.release.dataset_snapshot.check_inputs"
_EVAL_PREFIXES: Final = ("eval", "test", "holdout", "validation", "val")
Severity = Literal["error", "warning"]

__all__ = [
    "REPORT_NAME",
    "REQUIRED_TRAINING_MODULES",
    "SCHEMA_VERSION",
    "DependencyProbe",
    "TrainingPreflightIssue",
    "TrainingPreflightReport",
    "TrainingPreflightRequest",
    "build_training_preflight_report",
    "write_training_preflight_report",
]


@dataclass(frozen=True, slots=True)
class TrainingPreflightRequest:
    """Inputs needed before launching a Carbon-backed training run."""

    dataset_dir: Path
    carbon_model_dir: Path
    training_config: Path
    run_dir: Path
    allow_fixture_dataset: bool = False
    require_native_runtime: bool = True


@dataclass(frozen=True, slots=True)
class TrainingPreflightIssue:
    """One preflight issue."""

    severity: Severity
    code: str
    path: str
    message: str

    def to_dict(self) -> dict[str, str]:
        return {
            "severity": self.severity,
            "code": self.code,
            "path": self.path,
            "message": self.message,
        }


@dataclass(frozen=True, slots=True)
class DependencyProbe:
    """Importability probe for one training dependency."""

    import_name: str
    package: str
    required: bool
    available: bool
    version: str | None
    reason: str

    def to_dict(self) -> dict[str, object]:
        return {
            "import_name": self.import_name,
            "package": self.package,
            "required": self.required,
            "available": self.available,
            "version": self.version,
            "reason": self.reason,
        }


@dataclass(frozen=True, slots=True)
class TrainingPreflightReport:
    """Machine-readable readiness evidence for the real training path."""

    schema_version: str
    generated_by: str
    generated_at: str
    ok: bool
    dataset_snapshot_id: str | None
    training_config: dict[str, object]
    run_dir: dict[str, object]
    dataset: dict[str, object]
    carbon: dict[str, object]
    dependencies: tuple[DependencyProbe, ...]
    issues: tuple[TrainingPreflightIssue, ...]

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "generated_by": self.generated_by,
            "generated_at": self.generated_at,
            "ok": self.ok,
            "dataset_snapshot_id": self.dataset_snapshot_id,
            "training_config": self.training_config,
            "run_dir": self.run_dir,
            "dataset": self.dataset,
            "carbon": self.carbon,
            "dependencies": [probe.to_dict() for probe in self.dependencies],
            "issues": [issue.to_dict() for issue in self.issues],
        }


DependencyProbeFn = Callable[[str, bool], DependencyProbe]


def build_training_preflight_report(
    request: TrainingPreflightRequest,
    *,
    generated_at: str | None = None,
    dependency_probe: DependencyProbeFn | None = None,
) -> TrainingPreflightReport:
    """Build clean-machine readiness evidence for Carbon-backed training."""
    issues: list[TrainingPreflightIssue] = []
    dependency_probe = dependency_probe or _probe_dependency
    dataset = _inspect_dataset(request.dataset_dir, request.allow_fixture_dataset, issues)
    carbon = _inspect_carbon_model_dir(request.carbon_model_dir, issues)
    training_config = _inspect_training_config(request.training_config, issues)
    run_dir = _inspect_run_dir(request.run_dir)
    dependencies = tuple(
        dependency_probe(name, request.require_native_runtime) for name in REQUIRED_TRAINING_MODULES
    )
    for probe in dependencies:
        if probe.required and not probe.available:
            _issue(
                issues,
                "error",
                "training.dependency_unavailable",
                probe.import_name,
                probe.reason,
            )
    snapshot_id = dataset.get("snapshot_id")
    return TrainingPreflightReport(
        schema_version=SCHEMA_VERSION,
        generated_by=GENERATED_BY,
        generated_at=_utc_now() if generated_at is None else generated_at,
        ok=not any(issue.severity == "error" for issue in issues),
        dataset_snapshot_id=snapshot_id if isinstance(snapshot_id, str) else None,
        training_config=training_config,
        run_dir=run_dir,
        dataset=dataset,
        carbon=carbon,
        dependencies=dependencies,
        issues=tuple(issues),
    )


def write_training_preflight_report(
    request: TrainingPreflightRequest,
    output: Path | None = None,
    *,
    generated_at: str | None = None,
    dependency_probe: DependencyProbeFn | None = None,
) -> TrainingPreflightReport:
    """Write ``training_preflight_report.json`` and return the report."""
    report = build_training_preflight_report(
        request,
        generated_at=generated_at,
        dependency_probe=dependency_probe,
    )
    output = request.run_dir / REPORT_NAME if output is None else output
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(report.to_dict(), indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return report


def _inspect_dataset(
    dataset_dir: Path,
    allow_fixture_dataset: bool,
    issues: list[TrainingPreflightIssue],
) -> dict[str, object]:
    core_files: dict[str, dict[str, object] | None] = {}
    dataset: dict[str, object] = {
        "path": _public_path_reference(dataset_dir),
        "core_files": core_files,
        "files": [],
        "splits": {},
    }
    if not dataset_dir.is_dir():
        _issue(issues, "error", "dataset.dir_missing", dataset_dir, "dataset_dir is missing")
        return dataset
    for name in _DATASET_CORE_FILES:
        path = dataset_dir / name
        core_files[name] = _file_identity(path, public_path=name) if path.is_file() else None
        if not path.is_file():
            _issue(issues, "error", f"dataset.{name}.missing", path, f"{name} is required")
    package = _load_json_object(dataset_dir / "dataset_package.json", issues, "dataset.package")
    manifest = _load_json_object(dataset_dir / "dataset_manifest.json", issues, "dataset.manifest")
    integrity = _load_json_object(dataset_dir / "split_integrity.json", issues, "dataset.integrity")
    input_check = _load_json_object(
        dataset_dir / "dataset_input_check_report.json", issues, "dataset.input_check"
    )
    snapshot_report = _load_json_object(
        dataset_dir / "dataset_snapshot_report.json", issues, "dataset.snapshot_report"
    )
    checksum_entries = _parse_sha256sums(dataset_dir / "SHA256SUMS", issues)
    if manifest is None:
        return dataset
    snapshot_id = _text(manifest.get("snapshot_id"))
    dataset["snapshot_id"] = snapshot_id
    if snapshot_id is not None and not allow_fixture_dataset and _looks_like_fixture(snapshot_id):
        _issue(
            issues,
            "error",
            "dataset.fixture_snapshot",
            dataset_dir / "dataset_manifest.json",
            "fixture/test dataset snapshots cannot back a Carbon training run",
        )
    splits = manifest.get("splits")
    if isinstance(splits, dict):
        dataset["splits"] = splits
        _require_train_and_eval_splits(splits, issues, dataset_dir / "dataset_manifest.json")
    else:
        _issue(issues, "error", "dataset.splits.invalid", dataset_dir, "splits must be an object")
    raw_files = manifest.get("files")
    if isinstance(raw_files, list):
        dataset["files"] = _inspect_dataset_files(dataset_dir, raw_files, checksum_entries, issues)
    else:
        _issue(issues, "error", "dataset.files.invalid", dataset_dir, "files must be a list")
    if package is not None:
        _verify_dataset_package_metadata(dataset_dir, manifest, package, issues)
    if integrity is not None:
        _verify_integrity_report(dataset_dir, manifest, integrity, issues)
    if input_check is not None:
        _verify_input_check_report(
            dataset_dir,
            manifest=manifest,
            input_check=input_check,
            snapshot_report=snapshot_report,
            issues=issues,
        )
    if snapshot_report is not None:
        _verify_snapshot_report(
            dataset_dir,
            manifest=manifest,
            snapshot_report=snapshot_report,
            issues=issues,
        )
    _verify_core_checksums(dataset_dir, checksum_entries, issues)
    return dataset


def _inspect_carbon_model_dir(
    carbon_model_dir: Path,
    issues: list[TrainingPreflightIssue],
) -> dict[str, object]:
    artifacts: dict[str, dict[str, object] | None] = {}
    carbon: dict[str, object] = {
        "path": _public_path_reference(carbon_model_dir),
        "local_files_only": True,
        "artifacts": artifacts,
    }
    if not carbon_model_dir.is_dir():
        _issue(
            issues,
            "error",
            "carbon.dir_missing",
            carbon_model_dir,
            "carbon_model_dir must point at a local Transformers model directory",
        )
        return carbon
    config = carbon_model_dir / "config.json"
    artifacts["config"] = (
        _file_identity(config, public_path="config.json") if config.is_file() else None
    )
    if not config.is_file():
        _issue(issues, "error", "carbon.config_missing", config, "config.json is required")
    tokenizer = _first_existing(
        carbon_model_dir,
        ("tokenizer.json", "tokenizer_config.json", "vocab.txt", "spiece.model"),
    )
    artifacts["tokenizer"] = (
        _file_identity(tokenizer, public_path=tokenizer.name) if tokenizer is not None else None
    )
    if tokenizer is None:
        _issue(
            issues,
            "error",
            "carbon.tokenizer_missing",
            carbon_model_dir,
            "local tokenizer files are required",
        )
    weights = _first_existing(
        carbon_model_dir,
        ("model.safetensors", "model.safetensors.index.json", "pytorch_model.bin"),
    )
    artifacts["weights"] = (
        _file_identity(weights, public_path=weights.name) if weights is not None else None
    )
    if weights is None:
        _issue(
            issues,
            "error",
            "carbon.weights_missing",
            carbon_model_dir,
            "local model weight files are required",
        )
    return carbon


def _inspect_training_config(
    training_config: Path,
    issues: list[TrainingPreflightIssue],
) -> dict[str, object]:
    payload: dict[str, object] = {
        "path": _public_path_reference(training_config),
        "sha256": None,
        "size_bytes": None,
        "top_level_keys": [],
        "resolved": None,
    }
    if not training_config.is_file():
        _issue(
            issues,
            "error",
            "training_config.missing",
            training_config,
            "training config file is required",
        )
        return payload
    payload["sha256"] = sha256_file(training_config)
    payload["size_bytes"] = training_config.stat().st_size
    try:
        decoded = yaml.safe_load(training_config.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        _issue(
            issues,
            "error",
            "training_config.invalid_yaml",
            training_config,
            str(exc),
        )
        return payload
    if not isinstance(decoded, dict):
        _issue(
            issues,
            "error",
            "training_config.invalid",
            training_config,
            "training config must be a YAML mapping",
        )
        return payload
    payload["top_level_keys"] = sorted(str(key) for key in decoded)
    try:
        resolved = load_config(training_config)
    except GenoLeWMError as exc:
        _issue(
            issues,
            "error",
            "training_config.schema_invalid",
            training_config,
            exc.message or str(exc),
        )
        return payload
    payload["resolved"] = config_to_dict(resolved)
    _verify_training_horizon(resolved, training_config, issues)
    for key in ("encoder", "data", "predictor", "action", "training", "optimizer"):
        if key not in decoded:
            _issue(
                issues,
                "warning",
                f"training_config.{key}.missing",
                training_config,
                f"{key} block is not present in the training config",
            )
    return payload


def _verify_training_horizon(
    config: object,
    training_config: Path,
    issues: list[TrainingPreflightIssue],
) -> None:
    max_steps = getattr(getattr(config, "training", None), "max_steps", None)
    if not _is_positive_int(max_steps):
        _issue(
            issues,
            "error",
            "training_config.training.max_steps_invalid",
            training_config,
            "training.max_steps must be a positive integer",
        )
        return
    assert isinstance(max_steps, int)
    collapse_log_every = getattr(
        getattr(config, "training", None), "collapse_log_every_steps", None
    )
    if not _is_positive_int(collapse_log_every):
        _issue(
            issues,
            "error",
            "training_config.training.collapse_log_every_steps_invalid",
            training_config,
            "training.collapse_log_every_steps must be a positive integer",
        )
    optimizer = getattr(config, "optimizer", None)
    warmup_steps = getattr(optimizer, "warmup_steps", None)
    schedule = getattr(optimizer, "schedule", None)
    if schedule == "wsd" and isinstance(warmup_steps, int) and max_steps <= warmup_steps:
        _issue(
            issues,
            "error",
            "training_config.training.max_steps_wsd_warmup",
            training_config,
            "training.max_steps must exceed optimizer.warmup_steps for the WSD schedule",
        )


def _inspect_run_dir(run_dir: Path) -> dict[str, object]:
    run_dir.mkdir(parents=True, exist_ok=True)
    return {
        "path": _public_path_reference(run_dir),
        "exists": run_dir.is_dir(),
        "preflight_report_path": REPORT_NAME,
    }


def _inspect_dataset_files(
    dataset_dir: Path,
    raw_files: list[Any],
    checksum_entries: dict[str, str],
    issues: list[TrainingPreflightIssue],
) -> list[dict[str, object]]:
    files: list[dict[str, object]] = []
    for index, item in enumerate(raw_files):
        if not isinstance(item, dict):
            _issue(
                issues,
                "error",
                "dataset.file.invalid",
                dataset_dir,
                f"files[{index}] must be an object",
            )
            continue
        relative = _text(item.get("path"))
        if relative is None:
            _issue(issues, "error", "dataset.file.path_missing", dataset_dir, "file path required")
            continue
        path = _safe_relative(dataset_dir, relative, issues, code="dataset.file.unsafe_path")
        if path is None:
            continue
        identity = _file_identity(path, public_path=relative) if path.is_file() else None
        if identity is None:
            _issue(issues, "error", "dataset.file.missing", path, "dataset file is missing")
            continue
        expected = _text(item.get("sha256"))
        if expected is not None and identity["sha256"] != expected:
            _issue(
                issues,
                "error",
                "dataset.file.hash_mismatch",
                path,
                "dataset_manifest.json hash does not match file contents",
            )
        checksum = checksum_entries.get(relative)
        if checksum is None:
            _issue(
                issues,
                "error",
                "dataset.file.checksum_missing",
                path,
                "SHA256SUMS does not cover this dataset file",
            )
        elif checksum != identity["sha256"]:
            _issue(
                issues,
                "error",
                "dataset.file.checksum_mismatch",
                path,
                "SHA256SUMS hash does not match file contents",
            )
        files.append(
            {
                "path": relative,
                "split": item.get("split"),
                "records": item.get("records"),
                **identity,
            }
        )
    return files


def _verify_dataset_package_metadata(
    dataset_dir: Path,
    manifest: dict[str, Any],
    package: dict[str, Any],
    issues: list[TrainingPreflightIssue],
) -> None:
    path = dataset_dir / "dataset_package.json"
    if package.get("generated_by") != _DATASET_PACKAGE_GENERATED_BY:
        _issue(
            issues,
            "error",
            "dataset.package.generated_by",
            path,
            f"generated_by must be {_DATASET_PACKAGE_GENERATED_BY}",
        )
    if package.get("snapshot_id") != manifest.get("snapshot_id"):
        _issue(
            issues,
            "error",
            "dataset.package.snapshot_mismatch",
            path,
            "dataset_package snapshot_id must match dataset_manifest",
        )
    if package.get("files") != manifest.get("files"):
        _issue(
            issues,
            "error",
            "dataset.package.files_mismatch",
            path,
            "dataset_package files must match dataset_manifest",
        )


def _verify_input_check_report(
    dataset_dir: Path,
    *,
    manifest: dict[str, Any],
    input_check: dict[str, Any],
    snapshot_report: dict[str, Any] | None,
    issues: list[TrainingPreflightIssue],
) -> None:
    path = dataset_dir / "dataset_input_check_report.json"
    code_prefix = "dataset.input_check"
    if input_check.get("schema_version") != SCHEMA_VERSION:
        _issue(
            issues, "error", f"{code_prefix}.schema_version", path, "schema_version must be 1.0.0"
        )
    if input_check.get("generated_by") != _DATASET_INPUT_CHECK_GENERATED_BY:
        _issue(
            issues,
            "error",
            f"{code_prefix}.generated_by",
            path,
            f"generated_by must be {_DATASET_INPUT_CHECK_GENERATED_BY}",
        )
    if input_check.get("snapshot_id") != manifest.get("snapshot_id"):
        _issue(
            issues,
            "error",
            f"{code_prefix}.snapshot_mismatch",
            path,
            "input check snapshot_id must match dataset_manifest",
        )
    raw_inputs = input_check.get("inputs")
    if not isinstance(raw_inputs, list) or not raw_inputs:
        _issue(
            issues,
            "error",
            f"{code_prefix}.inputs",
            path,
            "inputs must be a non-empty list",
        )
        return
    observed = [
        _normalized_input_check_entry(item) for item in raw_inputs if isinstance(item, dict)
    ]
    if None in observed or len(observed) != len(raw_inputs):
        _issue(
            issues,
            "error",
            f"{code_prefix}.inputs",
            path,
            "input entries must be public-safe source identity objects",
        )
        return
    if snapshot_report is None:
        return
    expected = _expected_input_check_entries(snapshot_report.get("files"))
    if observed != expected:
        _issue(
            issues,
            "error",
            f"{code_prefix}.stale",
            path,
            "input check report does not match dataset_snapshot_report source identities",
        )


def _verify_snapshot_report(
    dataset_dir: Path,
    *,
    manifest: dict[str, Any],
    snapshot_report: dict[str, Any],
    issues: list[TrainingPreflightIssue],
) -> None:
    path = dataset_dir / "dataset_snapshot_report.json"
    if snapshot_report.get("schema_version") != SCHEMA_VERSION:
        _issue(
            issues,
            "error",
            "dataset.snapshot_report.schema_version",
            path,
            "schema_version must be 1.0.0",
        )
    if snapshot_report.get("generated_by") != _DATASET_SNAPSHOT_GENERATED_BY:
        _issue(
            issues,
            "error",
            "dataset.snapshot_report.generated_by",
            path,
            f"generated_by must be {_DATASET_SNAPSHOT_GENERATED_BY}",
        )
    if snapshot_report.get("snapshot_id") != manifest.get("snapshot_id"):
        _issue(
            issues,
            "error",
            "dataset.snapshot_report.snapshot_mismatch",
            path,
            "snapshot report snapshot_id must match dataset_manifest",
        )
    for key, expected in (
        ("report_path", "dataset_snapshot_report.json"),
        ("metadata_path", "dataset_package.json"),
        ("input_check_path", "dataset_input_check_report.json"),
    ):
        if snapshot_report.get(key) != expected:
            _issue(
                issues,
                "error",
                f"dataset.snapshot_report.{key}",
                path,
                f"{key} must be {expected}",
            )
    _verify_snapshot_artifact_identity(
        snapshot_report.get("input_check"),
        expected_relative="dataset_input_check_report.json",
        dataset_dir=dataset_dir,
        code_prefix="dataset.snapshot_report.input_check",
        issues=issues,
    )
    _verify_snapshot_files_match_manifest(
        dataset_dir,
        manifest=manifest,
        snapshot_report=snapshot_report,
        issues=issues,
    )


def _verify_snapshot_artifact_identity(
    raw: object,
    *,
    expected_relative: str,
    dataset_dir: Path,
    code_prefix: str,
    issues: list[TrainingPreflightIssue],
) -> None:
    path = dataset_dir / expected_relative
    if not isinstance(raw, dict):
        _issue(issues, "error", code_prefix, path, f"{expected_relative} identity is required")
        return
    if raw.get("path") != expected_relative:
        _issue(
            issues,
            "error",
            f"{code_prefix}.path",
            path,
            f"path must be {expected_relative}",
        )
    if not path.is_file():
        _issue(issues, "error", f"{code_prefix}.missing", path, f"{expected_relative} is missing")
        return
    identity = _file_identity(path, public_path=expected_relative)
    if raw.get("sha256") != identity["sha256"] or raw.get("size_bytes") != identity["size_bytes"]:
        _issue(
            issues,
            "error",
            f"{code_prefix}.stale",
            path,
            f"{expected_relative} identity does not match current file contents",
        )


def _verify_snapshot_files_match_manifest(
    dataset_dir: Path,
    *,
    manifest: dict[str, Any],
    snapshot_report: dict[str, Any],
    issues: list[TrainingPreflightIssue],
) -> None:
    path = dataset_dir / "dataset_snapshot_report.json"
    raw_snapshot_files = snapshot_report.get("files")
    raw_manifest_files = manifest.get("files")
    if not isinstance(raw_snapshot_files, list) or not raw_snapshot_files:
        _issue(
            issues,
            "error",
            "dataset.snapshot_report.files",
            path,
            "files must be a non-empty list",
        )
        return
    if not isinstance(raw_manifest_files, list):
        return
    snapshot_paths = [
        item.get("path") if isinstance(item, dict) else None for item in raw_snapshot_files
    ]
    manifest_paths = [
        item.get("path") if isinstance(item, dict) else None for item in raw_manifest_files
    ]
    if snapshot_paths != manifest_paths:
        _issue(
            issues,
            "error",
            "dataset.snapshot_report.files_mismatch",
            path,
            "dataset_snapshot_report files must match dataset_manifest paths",
        )


def _expected_input_check_entries(snapshot_files: object) -> list[dict[str, object] | None]:
    if not isinstance(snapshot_files, list):
        return []
    expected: list[dict[str, object] | None] = []
    for item in snapshot_files:
        if not isinstance(item, dict):
            expected.append(None)
            continue
        path = item.get("path")
        source_path = item.get("source_path")
        split = item.get("split")
        description = item.get("description")
        source_sha256 = item.get("source_sha256")
        source_size_bytes = item.get("source_size_bytes")
        if not all(
            isinstance(value, str)
            for value in (path, source_path, split, description, source_sha256)
        ):
            expected.append(None)
            continue
        if not _is_positive_int(source_size_bytes):
            expected.append(None)
            continue
        expected.append(
            {
                "source_path": source_path,
                "staged_path": path,
                "split": split,
                "description": description,
                "sha256": source_sha256,
                "size_bytes": source_size_bytes,
            }
        )
    return expected


def _normalized_input_check_entry(item: dict[str, object]) -> dict[str, object] | None:
    source_path = item.get("source_path")
    staged_path = item.get("staged_path")
    split = item.get("split")
    description = item.get("description")
    sha256 = item.get("sha256")
    size_bytes = item.get("size_bytes")
    if not isinstance(source_path, str) or not _is_public_relative_reference(source_path):
        return None
    if not isinstance(staged_path, str) or not _is_public_relative_reference(staged_path):
        return None
    if not isinstance(split, str):
        return None
    if not isinstance(description, str):
        return None
    if not isinstance(sha256, str) or not _looks_like_sha256(sha256):
        return None
    if not _is_positive_int(size_bytes):
        return None
    return {
        "source_path": source_path,
        "staged_path": staged_path,
        "split": split,
        "description": description,
        "sha256": sha256,
        "size_bytes": size_bytes,
    }


def _verify_integrity_report(
    dataset_dir: Path,
    manifest: dict[str, Any],
    integrity: dict[str, Any],
    issues: list[TrainingPreflightIssue],
) -> None:
    if integrity.get("snapshot_id") != manifest.get("snapshot_id"):
        _issue(
            issues,
            "error",
            "dataset.integrity.snapshot_mismatch",
            dataset_dir / "split_integrity.json",
            "split_integrity snapshot_id must match dataset_manifest",
        )
    observed_manifest_hash = sha256_file(dataset_dir / "dataset_manifest.json")
    if integrity.get("manifest_sha256") != observed_manifest_hash:
        _issue(
            issues,
            "error",
            "dataset.integrity.manifest_hash_mismatch",
            dataset_dir / "split_integrity.json",
            "split_integrity manifest hash is stale",
        )
    checks = integrity.get("leakage_checks")
    if isinstance(checks, list):
        for index, check in enumerate(checks):
            if isinstance(check, dict) and check.get("status") != "passed":
                _issue(
                    issues,
                    "error",
                    "dataset.integrity.leakage_failed",
                    dataset_dir / "split_integrity.json",
                    f"leakage check {index} did not pass",
                )
    else:
        _issue(
            issues,
            "error",
            "dataset.integrity.leakage_missing",
            dataset_dir / "split_integrity.json",
            "leakage_checks must be a list",
        )


def _verify_core_checksums(
    dataset_dir: Path,
    checksum_entries: dict[str, str],
    issues: list[TrainingPreflightIssue],
) -> None:
    for relative in _DATASET_CORE_FILES[:-1]:
        path = dataset_dir / relative
        if path.is_file() and checksum_entries.get(relative) != sha256_file(path):
            _issue(
                issues,
                "error",
                f"dataset.{relative}.checksum_mismatch",
                path,
                "SHA256SUMS does not match the generated dataset package file",
            )


def _parse_sha256sums(path: Path, issues: list[TrainingPreflightIssue]) -> dict[str, str]:
    if not path.is_file():
        return {}
    entries: dict[str, str] = {}
    for line_no, raw_line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        if len(parts) != 2:
            _issue(
                issues,
                "error",
                "dataset.checksums.invalid",
                path,
                f"invalid SHA256SUMS line {line_no}",
            )
            continue
        digest, relative = parts
        entries[relative] = "sha256:" + digest.lower()
    return entries


def _require_train_and_eval_splits(
    splits: dict[Any, Any],
    issues: list[TrainingPreflightIssue],
    path: Path,
) -> None:
    split_names = {name for name in splits if isinstance(name, str)}
    has_train = any(name.lower().startswith("train") for name in split_names)
    has_eval = any(
        not name.lower().startswith("train") and name.lower().startswith(_EVAL_PREFIXES)
        for name in split_names
    )
    if not has_train:
        _issue(issues, "error", "dataset.train_split_missing", path, "a train split is required")
    if not has_eval:
        _issue(
            issues,
            "error",
            "dataset.eval_split_missing",
            path,
            "an eval/test/holdout split is required",
        )


def _load_json_object(
    path: Path,
    issues: list[TrainingPreflightIssue],
    code_prefix: str,
) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        _issue(issues, "error", f"{code_prefix}.invalid_json", path, str(exc))
        return None
    if not isinstance(payload, dict):
        _issue(issues, "error", f"{code_prefix}.invalid", path, "JSON must be an object")
        return None
    return payload


def _probe_dependency(import_name: str, required: bool) -> DependencyProbe:
    package = import_name.split(".", 1)[0]
    try:
        spec = importlib.util.find_spec(import_name)
    except ModuleNotFoundError:
        spec = None
    if spec is None:
        return DependencyProbe(
            import_name=import_name,
            package=package,
            required=required,
            available=False,
            version=None,
            reason=f"{import_name} is not importable",
        )
    try:
        version = importlib.metadata.version(package)
    except importlib.metadata.PackageNotFoundError:
        version = None
    return DependencyProbe(
        import_name=import_name,
        package=package,
        required=required,
        available=True,
        version=version,
        reason="importable",
    )


def _safe_relative(
    root: Path,
    relative: str,
    issues: list[TrainingPreflightIssue],
    *,
    code: str,
) -> Path | None:
    candidate = Path(relative)
    if candidate.is_absolute() or ".." in candidate.parts:
        _issue(issues, "error", code, relative, "path must stay inside dataset_dir")
        return None
    return root / candidate


def _file_identity(path: Path, *, public_path: str | None = None) -> dict[str, object]:
    return {
        "path": public_path or _public_path_reference(path),
        "sha256": sha256_file(path),
        "size_bytes": path.stat().st_size,
    }


def _public_path_reference(path: Path) -> str:
    if not path.is_absolute() and ".." not in path.parts:
        return path.as_posix()
    return path.name


def _first_existing(root: Path, names: tuple[str, ...]) -> Path | None:
    for name in names:
        path = root / name
        if path.is_file():
            return path
    return None


def _text(value: object) -> str | None:
    return value if isinstance(value, str) and value.strip() else None


def _is_public_relative_reference(value: str) -> bool:
    path = Path(value)
    return bool(value.strip()) and not path.is_absolute() and ".." not in path.parts


def _is_positive_int(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value > 0


def _looks_like_sha256(value: str) -> bool:
    prefix = "sha256:"
    digest = value.removeprefix(prefix)
    return (
        value.startswith(prefix)
        and len(digest) == 64
        and all(char in "0123456789abcdefABCDEF" for char in digest)
    )


def _looks_like_fixture(value: str) -> bool:
    lowered = value.lower()
    return any(token in lowered for token in ("fixture", "test", "smoke", "dummy"))


def _issue(
    issues: list[TrainingPreflightIssue],
    severity: Severity,
    code: str,
    path: str | Path,
    message: str,
) -> None:
    issues.append(
        TrainingPreflightIssue(
            severity=severity,
            code=code,
            path=str(path),
            message=message,
        )
    )


def _utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")
