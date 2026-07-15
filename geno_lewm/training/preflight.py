# SPDX-License-Identifier: Apache-2.0
"""Clean-machine preflight for Carbon-backed GenoLeWM training."""

from __future__ import annotations

import importlib
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
from geno_lewm.training._phase_contract import (
    PHASE2_ADAPTER_UNAVAILABLE_CODE,
    PHASE2_ADAPTER_UNAVAILABLE_MESSAGE,
)

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
MIN_CUDA_VRAM_GB: Final = 40.0
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
_LEGACY_DATASET_SCHEMA_VERSION: Final = "1.0.0"
_ARTIFACT_ROLE_DATASET_SCHEMA_VERSION: Final = "1.1.0"
_DATASET_ARTIFACT_ROLES: Final = frozenset({"split_data", "split_companion", "evidence"})
_MEMBERSHIP_STORE_FILES: Final = frozenset(
    {
        "manifest.json",
        "memberships.parquet",
        "lookup.sqlite",
        "snapshot-lineage.json",
        "build-receipt.json",
    }
)
_MEMBERSHIP_STORE_BINDING_KEYS: Final = frozenset(
    {"path", "artifact_id", "content_identity", "physical_identity", "rowset_sha256"}
)
_SPLIT_REPORT_BINDING_KEYS: Final = frozenset(
    {"path", "schema_path", "artifact_id", "schema_version"}
)
_EVAL_PREFIXES: Final = ("eval", "test", "holdout", "validation", "val")
Severity = Literal["error", "warning"]

__all__ = [
    "MIN_CUDA_VRAM_GB",
    "REPORT_NAME",
    "REQUIRED_TRAINING_MODULES",
    "SCHEMA_VERSION",
    "AcceleratorProbe",
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
    require_accelerator: bool = True
    min_cuda_vram_gb: float = MIN_CUDA_VRAM_GB


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
class AcceleratorProbe:
    """CUDA accelerator readiness probe for Carbon-backed training."""

    requested_device: str | None
    required: bool
    available: bool
    device_count: int
    device_name: str | None
    total_memory_bytes: int | None
    min_memory_bytes: int
    reason: str
    issue_code: str | None = None

    def to_dict(self) -> dict[str, object]:
        return {
            "requested_device": self.requested_device,
            "required": self.required,
            "available": self.available,
            "device_count": self.device_count,
            "device_name": self.device_name,
            "total_memory_bytes": self.total_memory_bytes,
            "min_memory_bytes": self.min_memory_bytes,
            "reason": self.reason,
            "issue_code": self.issue_code,
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
    accelerator: AcceleratorProbe
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
            "accelerator": self.accelerator.to_dict(),
            "dependencies": [probe.to_dict() for probe in self.dependencies],
            "issues": [issue.to_dict() for issue in self.issues],
        }


DependencyProbeFn = Callable[[str, bool], DependencyProbe]
AcceleratorProbeFn = Callable[[str | None, bool, int], AcceleratorProbe]


def build_training_preflight_report(
    request: TrainingPreflightRequest,
    *,
    generated_at: str | None = None,
    dependency_probe: DependencyProbeFn | None = None,
    accelerator_probe: AcceleratorProbeFn | None = None,
) -> TrainingPreflightReport:
    """Build clean-machine readiness evidence for Carbon-backed training."""
    issues: list[TrainingPreflightIssue] = []
    dependency_probe = dependency_probe or _probe_dependency
    accelerator_probe = accelerator_probe or _probe_accelerator
    dataset = _inspect_dataset(request.dataset_dir, request.allow_fixture_dataset, issues)
    carbon = _inspect_carbon_model_dir(request.carbon_model_dir, issues)
    training_config = _inspect_training_config(request.training_config, issues)
    run_dir = _inspect_run_dir(request.run_dir)
    min_cuda_memory_bytes = _min_cuda_memory_bytes(request.min_cuda_vram_gb)
    accelerator = accelerator_probe(
        _requested_training_device(training_config),
        request.require_accelerator,
        min_cuda_memory_bytes,
    )
    dependencies = tuple(
        dependency_probe(name, request.require_native_runtime) for name in REQUIRED_TRAINING_MODULES
    )
    if accelerator.required and not accelerator.available:
        _issue(
            issues,
            "error",
            accelerator.issue_code or "training.accelerator_unavailable",
            request.training_config,
            accelerator.reason,
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
        accelerator=accelerator,
        dependencies=dependencies,
        issues=tuple(issues),
    )


def write_training_preflight_report(
    request: TrainingPreflightRequest,
    output: Path | None = None,
    *,
    generated_at: str | None = None,
    dependency_probe: DependencyProbeFn | None = None,
    accelerator_probe: AcceleratorProbeFn | None = None,
) -> TrainingPreflightReport:
    """Write ``training_preflight_report.json`` and return the report."""
    report = build_training_preflight_report(
        request,
        generated_at=generated_at,
        dependency_probe=dependency_probe,
        accelerator_probe=accelerator_probe,
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
    dataset_schema_version = _text(manifest.get("schema_version"))
    dataset["schema_version"] = dataset_schema_version
    if dataset_schema_version not in {
        _LEGACY_DATASET_SCHEMA_VERSION,
        _ARTIFACT_ROLE_DATASET_SCHEMA_VERSION,
    }:
        _issue(
            issues,
            "error",
            "dataset.schema_version",
            dataset_dir / "dataset_manifest.json",
            "dataset schema_version must be 1.0.0 or 1.1.0",
        )
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
        dataset["files"] = _inspect_dataset_files(
            dataset_dir,
            raw_files,
            checksum_entries,
            issues,
            schema_version=dataset_schema_version,
        )
    else:
        _issue(issues, "error", "dataset.files.invalid", dataset_dir, "files must be a list")
    if package is not None:
        _verify_dataset_package_metadata(dataset_dir, manifest, package, issues)
    binding = _inspect_membership_and_split_evidence(
        dataset_dir,
        manifest=manifest,
        issues=issues,
    )
    if binding is not None:
        dataset["membership_and_split_evidence"] = binding
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
    _verify_training_phase(resolved, training_config, issues)
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


def _verify_training_phase(
    config: object,
    training_config: Path,
    issues: list[TrainingPreflightIssue],
) -> None:
    if getattr(config, "phase", None) != "phase2":
        return
    _issue(
        issues,
        "error",
        PHASE2_ADAPTER_UNAVAILABLE_CODE,
        training_config,
        PHASE2_ADAPTER_UNAVAILABLE_MESSAGE,
    )


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
    *,
    schema_version: str | None,
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
        semantics = _dataset_file_semantics(
            item,
            index=index,
            schema_version=schema_version,
            path=path,
            issues=issues,
        )
        files.append({"path": relative, **semantics, **identity})
    if schema_version == _ARTIFACT_ROLE_DATASET_SCHEMA_VERSION:
        _verify_dataset_companions(files, dataset_dir=dataset_dir, issues=issues)
    return files


def _dataset_file_semantics(
    item: dict[str, Any],
    *,
    index: int,
    schema_version: str | None,
    path: Path,
    issues: list[TrainingPreflightIssue],
) -> dict[str, object]:
    if schema_version != _ARTIFACT_ROLE_DATASET_SCHEMA_VERSION:
        if "artifact_role" in item or "companion_of" in item:
            _issue(
                issues,
                "error",
                "dataset.file.role_forbidden",
                path,
                "schema 1.0.0 files must not declare artifact roles",
            )
        return {"split": item.get("split"), "records": item.get("records")}

    role = item.get("artifact_role")
    if not isinstance(role, str) or role not in _DATASET_ARTIFACT_ROLES:
        _issue(
            issues,
            "error",
            "dataset.file.artifact_role",
            path,
            f"files[{index}].artifact_role is invalid",
        )
        return {"artifact_role": role}
    semantics: dict[str, object] = {"artifact_role": role}
    if role == "evidence":
        if any(key in item for key in ("split", "records", "companion_of")):
            _issue(
                issues,
                "error",
                "dataset.file.evidence_fields",
                path,
                "evidence files must not declare split, records, or companion_of",
            )
        return semantics

    split = item.get("split")
    records = item.get("records")
    if not isinstance(split, str) or not split:
        _issue(
            issues,
            "error",
            "dataset.file.split",
            path,
            "split_data and split_companion files require split",
        )
    else:
        semantics["split"] = split
    if not _is_non_negative_int(records):
        _issue(
            issues,
            "error",
            "dataset.file.records",
            path,
            "split_data and split_companion files require non-negative records",
        )
    else:
        semantics["records"] = records
    if role == "split_companion":
        companion_of = item.get("companion_of")
        if not isinstance(companion_of, str) or not companion_of:
            _issue(
                issues,
                "error",
                "dataset.file.companion_of",
                path,
                "split_companion files require companion_of",
            )
        else:
            semantics["companion_of"] = companion_of
    elif "companion_of" in item:
        _issue(
            issues,
            "error",
            "dataset.file.companion_of",
            path,
            "only split_companion files may declare companion_of",
        )
    return semantics


def _verify_dataset_companions(
    files: list[dict[str, object]],
    *,
    dataset_dir: Path,
    issues: list[TrainingPreflightIssue],
) -> None:
    by_path = {item.get("path"): item for item in files if isinstance(item.get("path"), str)}
    for item in files:
        if item.get("artifact_role") != "split_companion":
            continue
        target = by_path.get(item.get("companion_of"))
        if (
            target is None
            or target.get("artifact_role") != "split_data"
            or (item.get("split"), item.get("records"))
            != (target.get("split"), target.get("records"))
        ):
            _issue(
                issues,
                "error",
                "dataset.file.companion_binding",
                dataset_dir / str(item.get("path")),
                "split companion binding does not match one split_data artifact",
            )


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
    if package.get("schema_version") != manifest.get("schema_version"):
        _issue(
            issues,
            "error",
            "dataset.package.schema_mismatch",
            path,
            "dataset_package schema_version must match dataset_manifest",
        )
    if package.get("files") != manifest.get("files"):
        _issue(
            issues,
            "error",
            "dataset.package.files_mismatch",
            path,
            "dataset_package files must match dataset_manifest",
        )
    if package.get("membership_and_split_evidence") != manifest.get(
        "membership_and_split_evidence"
    ):
        _issue(
            issues,
            "error",
            "dataset.package.membership_binding_mismatch",
            path,
            "dataset_package membership and split evidence must match dataset_manifest",
        )


def _inspect_membership_and_split_evidence(
    dataset_dir: Path,
    *,
    manifest: dict[str, Any],
    issues: list[TrainingPreflightIssue],
) -> dict[str, object] | None:
    raw = manifest.get("membership_and_split_evidence")
    schema_version = manifest.get("schema_version")
    if raw is None:
        return None
    path = dataset_dir / "dataset_manifest.json"
    if schema_version != _ARTIFACT_ROLE_DATASET_SCHEMA_VERSION:
        _issue(
            issues,
            "error",
            "dataset.membership_binding.schema_version",
            path,
            "membership and split evidence requires dataset schema 1.1.0",
        )
        return None
    if not isinstance(raw, dict) or set(raw) != {"membership_store", "report"}:
        _issue(
            issues,
            "error",
            "dataset.membership_binding.shape",
            path,
            "membership_and_split_evidence must contain exactly membership_store and report",
        )
        return None
    store = raw.get("membership_store")
    report_binding = raw.get("report")
    if not isinstance(store, dict) or set(store) != _MEMBERSHIP_STORE_BINDING_KEYS:
        _issue(
            issues,
            "error",
            "dataset.membership_binding.store",
            path,
            "membership store binding fields are invalid",
        )
        return None
    if not isinstance(report_binding, dict) or set(report_binding) != _SPLIT_REPORT_BINDING_KEYS:
        _issue(
            issues,
            "error",
            "dataset.membership_binding.report",
            path,
            "split report binding fields are invalid",
        )
        return None
    if any(
        not isinstance(value, str) or not value
        for value in (*store.values(), *report_binding.values())
    ):
        _issue(
            issues,
            "error",
            "dataset.membership_binding.values",
            path,
            "membership and split evidence binding values must be non-empty strings",
        )
        return None

    raw_files = manifest.get("files")
    by_path = (
        {
            item.get("path"): item
            for item in raw_files
            if isinstance(raw_files, list) and isinstance(item, dict)
        }
        if isinstance(raw_files, list)
        else {}
    )
    store_root = str(store["path"])
    store_root_path = _safe_relative(
        dataset_dir,
        store_root,
        issues,
        code="dataset.membership_binding.store_path",
    )
    report_path = _safe_relative(
        dataset_dir,
        str(report_binding["path"]),
        issues,
        code="dataset.membership_binding.report_path",
    )
    schema_path = _safe_relative(
        dataset_dir,
        str(report_binding["schema_path"]),
        issues,
        code="dataset.membership_binding.schema_path",
    )
    if store_root_path is None or report_path is None or schema_path is None:
        return None
    required_paths = {
        *(f"{store_root}/{name}" for name in _MEMBERSHIP_STORE_FILES),
        str(report_binding["path"]),
        str(report_binding["schema_path"]),
    }
    invalid_paths = sorted(
        relative
        for relative in required_paths
        if not isinstance(by_path.get(relative), dict)
        or by_path[relative].get("artifact_role") != "evidence"
    )
    if invalid_paths:
        _issue(
            issues,
            "error",
            "dataset.membership_binding.evidence_files",
            path,
            f"membership evidence paths are not declared evidence files: {invalid_paths}",
        )

    store_manifest = _load_json_object(
        store_root_path / "manifest.json",
        issues,
        "dataset.membership_store.manifest",
    )
    expected_store = {
        key: store[key]
        for key in ("artifact_id", "content_identity", "physical_identity", "rowset_sha256")
    }
    if store_manifest is not None:
        observed_store = {key: store_manifest.get(key) for key in expected_store}
        if observed_store != expected_store:
            _issue(
                issues,
                "error",
                "dataset.membership_binding.store_identity",
                store_root_path / "manifest.json",
                "membership store manifest identities do not match the dataset binding",
            )

    report = _load_json_object(
        report_path,
        issues,
        "dataset.membership_split_report",
    )
    if report is not None:
        observed_report = {
            "artifact_id": report.get("artifact_id"),
            "schema_version": report.get("schema_version"),
        }
        expected_report = {
            "artifact_id": report_binding["artifact_id"],
            "schema_version": report_binding["schema_version"],
        }
        report_store = report.get("membership_store")
        observed_report_store = (
            {key: report_store.get(key) for key in expected_store}
            if isinstance(report_store, dict)
            else None
        )
        claim = report.get("claim_boundary")
        if (
            observed_report != expected_report
            or observed_report_store != expected_store
            or not isinstance(claim, dict)
            or claim.get("publication_eligible") is not True
        ):
            _issue(
                issues,
                "error",
                "dataset.membership_binding.report_identity",
                report_path,
                "split report does not match the publication-eligible dataset binding",
            )
    return {
        "membership_store": dict(store),
        "report": dict(report_binding),
    }


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
        return
    for index, (snapshot_file, manifest_file) in enumerate(
        zip(raw_snapshot_files, raw_manifest_files, strict=True)
    ):
        if not isinstance(snapshot_file, dict) or not isinstance(manifest_file, dict):
            continue
        for key in ("split", "records", "artifact_role", "companion_of", "description"):
            if snapshot_file.get(key) != manifest_file.get(key):
                _issue(
                    issues,
                    "error",
                    f"dataset.snapshot_report.file.{key}",
                    path,
                    f"snapshot file {index} {key} does not match dataset_manifest",
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
        source_sha256 = item.get("source_sha256")
        source_size_bytes = item.get("source_size_bytes")
        if not all(isinstance(value, str) for value in (path, source_path, source_sha256)):
            expected.append(None)
            continue
        if not _is_positive_int(source_size_bytes):
            expected.append(None)
            continue
        semantic_fields = _normalized_snapshot_input_semantic_fields(item)
        if semantic_fields is None:
            expected.append(None)
            continue
        expected.append(
            {
                "source_path": source_path,
                "staged_path": path,
                **semantic_fields,
                "sha256": source_sha256,
                "size_bytes": source_size_bytes,
            }
        )
    return expected


def _normalized_input_check_entry(item: dict[str, object]) -> dict[str, object] | None:
    source_path = item.get("source_path")
    staged_path = item.get("staged_path")
    sha256 = item.get("sha256")
    size_bytes = item.get("size_bytes")
    if not isinstance(source_path, str) or not _is_public_relative_reference(source_path):
        return None
    if not isinstance(staged_path, str) or not _is_public_relative_reference(staged_path):
        return None
    if not isinstance(sha256, str) or not _looks_like_sha256(sha256):
        return None
    if not _is_positive_int(size_bytes):
        return None
    semantic_fields = _normalized_snapshot_input_semantic_fields(item)
    if semantic_fields is None:
        return None
    return {
        "source_path": source_path,
        "staged_path": staged_path,
        **semantic_fields,
        "sha256": sha256,
        "size_bytes": size_bytes,
    }


def _normalized_snapshot_input_semantic_fields(
    item: dict[str, object],
) -> dict[str, object] | None:
    if "artifact_role" not in item:
        split = item.get("split")
        description = item.get("description")
        if not isinstance(split, str) or not isinstance(description, str):
            return None
        return {"split": split, "description": description}

    role = item.get("artifact_role")
    if not isinstance(role, str) or role not in _DATASET_ARTIFACT_ROLES:
        return None
    semantics: dict[str, object] = {"artifact_role": role}
    if role == "evidence":
        if "split" in item or "companion_of" in item:
            return None
    else:
        split = item.get("split")
        if not isinstance(split, str) or not split:
            return None
        semantics["split"] = split
    if role == "split_companion":
        companion_of = item.get("companion_of")
        if not isinstance(companion_of, str) or not companion_of:
            return None
        semantics["companion_of"] = companion_of
    elif "companion_of" in item:
        return None
    if "description" in item:
        description = item.get("description")
        if not isinstance(description, str) or not description:
            return None
        semantics["description"] = description
    return semantics


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


def _requested_training_device(training_config: dict[str, object]) -> str | None:
    resolved = training_config.get("resolved")
    if not isinstance(resolved, dict):
        return None
    runtime = resolved.get("runtime")
    if not isinstance(runtime, dict):
        return None
    device = runtime.get("device")
    return device if isinstance(device, str) else None


def _min_cuda_memory_bytes(min_cuda_vram_gb: float) -> int:
    if isinstance(min_cuda_vram_gb, bool) or min_cuda_vram_gb <= 0:
        return int(MIN_CUDA_VRAM_GB * 1024**3)
    return int(float(min_cuda_vram_gb) * 1024**3)


def _probe_accelerator(
    requested_device: str | None,
    required: bool,
    min_memory_bytes: int,
) -> AcceleratorProbe:
    if requested_device != "cuda":
        return AcceleratorProbe(
            requested_device=requested_device,
            required=required,
            available=False,
            device_count=0,
            device_name=None,
            total_memory_bytes=None,
            min_memory_bytes=min_memory_bytes,
            reason="Carbon-backed first-experiment training requires runtime.device: cuda",
            issue_code="training_config.runtime.device_not_cuda",
        )
    try:
        torch = importlib.import_module("torch")
    except ImportError:
        return AcceleratorProbe(
            requested_device=requested_device,
            required=required,
            available=False,
            device_count=0,
            device_name=None,
            total_memory_bytes=None,
            min_memory_bytes=min_memory_bytes,
            reason="torch is required to probe CUDA availability",
            issue_code="training.cuda.torch_unavailable",
        )
    cuda = getattr(torch, "cuda", None)
    if cuda is None or not bool(cuda.is_available()):
        return AcceleratorProbe(
            requested_device=requested_device,
            required=required,
            available=False,
            device_count=0,
            device_name=None,
            total_memory_bytes=None,
            min_memory_bytes=min_memory_bytes,
            reason="CUDA is not available for Carbon-backed training",
            issue_code="training.cuda.unavailable",
        )
    device_count = int(cuda.device_count()) if callable(getattr(cuda, "device_count", None)) else 0
    device_name = None
    total_memory_bytes = None
    try:
        props = cuda.get_device_properties(0)
        device_name = str(getattr(props, "name", None) or cuda.get_device_name(0))
        total_memory_bytes = int(props.total_memory)
    except Exception:
        get_name = getattr(cuda, "get_device_name", None)
        device_name = str(get_name(0)) if callable(get_name) else None
    if total_memory_bytes is None:
        return AcceleratorProbe(
            requested_device=requested_device,
            required=required,
            available=False,
            device_count=device_count,
            device_name=device_name,
            total_memory_bytes=None,
            min_memory_bytes=min_memory_bytes,
            reason="CUDA device memory could not be probed",
            issue_code="training.cuda.memory_unknown",
        )
    if total_memory_bytes < min_memory_bytes:
        return AcceleratorProbe(
            requested_device=requested_device,
            required=required,
            available=False,
            device_count=device_count,
            device_name=device_name,
            total_memory_bytes=total_memory_bytes,
            min_memory_bytes=min_memory_bytes,
            reason=(
                f"CUDA device memory {total_memory_bytes} bytes is below the required "
                f"{min_memory_bytes} bytes"
            ),
            issue_code="training.cuda.vram_too_low",
        )
    return AcceleratorProbe(
        requested_device=requested_device,
        required=required,
        available=True,
        device_count=device_count,
        device_name=device_name,
        total_memory_bytes=total_memory_bytes,
        min_memory_bytes=min_memory_bytes,
        reason="cuda accelerator satisfies the training preflight requirement",
    )


def _safe_relative(
    root: Path,
    relative: str,
    issues: list[TrainingPreflightIssue],
    *,
    code: str,
) -> Path | None:
    candidate = Path(relative)
    if candidate.is_absolute() or ".." in candidate.parts or "\\" in relative:
        _issue(issues, "error", code, relative, "path must stay inside dataset_dir")
        return None
    resolved = root
    for part in candidate.parts:
        resolved /= part
        if resolved.is_symlink():
            _issue(issues, "error", code, relative, "path must not traverse symbolic links")
            return None
    return resolved


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


def _is_non_negative_int(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value >= 0


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
