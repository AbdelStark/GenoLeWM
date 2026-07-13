# SPDX-License-Identifier: Apache-2.0
"""Compare real training-run archives for deterministic reproducibility evidence."""

from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Final

import yaml

from geno_lewm.errors import GenoLeWMError, InputError, exit_code_for
from tools.release.training_run import TrainingRunManifest, verify_training_run_manifest

REPORT_NAME: Final = "training_reproducibility_report.json"
SCHEMA_VERSION: Final = "2.0.0"
GENERATED_BY: Final = "tools.release.training_reproducibility"
_DETERMINISTIC_PAIR_FIELDS: Final = (
    "run_id",
    "commit_sha",
    "package_version",
    "dataset_snapshot_id",
    "seeds",
    "determinism",
)
_THROUGHPUT_IDENTITY_FIELDS: Final = (
    "run_id",
    "commit_sha",
    "package_version",
    "dataset_snapshot_id",
    "seeds",
)
_REQUIRED_PAIR_ARTIFACT_KINDS: Final = frozenset(
    {"dataset_manifest", "training_config", "checkpoint"}
)
_THROUGHPUT_KEYS: Final = (
    "samples_per_second",
    "samples_per_s",
    "throughput_samples_per_s",
    "train_samples_per_second",
)
_ELAPSED_KEYS: Final = ("elapsed_seconds", "wall_time_seconds", "duration_seconds")
EXPECTED_STEPS: Final = 500
EXPECTED_SAMPLE_COUNT: Final = 4_000


@dataclass(frozen=True, slots=True)
class ReproducibilityBlocker:
    """One failed reproducibility-evidence requirement."""

    code: str
    message: str
    details: dict[str, object]

    def to_dict(self) -> dict[str, object]:
        return {
            "code": self.code,
            "message": self.message,
            "details": self.details,
        }


@dataclass(frozen=True, slots=True)
class ArtifactIdentity:
    """Stable identity for one training-run artifact."""

    key: str
    kind: str
    path: str
    sha256: str
    size_bytes: int

    def to_dict(self) -> dict[str, object]:
        return {
            "kind": self.kind,
            "path": self.path,
            "sha256": self.sha256,
            "size_bytes": self.size_bytes,
        }


@dataclass(frozen=True, slots=True)
class TrainingRunEvidence:
    """Validated evidence loaded from one training-run archive."""

    label: str
    run_dir: str
    run_id: str
    commit_sha: str
    package_version: str
    dataset_snapshot_id: str
    seeds: dict[str, int]
    determinism: str
    determinism_payload: dict[str, Any]
    deterministic_enabled: bool
    steps_completed: int | None
    sample_count: int | None
    new_sample_count: int | None
    resumed_from_step: int | None
    resume_checkpoint: object
    resume_checkpoint_recorded: bool
    nan_loss_count: int | None
    collapse_alert_count: int | None
    elapsed_seconds: float | None
    samples_per_second: float | None
    training_config: dict[str, Any]
    accelerator_name: str | None
    runtime_identity: dict[str, Any]
    artifacts: tuple[ArtifactIdentity, ...]

    def artifact_map(self) -> dict[str, ArtifactIdentity]:
        return {artifact.key: artifact for artifact in self.artifacts}

    def to_dict(self) -> dict[str, object]:
        return {
            "label": self.label,
            "run_dir": self.run_dir,
            "run_id": self.run_id,
            "commit_sha": self.commit_sha,
            "package_version": self.package_version,
            "dataset_snapshot_id": self.dataset_snapshot_id,
            "seeds": self.seeds,
            "determinism": self.determinism,
            "determinism_payload": self.determinism_payload,
            "deterministic_enabled": self.deterministic_enabled,
            "steps_completed": self.steps_completed,
            "sample_count": self.sample_count,
            "new_sample_count": self.new_sample_count,
            "resumed_from_step": self.resumed_from_step,
            "resume_checkpoint": self.resume_checkpoint,
            "resume_checkpoint_recorded": self.resume_checkpoint_recorded,
            "nan_loss_count": self.nan_loss_count,
            "collapse_alert_count": self.collapse_alert_count,
            "elapsed_seconds": self.elapsed_seconds,
            "samples_per_second": self.samples_per_second,
            "training_config": self.training_config,
            "accelerator_name": self.accelerator_name,
            "runtime_identity": self.runtime_identity,
            "artifacts": {artifact.key: artifact.to_dict() for artifact in self.artifacts},
        }


@dataclass(frozen=True, slots=True)
class DeterministicPairComparison:
    """Bit-exact deterministic rerun comparison."""

    ok: bool
    matched_fields: tuple[str, ...]
    matched_artifacts: tuple[str, ...]
    blockers: tuple[ReproducibilityBlocker, ...]

    def to_dict(self) -> dict[str, object]:
        return {
            "ok": self.ok,
            "matched_fields": list(self.matched_fields),
            "matched_artifacts": list(self.matched_artifacts),
            "blockers": [blocker.to_dict() for blocker in self.blockers],
        }


@dataclass(frozen=True, slots=True)
class RunContractValidation:
    """Per-arm completion and safety checks for the H200 N-D-D-N suite."""

    ok: bool
    expected_steps: int
    expected_sample_count: int
    blockers: tuple[ReproducibilityBlocker, ...]

    def to_dict(self) -> dict[str, object]:
        return {
            "ok": self.ok,
            "expected_steps": self.expected_steps,
            "expected_sample_count": self.expected_sample_count,
            "blockers": [blocker.to_dict() for blocker in self.blockers],
        }


@dataclass(frozen=True, slots=True)
class ThroughputComparison:
    """Conservative throughput comparison across counterbalanced repeat pairs."""

    ok: bool
    status: str
    max_drop_fraction: float
    max_repeat_spread_fraction: float
    baseline_samples_per_second: dict[str, float]
    baseline_max_samples_per_second: float | None
    baseline_spread_fraction: float | None
    deterministic_samples_per_second: dict[str, float]
    deterministic_min_samples_per_second: float | None
    deterministic_spread_fraction: float | None
    drop_fraction: float | None
    threshold_evaluated: bool
    blockers: tuple[ReproducibilityBlocker, ...]

    def to_dict(self) -> dict[str, object]:
        return {
            "ok": self.ok,
            "status": self.status,
            "max_drop_fraction": self.max_drop_fraction,
            "max_repeat_spread_fraction": self.max_repeat_spread_fraction,
            "baseline_samples_per_second": self.baseline_samples_per_second,
            "baseline_max_samples_per_second": self.baseline_max_samples_per_second,
            "baseline_spread_fraction": self.baseline_spread_fraction,
            "baseline_spread_definition": "(max-min)/max",
            "deterministic_samples_per_second": self.deterministic_samples_per_second,
            "deterministic_min_samples_per_second": self.deterministic_min_samples_per_second,
            "deterministic_spread_fraction": self.deterministic_spread_fraction,
            "deterministic_spread_definition": "(max-min)/max",
            "drop_fraction": self.drop_fraction,
            "drop_definition": "max(0,(baseline_max-det_min)/baseline_max)",
            "threshold_evaluated": self.threshold_evaluated,
            "blockers": [blocker.to_dict() for blocker in self.blockers],
        }


@dataclass(frozen=True, slots=True)
class TrainingReproducibilityReport:
    """Machine-readable deterministic training evidence."""

    schema_version: str
    generated_by: str
    generated_at: str
    ok: bool
    max_throughput_drop_fraction: float
    max_repeat_spread_fraction: float
    runs: tuple[TrainingRunEvidence, ...]
    run_contract: RunContractValidation
    deterministic_pair: DeterministicPairComparison
    throughput: ThroughputComparison

    @property
    def blockers(self) -> tuple[ReproducibilityBlocker, ...]:
        return (
            *self.run_contract.blockers,
            *self.deterministic_pair.blockers,
            *self.throughput.blockers,
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "generated_by": self.generated_by,
            "generated_at": self.generated_at,
            "ok": self.ok,
            "max_throughput_drop_fraction": self.max_throughput_drop_fraction,
            "max_repeat_spread_fraction": self.max_repeat_spread_fraction,
            "runs": [run.to_dict() for run in self.runs],
            "run_contract": self.run_contract.to_dict(),
            "deterministic_pair": self.deterministic_pair.to_dict(),
            "throughput": self.throughput.to_dict(),
            "blockers": [blocker.to_dict() for blocker in self.blockers],
        }


def build_training_reproducibility_report(
    *,
    deterministic_run_a: Path,
    deterministic_run_b: Path,
    baseline_run_a: Path | None = None,
    baseline_run_b: Path | None = None,
    max_throughput_drop_fraction: float = 0.15,
    max_repeat_spread_fraction: float = 0.05,
    require_preflight: bool = False,
    generated_at: str | None = None,
) -> TrainingReproducibilityReport:
    """Build a report for issue #47's remaining real-run evidence gates."""
    _validate_fraction_threshold(
        max_throughput_drop_fraction,
        field="max_throughput_drop_fraction",
    )
    _validate_fraction_threshold(
        max_repeat_spread_fraction,
        field="max_repeat_spread_fraction",
    )
    baseline_a = (
        None
        if baseline_run_a is None
        else load_training_run_evidence(
            baseline_run_a,
            label="baseline_a",
            require_preflight=require_preflight,
        )
    )
    run_a = load_training_run_evidence(
        deterministic_run_a,
        label="deterministic_a",
        require_preflight=require_preflight,
    )
    run_b = load_training_run_evidence(
        deterministic_run_b,
        label="deterministic_b",
        require_preflight=require_preflight,
    )
    baseline_b = (
        None
        if baseline_run_b is None
        else load_training_run_evidence(
            baseline_run_b,
            label="baseline_b",
            require_preflight=require_preflight,
        )
    )
    deterministic_pair = compare_deterministic_pair(run_a, run_b)
    throughput = compare_throughput(
        baseline_a,
        run_a,
        run_b,
        baseline_b,
        max_drop_fraction=max_throughput_drop_fraction,
        max_repeat_spread_fraction=max_repeat_spread_fraction,
    )
    available_runs = (baseline_a, run_a, run_b, baseline_b)
    available_paths = (baseline_run_a, deterministic_run_a, deterministic_run_b, baseline_run_b)
    runs = tuple(run for run in available_runs if run is not None)
    source_dirs = tuple(
        path for run, path in zip(available_runs, available_paths, strict=True) if run is not None
    )
    run_contract = validate_run_contract(runs, source_dirs=source_dirs)
    ok = run_contract.ok and deterministic_pair.ok and throughput.ok
    return TrainingReproducibilityReport(
        schema_version=SCHEMA_VERSION,
        generated_by=GENERATED_BY,
        generated_at=generated_at or _utc_now(),
        ok=ok,
        max_throughput_drop_fraction=max_throughput_drop_fraction,
        max_repeat_spread_fraction=max_repeat_spread_fraction,
        runs=runs,
        run_contract=run_contract,
        deterministic_pair=deterministic_pair,
        throughput=throughput,
    )


def load_training_run_evidence(
    run_dir: Path,
    *,
    label: str,
    require_preflight: bool = False,
) -> TrainingRunEvidence:
    """Load one validated training-run archive into comparison-ready evidence."""
    manifest = verify_training_run_manifest(run_dir, require_preflight=require_preflight)
    metrics = _load_metrics_payload(run_dir, manifest)
    training_config = _load_training_config_payload(run_dir, manifest)
    preflight = _load_training_preflight_payload(run_dir, manifest)
    runtime_identity = _runtime_identity(manifest, preflight)
    accelerator = preflight.get("accelerator")
    accelerator_name = (
        accelerator.get("device_name")
        if isinstance(accelerator, dict) and isinstance(accelerator.get("device_name"), str)
        else None
    )
    determinism_payload = _parse_determinism_payload(manifest.determinism)
    return TrainingRunEvidence(
        label=label,
        run_dir=_public_path(run_dir),
        run_id=manifest.run_id,
        commit_sha=manifest.commit_sha,
        package_version=manifest.package_version,
        dataset_snapshot_id=manifest.dataset_snapshot_id,
        seeds=dict(manifest.seeds),
        determinism=manifest.determinism,
        determinism_payload=determinism_payload,
        deterministic_enabled=_deterministic_enabled(determinism_payload),
        steps_completed=_positive_int(_metric_value(metrics, "steps_completed")),
        sample_count=_extract_sample_count(metrics),
        new_sample_count=_positive_int(_metric_value(metrics, "new_sample_count")),
        resumed_from_step=_nonnegative_int(_metric_value(metrics, "resumed_from_step")),
        resume_checkpoint=metrics.get("resume_checkpoint"),
        resume_checkpoint_recorded="resume_checkpoint" in metrics,
        nan_loss_count=_nonnegative_int(_metric_value(metrics, "nan_loss_count")),
        collapse_alert_count=_nonnegative_int(_metric_value(metrics, "collapse_alert_count")),
        elapsed_seconds=_extract_elapsed_seconds(metrics),
        samples_per_second=_extract_samples_per_second(metrics),
        training_config=training_config,
        accelerator_name=accelerator_name,
        runtime_identity=runtime_identity,
        artifacts=_artifact_identities(manifest),
    )


def validate_run_contract(
    runs: tuple[TrainingRunEvidence, ...],
    *,
    source_dirs: tuple[Path | None, ...] | None = None,
) -> RunContractValidation:
    """Require every arm to complete the predeclared 500-step workload."""
    blockers: list[ReproducibilityBlocker] = []
    expected_labels = ("baseline_a", "deterministic_a", "deterministic_b", "baseline_b")
    observed_labels = tuple(run.label for run in runs)
    if observed_labels != expected_labels:
        blockers.append(
            _blocker(
                "run_contract.execution_order_mismatch",
                "the suite must bind four arms in counterbalanced N-D-D-N order",
                expected=list(expected_labels),
                observed=list(observed_labels),
            )
        )
    run_dirs = (
        tuple(Path(path).resolve() for path in source_dirs if path is not None)
        if source_dirs is not None
        else tuple(Path(run.run_dir) for run in runs)
    )
    if len(run_dirs) != len(runs) or len(set(run_dirs)) != len(run_dirs):
        blockers.append(
            _blocker(
                "run_contract.run_directory_reused",
                "every counterbalanced arm must use an independent run directory",
                observed=[_public_path(path) for path in run_dirs],
            )
        )
    expected_runtime = runs[0].runtime_identity if runs else {}
    for run in runs:
        if run.steps_completed != EXPECTED_STEPS:
            blockers.append(
                _blocker(
                    "run_contract.steps_completed_mismatch",
                    "every reproducibility arm must complete exactly 500 optimizer steps",
                    run=run.run_dir,
                    expected=EXPECTED_STEPS,
                    observed=run.steps_completed,
                )
            )
        if run.sample_count != EXPECTED_SAMPLE_COUNT:
            blockers.append(
                _blocker(
                    "run_contract.sample_count_mismatch",
                    "every reproducibility arm must process exactly 4000 samples",
                    run=run.run_dir,
                    expected=EXPECTED_SAMPLE_COUNT,
                    observed=run.sample_count,
                )
            )
        if run.new_sample_count != EXPECTED_SAMPLE_COUNT:
            blockers.append(
                _blocker(
                    "run_contract.new_sample_count_mismatch",
                    "every arm must process 4000 new samples in a fresh process",
                    run=run.run_dir,
                    expected=EXPECTED_SAMPLE_COUNT,
                    observed=run.new_sample_count,
                )
            )
        if run.resumed_from_step != 0:
            blockers.append(
                _blocker(
                    "run_contract.resumed_from_step_nonzero",
                    "every reproducibility arm must start from optimizer step zero",
                    run=run.run_dir,
                    expected=0,
                    observed=run.resumed_from_step,
                )
            )
        if not run.resume_checkpoint_recorded or run.resume_checkpoint is not None:
            blockers.append(
                _blocker(
                    "run_contract.resume_checkpoint_present",
                    "fresh reproducibility arms must explicitly record no resume checkpoint",
                    run=run.run_dir,
                    recorded=run.resume_checkpoint_recorded,
                    observed=run.resume_checkpoint,
                )
            )
        if run.nan_loss_count != 0:
            blockers.append(
                _blocker(
                    "run_contract.nan_loss_detected",
                    "every reproducibility arm must record zero non-finite losses",
                    run=run.run_dir,
                    expected=0,
                    observed=run.nan_loss_count,
                )
            )
        if run.collapse_alert_count != 0:
            blockers.append(
                _blocker(
                    "run_contract.collapse_alert_detected",
                    "every reproducibility arm must record zero collapse alerts",
                    run=run.run_dir,
                    expected=0,
                    observed=run.collapse_alert_count,
                )
            )
        if run.elapsed_seconds is None:
            blockers.append(
                _blocker(
                    "run_contract.elapsed_seconds_missing",
                    "every reproducibility arm must record a positive elapsed_seconds value",
                    run=run.run_dir,
                    observed=run.elapsed_seconds,
                )
            )
        elif not _throughput_metric_is_consistent(run):
            blockers.append(
                _blocker(
                    "run_contract.throughput_metric_inconsistent",
                    "reported samples_per_second must equal new_sample_count / elapsed_seconds",
                    run=run.run_dir,
                    reported=run.samples_per_second,
                    recomputed=_recomputed_samples_per_second(run),
                    new_sample_count=run.new_sample_count,
                    elapsed_seconds=run.elapsed_seconds,
                )
            )
        accelerator = run.runtime_identity.get("accelerator")
        accelerator_ready = (
            isinstance(accelerator, dict)
            and accelerator.get("requested_device") == "cuda"
            and accelerator.get("required") is True
            and accelerator.get("available") is True
            and accelerator.get("device_count") == 1
        )
        if (
            not accelerator_ready
            or run.accelerator_name is None
            or "H200" not in run.accelerator_name.upper()
        ):
            blockers.append(
                _blocker(
                    "run_contract.not_h200",
                    "every reproducibility arm must run on an NVIDIA H200",
                    run=run.run_dir,
                    observed=run.accelerator_name,
                )
            )
        if run.runtime_identity != expected_runtime:
            blockers.append(
                _blocker(
                    "run_contract.runtime_identity_mismatch",
                    "all four arms must use identical hardware and software runtime identities",
                    run=run.run_dir,
                    expected=expected_runtime,
                    observed=run.runtime_identity,
                )
            )
        if run.label.startswith("baseline_"):
            if run.training_config.get("deterministic") is not False:
                blockers.append(
                    _blocker(
                        "run_contract.baseline_config_not_nondeterministic",
                        "baseline training config must set deterministic=false",
                        run=run.run_dir,
                        observed=run.training_config.get("deterministic"),
                    )
                )
            if run.determinism_payload.get("deterministic") is not False:
                blockers.append(
                    _blocker(
                        "run_contract.baseline_deterministic_flag_enabled",
                        "baseline runtime must record deterministic=false",
                        run=run.run_dir,
                        observed=run.determinism_payload.get("deterministic"),
                    )
                )
            if run.determinism_payload.get("torch_deterministic_algorithms") is not False:
                blockers.append(
                    _blocker(
                        "run_contract.baseline_algorithms_enabled",
                        "baseline runtime must disable torch deterministic algorithms",
                        run=run.run_dir,
                        observed=run.determinism_payload.get("torch_deterministic_algorithms"),
                    )
                )
            if (
                "cublas_workspace_config" not in run.determinism_payload
                or run.determinism_payload.get("cublas_workspace_config") is not None
            ):
                blockers.append(
                    _blocker(
                        "run_contract.baseline_cublas_workspace_present",
                        "baseline process must start without CUBLAS_WORKSPACE_CONFIG",
                        run=run.run_dir,
                        observed=run.determinism_payload.get("cublas_workspace_config"),
                    )
                )
        else:
            if run.training_config.get("deterministic") is not True:
                blockers.append(
                    _blocker(
                        "run_contract.deterministic_config_disabled",
                        "deterministic arm config must set deterministic=true",
                        run=run.run_dir,
                        observed=run.training_config.get("deterministic"),
                    )
                )
            if (
                run.determinism_payload.get("deterministic") is not True
                or run.determinism_payload.get("torch_deterministic_algorithms") is not True
            ):
                blockers.append(
                    _blocker(
                        "run_contract.deterministic_algorithms_disabled",
                        "deterministic arms must enable torch deterministic algorithms",
                        run=run.run_dir,
                        observed=run.determinism_payload,
                    )
                )
            if run.determinism_payload.get("cublas_workspace_config") != ":4096:8":
                blockers.append(
                    _blocker(
                        "run_contract.deterministic_cublas_workspace_mismatch",
                        "deterministic arms must pin CUBLAS_WORKSPACE_CONFIG=:4096:8",
                        run=run.run_dir,
                        expected=":4096:8",
                        observed=run.determinism_payload.get("cublas_workspace_config"),
                    )
                )
    return RunContractValidation(
        ok=not blockers,
        expected_steps=EXPECTED_STEPS,
        expected_sample_count=EXPECTED_SAMPLE_COUNT,
        blockers=tuple(blockers),
    )


def compare_deterministic_pair(
    run_a: TrainingRunEvidence,
    run_b: TrainingRunEvidence,
) -> DeterministicPairComparison:
    """Compare two deterministic runs for bit-exact checkpoint evidence."""
    blockers: list[ReproducibilityBlocker] = []
    matched_fields: list[str] = []
    for field in _DETERMINISTIC_PAIR_FIELDS:
        left = getattr(run_a, field)
        right = getattr(run_b, field)
        if left == right:
            matched_fields.append(field)
        else:
            blockers.append(
                _blocker(
                    "deterministic_pair.field_mismatch",
                    "deterministic run identity fields must match",
                    field=field,
                    run_a=left,
                    run_b=right,
                )
            )
    if not run_a.deterministic_enabled:
        blockers.append(
            _blocker(
                "deterministic_pair.run_a_not_deterministic",
                "first deterministic run does not record deterministic torch controls",
                run=run_a.run_dir,
            )
        )
    if not run_b.deterministic_enabled:
        blockers.append(
            _blocker(
                "deterministic_pair.run_b_not_deterministic",
                "second deterministic run does not record deterministic torch controls",
                run=run_b.run_dir,
            )
        )

    matched_artifacts = _compare_pair_artifacts(run_a, run_b, blockers)
    return DeterministicPairComparison(
        ok=not blockers,
        matched_fields=tuple(matched_fields),
        matched_artifacts=tuple(matched_artifacts),
        blockers=tuple(blockers),
    )


def compare_throughput(
    baseline_a: TrainingRunEvidence | None,
    run_a: TrainingRunEvidence,
    run_b: TrainingRunEvidence,
    baseline_b: TrainingRunEvidence | None,
    *,
    max_drop_fraction: float,
    max_repeat_spread_fraction: float = 0.05,
) -> ThroughputComparison:
    """Compare counterbalanced baseline and deterministic repeat pairs."""
    _validate_fraction_threshold(max_drop_fraction, field="max_drop_fraction")
    _validate_fraction_threshold(
        max_repeat_spread_fraction,
        field="max_repeat_spread_fraction",
    )
    blockers: list[ReproducibilityBlocker] = []
    missing_baselines = [
        label
        for label, run in (("baseline_a", baseline_a), ("baseline_b", baseline_b))
        if run is None
    ]
    if missing_baselines:
        blockers.append(
            _blocker(
                "throughput.missing_baseline_runs",
                "counterbalanced throughput evidence requires two baseline run directories",
                missing=missing_baselines,
            )
        )
        return ThroughputComparison(
            ok=False,
            status="invalid",
            max_drop_fraction=max_drop_fraction,
            max_repeat_spread_fraction=max_repeat_spread_fraction,
            baseline_samples_per_second={},
            baseline_max_samples_per_second=None,
            baseline_spread_fraction=None,
            deterministic_samples_per_second={},
            deterministic_min_samples_per_second=None,
            deterministic_spread_fraction=None,
            drop_fraction=None,
            threshold_evaluated=False,
            blockers=tuple(blockers),
        )

    assert baseline_a is not None
    assert baseline_b is not None
    baselines = (baseline_a, baseline_b)
    deterministic_runs = (run_a, run_b)
    all_runs = (baseline_a, run_a, run_b, baseline_b)

    blockers.extend(
        _blocker(
            "throughput.baseline_is_deterministic",
            "baseline runs must not enable deterministic torch controls",
            run=baseline.run_dir,
        )
        for baseline in baselines
        if baseline.deterministic_enabled
    )
    blockers.extend(
        _blocker(
            "throughput.deterministic_run_not_deterministic",
            "deterministic throughput runs must record deterministic torch controls",
            run=run.run_dir,
        )
        for run in deterministic_runs
        if not run.deterministic_enabled
    )

    expected_dataset_manifest = _single_artifact(run_a, "dataset_manifest")
    for observed_run in all_runs:
        observed_dataset_manifest = _single_artifact(observed_run, "dataset_manifest")
        if not _artifacts_match(expected_dataset_manifest, observed_dataset_manifest):
            blockers.append(
                _blocker(
                    "throughput.dataset_manifest_mismatch",
                    "all four throughput arms must use byte-identical dataset manifests",
                    expected=None
                    if expected_dataset_manifest is None
                    else expected_dataset_manifest.to_dict(),
                    observed=None
                    if observed_dataset_manifest is None
                    else observed_dataset_manifest.to_dict(),
                    run=observed_run.run_dir,
                )
            )

    expected_baseline_config = dict(run_a.training_config)
    expected_baseline_config["deterministic"] = False
    blockers.extend(
        _blocker(
            "throughput.baseline_config_mismatch",
            "baseline configs may differ from deterministic config only by deterministic=false",
            expected=expected_baseline_config,
            observed=baseline.training_config,
            run=baseline.run_dir,
        )
        for baseline in baselines
        if baseline.training_config != expected_baseline_config
    )

    baseline_config_a = _single_artifact(baseline_a, "training_config")
    baseline_config_b = _single_artifact(baseline_b, "training_config")
    if not _artifacts_match(baseline_config_a, baseline_config_b):
        blockers.append(
            _blocker(
                "throughput.baseline_config_artifact_mismatch",
                "baseline repeats must use byte-identical training config artifacts",
                baseline_a=None if baseline_config_a is None else baseline_config_a.to_dict(),
                baseline_b=None if baseline_config_b is None else baseline_config_b.to_dict(),
            )
        )

    for field in _THROUGHPUT_IDENTITY_FIELDS:
        expected = getattr(run_a, field)
        for observed in all_runs:
            value = getattr(observed, field)
            if value != expected:
                blockers.append(
                    _blocker(
                        "throughput.identity_mismatch",
                        "all four runs must use the same run, commit, package, dataset, and seeds",
                        field=field,
                        expected=expected,
                        observed=value,
                        run=observed.run_dir,
                    )
                )

    expected_sample_count = run_a.sample_count
    for run in all_runs:
        if run.sample_count is None:
            blockers.append(
                _blocker(
                    "throughput.missing_sample_count",
                    "every throughput arm must record a positive sample_count",
                    run=run.run_dir,
                )
            )
        elif expected_sample_count is not None and run.sample_count != expected_sample_count:
            blockers.append(
                _blocker(
                    "throughput.sample_count_mismatch",
                    "all four throughput arms must measure the same sample count",
                    expected=expected_sample_count,
                    observed=run.sample_count,
                    run=run.run_dir,
                )
            )

    baseline_rates = {
        run.label: run.samples_per_second for run in baselines if run.samples_per_second is not None
    }
    deterministic_rates = {
        run.label: run.samples_per_second
        for run in deterministic_runs
        if run.samples_per_second is not None
    }
    blockers.extend(
        _blocker(
            "throughput.missing_baseline_rate",
            "both baseline metrics must record samples_per_second and elapsed_seconds",
            run=run.run_dir,
        )
        for run in baselines
        if run.samples_per_second is None
    )
    blockers.extend(
        _blocker(
            "throughput.missing_deterministic_rate",
            "both deterministic metrics must record samples_per_second and elapsed_seconds",
            run=run.run_dir,
        )
        for run in deterministic_runs
        if run.samples_per_second is None
    )

    baseline_min = min(baseline_rates.values()) if len(baseline_rates) == 2 else None
    baseline_max = max(baseline_rates.values()) if len(baseline_rates) == 2 else None
    baseline_spread = (
        None
        if baseline_min is None or baseline_max is None
        else (baseline_max - baseline_min) / baseline_max
    )
    deterministic_min = min(deterministic_rates.values()) if len(deterministic_rates) == 2 else None
    deterministic_max = max(deterministic_rates.values()) if len(deterministic_rates) == 2 else None
    deterministic_spread = (
        None
        if deterministic_min is None or deterministic_max is None
        else (deterministic_max - deterministic_min) / deterministic_max
    )
    drop_fraction = (
        None
        if baseline_max is None or deterministic_min is None
        else max(0.0, (baseline_max - deterministic_min) / baseline_max)
    )
    validation_failed = bool(blockers)
    baseline_spread_inconclusive = (
        baseline_spread is not None and baseline_spread > max_repeat_spread_fraction
    )
    deterministic_spread_inconclusive = (
        deterministic_spread is not None and deterministic_spread > max_repeat_spread_fraction
    )
    if baseline_spread_inconclusive:
        blockers.append(
            _blocker(
                "throughput.baseline_repeat_spread_inconclusive",
                "baseline repeat spread exceeds the predeclared noise guard",
                baseline_spread_fraction=baseline_spread,
                max_repeat_spread_fraction=max_repeat_spread_fraction,
            )
        )
    if deterministic_spread_inconclusive:
        blockers.append(
            _blocker(
                "throughput.deterministic_repeat_spread_inconclusive",
                "deterministic repeat spread exceeds the predeclared noise guard",
                deterministic_spread_fraction=deterministic_spread,
                max_repeat_spread_fraction=max_repeat_spread_fraction,
            )
        )
    repeats_inconclusive = baseline_spread_inconclusive or deterministic_spread_inconclusive
    threshold_evaluated = (
        drop_fraction is not None and not validation_failed and not repeats_inconclusive
    )
    if threshold_evaluated and drop_fraction is not None and drop_fraction > max_drop_fraction:
        blockers.append(
            _blocker(
                "throughput.drop_exceeds_threshold",
                "deterministic throughput drop exceeds the configured threshold",
                baseline_max_samples_per_second=baseline_max,
                deterministic_min_samples_per_second=deterministic_min,
                drop_fraction=drop_fraction,
                max_drop_fraction=max_drop_fraction,
            )
        )

    if validation_failed:
        status = "invalid"
    elif repeats_inconclusive:
        status = "inconclusive"
    elif not blockers:
        status = "pass"
    elif any(blocker.code == "throughput.drop_exceeds_threshold" for blocker in blockers):
        status = "fail"
    else:
        status = "invalid"

    return ThroughputComparison(
        ok=status == "pass",
        status=status,
        max_drop_fraction=max_drop_fraction,
        max_repeat_spread_fraction=max_repeat_spread_fraction,
        baseline_samples_per_second=dict(baseline_rates),
        baseline_max_samples_per_second=baseline_max,
        baseline_spread_fraction=baseline_spread,
        deterministic_samples_per_second=dict(deterministic_rates),
        deterministic_min_samples_per_second=deterministic_min,
        deterministic_spread_fraction=deterministic_spread,
        drop_fraction=drop_fraction,
        threshold_evaluated=threshold_evaluated,
        blockers=tuple(blockers),
    )


def write_training_reproducibility_report(
    *,
    deterministic_run_a: Path,
    deterministic_run_b: Path,
    output: Path,
    baseline_run_a: Path | None = None,
    baseline_run_b: Path | None = None,
    max_throughput_drop_fraction: float = 0.15,
    max_repeat_spread_fraction: float = 0.05,
    require_preflight: bool = False,
) -> TrainingReproducibilityReport:
    """Build and write a normalized training reproducibility report."""
    report = build_training_reproducibility_report(
        deterministic_run_a=deterministic_run_a,
        deterministic_run_b=deterministic_run_b,
        baseline_run_a=baseline_run_a,
        baseline_run_b=baseline_run_b,
        max_throughput_drop_fraction=max_throughput_drop_fraction,
        max_repeat_spread_fraction=max_repeat_spread_fraction,
        require_preflight=require_preflight,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(report.to_dict(), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return report


def main(argv: list[str] | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    try:
        report = write_training_reproducibility_report(
            deterministic_run_a=args.deterministic_run_a,
            deterministic_run_b=args.deterministic_run_b,
            baseline_run_a=args.baseline_run_a,
            baseline_run_b=args.baseline_run_b,
            output=args.output,
            max_throughput_drop_fraction=args.max_throughput_drop,
            max_repeat_spread_fraction=args.max_repeat_spread,
            require_preflight=args.require_preflight,
        )
    except GenoLeWMError as exc:
        sys.stderr.write(f"error: {exc}\n")
        return exit_code_for(exc)
    sys.stdout.write(json.dumps(report.to_dict(), indent=2, sort_keys=True) + "\n")
    return 0 if report.ok else 2


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Compare completed training-run archives for deterministic rerun and "
            "deterministic-throughput evidence."
        ),
    )
    parser.add_argument("--deterministic-run-a", type=Path, required=True)
    parser.add_argument("--deterministic-run-b", type=Path, required=True)
    parser.add_argument("--baseline-run-a", type=Path, required=True)
    parser.add_argument("--baseline-run-b", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-throughput-drop", type=float, default=0.15)
    parser.add_argument("--max-repeat-spread", type=float, default=0.05)
    parser.add_argument(
        "--require-preflight",
        action="store_true",
        help="Require a validated training_preflight_report.json artifact in each run.",
    )
    return parser


def _compare_pair_artifacts(
    run_a: TrainingRunEvidence,
    run_b: TrainingRunEvidence,
    blockers: list[ReproducibilityBlocker],
) -> list[str]:
    left = {
        key: artifact
        for key, artifact in run_a.artifact_map().items()
        if artifact.kind in _REQUIRED_PAIR_ARTIFACT_KINDS
    }
    right = {
        key: artifact
        for key, artifact in run_b.artifact_map().items()
        if artifact.kind in _REQUIRED_PAIR_ARTIFACT_KINDS
    }
    matched: list[str] = []
    missing_left = sorted(set(right) - set(left))
    missing_right = sorted(set(left) - set(right))
    if missing_left or missing_right:
        blockers.append(
            _blocker(
                "deterministic_pair.artifact_set_mismatch",
                "deterministic runs must expose matching dataset, config, and checkpoint artifacts",
                missing_from_run_a=missing_left,
                missing_from_run_b=missing_right,
            )
        )
    observed_kinds = {artifact.kind for artifact in (*left.values(), *right.values())}
    missing_kinds = sorted(_REQUIRED_PAIR_ARTIFACT_KINDS - observed_kinds)
    if missing_kinds:
        blockers.append(
            _blocker(
                "deterministic_pair.missing_required_artifact_kind",
                "deterministic comparison requires dataset, config, and checkpoint artifacts",
                missing_kinds=missing_kinds,
            )
        )
    for key in sorted(set(left) & set(right)):
        left_artifact = left[key]
        right_artifact = right[key]
        if (
            left_artifact.sha256 == right_artifact.sha256
            and left_artifact.size_bytes == right_artifact.size_bytes
        ):
            matched.append(key)
            continue
        blockers.append(
            _blocker(
                "deterministic_pair.artifact_mismatch",
                "deterministic reruns must produce bit-exact config, dataset, and checkpoint artifacts",
                artifact=key,
                run_a_sha256=left_artifact.sha256,
                run_b_sha256=right_artifact.sha256,
                run_a_size_bytes=left_artifact.size_bytes,
                run_b_size_bytes=right_artifact.size_bytes,
            )
        )
    return matched


def _load_metrics_payload(run_dir: Path, manifest: TrainingRunManifest) -> dict[str, Any]:
    metrics_artifacts = tuple(
        artifact for artifact in manifest.artifacts if artifact.kind == "metrics"
    )
    if len(metrics_artifacts) != 1:
        raise InputError(
            "training-run manifest must contain exactly one metrics artifact",
            details={"observed": len(metrics_artifacts)},
        )
    path = run_dir / metrics_artifacts[0].path
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise InputError("failed to read training metrics", details={"path": str(path)}) from exc
    except json.JSONDecodeError as exc:
        raise InputError(
            "training metrics JSON is invalid",
            details={"path": str(path), "line": exc.lineno, "column": exc.colno},
        ) from exc
    if not isinstance(payload, dict):
        raise InputError("training metrics must be a JSON object", details={"path": str(path)})
    return payload


def _load_training_config_payload(
    run_dir: Path,
    manifest: TrainingRunManifest,
) -> dict[str, Any]:
    config_artifacts = tuple(
        artifact for artifact in manifest.artifacts if artifact.kind == "training_config"
    )
    if len(config_artifacts) != 1:
        raise InputError(
            "training-run manifest must contain exactly one training config artifact",
            details={"observed": len(config_artifacts)},
        )
    path = run_dir / config_artifacts[0].path
    try:
        payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, yaml.YAMLError) as exc:
        raise InputError(
            "training config YAML is invalid",
            details={"path": str(path)},
        ) from exc
    if not isinstance(payload, dict) or any(not isinstance(key, str) for key in payload):
        raise InputError(
            "training config must be a string-keyed YAML object",
            details={"path": str(path)},
        )
    return payload


def _load_training_preflight_payload(
    run_dir: Path,
    manifest: TrainingRunManifest,
) -> dict[str, Any]:
    artifacts = tuple(
        artifact for artifact in manifest.artifacts if artifact.kind == "training_preflight"
    )
    if len(artifacts) != 1:
        return {}
    path = run_dir / artifacts[0].path
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise InputError(
            "training preflight report JSON is invalid",
            details={"path": str(path)},
        ) from exc
    if not isinstance(payload, dict):
        raise InputError(
            "training preflight report must be a JSON object",
            details={"path": str(path)},
        )
    return payload


def _runtime_identity(
    manifest: TrainingRunManifest,
    preflight: dict[str, Any],
) -> dict[str, Any]:
    accelerator = preflight.get("accelerator")
    accelerator_identity = (
        {
            key: accelerator.get(key)
            for key in (
                "requested_device",
                "required",
                "available",
                "device_count",
                "device_name",
                "total_memory_bytes",
            )
        }
        if isinstance(accelerator, dict)
        else None
    )
    raw_dependencies = preflight.get("dependencies")
    dependencies = []
    if isinstance(raw_dependencies, list):
        for dependency in raw_dependencies:
            if not isinstance(dependency, dict):
                continue
            dependencies.append(
                {
                    key: dependency.get(key)
                    for key in ("import_name", "package", "required", "available", "version")
                }
            )
    dependencies.sort(key=lambda item: str(item.get("import_name")))
    carbon = preflight.get("carbon")
    carbon_identity = (
        {key: carbon.get(key) for key in ("local_files_only", "artifacts")}
        if isinstance(carbon, dict)
        else None
    )
    return {
        "hardware": list(manifest.hardware),
        "runtime": list(manifest.runtime),
        "accelerator": accelerator_identity,
        "dependencies": dependencies,
        "carbon": carbon_identity,
    }


def _artifact_identities(manifest: TrainingRunManifest) -> tuple[ArtifactIdentity, ...]:
    artifacts = [
        ArtifactIdentity(
            key=f"{artifact.kind}:{artifact.path}",
            kind=artifact.kind,
            path=artifact.path,
            sha256=artifact.sha256,
            size_bytes=artifact.size_bytes,
        )
        for artifact in manifest.artifacts
    ]
    return tuple(sorted(artifacts, key=lambda artifact: artifact.key))


def _single_artifact(run: TrainingRunEvidence, kind: str) -> ArtifactIdentity | None:
    matches = tuple(artifact for artifact in run.artifacts if artifact.kind == kind)
    return matches[0] if len(matches) == 1 else None


def _artifacts_match(
    left: ArtifactIdentity | None,
    right: ArtifactIdentity | None,
) -> bool:
    return (
        left is not None
        and right is not None
        and left.sha256 == right.sha256
        and left.size_bytes == right.size_bytes
    )


def _parse_determinism_payload(determinism: str) -> dict[str, Any]:
    try:
        payload = json.loads(determinism)
    except json.JSONDecodeError:
        return {}
    if not isinstance(payload, dict):
        return {}
    return payload


def _deterministic_enabled(payload: dict[str, Any]) -> bool:
    return (
        payload.get("deterministic") is True
        and payload.get("torch_deterministic_algorithms") is True
    )


def _extract_samples_per_second(payload: dict[str, Any]) -> float | None:
    for key in _THROUGHPUT_KEYS:
        value = _positive_float(_metric_value(payload, key))
        if value is not None:
            return value
    sample_count = _positive_int(_metric_value(payload, "new_sample_count"))
    if sample_count is None:
        sample_count = _extract_sample_count(payload)
    if sample_count is None:
        return None
    for key in _ELAPSED_KEYS:
        elapsed = _positive_float(_metric_value(payload, key))
        if elapsed is not None:
            return sample_count / elapsed
    return None


def _extract_elapsed_seconds(payload: dict[str, Any]) -> float | None:
    for key in _ELAPSED_KEYS:
        elapsed = _positive_float(_metric_value(payload, key))
        if elapsed is not None:
            return elapsed
    return None


def _recomputed_samples_per_second(run: TrainingRunEvidence) -> float | None:
    if run.new_sample_count is None or run.elapsed_seconds is None:
        return None
    return run.new_sample_count / run.elapsed_seconds


def _throughput_metric_is_consistent(run: TrainingRunEvidence) -> bool:
    recomputed = _recomputed_samples_per_second(run)
    return (
        recomputed is not None
        and run.samples_per_second is not None
        and math.isclose(run.samples_per_second, recomputed, rel_tol=1e-9, abs_tol=1e-12)
    )


def _extract_sample_count(payload: dict[str, Any]) -> int | None:
    value = _metric_value(payload, "sample_count")
    if isinstance(value, dict):
        value = value.get("value")
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        return None
    return value


def _metric_value(payload: dict[str, Any], key: str) -> Any:
    if key in payload:
        return payload[key]
    metrics = payload.get("metrics")
    if isinstance(metrics, dict):
        return metrics.get(key)
    return None


def _positive_float(value: Any) -> float | None:
    if isinstance(value, dict):
        value = value.get("value")
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    converted = float(value)
    if not math.isfinite(converted) or converted <= 0:
        return None
    return converted


def _positive_int(value: Any) -> int | None:
    if isinstance(value, dict):
        value = value.get("value")
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        return None
    return value


def _nonnegative_int(value: Any) -> int | None:
    if isinstance(value, dict):
        value = value.get("value")
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        return None
    return value


def _validate_fraction_threshold(value: float, *, field: str) -> None:
    if not math.isfinite(value) or value < 0 or value >= 1:
        raise InputError(
            f"{field} must be in [0, 1)",
            details={field: value},
        )


def _blocker(code: str, message: str, **details: object) -> ReproducibilityBlocker:
    return ReproducibilityBlocker(code=code, message=message, details=dict(details))


def _public_path(path: Path) -> str:
    if path.is_absolute():
        return path.name
    if ".." in path.parts or not path.parts:
        return path.name
    return path.as_posix()


def _utc_now() -> str:
    return datetime.now(tz=timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
