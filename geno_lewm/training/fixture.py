# SPDX-License-Identifier: Apache-2.0
"""Deterministic fixture-tier trainer for the ``geno-lewm-train`` smoke path.

This module is intentionally not the Carbon-backed trainer. It gives the
release workflow a clean-machine, dependency-light training command that
exercises config resolution, checkpointing, deterministic resume, metrics,
logs, and training-run metadata before the heavy ML stack is available.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from geno_lewm.config import GenoLeWMConfig, write_resolved_config
from geno_lewm.errors import InputError

__all__ = [
    "FIXTURE_CHECKPOINT_NAME",
    "FIXTURE_DATASET_MANIFEST_NAME",
    "FIXTURE_LOG_NAME",
    "FIXTURE_METRICS_NAME",
    "FIXTURE_TRAINING_METADATA_NAME",
    "FixtureTrainingReport",
    "run_fixture_training",
]

FIXTURE_CHECKPOINT_NAME = "fixture_predictor_checkpoint.json"
FIXTURE_DATASET_MANIFEST_NAME = "dataset_manifest.json"
FIXTURE_LOG_NAME = "train.log"
FIXTURE_METRICS_NAME = "metrics.json"
FIXTURE_TRAINING_METADATA_NAME = "training_run.json"
_SCHEMA_VERSION = "1.0.0"
_TRAINING_RUN_PACKAGE_GENERATED_BY = "tools.release.training_run"


@dataclass(frozen=True, slots=True)
class FixtureTrainingReport:
    """Summary returned by the deterministic fixture trainer."""

    run_id: str
    run_dir: Path
    steps_requested: int
    steps_completed: int
    resumed_from_step: int
    final_loss: float
    checkpoint_path: Path
    metrics_path: Path
    log_path: Path
    config_path: Path
    dataset_manifest_path: Path
    training_metadata_path: Path

    def to_dict(self) -> dict[str, object]:
        return {
            "run_id": self.run_id,
            "run_dir": str(self.run_dir),
            "steps_requested": self.steps_requested,
            "steps_completed": self.steps_completed,
            "resumed_from_step": self.resumed_from_step,
            "final_loss": self.final_loss,
            "checkpoint_path": str(self.checkpoint_path),
            "metrics_path": str(self.metrics_path),
            "log_path": str(self.log_path),
            "config_path": str(self.config_path),
            "dataset_manifest_path": str(self.dataset_manifest_path),
            "training_metadata_path": str(self.training_metadata_path),
        }


@dataclass(frozen=True, slots=True)
class _FixtureState:
    step: int
    seed: int
    weight: float
    target: float
    loss: float


def run_fixture_training(
    *,
    config: GenoLeWMConfig,
    run_dir: Path,
    steps: int = 50,
    resume_from: Path | None = None,
    command: str,
    commit_sha: str,
    package_version: str,
) -> FixtureTrainingReport:
    """Run a deterministic scalar smoke trainer and write release artifacts.

    ``steps`` is the target total step count. When ``resume_from`` is
    supplied, the checkpoint's current step must be lower than ``steps``;
    the resumed run continues with the same deterministic sample stream.
    """
    _require_positive_steps(steps)
    run_dir.mkdir(parents=True, exist_ok=True)
    config_path = write_resolved_config(config, run_dir / "config.resolved.yaml")
    dataset_manifest_path = _write_fixture_dataset_manifest(config, run_dir)

    state = (
        _load_checkpoint(resume_from, expected_seed=config.seed)
        if resume_from is not None
        else _initial_state(config.seed)
    )
    resumed_from_step = state.step
    if resumed_from_step >= steps:
        raise InputError(
            "fixture resume checkpoint is already at or beyond --steps",
            details={"checkpoint_step": resumed_from_step, "steps": steps},
        )

    log_path = run_dir / FIXTURE_LOG_NAME
    mode = "a" if resumed_from_step else "w"
    with log_path.open(mode, encoding="utf-8") as log:
        if resumed_from_step:
            log.write(
                f"resume_from_step={resumed_from_step} target_steps={steps} seed={config.seed}\n"
            )
        for step in range(resumed_from_step + 1, steps + 1):
            state = _train_step(state, step=step)
            log.write(
                f"step={step} loss={state.loss:.12f} weight={state.weight:.12f} "
                "collapse_var_min=1.0 nan_loss=false\n"
            )

    checkpoint_path = run_dir / FIXTURE_CHECKPOINT_NAME
    _write_checkpoint(config, state, checkpoint_path)
    metrics_path = run_dir / FIXTURE_METRICS_NAME
    _write_metrics(config, state, metrics_path, steps=steps, resumed_from_step=resumed_from_step)
    training_metadata_path = run_dir / FIXTURE_TRAINING_METADATA_NAME
    _write_training_metadata(
        config,
        training_metadata_path,
        command=command,
        commit_sha=commit_sha,
        package_version=package_version,
    )
    return FixtureTrainingReport(
        run_id=config.run_id,
        run_dir=run_dir,
        steps_requested=steps,
        steps_completed=state.step,
        resumed_from_step=resumed_from_step,
        final_loss=state.loss,
        checkpoint_path=checkpoint_path,
        metrics_path=metrics_path,
        log_path=log_path,
        config_path=config_path,
        dataset_manifest_path=dataset_manifest_path,
        training_metadata_path=training_metadata_path,
    )


def _train_step(state: _FixtureState, *, step: int) -> _FixtureState:
    x = _unit_float(state.seed, "x", step) * 2.0 - 1.0
    y = state.target * x
    prediction = state.weight * x
    error = prediction - y
    loss = error * error
    lr = 0.2 / math.sqrt(step + 3.0)
    weight = state.weight - (2.0 * error * x * lr)
    return _FixtureState(
        step=step,
        seed=state.seed,
        weight=weight,
        target=state.target,
        loss=loss,
    )


def _initial_state(seed: int) -> _FixtureState:
    weight = _unit_float(seed, "init", 0) - 0.5
    target = 0.25 + _unit_float(seed, "target", 0) * 0.5
    loss = (weight - target) * (weight - target)
    return _FixtureState(step=0, seed=seed, weight=weight, target=target, loss=loss)


def _write_fixture_dataset_manifest(config: GenoLeWMConfig, run_dir: Path) -> Path:
    path = run_dir / FIXTURE_DATASET_MANIFEST_NAME
    payload = {
        "schema_version": _SCHEMA_VERSION,
        "snapshot_id": "geno-lewm-fixture-snv-smoke-v1",
        "generated_by": "geno-lewm-train --fixture-smoke",
        "run_id": config.run_id,
        "sources": [
            {
                "name": "deterministic scalar SNV smoke stream",
                "revision": "fixture-v1",
                "license": "Apache-2.0",
                "notes": "CI fixture only; not Carbon, gnomAD, or ClinVar evidence.",
            }
        ],
        "splits": {"train": {"records": 1, "description": "deterministic scalar stream"}},
    }
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def _write_checkpoint(config: GenoLeWMConfig, state: _FixtureState, path: Path) -> None:
    payload = {
        "schema_version": _SCHEMA_VERSION,
        "model_kind": "fixture_scalar_predictor",
        "run_id": config.run_id,
        "step": state.step,
        "seed": state.seed,
        "deterministic": config.deterministic,
        "weight": state.weight,
        "target": state.target,
        "loss": state.loss,
        "warning": "Fixture smoke checkpoint; not a Carbon-backed GenoLeWM model.",
    }
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_metrics(
    config: GenoLeWMConfig,
    state: _FixtureState,
    path: Path,
    *,
    steps: int,
    resumed_from_step: int,
) -> None:
    payload = {
        "schema_version": _SCHEMA_VERSION,
        "run_id": config.run_id,
        "sample_count": steps,
        "resumed_from_step": resumed_from_step,
        "metrics": {
            "train_loss": state.loss,
            "fixture_weight": state.weight,
            "fixture_target": state.target,
            "collapse_var_min": {"value": 1.0},
            "nan_loss_count": 0,
        },
        "claim_boundary": "Fixture smoke metrics only; not paper or model-quality evidence.",
    }
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_training_metadata(
    config: GenoLeWMConfig,
    path: Path,
    *,
    command: str,
    commit_sha: str,
    package_version: str,
) -> None:
    payload = {
        "schema_version": _SCHEMA_VERSION,
        "run_id": config.run_id,
        "generated_by": _TRAINING_RUN_PACKAGE_GENERATED_BY,
        "generated_at": _utc_now(),
        "command": command,
        "commit_sha": commit_sha,
        "package_version": package_version,
        "dataset_snapshot_id": "geno-lewm-fixture-snv-smoke-v1",
        "dataset_manifest": FIXTURE_DATASET_MANIFEST_NAME,
        "training_config": "config.resolved.yaml",
        "metrics": FIXTURE_METRICS_NAME,
        "logs": [FIXTURE_LOG_NAME],
        "checkpoint_files": [FIXTURE_CHECKPOINT_NAME],
        "status": "completed",
        "hardware": ["Fixture smoke run; no accelerator required."],
        "runtime": ["Python stdlib fixture backend; Carbon and torch are not loaded."],
        "seeds": {"python": config.seed, "fixture_stream": config.seed},
        "determinism": (
            "The fixture stream is deterministic by construction; this does not establish "
            "Carbon-backed GPU determinism."
        ),
        "monitoring": {"collapse_monitoring": True, "nan_monitoring": True},
        "result_summary": (
            "Completed deterministic fixture smoke training archive for CLI and release "
            "plumbing validation."
        ),
        "limitations": [
            "Fixture-scale evidence only; paper claims require the real Carbon-backed run.",
            "The checkpoint is not a GenoLeWM model and must not be released as model evidence.",
        ],
    }
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _load_checkpoint(path: Path, *, expected_seed: int) -> _FixtureState:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise InputError("failed to read fixture checkpoint", details={"path": str(path)}) from exc
    except json.JSONDecodeError as exc:
        raise InputError(
            "fixture checkpoint JSON is invalid",
            details={"path": str(path), "line": exc.lineno, "column": exc.colno},
        ) from exc
    if not isinstance(payload, dict):
        raise InputError("fixture checkpoint must be a JSON object")
    if payload.get("model_kind") != "fixture_scalar_predictor":
        raise InputError("checkpoint is not a fixture smoke checkpoint")
    seed = _required_int(payload, "seed")
    if seed != expected_seed:
        raise InputError(
            "fixture checkpoint seed does not match resolved config",
            details={"checkpoint_seed": seed, "config_seed": expected_seed},
        )
    return _FixtureState(
        step=_required_int(payload, "step"),
        seed=seed,
        weight=_required_float(payload, "weight"),
        target=_required_float(payload, "target"),
        loss=_required_float(payload, "loss"),
    )


def _unit_float(seed: int, name: str, step: int) -> float:
    raw = f"{seed}:{name}:{step}".encode()
    digest = hashlib.sha256(raw).digest()
    value = int.from_bytes(digest[:8], "big")
    return value / float(2**64 - 1)


def _required_int(payload: dict[str, object], key: str) -> int:
    value = payload.get(key)
    if isinstance(value, bool) or not isinstance(value, int):
        raise InputError(f"fixture checkpoint {key} must be an integer")
    return value


def _required_float(payload: dict[str, object], key: str) -> float:
    value = payload.get(key)
    if isinstance(value, bool) or not isinstance(value, int | float) or not math.isfinite(value):
        raise InputError(f"fixture checkpoint {key} must be a finite number")
    return float(value)


def _require_positive_steps(steps: int) -> None:
    if isinstance(steps, bool) or not isinstance(steps, int) or steps <= 0:
        raise InputError("fixture training --steps must be a positive integer")


def _utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")
