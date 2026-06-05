"""Tests for the real Carbon-backed training launcher boundary."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from geno_lewm.config import load_config
from geno_lewm.errors import InputError
from geno_lewm.training.real import (
    _collapse_var_min,
    _dataset_fallback_sources,
    _dataset_files,
    _load_dataset_manifest,
    _load_windows,
    _move_trainable_to_device,
    _nan_loss_count,
    _training_device,
    _validate_resume_checkpoint_payload,
    _write_metrics,
)
from geno_lewm.training.trainer import TorchTrainerStepResult, TrainerSeeds
from tests.unit.test_training_preflight import _write_release_dataset, _write_training_config


def _step_result(*, loss: float, var: float, step: int = 1) -> TorchTrainerStepResult:
    return TorchTrainerStepResult(
        step=step,
        lr_multiplier=1.0,
        loss=loss,
        pred_loss=loss,
        kl_reg=0.0,
        action_count=1,
        pred_var_per_dim=var,
    )


def test_nan_loss_count_counts_nonfinite_losses() -> None:
    results = [
        _step_result(loss=0.5, var=1.0),
        _step_result(loss=float("nan"), var=1.0),
        _step_result(loss=float("inf"), var=1.0),
        _step_result(loss=0.3, var=1.0),
    ]
    assert _nan_loss_count(results) == 2


def test_collapse_var_min_returns_minimum_and_handles_empty() -> None:
    results = [
        _step_result(loss=0.5, var=0.8),
        _step_result(loss=0.4, var=0.2),
        _step_result(loss=0.3, var=0.9),
    ]
    assert _collapse_var_min(results) == pytest.approx(0.2)
    assert _collapse_var_min([]) == 0.0


def test_write_metrics_emits_real_nan_and_collapse_floor(tmp_path: Path) -> None:
    config = load_config(_write_training_config(tmp_path))
    results = [
        _step_result(loss=0.5, var=0.8, step=1),
        TorchTrainerStepResult(
            step=2,
            lr_multiplier=1.0,
            loss=float("nan"),
            pred_loss=float("nan"),
            kl_reg=0.0,
            action_count=1,
            pred_var_per_dim=0.2,
        ),
        _step_result(loss=0.3, var=0.9, step=3),
    ]
    path = tmp_path / "metrics.json"
    _write_metrics(
        path,
        config=config,
        steps=3,
        resumed_from_step=0,
        sample_count=24,
        final_loss=0.3,
        step_results=results,
        collapse_alert_count=1,
        dataset_snapshot_id="geno-lewm-data-v0.1.0-r1",
        resume_checkpoint_path=None,
    )
    metrics = json.loads(path.read_text(encoding="utf-8"))["metrics"]
    assert metrics["nan_loss_count"] == 1
    assert metrics["collapse_var_min"]["value"] == pytest.approx(0.2)
    assert metrics["collapse_alert_count"] == 1


def test_training_device_uses_runtime_config_device(tmp_path: Path) -> None:
    config = load_config(_write_training_config(tmp_path))

    assert _training_device(config) == "cuda"


def test_release_training_loader_prefers_placed_windows(tmp_path: Path) -> None:
    pytest.importorskip("pyarrow")
    dataset_dir = _write_release_dataset(tmp_path)
    files = _dataset_files(_load_dataset_manifest(dataset_dir))

    windows = tuple(_load_windows(dataset_dir, files))

    assert len(windows) == 1
    assert windows[0].chrom == "1"
    assert windows[0].source == "gnomad_common"
    assert _dataset_fallback_sources(windows) == {"clinvar": "synthetic_snv"}


def test_move_trainable_to_device_invokes_module_to_for_accelerator() -> None:
    module = _DeviceModule()

    moved = _move_trainable_to_device(module, "cuda", label="predictor")

    assert moved is module
    assert module.devices == ["cuda"]


def test_move_trainable_to_device_leaves_cpu_module_in_place() -> None:
    module = _DeviceModule()

    moved = _move_trainable_to_device(module, "cpu", label="predictor")

    assert moved is module
    assert module.devices == []


def test_validate_resume_checkpoint_accepts_matching_identity(tmp_path: Path) -> None:
    config = load_config(_write_training_config(tmp_path))
    seeds = TrainerSeeds.from_base_seed(config.seed)

    checkpoint = _validate_resume_checkpoint_payload(
        _resume_payload(config, seeds, steps_completed=3),
        path=tmp_path / "predictor_checkpoint.pt",
        config=config,
        dataset_snapshot_id="geno-lewm-data-v0.1.0-r1",
        seeds=seeds,
        target_steps=5,
    )

    assert checkpoint.steps_completed == 3


def test_validate_resume_checkpoint_rejects_dataset_mismatch(tmp_path: Path) -> None:
    config = load_config(_write_training_config(tmp_path))
    seeds = TrainerSeeds.from_base_seed(config.seed)
    payload = _resume_payload(config, seeds, steps_completed=3)
    payload["dataset_snapshot_id"] = "geno-lewm-data-old"

    with pytest.raises(InputError, match="dataset_snapshot_id"):
        _validate_resume_checkpoint_payload(
            payload,
            path=tmp_path / "predictor_checkpoint.pt",
            config=config,
            dataset_snapshot_id="geno-lewm-data-v0.1.0-r1",
            seeds=seeds,
            target_steps=5,
        )


def test_validate_resume_checkpoint_rejects_config_mismatch(tmp_path: Path) -> None:
    config = load_config(_write_training_config(tmp_path))
    seeds = TrainerSeeds.from_base_seed(config.seed)
    payload = _resume_payload(config, seeds, steps_completed=3)
    payload["config"]["data.batch_size"] = config.data.batch_size + 1

    with pytest.raises(InputError, match="config does not match"):
        _validate_resume_checkpoint_payload(
            payload,
            path=tmp_path / "predictor_checkpoint.pt",
            config=config,
            dataset_snapshot_id="geno-lewm-data-v0.1.0-r1",
            seeds=seeds,
            target_steps=5,
        )


def test_validate_resume_checkpoint_rejects_finished_checkpoint(tmp_path: Path) -> None:
    config = load_config(_write_training_config(tmp_path))
    seeds = TrainerSeeds.from_base_seed(config.seed)

    with pytest.raises(InputError, match="already at or beyond"):
        _validate_resume_checkpoint_payload(
            _resume_payload(config, seeds, steps_completed=5),
            path=tmp_path / "predictor_checkpoint.pt",
            config=config,
            dataset_snapshot_id="geno-lewm-data-v0.1.0-r1",
            seeds=seeds,
            target_steps=5,
        )


def _resume_payload(config, seeds: TrainerSeeds, *, steps_completed: int) -> dict[str, object]:
    return {
        "schema_version": "1.0.0",
        "run_id": config.run_id,
        "dataset_snapshot_id": "geno-lewm-data-v0.1.0-r1",
        "steps_completed": steps_completed,
        "seeds": seeds.to_dict(),
        "config": {
            "run_id": config.run_id,
            "seed": config.seed,
            "deterministic": config.deterministic,
            "data.batch_size": config.data.batch_size,
            "predictor.d_state": config.predictor.d_state,
            "action.d_action": config.action.d_action,
        },
        "predictor": {},
        "action_encoder": {},
        "optimizer": {},
    }


class _DeviceModule:
    def __init__(self) -> None:
        self.devices: list[str] = []

    def to(self, device: str) -> _DeviceModule:
        self.devices.append(device)
        return self
