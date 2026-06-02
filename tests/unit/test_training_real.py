"""Tests for the real Carbon-backed training launcher boundary."""

from __future__ import annotations

from pathlib import Path

import pytest

from geno_lewm.config import load_config
from geno_lewm.errors import InputError
from geno_lewm.training.real import _validate_resume_checkpoint_payload
from geno_lewm.training.trainer import TrainerSeeds
from tests.unit.test_training_preflight import _write_training_config


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
