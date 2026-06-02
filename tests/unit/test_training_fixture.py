"""Tests for deterministic fixture smoke training."""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path

import pytest

from geno_lewm.config import load_default
from geno_lewm.errors import InputError
from geno_lewm.training import run_fixture_training
from tools.release.training_run import build_training_run_package, verify_training_run_manifest


def test_fixture_training_is_resume_equivalent(tmp_path: Path) -> None:
    cfg = dataclasses.replace(
        load_default("train"), run_id="fixture-run", seed=7, deterministic=True
    )

    full = run_fixture_training(
        config=cfg,
        run_dir=tmp_path / "full",
        steps=50,
        command="geno-lewm-train --fixture-smoke --steps 50",
        commit_sha="abcdef123456",
        package_version="0.1.0.dev0",
    )
    first_half = run_fixture_training(
        config=cfg,
        run_dir=tmp_path / "resume",
        steps=25,
        command="geno-lewm-train --fixture-smoke --steps 25",
        commit_sha="abcdef123456",
        package_version="0.1.0.dev0",
    )
    resumed = run_fixture_training(
        config=cfg,
        run_dir=tmp_path / "resume",
        steps=50,
        resume_from=first_half.checkpoint_path,
        command="geno-lewm-train --fixture-smoke --resume-from fixture_predictor_checkpoint.json",
        commit_sha="abcdef123456",
        package_version="0.1.0.dev0",
    )

    assert resumed.steps_completed == full.steps_completed == 50
    assert resumed.resumed_from_step == 25
    assert resumed.final_loss == pytest.approx(full.final_loss)
    assert _checkpoint(resumed.checkpoint_path)["weight"] == pytest.approx(
        _checkpoint(full.checkpoint_path)["weight"]
    )


def test_fixture_training_writes_training_run_package_inputs(tmp_path: Path) -> None:
    cfg = dataclasses.replace(
        load_default("train"), run_id="fixture-run", seed=3, deterministic=True
    )
    report = run_fixture_training(
        config=cfg,
        run_dir=tmp_path,
        steps=5,
        command="geno-lewm-train --fixture-smoke --steps 5",
        commit_sha="abcdef123456",
        package_version="0.1.0.dev0",
    )

    package = build_training_run_package(tmp_path, report.training_metadata_path)
    manifest = verify_training_run_manifest(tmp_path)

    assert package.run_id == "fixture-run"
    assert manifest.run_id == "fixture-run"
    assert manifest.dataset_snapshot_id == "geno-lewm-fixture-snv-smoke-v1"
    assert {artifact.kind for artifact in manifest.artifacts} >= {
        "checkpoint",
        "dataset_manifest",
        "log",
        "metrics",
        "training_config",
    }


def test_fixture_training_rejects_seed_mismatch_on_resume(tmp_path: Path) -> None:
    cfg = dataclasses.replace(load_default("train"), run_id="fixture-run", seed=1)
    first = run_fixture_training(
        config=cfg,
        run_dir=tmp_path,
        steps=1,
        command="geno-lewm-train --fixture-smoke --steps 1",
        commit_sha="abcdef123456",
        package_version="0.1.0.dev0",
    )
    changed = dataclasses.replace(cfg, seed=2)

    with pytest.raises(InputError, match="seed does not match"):
        run_fixture_training(
            config=changed,
            run_dir=tmp_path,
            steps=2,
            resume_from=first.checkpoint_path,
            command="geno-lewm-train --fixture-smoke --resume-from checkpoint",
            commit_sha="abcdef123456",
            package_version="0.1.0.dev0",
        )


def _checkpoint(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))
