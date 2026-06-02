# SPDX-License-Identifier: Apache-2.0
"""Fixture-backed ML smoke tests for hosted CI.

These tests are intentionally not paper evidence. They exercise the
small public fixture path that should fail quickly when training
artifacts become non-finite, collapse diagnostics disappear, or
deterministic resume stops reproducing a full uninterrupted run.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import math
from pathlib import Path

import pytest

from geno_lewm.config import load_default
from geno_lewm.training import CollapseMonitor, run_fixture_training


def test_fixture_smoke_loss_health_and_claim_boundary(tmp_path: Path) -> None:
    """Catch non-finite loss, missing collapse health, and paper-claim drift."""
    cfg = dataclasses.replace(
        load_default("train"),
        run_id="ml-smoke-fixture",
        seed=17,
        deterministic=True,
    )

    first = run_fixture_training(
        config=cfg,
        run_dir=tmp_path / "first-step",
        steps=1,
        command="geno-lewm-train --fixture-smoke --steps 1",
        commit_sha="abcdef123456",
        package_version="0.1.0.dev0",
    )
    final = run_fixture_training(
        config=cfg,
        run_dir=tmp_path / "final",
        steps=20,
        command="geno-lewm-train --fixture-smoke --steps 20",
        commit_sha="abcdef123456",
        package_version="0.1.0.dev0",
    )

    metrics = _json(final.metrics_path)
    train_loss = metrics["metrics"]["train_loss"]
    assert isinstance(train_loss, int | float)
    assert math.isfinite(float(train_loss))
    assert final.final_loss < first.final_loss
    assert metrics["metrics"]["nan_loss_count"] == 0
    assert metrics["metrics"]["collapse_var_min"]["value"] >= 1.0
    assert "not paper or model-quality evidence" in metrics["claim_boundary"]
    assert "nan_loss=false" in final.log_path.read_text(encoding="utf-8")

    manifest = _json(final.dataset_manifest_path)
    assert manifest["snapshot_id"] == "geno-lewm-fixture-snv-smoke-v1"
    assert "not Carbon, gnomAD, or ClinVar evidence" in manifest["sources"][0]["notes"]


def test_fixture_resume_reproduces_full_checkpoint_identity(tmp_path: Path) -> None:
    """Catch nondeterministic fixture resume before release-package tests run."""
    cfg = dataclasses.replace(
        load_default("train"),
        run_id="ml-smoke-resume",
        seed=7,
        deterministic=True,
    )

    full = run_fixture_training(
        config=cfg,
        run_dir=tmp_path / "full",
        steps=30,
        command="geno-lewm-train --fixture-smoke --steps 30",
        commit_sha="abcdef123456",
        package_version="0.1.0.dev0",
    )
    partial = run_fixture_training(
        config=cfg,
        run_dir=tmp_path / "resume",
        steps=10,
        command="geno-lewm-train --fixture-smoke --steps 10",
        commit_sha="abcdef123456",
        package_version="0.1.0.dev0",
    )
    resumed = run_fixture_training(
        config=cfg,
        run_dir=tmp_path / "resume",
        steps=30,
        resume_from=partial.checkpoint_path,
        command="geno-lewm-train --fixture-smoke --resume-from fixture_predictor_checkpoint.json",
        commit_sha="abcdef123456",
        package_version="0.1.0.dev0",
    )

    assert resumed.resumed_from_step == 10
    assert resumed.final_loss == pytest.approx(full.final_loss)
    assert _sha256(resumed.checkpoint_path) == _sha256(full.checkpoint_path)
    assert _json(resumed.metrics_path)["resumed_from_step"] == 10


def test_collapse_monitor_smoke_flags_degenerate_predictions() -> None:
    """Catch collapse-monitor regressions without importing a heavy ML stack."""
    monitor = CollapseMonitor()

    healthy = monitor.observe(
        [[-1.0, -1.0], [0.0, 0.0], [1.0, 1.0], [2.0, 2.0]],
        [[-1.0, -1.0], [0.0, 0.0], [1.0, 1.0], [2.0, 2.0]],
        kl_reg=0.0,
        step=1,
        force=True,
    )
    degenerate = monitor.observe(
        [[0.0, 0.0], [0.0, 0.0], [0.0, 0.0], [0.0, 0.0]],
        [[-2.0, -1.0], [0.0, 1.0], [2.0, 3.0], [4.0, 5.0]],
        kl_reg=0.0,
        step=2,
        force=True,
    )

    assert healthy is not None
    assert not healthy.tripped
    assert degenerate is not None
    assert degenerate.tripped
    assert "pred_var_per_dim" in {alert.criterion for alert in degenerate.alerts}


def _json(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()
