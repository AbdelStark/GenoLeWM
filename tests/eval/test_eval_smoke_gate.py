# SPDX-License-Identifier: Apache-2.0
"""Hosted eval smoke-regression gate tests."""

from __future__ import annotations

import json
from pathlib import Path

from tools.ci.eval_smoke_gate import EvalSmokeThresholds, main, run_eval_smoke_gate
from tools.release.eval_report import load_report_input


def test_eval_smoke_gate_generates_release_eval_artifacts(tmp_path: Path) -> None:
    """Catch eval/report plumbing regressions using generated public fixtures."""
    summary_path = tmp_path / "summary.json"

    summary = run_eval_smoke_gate(work_dir=tmp_path / "work", summary_json=summary_path)

    assert summary["ok"] is True
    assert summary["generated_by"] == "tools.ci.eval_smoke_gate"
    assert summary["regressions"] == []
    observed = summary["observed"]
    assert isinstance(observed, dict)
    assert observed["auroc"] == 1.0
    assert observed["average_precision"] == 1.0
    assert observed["balanced_accuracy"] == 1.0
    assert observed["auroc_delta_vs_baseline"] > 0.0
    real_model_path = summary["real_model_path"]
    assert isinstance(real_model_path, dict)
    assert real_model_path["status"] == "not_attempted"
    assert "generated public fixture artifacts only" in real_model_path["reason"]
    persisted = json.loads(summary_path.read_text(encoding="utf-8"))
    assert persisted == summary

    artifacts = summary["artifacts"]
    assert isinstance(artifacts, dict)
    aggregate = load_report_input(tmp_path / "work" / artifacts["aggregate_metrics_json"])
    assert aggregate.generated_by == "geno-lewm-eval-all"
    assert "eval_config" in dict(aggregate.artifacts)
    report = tmp_path / "work" / artifacts["eval_report"]
    assert "## Negative Findings" in report.read_text(encoding="utf-8")


def test_eval_smoke_gate_fails_when_threshold_is_crossed(tmp_path: Path) -> None:
    """Prove the hosted gate blocks metric regressions instead of only logging them."""
    summary = run_eval_smoke_gate(
        work_dir=tmp_path / "work",
        thresholds=EvalSmokeThresholds(min_auroc=1.01),
    )

    assert summary["ok"] is False
    assert summary["regressions"] == [{"metric": "auroc", "minimum": 1.01, "observed": 1.0}]


def test_eval_smoke_gate_cli_returns_one_for_regression(tmp_path: Path) -> None:
    """Keep the CI entrypoint wired to a failing exit code on threshold crossings."""
    summary_path = tmp_path / "summary.json"

    rc = main(
        [
            "--work-dir",
            str(tmp_path / "work"),
            "--summary-json",
            str(summary_path),
            "--min-auroc",
            "1.01",
        ]
    )

    assert rc == 1
    assert json.loads(summary_path.read_text(encoding="utf-8"))["ok"] is False
