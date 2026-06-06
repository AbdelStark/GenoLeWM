"""Tests for the implemented ``geno-lewm-rollout`` metric aggregation path."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from geno_lewm._artifact_sources import ROLLOUT_STATES_GENERATED_BY
from geno_lewm.cli import _dispatch
from geno_lewm.cli.rollout import app
from geno_lewm.provenance import sha256_bytes
from tools.release.eval_report import load_report_input
from tools.release.v02_benchmark_readiness import build_readiness_report


def test_rollout_requires_states_jsonl(capsys: pytest.CaptureFixture[str]) -> None:
    rc = _dispatch.run_app(app, argv=["--quiet", "--no-banner"])
    captured = capsys.readouterr()

    assert rc == 2
    assert "requires --states-jsonl" in captured.err


def test_rollout_aggregates_measured_state_rows(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    states = tmp_path / "eval" / "rollout_states.jsonl"
    states.parent.mkdir()
    output = tmp_path / "eval_metrics.json"
    _write_state_rows(states)

    rc = _dispatch.run_app(
        app,
        argv=[
            "--quiet",
            "--no-banner",
            "--states-jsonl",
            str(states),
            "--output-metrics",
            str(output),
            "--artifact-root",
            str(tmp_path),
            "--recall-k",
            "10",
            "--model-id",
            sha256_bytes(b"model"),
            "--model-release",
            "geno-lewm-v0.2.0-r1",
            "--dataset-snapshot",
            "geno-lewm-data-v0.2.0-r1",
            "--commit",
            "abcdef1234567890",
            "--hardware",
            "Apple M3 Max CPU",
            "--checkpoint",
            str(tmp_path / "model" / "predictor.safetensors"),
            "--config-artifact",
            str(tmp_path / "model" / "train_config.yaml"),
            "--dataset-manifest",
            str(tmp_path / "dataset" / "dataset_manifest.json"),
            "--efficiency-report",
            str(tmp_path / "model" / "efficiency_report.json"),
            "--rollout-state-examples-report",
            str(tmp_path / "eval" / "rollout_state_examples_report.json"),
            "--rollout-state-rows-report",
            str(tmp_path / "eval" / "rollout_state_rows_report.json"),
        ],
    )
    captured = capsys.readouterr()

    assert rc == 0
    summary = json.loads(captured.out)
    assert summary == {
        "metrics": 6,
        "metrics_json": str(output),
        "recall_k": 10,
        "rows": 3,
        "splits": ["rollout_phased_haplotypes", "rollout_synthetic_edit_chains"],
    }
    report = load_report_input(output)
    assert report.generated_by == "geno-lewm-eval"
    metrics = {(metric.split, metric.name): metric for metric in report.metrics}
    phased_cosine = metrics[("rollout_phased_haplotypes", "cosine_similarity_mean")]
    assert phased_cosine.value == pytest.approx(0.8)
    assert phased_cosine.baseline == "source_state"
    assert phased_cosine.baseline_value == pytest.approx(0.0)
    assert phased_cosine.delta_vs_baseline == pytest.approx(0.8)
    phased_recall = metrics[("rollout_phased_haplotypes", "recall_at_k")]
    assert phased_recall.value == pytest.approx(0.5)
    artifacts = dict(report.artifacts)
    assert artifacts["rollout_states"] == "eval/rollout_states.jsonl"
    assert artifacts["baseline_rollout_states"] == "eval/rollout_states.jsonl"
    assert artifacts["rollout_state_examples_report"] == "eval/rollout_state_examples_report.json"
    assert artifacts["rollout_state_rows_report"] == "eval/rollout_state_rows_report.json"
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["rollout_stratification"][0]["k"] == 5
    readiness = build_readiness_report(metrics_json=(output,))
    rows = {
        str(row["benchmark_id"]): row
        for row in readiness["benchmark_rows"]
        if isinstance(row, dict)
    }
    assert rows["rollout_phased_haplotypes"]["status"] == "pass"
    assert rows["rollout_synthetic_edit_chains"]["status"] == "pass"


def test_rollout_rejects_wrong_generated_by(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    states = tmp_path / "rollout_states.jsonl"
    row = _state_row("bad", split="rollout_phased_haplotypes", horizon=5)
    row["generated_by"] = "manual"
    states.write_text(json.dumps(row) + "\n", encoding="utf-8")

    rc = _dispatch.run_app(
        app,
        argv=[
            "--quiet",
            "--no-banner",
            "--states-jsonl",
            str(states),
            "--output-metrics",
            str(tmp_path / "eval_metrics.json"),
            "--model-id",
            sha256_bytes(b"model"),
            "--model-release",
            "geno-lewm-v0.2.0-r1",
            "--dataset-snapshot",
            "geno-lewm-data-v0.2.0-r1",
            "--commit",
            "abcdef1234567890",
            "--hardware",
            "Apple M3 Max CPU",
            "--checkpoint",
            str(tmp_path / "predictor.safetensors"),
            "--config-artifact",
            str(tmp_path / "train_config.yaml"),
            "--dataset-manifest",
            str(tmp_path / "dataset_manifest.json"),
            "--efficiency-report",
            str(tmp_path / "efficiency_report.json"),
        ],
    )
    captured = capsys.readouterr()

    assert rc == 2
    assert "rollout state generated_by is invalid" in captured.err


def _write_state_rows(path: Path) -> None:
    rows = [
        _state_row(
            "phased-k5-a",
            split="rollout_phased_haplotypes",
            horizon=5,
            predicted=[1.0, 0.0],
            target_rank=1,
            baseline_rank=99,
        ),
        _state_row(
            "phased-k5-b",
            split="rollout_phased_haplotypes",
            horizon=5,
            predicted=[0.6, 0.8],
            target_rank=11,
            baseline_rank=99,
        ),
        _state_row(
            "synthetic-k13-a",
            split="rollout_synthetic_edit_chains",
            horizon=13,
            predicted=[0.8, 0.6],
            target_rank=1,
            baseline_rank=99,
        ),
    ]
    path.write_text("\n".join(json.dumps(row, sort_keys=True) for row in rows) + "\n")


def _state_row(
    row_id: str,
    *,
    split: str,
    horizon: int,
    predicted: list[float] | None = None,
    target_rank: int = 1,
    baseline_rank: int = 99,
) -> dict[str, object]:
    return {
        "generated_by": ROLLOUT_STATES_GENERATED_BY,
        "id": row_id,
        "split": split,
        "k": horizon,
        "source_state": [0.0, 1.0],
        "predicted_state": [1.0, 0.0] if predicted is None else predicted,
        "target_state": [1.0, 0.0],
        "target_rank": target_rank,
        "baseline_target_rank": baseline_rank,
    }
