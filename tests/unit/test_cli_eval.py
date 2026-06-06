"""Tests for the implemented ``geno-lewm-eval`` CLI slice."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from geno_lewm._artifact_sources import CARBON_ZERO_SHOT_GENERATED_BY, SCORE_JSONL_GENERATED_BY
from geno_lewm.cli import _dispatch
from geno_lewm.cli.eval import app
from tools.release.eval_report import load_report_input


def test_eval_requires_score_artifacts(
    capsys: pytest.CaptureFixture[str],
) -> None:
    rc = _dispatch.run_app(app, argv=["--quiet", "--no-banner"])
    captured = capsys.readouterr()

    assert rc == 2
    assert "requires --scores-jsonl" in captured.err
    assert "research tool" not in captured.err


def test_eval_requires_baseline_name_with_baseline_scores(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    scores = tmp_path / "scores.jsonl"
    labels = tmp_path / "labels.jsonl"
    baseline = tmp_path / "baseline_scores.jsonl"
    _write_jsonl(
        scores,
        [
            {"chrom": "1", "pos": 10, "ref": "A", "alt": "T", "sigma_calibrated": 0.9},
            {"chrom": "1", "pos": 20, "ref": "G", "alt": "A", "sigma_calibrated": 0.8},
            {"chrom": "1", "pos": 30, "ref": "C", "alt": "G", "sigma_calibrated": 0.2},
            {"chrom": "1", "pos": 40, "ref": "T", "alt": "C", "sigma_calibrated": 0.1},
        ],
    )
    _write_jsonl(
        labels,
        [
            {"chrom": "1", "pos": 10, "ref": "A", "alt": "T", "clinical_significance": "P"},
            {"chrom": "1", "pos": 20, "ref": "G", "alt": "A", "clinical_significance": "LP"},
            {"chrom": "1", "pos": 30, "ref": "C", "alt": "G", "clinical_significance": "B"},
            {"chrom": "1", "pos": 40, "ref": "T", "alt": "C", "clinical_significance": "LB"},
        ],
    )
    _write_jsonl(
        baseline,
        [
            {"chrom": "1", "pos": 10, "ref": "A", "alt": "T", "sigma_calibrated": 0.8},
            {"chrom": "1", "pos": 20, "ref": "G", "alt": "A", "sigma_calibrated": 0.3},
            {"chrom": "1", "pos": 30, "ref": "C", "alt": "G", "sigma_calibrated": 0.7},
            {"chrom": "1", "pos": 40, "ref": "T", "alt": "C", "sigma_calibrated": 0.2},
        ],
    )

    rc = _dispatch.run_app(
        app,
        argv=[
            "--quiet",
            "--no-banner",
            "--scores-jsonl",
            str(scores),
            "--labels-jsonl",
            str(labels),
            "--baseline-scores-jsonl",
            str(baseline),
            "--output-metrics",
            str(tmp_path / "metrics.json"),
            "--model-id",
            "sha256:" + "b" * 64,
            "--model-release",
            "geno-lewm-v0.1.0-r1",
            "--dataset-snapshot",
            "geno-lewm-data-v0.1.0-r1",
            "--commit",
            "abcdef1234567890",
            "--hardware",
            "local CPU eval fixture",
            "--checkpoint",
            str(tmp_path / "model" / "predictor.safetensors"),
            "--config-artifact",
            str(tmp_path / "model" / "train_config.yaml"),
            "--dataset-manifest",
            str(tmp_path / "dataset" / "dataset_manifest.json"),
            "--efficiency-report",
            str(tmp_path / "model" / "efficiency_report.json"),
        ],
    )
    captured = capsys.readouterr()

    assert rc == 2
    assert "requires --baseline-name" in captured.err


def test_eval_writes_release_metrics_json(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    scores = tmp_path / "scores.jsonl"
    labels = tmp_path / "labels.jsonl"
    baseline = tmp_path / "baseline_scores.jsonl"
    output = tmp_path / "metrics.json"
    output_config = tmp_path / "eval_config.effective.yaml"
    _write_jsonl(
        scores,
        [
            {"chrom": "1", "pos": 10, "ref": "A", "alt": "T", "sigma_calibrated": 0.9},
            {"chrom": "1", "pos": 20, "ref": "G", "alt": "A", "sigma_calibrated": 0.8},
            {"chrom": "1", "pos": 30, "ref": "C", "alt": "G", "sigma_calibrated": 0.2},
            {"chrom": "1", "pos": 40, "ref": "T", "alt": "C", "sigma_calibrated": 0.1},
        ],
    )
    _write_jsonl(
        labels,
        [
            {"chrom": "1", "pos": 10, "ref": "A", "alt": "T", "clinical_significance": "P"},
            {"chrom": "1", "pos": 20, "ref": "G", "alt": "A", "clinical_significance": "LP"},
            {"chrom": "1", "pos": 30, "ref": "C", "alt": "G", "clinical_significance": "B"},
            {"chrom": "1", "pos": 40, "ref": "T", "alt": "C", "clinical_significance": "LB"},
        ],
    )
    _write_jsonl(
        baseline,
        [
            {"chrom": "1", "pos": 10, "ref": "A", "alt": "T", "sigma_calibrated": 0.8},
            {"chrom": "1", "pos": 20, "ref": "G", "alt": "A", "sigma_calibrated": 0.3},
            {"chrom": "1", "pos": 30, "ref": "C", "alt": "G", "sigma_calibrated": 0.7},
            {"chrom": "1", "pos": 40, "ref": "T", "alt": "C", "sigma_calibrated": 0.2},
        ],
    )

    rc = _dispatch.run_app(
        app,
        argv=[
            "--quiet",
            "--no-banner",
            "--scores-jsonl",
            str(scores),
            "--labels-jsonl",
            str(labels),
            "--baseline-scores-jsonl",
            str(baseline),
            "--baseline-name",
            "carbon_zero_shot",
            "--baseline-score-field",
            "carbon_zero_shot_score",
            "--output-metrics",
            str(output),
            "--model-id",
            "sha256:" + "b" * 64,
            "--model-release",
            "geno-lewm-v0.1.0-r1",
            "--dataset-snapshot",
            "geno-lewm-data-v0.1.0-r1",
            "--commit",
            "abcdef1234567890",
            "--hardware",
            "local CPU eval fixture",
            "--checkpoint",
            str(tmp_path / "model" / "predictor.safetensors"),
            "--config-artifact",
            str(tmp_path / "model" / "train_config.yaml"),
            "--dataset-manifest",
            str(tmp_path / "dataset" / "dataset_manifest.json"),
            "--efficiency-report",
            str(tmp_path / "model" / "efficiency_report.json"),
            "--bootstrap-resamples",
            "25",
            "--bootstrap-seed",
            "123",
        ],
    )
    captured = capsys.readouterr()

    assert rc == 0
    summary = json.loads(captured.out)
    assert summary["metrics_json"] == str(output)
    assert summary["auroc"] == 1.0
    assert summary["baseline_name"] == "carbon_zero_shot"
    assert summary["baseline_auroc"] == 0.75
    assert summary["auroc_delta_vs_baseline"] == 0.25
    assert summary["bootstrap_resamples"] == 25
    assert summary["auroc_ci"] == [1.0, 1.0]
    assert summary["evaluated_variant_keys_sha256"].startswith("sha256:")
    assert output_config.is_file()
    report_input = load_report_input(output)
    assert report_input.generated_by == "geno-lewm-eval"
    assert report_input.metrics[0].name == "auroc"
    assert report_input.metrics[0].baseline == "carbon_zero_shot"
    assert report_input.metrics[0].baseline_value == 0.75
    assert report_input.metrics[0].evaluated_variant_keys_sha256 == (
        report_input.metrics[0].baseline_evaluated_variant_keys_sha256
    )
    assert report_input.metrics[0].ci_low == 1.0
    artifacts = dict(report_input.artifacts)
    assert artifacts["checkpoint"] == "model/predictor.safetensors"
    assert artifacts["config"] == "model/train_config.yaml"
    assert artifacts["dataset_manifest"] == "dataset/dataset_manifest.json"
    assert artifacts["eval_config"] == "eval_config.effective.yaml"
    assert artifacts["efficiency_report"] == "model/efficiency_report.json"
    assert artifacts["scores"] == "scores.jsonl"
    assert artifacts["labels"] == "labels.jsonl"
    assert artifacts["baseline_scores"] == "baseline_scores.jsonl"


def test_eval_writes_continuous_spearman_metrics_json(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    scores = tmp_path / "scores.jsonl"
    labels = tmp_path / "labels.jsonl"
    baseline = tmp_path / "baseline_scores.jsonl"
    output = tmp_path / "metrics.json"
    _write_jsonl(
        scores,
        [
            {"chrom": "1", "pos": 10, "ref": "A", "alt": "T", "sigma_calibrated": 0.1},
            {"chrom": "1", "pos": 20, "ref": "G", "alt": "A", "sigma_calibrated": 0.2},
            {"chrom": "1", "pos": 30, "ref": "C", "alt": "G", "sigma_calibrated": 0.3},
            {"chrom": "1", "pos": 40, "ref": "T", "alt": "C", "sigma_calibrated": 0.4},
        ],
    )
    _write_jsonl(
        labels,
        [
            {"chrom": "1", "pos": 10, "ref": "A", "alt": "T", "functional_score": 0.1},
            {"chrom": "1", "pos": 20, "ref": "G", "alt": "A", "functional_score": 0.2},
            {"chrom": "1", "pos": 30, "ref": "C", "alt": "G", "functional_score": 0.3},
            {"chrom": "1", "pos": 40, "ref": "T", "alt": "C", "functional_score": 0.4},
        ],
    )
    _write_jsonl(
        baseline,
        [
            {"chrom": "1", "pos": 10, "ref": "A", "alt": "T", "sigma_calibrated": 0.4},
            {"chrom": "1", "pos": 20, "ref": "G", "alt": "A", "sigma_calibrated": 0.3},
            {"chrom": "1", "pos": 30, "ref": "C", "alt": "G", "sigma_calibrated": 0.2},
            {"chrom": "1", "pos": 40, "ref": "T", "alt": "C", "sigma_calibrated": 0.1},
        ],
    )

    rc = _dispatch.run_app(
        app,
        argv=[
            "--quiet",
            "--no-banner",
            "--scores-jsonl",
            str(scores),
            "--labels-jsonl",
            str(labels),
            "--metric-mode",
            "spearman",
            "--label-field",
            "functional_score",
            "--split",
            "brca2",
            "--baseline-scores-jsonl",
            str(baseline),
            "--baseline-name",
            "carbon_zero_shot",
            "--baseline-score-field",
            "carbon_zero_shot_score",
            "--output-metrics",
            str(output),
            "--model-id",
            "sha256:" + "b" * 64,
            "--model-release",
            "geno-lewm-v0.2.0-r1",
            "--dataset-snapshot",
            "geno-lewm-data-v0.2.0-r1",
            "--commit",
            "abcdef1234567890",
            "--hardware",
            "local CPU eval fixture",
            "--checkpoint",
            str(tmp_path / "model" / "predictor.safetensors"),
            "--config-artifact",
            str(tmp_path / "model" / "train_config.yaml"),
            "--dataset-manifest",
            str(tmp_path / "dataset" / "dataset_manifest.json"),
            "--efficiency-report",
            str(tmp_path / "model" / "efficiency_report.json"),
            "--bootstrap-resamples",
            "25",
            "--bootstrap-seed",
            "123",
        ],
    )
    captured = capsys.readouterr()

    assert rc == 0
    summary = json.loads(captured.out)
    assert summary["metrics_json"] == str(output)
    assert summary["split"] == "brca2"
    assert summary["label_field"] == "functional_score"
    assert summary["spearman_rho"] == pytest.approx(1.0)
    assert summary["baseline_name"] == "carbon_zero_shot"
    assert summary["baseline_spearman_rho"] == pytest.approx(-1.0)
    assert summary["spearman_rho_delta_vs_baseline"] == pytest.approx(2.0)
    assert summary["spearman_rho_ci"] == [1.0, 1.0]
    report_input = load_report_input(output)
    assert report_input.generated_by == "geno-lewm-eval"
    assert report_input.metrics[0].name == "spearman_rho"
    assert report_input.metrics[0].split == "brca2"
    assert report_input.metrics[0].baseline == "carbon_zero_shot"
    assert report_input.metrics[0].baseline_value == pytest.approx(-1.0)
    assert report_input.metrics[0].delta_vs_baseline == pytest.approx(2.0)
    assert report_input.metrics[0].evaluated_variant_keys_sha256 == (
        report_input.metrics[0].baseline_evaluated_variant_keys_sha256
    )


def test_eval_records_paths_relative_to_artifact_root(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    release_root = tmp_path / "release"
    model_dir = release_root / "model"
    eval_dir = model_dir / "eval"
    dataset_dir = release_root / "dataset"
    eval_dir.mkdir(parents=True)
    dataset_dir.mkdir(parents=True)
    scores = eval_dir / "scores.jsonl"
    labels = dataset_dir / "labels.jsonl"
    output = model_dir / "eval_metrics.json"
    output_config = model_dir / "eval_config.effective.yaml"
    _write_minimal_score_label_artifacts(scores, labels)

    rc = _dispatch.run_app(
        app,
        argv=[
            "--quiet",
            "--no-banner",
            "--scores-jsonl",
            str(scores),
            "--labels-jsonl",
            str(labels),
            "--output-metrics",
            str(output),
            "--artifact-root",
            str(release_root),
            "--model-id",
            "sha256:" + "b" * 64,
            "--model-release",
            "geno-lewm-v0.1.0-r1",
            "--dataset-snapshot",
            "geno-lewm-data-v0.1.0-r1",
            "--commit",
            "abcdef1234567890",
            "--hardware",
            "local CPU eval fixture",
            "--checkpoint",
            str(model_dir / "predictor.safetensors"),
            "--config-artifact",
            str(model_dir / "train_config.yaml"),
            "--dataset-manifest",
            str(dataset_dir / "dataset_manifest.json"),
            "--efficiency-report",
            str(model_dir / "efficiency_report.json"),
            "--bootstrap-resamples",
            "0",
        ],
    )
    capsys.readouterr()

    assert rc == 0
    assert output_config.is_file()
    report_input = load_report_input(output)
    artifacts = dict(report_input.artifacts)
    assert artifacts["checkpoint"] == "model/predictor.safetensors"
    assert artifacts["config"] == "model/train_config.yaml"
    assert artifacts["dataset_manifest"] == "dataset/dataset_manifest.json"
    assert artifacts["eval_config"] == "model/eval_config.effective.yaml"
    assert artifacts["efficiency_report"] == "model/efficiency_report.json"
    assert artifacts["scores"] == "model/eval/scores.jsonl"
    assert artifacts["labels"] == "dataset/labels.jsonl"


def test_eval_rejects_absolute_artifacts_outside_artifact_root(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    release_root = tmp_path / "release"
    outside = tmp_path / "outside"
    release_root.mkdir()
    outside.mkdir()
    scores = outside / "scores.jsonl"
    labels = release_root / "labels.jsonl"
    _write_minimal_score_label_artifacts(scores, labels)

    rc = _dispatch.run_app(
        app,
        argv=[
            "--quiet",
            "--no-banner",
            "--scores-jsonl",
            str(scores),
            "--labels-jsonl",
            str(labels),
            "--output-metrics",
            str(release_root / "eval_metrics.json"),
            "--model-id",
            "sha256:" + "b" * 64,
            "--model-release",
            "geno-lewm-v0.1.0-r1",
            "--dataset-snapshot",
            "geno-lewm-data-v0.1.0-r1",
            "--commit",
            "abcdef1234567890",
            "--hardware",
            "local CPU eval fixture",
            "--checkpoint",
            str(release_root / "predictor.safetensors"),
            "--config-artifact",
            str(release_root / "train_config.yaml"),
            "--dataset-manifest",
            str(release_root / "dataset_manifest.json"),
            "--efficiency-report",
            str(release_root / "efficiency_report.json"),
        ],
    )
    captured = capsys.readouterr()

    assert rc == 2
    assert "artifact paths must stay inside --artifact-root" in captured.err


def _write_jsonl(path: Path, records: list[dict[str, object]]) -> None:
    rows = [_with_source_marker(path, record) for record in records]
    path.write_text(
        "".join(json.dumps(record, sort_keys=True) + "\n" for record in rows),
        encoding="utf-8",
    )


def _with_source_marker(path: Path, record: dict[str, object]) -> dict[str, object]:
    row = dict(record)
    if "generated_by" in row:
        return row
    if "sigma_calibrated" in row and path.name == "scores.jsonl":
        row["generated_by"] = SCORE_JSONL_GENERATED_BY
    elif path.name == "baseline_scores.jsonl":
        row["generated_by"] = CARBON_ZERO_SHOT_GENERATED_BY
        if "sigma_calibrated" in row and "carbon_zero_shot_score" not in row:
            row["carbon_zero_shot_score"] = row.pop("sigma_calibrated")
    return row


def _write_minimal_score_label_artifacts(scores: Path, labels: Path) -> None:
    _write_jsonl(
        scores,
        [
            {"chrom": "1", "pos": 10, "ref": "A", "alt": "T", "sigma_calibrated": 0.9},
            {"chrom": "1", "pos": 20, "ref": "G", "alt": "A", "sigma_calibrated": 0.8},
            {"chrom": "1", "pos": 30, "ref": "C", "alt": "G", "sigma_calibrated": 0.2},
            {"chrom": "1", "pos": 40, "ref": "T", "alt": "C", "sigma_calibrated": 0.1},
        ],
    )
    _write_jsonl(
        labels,
        [
            {"chrom": "1", "pos": 10, "ref": "A", "alt": "T", "clinical_significance": "P"},
            {"chrom": "1", "pos": 20, "ref": "G", "alt": "A", "clinical_significance": "LP"},
            {"chrom": "1", "pos": 30, "ref": "C", "alt": "G", "clinical_significance": "B"},
            {"chrom": "1", "pos": 40, "ref": "T", "alt": "C", "clinical_significance": "LB"},
        ],
    )
