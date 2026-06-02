"""Tests for measured score/label evaluation helpers."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from geno_lewm._artifact_sources import SCORE_JSONL_GENERATED_BY
from geno_lewm.errors import InputError
from geno_lewm.evaluation import (
    _require_score_jsonl_generated_by,
    build_eval_report_payload,
    evaluate_score_labels,
)
from tools.release.eval_report import parse_report_input


def test_evaluate_score_labels_computes_binary_metrics(tmp_path: Path) -> None:
    scores, labels = _write_score_label_artifacts(tmp_path)

    result = evaluate_score_labels(
        scores,
        labels,
        split="eval_clinvar_coding",
        bootstrap_resamples=25,
        bootstrap_seed=123,
    )

    assert result.split == "eval_clinvar_coding"
    assert result.labelled_variants == 4
    assert result.evaluated_variants == 4
    assert result.positive_variants == 2
    assert result.negative_variants == 2
    assert result.extra_score_variants == 1
    assert result.auroc == 1.0
    assert result.average_precision == 1.0
    assert result.balanced_accuracy == 1.0
    assert result.accuracy == 1.0
    assert result.bootstrap_resamples == 25
    assert result.bootstrap_seed == 123
    assert result.auroc_ci_low == 1.0
    assert result.auroc_ci_high == 1.0
    assert result.evaluated_variant_keys_sha256.startswith("sha256:")
    assert result.to_summary_dict()["evaluated_variant_keys_sha256"] == (
        result.evaluated_variant_keys_sha256
    )


def test_eval_report_payload_matches_release_report_schema(tmp_path: Path) -> None:
    scores, labels = _write_score_label_artifacts(tmp_path)
    result = evaluate_score_labels(scores, labels, bootstrap_resamples=25, bootstrap_seed=123)
    baseline = evaluate_score_labels(
        _write_baseline_scores(tmp_path),
        labels,
        bootstrap_resamples=25,
        bootstrap_seed=123,
    )

    payload = build_eval_report_payload(
        result,
        model_id="sha256:" + "a" * 64,
        model_release="geno-lewm-v0.1.0-r1",
        dataset_snapshot="geno-lewm-data-v0.1.0-r1",
        commit="abcdef1234567890",
        hardware="local CPU eval fixture",
        checkpoint="model/predictor.safetensors",
        config="model/train_config.yaml",
        dataset_manifest="dataset/dataset_manifest.json",
        eval_config="eval_config.effective.yaml",
        efficiency_report="model/efficiency_report.json",
        scores=scores,
        labels=labels,
        baseline_result=baseline,
        baseline_name="carbon_zero_shot",
        baseline_scores=tmp_path / "baseline_scores.jsonl",
        generated_at="2026-06-01T00:00:00Z",
    )

    parsed = parse_report_input(payload)

    assert parsed.generated_by == "geno-lewm-eval"
    assert parsed.metrics[0].name == "auroc"
    assert parsed.metrics[0].value == 1.0
    assert parsed.metrics[0].ci_low == 1.0
    assert parsed.metrics[0].ci_high == 1.0
    assert parsed.metrics[0].baseline == "carbon_zero_shot"
    assert parsed.metrics[0].baseline_value == 0.75
    assert parsed.metrics[0].delta_vs_baseline == 0.25
    assert parsed.metrics[0].evaluated_variant_keys_sha256 == (
        parsed.metrics[0].baseline_evaluated_variant_keys_sha256
    )
    assert dict(parsed.artifacts)["baseline_scores"].endswith("baseline_scores.jsonl")
    assert dict(parsed.artifacts)["eval_config"] == "eval_config.effective.yaml"
    assert dict(parsed.artifacts)["efficiency_report"] == "model/efficiency_report.json"


def test_eval_report_payload_rejects_baseline_on_different_variant_keys(
    tmp_path: Path,
) -> None:
    scores, labels = _write_score_label_artifacts(tmp_path)
    result = evaluate_score_labels(scores, labels, bootstrap_resamples=0)
    baseline_labels = tmp_path / "baseline_labels.jsonl"
    baseline_scores = tmp_path / "baseline_scores_mismatched.jsonl"
    _write_jsonl(
        baseline_labels,
        [
            {"chrom": "2", "pos": 10, "ref": "A", "alt": "T", "clinical_significance": "P"},
            {"chrom": "2", "pos": 20, "ref": "G", "alt": "A", "clinical_significance": "LP"},
            {"chrom": "2", "pos": 30, "ref": "C", "alt": "G", "clinical_significance": "B"},
            {"chrom": "2", "pos": 40, "ref": "T", "alt": "C", "clinical_significance": "LB"},
        ],
    )
    _write_jsonl(
        baseline_scores,
        [
            {"chrom": "2", "pos": 10, "ref": "A", "alt": "T", "sigma_calibrated": 0.8},
            {"chrom": "2", "pos": 20, "ref": "G", "alt": "A", "sigma_calibrated": 0.3},
            {"chrom": "2", "pos": 30, "ref": "C", "alt": "G", "sigma_calibrated": 0.7},
            {"chrom": "2", "pos": 40, "ref": "T", "alt": "C", "sigma_calibrated": 0.2},
        ],
    )
    baseline = evaluate_score_labels(
        baseline_scores,
        baseline_labels,
        bootstrap_resamples=0,
    )

    assert result.evaluated_variant_keys_sha256 != baseline.evaluated_variant_keys_sha256
    with pytest.raises(InputError, match="baseline metrics are not comparable"):
        build_eval_report_payload(
            result,
            model_id="sha256:" + "a" * 64,
            model_release="geno-lewm-v0.1.0-r1",
            dataset_snapshot="geno-lewm-data-v0.1.0-r1",
            commit="abcdef1234567890",
            hardware="local CPU eval fixture",
            checkpoint="model/predictor.safetensors",
            config="model/train_config.yaml",
            dataset_manifest="dataset/dataset_manifest.json",
            eval_config="eval_config.effective.yaml",
            efficiency_report="model/efficiency_report.json",
            scores=scores,
            labels=labels,
            baseline_result=baseline,
            baseline_name="carbon_zero_shot",
            baseline_scores=baseline_scores,
            generated_at="2026-06-01T00:00:00Z",
        )


def test_eval_report_payload_requires_baseline_score_artifact(tmp_path: Path) -> None:
    scores, labels = _write_score_label_artifacts(tmp_path)
    result = evaluate_score_labels(scores, labels, bootstrap_resamples=25, bootstrap_seed=123)
    baseline = evaluate_score_labels(
        _write_baseline_scores(tmp_path),
        labels,
        bootstrap_resamples=25,
        bootstrap_seed=123,
    )

    with pytest.raises(InputError, match="baseline_result, baseline_name, and baseline_scores"):
        build_eval_report_payload(
            result,
            model_id="sha256:" + "a" * 64,
            model_release="geno-lewm-v0.1.0-r1",
            dataset_snapshot="geno-lewm-data-v0.1.0-r1",
            commit="abcdef1234567890",
            hardware="local CPU eval fixture",
            checkpoint="model/predictor.safetensors",
            config="model/train_config.yaml",
            dataset_manifest="dataset/dataset_manifest.json",
            eval_config="eval_config.effective.yaml",
            efficiency_report="model/efficiency_report.json",
            scores=scores,
            labels=labels,
            baseline_result=baseline,
            baseline_name="carbon_zero_shot",
            generated_at="2026-06-01T00:00:00Z",
        )


def test_evaluate_score_labels_can_require_score_generator(tmp_path: Path) -> None:
    scores, labels = _write_score_label_artifacts(tmp_path)
    rows = [json.loads(line) for line in scores.read_text(encoding="utf-8").splitlines()]
    for row in rows:
        row["generated_by"] = SCORE_JSONL_GENERATED_BY
    _write_jsonl(scores, rows)

    _require_score_jsonl_generated_by(scores, expected=SCORE_JSONL_GENERATED_BY)
    result = evaluate_score_labels(scores, labels, bootstrap_resamples=0)

    assert result.evaluated_variants == 4

    rows[0]["generated_by"] = "manual-editor"
    _write_jsonl(scores, rows)
    with pytest.raises(InputError, match="generated_by"):
        _require_score_jsonl_generated_by(scores, expected=SCORE_JSONL_GENERATED_BY)


def test_evaluate_score_labels_can_record_ci_omission_reason(tmp_path: Path) -> None:
    scores, labels = _write_score_label_artifacts(tmp_path)

    result = evaluate_score_labels(scores, labels, bootstrap_resamples=0)
    metrics = result.to_report_metrics()

    assert result.auroc_ci_low is None
    assert "ci_low" not in metrics[0]
    assert "bootstrap_resamples=0" in str(metrics[0]["notes"])


def test_evaluate_score_labels_rejects_missing_labelled_scores(tmp_path: Path) -> None:
    scores = tmp_path / "scores.jsonl"
    labels = tmp_path / "labels.jsonl"
    _write_jsonl(
        scores,
        [
            {"chrom": "1", "pos": 10, "ref": "A", "alt": "T", "sigma_calibrated": 0.9},
            {"chrom": "1", "pos": 30, "ref": "C", "alt": "G", "sigma_calibrated": 0.1},
        ],
    )
    _write_jsonl(
        labels,
        [
            {"chrom": "1", "pos": 10, "ref": "A", "alt": "T", "clinical_significance": "P"},
            {"chrom": "1", "pos": 20, "ref": "G", "alt": "A", "clinical_significance": "LP"},
            {"chrom": "1", "pos": 30, "ref": "C", "alt": "G", "clinical_significance": "B"},
        ],
    )

    with pytest.raises(InputError, match="missing labelled variants"):
        evaluate_score_labels(scores, labels)


def test_evaluate_score_labels_requires_positive_and_negative_labels(tmp_path: Path) -> None:
    scores = tmp_path / "scores.jsonl"
    labels = tmp_path / "labels.jsonl"
    _write_jsonl(
        scores,
        [
            {"chrom": "1", "pos": 10, "ref": "A", "alt": "T", "sigma_calibrated": 0.9},
            {"chrom": "1", "pos": 20, "ref": "G", "alt": "A", "sigma_calibrated": 0.8},
        ],
    )
    _write_jsonl(
        labels,
        [
            {"chrom": "1", "pos": 10, "ref": "A", "alt": "T", "clinical_significance": "P"},
            {"chrom": "1", "pos": 20, "ref": "G", "alt": "A", "clinical_significance": "LP"},
        ],
    )

    with pytest.raises(InputError, match="at least one positive and one negative"):
        evaluate_score_labels(scores, labels)


def _write_score_label_artifacts(root: Path) -> tuple[Path, Path]:
    scores = root / "scores.jsonl"
    labels = root / "labels.jsonl"
    _write_jsonl(
        scores,
        [
            {"chrom": "1", "pos": 10, "ref": "A", "alt": "T", "sigma_calibrated": 0.9},
            {"chrom": "1", "pos": 20, "ref": "G", "alt": "A", "sigma_calibrated": 0.8},
            {"chrom": "1", "pos": 30, "ref": "C", "alt": "G", "sigma_calibrated": 0.2},
            {"chrom": "1", "pos": 40, "ref": "T", "alt": "C", "sigma_calibrated": 0.1},
            {"chrom": "1", "pos": 50, "ref": "A", "alt": "C", "sigma_calibrated": 0.5},
        ],
    )
    _write_jsonl(
        labels,
        [
            {"chrom": "1", "pos": 10, "ref": "A", "alt": "T", "clinical_significance": "P"},
            {
                "chrom": "1",
                "pos": 20,
                "ref": "G",
                "alt": "A",
                "clinical_significance": "Pathogenic/Likely pathogenic",
            },
            {"chrom": "1", "pos": 30, "ref": "C", "alt": "G", "clinical_significance": "B"},
            {
                "chrom": "1",
                "pos": 40,
                "ref": "T",
                "alt": "C",
                "clinical_significance": "Benign/Likely benign",
            },
            {"chrom": "1", "pos": 60, "ref": "C", "alt": "T", "clinical_significance": "VUS"},
        ],
    )
    return scores, labels


def _write_baseline_scores(root: Path) -> Path:
    path = root / "baseline_scores.jsonl"
    _write_jsonl(
        path,
        [
            {"chrom": "1", "pos": 10, "ref": "A", "alt": "T", "sigma_calibrated": 0.8},
            {"chrom": "1", "pos": 20, "ref": "G", "alt": "A", "sigma_calibrated": 0.3},
            {"chrom": "1", "pos": 30, "ref": "C", "alt": "G", "sigma_calibrated": 0.7},
            {"chrom": "1", "pos": 40, "ref": "T", "alt": "C", "sigma_calibrated": 0.2},
        ],
    )
    return path


def _write_jsonl(path: Path, records: list[dict[str, object]]) -> None:
    path.write_text(
        "".join(json.dumps(record, sort_keys=True) + "\n" for record in records),
        encoding="utf-8",
    )
