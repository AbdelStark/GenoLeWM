"""Tests for measured score/label evaluation helpers."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from geno_lewm._artifact_sources import SCORE_JSONL_GENERATED_BY
from geno_lewm.errors import InputError
from geno_lewm.evaluation import (
    BinaryEvalResult,
    VariantKey,
    _continuous_bootstrap_intervals,
    _quantile,
    _require_score_jsonl_generated_by,
    build_continuous_eval_report_payload,
    build_eval_report_payload,
    evaluate_continuous_score_labels,
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


def test_continuous_eval_report_payload_matches_release_report_schema(
    tmp_path: Path,
) -> None:
    scores, labels, baseline_scores = _write_continuous_score_label_artifacts(tmp_path)
    result = evaluate_continuous_score_labels(
        scores,
        labels,
        label_field="functional_score",
        split="brca2",
        bootstrap_resamples=25,
        bootstrap_seed=123,
    )
    baseline = evaluate_continuous_score_labels(
        baseline_scores,
        labels,
        label_field="functional_score",
        split="brca2",
        bootstrap_resamples=25,
        bootstrap_seed=123,
    )

    payload = build_continuous_eval_report_payload(
        result,
        model_id="sha256:" + "a" * 64,
        model_release="geno-lewm-v0.2.0-r1",
        dataset_snapshot="geno-lewm-data-v0.2.0-r1",
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

    parsed = parse_report_input(payload)

    assert result.spearman_rho == pytest.approx(1.0)
    assert baseline.spearman_rho == pytest.approx(-1.0)
    assert result.spearman_rho_ci_low == pytest.approx(1.0)
    assert result.spearman_rho_ci_high == pytest.approx(1.0)
    assert parsed.generated_by == "geno-lewm-eval"
    assert parsed.metrics[0].name == "spearman_rho"
    assert parsed.metrics[0].split == "brca2"
    assert parsed.metrics[0].value == pytest.approx(1.0)
    assert parsed.metrics[0].baseline == "carbon_zero_shot"
    assert parsed.metrics[0].baseline_value == pytest.approx(-1.0)
    assert parsed.metrics[0].delta_vs_baseline == pytest.approx(2.0)
    assert parsed.metrics[0].evaluated_variant_keys_sha256 == (
        parsed.metrics[0].baseline_evaluated_variant_keys_sha256
    )
    assert "spearman_rho=1" in payload["conclusions"][1]
    assert result.to_summary_dict()["spearman_rho_ci"] == [1.0, 1.0]


def test_eval_report_payloads_allow_missing_baseline(tmp_path: Path) -> None:
    scores, labels = _write_score_label_artifacts(tmp_path)
    result = evaluate_score_labels(scores, labels, bootstrap_resamples=0)

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
        generated_at="2026-06-01T00:00:00Z",
    )

    assert "baseline" not in payload["metrics"][0]
    assert "Compared with" not in payload["conclusions"][1]

    continuous_scores, continuous_labels, _baseline_scores = (
        _write_continuous_score_label_artifacts(tmp_path)
    )
    continuous_result = evaluate_continuous_score_labels(
        continuous_scores,
        continuous_labels,
        label_field="functional_score",
        bootstrap_resamples=0,
    )
    continuous_payload = build_continuous_eval_report_payload(
        continuous_result,
        model_id="sha256:" + "a" * 64,
        model_release="geno-lewm-v0.2.0-r1",
        dataset_snapshot="geno-lewm-data-v0.2.0-r1",
        commit="abcdef1234567890",
        hardware="local CPU eval fixture",
        checkpoint="model/predictor.safetensors",
        config="model/train_config.yaml",
        dataset_manifest="dataset/dataset_manifest.json",
        eval_config="eval_config.effective.yaml",
        efficiency_report="model/efficiency_report.json",
        scores=continuous_scores,
        labels=continuous_labels,
        generated_at="2026-06-01T00:00:00Z",
    )

    assert "baseline" not in continuous_payload["metrics"][0]
    assert "Compared with" not in continuous_payload["conclusions"][1]
    assert continuous_result.to_summary_dict()["spearman_rho_ci"] is None


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


def test_evaluate_continuous_score_labels_rejects_degenerate_inputs(tmp_path: Path) -> None:
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
            {"chrom": "1", "pos": 10, "ref": "A", "alt": "T", "functional_score": 0.1},
            {"chrom": "1", "pos": 20, "ref": "G", "alt": "A", "functional_score": 0.1},
        ],
    )

    with pytest.raises(InputError, match="non-constant labels and scores"):
        evaluate_continuous_score_labels(
            scores,
            labels,
            label_field="functional_score",
            bootstrap_resamples=0,
        )


def test_evaluate_continuous_score_labels_validates_options_and_inputs(
    tmp_path: Path,
) -> None:
    scores, labels, _baseline_scores = _write_continuous_score_label_artifacts(tmp_path)

    with pytest.raises(InputError, match="score_field must be a non-empty string"):
        evaluate_continuous_score_labels(scores, labels, score_field="")
    with pytest.raises(InputError, match="label_field must be a non-empty string"):
        evaluate_continuous_score_labels(scores, labels, label_field="")
    with pytest.raises(InputError, match="split must be a non-empty string"):
        evaluate_continuous_score_labels(scores, labels, split=" ")

    missing_scores = tmp_path / "continuous_missing_scores.jsonl"
    _write_jsonl(
        missing_scores,
        [
            {"chrom": "1", "pos": 10, "ref": "A", "alt": "T", "sigma_calibrated": 0.1},
            {"chrom": "1", "pos": 20, "ref": "G", "alt": "A", "sigma_calibrated": 0.2},
        ],
    )
    with pytest.raises(InputError, match="missing labelled variants"):
        evaluate_continuous_score_labels(
            missing_scores,
            labels,
            label_field="functional_score",
        )

    one_score = tmp_path / "continuous_one_score.jsonl"
    one_label = tmp_path / "continuous_one_label.jsonl"
    _write_jsonl(
        one_score,
        [{"chrom": "1", "pos": 10, "ref": "A", "alt": "T", "sigma_calibrated": 0.1}],
    )
    _write_jsonl(
        one_label,
        [{"chrom": "1", "pos": 10, "ref": "A", "alt": "T", "functional_score": 0.1}],
    )
    with pytest.raises(InputError, match="at least two matched variants"):
        evaluate_continuous_score_labels(
            one_score,
            one_label,
            label_field="functional_score",
        )


def test_evaluate_continuous_score_labels_rejects_jsonl_shape_errors(
    tmp_path: Path,
) -> None:
    scores, _labels, _baseline_scores = _write_continuous_score_label_artifacts(tmp_path)
    labels = tmp_path / "bad_continuous_labels.jsonl"

    _write_jsonl(
        labels,
        [
            {"chrom": "1", "pos": 10, "ref": "A", "alt": "T", "functional_score": 0.1},
            {"chrom": "1", "pos": 10, "ref": "A", "alt": "T", "functional_score": 0.2},
        ],
    )
    with pytest.raises(InputError, match="duplicate variant keys"):
        evaluate_continuous_score_labels(scores, labels, label_field="functional_score")

    _write_jsonl(labels, [{"chrom": "1", "pos": 10, "ref": "A", "alt": "T"}])
    with pytest.raises(InputError, match="functional_score must be a finite number"):
        evaluate_continuous_score_labels(scores, labels, label_field="functional_score")

    labels.write_text("\n\n", encoding="utf-8")
    with pytest.raises(InputError, match="contains no records"):
        evaluate_continuous_score_labels(scores, labels, label_field="functional_score")


def test_continuous_bootstrap_and_quantile_edge_cases() -> None:
    with pytest.raises(InputError, match="no non-degenerate samples"):
        _continuous_bootstrap_intervals(
            [1.0, 1.0],
            [0.5, 0.5],
            resamples=3,
            seed=0,
            ci_level=0.95,
        )

    with pytest.raises(InputError, match="empty sample"):
        _quantile([], 0.5)
    assert _quantile([0.25], 0.5) == 0.25
    assert _quantile([0.0, 0.5, 1.0], 0.5) == 0.5


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


def test_variant_keys_and_eval_options_validate_inputs(tmp_path: Path) -> None:
    scores, labels = _write_score_label_artifacts(tmp_path)

    with pytest.raises(InputError, match="chrom must be non-empty"):
        VariantKey("", 1, "A", "C")
    with pytest.raises(InputError, match="pos must be positive"):
        VariantKey("1", 0, "A", "C")
    with pytest.raises(InputError, match="ref and alt must be non-empty"):
        VariantKey("1", 1, "", "C")

    with pytest.raises(InputError, match="score_field must be a non-empty string"):
        evaluate_score_labels(scores, labels, score_field="")
    with pytest.raises(InputError, match="threshold must be a finite number"):
        evaluate_score_labels(scores, labels, threshold=float("nan"))
    with pytest.raises(InputError, match="split must be a non-empty string"):
        evaluate_score_labels(scores, labels, split=" ")
    with pytest.raises(InputError, match="bootstrap_resamples must be an integer"):
        evaluate_score_labels(scores, labels, bootstrap_resamples=True)  # type: ignore[arg-type]
    with pytest.raises(InputError, match="bootstrap_resamples must be non-negative"):
        evaluate_score_labels(scores, labels, bootstrap_resamples=-1)
    with pytest.raises(InputError, match="bootstrap_seed must be an integer"):
        evaluate_score_labels(scores, labels, bootstrap_seed=False)  # type: ignore[arg-type]
    with pytest.raises(InputError, match="ci_level must be greater than 0"):
        evaluate_score_labels(scores, labels, ci_level=1.0)


def test_evaluate_score_labels_rejects_label_jsonl_shape_errors(tmp_path: Path) -> None:
    scores, _labels = _write_score_label_artifacts(tmp_path)
    labels = tmp_path / "bad_labels.jsonl"

    with pytest.raises(InputError, match="JSONL artifact is missing"):
        evaluate_score_labels(scores, tmp_path / "missing.jsonl")

    labels.write_text("{not-json}\n", encoding="utf-8")
    with pytest.raises(InputError, match="invalid JSON"):
        evaluate_score_labels(scores, labels)

    labels.write_text("[1, 2, 3]\n", encoding="utf-8")
    with pytest.raises(InputError, match="records must be objects"):
        evaluate_score_labels(scores, labels)

    _write_jsonl(labels, [{"chrom": "1", "pos": 10, "ref": "A", "clinical_significance": "P"}])
    with pytest.raises(InputError, match="missing a key field"):
        evaluate_score_labels(scores, labels)

    _write_jsonl(
        labels,
        [{"chrom": "1", "pos": "bad", "ref": "A", "alt": "C", "clinical_significance": "P"}],
    )
    with pytest.raises(InputError, match="variant pos must be an integer"):
        evaluate_score_labels(scores, labels)

    _write_jsonl(labels, [{"chrom": "1", "pos": 10, "ref": "A", "alt": "C"}])
    with pytest.raises(InputError, match="clinical_significance or label"):
        evaluate_score_labels(scores, labels)

    _write_jsonl(
        labels,
        [
            {"chrom": "1", "pos": 10, "ref": "A", "alt": "T", "clinical_significance": "P"},
            {"chrom": "1", "pos": 10, "ref": "A", "alt": "T", "clinical_significance": "LP"},
        ],
    )
    with pytest.raises(InputError, match="duplicate variant keys"):
        evaluate_score_labels(scores, labels)

    _write_jsonl(labels, [{"chrom": "1", "pos": 10, "ref": "A", "alt": "T", "label": "VUS"}])
    with pytest.raises(InputError, match="no P/LP/B/LB variants"):
        evaluate_score_labels(scores, labels)


def test_evaluate_score_labels_rejects_score_jsonl_shape_errors(tmp_path: Path) -> None:
    scores = tmp_path / "bad_scores.jsonl"
    labels = tmp_path / "labels.jsonl"
    _write_jsonl(
        labels,
        [
            {"chrom": "1", "pos": 10, "ref": "A", "alt": "T", "clinical_significance": "P"},
            {"chrom": "1", "pos": 20, "ref": "G", "alt": "A", "clinical_significance": "B"},
        ],
    )

    _write_jsonl(
        scores,
        [
            {"chrom": "1", "pos": 10, "ref": "A", "alt": "T", "sigma_calibrated": 0.9},
            {"chrom": "1", "pos": 10, "ref": "A", "alt": "T", "sigma_calibrated": 0.8},
            {"chrom": "1", "pos": 20, "ref": "G", "alt": "A", "sigma_calibrated": 0.1},
        ],
    )
    with pytest.raises(InputError, match="duplicate variant keys"):
        evaluate_score_labels(scores, labels)

    _write_jsonl(
        scores,
        [
            {"chrom": "1", "pos": 10, "ref": "A", "alt": "T", "sigma_calibrated": True},
            {"chrom": "1", "pos": 20, "ref": "G", "alt": "A", "sigma_calibrated": 0.1},
        ],
    )
    with pytest.raises(InputError, match="sigma_calibrated must be a finite number"):
        evaluate_score_labels(scores, labels)


def test_eval_result_rejects_partial_confidence_interval_bounds() -> None:
    result = BinaryEvalResult(
        split="eval",
        score_field="sigma_calibrated",
        threshold=0.5,
        labelled_variants=2,
        evaluated_variants=2,
        positive_variants=1,
        negative_variants=1,
        extra_score_variants=0,
        auroc=1.0,
        average_precision=1.0,
        accuracy=1.0,
        balanced_accuracy=1.0,
        sensitivity=1.0,
        specificity=1.0,
        ci_level=0.95,
        bootstrap_resamples=10,
        bootstrap_seed=1,
        auroc_ci_low=0.9,
    )

    with pytest.raises(InputError, match="bounds must be supplied together"):
        result.to_report_metrics()
    with pytest.raises(InputError, match="bounds must be supplied together"):
        result.to_summary_dict()


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


def _write_continuous_score_label_artifacts(root: Path) -> tuple[Path, Path, Path]:
    scores = root / "continuous_scores.jsonl"
    labels = root / "continuous_labels.jsonl"
    baseline_scores = root / "continuous_baseline_scores.jsonl"
    _write_jsonl(
        scores,
        [
            {"chrom": "1", "pos": 10, "ref": "A", "alt": "T", "sigma_calibrated": 0.1},
            {"chrom": "1", "pos": 20, "ref": "G", "alt": "A", "sigma_calibrated": 0.2},
            {"chrom": "1", "pos": 30, "ref": "C", "alt": "G", "sigma_calibrated": 0.3},
            {"chrom": "1", "pos": 40, "ref": "T", "alt": "C", "sigma_calibrated": 0.4},
            {"chrom": "1", "pos": 50, "ref": "A", "alt": "C", "sigma_calibrated": 0.5},
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
        baseline_scores,
        [
            {"chrom": "1", "pos": 10, "ref": "A", "alt": "T", "sigma_calibrated": 0.4},
            {"chrom": "1", "pos": 20, "ref": "G", "alt": "A", "sigma_calibrated": 0.3},
            {"chrom": "1", "pos": 30, "ref": "C", "alt": "G", "sigma_calibrated": 0.2},
            {"chrom": "1", "pos": 40, "ref": "T", "alt": "C", "sigma_calibrated": 0.1},
        ],
    )
    return scores, labels, baseline_scores


def _write_jsonl(path: Path, records: list[dict[str, object]]) -> None:
    path.write_text(
        "".join(json.dumps(record, sort_keys=True) + "\n" for record in records),
        encoding="utf-8",
    )
