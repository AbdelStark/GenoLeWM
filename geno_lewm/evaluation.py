# SPDX-License-Identifier: Apache-2.0
"""Artifact-level evaluation helpers for first-experiment score files."""

from __future__ import annotations

import json
import math
import random
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Final

from geno_lewm.errors import InputError
from geno_lewm.provenance import canonical_json_sha256

__all__ = [
    "DEFAULT_BOOTSTRAP_RESAMPLES",
    "DEFAULT_BOOTSTRAP_SEED",
    "DEFAULT_CI_LEVEL",
    "DEFAULT_EVAL_SCORE_FIELD",
    "DEFAULT_EVAL_THRESHOLD",
    "NEGATIVE_CLINVAR_LABELS",
    "POSITIVE_CLINVAR_LABELS",
    "BinaryEvalResult",
    "ContinuousEvalResult",
    "VariantKey",
    "build_continuous_eval_report_payload",
    "build_eval_report_payload",
    "evaluate_continuous_score_labels",
    "evaluate_score_labels",
]

DEFAULT_EVAL_SCORE_FIELD: Final = "sigma_calibrated"
DEFAULT_EVAL_THRESHOLD: Final = 0.5
DEFAULT_BOOTSTRAP_RESAMPLES: Final = 1000
DEFAULT_BOOTSTRAP_SEED: Final = 0
DEFAULT_CI_LEVEL: Final = 0.95
POSITIVE_CLINVAR_LABELS: Final = frozenset({"P", "LP"})
NEGATIVE_CLINVAR_LABELS: Final = frozenset({"B", "LB"})
_LABEL_ALIASES: Final = {
    "P": "P",
    "PATHOGENIC": "P",
    "LP": "LP",
    "LIKELY_PATHOGENIC": "LP",
    "B": "B",
    "BENIGN": "B",
    "LB": "LB",
    "LIKELY_BENIGN": "LB",
}


@dataclass(frozen=True, slots=True, order=True)
class VariantKey:
    """Comparable SNV key shared by score and label artifacts."""

    chrom: str
    pos: int
    ref: str
    alt: str

    def __post_init__(self) -> None:
        if not self.chrom:
            raise InputError("variant chrom must be non-empty")
        if self.pos <= 0:
            raise InputError("variant pos must be positive", details={"pos": self.pos})
        if not self.ref or not self.alt:
            raise InputError("variant ref and alt must be non-empty")
        object.__setattr__(self, "ref", self.ref.upper())
        object.__setattr__(self, "alt", self.alt.upper())

    def to_dict(self) -> dict[str, str | int]:
        """Return the JSON-native variant-key payload."""
        return {"chrom": self.chrom, "pos": self.pos, "ref": self.ref, "alt": self.alt}


@dataclass(frozen=True, slots=True)
class BinaryEvalResult:
    """Measured binary evaluation over labelled score records."""

    split: str
    score_field: str
    threshold: float
    labelled_variants: int
    evaluated_variants: int
    positive_variants: int
    negative_variants: int
    extra_score_variants: int
    auroc: float
    average_precision: float
    accuracy: float
    balanced_accuracy: float
    sensitivity: float
    specificity: float
    ci_level: float
    bootstrap_resamples: int
    bootstrap_seed: int
    auroc_ci_low: float | None = None
    auroc_ci_high: float | None = None
    average_precision_ci_low: float | None = None
    average_precision_ci_high: float | None = None
    accuracy_ci_low: float | None = None
    accuracy_ci_high: float | None = None
    balanced_accuracy_ci_low: float | None = None
    balanced_accuracy_ci_high: float | None = None
    _evaluated_variant_keys_sha256: str = field(default="", init=False, repr=False, compare=False)

    @property
    def evaluated_variant_keys_sha256(self) -> str:
        """SHA-256 identity of the sorted evaluated variant-key set, when known."""
        return self._evaluated_variant_keys_sha256

    def to_report_metrics(self) -> list[dict[str, object]]:
        """Render metrics in ``tools.release.eval_report`` input shape."""
        notes = f"positive=P/LP and negative=B/LB ClinVar labels; scores use {self.score_field}"
        variant_hash = _evaluated_variant_hash_payload(self.evaluated_variant_keys_sha256)
        metrics = [
            {
                "name": "auroc",
                "value": self.auroc,
                "split": self.split,
                "unit": "area",
                "higher_is_better": True,
                "n": self.evaluated_variants,
                "notes": notes,
                **variant_hash,
                **_ci_payload(self.auroc_ci_low, self.auroc_ci_high),
            },
            {
                "name": "average_precision",
                "value": self.average_precision,
                "split": self.split,
                "unit": "area",
                "higher_is_better": True,
                "n": self.evaluated_variants,
                "notes": notes,
                **variant_hash,
                **_ci_payload(
                    self.average_precision_ci_low,
                    self.average_precision_ci_high,
                ),
            },
            {
                "name": "balanced_accuracy",
                "value": self.balanced_accuracy,
                "split": self.split,
                "unit": "fraction",
                "higher_is_better": True,
                "n": self.evaluated_variants,
                "notes": f"threshold={self.threshold:g}",
                **variant_hash,
                **_ci_payload(
                    self.balanced_accuracy_ci_low,
                    self.balanced_accuracy_ci_high,
                ),
            },
            {
                "name": "accuracy",
                "value": self.accuracy,
                "split": self.split,
                "unit": "fraction",
                "higher_is_better": True,
                "n": self.evaluated_variants,
                "notes": f"threshold={self.threshold:g}",
                **variant_hash,
                **_ci_payload(self.accuracy_ci_low, self.accuracy_ci_high),
            },
        ]
        ci_note = self._ci_note()
        for metric in metrics:
            metric["notes"] = f"{metric['notes']}; {ci_note}"
        return metrics

    def to_summary_dict(self) -> dict[str, object]:
        """Return a compact JSON summary for CLI stdout."""
        return {
            "split": self.split,
            "score_field": self.score_field,
            "threshold": self.threshold,
            "labelled_variants": self.labelled_variants,
            "evaluated_variants": self.evaluated_variants,
            "evaluated_variant_keys_sha256": self.evaluated_variant_keys_sha256 or None,
            "positive_variants": self.positive_variants,
            "negative_variants": self.negative_variants,
            "extra_score_variants": self.extra_score_variants,
            "auroc": self.auroc,
            "average_precision": self.average_precision,
            "accuracy": self.accuracy,
            "balanced_accuracy": self.balanced_accuracy,
            "sensitivity": self.sensitivity,
            "specificity": self.specificity,
            "ci_level": self.ci_level,
            "bootstrap_resamples": self.bootstrap_resamples,
            "bootstrap_seed": self.bootstrap_seed,
            "auroc_ci": _ci_summary(self.auroc_ci_low, self.auroc_ci_high),
            "average_precision_ci": _ci_summary(
                self.average_precision_ci_low,
                self.average_precision_ci_high,
            ),
            "accuracy_ci": _ci_summary(self.accuracy_ci_low, self.accuracy_ci_high),
            "balanced_accuracy_ci": _ci_summary(
                self.balanced_accuracy_ci_low,
                self.balanced_accuracy_ci_high,
            ),
        }

    def _ci_note(self) -> str:
        if self.bootstrap_resamples <= 0:
            return "confidence intervals omitted because bootstrap_resamples=0"
        return (
            f"{self.ci_level:g} stratified bootstrap confidence interval; "
            f"resamples={self.bootstrap_resamples}; seed={self.bootstrap_seed}"
        )


@dataclass(frozen=True, slots=True)
class ContinuousEvalResult:
    """Measured continuous-label evaluation over matched score records."""

    split: str
    score_field: str
    label_field: str
    labelled_variants: int
    evaluated_variants: int
    extra_score_variants: int
    spearman_rho: float
    ci_level: float
    bootstrap_resamples: int
    bootstrap_seed: int
    spearman_rho_ci_low: float | None = None
    spearman_rho_ci_high: float | None = None
    _evaluated_variant_keys_sha256: str = field(default="", init=False, repr=False, compare=False)

    @property
    def evaluated_variant_keys_sha256(self) -> str:
        """SHA-256 identity of the sorted evaluated variant-key set, when known."""
        return self._evaluated_variant_keys_sha256

    def to_report_metrics(self) -> list[dict[str, object]]:
        """Render Spearman correlation in ``tools.release.eval_report`` input shape."""
        notes = (
            f"continuous labels use {self.label_field}; scores use {self.score_field}; "
            f"{self._ci_note()}"
        )
        return [
            {
                "name": "spearman_rho",
                "value": self.spearman_rho,
                "split": self.split,
                "unit": "correlation",
                "higher_is_better": True,
                "n": self.evaluated_variants,
                "notes": notes,
                **_evaluated_variant_hash_payload(self.evaluated_variant_keys_sha256),
                **_ci_payload(self.spearman_rho_ci_low, self.spearman_rho_ci_high),
            }
        ]

    def to_summary_dict(self) -> dict[str, object]:
        """Return a compact JSON summary for CLI stdout."""
        return {
            "split": self.split,
            "score_field": self.score_field,
            "label_field": self.label_field,
            "labelled_variants": self.labelled_variants,
            "evaluated_variants": self.evaluated_variants,
            "evaluated_variant_keys_sha256": self.evaluated_variant_keys_sha256 or None,
            "extra_score_variants": self.extra_score_variants,
            "spearman_rho": self.spearman_rho,
            "ci_level": self.ci_level,
            "bootstrap_resamples": self.bootstrap_resamples,
            "bootstrap_seed": self.bootstrap_seed,
            "spearman_rho_ci": _ci_summary(
                self.spearman_rho_ci_low,
                self.spearman_rho_ci_high,
            ),
        }

    def _ci_note(self) -> str:
        if self.bootstrap_resamples <= 0:
            return "confidence intervals omitted because bootstrap_resamples=0"
        return (
            f"{self.ci_level:g} bootstrap confidence interval; "
            f"resamples={self.bootstrap_resamples}; seed={self.bootstrap_seed}"
        )


@dataclass(frozen=True, slots=True)
class _BinaryMetricValues:
    auroc: float
    average_precision: float
    accuracy: float
    balanced_accuracy: float
    sensitivity: float
    specificity: float


def evaluate_score_labels(
    scores_jsonl: str | Path,
    labels_jsonl: str | Path,
    *,
    score_field: str = DEFAULT_EVAL_SCORE_FIELD,
    threshold: float = DEFAULT_EVAL_THRESHOLD,
    split: str = "eval_clinvar",
    bootstrap_resamples: int = DEFAULT_BOOTSTRAP_RESAMPLES,
    bootstrap_seed: int = DEFAULT_BOOTSTRAP_SEED,
    ci_level: float = DEFAULT_CI_LEVEL,
) -> BinaryEvalResult:
    """Evaluate score JSONL against held-out ClinVar-style label JSONL."""
    if not isinstance(score_field, str) or not score_field:
        raise InputError("score_field must be a non-empty string")
    _require_finite_number("threshold", threshold)
    if not isinstance(split, str) or not split.strip():
        raise InputError("split must be a non-empty string")
    resamples = _require_non_negative_int("bootstrap_resamples", bootstrap_resamples)
    seed = _require_int("bootstrap_seed", bootstrap_seed)
    level = _require_ci_level(ci_level)

    labels = _load_label_map(Path(labels_jsonl))
    scores = _load_score_map(Path(scores_jsonl), score_field=score_field)
    missing = sorted(set(labels) - set(scores))
    if missing:
        raise InputError(
            "scores are missing labelled variants",
            details={"missing": [key.to_dict() for key in missing[:10]], "count": len(missing)},
            remediation="score every held-out labelled variant before generating metrics",
        )

    keys = sorted(labels)
    y_true = [labels[key] for key in keys]
    y_score = [scores[key] for key in keys]
    positives = sum(1 for value in y_true if value)
    negatives = len(y_true) - positives
    if positives == 0 or negatives == 0:
        raise InputError(
            "label evaluation requires at least one positive and one negative variant",
            details={"positive": positives, "negative": negatives},
        )

    values = _binary_metric_values(y_true, y_score, threshold=threshold)
    intervals = _bootstrap_intervals(
        y_true,
        y_score,
        threshold=threshold,
        resamples=resamples,
        seed=seed,
        ci_level=level,
    )
    return _with_evaluated_variant_keys_sha256(
        BinaryEvalResult(
            split=split.strip(),
            score_field=score_field,
            threshold=float(threshold),
            labelled_variants=len(labels),
            evaluated_variants=len(y_true),
            positive_variants=positives,
            negative_variants=negatives,
            extra_score_variants=len(set(scores) - set(labels)),
            auroc=values.auroc,
            average_precision=values.average_precision,
            accuracy=values.accuracy,
            balanced_accuracy=values.balanced_accuracy,
            sensitivity=values.sensitivity,
            specificity=values.specificity,
            ci_level=level,
            bootstrap_resamples=resamples,
            bootstrap_seed=seed,
            auroc_ci_low=intervals.get("auroc", (None, None))[0],
            auroc_ci_high=intervals.get("auroc", (None, None))[1],
            average_precision_ci_low=intervals.get("average_precision", (None, None))[0],
            average_precision_ci_high=intervals.get("average_precision", (None, None))[1],
            accuracy_ci_low=intervals.get("accuracy", (None, None))[0],
            accuracy_ci_high=intervals.get("accuracy", (None, None))[1],
            balanced_accuracy_ci_low=intervals.get("balanced_accuracy", (None, None))[0],
            balanced_accuracy_ci_high=intervals.get("balanced_accuracy", (None, None))[1],
        ),
        _variant_keys_sha256(keys),
    )


def evaluate_continuous_score_labels(
    scores_jsonl: str | Path,
    labels_jsonl: str | Path,
    *,
    score_field: str = DEFAULT_EVAL_SCORE_FIELD,
    label_field: str = "value",
    split: str = "eval_continuous",
    bootstrap_resamples: int = DEFAULT_BOOTSTRAP_RESAMPLES,
    bootstrap_seed: int = DEFAULT_BOOTSTRAP_SEED,
    ci_level: float = DEFAULT_CI_LEVEL,
) -> ContinuousEvalResult:
    """Evaluate score JSONL against held-out continuous labels using Spearman rho."""
    if not isinstance(score_field, str) or not score_field:
        raise InputError("score_field must be a non-empty string")
    if not isinstance(label_field, str) or not label_field:
        raise InputError("label_field must be a non-empty string")
    if not isinstance(split, str) or not split.strip():
        raise InputError("split must be a non-empty string")
    resamples = _require_non_negative_int("bootstrap_resamples", bootstrap_resamples)
    seed = _require_int("bootstrap_seed", bootstrap_seed)
    level = _require_ci_level(ci_level)

    labels = _load_continuous_label_map(Path(labels_jsonl), label_field=label_field)
    scores = _load_score_map(Path(scores_jsonl), score_field=score_field)
    missing = sorted(set(labels) - set(scores))
    if missing:
        raise InputError(
            "scores are missing labelled variants",
            details={"missing": [key.to_dict() for key in missing[:10]], "count": len(missing)},
            remediation="score every held-out labelled variant before generating metrics",
        )

    keys = sorted(labels)
    y_true = [labels[key] for key in keys]
    y_score = [scores[key] for key in keys]
    if len(keys) < 2:
        raise InputError("continuous evaluation requires at least two matched variants")
    rho = _spearman_rho(y_true, y_score)
    intervals = _continuous_bootstrap_intervals(
        y_true,
        y_score,
        resamples=resamples,
        seed=seed,
        ci_level=level,
    )
    return _with_continuous_evaluated_variant_keys_sha256(
        ContinuousEvalResult(
            split=split.strip(),
            score_field=score_field,
            label_field=label_field,
            labelled_variants=len(labels),
            evaluated_variants=len(keys),
            extra_score_variants=len(set(scores) - set(labels)),
            spearman_rho=rho,
            ci_level=level,
            bootstrap_resamples=resamples,
            bootstrap_seed=seed,
            spearman_rho_ci_low=intervals.get("spearman_rho", (None, None))[0],
            spearman_rho_ci_high=intervals.get("spearman_rho", (None, None))[1],
        ),
        _variant_keys_sha256(keys),
    )


def build_eval_report_payload(
    result: BinaryEvalResult,
    *,
    model_id: str,
    model_release: str,
    dataset_snapshot: str,
    commit: str,
    hardware: str,
    checkpoint: str | Path,
    config: str | Path,
    dataset_manifest: str | Path,
    eval_config: str | Path,
    efficiency_report: str | Path,
    scores: str | Path,
    labels: str | Path,
    baseline_result: BinaryEvalResult | None = None,
    baseline_name: str | None = None,
    baseline_scores: str | Path | None = None,
    generated_at: str | None = None,
) -> dict[str, object]:
    """Build measured metrics JSON accepted by ``tools.release.eval_report``."""
    _require_baseline_inputs(
        baseline_result=baseline_result,
        baseline_name=baseline_name,
        baseline_scores=baseline_scores,
    )
    generated = generated_at or datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
    evaluated = result.evaluated_variants
    positives = result.positive_variants
    negatives = result.negative_variants
    return {
        "schema_version": "1.0.0",
        "generated_by": "geno-lewm-eval",
        "generated_at": generated,
        "model_id": _required_text("model_id", model_id),
        "model_release": _required_text("model_release", model_release),
        "dataset_snapshot": _required_text("dataset_snapshot", dataset_snapshot),
        "commit": _required_text("commit", commit),
        "hardware": _required_text("hardware", hardware),
        "metrics": _report_metrics(
            result,
            baseline_result=baseline_result,
            baseline_name=baseline_name,
        ),
        "artifacts": {
            "checkpoint": str(checkpoint),
            "config": str(config),
            "dataset_manifest": str(dataset_manifest),
            "eval_config": str(eval_config),
            "efficiency_report": str(efficiency_report),
            "scores": str(scores),
            "labels": str(labels),
            **({} if baseline_scores is None else {"baseline_scores": str(baseline_scores)}),
        },
        "limitations": [
            (
                "Artifact-level binary ClinVar evaluation only; labels outside P/LP/B/LB "
                "are excluded from the measured set."
            ),
            (
                "Confidence intervals use deterministic stratified bootstrap resampling; "
                "if bootstrap_resamples is zero, metric notes state that intervals were omitted."
            ),
            "The metrics do not establish clinical utility or deployment readiness.",
        ],
        "negative_findings": [
            "This report does not measure non-coding, multi-edit, or prospective clinical utility.",
            "Failures and omitted intervals must be read from the metric notes and limitations.",
        ],
        "conclusions": [
            (
                f"The score artifact was evaluated on {evaluated} labelled variants "
                f"({positives} positive, {negatives} negative) from {result.split}."
            ),
            _summary_conclusion(
                result, baseline_result=baseline_result, baseline_name=baseline_name
            ),
        ],
    }


def build_continuous_eval_report_payload(
    result: ContinuousEvalResult,
    *,
    model_id: str,
    model_release: str,
    dataset_snapshot: str,
    commit: str,
    hardware: str,
    checkpoint: str | Path,
    config: str | Path,
    dataset_manifest: str | Path,
    eval_config: str | Path,
    efficiency_report: str | Path,
    scores: str | Path,
    labels: str | Path,
    baseline_result: ContinuousEvalResult | None = None,
    baseline_name: str | None = None,
    baseline_scores: str | Path | None = None,
    generated_at: str | None = None,
) -> dict[str, object]:
    """Build measured continuous metrics JSON accepted by ``eval_report``."""
    _require_baseline_inputs(
        baseline_result=baseline_result,
        baseline_name=baseline_name,
        baseline_scores=baseline_scores,
    )
    generated = generated_at or datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
    return {
        "schema_version": "1.0.0",
        "generated_by": "geno-lewm-eval",
        "generated_at": generated,
        "model_id": _required_text("model_id", model_id),
        "model_release": _required_text("model_release", model_release),
        "dataset_snapshot": _required_text("dataset_snapshot", dataset_snapshot),
        "commit": _required_text("commit", commit),
        "hardware": _required_text("hardware", hardware),
        "metrics": _continuous_report_metrics(
            result,
            baseline_result=baseline_result,
            baseline_name=baseline_name,
        ),
        "artifacts": {
            "checkpoint": str(checkpoint),
            "config": str(config),
            "dataset_manifest": str(dataset_manifest),
            "eval_config": str(eval_config),
            "efficiency_report": str(efficiency_report),
            "scores": str(scores),
            "labels": str(labels),
            **({} if baseline_scores is None else {"baseline_scores": str(baseline_scores)}),
        },
        "limitations": [
            "Artifact-level continuous-label evaluation only; labels are not clinical outcomes.",
            (
                "Confidence intervals use deterministic bootstrap resampling; if "
                "bootstrap_resamples is zero, metric notes state that intervals were omitted."
            ),
            "The metrics do not establish clinical utility or deployment readiness.",
        ],
        "negative_findings": [
            "This report does not measure prospective clinical utility.",
            "Correlation metrics do not establish calibration or causal edit effects.",
        ],
        "conclusions": [
            (
                f"The score artifact was evaluated on {result.evaluated_variants} "
                f"continuous-label variants from {result.split}."
            ),
            _continuous_summary_conclusion(
                result,
                baseline_result=baseline_result,
                baseline_name=baseline_name,
            ),
        ],
    }


def _require_baseline_inputs(
    *,
    baseline_result: BinaryEvalResult | ContinuousEvalResult | None,
    baseline_name: str | None,
    baseline_scores: str | Path | None,
) -> None:
    has_result = baseline_result is not None
    has_name = bool(baseline_name and baseline_name.strip())
    has_scores = baseline_scores is not None and bool(str(baseline_scores).strip())
    if any((has_result, has_name, has_scores)) and not all((has_result, has_name, has_scores)):
        raise InputError(
            "baseline_result, baseline_name, and baseline_scores must be supplied together",
            details={
                "baseline_result": has_result,
                "baseline_name": has_name,
                "baseline_scores": has_scores,
            },
        )


def _report_metrics(
    result: BinaryEvalResult,
    *,
    baseline_result: BinaryEvalResult | None,
    baseline_name: str | None,
) -> list[dict[str, object]]:
    metrics = result.to_report_metrics()
    if baseline_result is None:
        return metrics
    name = _required_text("baseline_name", baseline_name or "")
    _require_comparable_baseline(result, baseline_result)
    baseline_by_name = {metric["name"]: metric for metric in baseline_result.to_report_metrics()}
    for metric in metrics:
        baseline_metric = baseline_by_name.get(metric["name"])
        if baseline_metric is None:
            raise InputError(
                "baseline result is missing a metric",
                details={"metric": metric["name"]},
            )
        value = _require_finite_number("metric.value", metric["value"])
        baseline_value = _require_finite_number("baseline_metric.value", baseline_metric["value"])
        higher_is_better = metric.get("higher_is_better")
        if not isinstance(higher_is_better, bool):
            raise InputError("metric higher_is_better must be a boolean")
        metric["baseline"] = name
        metric["baseline_value"] = baseline_value
        metric["delta_vs_baseline"] = (
            value - baseline_value if higher_is_better else baseline_value - value
        )
        if baseline_result.evaluated_variant_keys_sha256:
            metric["baseline_evaluated_variant_keys_sha256"] = (
                baseline_result.evaluated_variant_keys_sha256
            )
        metric["notes"] = f"{metric['notes']}; baseline {name} uses {baseline_result.score_field}"
    return metrics


def _continuous_report_metrics(
    result: ContinuousEvalResult,
    *,
    baseline_result: ContinuousEvalResult | None,
    baseline_name: str | None,
) -> list[dict[str, object]]:
    metrics = result.to_report_metrics()
    if baseline_result is None:
        return metrics
    name = _required_text("baseline_name", baseline_name or "")
    _require_comparable_continuous_baseline(result, baseline_result)
    baseline_by_name = {metric["name"]: metric for metric in baseline_result.to_report_metrics()}
    for metric in metrics:
        baseline_metric = baseline_by_name.get(metric["name"])
        if baseline_metric is None:
            raise InputError(
                "baseline result is missing a metric",
                details={"metric": metric["name"]},
            )
        value = _require_finite_number("metric.value", metric["value"])
        baseline_value = _require_finite_number("baseline_metric.value", baseline_metric["value"])
        metric["baseline"] = name
        metric["baseline_value"] = baseline_value
        metric["delta_vs_baseline"] = value - baseline_value
        if baseline_result.evaluated_variant_keys_sha256:
            metric["baseline_evaluated_variant_keys_sha256"] = (
                baseline_result.evaluated_variant_keys_sha256
            )
        metric["notes"] = f"{metric['notes']}; baseline {name} uses {baseline_result.score_field}"
    return metrics


def _require_comparable_baseline(
    result: BinaryEvalResult,
    baseline_result: BinaryEvalResult,
) -> None:
    expected = {
        "split": result.split,
        "labelled_variants": result.labelled_variants,
        "evaluated_variants": result.evaluated_variants,
        "positive_variants": result.positive_variants,
        "negative_variants": result.negative_variants,
        "threshold": result.threshold,
    }
    observed = {
        "split": baseline_result.split,
        "labelled_variants": baseline_result.labelled_variants,
        "evaluated_variants": baseline_result.evaluated_variants,
        "positive_variants": baseline_result.positive_variants,
        "negative_variants": baseline_result.negative_variants,
        "threshold": baseline_result.threshold,
    }
    result_hash = result.evaluated_variant_keys_sha256
    baseline_hash = baseline_result.evaluated_variant_keys_sha256
    if result_hash or baseline_hash:
        expected["evaluated_variant_keys_sha256"] = result_hash
        observed["evaluated_variant_keys_sha256"] = baseline_hash
    mismatches = {
        key: {"expected": expected_value, "observed": observed[key]}
        for key, expected_value in expected.items()
        if observed[key] != expected_value
    }
    if mismatches:
        raise InputError(
            "baseline metrics are not comparable to evaluated metrics",
            details={"mismatches": mismatches},
        )


def _require_comparable_continuous_baseline(
    result: ContinuousEvalResult,
    baseline_result: ContinuousEvalResult,
) -> None:
    expected = {
        "split": result.split,
        "label_field": result.label_field,
        "labelled_variants": result.labelled_variants,
        "evaluated_variants": result.evaluated_variants,
    }
    observed = {
        "split": baseline_result.split,
        "label_field": baseline_result.label_field,
        "labelled_variants": baseline_result.labelled_variants,
        "evaluated_variants": baseline_result.evaluated_variants,
    }
    result_hash = result.evaluated_variant_keys_sha256
    baseline_hash = baseline_result.evaluated_variant_keys_sha256
    if result_hash or baseline_hash:
        expected["evaluated_variant_keys_sha256"] = result_hash
        observed["evaluated_variant_keys_sha256"] = baseline_hash
    mismatches = {
        key: {"expected": expected_value, "observed": observed[key]}
        for key, expected_value in expected.items()
        if observed[key] != expected_value
    }
    if mismatches:
        raise InputError(
            "baseline metrics are not comparable to evaluated metrics",
            details={"mismatches": mismatches},
        )


def _summary_conclusion(
    result: BinaryEvalResult,
    *,
    baseline_result: BinaryEvalResult | None,
    baseline_name: str | None,
) -> str:
    base = (
        f"auroc={result.auroc:.6g}, average_precision={result.average_precision:.6g}, "
        f"accuracy={result.accuracy:.6g}, and balanced_accuracy="
        f"{result.balanced_accuracy:.6g} at threshold {result.threshold:g}."
    )
    if baseline_result is None:
        return base
    name = _required_text("baseline_name", baseline_name or "")
    return (
        f"{base} Compared with {name}, auroc delta="
        f"{result.auroc - baseline_result.auroc:.6g}, average_precision delta="
        f"{result.average_precision - baseline_result.average_precision:.6g}, "
        f"accuracy delta={result.accuracy - baseline_result.accuracy:.6g}, "
        f"and balanced_accuracy delta="
        f"{result.balanced_accuracy - baseline_result.balanced_accuracy:.6g}."
    )


def _continuous_summary_conclusion(
    result: ContinuousEvalResult,
    *,
    baseline_result: ContinuousEvalResult | None,
    baseline_name: str | None,
) -> str:
    base = f"spearman_rho={result.spearman_rho:.6g} using label field {result.label_field}."
    if baseline_result is None:
        return base
    name = _required_text("baseline_name", baseline_name or "")
    return (
        f"{base} Compared with {name}, spearman_rho delta="
        f"{result.spearman_rho - baseline_result.spearman_rho:.6g}."
    )


def _variant_keys_sha256(keys: list[VariantKey]) -> str:
    return canonical_json_sha256([key.to_dict() for key in keys])


def _with_evaluated_variant_keys_sha256(
    result: BinaryEvalResult,
    digest: str,
) -> BinaryEvalResult:
    object.__setattr__(result, "_evaluated_variant_keys_sha256", digest)
    return result


def _with_continuous_evaluated_variant_keys_sha256(
    result: ContinuousEvalResult,
    digest: str,
) -> ContinuousEvalResult:
    object.__setattr__(result, "_evaluated_variant_keys_sha256", digest)
    return result


def _evaluated_variant_hash_payload(digest: str) -> dict[str, str]:
    if not digest:
        return {}
    return {"evaluated_variant_keys_sha256": digest}


def _binary_metric_values(
    labels: list[bool],
    scores: list[float],
    *,
    threshold: float,
) -> _BinaryMetricValues:
    predictions = [score >= threshold for score in scores]
    tp = sum(
        1 for observed, predicted in zip(labels, predictions, strict=True) if observed and predicted
    )
    tn = sum(
        1
        for observed, predicted in zip(labels, predictions, strict=True)
        if not observed and not predicted
    )
    fp = sum(
        1
        for observed, predicted in zip(labels, predictions, strict=True)
        if not observed and predicted
    )
    fn = sum(
        1
        for observed, predicted in zip(labels, predictions, strict=True)
        if observed and not predicted
    )
    sensitivity = tp / (tp + fn)
    specificity = tn / (tn + fp)
    return _BinaryMetricValues(
        auroc=_auroc(labels, scores),
        average_precision=_average_precision(labels, scores),
        accuracy=(tp + tn) / len(labels),
        balanced_accuracy=(sensitivity + specificity) / 2.0,
        sensitivity=sensitivity,
        specificity=specificity,
    )


def _bootstrap_intervals(
    labels: list[bool],
    scores: list[float],
    *,
    threshold: float,
    resamples: int,
    seed: int,
    ci_level: float,
) -> dict[str, tuple[float, float]]:
    if resamples <= 0:
        return {}
    positives = [score for label, score in zip(labels, scores, strict=True) if label]
    negatives = [score for label, score in zip(labels, scores, strict=True) if not label]
    rng = random.Random(seed)
    samples: dict[str, list[float]] = {
        "auroc": [],
        "average_precision": [],
        "accuracy": [],
        "balanced_accuracy": [],
    }
    for _ in range(resamples):
        sampled_pos = rng.choices(positives, k=len(positives))
        sampled_neg = rng.choices(negatives, k=len(negatives))
        sampled_labels = [True] * len(sampled_pos) + [False] * len(sampled_neg)
        sampled_scores = sampled_pos + sampled_neg
        values = _binary_metric_values(sampled_labels, sampled_scores, threshold=threshold)
        samples["auroc"].append(values.auroc)
        samples["average_precision"].append(values.average_precision)
        samples["accuracy"].append(values.accuracy)
        samples["balanced_accuracy"].append(values.balanced_accuracy)
    return {
        name: _confidence_interval(values, ci_level=ci_level) for name, values in samples.items()
    }


def _continuous_bootstrap_intervals(
    labels: list[float],
    scores: list[float],
    *,
    resamples: int,
    seed: int,
    ci_level: float,
) -> dict[str, tuple[float, float]]:
    if resamples <= 0:
        return {}
    rng = random.Random(seed)
    values: list[float] = []
    pairs = list(zip(labels, scores, strict=True))
    for _ in range(resamples):
        sample = rng.choices(pairs, k=len(pairs))
        sampled_labels = [label for label, _score in sample]
        sampled_scores = [score for _label, score in sample]
        try:
            values.append(_spearman_rho(sampled_labels, sampled_scores))
        except InputError:
            continue
    if not values:
        raise InputError("continuous bootstrap produced no non-degenerate samples")
    return {"spearman_rho": _confidence_interval(values, ci_level=ci_level)}


def _confidence_interval(values: list[float], *, ci_level: float) -> tuple[float, float]:
    alpha = (1.0 - ci_level) / 2.0
    ordered = sorted(values)
    return (_quantile(ordered, alpha), _quantile(ordered, 1.0 - alpha))


def _quantile(ordered: list[float], q: float) -> float:
    if not ordered:
        raise InputError("cannot compute a quantile from an empty sample")
    if len(ordered) == 1:
        return ordered[0]
    index = q * (len(ordered) - 1)
    low = math.floor(index)
    high = math.ceil(index)
    if low == high:
        return ordered[low]
    weight = index - low
    return ordered[low] * (1.0 - weight) + ordered[high] * weight


def _ci_payload(low: float | None, high: float | None) -> dict[str, float]:
    if low is None and high is None:
        return {}
    if low is None or high is None:
        raise InputError("confidence interval bounds must be supplied together")
    return {"ci_low": low, "ci_high": high}


def _ci_summary(low: float | None, high: float | None) -> list[float] | None:
    if low is None and high is None:
        return None
    if low is None or high is None:
        raise InputError("confidence interval bounds must be supplied together")
    return [low, high]


def _load_label_map(path: Path) -> dict[VariantKey, bool]:
    labels: dict[VariantKey, bool] = {}
    skipped = 0
    for line_no, record in _iter_jsonl(path):
        key = _variant_key(record, path=path, line_no=line_no)
        label = _clinical_label(record, path=path, line_no=line_no)
        if label is None:
            skipped += 1
            continue
        if key in labels:
            raise InputError(
                "label JSONL contains duplicate variant keys",
                details={"path": str(path), "line": line_no, "variant": key.to_dict()},
            )
        labels[key] = label
    if not labels:
        raise InputError(
            "label JSONL contains no P/LP/B/LB variants",
            details={"path": str(path), "skipped": skipped},
        )
    return labels


def _load_score_map(
    path: Path,
    *,
    score_field: str,
) -> dict[VariantKey, float]:
    scores: dict[VariantKey, float] = {}
    for line_no, record in _iter_jsonl(path):
        key = _variant_key(record, path=path, line_no=line_no)
        if key in scores:
            raise InputError(
                "score JSONL contains duplicate variant keys",
                details={"path": str(path), "line": line_no, "variant": key.to_dict()},
            )
        value = record.get(score_field)
        scores[key] = _require_finite_number(
            score_field,
            value,
            details={"path": str(path), "line": line_no},
        )
    if not scores:
        raise InputError("score JSONL contains no score records", details={"path": str(path)})
    return scores


def _load_continuous_label_map(
    path: Path,
    *,
    label_field: str,
) -> dict[VariantKey, float]:
    labels: dict[VariantKey, float] = {}
    for line_no, record in _iter_jsonl(path):
        key = _variant_key(record, path=path, line_no=line_no)
        if key in labels:
            raise InputError(
                "label JSONL contains duplicate variant keys",
                details={"path": str(path), "line": line_no, "variant": key.to_dict()},
            )
        labels[key] = _require_finite_number(
            label_field,
            record.get(label_field),
            details={"path": str(path), "line": line_no},
        )
    if not labels:
        raise InputError("label JSONL contains no continuous labels", details={"path": str(path)})
    return labels


def _require_score_jsonl_generated_by(path: str | Path, *, expected: str) -> None:
    resolved = Path(path)
    for line_no, record in _iter_jsonl(resolved):
        _require_generated_by(record, expected=expected, path=resolved, line_no=line_no)


def _require_generated_by(
    record: dict[str, Any],
    *,
    expected: str,
    path: Path,
    line_no: int,
) -> None:
    observed = record.get("generated_by")
    if observed != expected:
        raise InputError(
            "score JSONL row generated_by is invalid",
            details={
                "path": str(path),
                "line": line_no,
                "expected": expected,
                "observed": observed,
            },
            remediation="regenerate the score artifact with the expected CLI",
        )


def _iter_jsonl(path: Path) -> list[tuple[int, dict[str, Any]]]:
    if not path.is_file():
        raise InputError("JSONL artifact is missing", details={"path": str(path)})
    records: list[tuple[int, dict[str, Any]]] = []
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        raise InputError("failed to read JSONL artifact", details={"path": str(path)}) from exc
    for line_no, raw_line in enumerate(lines, start=1):
        line = raw_line.strip()
        if not line:
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError as exc:
            raise InputError(
                "JSONL artifact contains invalid JSON",
                details={"path": str(path), "line": line_no, "column": exc.colno},
            ) from exc
        if not isinstance(record, dict):
            raise InputError(
                "JSONL artifact records must be objects",
                details={"path": str(path), "line": line_no},
            )
        records.append((line_no, record))
    if not records:
        raise InputError("JSONL artifact contains no records", details={"path": str(path)})
    return records


def _variant_key(record: dict[str, Any], *, path: Path, line_no: int) -> VariantKey:
    try:
        chrom = str(record["chrom"])
        pos = int(record["pos"])
        ref = str(record["ref"])
        alt = str(record["alt"])
    except KeyError as exc:
        raise InputError(
            "variant record is missing a key field",
            details={"path": str(path), "line": line_no, "field": str(exc)},
        ) from exc
    except (TypeError, ValueError) as exc:
        raise InputError(
            "variant pos must be an integer",
            details={"path": str(path), "line": line_no, "pos": record.get("pos")},
        ) from exc
    return VariantKey(chrom=chrom, pos=pos, ref=ref, alt=alt)


def _clinical_label(record: dict[str, Any], *, path: Path, line_no: int) -> bool | None:
    raw = record.get("clinical_significance", record.get("label"))
    if not isinstance(raw, str) or not raw.strip():
        raise InputError(
            "label records require clinical_significance or label",
            details={"path": str(path), "line": line_no},
        )
    normalized = _normalize_labels(raw)
    has_positive = any(label in POSITIVE_CLINVAR_LABELS for label in normalized)
    has_negative = any(label in NEGATIVE_CLINVAR_LABELS for label in normalized)
    if has_positive and not has_negative:
        return True
    if has_negative and not has_positive:
        return False
    return None


def _normalize_labels(raw: str) -> tuple[str, ...]:
    value = raw.strip().upper().replace("%20", "_").replace(" ", "_").replace("-", "_")
    chunks = value.replace("%2F", "/").replace("|", "/").replace(",", "/").split("/")
    labels: list[str] = []
    for chunk in chunks:
        label = _LABEL_ALIASES.get(chunk.strip())
        if label is not None and label not in labels:
            labels.append(label)
    return tuple(labels)


def _require_finite_number(
    name: str,
    value: object,
    *,
    details: dict[str, object] | None = None,
) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise InputError(f"{name} must be a finite number", details=details)
    number = float(value)
    if not math.isfinite(number):
        raise InputError(f"{name} must be a finite number", details=details)
    return number


def _require_int(name: str, value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise InputError(f"{name} must be an integer", details={name: value})
    return value


def _require_non_negative_int(name: str, value: object) -> int:
    number = _require_int(name, value)
    if number < 0:
        raise InputError(f"{name} must be non-negative", details={name: number})
    return number


def _require_ci_level(value: object) -> float:
    level = _require_finite_number("ci_level", value)
    if level <= 0.0 or level >= 1.0:
        raise InputError(
            "ci_level must be greater than 0 and less than 1", details={"ci_level": level}
        )
    return level


def _required_text(name: str, value: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise InputError(f"{name} must be a non-empty string")
    return value.strip()


def _auroc(labels: list[bool], scores: list[float]) -> float:
    n_pos = sum(1 for label in labels if label)
    n_neg = len(labels) - n_pos
    ranked = sorted(enumerate(scores), key=lambda item: item[1])
    rank_sum_pos = 0.0
    rank = 1
    index = 0
    while index < len(ranked):
        end = index + 1
        while end < len(ranked) and ranked[end][1] == ranked[index][1]:
            end += 1
        avg_rank = (rank + rank + (end - index) - 1) / 2.0
        for original_index, _score in ranked[index:end]:
            if labels[original_index]:
                rank_sum_pos += avg_rank
        rank += end - index
        index = end
    return (rank_sum_pos - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)


def _average_precision(labels: list[bool], scores: list[float]) -> float:
    ordered = sorted(zip(scores, labels, strict=True), key=lambda item: item[0], reverse=True)
    positives = sum(1 for label in labels if label)
    seen_positive = 0
    precision_sum = 0.0
    for rank, (_score, label) in enumerate(ordered, start=1):
        if not label:
            continue
        seen_positive += 1
        precision_sum += seen_positive / rank
    return precision_sum / positives


def _spearman_rho(labels: list[float], scores: list[float]) -> float:
    if len(labels) != len(scores):
        raise InputError("spearman inputs must have the same length")
    if len(labels) < 2:
        raise InputError("spearman correlation requires at least two values")
    label_ranks = _average_ranks(labels)
    score_ranks = _average_ranks(scores)
    return _pearson(label_ranks, score_ranks)


def _average_ranks(values: list[float]) -> list[float]:
    ranked = sorted(enumerate(values), key=lambda item: item[1])
    ranks = [0.0] * len(values)
    rank = 1
    index = 0
    while index < len(ranked):
        end = index + 1
        while end < len(ranked) and ranked[end][1] == ranked[index][1]:
            end += 1
        avg_rank = (rank + rank + (end - index) - 1) / 2.0
        for original_index, _value in ranked[index:end]:
            ranks[original_index] = avg_rank
        rank += end - index
        index = end
    return ranks


def _pearson(left: list[float], right: list[float]) -> float:
    left_mean = sum(left) / len(left)
    right_mean = sum(right) / len(right)
    left_centered = [value - left_mean for value in left]
    right_centered = [value - right_mean for value in right]
    left_ss = sum(value * value for value in left_centered)
    right_ss = sum(value * value for value in right_centered)
    if left_ss <= 0.0 or right_ss <= 0.0:
        raise InputError("spearman correlation requires non-constant labels and scores")
    covariance = sum(
        left_value * right_value
        for left_value, right_value in zip(left_centered, right_centered, strict=True)
    )
    return covariance / math.sqrt(left_ss * right_ss)
