# SPDX-License-Identifier: Apache-2.0
"""Dependency-closed v0.2 metric requirements used by installed evaluation CLIs."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final

from geno_lewm._evaluation_report import EvalReportInput, MetricResult
from geno_lewm.errors import InputError

__all__ = [
    "BENCHMARK_REQUIREMENTS",
    "REQUIRED_METRICS",
    "ROLLOUT_FIDELITY_REQUIREMENTS",
    "ROLLOUT_SPLITS",
    "VEP_REQUIREMENTS",
    "VEP_SPLITS",
    "BenchmarkRequirement",
    "require_v02_rollout_benchmark_metrics",
    "require_v02_vep_benchmark_metrics",
    "v02_rollout_benchmark_metric_findings",
    "v02_vep_benchmark_metric_findings",
]


@dataclass(frozen=True, slots=True)
class BenchmarkRequirement:
    """One required v0.2 benchmark coverage row."""

    benchmark_id: str
    track: str
    split: str | None
    required_metrics: tuple[str, ...]
    issue_refs: tuple[int, ...]
    required_baseline: str | None = None
    require_confidence_intervals: bool = False


REQUIRED_METRICS: Final = ("auroc", "average_precision", "balanced_accuracy", "accuracy")
VEP_REQUIREMENTS: Final = (
    BenchmarkRequirement(
        "clinvar_coding",
        "variant_effect_prediction",
        "clinvar_coding",
        REQUIRED_METRICS,
        (53, 55, 56, 197),
        required_baseline="carbon_zero_shot",
        require_confidence_intervals=True,
    ),
    BenchmarkRequirement(
        "clinvar_noncoding",
        "variant_effect_prediction",
        "clinvar_noncoding",
        REQUIRED_METRICS,
        (53, 55, 56, 197),
        required_baseline="carbon_zero_shot",
        require_confidence_intervals=True,
    ),
    BenchmarkRequirement(
        "brca2_saturation",
        "variant_effect_prediction",
        "brca2",
        ("spearman_rho",),
        (56, 197),
        required_baseline="carbon_zero_shot",
        require_confidence_intervals=True,
    ),
    BenchmarkRequirement(
        "traitgym_mendelian",
        "variant_effect_prediction",
        "traitgym_mendelian",
        ("spearman_rho",),
        (56, 197),
        required_baseline="carbon_zero_shot",
        require_confidence_intervals=True,
    ),
)
ROLLOUT_FIDELITY_REQUIREMENTS: Final = (
    BenchmarkRequirement(
        "rollout_phased_haplotypes",
        "latent_rollout_fidelity",
        "rollout_phased_haplotypes",
        ("cosine_similarity_mean", "l2_distance_mean", "recall_at_k"),
        (42, 57, 197),
    ),
    BenchmarkRequirement(
        "rollout_synthetic_edit_chains",
        "latent_rollout_fidelity",
        "rollout_synthetic_edit_chains",
        ("cosine_similarity_mean", "l2_distance_mean", "recall_at_k"),
        (42, 57, 197),
    ),
)
BENCHMARK_REQUIREMENTS: Final = VEP_REQUIREMENTS + ROLLOUT_FIDELITY_REQUIREMENTS
ROLLOUT_SPLITS: Final = frozenset(
    requirement.split for requirement in ROLLOUT_FIDELITY_REQUIREMENTS if requirement.split
)
VEP_SPLITS: Final = frozenset(
    requirement.split for requirement in VEP_REQUIREMENTS if requirement.split
)


def v02_vep_benchmark_metric_findings(
    reports: tuple[EvalReportInput, ...],
) -> list[dict[str, object]]:
    """Return missing/incomplete v0.2 VEP metric rows for measured eval reports."""
    metric_rows = tuple(metric for report in reports for metric in report.metrics)
    findings: list[dict[str, object]] = []
    for requirement in VEP_REQUIREMENTS:
        row = _benchmark_row(requirement, metric_rows)
        if row["status"] == "pass":
            continue
        findings.append(
            {
                "benchmark_id": row["benchmark_id"],
                "split": row["split"],
                "status": row["status"],
                "missing_metrics": row["missing_metrics"],
                "missing_confidence_intervals": row["missing_confidence_intervals"],
                "required_baseline": row["required_baseline"],
                "baseline_observed": row["baseline_observed"],
                "issue_refs": row["issue_refs"],
            }
        )
    return findings


def require_v02_vep_benchmark_metrics(reports: tuple[EvalReportInput, ...]) -> None:
    """Require the measured VEP metric rows needed before a v0.2 readiness report."""
    findings = v02_vep_benchmark_metric_findings(reports)
    if findings:
        raise InputError(
            "v0.2 VEP benchmark metric coverage is incomplete",
            details={"findings": findings},
            remediation=(
                "aggregate measured coding/non-coding ClinVar, BRCA2 saturation, "
                "and TraitGym Mendelian metrics with Carbon baseline deltas, confidence "
                "intervals, and evaluated variant-key identities"
            ),
        )


def v02_rollout_benchmark_metric_findings(
    reports: tuple[EvalReportInput, ...],
) -> list[dict[str, object]]:
    """Return missing/incomplete v0.2 rollout-fidelity rows for measured eval reports."""
    metric_rows = tuple(metric for report in reports for metric in report.metrics)
    findings: list[dict[str, object]] = []
    for requirement in ROLLOUT_FIDELITY_REQUIREMENTS:
        row = _benchmark_row(requirement, metric_rows)
        if row["status"] == "pass":
            continue
        findings.append(
            {
                "benchmark_id": row["benchmark_id"],
                "split": row["split"],
                "status": row["status"],
                "missing_metrics": row["missing_metrics"],
                "issue_refs": row["issue_refs"],
            }
        )
    return findings


def require_v02_rollout_benchmark_metrics(reports: tuple[EvalReportInput, ...]) -> None:
    """Require the measured rollout-fidelity rows needed before a v0.2 readiness report."""
    findings = v02_rollout_benchmark_metric_findings(reports)
    if findings:
        raise InputError(
            "v0.2 rollout-fidelity metric coverage is incomplete",
            details={"findings": findings},
            remediation=(
                "aggregate measured rollout phased-haplotype and synthetic edit-chain "
                "metrics with cosine similarity, L2 distance, and Recall@k rows"
            ),
        )


def _benchmark_row(
    requirement: BenchmarkRequirement,
    metrics: tuple[MetricResult, ...],
) -> dict[str, object]:
    split_metrics = tuple(metric for metric in metrics if metric.split == requirement.split)
    observed_names = sorted({_normalized_metric_name(metric) for metric in split_metrics})
    missing_metrics = [name for name in requirement.required_metrics if name not in observed_names]
    required_metric_rows = tuple(
        metric
        for metric in split_metrics
        if _normalized_metric_name(metric) in requirement.required_metrics
    )
    missing_confidence_intervals: list[str] = []
    if requirement.require_confidence_intervals:
        missing_confidence_intervals = sorted(
            {
                _normalized_metric_name(metric)
                for metric in required_metric_rows
                if not _metric_has_confidence_interval(metric)
            }
        )
    baseline_missing = False
    if requirement.required_baseline is not None:
        baseline_missing = not required_metric_rows or not all(
            _metric_has_baseline(metric, requirement.required_baseline)
            for metric in required_metric_rows
        )
    if not split_metrics:
        status = "missing"
    elif missing_metrics or baseline_missing or missing_confidence_intervals:
        status = "incomplete"
    else:
        status = "pass"
    return {
        "benchmark_id": requirement.benchmark_id,
        "track": requirement.track,
        "split": requirement.split,
        "status": status,
        "required_metrics": list(requirement.required_metrics),
        "observed_metrics": observed_names,
        "observed_values": _observed_metric_values(required_metric_rows),
        "confidence_intervals": _observed_confidence_intervals(required_metric_rows),
        "evaluated_variant_key_identities": _observed_variant_key_identities(required_metric_rows),
        "missing_metrics": missing_metrics,
        "confidence_intervals_required": requirement.require_confidence_intervals,
        "missing_confidence_intervals": missing_confidence_intervals,
        "required_baseline": requirement.required_baseline,
        "baseline_values": _observed_baseline_values(required_metric_rows),
        "delta_vs_baseline": _observed_baseline_deltas(required_metric_rows),
        "baseline_observed": (
            None if requirement.required_baseline is None else not baseline_missing
        ),
        "issue_refs": [f"#{number}" for number in requirement.issue_refs],
    }


def _normalized_metric_name(metric: MetricResult) -> str:
    prefix = f"{metric.split}_"
    if metric.name.startswith(prefix):
        return metric.name.removeprefix(prefix)
    return metric.name


def _metric_has_baseline(metric: MetricResult, required_baseline: str) -> bool:
    return (
        metric.baseline == required_baseline
        and metric.baseline_value is not None
        and metric.delta_vs_baseline is not None
        and metric.evaluated_variant_keys_sha256 is not None
        and metric.baseline_evaluated_variant_keys_sha256 == metric.evaluated_variant_keys_sha256
    )


def _metric_has_confidence_interval(metric: MetricResult) -> bool:
    return metric.ci_low is not None and metric.ci_high is not None


def _observed_metric_values(metrics: tuple[MetricResult, ...]) -> dict[str, float]:
    return {_normalized_metric_name(metric): metric.value for metric in metrics}


def _observed_confidence_intervals(
    metrics: tuple[MetricResult, ...],
) -> dict[str, dict[str, float]]:
    return {
        _normalized_metric_name(metric): {"ci_low": metric.ci_low, "ci_high": metric.ci_high}
        for metric in metrics
        if metric.ci_low is not None and metric.ci_high is not None
    }


def _observed_variant_key_identities(metrics: tuple[MetricResult, ...]) -> dict[str, str]:
    return {
        _normalized_metric_name(metric): metric.evaluated_variant_keys_sha256
        for metric in metrics
        if metric.evaluated_variant_keys_sha256 is not None
    }


def _observed_baseline_values(metrics: tuple[MetricResult, ...]) -> dict[str, float]:
    return {
        _normalized_metric_name(metric): metric.baseline_value
        for metric in metrics
        if metric.baseline_value is not None
    }


def _observed_baseline_deltas(metrics: tuple[MetricResult, ...]) -> dict[str, float]:
    return {
        _normalized_metric_name(metric): metric.delta_vs_baseline
        for metric in metrics
        if metric.delta_vs_baseline is not None
    }
