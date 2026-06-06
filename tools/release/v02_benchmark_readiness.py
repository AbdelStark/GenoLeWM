# SPDX-License-Identifier: Apache-2.0
"""Generate the v0.2 benchmark-readiness report.

This tool does not run private-data benchmarks and does not turn fixture
smokes into model-quality claims. It validates measured artifact inputs
that already exist, enumerates the RFC-0007/#197 suite coverage, and
records missing or failed rows explicitly.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Final

from geno_lewm.errors import GenoLeWMError, InputError, exit_code_for
from geno_lewm.provenance import sha256_file
from tools.release.efficiency_report import EfficiencyReport, load_efficiency_report
from tools.release.eval_report import EvalReportInput, MetricResult, load_report_input
from tools.release.rollout_speed_scope import (
    DECISION as ROLLOUT_SPEED_SCOPE_DECISION,
    GENERATED_BY as ROLLOUT_SPEED_SCOPE_GENERATED_BY,
    SCHEMA_VERSION as ROLLOUT_SPEED_SCOPE_SCHEMA_VERSION,
    STATUS as ROLLOUT_SPEED_SCOPE_STATUS,
)

SCHEMA_VERSION: Final = "1.0.0"
GENERATED_BY: Final = "tools.release.v02_benchmark_readiness"
ROLLOUT_GENERATED_BY: Final = "bench.rollout"
ROLLOUT_SCHEMA_VERSION: Final = "1.0.0"
ROLLOUT_SPEED_REQUIRED_METRICS: Final = ("k5_speedup", "k20_speedup")
RELEASE_INPUT_PLACEHOLDER_RE: Final = re.compile(
    r"\b(?:fixture|synthetic|readiness|placeholder|dummy|mock|fake|test)\b",
    re.IGNORECASE,
)
RELEASE_ARTIFACT_PLACEHOLDER_RE: Final = re.compile(
    r"\b(?:fixture|placeholder|dummy|mock|fake|test)\b",
    re.IGNORECASE,
)
COMMIT_RE: Final = re.compile(r"^[0-9a-fA-F]{7,40}$")
COMMAND_PATH_FLAGS: Final = frozenset(
    {
        "--metrics-json",
        "--rollout-speed-report",
        "--rollout-speed-scope-report",
        "--efficiency-report",
        "--output-json",
        "--out-dir",
        "--output",
    }
)


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


def build_readiness_report(
    *,
    metrics_json: tuple[Path, ...] = (),
    rollout_speed_report: Path | None = None,
    rollout_speed_scope_report: Path | None = None,
    efficiency_report: Path | None = None,
    command: tuple[str, ...] = (),
    require_release_inputs: bool = False,
) -> dict[str, object]:
    """Build a machine-readable v0.2 benchmark-readiness report."""
    metric_reports = tuple(load_report_input(path) for path in metrics_json)
    efficiency = (
        load_efficiency_report(efficiency_report) if efficiency_report is not None else None
    )
    _require_shared_identity(metric_reports)
    _require_efficiency_identity(metric_reports, efficiency)
    identity = _identity(metric_reports)
    metric_rows = tuple(metric for report in metric_reports for metric in report.metrics)
    rows = [_benchmark_row(requirement, metric_rows) for requirement in BENCHMARK_REQUIREMENTS]
    rows.append(_efficiency_row(efficiency))
    rows.append(
        _rollout_speed_row(
            rollout_speed_report,
            rollout_speed_scope_report=rollout_speed_scope_report,
            expected_commit=identity.get("commit"),
        )
    )
    if require_release_inputs:
        rows.append(
            _release_inputs_row(
                metric_reports=metric_reports,
                rollout_speed_report=rollout_speed_report,
                efficiency=efficiency,
            )
        )
    missing_or_failed = [
        str(row["benchmark_id"])
        for row in rows
        if str(row.get("status")) not in {"pass", "rescoped"}
    ]
    scope_decisions = _scope_decisions(rows)
    ok = not missing_or_failed
    artifact_inputs = _artifact_inputs(
        metrics_json=metrics_json,
        rollout_speed_report=rollout_speed_report,
        rollout_speed_scope_report=rollout_speed_scope_report,
        efficiency_report=efficiency_report,
    )
    return {
        "schema_version": SCHEMA_VERSION,
        "generated_by": GENERATED_BY,
        "generated_at": _utc_now(),
        "ok": ok,
        "model_id": identity.get("model_id"),
        "model_release": identity.get("model_release"),
        "dataset_snapshot": identity.get("dataset_snapshot"),
        "commit": identity.get("commit"),
        "hardware": identity.get("hardware"),
        "command": _public_safe_command(command),
        "release_inputs_required": require_release_inputs,
        "inputs": artifact_inputs,
        "benchmark_rows": rows,
        "scope_decisions": scope_decisions,
        "missing_or_failed_benchmarks": missing_or_failed,
        "metric_conclusions": _metric_conclusions(rows),
        "negative_findings": _negative_findings(
            missing_or_failed,
            scope_decisions=scope_decisions,
        ),
        "claim_boundary": (
            "This report validates benchmark coverage and artifact provenance only; it is not "
            "clinical, privacy, deployment, or model-quality evidence beyond the measured inputs."
        ),
    }


def write_readiness_report(
    *,
    output: Path,
    metrics_json: tuple[Path, ...] = (),
    rollout_speed_report: Path | None = None,
    rollout_speed_scope_report: Path | None = None,
    efficiency_report: Path | None = None,
    command: tuple[str, ...] = (),
    require_release_inputs: bool = False,
) -> dict[str, object]:
    """Build and write the v0.2 benchmark-readiness report."""
    report = build_readiness_report(
        metrics_json=metrics_json,
        rollout_speed_report=rollout_speed_report,
        rollout_speed_scope_report=rollout_speed_scope_report,
        efficiency_report=efficiency_report,
        command=command,
        require_release_inputs=require_release_inputs,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return report


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


def main(argv: list[str] | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    command = _command_from_args(args)
    try:
        report = write_readiness_report(
            output=args.output,
            metrics_json=tuple(args.metrics_json or ()),
            rollout_speed_report=args.rollout_speed_report,
            rollout_speed_scope_report=args.rollout_speed_scope_report,
            efficiency_report=args.efficiency_report,
            command=command,
            require_release_inputs=args.require_release_inputs or args.require_ok,
        )
    except GenoLeWMError as exc:
        sys.stderr.write(f"error: {exc}\n")
        return exit_code_for(exc)
    sys.stdout.write(f"wrote {args.output}\n")
    if args.require_ok and not bool(report["ok"]):
        return 1
    return 0


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


def _efficiency_row(report: EfficiencyReport | None) -> dict[str, object]:
    if report is None:
        return {
            "benchmark_id": "inference_efficiency",
            "track": "inference_efficiency",
            "status": "missing",
            "required_metrics": [
                "single_variant_latency_ms",
                "batched_throughput_variants_per_s",
                "peak_memory_bytes",
            ],
            "observed_metrics": [],
            "issue_refs": ["#56", "#58", "#197"],
        }
    measurements = report.measurements.to_dict()
    return {
        "benchmark_id": "inference_efficiency",
        "track": "inference_efficiency",
        "status": "pass",
        "required_metrics": [
            "single_variant_latency_ms",
            "batched_throughput_variants_per_s",
            "peak_memory_bytes",
        ],
        "observed_metrics": list(measurements),
        "observed_values": measurements,
        "model_id": report.model_id,
        "model_release": report.model_release,
        "dataset_snapshot": report.dataset_snapshot,
        "commit": report.commit,
        "hardware": report.hardware,
        "runtime": report.runtime,
        "command": _public_safe_command(tuple(report.command)),
        "warmup_batches": report.warmup_batches,
        "samples": report.samples,
        "limitations": list(report.limitations),
        "issue_refs": ["#56", "#58", "#197"],
    }


def _rollout_speed_row(
    path: Path | None,
    *,
    rollout_speed_scope_report: Path | None,
    expected_commit: str | None,
) -> dict[str, object]:
    if path is None and rollout_speed_scope_report is not None:
        raise InputError("rollout speed scope report requires a rollout speed report")
    if path is None:
        return {
            "benchmark_id": "ar_rollout_speed",
            "track": "rollout_performance",
            "status": "missing",
            "required_metrics": list(ROLLOUT_SPEED_REQUIRED_METRICS),
            "observed_metrics": [],
            "issue_refs": ["#42", "#197"],
        }
    payload = _load_json(path, label="rollout speed report")
    schema_version = payload.get("schema_version")
    if schema_version != ROLLOUT_SCHEMA_VERSION:
        raise InputError(
            "rollout speed report schema_version is invalid",
            details={"expected": ROLLOUT_SCHEMA_VERSION, "observed": schema_version},
        )
    generated_by = payload.get("generated_by")
    if generated_by != ROLLOUT_GENERATED_BY:
        raise InputError(
            "rollout speed report generated_by is invalid",
            details={"expected": ROLLOUT_GENERATED_BY, "observed": generated_by},
        )
    commit = _required_text(payload, "commit")
    if expected_commit is not None and commit != expected_commit:
        raise InputError(
            "rollout speed report commit does not match metrics release identity",
            details={"metrics_commit": expected_commit, "rollout_commit": commit},
        )
    command = _required_text_list(payload.get("command"), "rollout speed command")
    report_ok = _required_bool(payload, "ok")
    rows = _require_list(payload.get("rows"), "rollout speed rows")
    observed: dict[str, float] = {}
    failed: list[str] = []
    failed_target_rows: list[dict[str, object]] = []
    for row in rows:
        if not isinstance(row, dict):
            raise InputError("rollout speed rows must be objects")
        horizon = _required_int(row, "horizon")
        speedup = _required_number(row, "measured_speedup")
        target = _required_number(row, "target_speedup")
        observed[f"k{horizon}_speedup"] = speedup
        if not _required_bool(row, "target_met"):
            failed.append(f"K={horizon}: {speedup:.6g}x < {target:.6g}x")
            failed_target_rows.append(
                {
                    "horizon": horizon,
                    "measured_speedup": speedup,
                    "target_speedup": target,
                    "shortfall": max(0.0, target - speedup),
                }
            )
    missing_metrics = [
        metric for metric in ROLLOUT_SPEED_REQUIRED_METRICS if metric not in observed
    ]
    if not report_ok:
        failed.append("report ok=false")
    if failed:
        status = "failed"
    elif missing_metrics:
        status = "incomplete"
    else:
        status = "pass"
    if rollout_speed_scope_report is not None and status != "failed":
        raise InputError(
            "rollout speed scope report is only valid for failed rollout speed targets",
            details={"rollout_speed_status": status},
        )
    scope_decision = None
    if rollout_speed_scope_report is not None:
        scope_decision = _load_rollout_speed_scope_decision(
            rollout_speed_scope_report,
            rollout_speed_report=path,
            rollout_commit=commit,
            rollout_command=command,
            rollout_report_ok=report_ok,
            observed_values=observed,
            failed_targets=failed_target_rows,
        )
        status = "rescoped"
    row_payload: dict[str, object] = {
        "benchmark_id": "ar_rollout_speed",
        "track": "rollout_performance",
        "status": status,
        "required_metrics": list(ROLLOUT_SPEED_REQUIRED_METRICS),
        "observed_metrics": sorted(observed),
        "observed_values": observed,
        "missing_metrics": missing_metrics,
        "failed_targets": failed,
        "commit": commit,
        "command": _public_safe_command(tuple(command)),
        "issue_refs": ["#42", "#197"],
    }
    if scope_decision is not None:
        row_payload["scope_decision"] = scope_decision
    return row_payload


def _load_rollout_speed_scope_decision(
    path: Path,
    *,
    rollout_speed_report: Path,
    rollout_commit: str,
    rollout_command: list[str],
    rollout_report_ok: bool,
    observed_values: dict[str, float],
    failed_targets: list[dict[str, object]],
) -> dict[str, object]:
    payload = _load_json(path, label="rollout speed scope report")
    schema_version = payload.get("schema_version")
    if schema_version != ROLLOUT_SPEED_SCOPE_SCHEMA_VERSION:
        raise InputError(
            "rollout speed scope report schema_version is invalid",
            details={
                "expected": ROLLOUT_SPEED_SCOPE_SCHEMA_VERSION,
                "observed": schema_version,
            },
        )
    generated_by = payload.get("generated_by")
    if generated_by != ROLLOUT_SPEED_SCOPE_GENERATED_BY:
        raise InputError(
            "rollout speed scope report generated_by is invalid",
            details={
                "expected": ROLLOUT_SPEED_SCOPE_GENERATED_BY,
                "observed": generated_by,
            },
        )
    if payload.get("ok") is not True:
        raise InputError("rollout speed scope report must have ok=true")
    status = _required_text(payload, "status")
    if status != ROLLOUT_SPEED_SCOPE_STATUS:
        raise InputError(
            "rollout speed scope report status is invalid",
            details={"expected": ROLLOUT_SPEED_SCOPE_STATUS, "observed": status},
        )
    decision = _required_text(payload, "decision")
    if decision != ROLLOUT_SPEED_SCOPE_DECISION:
        raise InputError(
            "rollout speed scope report decision is invalid",
            details={"expected": ROLLOUT_SPEED_SCOPE_DECISION, "observed": decision},
        )
    issue_refs = _required_issue_refs(payload.get("issue_refs"))
    scope_command = _required_text_list(payload.get("command"), "rollout speed scope command")
    expected_scope_command = _public_safe_command(tuple(scope_command))
    if scope_command != expected_scope_command:
        raise InputError(
            "rollout speed scope report command must be public-safe",
            details={"expected": expected_scope_command, "observed": scope_command},
        )
    raw_identity = payload.get("rollout_speed_report")
    if not isinstance(raw_identity, dict):
        raise InputError("rollout speed scope report must bind rollout_speed_report")
    expected_identity = _file_identity(rollout_speed_report)
    identity_mismatches = {
        key: {"expected": expected_identity[key], "observed": raw_identity.get(key)}
        for key in ("path", "sha256", "size_bytes")
        if raw_identity.get(key) != expected_identity[key]
    }
    if identity_mismatches:
        raise InputError(
            "rollout speed scope report does not match rollout speed report",
            details={"mismatches": identity_mismatches},
        )
    raw_summary = payload.get("rollout_speed_summary")
    if not isinstance(raw_summary, dict):
        raise InputError("rollout speed scope report must bind rollout_speed_summary")
    summary_mismatches: dict[str, object] = {}
    expected_rollout_command = _public_safe_command(tuple(rollout_command))
    observed_rollout_command = _required_text_list(
        raw_summary.get("command"), "rollout speed scope summary command"
    )
    if observed_rollout_command != expected_rollout_command:
        summary_mismatches["command"] = {
            "expected": expected_rollout_command,
            "observed": observed_rollout_command,
        }
    if raw_summary.get("commit") != rollout_commit:
        summary_mismatches["commit"] = {
            "expected": rollout_commit,
            "observed": raw_summary.get("commit"),
        }
    if raw_summary.get("report_ok") != rollout_report_ok:
        summary_mismatches["report_ok"] = {
            "expected": rollout_report_ok,
            "observed": raw_summary.get("report_ok"),
        }
    if raw_summary.get("observed_values") != observed_values:
        summary_mismatches["observed_values"] = {
            "expected": observed_values,
            "observed": raw_summary.get("observed_values"),
        }
    if raw_summary.get("failed_targets") != failed_targets:
        summary_mismatches["failed_targets"] = {
            "expected": failed_targets,
            "observed": raw_summary.get("failed_targets"),
        }
    if summary_mismatches:
        raise InputError(
            "rollout speed scope report summary is stale",
            details={"mismatches": summary_mismatches},
        )
    _require_scope_negative_findings(payload.get("negative_findings"))
    _require_scope_claim_boundary(payload.get("claim_boundary"))
    return {
        "report": _file_identity(path),
        "decision": decision,
        "status": status,
        "generated_at": _require_utc_timestamp(payload.get("generated_at"), "generated_at"),
        "accepted_by": _required_text(payload, "accepted_by"),
        "accepted_at": _require_utc_timestamp(payload.get("accepted_at"), "accepted_at"),
        "decision_url": _require_url(payload.get("decision_url"), "decision_url"),
        "rationale": _required_text(payload, "rationale"),
        "replacement_target": _required_text(payload, "replacement_target"),
        "issue_refs": issue_refs,
    }


def _release_inputs_row(
    *,
    metric_reports: tuple[EvalReportInput, ...],
    rollout_speed_report: Path | None,
    efficiency: EfficiencyReport | None,
) -> dict[str, object]:
    findings: list[str] = []
    if not metric_reports:
        findings.append("at least one measured eval metrics JSON artifact is required")
    if rollout_speed_report is None:
        findings.append("a bench.rollout speed report is required")
    if efficiency is None:
        findings.append("an efficiency_report.json artifact is required")
    for index, report in enumerate(metric_reports, start=1):
        findings.extend(_metric_release_input_findings(report, input_index=index))
    if efficiency is not None:
        findings.extend(_efficiency_release_input_findings(efficiency))
    if findings:
        status = (
            "missing"
            if not metric_reports or rollout_speed_report is None or efficiency is None
            else "failed"
        )
    else:
        status = "pass"
    return {
        "benchmark_id": "release_inputs",
        "track": "artifact_provenance",
        "status": status,
        "required_metrics": [
            "package_relative_artifacts",
            "score_or_metrics_input_provenance",
            "non_fixture_release_identity",
            "efficiency_input_identities",
        ],
        "observed_metrics": []
        if findings
        else [
            "package_relative_artifacts",
            "score_or_metrics_input_provenance",
            "non_fixture_release_identity",
            "efficiency_input_identities",
        ],
        "findings": findings,
        "issue_refs": ["#56", "#197"],
    }


def _metric_release_input_findings(
    report: EvalReportInput,
    *,
    input_index: int,
) -> list[str]:
    prefix = f"metrics_json[{input_index}]"
    findings: list[str] = []
    text_fields = {
        "model_release": report.model_release,
        "dataset_snapshot": report.dataset_snapshot,
        "hardware": report.hardware,
    }
    if COMMIT_RE.fullmatch(report.commit) is None:
        findings.append(f"{prefix}.commit must be a 7-40 character hexadecimal SHA")
    findings.extend(_placeholder_findings(text_fields, prefix=prefix))
    artifacts = dict(report.artifacts)
    findings.extend(_artifact_findings(artifacts, prefix=f"{prefix}.artifacts"))
    artifact_keys = set(artifacts)
    has_raw_score_inputs = _has_artifact_key(artifact_keys, "scores") and _has_artifact_key(
        artifact_keys, "labels"
    )
    has_rollout_state_inputs = _has_artifact_key(artifact_keys, "rollout_states")
    has_aggregate_inputs = any(key.startswith("metrics_input_") for key in artifact_keys)
    if not has_raw_score_inputs and not has_rollout_state_inputs and not has_aggregate_inputs:
        findings.append(
            f"{prefix}.artifacts must include scores+labels, rollout_states, or metrics_input_* provenance"
        )
    if _report_has_vep_metrics(report) and not has_raw_score_inputs:
        findings.append(
            f"{prefix}.artifacts must include score and label provenance for VEP metrics"
        )
    if _report_has_rollout_metrics(report):
        required_rollout_artifacts = (
            "rollout_states",
            "baseline_rollout_states",
            "rollout_state_examples_report",
            "rollout_state_rows_report",
        )
        missing = [
            key for key in required_rollout_artifacts if not _has_artifact_key(artifact_keys, key)
        ]
        if missing:
            findings.append(
                f"{prefix}.artifacts must include rollout generation provenance: {', '.join(missing)}"
            )
    has_baseline_metrics = any(metric.baseline is not None for metric in report.metrics)
    has_baseline_artifact = any(_is_baseline_artifact_key(key) for key in artifact_keys)
    if has_baseline_metrics and not has_baseline_artifact:
        findings.append(f"{prefix}.artifacts must include baseline artifact provenance")
    return findings


def _report_has_rollout_metrics(report: EvalReportInput) -> bool:
    return any(metric.split in ROLLOUT_SPLITS for metric in report.metrics)


def _report_has_vep_metrics(report: EvalReportInput) -> bool:
    return any(metric.split in VEP_SPLITS for metric in report.metrics)


def _has_artifact_key(keys: set[str], name: str) -> bool:
    return name in keys or any(key.endswith(f".{name}") for key in keys)


def _is_baseline_artifact_key(key: str) -> bool:
    return key in {"baseline_scores", "baseline_rollout_states"} or key.endswith(
        (".baseline_scores", ".baseline_rollout_states")
    )


def _efficiency_release_input_findings(report: EfficiencyReport) -> list[str]:
    findings: list[str] = []
    if COMMIT_RE.fullmatch(report.commit) is None:
        findings.append("efficiency_report.commit must be a 7-40 character hexadecimal SHA")
    findings.extend(
        _placeholder_findings(
            {
                "model_release": report.model_release,
                "dataset_snapshot": report.dataset_snapshot,
                "hardware": report.hardware,
                "runtime": report.runtime,
            },
            prefix="efficiency_report",
        )
    )
    for key, identity in report.inputs:
        findings.extend(
            _path_findings(
                identity.path,
                field=f"efficiency_report.inputs.{key}.path",
                allow_inline=True,
            )
        )
        if RELEASE_ARTIFACT_PLACEHOLDER_RE.search(identity.path):
            findings.append(
                f"efficiency_report.inputs.{key}.path must not reference fixture/test artifacts"
            )
    return findings


def _artifact_findings(artifacts: dict[str, str], *, prefix: str) -> list[str]:
    findings: list[str] = []
    for key, value in sorted(artifacts.items()):
        field = f"{prefix}.{key}"
        findings.extend(_path_findings(value, field=field, allow_inline=False))
        if RELEASE_ARTIFACT_PLACEHOLDER_RE.search(value):
            findings.append(f"{field} must not reference fixture/test artifacts")
    return findings


def _path_findings(value: str, *, field: str, allow_inline: bool) -> list[str]:
    if allow_inline and value.startswith("inline:"):
        label = value.removeprefix("inline:")
        if not label or "/" in label or "\\" in label or label in {".", ".."}:
            return [f"{field} must use inline:<label> when inline paths are used"]
        return []
    if "://" in value:
        return [f"{field} must be package-relative"]
    if PurePosixPath(value).is_absolute() or PureWindowsPath(value).is_absolute() or "\\" in value:
        return [f"{field} must be package-relative"]
    candidate = Path(value)
    if ".." in candidate.parts or not candidate.parts:
        return [f"{field} must be package-relative"]
    return []


def _placeholder_findings(values: dict[str, str], *, prefix: str) -> list[str]:
    return [
        f"{prefix}.{field} must not look like fixture/test/readiness evidence"
        for field, value in sorted(values.items())
        if RELEASE_INPUT_PLACEHOLDER_RE.search(value)
    ]


def _require_shared_identity(reports: tuple[EvalReportInput, ...]) -> None:
    if not reports:
        return
    first = reports[0]
    for index, report in enumerate(reports[1:], start=2):
        mismatches = {
            "model_id": (first.model_id, report.model_id),
            "model_release": (first.model_release, report.model_release),
            "dataset_snapshot": (first.dataset_snapshot, report.dataset_snapshot),
            "commit": (first.commit, report.commit),
            "hardware": (first.hardware, report.hardware),
        }
        observed = {
            key: {"first": expected, "observed": value}
            for key, (expected, value) in mismatches.items()
            if expected != value
        }
        if observed:
            raise InputError(
                "v0.2 readiness metrics JSON artifacts do not share release identity",
                details={"input_index": index, "mismatches": observed},
            )


def _require_efficiency_identity(
    reports: tuple[EvalReportInput, ...],
    efficiency: EfficiencyReport | None,
) -> None:
    if not reports or efficiency is None:
        return
    first = reports[0]
    mismatches = {
        "model_id": (first.model_id, efficiency.model_id),
        "model_release": (first.model_release, efficiency.model_release),
        "dataset_snapshot": (first.dataset_snapshot, efficiency.dataset_snapshot),
        "commit": (first.commit, efficiency.commit),
        "hardware": (first.hardware, efficiency.hardware),
    }
    observed = {
        key: {"metrics_json": expected, "efficiency_report": value}
        for key, (expected, value) in mismatches.items()
        if expected != value
    }
    if observed:
        raise InputError(
            "v0.2 readiness efficiency report does not match metrics release identity",
            details={"mismatches": observed},
        )


def _identity(reports: tuple[EvalReportInput, ...]) -> dict[str, str | None]:
    if not reports:
        return {
            "model_id": None,
            "model_release": None,
            "dataset_snapshot": None,
            "commit": None,
            "hardware": None,
        }
    first = reports[0]
    return {
        "model_id": first.model_id,
        "model_release": first.model_release,
        "dataset_snapshot": first.dataset_snapshot,
        "commit": first.commit,
        "hardware": first.hardware,
    }


def _artifact_inputs(
    *,
    metrics_json: tuple[Path, ...],
    rollout_speed_report: Path | None,
    rollout_speed_scope_report: Path | None,
    efficiency_report: Path | None,
) -> dict[str, object]:
    inputs: dict[str, object] = {
        "metrics_json": [_file_identity(path) for path in metrics_json],
    }
    if rollout_speed_report is not None:
        inputs["rollout_speed_report"] = _file_identity(rollout_speed_report)
    if rollout_speed_scope_report is not None:
        inputs["rollout_speed_scope_report"] = _file_identity(rollout_speed_scope_report)
    if efficiency_report is not None:
        inputs["efficiency_report"] = _file_identity(efficiency_report)
    return inputs


def _file_identity(path: Path) -> dict[str, object]:
    if not path.is_file():
        raise InputError("readiness input artifact does not exist", details={"path": str(path)})
    return {
        "path": _public_safe_identity_path(path),
        "sha256": sha256_file(path),
        "size_bytes": path.stat().st_size,
    }


def _public_safe_identity_path(path: Path) -> str:
    return _public_safe_path_text(str(path))


def _public_safe_path_text(value: str) -> str:
    if "://" in value:
        return value
    posix_path = PurePosixPath(value)
    if posix_path.is_absolute():
        return posix_path.name
    windows_path = PureWindowsPath(value)
    if windows_path.is_absolute():
        return windows_path.name
    return value


def _public_safe_command(command: tuple[str, ...]) -> list[str]:
    result: list[str] = []
    sanitize_next = False
    for token in command:
        if sanitize_next:
            result.append(_public_safe_path_text(token))
            sanitize_next = False
            continue
        result.append(_public_safe_path_text(token))
        sanitize_next = token in COMMAND_PATH_FLAGS
    return result


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


def _metric_conclusions(rows: list[dict[str, object]]) -> list[str]:
    conclusions: list[str] = []
    for row in rows:
        benchmark = str(row["benchmark_id"])
        status = str(row["status"])
        if status == "pass":
            context = _format_row_context(row)
            values = _format_metric_values(row.get("observed_values"))
            deltas = _format_metric_values(row.get("delta_vs_baseline"))
            intervals = _format_confidence_intervals(row.get("confidence_intervals"))
            variant_identities = _format_text_values(row.get("evaluated_variant_key_identities"))
            conclusion = f"{benchmark} passed with measured artifact coverage"
            if context:
                conclusion += f" ({context})"
            conclusion += f": {values}." if values else "."
            if deltas:
                conclusion += f" Baseline deltas: {deltas}."
            if intervals:
                conclusion += f" Confidence intervals: {intervals}."
            if variant_identities:
                conclusion += f" Evaluated variant-key identities: {variant_identities}."
            conclusions.append(conclusion)
        elif status == "rescoped":
            context = _format_row_context(row)
            values = _format_metric_values(row.get("observed_values"))
            failed_targets = _format_text_list(row.get("failed_targets"))
            scope_decision = _format_scope_decision(row.get("scope_decision"))
            conclusion = f"{benchmark} was explicitly rescoped after failed measured targets"
            if context:
                conclusion += f" ({context})"
            conclusion += f": {values}." if values else "."
            if failed_targets:
                conclusion += f" Failed targets: {failed_targets}."
            if scope_decision:
                conclusion += f" Accepted re-scope: {scope_decision}."
            conclusions.append(conclusion)
        else:
            raw_issue_refs = row.get("issue_refs")
            issue_refs = (
                ", ".join(raw_issue_refs)
                if isinstance(raw_issue_refs, list)
                and all(isinstance(ref, str) for ref in raw_issue_refs)
                else ""
            )
            conclusions.append(
                f"{benchmark} is {status}; route remaining work through {issue_refs}."
            )
    return conclusions


def _scope_decisions(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    decisions: list[dict[str, object]] = []
    for row in rows:
        if row.get("status") != "rescoped":
            continue
        raw = row.get("scope_decision")
        if isinstance(raw, dict):
            decisions.append(
                {
                    "benchmark_id": row.get("benchmark_id"),
                    "report": raw.get("report"),
                    "decision": raw.get("decision"),
                    "status": raw.get("status"),
                    "generated_at": raw.get("generated_at"),
                    "accepted_by": raw.get("accepted_by"),
                    "accepted_at": raw.get("accepted_at"),
                    "decision_url": raw.get("decision_url"),
                    "rationale": raw.get("rationale"),
                    "replacement_target": raw.get("replacement_target"),
                    "issue_refs": raw.get("issue_refs"),
                }
            )
    return decisions


def _format_row_context(row: dict[str, object]) -> str:
    parts: list[str] = []
    track = row.get("track")
    split = row.get("split")
    if isinstance(track, str) and track:
        parts.append(f"track={track}")
    if isinstance(split, str) and split:
        parts.append(f"split={split}")
    return ", ".join(parts)


def _format_metric_values(raw: object) -> str:
    if not isinstance(raw, dict) or not raw:
        return ""
    items: list[str] = []
    for key in sorted(raw):
        value = raw[key]
        if isinstance(value, bool) or not isinstance(value, int | float):
            continue
        items.append(f"{key}={float(value):.6g}")
    return ", ".join(items)


def _format_confidence_intervals(raw: object) -> str:
    if not isinstance(raw, dict) or not raw:
        return ""
    items: list[str] = []
    for key in sorted(raw):
        value = raw[key]
        if not isinstance(value, dict):
            continue
        low = value.get("ci_low")
        high = value.get("ci_high")
        if (
            isinstance(low, bool)
            or isinstance(high, bool)
            or not isinstance(low, int | float)
            or not isinstance(high, int | float)
        ):
            continue
        items.append(f"{key}=[{float(low):.6g},{float(high):.6g}]")
    return ", ".join(items)


def _format_text_values(raw: object) -> str:
    if not isinstance(raw, dict) or not raw:
        return ""
    items: list[str] = []
    for key in sorted(raw):
        value = raw[key]
        if isinstance(value, str) and value:
            items.append(f"{key}={value}")
    return ", ".join(items)


def _format_text_list(raw: object) -> str:
    if not isinstance(raw, list) or not raw:
        return ""
    values = [value for value in raw if isinstance(value, str) and value]
    return "; ".join(values)


def _format_scope_decision(raw: object) -> str:
    if not isinstance(raw, dict) or not raw:
        return ""
    parts: list[str] = []
    for key in (
        "decision",
        "accepted_by",
        "accepted_at",
        "decision_url",
        "rationale",
        "replacement_target",
    ):
        value = raw.get(key)
        if isinstance(value, str) and value:
            parts.append(f"{key}={value}")
    issue_refs = raw.get("issue_refs")
    if isinstance(issue_refs, list) and issue_refs:
        refs = [ref for ref in issue_refs if isinstance(ref, str) and ref]
        if refs:
            parts.append(f"issue_refs={','.join(refs)}")
    return "; ".join(parts)


def _negative_findings(
    missing_or_failed: list[str],
    *,
    scope_decisions: list[dict[str, object]],
) -> list[str]:
    if not missing_or_failed:
        findings = [
            (
                "No clinical utility, privacy, runtime-assurance, or deployment claim is established "
                "by benchmark coverage alone."
            )
        ]
        if scope_decisions:
            findings.append(
                "At least one benchmark target was explicitly rescoped; original failed "
                "measurements remain recorded and are not passing speed evidence."
            )
        return findings
    return [
        (
            "The v0.2 benchmark suite is incomplete or below target for: "
            + ", ".join(missing_or_failed)
            + "."
        )
    ]


def _load_json(path: Path, *, label: str) -> dict[str, object]:
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise InputError(f"failed to read {label}", details={"path": str(path)}) from exc
    except json.JSONDecodeError as exc:
        raise InputError(
            f"{label} JSON is invalid",
            details={"path": str(path), "line": exc.lineno, "column": exc.colno},
        ) from exc
    if not isinstance(raw, dict):
        raise InputError(f"{label} payload must be a JSON object")
    return raw


def _require_list(raw: object, label: str) -> list[object]:
    if not isinstance(raw, list):
        raise InputError(f"{label} must be a JSON list")
    return raw


def _required_int(raw: dict[str, object], field: str) -> int:
    value = raw.get(field)
    if isinstance(value, bool) or not isinstance(value, int):
        raise InputError(f"{field} must be an integer")
    return value


def _required_number(raw: dict[str, object], field: str) -> float:
    value = raw.get(field)
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise InputError(f"{field} must be a number")
    return float(value)


def _required_text(raw: dict[str, object], field: str) -> str:
    value = raw.get(field)
    if not isinstance(value, str) or not value.strip():
        raise InputError(f"{field} must be a non-empty string")
    return value.strip()


def _require_text_value(value: object, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise InputError(f"{field} must be a non-empty string")
    return value.strip()


def _require_url(value: object, field: str) -> str:
    text = _require_text_value(value, field)
    if not text.startswith(("https://", "http://")):
        raise InputError(f"{field} must be an HTTP(S) URL")
    return text


def _require_utc_timestamp(value: object, field: str) -> str:
    text = _require_text_value(value, field)
    if not text.endswith("Z"):
        raise InputError(f"{field} must be a UTC timestamp ending in Z")
    try:
        datetime.fromisoformat(text.removesuffix("Z") + "+00:00")
    except ValueError as exc:
        raise InputError(f"{field} must be an ISO-8601 UTC timestamp") from exc
    return text


def _required_issue_refs(raw: object) -> list[str]:
    if not isinstance(raw, list):
        raise InputError("rollout speed scope report issue_refs must be a JSON list")
    refs: list[str] = []
    for ref in raw:
        if not isinstance(ref, str) or not ref.startswith("#") or not ref[1:].isdigit():
            raise InputError("rollout speed scope report issue_refs must be GitHub issue refs")
        refs.append(ref)
    if "#42" not in refs or "#197" not in refs:
        raise InputError("rollout speed scope report issue_refs must include #42 and #197")
    return refs


def _require_scope_negative_findings(raw: object) -> None:
    findings = _required_text_list(raw, "rollout speed scope report negative_findings")
    text = " ".join(findings).lower()
    if "not met" not in text or "not rollout-speed evidence" not in text:
        raise InputError(
            "rollout speed scope report negative_findings must preserve failed-target boundaries"
        )


def _require_scope_claim_boundary(raw: object) -> None:
    text = _require_text_value(raw, "claim_boundary")
    normalized = text.lower()
    if (
        "does not establish" not in normalized
        or "original rollout-speed targets were met" not in normalized
    ):
        raise InputError("rollout speed scope report claim_boundary must preserve scope limits")


def _required_text_list(raw: object, label: str) -> list[str]:
    if not isinstance(raw, list) or not raw:
        raise InputError(f"{label} must be a non-empty JSON list")
    if not all(isinstance(item, str) and item.strip() for item in raw):
        raise InputError(f"{label} entries must be non-empty strings")
    return [item.strip() for item in raw]


def _required_bool(raw: dict[str, object], field: str) -> bool:
    value = raw.get(field)
    if not isinstance(value, bool):
        raise InputError(f"{field} must be a boolean")
    return value


def _utc_now() -> str:
    return datetime.now(tz=timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate a v0.2 benchmark-readiness coverage report for #197.",
    )
    parser.add_argument(
        "--metrics-json",
        type=Path,
        action="append",
        help="Measured eval metrics JSON from geno-lewm-eval or geno-lewm-eval-all.",
    )
    parser.add_argument(
        "--rollout-speed-report",
        type=Path,
        help="Optional bench.rollout report JSON for RFC-0004 speed gates.",
    )
    parser.add_argument(
        "--rollout-speed-scope-report",
        type=Path,
        help="Optional accepted rollout-speed scope report for explicit #42 re-scope decisions.",
    )
    parser.add_argument(
        "--efficiency-report",
        type=Path,
        help="Optional validated efficiency_report.json for RFC-0007 efficiency coverage.",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--require-ok",
        action="store_true",
        help="Exit non-zero when the v0.2 readiness report is incomplete.",
    )
    parser.add_argument(
        "--require-release-inputs",
        action="store_true",
        help=("Require release-ready artifact provenance checks; implied by --require-ok."),
    )
    return parser


def _command_from_args(args: argparse.Namespace) -> tuple[str, ...]:
    command = ["python", "-m", "tools.release.v02_benchmark_readiness"]
    for path in args.metrics_json or ():
        command.extend(("--metrics-json", _public_safe_identity_path(path)))
    if args.rollout_speed_report is not None:
        command.extend(
            ("--rollout-speed-report", _public_safe_identity_path(args.rollout_speed_report))
        )
    if args.rollout_speed_scope_report is not None:
        command.extend(
            (
                "--rollout-speed-scope-report",
                _public_safe_identity_path(args.rollout_speed_scope_report),
            )
        )
    if args.efficiency_report is not None:
        command.extend(("--efficiency-report", _public_safe_identity_path(args.efficiency_report)))
    command.extend(("--output", _public_safe_identity_path(args.output)))
    if args.require_ok:
        command.append("--require-ok")
    if args.require_release_inputs:
        command.append("--require-release-inputs")
    return tuple(command)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
