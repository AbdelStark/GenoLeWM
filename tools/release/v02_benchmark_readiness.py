# SPDX-License-Identifier: Apache-2.0
"""Generate the v0.2 benchmark-readiness report.

This tool does not run private-data benchmarks and does not turn fixture
smokes into model-quality claims. It validates measured artifact inputs
that already exist, enumerates the v0.2 benchmark-suite coverage, and
records missing or failed rows explicitly.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Final

from geno_lewm._evaluation_report import EvalReportInput, MetricResult, load_report_input
from geno_lewm._v02_benchmark_metrics import (
    BENCHMARK_REQUIREMENTS,
    REQUIRED_METRICS,
    ROLLOUT_FIDELITY_REQUIREMENTS,
    ROLLOUT_SPLITS,
    VEP_REQUIREMENTS,
    VEP_SPLITS,
    BenchmarkRequirement,
    _benchmark_row,
    require_v02_rollout_benchmark_metrics,
    require_v02_vep_benchmark_metrics,
    v02_rollout_benchmark_metric_findings,
    v02_vep_benchmark_metric_findings,
)
from geno_lewm.errors import GenoLeWMError, InputError, exit_code_for
from geno_lewm.provenance import sha256_file
from tools.release.efficiency_report import EfficiencyReport, load_efficiency_report
from tools.release.rollout_speed_scope import (
    DECISION as ROLLOUT_SPEED_SCOPE_DECISION,
    GENERATED_BY as ROLLOUT_SPEED_SCOPE_GENERATED_BY,
    SCHEMA_VERSION as ROLLOUT_SPEED_SCOPE_SCHEMA_VERSION,
    STATUS as ROLLOUT_SPEED_SCOPE_STATUS,
)
from tools.release.v02_benchmark_suite import (
    GENERATED_BY as BENCHMARK_SUITE_GENERATED_BY,
    SCHEMA_VERSION as BENCHMARK_SUITE_SCHEMA_VERSION,
)

__all__ = [
    "BENCHMARK_REQUIREMENTS",
    "GENERATED_BY",
    "REQUIRED_METRICS",
    "ROLLOUT_FIDELITY_REQUIREMENTS",
    "ROLLOUT_SPLITS",
    "SCHEMA_VERSION",
    "VEP_REQUIREMENTS",
    "VEP_SPLITS",
    "BenchmarkRequirement",
    "EvalReportInput",
    "MetricResult",
    "build_readiness_report",
    "main",
    "require_v02_rollout_benchmark_metrics",
    "require_v02_vep_benchmark_metrics",
    "v02_rollout_benchmark_metric_findings",
    "v02_vep_benchmark_metric_findings",
    "write_readiness_report",
]

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
        "--suite-report",
        "--output-json",
        "--out-dir",
        "--output",
    }
)


def build_readiness_report(
    *,
    metrics_json: tuple[Path, ...] = (),
    rollout_speed_report: Path | None = None,
    rollout_speed_scope_report: Path | None = None,
    efficiency_report: Path | None = None,
    suite_report: Path | None = None,
    command: tuple[str, ...] = (),
    require_release_inputs: bool = False,
) -> dict[str, object]:
    """Build a machine-readable v0.2 benchmark-readiness report."""
    metric_reports = tuple(load_report_input(path) for path in metrics_json)
    efficiency = (
        load_efficiency_report(efficiency_report) if efficiency_report is not None else None
    )
    suite_payload = _load_suite_report(suite_report) if suite_report is not None else None
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
                metrics_json=metrics_json,
                metric_reports=metric_reports,
                rollout_speed_report=rollout_speed_report,
                efficiency=efficiency,
                suite_report=suite_report,
                suite_payload=suite_payload,
            )
        )
    missing_or_failed = [
        str(row["benchmark_id"])
        for row in rows
        if str(row.get("status")) not in {"pass", "documented_limitation"}
    ]
    readiness = _readiness_items(rows)
    blockers = _readiness_blockers(rows)
    scope_decisions = _scope_decisions(rows)
    ok = not missing_or_failed
    artifact_inputs = _artifact_inputs(
        metrics_json=metrics_json,
        rollout_speed_report=rollout_speed_report,
        rollout_speed_scope_report=rollout_speed_scope_report,
        efficiency_report=efficiency_report,
        suite_report=suite_report,
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
        "readiness": readiness,
        "blockers": blockers,
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
    suite_report: Path | None = None,
    command: tuple[str, ...] = (),
    require_release_inputs: bool = False,
) -> dict[str, object]:
    """Build and write the v0.2 benchmark-readiness report."""
    report = build_readiness_report(
        metrics_json=metrics_json,
        rollout_speed_report=rollout_speed_report,
        rollout_speed_scope_report=rollout_speed_scope_report,
        efficiency_report=efficiency_report,
        suite_report=suite_report,
        command=command,
        require_release_inputs=require_release_inputs,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return report


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
            suite_report=args.suite_report,
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
    _require_rollout_claim_boundary(payload.get("claim_boundary"))
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
        status = "documented_limitation"
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
    metrics_json: tuple[Path, ...],
    metric_reports: tuple[EvalReportInput, ...],
    rollout_speed_report: Path | None,
    efficiency: EfficiencyReport | None,
    suite_report: Path | None,
    suite_payload: dict[str, object] | None,
) -> dict[str, object]:
    findings: list[str] = []
    if not metric_reports:
        findings.append("at least one measured eval metrics JSON artifact is required")
    if rollout_speed_report is None:
        findings.append("a bench.rollout speed report is required")
    if efficiency is None:
        findings.append("an efficiency_report.json artifact is required")
    if suite_report is None:
        findings.append("a v0.2 benchmark suite report is required")
    for index, report in enumerate(metric_reports, start=1):
        findings.extend(_metric_release_input_findings(report, input_index=index))
    if efficiency is not None:
        findings.extend(_efficiency_release_input_findings(efficiency))
    if suite_payload is not None:
        findings.extend(
            _suite_release_input_findings(
                suite_payload,
                metrics_json=metrics_json,
                suite_report=suite_report,
            )
        )
    if findings:
        status = (
            "missing"
            if not metric_reports
            or rollout_speed_report is None
            or efficiency is None
            or suite_report is None
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
            "suite_output_identities",
        ],
        "observed_metrics": []
        if findings
        else [
            "package_relative_artifacts",
            "score_or_metrics_input_provenance",
            "non_fixture_release_identity",
            "efficiency_input_identities",
            "suite_output_identities",
        ],
        "checked_artifacts": _checked_release_input_artifacts(
            metric_reports=metric_reports,
            rollout_speed_report=rollout_speed_report,
            efficiency=efficiency,
            suite_report=suite_report,
            suite_payload=suite_payload,
        ),
        "findings": findings,
        "issue_refs": ["#56", "#197"],
    }


def _checked_release_input_artifacts(
    *,
    metric_reports: tuple[EvalReportInput, ...],
    rollout_speed_report: Path | None,
    efficiency: EfficiencyReport | None,
    suite_report: Path | None,
    suite_payload: dict[str, object] | None,
) -> dict[str, object]:
    checked: dict[str, object] = {
        "metrics_json": [
            {
                "input_index": index,
                "generated_by": report.generated_by,
                "artifacts": {
                    key: _public_safe_checked_artifact_path(value)
                    for key, value in sorted(report.artifacts)
                },
            }
            for index, report in enumerate(metric_reports, start=1)
        ]
    }
    if rollout_speed_report is not None:
        checked["rollout_speed_report"] = _file_identity(rollout_speed_report)
    if efficiency is not None:
        checked["efficiency_report"] = {
            "inputs": {
                key: identity.to_dict()
                for key, identity in sorted(efficiency.inputs, key=lambda item: item[0])
            }
        }
    if suite_report is not None:
        suite_checked: dict[str, object] = {"report": _file_identity(suite_report)}
        if suite_payload is not None:
            manifest = suite_payload.get("manifest")
            if isinstance(manifest, dict):
                suite_checked["manifest"] = _public_safe_identity_mapping(manifest)
            passed_step_outputs = _suite_passed_step_outputs(suite_payload)
            if passed_step_outputs:
                suite_checked["passed_step_outputs"] = passed_step_outputs
        checked["suite_report"] = suite_checked
    return checked


def _public_safe_identity_mapping(raw: dict[str, object]) -> dict[str, object]:
    identity: dict[str, object] = {}
    path = raw.get("path")
    if isinstance(path, str):
        identity["path"] = _public_safe_checked_artifact_path(path)
    sha256 = raw.get("sha256")
    if isinstance(sha256, str):
        identity["sha256"] = sha256
    size_bytes = raw.get("size_bytes")
    if isinstance(size_bytes, int) and not isinstance(size_bytes, bool):
        identity["size_bytes"] = size_bytes
    return identity


def _suite_passed_step_outputs(payload: dict[str, object]) -> list[dict[str, object]]:
    raw_steps = payload.get("steps")
    if not isinstance(raw_steps, list):
        return []
    step_outputs: list[dict[str, object]] = []
    for raw_step in raw_steps:
        if not isinstance(raw_step, dict):
            continue
        step_id = raw_step.get("id")
        raw_outputs = raw_step.get("output_identities")
        if not isinstance(step_id, str) or not isinstance(raw_outputs, list):
            continue
        outputs = [
            _public_safe_identity_mapping(raw_identity)
            for raw_identity in raw_outputs
            if isinstance(raw_identity, dict)
        ]
        if outputs:
            step_outputs.append({"step_id": step_id, "outputs": outputs})
    return step_outputs


def _public_safe_checked_artifact_path(value: str) -> str:
    if not _path_findings(value, field="artifact", allow_inline=False):
        return value
    if "://" in value:
        return "<non-package-relative-url>"
    redacted = _public_safe_path_text(value)
    if redacted != value:
        return redacted
    path = PureWindowsPath(value) if "\\" in value else PurePosixPath(value)
    return path.name or "<non-package-relative-path>"


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


def _suite_release_input_findings(
    payload: dict[str, object],
    *,
    metrics_json: tuple[Path, ...],
    suite_report: Path | None,
) -> list[str]:
    findings: list[str] = []
    if payload.get("execute") is not True:
        findings.append("suite_report.execute must be true")
    if payload.get("ok") is not True:
        findings.append("suite_report.ok must be true")
    if payload.get("status") != "pass":
        findings.append("suite_report.status must be pass")
    findings.extend(_suite_boundary_findings(payload))
    manifest = payload.get("manifest")
    if not isinstance(manifest, dict):
        findings.append("suite_report.manifest must be an artifact identity object")
    else:
        findings.extend(_artifact_identity_findings(manifest, field="suite_report.manifest"))
    steps = payload.get("steps")
    if not isinstance(steps, list) or not steps:
        findings.append("suite_report.steps must list executed benchmark steps")
        return findings
    output_identity_count = 0
    output_paths: set[str] = set()
    output_identity_keys: set[tuple[str, int]] = set()
    for index, raw_step in enumerate(steps, start=1):
        prefix = f"suite_report.steps[{index}]"
        if not isinstance(raw_step, dict):
            findings.append(f"{prefix} must be an object")
            continue
        if raw_step.get("status") != "pass":
            findings.append(f"{prefix}.status must be pass")
        declared_outputs = _suite_declared_outputs(raw_step, prefix=prefix, findings=findings)
        raw_identities = raw_step.get("output_identities")
        if not isinstance(raw_identities, list) or not raw_identities:
            findings.append(
                f"{prefix}.output_identities must list generated output file identities"
            )
            continue
        identity_paths: list[str] = []
        for output_index, raw_identity in enumerate(raw_identities, start=1):
            identity_field = f"{prefix}.output_identities[{output_index}]"
            if isinstance(raw_identity, dict):
                path = raw_identity.get("path")
                if isinstance(path, str):
                    identity_paths.append(path)
                    output_paths.add(path)
                identity_key = _artifact_identity_key(raw_identity)
                if identity_key is not None:
                    output_identity_keys.add(identity_key)
                findings.extend(_artifact_identity_findings(raw_identity, field=identity_field))
            else:
                findings.append(f"{identity_field} must be an artifact identity object")
        output_identity_count += len(raw_identities)
        if declared_outputs and sorted(identity_paths) != sorted(declared_outputs):
            findings.append(f"{prefix}.output_identities must match declared outputs")
    if output_identity_count == 0:
        findings.append("suite_report must record at least one output identity")
    if suite_report is not None:
        missing_metrics: list[str] = []
        for path in metrics_json:
            expected_path = _suite_relative_input_path(path, root=suite_report.parent)
            if expected_path in output_paths:
                continue
            if _suite_output_identity_matches_file(path, output_identity_keys):
                continue
            missing_metrics.append(expected_path)
        if missing_metrics:
            findings.append(
                "suite_report.output_identities must include metrics_json inputs: "
                + ", ".join(missing_metrics)
            )
    return findings


def _suite_boundary_findings(payload: dict[str, object]) -> list[str]:
    findings: list[str] = []
    negative_findings = payload.get("negative_findings")
    if not isinstance(negative_findings, list) or not any(
        isinstance(item, str) and item.strip() for item in negative_findings
    ):
        findings.append("suite_report.negative_findings must be a non-empty text list")
    else:
        boundary_text = "\n".join(
            item.lower() for item in negative_findings if isinstance(item, str)
        )
        if "measured" not in boundary_text or (
            "validator" not in boundary_text and "validate" not in boundary_text
        ):
            findings.append(
                "suite_report.negative_findings must preserve measured-claim validator limits"
            )
    claim_boundary = payload.get("claim_boundary")
    if not isinstance(claim_boundary, str) or not claim_boundary.strip():
        findings.append("suite_report.claim_boundary must be non-empty text")
    else:
        claim_text = claim_boundary.lower()
        if "model-quality" not in claim_text or "validate separately" not in claim_text:
            findings.append("suite_report.claim_boundary must preserve measured-claim limits")
    return findings


def _suite_relative_input_path(path: Path, *, root: Path) -> str:
    if path.is_absolute():
        try:
            return path.resolve().relative_to(root.resolve()).as_posix()
        except ValueError:
            return _public_safe_identity_path(path)
    return path.as_posix()


def _suite_declared_outputs(
    raw_step: dict[str, object],
    *,
    prefix: str,
    findings: list[str],
) -> list[str]:
    raw_outputs = raw_step.get("outputs")
    if not isinstance(raw_outputs, list) or not raw_outputs:
        findings.append(f"{prefix}.outputs must list declared output paths")
        return []
    outputs: list[str] = []
    for output_index, raw_output in enumerate(raw_outputs, start=1):
        field = f"{prefix}.outputs[{output_index}]"
        if not isinstance(raw_output, str) or not raw_output:
            findings.append(f"{field} must be a package-relative path")
            continue
        findings.extend(_artifact_path_release_findings(raw_output, field=field))
        outputs.append(raw_output)
    return outputs


def _artifact_identity_findings(raw: dict[str, object], *, field: str) -> list[str]:
    findings: list[str] = []
    path = raw.get("path")
    if not isinstance(path, str) or not path:
        findings.append(f"{field}.path must be a package-relative path")
    else:
        findings.extend(_artifact_path_release_findings(path, field=f"{field}.path"))
    sha256 = raw.get("sha256")
    if not isinstance(sha256, str) or not sha256.startswith("sha256:"):
        findings.append(f"{field}.sha256 must be a sha256:<hex> identity")
    size_bytes = raw.get("size_bytes")
    if isinstance(size_bytes, bool) or not isinstance(size_bytes, int) or size_bytes <= 0:
        findings.append(f"{field}.size_bytes must be a positive integer")
    return findings


def _artifact_identity_key(raw: dict[str, object]) -> tuple[str, int] | None:
    sha256 = raw.get("sha256")
    size_bytes = raw.get("size_bytes")
    if not isinstance(sha256, str) or not isinstance(size_bytes, int):
        return None
    if isinstance(size_bytes, bool):
        return None
    return (sha256, size_bytes)


def _suite_output_identity_matches_file(
    path: Path,
    output_identity_keys: set[tuple[str, int]],
) -> bool:
    if not output_identity_keys or not path.is_file():
        return False
    return (sha256_file(path), path.stat().st_size) in output_identity_keys


def _artifact_path_release_findings(path: str, *, field: str) -> list[str]:
    findings = _path_findings(path, field=field, allow_inline=False)
    if RELEASE_ARTIFACT_PLACEHOLDER_RE.search(path):
        findings.append(f"{field} must not reference fixture/test artifacts")
    return findings


def _artifact_findings(artifacts: dict[str, str], *, prefix: str) -> list[str]:
    findings: list[str] = []
    for key, value in sorted(artifacts.items()):
        field = f"{prefix}.{key}"
        findings.extend(_artifact_path_release_findings(value, field=field))
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
    suite_report: Path | None,
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
    if suite_report is not None:
        inputs["suite_report"] = _file_identity(suite_report)
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
        elif status == "documented_limitation":
            context = _format_row_context(row)
            values = _format_metric_values(row.get("observed_values"))
            failed_targets = _format_text_list(row.get("failed_targets"))
            scope_decision = _format_scope_decision(row.get("scope_decision"))
            conclusion = f"{benchmark} is a documented limitation after failed measured targets"
            if context:
                conclusion += f" ({context})"
            conclusion += f": {values}." if values else "."
            if failed_targets:
                conclusion += f" Failed targets: {failed_targets}."
            if scope_decision:
                conclusion += f" Accepted limitation: {scope_decision}."
            conclusions.append(conclusion)
        else:
            context = _format_row_context(row)
            details = _format_nonpassing_details(row)
            raw_issue_refs = row.get("issue_refs")
            issue_refs = (
                ", ".join(raw_issue_refs)
                if isinstance(raw_issue_refs, list)
                and all(isinstance(ref, str) for ref in raw_issue_refs)
                else ""
            )
            conclusion = f"{benchmark} is {status}"
            if context:
                conclusion += f" ({context})"
            if details:
                conclusion += f": {details}"
            conclusion += f"; route remaining work through {issue_refs}."
            conclusions.append(conclusion)
    return conclusions


def _readiness_items(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    items: list[dict[str, object]] = []
    for row in rows:
        benchmark = str(row["benchmark_id"])
        status = str(row["status"])
        display_status = status.replace("_", " ")
        blockers = _row_blocker_codes(row)
        evidence = _row_evidence(row)
        message = f"{benchmark} is {display_status}"
        details = _format_nonpassing_details(row)
        if details:
            message += f": {details}"
        items.append(
            {
                "code": benchmark,
                "ok": status in {"pass", "documented_limitation"},
                "status": status,
                "message": message,
                "evidence": evidence,
                "blockers": blockers,
                "issue_refs": _row_issue_refs(row),
            }
        )
    return items


def _readiness_blockers(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    blockers: list[dict[str, object]] = []
    for row in rows:
        for code in _row_blocker_codes(row):
            details = _format_nonpassing_details(row)
            blockers.append(
                {
                    "code": code,
                    "benchmark_id": row["benchmark_id"],
                    "status": row["status"],
                    "message": details or f"{row['benchmark_id']} is {row['status']}",
                    "issue_refs": _row_issue_refs(row),
                }
            )
    return blockers


def _row_blocker_codes(row: dict[str, object]) -> list[str]:
    status = str(row["status"])
    if status in {"pass", "documented_limitation"}:
        return []
    return [f"benchmark.{row['benchmark_id']}.{status}"]


def _row_issue_refs(row: dict[str, object]) -> list[str]:
    raw = row.get("issue_refs")
    if not isinstance(raw, list):
        return []
    return [ref for ref in raw if isinstance(ref, str) and ref]


def _row_evidence(row: dict[str, object]) -> list[str]:
    evidence: list[str] = []
    values = _format_metric_values(row.get("observed_values"))
    if values:
        evidence.append(f"observed_values={values}")
    deltas = _format_metric_values(row.get("delta_vs_baseline"))
    if deltas:
        evidence.append(f"delta_vs_baseline={deltas}")
    intervals = _format_confidence_intervals(row.get("confidence_intervals"))
    if intervals:
        evidence.append(f"confidence_intervals={intervals}")
    variant_identities = _format_text_values(row.get("evaluated_variant_key_identities"))
    if variant_identities:
        evidence.append(f"evaluated_variant_key_identities={variant_identities}")
    command = _format_command(row.get("command"))
    if command:
        evidence.append(f"command={command}")
    scope_decision = _format_scope_decision(row.get("scope_decision"))
    if scope_decision:
        evidence.append(f"scope_decision={scope_decision}")
    checked_artifacts = _format_checked_artifacts(row.get("checked_artifacts"))
    if checked_artifacts:
        evidence.append(f"checked_artifacts={checked_artifacts}")
    return evidence


def _format_command(raw: object) -> str:
    if not isinstance(raw, list) or not raw:
        return ""
    tokens = [token for token in raw if isinstance(token, str) and token]
    return " ".join(tokens)


def _format_checked_artifacts(raw: object) -> str:
    if not isinstance(raw, dict) or not raw:
        return ""
    parts: list[str] = []
    metrics = raw.get("metrics_json")
    if isinstance(metrics, list):
        metric_parts: list[str] = []
        for item in metrics:
            if not isinstance(item, dict):
                continue
            index = item.get("input_index")
            artifacts = item.get("artifacts")
            if not isinstance(index, int) or not isinstance(artifacts, dict):
                continue
            keys = sorted(key for key in artifacts if isinstance(key, str))
            if keys:
                metric_parts.append(f"metrics_json[{index}]={','.join(keys)}")
        if metric_parts:
            parts.append("; ".join(metric_parts))
    rollout = raw.get("rollout_speed_report")
    if isinstance(rollout, dict) and isinstance(rollout.get("path"), str):
        parts.append(f"rollout_speed_report={rollout['path']}")
    efficiency = raw.get("efficiency_report")
    if isinstance(efficiency, dict):
        inputs = efficiency.get("inputs")
        if isinstance(inputs, dict):
            keys = sorted(key for key in inputs if isinstance(key, str))
            if keys:
                parts.append(f"efficiency_inputs={','.join(keys)}")
    suite = raw.get("suite_report")
    if isinstance(suite, dict):
        report = suite.get("report")
        if isinstance(report, dict) and isinstance(report.get("path"), str):
            parts.append(f"suite_report={report['path']}")
        outputs = suite.get("passed_step_outputs")
        if isinstance(outputs, list):
            count = 0
            for step in outputs:
                if not isinstance(step, dict):
                    continue
                raw_step_outputs = step.get("outputs")
                if isinstance(raw_step_outputs, list):
                    count += len(raw_step_outputs)
            if count:
                parts.append(f"suite_outputs={count}")
    return "; ".join(parts)


def _scope_decisions(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    decisions: list[dict[str, object]] = []
    for row in rows:
        if row.get("status") != "documented_limitation":
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


def _format_comma_text_list(raw: object) -> str:
    if not isinstance(raw, list) or not raw:
        return ""
    values = [value for value in raw if isinstance(value, str) and value]
    return ", ".join(values)


def _format_nonpassing_details(row: dict[str, object]) -> str:
    parts: list[str] = []
    values = _format_metric_values(row.get("observed_values"))
    if values:
        parts.append(f"observed_values={values}")
    missing_metrics = _format_comma_text_list(row.get("missing_metrics"))
    if missing_metrics:
        parts.append(f"missing_metrics={missing_metrics}")
    missing_intervals = _format_comma_text_list(row.get("missing_confidence_intervals"))
    if missing_intervals:
        parts.append(f"missing_confidence_intervals={missing_intervals}")
    required_baseline = row.get("required_baseline")
    if (
        isinstance(required_baseline, str)
        and required_baseline
        and row.get("baseline_observed") is False
    ):
        parts.append(f"required_baseline={required_baseline} missing")
    failed_targets = _format_text_list(row.get("failed_targets"))
    if failed_targets:
        parts.append(f"failed_targets={failed_targets}")
    findings = _format_text_list(row.get("findings"))
    if findings:
        parts.append(f"findings={findings}")
    return "; ".join(parts)


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
                "At least one benchmark target is documented as a limitation; original failed "
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


def _load_suite_report(path: Path) -> dict[str, object]:
    payload = _load_json(path, label="v0.2 benchmark suite report")
    if payload.get("schema_version") != BENCHMARK_SUITE_SCHEMA_VERSION:
        raise InputError(
            "v0.2 benchmark suite report schema_version is unsupported",
            details={"observed": payload.get("schema_version")},
        )
    if payload.get("generated_by") != BENCHMARK_SUITE_GENERATED_BY:
        raise InputError(
            "v0.2 benchmark suite report generated_by is unsupported",
            details={"observed": payload.get("generated_by")},
        )
    return payload


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


def _require_rollout_claim_boundary(raw: object) -> None:
    text = _require_text_value(raw, "claim_boundary")
    lower = text.lower()
    required_terms = (
        "rollout speed",
        "not",
        "model-quality",
        "clinical",
        "privacy",
        "release-readiness",
    )
    if any(term not in lower for term in required_terms):
        raise InputError("rollout speed report claim_boundary must preserve benchmark limits")


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
        help="Optional bench.rollout report JSON for rollout-speed gates.",
    )
    parser.add_argument(
        "--rollout-speed-scope-report",
        type=Path,
        help="Optional accepted rollout-speed limitation report for the #42 target miss.",
    )
    parser.add_argument(
        "--efficiency-report",
        type=Path,
        help="Optional validated efficiency_report.json for efficiency coverage.",
    )
    parser.add_argument(
        "--suite-report",
        type=Path,
        help="Optional tools.release.v02_benchmark_suite report for suite output provenance.",
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
    if args.suite_report is not None:
        command.extend(("--suite-report", _public_safe_identity_path(args.suite_report)))
    command.extend(("--output", _public_safe_identity_path(args.output)))
    if args.require_ok:
        command.append("--require-ok")
    if args.require_release_inputs:
        command.append("--require-release-inputs")
    return tuple(command)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
