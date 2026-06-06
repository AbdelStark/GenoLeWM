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
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Final

from geno_lewm.errors import GenoLeWMError, InputError, exit_code_for
from geno_lewm.provenance import sha256_file
from tools.release.efficiency_report import EfficiencyReport, load_efficiency_report
from tools.release.eval_report import EvalReportInput, MetricResult, load_report_input

SCHEMA_VERSION: Final = "1.0.0"
GENERATED_BY: Final = "tools.release.v02_benchmark_readiness"
ROLLOUT_GENERATED_BY: Final = "bench.rollout"
ROLLOUT_SPEED_REQUIRED_METRICS: Final = ("k5_speedup", "k20_speedup")


@dataclass(frozen=True, slots=True)
class BenchmarkRequirement:
    """One required v0.2 benchmark coverage row."""

    benchmark_id: str
    track: str
    split: str | None
    required_metrics: tuple[str, ...]
    issue_refs: tuple[int, ...]
    required_baseline: str | None = None


REQUIRED_METRICS: Final = ("auroc", "average_precision", "balanced_accuracy", "accuracy")
VEP_REQUIREMENTS: Final = (
    BenchmarkRequirement(
        "clinvar_coding",
        "variant_effect_prediction",
        "clinvar_coding",
        REQUIRED_METRICS,
        (53, 55, 56, 197),
        required_baseline="carbon_zero_shot",
    ),
    BenchmarkRequirement(
        "clinvar_noncoding",
        "variant_effect_prediction",
        "clinvar_noncoding",
        REQUIRED_METRICS,
        (53, 55, 56, 197),
        required_baseline="carbon_zero_shot",
    ),
    BenchmarkRequirement(
        "brca2_saturation",
        "variant_effect_prediction",
        "brca2",
        ("spearman_rho",),
        (56, 197),
        required_baseline="carbon_zero_shot",
    ),
    BenchmarkRequirement(
        "traitgym_mendelian",
        "variant_effect_prediction",
        "traitgym_mendelian",
        ("spearman_rho",),
        (56, 197),
        required_baseline="carbon_zero_shot",
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


def build_readiness_report(
    *,
    metrics_json: tuple[Path, ...] = (),
    rollout_speed_report: Path | None = None,
    efficiency_report: Path | None = None,
    command: tuple[str, ...] = (),
) -> dict[str, object]:
    """Build a machine-readable v0.2 benchmark-readiness report."""
    metric_reports = tuple(load_report_input(path) for path in metrics_json)
    efficiency = (
        load_efficiency_report(efficiency_report) if efficiency_report is not None else None
    )
    _require_shared_identity(metric_reports)
    _require_efficiency_identity(metric_reports, efficiency)
    metric_rows = tuple(metric for report in metric_reports for metric in report.metrics)
    rows = [_benchmark_row(requirement, metric_rows) for requirement in BENCHMARK_REQUIREMENTS]
    rows.append(_efficiency_row(efficiency))
    rows.append(_rollout_speed_row(rollout_speed_report))
    missing_or_failed = [
        str(row["benchmark_id"]) for row in rows if str(row.get("status")) != "pass"
    ]
    ok = not missing_or_failed
    artifact_inputs = _artifact_inputs(
        metrics_json=metrics_json,
        rollout_speed_report=rollout_speed_report,
        efficiency_report=efficiency_report,
    )
    identity = _identity(metric_reports)
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
        "command": list(command),
        "inputs": artifact_inputs,
        "benchmark_rows": rows,
        "missing_or_failed_benchmarks": missing_or_failed,
        "metric_conclusions": _metric_conclusions(rows),
        "negative_findings": _negative_findings(missing_or_failed),
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
    efficiency_report: Path | None = None,
    command: tuple[str, ...] = (),
) -> dict[str, object]:
    """Build and write the v0.2 benchmark-readiness report."""
    report = build_readiness_report(
        metrics_json=metrics_json,
        rollout_speed_report=rollout_speed_report,
        efficiency_report=efficiency_report,
        command=command,
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
            efficiency_report=args.efficiency_report,
            command=command,
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
    baseline_missing = False
    if requirement.required_baseline is not None:
        required_metric_rows = tuple(
            metric
            for metric in split_metrics
            if _normalized_metric_name(metric) in requirement.required_metrics
        )
        baseline_missing = not required_metric_rows or not all(
            _metric_has_baseline(metric, requirement.required_baseline)
            for metric in required_metric_rows
        )
    if not split_metrics:
        status = "missing"
    elif missing_metrics or baseline_missing:
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
        "missing_metrics": missing_metrics,
        "required_baseline": requirement.required_baseline,
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
    return {
        "benchmark_id": "inference_efficiency",
        "track": "inference_efficiency",
        "status": "pass",
        "required_metrics": [
            "single_variant_latency_ms",
            "batched_throughput_variants_per_s",
            "peak_memory_bytes",
        ],
        "observed_metrics": list(report.measurements.to_dict()),
        "model_release": report.model_release,
        "dataset_snapshot": report.dataset_snapshot,
        "issue_refs": ["#56", "#58", "#197"],
    }


def _rollout_speed_row(path: Path | None) -> dict[str, object]:
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
    generated_by = payload.get("generated_by")
    if generated_by != ROLLOUT_GENERATED_BY:
        raise InputError(
            "rollout speed report generated_by is invalid",
            details={"expected": ROLLOUT_GENERATED_BY, "observed": generated_by},
        )
    rows = _require_list(payload.get("rows"), "rollout speed rows")
    observed: dict[str, float] = {}
    failed: list[str] = []
    for row in rows:
        if not isinstance(row, dict):
            raise InputError("rollout speed rows must be objects")
        horizon = _required_int(row, "horizon")
        speedup = _required_number(row, "measured_speedup")
        target = _required_number(row, "target_speedup")
        observed[f"k{horizon}_speedup"] = speedup
        if not bool(row.get("target_met")):
            failed.append(f"K={horizon}: {speedup:.6g}x < {target:.6g}x")
    missing_metrics = [
        metric for metric in ROLLOUT_SPEED_REQUIRED_METRICS if metric not in observed
    ]
    if not bool(payload.get("ok")):
        failed.append("report ok=false")
    if failed:
        status = "failed"
    elif missing_metrics:
        status = "incomplete"
    else:
        status = "pass"
    return {
        "benchmark_id": "ar_rollout_speed",
        "track": "rollout_performance",
        "status": status,
        "required_metrics": list(ROLLOUT_SPEED_REQUIRED_METRICS),
        "observed_metrics": sorted(observed),
        "observed_values": observed,
        "missing_metrics": missing_metrics,
        "failed_targets": failed,
        "commit": payload.get("commit"),
        "issue_refs": ["#42", "#197"],
    }


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
    efficiency_report: Path | None,
) -> dict[str, object]:
    inputs: dict[str, object] = {
        "metrics_json": [_file_identity(path) for path in metrics_json],
    }
    if rollout_speed_report is not None:
        inputs["rollout_speed_report"] = _file_identity(rollout_speed_report)
    if efficiency_report is not None:
        inputs["efficiency_report"] = _file_identity(efficiency_report)
    return inputs


def _file_identity(path: Path) -> dict[str, object]:
    if not path.is_file():
        raise InputError("readiness input artifact does not exist", details={"path": str(path)})
    return {
        "path": str(path),
        "sha256": sha256_file(path),
        "size_bytes": path.stat().st_size,
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


def _metric_conclusions(rows: list[dict[str, object]]) -> list[str]:
    conclusions: list[str] = []
    for row in rows:
        benchmark = str(row["benchmark_id"])
        status = str(row["status"])
        if status == "pass":
            conclusions.append(f"{benchmark} has measured artifact coverage for this report.")
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


def _negative_findings(missing_or_failed: list[str]) -> list[str]:
    if not missing_or_failed:
        return [
            (
                "No clinical utility, privacy, runtime-assurance, or deployment claim is established "
                "by benchmark coverage alone."
            )
        ]
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
    return parser


def _command_from_args(args: argparse.Namespace) -> tuple[str, ...]:
    command = ["python", "-m", "tools.release.v02_benchmark_readiness"]
    for path in args.metrics_json or ():
        command.extend(("--metrics-json", str(path)))
    if args.rollout_speed_report is not None:
        command.extend(("--rollout-speed-report", str(args.rollout_speed_report)))
    if args.efficiency_report is not None:
        command.extend(("--efficiency-report", str(args.efficiency_report)))
    command.extend(("--output", str(args.output)))
    if args.require_ok:
        command.append("--require-ok")
    return tuple(command)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
