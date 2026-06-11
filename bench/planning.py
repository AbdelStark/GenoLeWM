# SPDX-License-Identifier: Apache-2.0
"""Planning-loop benchmarks for the planning contract CEM solver.

This benchmark times the pure CEM loop, ``ActionSampler`` proposal
sampling/refitting, and edit-cost integration. It also records one
deterministic default-config planning call so issue #59 performance
evidence can be collected on named hardware without implying learned
model quality.
"""

from __future__ import annotations

import argparse
import datetime as _dt
import json
import math
import platform as _platform
import subprocess
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from bench._harness import (
    DEFAULT_RESULTS_DIR,
    BenchResult,
    current_commit,
    machine_id,
    report_to_stdout,
    time_callable,
)
from geno_lewm.action import EditType, RelEdit
from geno_lewm.errors import InputError
from geno_lewm.planning.cem import CandidateEvaluation, PlanningConfig, PlanningResult, plan
from geno_lewm.planning.sampling import ActionSampler

GENERATED_BY = "bench.planning"
SCHEMA_VERSION = "1.0.0"
DEFAULT_HORIZONS = (1, 3, 10)
DEFAULT_SEED = 20260609
DEFAULT_WINDOW_BP = 12_288
DEFAULT_TARGET_PROFILE = "none"
TARGET_SECONDS_BY_PROFILE = {
    "h100": 1.0,
    "m3-max": 30.0,
}
SUPPORTED_TARGET_PROFILES = (DEFAULT_TARGET_PROFILE, *TARGET_SECONDS_BY_PROFILE)


@dataclass(frozen=True, slots=True)
class PlanningBenchmarkConfig:
    """Configuration for CEM solver and default-planning benchmarks."""

    horizons: tuple[int, ...] = DEFAULT_HORIZONS
    n_samples: int = 256
    n_elite: int = 32
    iters: int = 50
    warmup: int = 5
    seed: int = DEFAULT_SEED
    window_bp: int = DEFAULT_WINDOW_BP
    edge_margin: int = 64
    position_bin_bp: int = 8
    default_horizon: int = 5
    default_iterations: int = 5
    default_samples: int = 1024
    default_elite: int = 64
    default_stopping_eps: float = 0.05
    default_patience: int = 2
    target_profile: str = DEFAULT_TARGET_PROFILE
    target_seconds: float | None = None


def build_planning_report(config: PlanningBenchmarkConfig) -> dict[str, object]:
    """Run the planning benchmarks and return a machine-readable report."""
    return _build_planning_report(config, command=())


def _build_planning_report(
    config: PlanningBenchmarkConfig,
    *,
    command: Sequence[str],
) -> dict[str, object]:
    _validate_config(config)
    hardware = _hardware_summary()
    rows: list[dict[str, object]] = []

    for horizon in config.horizons:
        benchmark = _benchmark_cem_iteration(config, horizon=horizon)
        rows.append(_summarize_benchmark(benchmark, horizon=horizon, config=config))

    default_call = _default_call_summary(config, hardware=hardware)
    ok = all(_positive_median(row) for row in rows) and bool(default_call["deterministic_match"])
    if default_call["target_met"] is False:
        ok = False

    return {
        "schema_version": SCHEMA_VERSION,
        "generated_by": GENERATED_BY,
        "generated_at": _utc_now(),
        "commit": current_commit(),
        "machine": machine_id(),
        "hardware": hardware,
        "command": list(command),
        "config": asdict(config),
        "rows": rows,
        "default_call": default_call,
        "ok": ok,
        "claim_boundary": (
            "This benchmark measures the pure CEM solver, sampler, and cost loop. "
            "It does not measure Carbon encoding, learned predictor quality, clinical "
            "utility, privacy assurance, or runtime assurance. Hardware target rows are "
            "valid only when the report was generated on the stated hardware profile."
        ),
    }


def write_planning_report(
    payload: dict[str, object],
    *,
    out_dir: Path = DEFAULT_RESULTS_DIR,
) -> Path:
    """Persist the aggregate planning performance report."""
    machine = str(payload.get("machine") or machine_id())
    target_dir = out_dir / machine
    target_dir.mkdir(parents=True, exist_ok=True)
    path = target_dir / "planning.performance.json"
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def write_planning_benchmarks(
    payload: dict[str, object],
    *,
    out_dir: Path = DEFAULT_RESULTS_DIR,
) -> tuple[Path, ...]:
    """Persist per-workload ``BenchResult`` JSON for regression detection."""
    paths: list[Path] = []
    for row in _require_rows(payload):
        result = row.get("result")
        if not isinstance(result, dict):
            raise InputError("planning benchmark row is missing result payload")
        name = _required_str(result, "name")
        metadata = result.get("metadata")
        if not isinstance(metadata, dict):
            raise InputError("planning benchmark result is missing metadata")
        machine = str(metadata.get("machine") or payload.get("machine") or machine_id())
        target_dir = out_dir / machine
        target_dir.mkdir(parents=True, exist_ok=True)
        path = target_dir / f"{_sanitize_filename(name)}.json"
        path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        paths.append(path)
    return tuple(paths)


def target_seconds_for_profile(profile: str) -> float | None:
    """Return the default-planning target seconds for a hardware profile."""
    if profile == DEFAULT_TARGET_PROFILE:
        return None
    try:
        return TARGET_SECONDS_BY_PROFILE[profile]
    except KeyError as exc:
        raise InputError(
            "unsupported planning target profile",
            details={"profile": profile, "supported": list(SUPPORTED_TARGET_PROFILES)},
        ) from exc


def _benchmark_cem_iteration(config: PlanningBenchmarkConfig, *, horizon: int) -> BenchResult:
    window = _window(config.window_bp)
    sampler = ActionSampler(
        window,
        edge_margin=config.edge_margin,
        position_bin_bp=config.position_bin_bp,
    )
    planning_config = PlanningConfig(
        horizon=horizon,
        n_iterations=1,
        n_samples=config.n_samples,
        n_elite=config.n_elite,
        stopping_eps=0.0,
        patience=1,
        seed=config.seed,
    )
    evaluate = _make_evaluator(
        horizon=horizon,
        window_bp=config.window_bp,
        edge_margin=config.edge_margin,
    )

    def workload() -> PlanningResult:
        return plan(evaluate, sampler, config=planning_config)

    return time_callable(
        f"planning.cem_iter.k{horizon}",
        workload,
        iters=config.iters,
        warmup=config.warmup,
        dtype="n/a",
        extra=_metadata_extra(config, planning_config),
    )


def _default_call_summary(
    config: PlanningBenchmarkConfig,
    *,
    hardware: dict[str, object],
) -> dict[str, object]:
    planning_config = PlanningConfig(
        horizon=config.default_horizon,
        n_iterations=config.default_iterations,
        n_samples=config.default_samples,
        n_elite=config.default_elite,
        stopping_eps=config.default_stopping_eps,
        patience=config.default_patience,
        seed=config.seed,
    )
    first = _run_single_plan(config, planning_config)
    second = _run_single_plan(config, planning_config)
    deterministic_match = _same_result(first, second)
    target_seconds = (
        config.target_seconds
        if config.target_seconds is not None
        else target_seconds_for_profile(config.target_profile)
    )
    profile_match = _target_profile_match(config.target_profile, hardware)
    elapsed_target_met = None if target_seconds is None else first.elapsed_seconds <= target_seconds
    target_failures: list[str] = []
    if profile_match is False:
        target_failures.append("hardware_profile_mismatch")
    if elapsed_target_met is False:
        target_failures.append("elapsed_seconds_exceeded_target")
    target_met = None
    if target_seconds is not None:
        target_met = elapsed_target_met is True and profile_match is not False
    return {
        "benchmark": "planning.default_call",
        "target_profile": None
        if config.target_profile == DEFAULT_TARGET_PROFILE
        else config.target_profile,
        "hardware_profile_match": profile_match,
        "target_seconds": target_seconds,
        "target_met": target_met,
        "target_failures": target_failures,
        "deterministic_match": deterministic_match,
        "repeat_elapsed_seconds": second.elapsed_seconds,
        "config": {
            "horizon": planning_config.horizon,
            "n_iterations": planning_config.n_iterations,
            "n_samples": planning_config.n_samples,
            "n_elite": planning_config.n_elite,
            "stopping_eps": planning_config.stopping_eps,
            "patience": planning_config.patience,
            "seed": planning_config.seed,
        },
        "result": _planning_result_payload(first),
        "repeat_result": _planning_result_payload(second),
    }


def _run_single_plan(
    benchmark_config: PlanningBenchmarkConfig,
    planning_config: PlanningConfig,
) -> PlanningResult:
    window = _window(benchmark_config.window_bp)
    sampler = ActionSampler(
        window,
        edge_margin=benchmark_config.edge_margin,
        position_bin_bp=benchmark_config.position_bin_bp,
    )
    evaluate = _make_evaluator(
        horizon=planning_config.horizon,
        window_bp=benchmark_config.window_bp,
        edge_margin=benchmark_config.edge_margin,
    )
    return plan(evaluate, sampler, config=planning_config)


def _summarize_benchmark(
    result: BenchResult,
    *,
    horizon: int,
    config: PlanningBenchmarkConfig,
) -> dict[str, object]:
    return {
        "benchmark": result.name,
        "horizon": horizon,
        "n_iterations": 1,
        "n_samples": config.n_samples,
        "n_elite": config.n_elite,
        "median_seconds": result.median_ns / 1_000_000_000.0,
        "result": result.to_json(),
    }


def _planning_result_payload(result: PlanningResult) -> dict[str, object]:
    return {
        "best_edits": [_edit_payload(edit) for edit in result.best_edits],
        "best_distance": result.best_distance,
        "best_cost": result.best_cost,
        "best_objective": result.best_objective,
        "n_evaluations": result.n_evaluations,
        "logical_step_evaluations": result.n_evaluations * max(1, len(result.best_edits)),
        "elapsed_seconds": result.elapsed_seconds,
        "stopped_reason": result.stopped_reason,
        "iterations": [
            {
                "iteration": item.iteration,
                "best_distance": item.best_distance,
                "best_cost": item.best_cost,
                "best_objective": item.best_objective,
                "elite_mean_distance": item.elite_mean_distance,
                "elite_mean_objective": item.elite_mean_objective,
                "n_candidates": item.n_candidates,
            }
            for item in result.iterations
        ],
    }


def _same_result(left: PlanningResult, right: PlanningResult) -> bool:
    return (
        left.best_edits == right.best_edits
        and left.best_distance == right.best_distance
        and left.best_cost == right.best_cost
        and left.best_objective == right.best_objective
        and left.n_evaluations == right.n_evaluations
        and left.stopped_reason == right.stopped_reason
        and tuple(left.iterations) == tuple(right.iterations)
    )


def _make_evaluator(
    *,
    horizon: int,
    window_bp: int,
    edge_margin: int,
) -> Any:
    target_positions = _target_positions(
        horizon=horizon,
        window_bp=window_bp,
        edge_margin=edge_margin,
    )

    def evaluate(edits: Sequence[RelEdit]) -> CandidateEvaluation:
        if len(edits) != horizon:
            raise InputError(
                "planning benchmark evaluator received wrong horizon",
                details={"expected": horizon, "observed": len(edits)},
            )
        position_loss = 0.0
        type_penalty = 0.0
        for idx, edit in enumerate(edits):
            position_loss += float(edit.rel_pos - target_positions[idx]) ** 2
            if edit.edit_type is not EditType.SNV:
                type_penalty += 4.0
        return CandidateEvaluation(distance=math.sqrt(position_loss + type_penalty))

    return evaluate


def _target_positions(*, horizon: int, window_bp: int, edge_margin: int) -> tuple[int, ...]:
    _require_positive("horizon", horizon)
    interior = window_bp - 2 * edge_margin
    if interior <= 0:
        raise InputError(
            "edge_margin leaves no benchmark window interior",
            details={"window_bp": window_bp, "edge_margin": edge_margin},
        )
    step = interior / float(horizon + 1)
    return tuple(
        edge_margin + min(interior - 1, max(0, round((idx + 1) * step))) for idx in range(horizon)
    )


def _window(window_bp: int) -> str:
    _require_positive("window_bp", window_bp)
    repeats = math.ceil(window_bp / 4)
    return ("ACGT" * repeats)[:window_bp]


def _metadata_extra(
    config: PlanningBenchmarkConfig,
    planning_config: PlanningConfig,
) -> dict[str, str]:
    return {
        "horizon": str(planning_config.horizon),
        "n_iterations": str(planning_config.n_iterations),
        "n_samples": str(planning_config.n_samples),
        "n_elite": str(planning_config.n_elite),
        "seed": str(config.seed),
        "window_bp": str(config.window_bp),
        "edge_margin": str(config.edge_margin),
        "position_bin_bp": str(config.position_bin_bp),
        "target": "deterministic_position_type_distance",
    }


def _hardware_summary() -> dict[str, object]:
    return {
        "platform": _platform.platform(terse=True),
        "system": _platform.system(),
        "machine": _platform.machine(),
        "processor": _platform.processor(),
        "cpu_brand": _cpu_brand(),
        "gpu_names": _gpu_names(),
    }


def _cpu_brand() -> str | None:
    if _platform.system() == "Darwin":
        return _run_text(("sysctl", "-n", "machdep.cpu.brand_string"))
    if _platform.system() == "Linux":
        try:
            text = Path("/proc/cpuinfo").read_text(encoding="utf-8")
        except OSError:
            return None
        for line in text.splitlines():
            if line.lower().startswith("model name"):
                _key, _sep, value = line.partition(":")
                return value.strip() or None
    return None


def _gpu_names() -> tuple[str, ...]:
    output = _run_text(
        (
            "nvidia-smi",
            "--query-gpu=name",
            "--format=csv,noheader",
        )
    )
    if output is None:
        return ()
    return tuple(line.strip() for line in output.splitlines() if line.strip())


def _run_text(command: Sequence[str]) -> str | None:
    try:
        completed = subprocess.run(
            list(command),
            check=True,
            capture_output=True,
            text=True,
        )
    except (FileNotFoundError, subprocess.CalledProcessError):
        return None
    return completed.stdout.strip() or None


def _target_profile_match(
    profile: str,
    hardware: dict[str, object],
) -> bool | None:
    if profile == DEFAULT_TARGET_PROFILE:
        return None
    if profile == "m3-max":
        cpu_brand = hardware.get("cpu_brand")
        return isinstance(cpu_brand, str) and "m3 max" in cpu_brand.lower()
    if profile == "h100":
        gpu_names = hardware.get("gpu_names")
        if not isinstance(gpu_names, tuple):
            return False
        return any("h100" in name.lower() for name in gpu_names)
    raise InputError(
        "unsupported planning target profile",
        details={"profile": profile, "supported": list(SUPPORTED_TARGET_PROFILES)},
    )


def _edit_payload(edit: RelEdit) -> dict[str, object]:
    return {
        "rel_pos": edit.rel_pos,
        "edit_type": edit.edit_type.name,
        "edit_type_id": int(edit.edit_type),
        "ref_bases": edit.ref_bases,
        "alt_bases": edit.alt_bases,
    }


def _positive_median(row: dict[str, object]) -> bool:
    result = row.get("result")
    if not isinstance(result, dict):
        return False
    median = result.get("median_ns")
    return isinstance(median, int | float) and median > 0


def _validate_config(config: PlanningBenchmarkConfig) -> None:
    if not config.horizons:
        raise InputError("at least one planning benchmark horizon is required")
    for horizon in config.horizons:
        _require_positive("horizon", horizon)
    _require_positive("n_samples", config.n_samples)
    _require_positive("n_elite", config.n_elite)
    _require_positive("iters", config.iters)
    _require_nonnegative("warmup", config.warmup)
    _require_positive("window_bp", config.window_bp)
    _require_nonnegative("edge_margin", config.edge_margin)
    _require_positive("position_bin_bp", config.position_bin_bp)
    _require_positive("default_horizon", config.default_horizon)
    _require_positive("default_iterations", config.default_iterations)
    _require_positive("default_samples", config.default_samples)
    _require_positive("default_elite", config.default_elite)
    _require_positive("default_patience", config.default_patience)
    _require_nonnegative_float("default_stopping_eps", config.default_stopping_eps)
    if config.n_elite > config.n_samples:
        raise InputError(
            "n_elite must be <= n_samples",
            details={"n_elite": config.n_elite, "n_samples": config.n_samples},
        )
    if config.default_elite > config.default_samples:
        raise InputError(
            "default_elite must be <= default_samples",
            details={
                "default_elite": config.default_elite,
                "default_samples": config.default_samples,
            },
        )
    if 2 * config.edge_margin >= config.window_bp:
        raise InputError(
            "edge_margin leaves no benchmark window interior",
            details={"window_bp": config.window_bp, "edge_margin": config.edge_margin},
        )
    if config.target_profile not in SUPPORTED_TARGET_PROFILES:
        raise InputError(
            "unsupported planning target profile",
            details={
                "target_profile": config.target_profile,
                "supported": list(SUPPORTED_TARGET_PROFILES),
            },
        )
    if config.target_seconds is not None:
        _require_positive_float("target_seconds", config.target_seconds)


def _require_rows(payload: dict[str, object]) -> tuple[dict[str, object], ...]:
    rows = payload.get("rows")
    if not isinstance(rows, list):
        raise InputError("planning report rows must be a list")
    out: list[dict[str, object]] = []
    for index, row in enumerate(rows):
        if not isinstance(row, dict):
            raise InputError("planning report row must be an object", details={"index": index})
        out.append(row)
    return tuple(out)


def _required_str(payload: dict[str, object], key: str) -> str:
    value = payload.get(key)
    if not isinstance(value, str) or not value:
        raise InputError("planning benchmark result field must be a string", details={"field": key})
    return value


def _require_positive(name: str, value: int) -> None:
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise InputError(
            f"{name} must be a positive integer",
            details={"field": name, "value": value, "type": type(value).__name__},
        )


def _require_nonnegative(name: str, value: int) -> None:
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise InputError(
            f"{name} must be a non-negative integer",
            details={"field": name, "value": value, "type": type(value).__name__},
        )


def _require_positive_float(name: str, value: float) -> None:
    if (
        isinstance(value, bool)
        or not isinstance(value, int | float)
        or not math.isfinite(float(value))
        or float(value) <= 0.0
    ):
        raise InputError(
            f"{name} must be a positive finite number",
            details={"field": name, "value": value, "type": type(value).__name__},
        )


def _require_nonnegative_float(name: str, value: float) -> None:
    if (
        isinstance(value, bool)
        or not isinstance(value, int | float)
        or not math.isfinite(float(value))
        or float(value) < 0.0
    ):
        raise InputError(
            f"{name} must be a non-negative finite number",
            details={"field": name, "value": value, "type": type(value).__name__},
        )


def _sanitize_filename(name: str) -> str:
    return name.replace("/", "__").replace("\\", "__")


def _utc_now() -> str:
    return _dt.datetime.now(tz=_dt.timezone.utc).isoformat(timespec="seconds")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="bench.planning",
        description="Benchmark the pure planning contract CEM planning loop.",
    )
    parser.add_argument("--k", dest="horizons", type=int, action="append")
    parser.add_argument("--samples", type=int, default=256)
    parser.add_argument("--elite", type=int, default=32)
    parser.add_argument("--iters", type=int, default=50)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--window-bp", type=int, default=DEFAULT_WINDOW_BP)
    parser.add_argument("--edge-margin", type=int, default=64)
    parser.add_argument("--position-bin-bp", type=int, default=8)
    parser.add_argument("--default-horizon", type=int, default=5)
    parser.add_argument("--default-iterations", type=int, default=5)
    parser.add_argument("--default-samples", type=int, default=1024)
    parser.add_argument("--default-elite", type=int, default=64)
    parser.add_argument("--default-stopping-eps", type=float, default=0.05)
    parser.add_argument("--default-patience", type=int, default=2)
    parser.add_argument(
        "--target-profile",
        choices=SUPPORTED_TARGET_PROFILES,
        default=DEFAULT_TARGET_PROFILE,
        help="Hardware profile for default-call target validation.",
    )
    parser.add_argument(
        "--target-seconds",
        type=float,
        help="Override the default-call elapsed-time target in seconds.",
    )
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_RESULTS_DIR)
    parser.add_argument("--output-json", "--output", dest="output_json", type=Path)
    parser.add_argument("--no-write", action="store_true")
    parser.add_argument(
        "--require-targets",
        action="store_true",
        help="exit non-zero when the selected default-call target is not met",
    )
    args = parser.parse_args(argv)

    config = PlanningBenchmarkConfig(
        horizons=tuple(args.horizons or DEFAULT_HORIZONS),
        n_samples=args.samples,
        n_elite=args.elite,
        iters=args.iters,
        warmup=args.warmup,
        seed=args.seed,
        window_bp=args.window_bp,
        edge_margin=args.edge_margin,
        position_bin_bp=args.position_bin_bp,
        default_horizon=args.default_horizon,
        default_iterations=args.default_iterations,
        default_samples=args.default_samples,
        default_elite=args.default_elite,
        default_stopping_eps=args.default_stopping_eps,
        default_patience=args.default_patience,
        target_profile=args.target_profile,
        target_seconds=args.target_seconds,
    )
    if (
        args.require_targets
        and config.target_profile == DEFAULT_TARGET_PROFILE
        and (config.target_seconds is None)
    ):
        raise InputError(
            "--require-targets requires --target-profile or --target-seconds",
            remediation="pass --target-profile m3-max, --target-profile h100, or --target-seconds",
        )

    payload = _build_planning_report(config, command=_command_from_args(args))
    for row in _require_rows(payload):
        result = row.get("result")
        if isinstance(result, dict):
            report_to_stdout(_bench_result_from_json(result))
    print(json.dumps(payload, indent=2, sort_keys=True))

    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        print(f"[bench] wrote {args.output_json}")
    elif not args.no_write:
        for path in write_planning_benchmarks(payload, out_dir=args.out_dir):
            print(f"[bench] wrote {path}")
        path = write_planning_report(payload, out_dir=args.out_dir)
        print(f"[bench] wrote {path}")

    if args.require_targets and not bool(payload["ok"]):
        return 1
    return 0


def _bench_result_from_json(payload: dict[str, object]) -> BenchResult:
    from bench._harness import BenchMetadata

    metadata = payload.get("metadata")
    if not isinstance(metadata, dict):
        raise InputError("benchmark result metadata must be an object")
    return BenchResult(
        name=_required_str(payload, "name"),
        iters=_required_int(payload, "iters"),
        warmup=_required_int(payload, "warmup"),
        samples_ns=_required_int_tuple(payload, "samples_ns"),
        median_ns=_required_int(payload, "median_ns"),
        p25_ns=_required_int(payload, "p25_ns"),
        p75_ns=_required_int(payload, "p75_ns"),
        iqr_ns=_required_int(payload, "iqr_ns"),
        metadata=BenchMetadata(
            commit=str(metadata["commit"]),
            timestamp=str(metadata["timestamp"]),
            machine=str(metadata["machine"]),
            python_version=str(metadata["python_version"]),
            platform=str(metadata["platform"]),
            dtype=str(metadata["dtype"]),
            extra=dict(metadata.get("extra") or {}),
        ),
        schema_version=str(payload["schema_version"]),
    )


def _required_int(payload: dict[str, object], key: str) -> int:
    value = payload.get(key)
    if not isinstance(value, int) or isinstance(value, bool):
        raise InputError(
            "planning benchmark result field must be an integer", details={"field": key}
        )
    return value


def _required_int_tuple(payload: dict[str, object], key: str) -> tuple[int, ...]:
    value = payload.get(key)
    if not isinstance(value, list):
        raise InputError("planning benchmark result field must be a list", details={"field": key})
    out: list[int] = []
    for index, item in enumerate(value):
        if not isinstance(item, int) or isinstance(item, bool):
            raise InputError(
                "planning benchmark result samples must be integers",
                details={"field": key, "index": index},
            )
        out.append(item)
    return tuple(out)


def _command_from_args(args: argparse.Namespace) -> tuple[str, ...]:
    command = ["python", "-m", "bench.planning"]
    for horizon in args.horizons or DEFAULT_HORIZONS:
        command.extend(("--k", str(horizon)))
    command.extend(
        (
            "--samples",
            str(args.samples),
            "--elite",
            str(args.elite),
            "--iters",
            str(args.iters),
            "--warmup",
            str(args.warmup),
            "--seed",
            str(args.seed),
            "--window-bp",
            str(args.window_bp),
            "--edge-margin",
            str(args.edge_margin),
            "--position-bin-bp",
            str(args.position_bin_bp),
            "--default-horizon",
            str(args.default_horizon),
            "--default-iterations",
            str(args.default_iterations),
            "--default-samples",
            str(args.default_samples),
            "--default-elite",
            str(args.default_elite),
            "--default-stopping-eps",
            str(args.default_stopping_eps),
            "--default-patience",
            str(args.default_patience),
            "--target-profile",
            str(args.target_profile),
        )
    )
    if args.target_seconds is not None:
        command.extend(("--target-seconds", str(args.target_seconds)))
    if args.output_json is not None:
        command.extend(("--output-json", str(args.output_json)))
    elif args.no_write:
        command.append("--no-write")
    else:
        command.extend(("--out-dir", str(args.out_dir)))
    if args.require_targets:
        command.append("--require-targets")
    return tuple(command)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
