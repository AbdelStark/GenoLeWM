# SPDX-License-Identifier: Apache-2.0
"""Autoregressive rollout benchmarks for predictor contract speed targets.

The benchmark compares the current ``ARPredictor`` rollout path against
the naive repeated one-step ``Predictor.forward`` loop. It records the
measured speedup for each requested horizon and can optionally fail when
the predictor contract targets are not met.
"""

from __future__ import annotations

import argparse
import datetime as _dt
import json
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from bench._harness import (
    DEFAULT_RESULTS_DIR,
    BenchResult,
    current_commit,
    machine_id,
    time_callable,
)
from geno_lewm.errors import InputError
from geno_lewm.provenance import canonical_json_sha256

GENERATED_BY = "bench.rollout"
SCHEMA_VERSION = "1.0.0"
BENCHMARK_RESULT_SCHEMA_VERSION = "1.0.0"
DEFAULT_HORIZONS = (5, 20)


@dataclass(frozen=True, slots=True)
class RolloutBenchmarkConfig:
    """Configuration for the AR rollout speed benchmark."""

    horizons: tuple[int, ...] = DEFAULT_HORIZONS
    batch_size: int = 4
    d_state: int = 64
    d_action: int = 32
    d_hidden: int = 64
    n_heads: int = 4
    n_cross_layers: int = 2
    n_self_layers: int = 1
    ffn_dim: int = 128
    iters: int = 30
    warmup: int = 5
    seed: int = 20260606
    device: str = "cpu"
    dtype: str = "fp32"


def build_rollout_speed_report(config: RolloutBenchmarkConfig) -> dict[str, object]:
    """Run the rollout benchmark and return a machine-readable report."""
    return _build_rollout_speed_report(config, command=())


def _build_rollout_speed_report(
    config: RolloutBenchmarkConfig,
    *,
    command: Sequence[str],
) -> dict[str, object]:
    _validate_config(config)
    torch = _load_torch()
    from geno_lewm.predictor import ARPredictor, Predictor

    torch.manual_seed(config.seed)
    predictor: Any = Predictor(
        d_state=config.d_state,
        d_action=config.d_action,
        d_hidden=config.d_hidden,
        n_heads=config.n_heads,
        n_cross_layers=config.n_cross_layers,
        n_self_layers=config.n_self_layers,
        ffn_dim=config.ffn_dim,
        max_actions=max(config.horizons),
    )
    predictor = predictor.to(config.device)
    predictor.eval()
    rollout = ARPredictor(predictor)

    rows = []
    for horizon in config.horizons:
        state = torch.nn.functional.normalize(
            torch.randn(config.batch_size, config.d_state, device=config.device),
            dim=-1,
        )
        actions = torch.randn(
            config.batch_size,
            horizon,
            config.d_action,
            device=config.device,
        )
        if config.dtype == "fp16":
            predictor = predictor.half()
            state = state.half()
            actions = actions.half()
            rollout = ARPredictor(predictor)
        elif config.dtype == "bf16":
            predictor = predictor.bfloat16()
            state = state.bfloat16()
            actions = actions.bfloat16()
            rollout = ARPredictor(predictor)

        def cached_workload(
            *,
            rollout_obj: Any = rollout,
            state_tensor: Any = state,
            action_tensor: Any = actions,
        ) -> Any:
            return rollout_obj.rollout_tensor(state_tensor, action_tensor)

        def naive_workload(
            *,
            predictor_obj: Any = predictor,
            state_tensor: Any = state,
            action_tensor: Any = actions,
        ) -> Any:
            return _naive_rollout_tensor(torch, predictor_obj, state_tensor, action_tensor)

        cached = time_callable(
            f"rollout.ar_cached.k{horizon}",
            cached_workload,
            iters=config.iters,
            warmup=config.warmup,
            dtype=config.dtype,
            extra=_metadata_extra(config, horizon),
        )
        naive = time_callable(
            f"rollout.ar_naive.k{horizon}",
            naive_workload,
            iters=config.iters,
            warmup=config.warmup,
            dtype=config.dtype,
            extra=_metadata_extra(config, horizon),
        )
        rows.append(summarize_speed_row(horizon=horizon, naive=naive, cached=cached))

    payload: dict[str, object] = {
        "schema_version": SCHEMA_VERSION,
        "generated_by": GENERATED_BY,
        "generated_at": _utc_now(),
        "commit": current_commit(),
        "machine": machine_id(),
        "command": list(command),
        "config": asdict(config),
        "rows": rows,
        "ok": all(bool(row["target_met"]) for row in rows),
        "claim_boundary": (
            "This benchmark measures local predictor rollout speed only; it is not "
            "model-quality, clinical, privacy, or release-readiness evidence."
        ),
    }
    payload["benchmark_result_sha256"] = rollout_benchmark_result_sha256(payload)
    return payload


def rollout_benchmark_result_payload(payload: dict[str, object]) -> dict[str, object]:
    """Return the timing-result identity payload, excluding run metadata."""
    rows = payload.get("rows")
    if not isinstance(rows, list):
        raise InputError("rollout benchmark report rows must be a list")
    return {
        "schema_version": BENCHMARK_RESULT_SCHEMA_VERSION,
        "generated_by": _required_report_text(payload, "generated_by"),
        "config": payload.get("config"),
        "command": payload.get("command"),
        "ok": payload.get("ok"),
        "claim_boundary": _required_report_text(payload, "claim_boundary"),
        "rows": [_rollout_row_identity(row) for row in rows],
    }


def rollout_benchmark_result_sha256(payload: dict[str, object]) -> str:
    """Hash the timing-result identity payload with canonical JSON."""
    return canonical_json_sha256(rollout_benchmark_result_payload(payload))


def summarize_speed_row(
    *,
    horizon: int,
    naive: BenchResult,
    cached: BenchResult,
) -> dict[str, object]:
    """Summarize one horizon's speedup from paired benchmark results."""
    if cached.median_ns <= 0 or naive.median_ns <= 0:
        raise InputError("rollout benchmark medians must be positive")
    speedup = naive.median_ns / cached.median_ns
    target = target_speedup_for_horizon(horizon)
    return {
        "horizon": horizon,
        "target_speedup": target,
        "measured_speedup": speedup,
        "target_met": speedup >= target,
        "cached": cached.to_json(),
        "naive": naive.to_json(),
    }


def _rollout_row_identity(row: object) -> dict[str, object]:
    if not isinstance(row, dict):
        raise InputError("rollout benchmark rows must be objects")
    return {
        "horizon": row.get("horizon"),
        "target_speedup": row.get("target_speedup"),
        "measured_speedup": row.get("measured_speedup"),
        "target_met": row.get("target_met"),
        "cached": _bench_result_identity(row.get("cached")),
        "naive": _bench_result_identity(row.get("naive")),
    }


def _bench_result_identity(raw: object) -> dict[str, object]:
    if not isinstance(raw, dict):
        raise InputError("rollout benchmark row cached/naive entries must be objects")
    return {
        "schema_version": raw.get("schema_version"),
        "name": raw.get("name"),
        "iters": raw.get("iters"),
        "warmup": raw.get("warmup"),
        "samples_ns": raw.get("samples_ns"),
        "median_ns": raw.get("median_ns"),
        "p25_ns": raw.get("p25_ns"),
        "p75_ns": raw.get("p75_ns"),
        "iqr_ns": raw.get("iqr_ns"),
    }


def _required_report_text(payload: dict[str, object], key: str) -> str:
    value = payload.get(key)
    if not isinstance(value, str) or not value:
        raise InputError(f"rollout benchmark report {key} must be a non-empty string")
    return value


def target_speedup_for_horizon(horizon: int) -> float:
    """Return the predictor contract speedup target for a rollout horizon."""
    if horizon >= 20:
        return 5.0
    if horizon >= 5:
        return 2.0
    return 1.0


def write_rollout_speed_report(
    payload: dict[str, object],
    *,
    out_dir: Path = DEFAULT_RESULTS_DIR,
) -> Path:
    """Persist a rollout speed report under the benchmark results tree."""
    machine = str(payload.get("machine") or machine_id())
    target_dir = out_dir / machine
    target_dir.mkdir(parents=True, exist_ok=True)
    path = target_dir / "rollout.ar_speed.json"
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def _naive_rollout_tensor(torch: Any, predictor: Any, state: Any, actions: Any) -> Any:
    mask = torch.ones(actions.shape[0], 1, dtype=torch.bool, device=actions.device)
    current = state
    outputs = []
    with torch.no_grad():
        for step in range(actions.shape[1]):
            prediction = predictor(current, actions[:, step : step + 1, :], mask)[:, 0, :]
            outputs.append(prediction)
            current = prediction
        return torch.stack(outputs, dim=1)


def _metadata_extra(config: RolloutBenchmarkConfig, horizon: int) -> dict[str, str]:
    return {
        "horizon": str(horizon),
        "batch_size": str(config.batch_size),
        "d_state": str(config.d_state),
        "d_action": str(config.d_action),
        "d_hidden": str(config.d_hidden),
        "n_heads": str(config.n_heads),
        "n_cross_layers": str(config.n_cross_layers),
        "n_self_layers": str(config.n_self_layers),
        "ffn_dim": str(config.ffn_dim),
    }


def _validate_config(config: RolloutBenchmarkConfig) -> None:
    if not config.horizons:
        raise InputError("at least one rollout horizon is required")
    for horizon in config.horizons:
        _require_positive("horizon", horizon)
    _require_positive("batch_size", config.batch_size)
    _require_positive("d_state", config.d_state)
    _require_positive("d_action", config.d_action)
    _require_positive("d_hidden", config.d_hidden)
    _require_positive("n_heads", config.n_heads)
    _require_positive("n_cross_layers", config.n_cross_layers)
    _require_positive("n_self_layers", config.n_self_layers)
    _require_positive("ffn_dim", config.ffn_dim)
    _require_positive("iters", config.iters)
    if config.warmup < 0:
        raise InputError("warmup must be non-negative", details={"warmup": config.warmup})
    if config.d_hidden % config.n_heads != 0:
        raise InputError(
            "d_hidden must be divisible by n_heads",
            details={"d_hidden": config.d_hidden, "n_heads": config.n_heads},
        )
    if config.dtype not in {"fp32", "fp16", "bf16"}:
        raise InputError(
            "unsupported rollout benchmark dtype",
            details={"dtype": config.dtype, "supported": ["fp32", "fp16", "bf16"]},
        )


def _require_positive(name: str, value: int) -> None:
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise InputError(
            f"{name} must be a positive integer",
            details={"field": name, "value": value, "type": type(value).__name__},
        )


def _load_torch() -> Any:
    try:
        import torch  # type: ignore[import-not-found]
    except ImportError as exc:  # pragma: no cover - depends on optional runtime.
        raise InputError(
            "bench.rollout requires PyTorch",
            remediation="run with `uv run --extra train python -m bench.rollout`",
        ) from exc
    return torch


def _utc_now() -> str:
    return _dt.datetime.now(tz=_dt.timezone.utc).isoformat(timespec="seconds")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="bench.rollout",
        description="Benchmark AR rollout speed against repeated one-step forward calls.",
    )
    parser.add_argument("--k", dest="horizons", type=int, action="append")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--d-state", type=int, default=64)
    parser.add_argument("--d-action", type=int, default=32)
    parser.add_argument("--d-hidden", type=int, default=64)
    parser.add_argument("--n-heads", type=int, default=4)
    parser.add_argument("--n-cross-layers", type=int, default=2)
    parser.add_argument("--n-self-layers", type=int, default=1)
    parser.add_argument("--ffn-dim", type=int, default=128)
    parser.add_argument("--iters", type=int, default=30)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--seed", type=int, default=20260606)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--dtype", default="fp32", choices=["fp32", "fp16", "bf16"])
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_RESULTS_DIR)
    parser.add_argument("--output-json", type=Path)
    parser.add_argument("--no-write", action="store_true")
    parser.add_argument(
        "--require-targets",
        action="store_true",
        help="exit non-zero when any requested horizon misses the predictor contract speedup target",
    )
    args = parser.parse_args(argv)

    config = RolloutBenchmarkConfig(
        horizons=tuple(args.horizons or DEFAULT_HORIZONS),
        batch_size=args.batch_size,
        d_state=args.d_state,
        d_action=args.d_action,
        d_hidden=args.d_hidden,
        n_heads=args.n_heads,
        n_cross_layers=args.n_cross_layers,
        n_self_layers=args.n_self_layers,
        ffn_dim=args.ffn_dim,
        iters=args.iters,
        warmup=args.warmup,
        seed=args.seed,
        device=args.device,
        dtype=args.dtype,
    )
    payload = _build_rollout_speed_report(config, command=_command_from_args(args))
    print(json.dumps(payload, indent=2, sort_keys=True))
    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        print(f"[bench] wrote {args.output_json}")
    elif not args.no_write:
        path = write_rollout_speed_report(payload, out_dir=args.out_dir)
        print(f"[bench] wrote {path}")
    if args.require_targets and not bool(payload["ok"]):
        return 1
    return 0


def _command_from_args(args: argparse.Namespace) -> tuple[str, ...]:
    command = ["python", "-m", "bench.rollout"]
    for horizon in args.horizons or DEFAULT_HORIZONS:
        command.extend(("--k", str(horizon)))
    command.extend(
        (
            "--batch-size",
            str(args.batch_size),
            "--d-state",
            str(args.d_state),
            "--d-action",
            str(args.d_action),
            "--d-hidden",
            str(args.d_hidden),
            "--n-heads",
            str(args.n_heads),
            "--n-cross-layers",
            str(args.n_cross_layers),
            "--n-self-layers",
            str(args.n_self_layers),
            "--ffn-dim",
            str(args.ffn_dim),
            "--iters",
            str(args.iters),
            "--warmup",
            str(args.warmup),
            "--seed",
            str(args.seed),
            "--device",
            str(args.device),
            "--dtype",
            str(args.dtype),
        )
    )
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
