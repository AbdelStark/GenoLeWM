"""Tests for the AR rollout benchmark report helpers."""

from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import pytest

from bench._harness import BenchMetadata, BenchResult
from bench.rollout import (
    RolloutBenchmarkConfig,
    _command_from_args,
    rollout_benchmark_result_payload,
    rollout_benchmark_result_sha256,
    summarize_speed_row,
    target_speedup_for_horizon,
    write_rollout_speed_report,
)
from geno_lewm.errors import InputError


def test_target_speedup_for_horizon_matches_rfc_thresholds() -> None:
    assert target_speedup_for_horizon(1) == 1.0
    assert target_speedup_for_horizon(5) == 2.0
    assert target_speedup_for_horizon(19) == 2.0
    assert target_speedup_for_horizon(20) == 5.0


def test_summarize_speed_row_records_pass_fail() -> None:
    row = summarize_speed_row(
        horizon=5,
        naive=_bench_result("naive", 2_000),
        cached=_bench_result("cached", 1_000),
    )
    assert row["target_speedup"] == 2.0
    assert row["measured_speedup"] == 2.0
    assert row["target_met"] is True

    failed = summarize_speed_row(
        horizon=20,
        naive=_bench_result("naive", 4_000),
        cached=_bench_result("cached", 1_000),
    )
    assert failed["target_speedup"] == 5.0
    assert failed["measured_speedup"] == 4.0
    assert failed["target_met"] is False


def test_summarize_speed_row_rejects_non_positive_medians() -> None:
    with pytest.raises(InputError):
        summarize_speed_row(
            horizon=5,
            naive=_bench_result("naive", 2_000),
            cached=_bench_result("cached", 0),
        )


def test_rollout_benchmark_result_hash_excludes_run_metadata() -> None:
    payload = _rollout_payload()
    rerun = deepcopy(payload)
    rerun["generated_at"] = "2026-06-07T00:00:00+00:00"
    rerun["commit"] = "different"
    rerun["machine"] = "different-host"
    rerun["rows"][0]["cached"]["metadata"]["timestamp"] = "2026-06-07T00:00:00+00:00"
    rerun["rows"][0]["cached"]["metadata"]["machine"] = "different-host"
    rerun["rows"][0]["naive"]["metadata"]["commit"] = "different"

    assert rollout_benchmark_result_payload(payload)["schema_version"] == "1.0.0"
    assert rollout_benchmark_result_sha256(payload).startswith("sha256:")
    assert rollout_benchmark_result_sha256(payload) == rollout_benchmark_result_sha256(rerun)

    changed = _rollout_payload()
    changed["rows"][0]["cached"]["median_ns"] = 900
    assert rollout_benchmark_result_sha256(changed) != rollout_benchmark_result_sha256(payload)


def test_write_rollout_speed_report(tmp_path: Path) -> None:
    payload = {
        "generated_by": "bench.rollout",
        "machine": "test-machine",
        "rows": [],
        "ok": False,
    }

    path = write_rollout_speed_report(payload, out_dir=tmp_path)

    assert path == tmp_path / "test-machine" / "rollout.ar_speed.json"
    assert json.loads(path.read_text(encoding="utf-8"))["generated_by"] == "bench.rollout"


def test_rollout_benchmark_config_is_constructible() -> None:
    config = RolloutBenchmarkConfig(horizons=(5, 20), batch_size=2)
    assert config.horizons == (5, 20)
    assert config.batch_size == 2


def test_rollout_command_records_explicit_invocation() -> None:
    args = SimpleNamespace(
        horizons=[5, 20],
        batch_size=2,
        d_state=16,
        d_action=8,
        d_hidden=16,
        n_heads=4,
        n_cross_layers=2,
        n_self_layers=1,
        ffn_dim=32,
        iters=30,
        warmup=5,
        seed=20260606,
        device="cpu",
        dtype="fp32",
        output_json=Path("rollout.ar_speed.json"),
        no_write=False,
        out_dir=Path("bench/results"),
        require_targets=True,
    )

    command = _command_from_args(args)

    assert command[:3] == ("python", "-m", "bench.rollout")
    assert command.count("--k") == 2
    assert "--output-json" in command
    assert "--require-targets" in command


def _bench_result(name: str, median_ns: int) -> BenchResult:
    return BenchResult(
        name=name,
        iters=3,
        warmup=1,
        samples_ns=(median_ns, median_ns, median_ns),
        median_ns=median_ns,
        p25_ns=median_ns,
        p75_ns=median_ns,
        iqr_ns=0,
        metadata=BenchMetadata(
            commit="abc1234",
            timestamp="2026-06-06T00:00:00+00:00",
            machine="unit",
            python_version="3.12",
            platform="test",
            dtype="fp32",
            extra={},
        ),
    )


def _rollout_payload() -> dict[str, object]:
    return {
        "schema_version": "1.0.0",
        "generated_by": "bench.rollout",
        "generated_at": "2026-06-06T00:00:00+00:00",
        "commit": "abc1234",
        "machine": "unit",
        "command": ["python", "-m", "bench.rollout"],
        "config": {"horizons": [5], "dtype": "fp32"},
        "rows": [
            summarize_speed_row(
                horizon=5,
                naive=_bench_result("naive", 2_000),
                cached=_bench_result("cached", 1_000),
            )
        ],
        "ok": True,
        "claim_boundary": (
            "This benchmark measures local predictor rollout speed only; it is not "
            "model-quality, clinical, privacy, or release-readiness evidence."
        ),
    }
