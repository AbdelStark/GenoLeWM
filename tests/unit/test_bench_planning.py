"""Tests for the CEM planning benchmark report."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from bench.planning import (
    PlanningBenchmarkConfig,
    _command_from_args,
    _target_profile_match,
    build_planning_report,
    main,
    target_seconds_for_profile,
    write_planning_benchmarks,
    write_planning_report,
)
from geno_lewm.errors import InputError


def test_build_planning_report_records_real_cem_timings() -> None:
    report = build_planning_report(
        PlanningBenchmarkConfig(
            horizons=(1,),
            n_samples=8,
            n_elite=2,
            iters=2,
            warmup=0,
            window_bp=256,
            edge_margin=4,
            default_horizon=2,
            default_iterations=2,
            default_samples=12,
            default_elite=3,
            default_patience=2,
            target_seconds=30.0,
        )
    )

    assert report["generated_by"] == "bench.planning"
    assert report["ok"] is True
    assert "not_implemented" not in json.dumps(report)
    rows = report["rows"]
    assert isinstance(rows, list)
    assert rows[0]["benchmark"] == "planning.cem_iter.k1"
    assert rows[0]["median_seconds"] > 0
    default = report["default_call"]
    assert default["target_seconds"] == 30.0
    assert default["target_met"] is True
    assert default["deterministic_match"] is True
    assert default["result"]["n_evaluations"] == 24


def test_write_planning_outputs(tmp_path: Path) -> None:
    report = build_planning_report(
        PlanningBenchmarkConfig(
            horizons=(1,),
            n_samples=4,
            n_elite=1,
            iters=1,
            warmup=0,
            window_bp=128,
            edge_margin=4,
            default_horizon=1,
            default_iterations=1,
            default_samples=4,
            default_elite=1,
        )
    )

    benchmark_paths = write_planning_benchmarks(report, out_dir=tmp_path)
    report_path = write_planning_report(report, out_dir=tmp_path)

    assert len(benchmark_paths) == 1
    assert benchmark_paths[0].name == "planning.cem_iter.k1.json"
    assert json.loads(benchmark_paths[0].read_text(encoding="utf-8"))["name"] == (
        "planning.cem_iter.k1"
    )
    assert report_path.name == "planning.performance.json"
    assert json.loads(report_path.read_text(encoding="utf-8"))["generated_by"] == ("bench.planning")


def test_target_seconds_for_profile() -> None:
    assert target_seconds_for_profile("none") is None
    assert target_seconds_for_profile("h100") == 1.0
    assert target_seconds_for_profile("m3-max") == 30.0
    with pytest.raises(InputError):
        target_seconds_for_profile("rtx-4090")


def test_target_profile_match_requires_named_hardware() -> None:
    assert (
        _target_profile_match(
            "m3-max",
            {"cpu_brand": "Apple M3 Max", "gpu_names": ()},
        )
        is True
    )
    assert (
        _target_profile_match(
            "h100",
            {"cpu_brand": "Intel", "gpu_names": ("NVIDIA H100 80GB HBM3",)},
        )
        is True
    )
    assert (
        _target_profile_match(
            "m3-max",
            {"cpu_brand": "Apple M4 Max", "gpu_names": ()},
        )
        is False
    )


def test_planning_command_records_explicit_invocation() -> None:
    args = SimpleNamespace(
        horizons=[1, 3],
        samples=16,
        elite=4,
        iters=3,
        warmup=1,
        seed=20260609,
        window_bp=512,
        edge_margin=8,
        position_bin_bp=4,
        default_horizon=5,
        default_iterations=5,
        default_samples=1024,
        default_elite=64,
        default_stopping_eps=0.05,
        default_patience=2,
        target_profile="m3-max",
        target_seconds=None,
        output_json=Path("planning.performance.json"),
        no_write=False,
        out_dir=Path("bench/results"),
        require_targets=True,
    )

    command = _command_from_args(args)

    assert command[:3] == ("python", "-m", "bench.planning")
    assert command.count("--k") == 2
    assert "--target-profile" in command
    assert "m3-max" in command
    assert "--output-json" in command
    assert "--require-targets" in command


def test_planning_main_accepts_readme_output_alias(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    output = tmp_path / "planning.performance.json"

    rc = main(
        [
            "--k",
            "1",
            "--samples",
            "4",
            "--elite",
            "1",
            "--iters",
            "1",
            "--warmup",
            "0",
            "--window-bp",
            "128",
            "--edge-margin",
            "4",
            "--default-horizon",
            "1",
            "--default-iterations",
            "1",
            "--default-samples",
            "4",
            "--default-elite",
            "1",
            "--output",
            str(output),
        ]
    )
    captured = capsys.readouterr()

    assert rc == 0
    assert output.is_file()
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["generated_by"] == "bench.planning"
    assert "--output-json" in payload["command"]
    assert str(output) in payload["command"]
    assert f"[bench] wrote {output}" in captured.out
