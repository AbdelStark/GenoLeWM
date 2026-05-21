# SPDX-License-Identifier: Apache-2.0
"""Tests for ``tools.ci.perf_regression`` (RFC-0016 §3.7)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from tools.ci import perf_regression


def _write_harness_result(
    dirpath: Path,
    name: str,
    median_ns: int,
    *,
    iters: int = 100,
    machine: str = "m1",
) -> Path:
    (dirpath / machine).mkdir(parents=True, exist_ok=True)
    target = dirpath / machine / f"{name}.json"
    payload = {
        "schema_version": "1.0.0",
        "name": name,
        "iters": iters,
        "warmup": 0,
        "samples_ns": [median_ns] * max(iters, 1),
        "median_ns": median_ns,
        "p25_ns": median_ns,
        "p75_ns": median_ns,
        "iqr_ns": 0,
        "metadata": {
            "commit": "abc",
            "timestamp": "t",
            "machine": machine,
            "python_version": "3.12",
            "platform": "linux",
            "dtype": "n/a",
            "extra": {},
        },
    }
    target.write_text(json.dumps(payload), encoding="utf-8")
    return target


def _write_pytest_benchmark_json(path: Path, entries: dict[str, float]) -> Path:
    payload = {
        "benchmarks": [
            {"name": name, "stats": {"median": seconds}} for name, seconds in entries.items()
        ]
    }
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


# ---------------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------------


def test_load_samples_harness_directory(tmp_path: Path) -> None:
    _write_harness_result(tmp_path, "a", 100)
    _write_harness_result(tmp_path, "b", 200)
    samples = sorted(perf_regression.load_samples(tmp_path), key=lambda s: s.name)
    assert [s.name for s in samples] == ["a", "b"]
    assert samples[0].median_ns == 100


def test_load_samples_pytest_benchmark_file(tmp_path: Path) -> None:
    target = tmp_path / "bench.json"
    _write_pytest_benchmark_json(target, {"x": 0.001, "y": 0.002})
    samples = sorted(perf_regression.load_samples(target), key=lambda s: s.name)
    assert [s.name for s in samples] == ["x", "y"]
    # 1 ms → 1_000_000 ns; 2 ms → 2_000_000 ns.
    assert samples[0].median_ns == pytest.approx(1_000_000.0)
    assert samples[1].median_ns == pytest.approx(2_000_000.0)


def test_load_samples_skips_placeholder_results(tmp_path: Path) -> None:
    # iters == 0 (planning placeholder) should be skipped.
    _write_harness_result(tmp_path, "planning.cem", 0, iters=0)
    samples = list(perf_regression.load_samples(tmp_path))
    assert samples == []


def test_load_samples_skips_unknown_shapes(tmp_path: Path) -> None:
    (tmp_path / "junk.json").write_text(json.dumps({"hello": "world"}), encoding="utf-8")
    assert list(perf_regression.load_samples(tmp_path)) == []


def test_load_samples_tolerates_malformed_json(tmp_path: Path) -> None:
    (tmp_path / "broken.json").write_text("{not-json", encoding="utf-8")
    _write_harness_result(tmp_path, "good", 100)
    samples = list(perf_regression.load_samples(tmp_path))
    assert [s.name for s in samples] == ["good"]


# ---------------------------------------------------------------------------
# compare()
# ---------------------------------------------------------------------------


def test_compare_pairs_by_name() -> None:
    src = Path("dummy")
    current = [
        perf_regression.BenchSample(name="a", median_ns=120, source=src),
        perf_regression.BenchSample(name="b", median_ns=200, source=src),
        perf_regression.BenchSample(name="new", median_ns=50, source=src),
    ]
    baseline = [
        perf_regression.BenchSample(name="a", median_ns=100, source=src),
        perf_regression.BenchSample(name="b", median_ns=200, source=src),
        perf_regression.BenchSample(name="dropped", median_ns=10, source=src),
    ]
    paired, missing_baseline, missing_current = perf_regression.compare(current, baseline)
    assert [c.name for c in paired] == ["a", "b"]
    assert paired[0].ratio == pytest.approx(1.2)
    assert paired[1].ratio == pytest.approx(1.0)
    assert missing_baseline == ["new"]
    assert missing_current == ["dropped"]


def test_compare_skips_zero_baseline() -> None:
    src = Path("dummy")
    current = [perf_regression.BenchSample(name="a", median_ns=100, source=src)]
    baseline = [perf_regression.BenchSample(name="a", median_ns=0, source=src)]
    paired, _, _ = perf_regression.compare(current, baseline)
    assert paired == []


# ---------------------------------------------------------------------------
# main()
# ---------------------------------------------------------------------------


def test_main_no_regression(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    base = tmp_path / "base"
    curr = tmp_path / "curr"
    _write_harness_result(base, "a", 1000)
    _write_harness_result(curr, "a", 1030)  # 3 % over baseline → passes at 5 %.
    code = perf_regression.main(
        ["--current", str(curr), "--baseline", str(base), "--threshold", "0.05"]
    )
    captured = capsys.readouterr()
    assert code == 0
    assert "ok" in captured.out


def test_main_detects_regression(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    base = tmp_path / "base"
    curr = tmp_path / "curr"
    _write_harness_result(base, "a", 1000)
    _write_harness_result(curr, "a", 1100)  # 10 % over baseline → regress at 5 %.
    code = perf_regression.main(
        ["--current", str(curr), "--baseline", str(base), "--threshold", "0.05"]
    )
    captured = capsys.readouterr()
    assert code == 1
    assert "REGRESS" in captured.out
    assert "1 regression" in captured.err


def test_main_missing_baseline_is_warmup(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    curr = tmp_path / "curr"
    _write_harness_result(curr, "a", 1000)
    code = perf_regression.main(
        ["--current", str(curr), "--baseline", str(tmp_path / "no_baseline")]
    )
    captured = capsys.readouterr()
    assert code == 0
    assert "no baseline" in captured.err


def test_main_missing_current_is_error(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    code = perf_regression.main(
        ["--current", str(tmp_path / "missing"), "--baseline", str(tmp_path)]
    )
    captured = capsys.readouterr()
    assert code == 2
    assert "does not exist" in captured.err


def test_main_empty_current_is_error(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    base = tmp_path / "base"
    curr = tmp_path / "curr"
    base.mkdir()
    curr.mkdir()
    code = perf_regression.main(["--current", str(curr), "--baseline", str(base)])
    captured = capsys.readouterr()
    assert code == 2
    assert "no samples" in captured.err


def test_main_new_benchmark_passes(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    base = tmp_path / "base"
    curr = tmp_path / "curr"
    base.mkdir()
    _write_harness_result(curr, "new", 5000)
    code = perf_regression.main(["--current", str(curr), "--baseline", str(base)])
    captured = capsys.readouterr()
    assert code == 0
    assert "New benchmarks (no baseline yet)" in captured.out
    assert "new" in captured.out


# ---------------------------------------------------------------------------
# Report formatting
# ---------------------------------------------------------------------------


def test_format_report_humanizes_units() -> None:
    paired = [
        perf_regression.Comparison(name="a", current_ns=1_200, baseline_ns=1_000, ratio=1.2),
    ]
    report = perf_regression.format_report(paired, [], [], threshold=0.05)
    assert "1.20 µs" in report
    assert "+20.0%" in report
    assert "REGRESS" in report
