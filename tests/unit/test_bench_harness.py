# SPDX-License-Identifier: Apache-2.0
"""Tests for ``bench._harness`` (RFC-0016 §3.4).

Covers the public surface of the benchmark library: result schema,
percentile helper, ``time_callable`` smoke, machine fingerprinting,
JSON persistence, and the human-readable formatter.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from bench import _harness

# ---------------------------------------------------------------------------
# BenchResult / BenchMetadata
# ---------------------------------------------------------------------------


def test_bench_result_to_json_round_trips() -> None:
    meta = _harness.BenchMetadata(
        commit="abc1234",
        timestamp="2026-05-21T00:00:00+00:00",
        machine="test-machine",
        python_version="3.12.5",
        platform="linux-x86_64",
        dtype="bf16",
        extra={"window_bytes": "4096"},
    )
    result = _harness.BenchResult(
        name="x.y",
        iters=10,
        warmup=2,
        samples_ns=(100, 110, 120, 130, 140),
        median_ns=120,
        p25_ns=110,
        p75_ns=130,
        iqr_ns=20,
        metadata=meta,
    )
    encoded = json.dumps(result.to_json())
    decoded = json.loads(encoded)
    assert decoded["name"] == "x.y"
    assert decoded["schema_version"] == _harness.RESULT_SCHEMA_VERSION
    assert decoded["metadata"]["machine"] == "test-machine"
    assert decoded["metadata"]["extra"] == {"window_bytes": "4096"}


# ---------------------------------------------------------------------------
# Percentiles
# ---------------------------------------------------------------------------


def test_percentile_empty_raises() -> None:
    with pytest.raises(ValueError, match="empty"):
        _harness._percentile([], 50)


def test_percentile_single() -> None:
    assert _harness._percentile([42], 50) == 42


def test_percentile_exact_rank() -> None:
    # 5 samples; rank for 50% = 2.0 exactly → samples[2].
    assert _harness._percentile([10, 20, 30, 40, 50], 50) == 30


def test_percentile_interpolated() -> None:
    # 2 samples, p50 → halfway = 15.
    assert _harness._percentile([10, 20], 50) == 15


def test_median_iqr_helper() -> None:
    median, iqr = _harness.median_iqr([100, 200, 300, 400, 500])
    assert median == 300
    # P75 - P25 over a uniform sequence of 5 samples = 200.
    assert iqr == 200


def test_median_iqr_empty_raises() -> None:
    with pytest.raises(ValueError, match="empty"):
        _harness.median_iqr([])


# ---------------------------------------------------------------------------
# time_callable
# ---------------------------------------------------------------------------


def test_time_callable_runs_required_iterations() -> None:
    calls = 0

    def fn() -> None:
        nonlocal calls
        calls += 1

    result = _harness.time_callable("noop", fn, iters=15, warmup=3)
    assert calls == 18  # warmup + iters
    assert result.iters == 15
    assert result.warmup == 3
    assert len(result.samples_ns) == 15
    assert result.median_ns >= 0
    assert result.iqr_ns >= 0
    assert result.metadata.machine != ""


def test_time_callable_rejects_zero_iters() -> None:
    with pytest.raises(ValueError, match="iters"):
        _harness.time_callable("x", lambda: None, iters=0, warmup=0)


def test_time_callable_rejects_negative_warmup() -> None:
    with pytest.raises(ValueError, match="warmup"):
        _harness.time_callable("x", lambda: None, iters=1, warmup=-1)


# ---------------------------------------------------------------------------
# Machine fingerprinting
# ---------------------------------------------------------------------------


def test_machine_id_from_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("GENO_LEWM_BENCH_MACHINE", "h100-node-7")
    assert _harness.machine_id() == "h100-node-7"


def test_machine_id_sanitises(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("GENO_LEWM_BENCH_MACHINE", "M3 Max / lab #7")
    out = _harness.machine_id()
    assert "/" not in out
    assert "#" not in out
    assert " " not in out
    # Stripped, but content preserved as ASCII-safe slug.
    assert "M3-Max" in out
    assert "lab-7" in out


def test_machine_id_falls_back_to_platform(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("GENO_LEWM_BENCH_MACHINE", raising=False)
    monkeypatch.setattr("platform.node", lambda: "localhost")
    out = _harness.machine_id()
    assert out  # non-empty
    assert "-" in out  # system-machine fallback


def test_sanitize_empty_returns_unknown() -> None:
    assert _harness._sanitize("####") == "unknown"


def test_current_commit_in_repo() -> None:
    sha = _harness.current_commit()
    # In this repo, current_commit should resolve. CI runners with
    # detached HEAD still return a value.
    assert sha != ""


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------


def test_write_result_writes_under_machine_dir(tmp_path: Path) -> None:
    meta = _harness.BenchMetadata(
        commit="abc",
        timestamp="2026-05-21T00:00:00+00:00",
        machine="m1",
        python_version="3.12",
        platform="linux",
        dtype="bf16",
        extra={},
    )
    result = _harness.BenchResult(
        name="inference.input_commitment",
        iters=1,
        warmup=0,
        samples_ns=(100,),
        median_ns=100,
        p25_ns=100,
        p75_ns=100,
        iqr_ns=0,
        metadata=meta,
    )
    path = _harness.write_result(result, out_dir=tmp_path)
    assert path == tmp_path / "m1" / "inference.input_commitment.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    assert data["name"] == "inference.input_commitment"


def test_write_result_sanitises_filename(tmp_path: Path) -> None:
    meta = _harness.BenchMetadata(
        commit="abc",
        timestamp="t",
        machine="m1",
        python_version="3.12",
        platform="linux",
        dtype="n/a",
        extra={},
    )
    # A name containing a slash should map to a single file, not a nested dir.
    result = _harness.BenchResult(
        name="a/b/c",
        iters=1,
        warmup=0,
        samples_ns=(0,),
        median_ns=0,
        p25_ns=0,
        p75_ns=0,
        iqr_ns=0,
        metadata=meta,
    )
    path = _harness.write_result(result, out_dir=tmp_path)
    assert path.name == "a__b__c.json"
    assert path.parent == tmp_path / "m1"


# ---------------------------------------------------------------------------
# Reporting helpers
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("ns", "expected"),
    [
        (500, "500 ns"),
        (1_500, "1.50 µs"),
        (1_500_000, "1.50 ms"),
        (1_500_000_000, "1.50 s"),
    ],
)
def test_humanize_ns(ns: int, expected: str) -> None:
    assert _harness.humanize_ns(ns) == expected


def test_report_to_stdout_writes_to_stream(tmp_path: Path) -> None:
    import io

    meta = _harness.BenchMetadata(
        commit="abc",
        timestamp="t",
        machine="m1",
        python_version="3.12",
        platform="linux",
        dtype="bf16",
        extra={},
    )
    result = _harness.BenchResult(
        name="x",
        iters=1,
        warmup=0,
        samples_ns=(1500,),
        median_ns=1500,
        p25_ns=1500,
        p75_ns=1500,
        iqr_ns=0,
        metadata=meta,
    )
    buf = io.StringIO()
    _harness.report_to_stdout(result, stream=buf)
    text = buf.getvalue()
    assert "x" in text
    assert "1.50 µs" in text
    assert "m1" in text
