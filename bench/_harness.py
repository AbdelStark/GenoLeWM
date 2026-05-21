# SPDX-License-Identifier: Apache-2.0
"""Shared benchmark harness library (RFC-0016 §3.4).

A benchmark is a callable timed over ``iters`` iterations after ``warmup``
warmups. The result captures median + IQR (P25/P75) in nanoseconds plus
metadata (commit, machine fingerprint, Python version, platform). Results
are persisted at ``bench/results/<machine>/<benchmark>.json`` so the
regression detector (``tools/ci/perf_regression.py``) can diff them.

This module is intentionally stdlib-only. Per-target benchmark scripts
that need ``torch`` / ``numpy`` / ``transformers`` should import those
lazily so a CPU-only laptop can still run the harness.

Usage::

    from bench._harness import BenchResult, time_callable, write_result


    def workload() -> None: ...


    result = time_callable("inference.commitment", workload, iters=200)
    write_result(result)

Determinism and noise:

- ``time_callable`` uses :func:`time.perf_counter_ns` and a small
  per-call closure so the python-level overhead is the same on every
  sample. The function under test owns its own warmth.
- The metadata block records the host's CPU brand, RAM (best effort),
  the running Python version, the platform string, the dtype hint
  (caller-provided), and the current git HEAD if a repository is
  available. Hardware is identified by stable strings so the harness
  remains reproducible across CI runners.
"""

from __future__ import annotations

import datetime as _dt
import json
import math
import os
import platform
import re
import statistics
import subprocess
import sys
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Constants

#: Schema version for ``bench/results/<machine>/<benchmark>.json``. Bump
#: when the result shape changes; the regression detector accepts any
#: schema version less than or equal to its own.
RESULT_SCHEMA_VERSION: str = "1.0.0"

#: Default location for persisted benchmark results.
DEFAULT_RESULTS_DIR: Path = Path(__file__).resolve().parent / "results"

# ---------------------------------------------------------------------------
# Result schema


@dataclass(frozen=True)
class BenchMetadata:
    """Reproducibility metadata captured with every result."""

    commit: str
    timestamp: str
    machine: str
    python_version: str
    platform: str
    dtype: str
    extra: Mapping[str, str] = field(default_factory=dict)

    def to_json(self) -> dict[str, Any]:
        out = dict(asdict(self))
        out["extra"] = dict(self.extra)
        return out


@dataclass(frozen=True)
class BenchResult:
    """Single benchmark run, ready for JSON serialization."""

    name: str
    iters: int
    warmup: int
    samples_ns: tuple[int, ...]
    median_ns: int
    p25_ns: int
    p75_ns: int
    iqr_ns: int
    metadata: BenchMetadata
    schema_version: str = RESULT_SCHEMA_VERSION

    def to_json(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "name": self.name,
            "iters": self.iters,
            "warmup": self.warmup,
            "samples_ns": list(self.samples_ns),
            "median_ns": self.median_ns,
            "p25_ns": self.p25_ns,
            "p75_ns": self.p75_ns,
            "iqr_ns": self.iqr_ns,
            "metadata": self.metadata.to_json(),
        }


# ---------------------------------------------------------------------------
# Timing


def time_callable(
    name: str,
    fn: Callable[[], object],
    *,
    iters: int = 100,
    warmup: int = 10,
    dtype: str = "n/a",
    extra: Mapping[str, str] | None = None,
) -> BenchResult:
    """Run ``fn`` ``iters`` times after ``warmup`` warmups; return stats.

    ``fn`` must be a no-argument callable. Each call is timed with
    :func:`time.perf_counter_ns`; the smallest possible Python-level
    wrapper is used so the overhead is fixed across samples.

    Raises:
        ValueError: ``iters`` ≤ 0 or ``warmup`` < 0.
    """
    if iters <= 0:
        raise ValueError(f"iters must be > 0, got {iters}")
    if warmup < 0:
        raise ValueError(f"warmup must be >= 0, got {warmup}")

    perf = time.perf_counter_ns
    for _ in range(warmup):
        fn()

    samples: list[int] = []
    for _ in range(iters):
        t0 = perf()
        fn()
        samples.append(perf() - t0)

    samples_sorted = sorted(samples)
    median = _percentile(samples_sorted, 50)
    p25 = _percentile(samples_sorted, 25)
    p75 = _percentile(samples_sorted, 75)
    iqr = p75 - p25

    return BenchResult(
        name=name,
        iters=iters,
        warmup=warmup,
        samples_ns=tuple(samples),
        median_ns=median,
        p25_ns=p25,
        p75_ns=p75,
        iqr_ns=iqr,
        metadata=collect_metadata(dtype=dtype, extra=extra or {}),
    )


def _percentile(sorted_samples: Sequence[int], pct: float) -> int:
    """Linear-interpolated percentile of a *pre-sorted* sequence."""
    if not sorted_samples:
        raise ValueError("cannot take percentile of empty sequence")
    if len(sorted_samples) == 1:
        return int(sorted_samples[0])
    rank = (pct / 100.0) * (len(sorted_samples) - 1)
    lo = math.floor(rank)
    hi = math.ceil(rank)
    if lo == hi:
        return int(sorted_samples[lo])
    frac = rank - lo
    interpolated = sorted_samples[lo] * (1 - frac) + sorted_samples[hi] * frac
    return round(interpolated)


# ---------------------------------------------------------------------------
# Metadata


def collect_metadata(*, dtype: str, extra: Mapping[str, str]) -> BenchMetadata:
    """Best-effort host/runtime fingerprint."""
    return BenchMetadata(
        commit=current_commit(),
        timestamp=_dt.datetime.now(tz=_dt.timezone.utc).isoformat(timespec="seconds"),
        machine=machine_id(),
        python_version=platform.python_version(),
        platform=platform.platform(terse=True),
        dtype=dtype,
        extra=dict(extra),
    )


def machine_id() -> str:
    """A stable, human-readable machine fingerprint.

    Order of preference:

    1. ``GENO_LEWM_BENCH_MACHINE`` environment variable (CI / explicit).
    2. ``platform.node()`` if it looks like a real hostname.
    3. ``platform.system()`` + ``platform.machine()`` fallback.

    The result is sanitised so it is safe as a directory name: ASCII
    letters, digits, ``-`` and ``_`` only.
    """
    raw = os.environ.get("GENO_LEWM_BENCH_MACHINE")
    if not raw:
        node = platform.node()
        if node and node not in {"localhost", "(none)"}:
            raw = node
        else:
            raw = f"{platform.system()}-{platform.machine()}"
    return _sanitize(raw)


_SAFE_CHARS = re.compile(r"[^A-Za-z0-9_-]+")


def _sanitize(name: str) -> str:
    sanitised = _SAFE_CHARS.sub("-", name.strip()).strip("-")
    return sanitised or "unknown"


def current_commit() -> str:
    """Resolve the current git HEAD; ``unknown`` if not in a repo."""
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
            cwd=Path(__file__).resolve().parent.parent,
        )
    except (FileNotFoundError, subprocess.CalledProcessError):
        return "unknown"
    return out.stdout.strip() or "unknown"


# ---------------------------------------------------------------------------
# Persistence


def write_result(
    result: BenchResult,
    *,
    out_dir: Path = DEFAULT_RESULTS_DIR,
) -> Path:
    """Persist a benchmark result; return the file written.

    The path layout is ``<out_dir>/<machine>/<name>.json``. ``<name>`` is
    sanitised (slashes become ``__``) so namespaced benchmarks like
    ``inference.commitment`` map to a single file ``inference.commitment.json``.
    """
    target_dir = out_dir / result.metadata.machine
    target_dir.mkdir(parents=True, exist_ok=True)
    target_path = target_dir / f"{_sanitize_filename(result.name)}.json"
    target_path.write_text(
        json.dumps(result.to_json(), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return target_path


def _sanitize_filename(name: str) -> str:
    """Make ``name`` safe for a single-file path component (no slashes)."""
    return name.replace("/", "__").replace("\\", "__")


# ---------------------------------------------------------------------------
# Reporting helpers (used by per-target scripts)


def humanize_ns(ns: float) -> str:
    """Render a nanosecond value with a sensible unit."""
    if ns < 1_000:
        return f"{ns:.0f} ns"
    if ns < 1_000_000:
        return f"{ns / 1_000:.2f} µs"
    if ns < 1_000_000_000:
        return f"{ns / 1_000_000:.2f} ms"
    return f"{ns / 1_000_000_000:.2f} s"


def report_to_stdout(result: BenchResult, *, stream: Any = None) -> None:
    """Pretty-print a result to ``stream`` (defaults to stdout)."""
    if stream is None:
        stream = sys.stdout
    median = humanize_ns(result.median_ns)
    iqr = humanize_ns(result.iqr_ns)
    print(
        f"[bench] {result.name}: median={median} iqr={iqr} "
        f"iters={result.iters} warmup={result.warmup} "
        f"machine={result.metadata.machine} commit={result.metadata.commit}",
        file=stream,
    )


# ---------------------------------------------------------------------------
# Median / IQR helpers exported for the regression detector


def median_iqr(samples_ns: Sequence[int]) -> tuple[int, int]:
    """Median + IQR of unsorted nanosecond samples."""
    if not samples_ns:
        raise ValueError("cannot compute median/IQR of empty samples")
    sorted_samples = sorted(samples_ns)
    median = int(statistics.median(sorted_samples))
    p25 = _percentile(sorted_samples, 25)
    p75 = _percentile(sorted_samples, 75)
    return median, p75 - p25
