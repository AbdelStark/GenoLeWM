# SPDX-License-Identifier: Apache-2.0
"""Performance regression detector (RFC-0016 §3.7).

Diff a current benchmark result tree against the committed baseline at
``bench/results/baseline/``. A benchmark *regresses* when its current
median exceeds the baseline median by more than ``--threshold``
(default 5 %, per RFC-0016 §3.7).

Two input formats are supported:

1. **bench-harness JSON** (``bench/_harness.BenchResult.to_json``).
   The per-target scripts under ``bench/`` write one file per
   benchmark at ``bench/results/<machine>/<name>.json``.

2. **pytest-benchmark JSON** (``--benchmark-json=...``). A single file
   with a ``benchmarks`` array. The ``stats.median`` field (seconds)
   is used; the ``name`` field is the benchmark identifier.

The detector picks the format from the file shape and is robust to a
mixed baseline tree (some files in format 1, the nightly file in
format 2).

Exit codes:

- ``0`` — no regression.
- ``1`` — at least one regression above the threshold.
- ``2`` — invalid inputs.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path

DEFAULT_THRESHOLD: float = 0.05
DEFAULT_BASELINE: Path = Path(__file__).resolve().parents[2] / "bench" / "results" / "baseline"


@dataclass(frozen=True)
class BenchSample:
    """One named measurement, with median expressed in nanoseconds."""

    name: str
    median_ns: float
    source: Path


@dataclass(frozen=True)
class Comparison:
    """Outcome of comparing one current sample to its baseline."""

    name: str
    current_ns: float
    baseline_ns: float
    ratio: float  # current / baseline (1.05 = +5%)

    @property
    def delta_pct(self) -> float:
        return (self.ratio - 1.0) * 100.0


def load_samples(path: Path) -> Iterator[BenchSample]:
    """Yield ``BenchSample`` from a single JSON file or a directory tree.

    Directories are walked recursively; ``*.json`` files are loaded and
    classified by shape. Files that contain neither format are skipped
    silently — the harness directory may hold READMEs or other notes.
    """
    if path.is_dir():
        for child in sorted(path.rglob("*.json")):
            yield from _load_file(child)
        return
    yield from _load_file(path)


def _load_file(path: Path) -> Iterator[BenchSample]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return

    # bench-harness format: dict with top-level "median_ns" + "name".
    if isinstance(data, dict) and "median_ns" in data and "name" in data:
        # iters == 0 is a placeholder (e.g., bench/planning.py before
        # the planning module lands). Skip those: there is nothing
        # meaningful to compare.
        if data.get("iters", 0) <= 0:
            return
        yield BenchSample(
            name=str(data["name"]),
            median_ns=float(data["median_ns"]),
            source=path,
        )
        return

    # pytest-benchmark format: dict with top-level "benchmarks" array.
    if isinstance(data, dict) and isinstance(data.get("benchmarks"), list):
        for entry in data["benchmarks"]:
            stats = entry.get("stats") or {}
            median_s = stats.get("median")
            name = entry.get("name") or entry.get("fullname")
            if name is None or median_s is None:
                continue
            yield BenchSample(
                name=str(name),
                median_ns=float(median_s) * 1_000_000_000.0,
                source=path,
            )


def compare(
    current: Sequence[BenchSample],
    baseline: Sequence[BenchSample],
) -> tuple[list[Comparison], list[str], list[str]]:
    """Pair samples by name; return (paired, missing_baseline, missing_current).

    ``missing_baseline`` is informational: a new benchmark has no
    baseline yet and cannot regress. ``missing_current`` is also
    informational: a baseline benchmark that the current run did not
    produce is a measurement gap, not a regression.
    """
    by_name_b = {s.name: s for s in baseline}
    by_name_c = {s.name: s for s in current}
    paired: list[Comparison] = []
    for name in sorted(set(by_name_c) & set(by_name_b)):
        c = by_name_c[name]
        b = by_name_b[name]
        if b.median_ns <= 0:
            continue
        paired.append(
            Comparison(
                name=name,
                current_ns=c.median_ns,
                baseline_ns=b.median_ns,
                ratio=c.median_ns / b.median_ns,
            )
        )
    missing_baseline = sorted(set(by_name_c) - set(by_name_b))
    missing_current = sorted(set(by_name_b) - set(by_name_c))
    return paired, missing_baseline, missing_current


def format_report(
    comparisons: Sequence[Comparison],
    missing_baseline: Sequence[str],
    missing_current: Sequence[str],
    threshold: float,
) -> str:
    lines: list[str] = []
    if comparisons:
        header = f"{'Benchmark':<48} {'Baseline':>14} {'Current':>14} {'Δ':>8}  Status"
        lines.append(header)
        lines.append("-" * len(header))
        for cmp_ in comparisons:
            status = "REGRESS" if cmp_.ratio > 1.0 + threshold else "ok"
            lines.append(
                f"{cmp_.name:<48} "
                f"{_humanize_ns(cmp_.baseline_ns):>14} "
                f"{_humanize_ns(cmp_.current_ns):>14} "
                f"{cmp_.delta_pct:>+7.1f}%  {status}"
            )
    else:
        lines.append("perf_regression: no overlapping benchmarks to compare.")
    if missing_baseline:
        lines.append("")
        lines.append("New benchmarks (no baseline yet):")
        lines.extend(f"  + {n}" for n in missing_baseline)
    if missing_current:
        lines.append("")
        lines.append("Baseline benchmarks missing from current run:")
        lines.extend(f"  - {n}" for n in missing_current)
    return "\n".join(lines) + "\n"


def _humanize_ns(ns: float) -> str:
    if ns < 1_000:
        return f"{ns:.0f} ns"
    if ns < 1_000_000:
        return f"{ns / 1_000:.2f} µs"
    if ns < 1_000_000_000:
        return f"{ns / 1_000_000:.2f} ms"
    return f"{ns / 1_000_000_000:.2f} s"


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="perf_regression",
        description="Performance regression detector (RFC-0016 §3.7).",
    )
    parser.add_argument(
        "--current",
        type=Path,
        required=True,
        help="path to the current run (JSON file or directory tree)",
    )
    parser.add_argument(
        "--baseline",
        type=Path,
        default=DEFAULT_BASELINE,
        help=f"path to the baseline (default: {DEFAULT_BASELINE})",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=DEFAULT_THRESHOLD,
        help=f"max allowed median ratio above baseline (default: {DEFAULT_THRESHOLD})",
    )
    args = parser.parse_args(argv)

    if not args.current.exists():
        print(
            f"perf_regression: --current path does not exist: {args.current}",
            file=sys.stderr,
        )
        return 2
    if not args.baseline.exists():
        # No baseline yet → nothing to compare against; emit a clear
        # message and exit 0. The first nightly run on a fresh tree
        # populates the baseline; subsequent runs gate on it.
        print(
            f"perf_regression: no baseline at {args.baseline}; treating as warm-up run.",
            file=sys.stderr,
        )
        return 0

    current = list(load_samples(args.current))
    baseline = list(load_samples(args.baseline))
    if not current:
        print(
            f"perf_regression: no samples loaded from {args.current}",
            file=sys.stderr,
        )
        return 2

    paired, missing_baseline, missing_current = compare(current, baseline)
    sys.stdout.write(format_report(paired, missing_baseline, missing_current, args.threshold))

    regressions = [c for c in paired if c.ratio > 1.0 + args.threshold]
    if regressions:
        print(
            f"perf_regression: {len(regressions)} regression(s) above "
            f"{args.threshold * 100:.1f}% threshold",
            file=sys.stderr,
        )
        return 1
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
