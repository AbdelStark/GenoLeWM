# SPDX-License-Identifier: Apache-2.0
"""Regression tests for RFC-0016 performance implementation status."""

from __future__ import annotations

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
RFC_0016 = REPO_ROOT / "rfcs" / "0016-performance-budget.md"


def test_rfc_0016_status_matches_current_benchmark_surface() -> None:
    text = RFC_0016.read_text(encoding="utf-8")
    normalized = " ".join(text.split())

    required = (
        "`bench.inference` can emit a validated release-efficiency report",
        "`bench.rollout` emits K=5/K=20 AR speed reports with target checks",
        "`bench.planning` emits CEM-loop and deterministic default-config planning reports",
        "named-hardware target profiles",
        "`tools.ci.perf_regression` plus the nightly performance workflow",
        "harness or pytest-benchmark outputs against baselines",
        "Public warm-cache, target-hardware real-model reports",
        "historical dashboards",
        "automated regression issue filing",
        "required per-PR benchmark gates remain open",
        "`bench/rollout.py`: K=5/K=20 AR rollout speedup benchmarks",
    )
    for fragment in required:
        assert fragment in normalized

    stale_fragments = (
        "Public real-model efficiency reports, regression dashboards, and benchmark CI gates remain open",
    )
    for fragment in stale_fragments:
        assert fragment not in normalized
