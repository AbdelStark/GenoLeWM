# SPDX-License-Identifier: Apache-2.0
"""Regression tests for RFC-0004 predictor implementation status."""

from __future__ import annotations

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
RFC_0004 = REPO_ROOT / "rfcs" / "0004-predictor-architecture.md"


def test_rfc_0004_status_matches_rollout_evidence_boundary() -> None:
    text = RFC_0004.read_text(encoding="utf-8")
    normalized = " ".join(text.split())

    required = (
        "manifest-backed rollout-state row generation",
        "`bench.rollout` / `tools.release.rollout_speed_scope` reporting",
        "v0.2.1-r1 evidence records K=5 speedup meeting the local 2x target",
        "K=20 speedup missing the RFC 5x target",
        "with #42 still open",
        "attention KV-cache speedups that meet the original K=20 target",
        "reported measurements were 2.41x at K=5 and 2.47x at K=20",
        "broader released-artifact validation",
        "trainer/evaluator integration remain open",
    )
    for fragment in required:
        assert fragment in normalized

    stale_fragments = (
        "attention KV-cache speedups, released-artifact validation, and full trainer/evaluator integration remain open",
    )
    for fragment in stale_fragments:
        assert fragment not in normalized
