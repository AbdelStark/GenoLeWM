# SPDX-License-Identifier: Apache-2.0
"""Regression tests for RFC-0009 surprise-scoring release status."""

from __future__ import annotations

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
RFC_0009 = REPO_ROOT / "rfcs" / "0009-surprise-based-pathogenicity-scoring.md"


def test_rfc_0009_matches_released_scoring_artifact_state() -> None:
    text = RFC_0009.read_text(encoding="utf-8")
    normalized = " ".join(text.split())

    required = (
        "public model/data artifact",
        "released terminal-demo replay evidence",
        "population-stratified calibration",
        "new runtime surfaces still need artifact-backed validation",
    )
    for fragment in required:
        assert fragment in normalized

    stale_fragments = (
        "Validation against released model/data artifacts",
        "clean-machine score transcript",
        "remain open.",
    )
    for fragment in stale_fragments:
        assert fragment not in normalized
