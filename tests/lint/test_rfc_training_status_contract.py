# SPDX-License-Identifier: Apache-2.0
"""Regression tests for RFC-0005 training implementation status."""

from __future__ import annotations

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
RFC_0005 = REPO_ROOT / "rfcs" / "0005-training-objective.md"


def test_rfc_0005_status_matches_current_training_surface() -> None:
    text = RFC_0005.read_text(encoding="utf-8")
    normalized = " ".join(text.split())

    required = (
        "preflight-gated Carbon training launcher",
        "training-run manifest/card/checksum packaging exist",
        "Completed clean-machine Carbon-backed training evidence",
        "deterministic real-run reproducibility evidence remain open",
    )
    for fragment in required:
        assert fragment in normalized

    stale_fragments = (
        "Carbon-state batch encoding, and the torch trainer core exist. Clean-machine Carbon-backed training and deterministic real-run evidence remain open",
    )
    for fragment in stale_fragments:
        assert fragment not in normalized
