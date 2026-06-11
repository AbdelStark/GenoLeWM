# SPDX-License-Identifier: Apache-2.0
"""Regression tests for RFC-0018 CLI implementation status."""

from __future__ import annotations

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
RFC_0018 = REPO_ROOT / "rfcs" / "0018-cli-design.md"


def test_rfc_0018_status_matches_current_cli_surface() -> None:
    text = RFC_0018.read_text(encoding="utf-8")
    normalized = " ".join(text.split())

    required = (
        "rollout aggregation",
        "manifest-backed planning",
        "safetensors export",
        "release/demo support paths exist",
        "Target-specific ONNX / Core ML / GGUF export",
        "full cache-build mode",
        "future runtime command validation remain open",
    )
    for fragment in required:
        assert fragment in normalized

    stale_fragments = (
        "Planning, rollout, export, and full real-artifact demo command validation remain open",
        "Planning, rollout, export",
    )
    for fragment in stale_fragments:
        assert fragment not in normalized
