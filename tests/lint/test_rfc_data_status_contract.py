# SPDX-License-Identifier: Apache-2.0
"""Regression tests for RFC-0006 data-pipeline implementation status."""

from __future__ import annotations

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
RFC_0006 = REPO_ROOT / "rfcs" / "0006-data-pipeline.md"


def test_rfc_0006_status_matches_current_dataset_release_surface() -> None:
    text = RFC_0006.read_text(encoding="utf-8")
    normalized = " ".join(text.split())

    required = (
        "public `abdelstark/geno-lewm-data` dataset package",
        "`dataset_manifest.json`",
        "`split_integrity.json`",
        "release dataset tuple-throughput measurement helper exist",
        "Broader published real-shard coverage",
        "target-hardware warm-cache throughput evidence",
        "expanded held-out data snapshots remain open",
    )
    for fragment in required:
        assert fragment in normalized

    stale_fragments = (
        "Published real shards, warm-cache throughput evidence, and public split artifacts remain open",
        "public split artifacts remain open",
    )
    for fragment in stale_fragments:
        assert fragment not in normalized
