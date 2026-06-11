# SPDX-License-Identifier: Apache-2.0
"""Regression tests for the RFC-0002 implementation-status boundary."""

from __future__ import annotations

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
RFC = REPO_ROOT / "rfcs" / "0002-state-encoder-carbon-integration.md"
CACHE_CLI = REPO_ROOT / "geno_lewm" / "cli" / "cache_windows.py"
CACHE = REPO_ROOT / "geno_lewm" / "encoder" / "cache.py"
CACHE_CLI_TEST = REPO_ROOT / "tests" / "unit" / "test_cli_cache_windows.py"
PUBLIC_API_SPEC = REPO_ROOT / "docs" / "spec" / "02-public-api.md"


def test_rfc_encoder_status_tracks_cache_maintenance_surface() -> None:
    text = RFC.read_text(encoding="utf-8")
    required = (
        "- **Updated:** 2026-06-11",
        "cache schema/read/write/reindex/repair primitives",
        "`geno-lewm-cache-windows --reindex/--repair` maintenance CLI",
        "lazy `CarbonStateEncoder` with injected-component tests",
        "full selected\n  corpus cache-build mode",
        "cache-build throughput evidence remain\n  open",
    )

    for fragment in required:
        assert fragment in text


def test_rfc_encoder_status_preserves_full_cache_build_boundary() -> None:
    text = RFC.read_text(encoding="utf-8")

    assert "full selected\n  corpus cache-build mode" in text
    assert "cache-build throughput evidence remain\n  open" in text
    assert "full selected-corpus cache-build throughput evidence remain open" not in text


def test_encoder_cache_status_tracks_live_cli_and_cache_modules() -> None:
    cli_text = CACHE_CLI.read_text(encoding="utf-8")
    cache_text = CACHE.read_text(encoding="utf-8")
    test_text = CACHE_CLI_TEST.read_text(encoding="utf-8")
    spec_text = PUBLIC_API_SPEC.read_text(encoding="utf-8")

    assert "reindex_cache" in cli_text
    assert "repair_cache" in cli_text
    assert "cache build mode is not yet implemented" in cli_text
    assert 'issue="#36"' in cli_text
    assert "CacheReindexReport" in cache_text
    assert "CacheRepairReport" in cache_text
    assert "write_shard" in cache_text
    assert "read_embedding" in cache_text
    assert "test_cache_windows_reindex_cli" in test_text
    assert "test_cache_windows_repair_cli" in test_text
    assert "repair/reindex local cache shards" in spec_text
    assert "full Carbon-corpus cache construction remains open" in spec_text
