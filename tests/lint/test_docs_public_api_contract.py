# SPDX-License-Identifier: Apache-2.0
"""Regression tests for public API contract documentation."""

from __future__ import annotations

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
PUBLIC_API_DOC = REPO_ROOT / "docs" / "api" / "public-surface.md"
PUBLIC_API_SNAPSHOT = REPO_ROOT / "tests" / "api" / "public_surface.json"
LEGACY_NAMESPACE = "att" + "estation"
LEGACY_RFC_FILENAME = "0011-verifiable-" + "inference-" + LEGACY_NAMESPACE + ".md"
LEGACY_PROVENANCE_PATHS = (
    REPO_ROOT / "geno_lewm" / LEGACY_NAMESPACE,
    REPO_ROOT / "tests" / "unit" / f"test_{LEGACY_NAMESPACE}_commitment.py",
    REPO_ROOT / "tests" / "unit" / f"test_{LEGACY_NAMESPACE}_manifest.py",
    REPO_ROOT / "tests" / "unit" / f"test_{LEGACY_NAMESPACE}_receipt.py",
    REPO_ROOT / "rfcs" / LEGACY_RFC_FILENAME,
)
CURRENT_PROVENANCE_PATHS = (
    REPO_ROOT / "geno_lewm" / "provenance",
    REPO_ROOT / "tests" / "unit" / "test_provenance_receipt.py",
)


def test_public_api_docs_point_to_snapshot_as_exhaustive_contract() -> None:
    text = PUBLIC_API_DOC.read_text(encoding="utf-8")

    assert "tests/api/public_surface.json" in text
    assert "exhaustive public symbol set" in text
    assert "uv run python tools/api/snapshot.py check" in text


def test_planning_solver_is_not_documented_as_stable_top_level_export() -> None:
    text = PUBLIC_API_DOC.read_text(encoding="utf-8")

    assert "PlanningResult" in text
    assert "not stable top-level exports yet" in text
    assert "geno_lewm.PlanningResult" not in text


def test_cli_scaffold_helpers_are_not_public_api() -> None:
    text = PUBLIC_API_DOC.read_text(encoding="utf-8")
    snapshot = PUBLIC_API_SNAPSHOT.read_text(encoding="utf-8")

    assert "CLI scaffold factory helpers" in text
    assert "build_stub_app" not in snapshot
    assert "make_cli_main" not in snapshot


def test_legacy_provenance_paths_are_removed() -> None:
    """The old trust-oriented namespace should stay gone."""
    for path in CURRENT_PROVENANCE_PATHS:
        assert path.exists(), path
    for path in LEGACY_PROVENANCE_PATHS:
        assert not path.exists(), path
