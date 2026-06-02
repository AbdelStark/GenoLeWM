# SPDX-License-Identifier: Apache-2.0
"""Regression tests for public API contract documentation."""

from __future__ import annotations

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
PUBLIC_API_SPEC = REPO_ROOT / "docs" / "spec" / "02-public-api.md"
PUBLIC_API_RFC = REPO_ROOT / "rfcs" / "0014-public-api-and-stability.md"
PUBLIC_API_SNAPSHOT = REPO_ROOT / "tests" / "api" / "public_surface.json"
AGENT_CONTEXT = REPO_ROOT / "AGENTS.md"
IMPLEMENTATION_TRACKER = REPO_ROOT / "docs" / "roadmap" / "IMPLEMENTATION.md"
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
    REPO_ROOT / "rfcs" / "0011-artifact-provenance-receipts.md",
)


def test_planning_solver_is_not_documented_as_stable_top_level_export() -> None:
    spec = PUBLIC_API_SPEC.read_text(encoding="utf-8")
    rfc = PUBLIC_API_RFC.read_text(encoding="utf-8")

    assert "geno_lewm.PlanningResult" not in spec
    assert "`SurpriseResult`, `PlanningResult`, `errors` submodule" not in rfc
    assert "not stable top-level exports yet" in spec
    assert "not stable top-level exports yet" in rfc


def test_public_api_docs_point_to_snapshot_as_exhaustive_contract() -> None:
    spec = PUBLIC_API_SPEC.read_text(encoding="utf-8")
    rfc = PUBLIC_API_RFC.read_text(encoding="utf-8")

    assert "tests/api/public_surface.json" in spec
    assert "tests/api/public_surface.json" in rfc
    assert "exhaustive public symbol set" in spec
    assert "exhaustive enforced symbol list" in rfc


def test_cli_scaffold_helpers_are_not_public_api() -> None:
    spec = PUBLIC_API_SPEC.read_text(encoding="utf-8")
    snapshot = PUBLIC_API_SNAPSHOT.read_text(encoding="utf-8")

    assert "CLI scaffold factory helpers" in spec
    assert "build_stub_app" not in snapshot
    assert "make_cli_main" not in snapshot


def test_legacy_provenance_paths_are_removed() -> None:
    """The old trust-oriented namespace and RFC filename should stay gone."""
    for path in CURRENT_PROVENANCE_PATHS:
        assert path.exists(), path
    for path in LEGACY_PROVENANCE_PATHS:
        assert not path.exists(), path


def test_agent_context_tracks_public_api_contract_guards() -> None:
    context = AGENT_CONTEXT.read_text(encoding="utf-8")
    tracker = IMPLEMENTATION_TRACKER.read_text(encoding="utf-8")

    for text in (context, tracker):
        assert "tests/api/public_surface.json" in text
        assert "duplicate-free `__all__`" in text
        assert "current public module-map docs" in text
        assert "not described as stable top-level exports" in text
