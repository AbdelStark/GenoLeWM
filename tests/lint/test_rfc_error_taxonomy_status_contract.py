# SPDX-License-Identifier: Apache-2.0
"""Regression tests for the RFC-0012 implementation-status boundary."""

from __future__ import annotations

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
RFC = REPO_ROOT / "rfcs" / "0012-error-taxonomy.md"
REGISTRY_GENERATOR = REPO_ROOT / "docs" / "_gen" / "build_registry_pages.py"
ERROR_SPEC = REPO_ROOT / "docs" / "spec" / "04-error-model.md"
MKDOCS = REPO_ROOT / "mkdocs.yml"


def test_rfc_error_taxonomy_status_tracks_generated_error_docs() -> None:
    text = RFC.read_text(encoding="utf-8")
    required = (
        "- **Updated:** 2026-06-11",
        "stable error-code registry",
        "CLI exit-code mapping",
        "docs-build\n  generation of `api/error-codes.md`",
        "docs-generation coverage",
        "lookup CLI and programmatic\n  remediation helpers remain future work",
    )

    for fragment in required:
        assert fragment in text


def test_rfc_error_taxonomy_future_work_excludes_landed_docs_generation() -> None:
    text = RFC.read_text(encoding="utf-8")

    assert "Auto-generated docs from `ERROR_CODES`" not in text
    assert "`geno-lewm errors lookup CODE` CLI" in text
    assert "Programmatic remediation" in text


def test_error_code_registry_docs_are_wired_into_docs_build() -> None:
    generator = REGISTRY_GENERATOR.read_text(encoding="utf-8")
    error_spec = ERROR_SPEC.read_text(encoding="utf-8")
    mkdocs = MKDOCS.read_text(encoding="utf-8")

    assert "build_error_codes" in generator
    assert 'mkdocs_gen_files.open("api/error-codes.md", "w")' in generator
    assert "geno_lewm.errors.ERROR_CODES" in generator
    assert "Exit code mapping" in generator
    assert "docs/api/error-codes.md" in error_spec
    assert "docs/_gen/build_registry_pages.py" in mkdocs
