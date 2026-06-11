# SPDX-License-Identifier: Apache-2.0
"""Regression tests for the RFC-0013 implementation-status boundary."""

from __future__ import annotations

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
RFC = REPO_ROOT / "rfcs" / "0013-observability.md"
REGISTRY_GENERATOR = REPO_ROOT / "docs" / "_gen" / "build_registry_pages.py"
METRICS_DOC = REPO_ROOT / "docs" / "api" / "metrics.md"
MKDOCS = REPO_ROOT / "mkdocs.yml"


def test_rfc_observability_status_tracks_registry_docs() -> None:
    text = RFC.read_text(encoding="utf-8")
    required = (
        "- **Updated:** 2026-06-11",
        "event/metric registry linting",
        "docs-build\n  generation of `api/log-events.md`",
        "`docs/api/metrics.md` registry reference",
        "OpenTelemetry export remains a future\n  optional sink",
    )

    for fragment in required:
        assert fragment in text


def test_rfc_observability_future_work_excludes_landed_registry_docs() -> None:
    text = RFC.read_text(encoding="utf-8")

    assert "Auto-generated registries (`EVENTS`, `METRICS`)" not in text
    assert "Pre-rendered registry snapshots for offline docs consumers" in text


def test_observability_registry_docs_are_wired_into_docs_build() -> None:
    generator = REGISTRY_GENERATOR.read_text(encoding="utf-8")
    metrics_doc = METRICS_DOC.read_text(encoding="utf-8")
    mkdocs = MKDOCS.read_text(encoding="utf-8")

    assert "build_log_events" in generator
    assert 'mkdocs_gen_files.open("api/log-events.md", "w")' in generator
    assert "geno_lewm.observability.EVENTS" in generator
    assert "::: geno_lewm.metrics.METRICS" in metrics_doc
    assert "docs/_gen/build_registry_pages.py" in mkdocs
