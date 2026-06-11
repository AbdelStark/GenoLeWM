# SPDX-License-Identifier: Apache-2.0
"""Regression tests for RFC-0010 runtime distribution status."""

from __future__ import annotations

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
RFC_0010 = REPO_ROOT / "rfcs" / "0010-on-device-personal-genome-deployment.md"


def test_rfc_0010_distribution_matches_published_package_state() -> None:
    text = RFC_0010.read_text(encoding="utf-8")
    normalized = " ".join(text.split())

    assert "geno-lewm==0.2.1" in text
    assert "PyPI**: current Python package channel" in text
    assert "trusted publishing through OIDC remains" in normalized

    stale_fragments = (
        "until the first PyPI tag is",
        "planned package channel after trusted publishing",
        "the first tag is released",
        "first public checkpoint package",
    )
    for fragment in stale_fragments:
        assert fragment not in text
