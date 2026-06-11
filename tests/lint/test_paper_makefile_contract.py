# SPDX-License-Identifier: Apache-2.0
"""Regression tests for the checked TeX paper Makefile."""

from __future__ import annotations

import re
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
PAPER_MAKEFILE = REPO_ROOT / "paper" / "Makefile"


def test_paper_pdf_target_tracks_checked_sources() -> None:
    text = PAPER_MAKEFILE.read_text(encoding="utf-8")
    match = re.search(r"^\$\(PAPER\)\.pdf:\s+(?P<deps>.+)$", text, flags=re.MULTILINE)

    assert match is not None
    deps = set(match.group("deps").split())
    assert {
        "$(PAPER).tex",
        "figures.tex",
        "tables.tex",
        "refs.bib",
        "neurips.sty",
    } <= deps


def test_paper_makefile_keeps_clean_targets() -> None:
    text = PAPER_MAKEFILE.read_text(encoding="utf-8")

    assert ".PHONY: all clean veryclean" in text
    assert re.search(r"^clean:\n\trm -f \$\(PAPER\)\.aux", text, flags=re.MULTILINE)
    assert re.search(r"^veryclean: clean\n\trm -f \$\(PAPER\)\.pdf", text, flags=re.MULTILINE)
