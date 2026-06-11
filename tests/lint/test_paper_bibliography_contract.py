# SPDX-License-Identifier: Apache-2.0
"""Regression tests for the checked TeX paper bibliography."""

from __future__ import annotations

import re
from collections import Counter
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
PAPER_DIR = REPO_ROOT / "paper"
REFS_BIB = PAPER_DIR / "refs.bib"
TEX_SOURCES = (
    PAPER_DIR / "main.tex",
    PAPER_DIR / "figures.tex",
    PAPER_DIR / "tables.tex",
)

_BIB_ENTRY_RE = re.compile(
    r"@(?P<kind>[A-Za-z]+)\s*\{\s*(?P<key>[^,\s]+)\s*,(?P<body>.*?)(?=^\s*@|\Z)",
    re.DOTALL | re.MULTILINE,
)
_CITE_RE = re.compile(r"\\cite[A-Za-z*]*\s*(?:\[[^\]]*\]\s*){0,2}\{(?P<keys>[^}]*)\}")
_LOCATOR_RE = re.compile(r"^\s*(doi|url|eprint)\s*=", re.IGNORECASE | re.MULTILINE)


def _bib_entries() -> dict[str, str]:
    text = REFS_BIB.read_text(encoding="utf-8")
    return {match.group("key"): match.group("body") for match in _BIB_ENTRY_RE.finditer(text)}


def _bib_keys_with_duplicates() -> list[str]:
    text = REFS_BIB.read_text(encoding="utf-8")
    keys = [match.group("key") for match in _BIB_ENTRY_RE.finditer(text)]
    return sorted(key for key, count in Counter(keys).items() if count > 1)


def _cited_keys() -> set[str]:
    cited: set[str] = set()
    for source in TEX_SOURCES:
        text = source.read_text(encoding="utf-8")
        for match in _CITE_RE.finditer(text):
            cited.update(key.strip() for key in match.group("keys").split(",") if key.strip())
    return cited


def test_paper_bibliography_keys_are_unique() -> None:
    assert _bib_keys_with_duplicates() == []


def test_paper_citations_and_bibliography_are_in_sync() -> None:
    bib_keys = set(_bib_entries())
    cited_keys = _cited_keys()

    assert sorted(cited_keys - bib_keys) == []
    assert sorted(bib_keys - cited_keys) == []


def test_paper_bibliography_entries_keep_verifiable_locator() -> None:
    missing_locator = [
        key for key, body in sorted(_bib_entries().items()) if _LOCATOR_RE.search(body) is None
    ]

    assert missing_locator == []
