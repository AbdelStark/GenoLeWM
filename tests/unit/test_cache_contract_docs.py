"""Repository-level claim-boundary checks for the corrected cache contract."""

from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def test_corrected_cache_contract_never_regresses_to_v2_wording() -> None:
    dossier = (ROOT / "paper" / "EVIDENCE_DOSSIER.md").read_text(encoding="utf-8")

    stale = re.compile(
        r"(?:corrected[^\n]{0,80}cache schema 2|corrected cache v2)",
        flags=re.IGNORECASE,
    )
    assert not stale.search(dossier)
    assert "Cache schema 2 is labeled replay-only" in dossier
    assert "Corrected cache v3" in dossier


def test_cache_v3_docs_keep_full_build_and_corrected_run_unclaimed() -> None:
    architecture = (ROOT / "ARCHITECTURE.md").read_text(encoding="utf-8")
    glossary = (ROOT / "docs" / "glossary.md").read_text(encoding="utf-8")

    assert "does not claim completion of a full cache build" in architecture
    assert re.search(r"No full cache build or\s+corrected model run is claimed", glossary)


def test_cache_v3_docs_state_the_posix_only_safe_io_boundary() -> None:
    architecture = (ROOT / "ARCHITECTURE.md").read_text(encoding="utf-8")
    glossary = (ROOT / "docs" / "glossary.md").read_text(encoding="utf-8")
    dossier = (ROOT / "paper" / "EVIDENCE_DOSSIER.md").read_text(encoding="utf-8")

    for text in (architecture, glossary, dossier):
        assert "Linux and macOS" in text
        assert "fails closed on Windows" in text
