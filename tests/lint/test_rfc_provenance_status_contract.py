# SPDX-License-Identifier: Apache-2.0
"""Regression tests for RFC-0011 provenance implementation status."""

from __future__ import annotations

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
RFC_0011 = REPO_ROOT / "rfcs" / "0011-artifact-provenance-receipts.md"


def test_rfc_0011_status_matches_current_receipt_release_surface() -> None:
    text = RFC_0011.read_text(encoding="utf-8")
    normalized = " ".join(text.split())

    required = (
        "`tools.release.batch_receipt_report` verification",
        "terminal-demo runtime preflight",
        "manifest identity, score/receipt artifacts, batch receipt summaries",
        "keep validating receipt emission for each new published checkpoint",
        "terminal-demo artifact set",
        "runtime-assurance modes beyond checksum provenance remain out of scope",
    )
    for fragment in required:
        assert fragment in normalized

    stale_fragments = (
        "validate receipt emission against the first published checkpoint and actual Carbon runtime artifacts",
        "legacy import package and receipt JSON field have been removed",
    )
    for fragment in stale_fragments:
        assert fragment not in normalized
