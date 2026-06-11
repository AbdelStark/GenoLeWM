# SPDX-License-Identifier: Apache-2.0
"""Regression tests for RFC-0015 testing/CI status."""

from __future__ import annotations

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
RFC_0015 = REPO_ROOT / "rfcs" / "0015-testing-strategy.md"
CI_WORKFLOW = REPO_ROOT / ".github" / "workflows" / "ci.yml"


def test_rfc_0015_matches_hosted_ml_and_eval_smoke_ci_state() -> None:
    rfc = RFC_0015.read_text(encoding="utf-8")
    normalized = " ".join(rfc.split())
    workflow = CI_WORKFLOW.read_text(encoding="utf-8")

    assert "hosted `tests/ml`" in normalized
    assert "generated-fixture eval-smoke CI gates exist" in normalized
    assert "ratcheted below the original 95% target" in normalized
    assert "full real-data eval remains a release-candidate gate" in normalized

    assert "ml-smoke:" in workflow
    assert "eval-smoke:" in workflow

    stale_fragments = (
        "Dedicated hosted `tests/ml` and eval-smoke CI gates remain open",
        "eval-smoke CI gates remain open",
    )
    for fragment in stale_fragments:
        assert fragment not in normalized
