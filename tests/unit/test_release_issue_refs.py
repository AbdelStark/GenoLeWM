"""Tests for live release issue reference serialization."""

from __future__ import annotations

from tools.release.issue_refs import (
    ALL_RELEASE_ISSUES,
    DATASET_ISSUE,
    DEMO_ISSUE,
    EVAL_ISSUE,
    ISSUE_URL_PREFIX,
    MODEL_RELEASE_ISSUE,
    PAPER_ISSUE,
    TRAINING_ISSUE,
    issue_ref_payload,
)


def test_all_release_issues_match_live_paper_demo_blockers() -> None:
    assert ALL_RELEASE_ISSUES == (
        DATASET_ISSUE,
        TRAINING_ISSUE,
        EVAL_ISSUE,
        DEMO_ISSUE,
        PAPER_ISSUE,
        MODEL_RELEASE_ISSUE,
    )
    assert ALL_RELEASE_ISSUES == (163, 164, 165, 166, 167, 101)


def test_issue_ref_payload_serializes_github_links() -> None:
    assert issue_ref_payload((DATASET_ISSUE, MODEL_RELEASE_ISSUE)) == [
        {
            "number": 163,
            "url": f"{ISSUE_URL_PREFIX}/163",
        },
        {
            "number": 101,
            "url": f"{ISSUE_URL_PREFIX}/101",
        },
    ]
