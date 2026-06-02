# SPDX-License-Identifier: Apache-2.0
"""Live GitHub issue references for first paper/demo release gates."""

from __future__ import annotations

from typing import Final

ISSUE_URL_PREFIX: Final = "https://github.com/AbdelStark/GenoLeWM/issues"
DATASET_ISSUE: Final = 163
TRAINING_ISSUE: Final = 164
EVAL_ISSUE: Final = 165
DEMO_ISSUE: Final = 166
PAPER_ISSUE: Final = 167
MODEL_RELEASE_ISSUE: Final = 101
ALL_RELEASE_ISSUES: Final = (
    DATASET_ISSUE,
    TRAINING_ISSUE,
    EVAL_ISSUE,
    DEMO_ISSUE,
    PAPER_ISSUE,
    MODEL_RELEASE_ISSUE,
)


def issue_ref_payload(issue_refs: tuple[int, ...]) -> list[dict[str, object]]:
    """Serialize issue numbers as stable machine-readable GitHub links."""
    return [
        {
            "number": number,
            "url": f"{ISSUE_URL_PREFIX}/{number}",
        }
        for number in issue_refs
    ]
