# SPDX-License-Identifier: Apache-2.0
"""Regression tests for the GenoLeWM-FX contract and decision docs."""

from __future__ import annotations

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
CONTRACT = REPO_ROOT / "docs" / "research" / "fx-experiment-contract.md"
REPORT = REPO_ROOT / "docs" / "research" / "fx-feasibility-report.md"
DECISION = REPO_ROOT / "docs" / "research" / "fx-decision-package.md"
RESCUE = REPO_ROOT / "docs" / "research" / "fx-borzoi-rescue-plan.md"
INDEX = REPO_ROOT / "docs" / "index.md"
README = REPO_ROOT / "README.md"
MKDOCS = REPO_ROOT / "mkdocs.yml"


def test_fx_contract_locks_required_baselines_and_claim_gates() -> None:
    text = CONTRACT.read_text(encoding="utf-8")
    normalized = " ".join(text.split())
    required = (
        "GenoLeWM-FX should model functional transitions",
        "zero-delta/no-edit baseline",
        "source-only or label-prior baseline",
        "linear or logistic probe on teacher features",
        "no reproducible public teacher-delta slice exists",
        "AlphaGenome is allowed only as a bounded sampled calibration/API baseline",
        "Failure is publishable as a kill report",
        "No medium or expensive Hugging Face job should launch",
    )
    forbidden = (
        "clinical utility",
        "deployment readiness",
        "broad variant-effect prediction superiority",
    )

    for fragment in required:
        assert fragment in normalized
    for fragment in forbidden:
        assert fragment in text


def test_fx_report_and_decision_package_publish_kill_path() -> None:
    report = REPORT.read_text(encoding="utf-8")
    decision = DECISION.read_text(encoding="utf-8")

    assert "Decision: **kill**." in report
    assert "no_public_teacher_delta_cache" in report
    assert "The parent epic takes the kill path" in decision
    assert "No FX demo ships" in decision
    assert "No FX positive-result paper section is added" in decision
    assert "no Stage B or Stage C job should launch" in decision
    assert "Borzoi rescue plan" in decision


def test_public_docs_link_fx_research_without_success_language() -> None:
    combined = "\n".join(path.read_text(encoding="utf-8") for path in (INDEX, README, MKDOCS))
    normalized = " ".join(combined.split())

    assert "research/fx-experiment-contract.md" in combined
    assert "research/fx-feasibility-report.md" in combined
    assert "research/fx-decision-package.md" in combined
    assert "research/fx-borzoi-rescue-plan.md" in combined
    assert "FX pivot" in combined
    assert "No GenoLeWM-FX model or demo ships" in combined
    assert "precomputed-Borzoi" in normalized
    assert "overlap audit" in normalized
    assert "GenoLeWM-FX improves" not in combined
    assert "GenoLeWM-FX outperforms" not in combined


def test_fx_borzoi_rescue_plan_is_overlap_first_and_claim_bounded() -> None:
    text = RESCUE.read_text(encoding="utf-8")
    normalized = " ".join(text.split())
    required = (
        "This plan does not reverse the #257 kill decision",
        "precomputed Borzoi scores",
        "more than 19 million common and low-frequency variants",
        "based on hg19",
        "at least 10,000 matched variants",
        "If this fails, stop and publish the overlap no-go report",
        "Use #266 rather than reopening #257",
        "#267 - lock rescue contract and coordinate rules",
        "#272 - publish the locked result or overlap kill report",
    )
    forbidden = (
        "clinical evidence",
        "deployment readiness",
        "broad VEP superiority",
        "proof of useful planning",
    )

    for fragment in required:
        assert fragment in normalized
    for fragment in forbidden:
        assert fragment in normalized
