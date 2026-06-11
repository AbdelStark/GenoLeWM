# SPDX-License-Identifier: Apache-2.0
"""Regression tests for the GenoLeWM-FX contract and decision docs."""

from __future__ import annotations

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
CONTRACT = REPO_ROOT / "docs" / "research" / "fx-experiment-contract.md"
REPORT = REPO_ROOT / "docs" / "research" / "fx-feasibility-report.md"
DECISION = REPO_ROOT / "docs" / "research" / "fx-decision-package.md"
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


def test_public_docs_link_fx_research_without_success_language() -> None:
    combined = "\n".join(path.read_text(encoding="utf-8") for path in (INDEX, README, MKDOCS))

    assert "research/fx-experiment-contract.md" in combined
    assert "research/fx-feasibility-report.md" in combined
    assert "research/fx-decision-package.md" in combined
    assert "FX pivot" in combined
    assert "No GenoLeWM-FX model or demo ships" in combined
    assert "GenoLeWM-FX improves" not in combined
    assert "GenoLeWM-FX outperforms" not in combined
