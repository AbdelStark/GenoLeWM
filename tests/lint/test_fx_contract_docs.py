# SPDX-License-Identifier: Apache-2.0
"""Regression tests for the GenoLeWM-FX contract and decision docs."""

from __future__ import annotations

import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
CONTRACT = REPO_ROOT / "docs" / "research" / "fx-experiment-contract.md"
REPORT = REPO_ROOT / "docs" / "research" / "fx-feasibility-report.md"
DECISION = REPO_ROOT / "docs" / "research" / "fx-decision-package.md"
RESCUE = REPO_ROOT / "docs" / "research" / "fx-borzoi-rescue-plan.md"
OVERLAP = REPO_ROOT / "docs" / "research" / "fx-borzoi-overlap-report.md"
OVERLAP_JSON = REPO_ROOT / "docs" / "research" / "fx-borzoi-overlap-report.json"
CACHE = REPO_ROOT / "docs" / "research" / "fx-borzoi-cache-report.md"
CACHE_MANIFEST = REPO_ROOT / "docs" / "research" / "fx-borzoi-cache-manifest.json"
BASELINE = REPO_ROOT / "docs" / "research" / "fx-borzoi-baseline-report.md"
BASELINE_JSON = REPO_ROOT / "docs" / "research" / "fx-borzoi-baseline-report.json"
RESIDUAL = REPO_ROOT / "docs" / "research" / "fx-borzoi-residual-report.md"
RESIDUAL_JSON = REPO_ROOT / "docs" / "research" / "fx-borzoi-residual-report.json"
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
    normalized_decision = " ".join(decision.split())

    assert "Decision: **kill**." in report
    assert "no_public_teacher_delta_cache" in report
    assert "The parent epic takes the kill path" in decision
    assert "No FX demo ships" in decision
    assert "No FX positive-result paper section is added" in decision
    assert "no Stage B or Stage C job should launch" in decision
    assert "Borzoi rescue plan" in decision
    assert "Borzoi alignment and overlap report" in decision
    assert "Borzoi score cache report" in decision
    assert "Borzoi baseline and saturation report" in decision
    assert "Borzoi residual model report" in decision
    assert "not a model-quality claim" in normalized_decision
    assert "does not open a model-quality or training claim" in normalized_decision
    assert "not a trained GenoLeWM-FX model result" in normalized_decision
    assert "does not support a positive locked-result claim" in normalized_decision


def test_public_docs_link_fx_research_without_success_language() -> None:
    combined = "\n".join(path.read_text(encoding="utf-8") for path in (INDEX, README, MKDOCS))
    normalized = " ".join(combined.split())

    assert "research/fx-experiment-contract.md" in combined
    assert "research/fx-feasibility-report.md" in combined
    assert "research/fx-decision-package.md" in combined
    assert "research/fx-borzoi-rescue-plan.md" in combined
    assert "research/fx-borzoi-overlap-report.md" in combined
    assert "research/fx-borzoi-cache-report.md" in combined
    assert "research/fx-borzoi-baseline-report.md" in combined
    assert "research/fx-borzoi-residual-report.md" in combined
    assert "FX pivot" in combined
    assert "No GenoLeWM-FX model or demo ships" in combined
    assert "precomputed-Borzoi" in normalized
    assert "small, non-significant lift" in normalized
    assert "full fipip table join is optional staged provenance" in normalized
    assert "GenoLeWM-FX improves" not in combined
    assert "GenoLeWM-FX outperforms" not in combined


def test_fx_borzoi_rescue_plan_is_overlap_first_and_claim_bounded() -> None:
    text = RESCUE.read_text(encoding="utf-8")
    normalized = " ".join(text.split())
    required = (
        "This plan does not reverse the #257 kill decision",
        "precomputed Borzoi scores",
        "TraitGym-native, row-aligned precomputed Borzoi score artifacts",
        "full fipip table exact join remains an optional",
        "must not claim exact fipip table overlap",
        "more than 19 million common and low-frequency variants",
        "based on hg19",
        "at least 10,000 matched variants",
        "records a go decision for the TraitGym-native row-aligned path",
        "manifest-backed cache with 11,400 rows",
        "records a go decision for the residual-model gate",
        "paired confidence intervals cross zero",
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


def test_fx_borzoi_overlap_report_is_go_and_claim_bounded() -> None:
    text = OVERLAP.read_text(encoding="utf-8")
    payload = json.loads(OVERLAP_JSON.read_text(encoding="utf-8"))

    assert payload["decision"] == "go_traitgym_native_borzoi"
    assert payload["ok_to_build_cache"] is True
    assert payload["traitgym_native_alignment"]["usable_rows"] == 11400
    assert payload["traitgym_native_alignment"]["variant_key_summary"]["duplicate_key_count"] == 0
    assert payload["fipip_exact_join"]["status"] == "not_run_full_table_not_staged"
    assert "Decision: **go_traitgym_native_borzoi**." in text
    assert "makes no exact fipip overlap claim" in text
    assert "not a model-quality result" in text


def test_fx_borzoi_cache_manifest_is_manifest_backed_and_claim_bounded() -> None:
    text = CACHE.read_text(encoding="utf-8")
    payload = json.loads(CACHE_MANIFEST.read_text(encoding="utf-8"))

    assert payload["row_count"] == 11400
    assert payload["cache_artifact"]["rows"] == 11400
    assert payload["cache_artifact"]["sha256"].startswith("sha256:")
    assert payload["target_kind"] == "teacher_derived_traitgym_native_borzoi_score"
    assert payload["fipip_exact_join_status"] == "not_run_full_table_not_staged"
    assert payload["excluded_rows"] == 0
    assert payload["unmatched_rows"] == 0
    assert payload["duplicate_variant_keys"] == 0
    assert "teacher-derived targets" in text
    assert "not a model-quality result" in text


def test_fx_borzoi_baseline_report_is_go_and_claim_bounded() -> None:
    text = BASELINE.read_text(encoding="utf-8")
    payload = json.loads(BASELINE_JSON.read_text(encoding="utf-8"))

    assert payload["decision"] == "go_residual_model"
    assert payload["ok_to_train_residual_model"] is True
    assert payload["blockers"] == []
    assert payload["strongest_simple_baseline"]["baseline_id"] == (
        "borzoi_plus_source_logistic_probe"
    )
    assert payload["strongest_simple_baseline"]["holdout_metrics"][1]["value"] < 0.8
    assert payload["leakage_audit"]["duplicate_variant_keys"] == 0
    assert payload["leakage_audit"]["train_holdout_key_overlap"] == 0
    assert "Decision: **go_residual_model**." in text
    assert "baseline and saturation gate only" in text
    assert "not a model-quality result" in text


def test_fx_borzoi_residual_report_is_non_positive_claim_bounded() -> None:
    text = RESIDUAL.read_text(encoding="utf-8")
    payload = json.loads(RESIDUAL_JSON.read_text(encoding="utf-8"))

    assert payload["decision"] == "residual_lift_candidate"
    assert payload["final_positive_claim_supported"] is False
    assert payload["prediction_artifact"]["rows"] == 3390
    ap_delta = next(
        metric
        for metric in payload["paired_deltas_vs_strongest_baseline"]
        if metric["name"] == "average_precision"
    )
    assert ap_delta["delta"] > 0
    assert ap_delta["ci95"][0] < 0
    assert "Final positive claim supported: **False**." in text
    assert "paired confidence interval crosses zero" in text
    assert "not a final GenoLeWM-FX model-quality result" in text
