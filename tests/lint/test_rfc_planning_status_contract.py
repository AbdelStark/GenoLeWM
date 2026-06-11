# SPDX-License-Identifier: Apache-2.0
"""Regression tests for the RFC-0008 implementation-status boundary."""

from __future__ import annotations

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
RFC = REPO_ROOT / "rfcs" / "0008-latent-planning.md"
CLI = REPO_ROOT / "geno_lewm" / "cli" / "plan.py"
BENCH = REPO_ROOT / "bench" / "planning.py"
HF_MODEL_CARD = REPO_ROOT / "docs" / "release" / "huggingface-model-card.md"


def test_rfc_records_current_planning_surface_and_limitations() -> None:
    text = RFC.read_text(encoding="utf-8")
    required = (
        "- **Updated:** 2026-06-11",
        "manifest-runtime\n  `geno-lewm-plan`",
        "explicit sequence-proxy smoke mode",
        "`bench.planning` deterministic pure-solver performance reporting",
        "best_distance=23.656930390534644",
        "after 384 evaluations",
        "stopped by patience",
        "not proof of useful\n  planning behavior",
        "Named target-hardware performance acceptance",
        "planning-quality evidence remain open",
    )

    for fragment in required:
        assert fragment in text


def test_rfc_api_sketch_uses_evaluator_first_cem_contract() -> None:
    text = RFC.read_text(encoding="utf-8")
    required = (
        "class CandidateEvaluation:",
        "best_edits: tuple[RelEdit, ...]",
        "best_cost: float",
        "best_objective: float",
        "n_evaluations: int",
        "stopped_reason: str",
        "evaluate: Callable[[Sequence[RelEdit]], float | CandidateEvaluation]",
        "cost_fn: Callable[[Sequence[RelEdit]], float] | None = None",
        "deterministic given a seed and deterministic\n`evaluate` callback",
    )
    stale_fragments = (
        "predictor: Predictor",
        "action_encoder: ActionEncoder",
        "best_edits: list[RelEdit]",
        "n_predictor_calls: int",
    )

    for fragment in required:
        assert fragment in text
    for fragment in stale_fragments:
        assert fragment not in text


def test_rfc_planning_evidence_tracks_code_and_release_docs() -> None:
    rfc_text = RFC.read_text(encoding="utf-8")
    cli_text = CLI.read_text(encoding="utf-8")
    bench_text = BENCH.read_text(encoding="utf-8")
    model_card_text = HF_MODEL_CARD.read_text(encoding="utf-8")
    normalized_rfc = " ".join(rfc_text.split())

    assert "MANIFEST_RUNTIME_MODE" in cli_text
    assert "SEQUENCE_PROXY_MODE" in cli_text
    assert "planning.performance.json" in bench_text
    assert "TARGET_SECONDS_BY_PROFILE" in bench_text
    assert "not useful-planning evidence" in model_card_text

    assert "Manifest-runtime output is released-artifact\nexecution evidence" in rfc_text
    assert "Sequence-proxy output is a local smoke path only" in rfc_text
    assert "without measuring Carbon encoding or learned model quality" in normalized_rfc
