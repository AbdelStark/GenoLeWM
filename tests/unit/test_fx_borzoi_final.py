# SPDX-License-Identifier: Apache-2.0
"""Tests for the final GenoLeWM-FX Borzoi outcome report."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from geno_lewm.errors import InputError
from tools.research.fx_borzoi_final import build_final_report, render_markdown


def test_final_report_publishes_no_positive_claim(tmp_path: Path) -> None:
    paths = _write_inputs(tmp_path, final_positive=False)

    report = build_final_report(
        overlap_report_path=paths["overlap"],
        cache_manifest_path=paths["cache"],
        baseline_report_path=paths["baseline"],
        residual_report_path=paths["residual"],
        generated_at="2026-06-11T15:00:00Z",
    )

    assert report["outcome"] == "no_positive_claim_fragile_lift"
    assert report["positive_claim_allowed"] is False
    assert report["task"]["fipip_exact_join_status"] == "not_run_full_table_not_staged"
    assert "paired AUPRC and AUROC confidence intervals cross zero" in " ".join(
        report["negative_findings"]
    )
    markdown = render_markdown(report)
    assert "Outcome: **no_positive_claim_fragile_lift**." in markdown
    assert "Positive claim allowed: **False**." in markdown


def test_final_report_rejects_broken_chain(tmp_path: Path) -> None:
    paths = _write_inputs(tmp_path, final_positive=False)
    overlap = json.loads(paths["overlap"].read_text(encoding="utf-8"))
    overlap["decision"] = "no_go"
    paths["overlap"].write_text(json.dumps(overlap, indent=2) + "\n", encoding="utf-8")

    with pytest.raises(InputError, match="passing overlap"):
        build_final_report(
            overlap_report_path=paths["overlap"],
            cache_manifest_path=paths["cache"],
            baseline_report_path=paths["baseline"],
            residual_report_path=paths["residual"],
        )


def _write_inputs(tmp_path: Path, *, final_positive: bool) -> dict[str, Path]:
    overlap = {
        "schema_version": "1.0.0",
        "generated_by": "fixture",
        "decision": "go_traitgym_native_borzoi",
        "traitgym_native_alignment": {
            "dataset": "songlab/TraitGym",
            "config": "complex_traits",
            "split": "test",
            "usable_rows": 4,
        },
    }
    cache = {
        "schema_version": "1.0.0",
        "generated_by": "fixture",
        "row_count": 4,
        "score_id": "Borzoi_L2_L2.plus.all",
        "score_column": "borzoi_score",
        "target_kind": "teacher_derived_traitgym_native_borzoi_score",
        "fipip_exact_join_status": "not_run_full_table_not_staged",
        "cache_artifact": {"path": "cache.parquet", "rows": 4},
    }
    baseline = {
        "schema_version": "1.0.0",
        "generated_by": "fixture",
        "decision": "go_residual_model",
        "strongest_simple_baseline": {
            "baseline_id": "borzoi_plus_source_logistic_probe",
            "holdout_metrics": [
                {"name": "auroc", "value": 0.65},
                {"name": "average_precision", "value": 0.22},
                {"name": "balanced_accuracy", "value": 0.51},
            ],
        },
    }
    residual = {
        "schema_version": "1.0.0",
        "generated_by": "fixture",
        "decision": "residual_lift_candidate",
        "prediction_artifact": {"path": "pred.parquet", "rows": 2},
        "residual_ensemble_metrics": [
            {"name": "auroc", "value": 0.66},
            {"name": "average_precision", "value": 0.23},
            {"name": "balanced_accuracy", "value": 0.50},
        ],
        "paired_deltas_vs_strongest_baseline": [
            {"name": "auroc", "delta": 0.01, "ci95": [-0.01, 0.02]},
            {"name": "average_precision", "delta": 0.01, "ci95": [-0.01, 0.02]},
        ],
        "seed_variance": {"average_precision_std": 0.001},
        "final_positive_claim_supported": final_positive,
    }
    payloads = {
        "overlap": overlap,
        "cache": cache,
        "baseline": baseline,
        "residual": residual,
    }
    paths: dict[str, Path] = {}
    for name, payload in payloads.items():
        path = tmp_path / f"{name}.json"
        path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        paths[name] = path
    return paths
