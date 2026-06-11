# SPDX-License-Identifier: Apache-2.0
"""Tests for the GenoLeWM-FX residual model gate."""

from __future__ import annotations

import json
from pathlib import Path

from tools.research.fx_borzoi_residual import build_residual_report, render_markdown


def test_residual_model_learns_controlled_nonlinear_residual(tmp_path: Path) -> None:
    prediction_rows, report = build_residual_report(
        cache_manifest_path=_write_cache_manifest(tmp_path),
        baseline_report_path=_write_baseline_report(tmp_path),
        output_predictions=tmp_path / "predictions.parquet",
        generated_at="2026-06-11T14:00:00Z",
        cache_rows=_nonlinear_rows(signal=True),
        seeds=(271, 272),
        bootstrap_samples=100,
    )

    assert len(prediction_rows) > 0
    assert report["decision"] == "residual_lift_candidate"
    assert report["final_positive_claim_supported"] is True
    delta = _metric(report, "average_precision")["delta"]
    assert delta > 0.2
    assert report["collapse_diagnostics"]["all_predictions_constant"] is False
    markdown = render_markdown(report)
    assert "experimental residual-model gate only" in markdown


def test_residual_model_backs_off_on_no_signal_control(tmp_path: Path) -> None:
    _, report = build_residual_report(
        cache_manifest_path=_write_cache_manifest(tmp_path),
        baseline_report_path=_write_baseline_report(tmp_path),
        output_predictions=tmp_path / "predictions.parquet",
        cache_rows=_nonlinear_rows(signal=False),
        seeds=(271, 272),
        bootstrap_samples=100,
    )

    assert report["final_positive_claim_supported"] is False
    assert _metric(report, "average_precision")["ci95"][0] <= 0


def _write_cache_manifest(tmp_path: Path) -> Path:
    payload = {
        "schema_version": "1.0.0",
        "target_kind": "teacher_derived_traitgym_native_borzoi_score",
        "row_count": 480,
        "fipip_exact_join_status": "not_run_full_table_not_staged",
        "cache_artifact": {
            "path": "fixture.parquet",
            "rows": 480,
            "sha256": "sha256:" + "0" * 64,
            "size_bytes": 480,
        },
    }
    path = tmp_path / "cache-manifest.json"
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return path


def _write_baseline_report(tmp_path: Path) -> Path:
    payload = {
        "schema_version": "1.0.0",
        "decision": "go_residual_model",
        "ok_to_train_residual_model": True,
        "strongest_simple_baseline": {
            "baseline_id": "borzoi_plus_source_logistic_probe",
        },
    }
    path = tmp_path / "baseline-report.json"
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return path


def _nonlinear_rows(*, signal: bool) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    row_index = 0
    for split, repeats in (("train", 80), ("holdout", 40)):
        for _ in range(repeats):
            for borzoi_score, maf in ((0.1, 0.1), (0.1, 0.9), (0.9, 0.1), (0.9, 0.9)):
                label = (
                    int((borzoi_score > 0.5) ^ (maf > 0.5)) if signal else int(borzoi_score > 0.5)
                )
                rows.append(
                    {
                        "row_index": row_index,
                        "chrom": "1" if split == "train" else "3",
                        "pos": row_index + 1,
                        "ref": "A",
                        "alt": "G",
                        "label": label,
                        "split": split,
                        "maf": maf,
                        "ld_score": 1.0,
                        "tss_dist": 100,
                        "borzoi_score": borzoi_score,
                        "target_kind": "teacher_derived_traitgym_native_borzoi_score",
                    }
                )
                row_index += 1
    return rows


def _metric(report: dict[str, object], name: str) -> dict[str, object]:
    metrics = report["paired_deltas_vs_strongest_baseline"]
    assert isinstance(metrics, list)
    for metric in metrics:
        assert isinstance(metric, dict)
        if metric["name"] == name:
            return metric
    raise AssertionError(name)
