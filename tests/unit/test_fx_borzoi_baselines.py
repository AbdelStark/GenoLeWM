# SPDX-License-Identifier: Apache-2.0
"""Tests for the GenoLeWM-FX Borzoi baseline gate."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from tools.research.fx_borzoi_baselines import build_baseline_report, render_markdown

pytest.importorskip("sklearn")

REPO_ROOT = Path(__file__).resolve().parents[2]
SOURCE_MANIFEST = REPO_ROOT / "configs" / "fx" / "borzoi_rescue_sources.json"


def test_borzoi_baseline_gate_allows_non_saturated_residual_path(tmp_path: Path) -> None:
    report = build_baseline_report(
        source_manifest_path=SOURCE_MANIFEST,
        cache_manifest_path=_write_cache_manifest(tmp_path),
        generated_at="2026-06-11T13:00:00Z",
        cache_rows=_fixture_rows(saturated=False),
    )

    assert report["decision"] == "go_residual_model"
    assert report["ok_to_train_residual_model"] is True
    assert report["blockers"] == []
    assert report["fipip_exact_join_status"] == "not_run_full_table_not_staged"
    baseline_ids = {baseline["baseline_id"] for baseline in report["baselines"]}
    assert {
        "label_prior_constant",
        "direct_traitgym_native_borzoi",
        "source_logistic_probe",
        "borzoi_plus_source_logistic_probe",
    } <= baseline_ids
    markdown = render_markdown(report)
    assert "Decision: **go_residual_model**." in markdown
    assert "baseline and saturation gate only" in markdown


def test_borzoi_baseline_gate_blocks_saturated_simple_baseline(tmp_path: Path) -> None:
    report = build_baseline_report(
        source_manifest_path=SOURCE_MANIFEST,
        cache_manifest_path=_write_cache_manifest(tmp_path),
        cache_rows=_fixture_rows(saturated=True),
    )

    assert report["decision"] == "no_go_baseline_gate"
    assert report["ok_to_train_residual_model"] is False
    blocker_codes = {blocker["code"] for blocker in report["blockers"]}
    assert "simple_baseline_auprc_saturated" in blocker_codes


def _write_cache_manifest(tmp_path: Path) -> Path:
    payload = {
        "schema_version": "1.0.0",
        "target_kind": "teacher_derived_traitgym_native_borzoi_score",
        "row_count": 8,
        "fipip_exact_join_status": "not_run_full_table_not_staged",
        "cache_artifact": {
            "path": "fixture.parquet",
            "rows": 8,
            "sha256": "sha256:" + "0" * 64,
            "size_bytes": 8,
        },
    }
    path = tmp_path / "cache-manifest.json"
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return path


def _fixture_rows(*, saturated: bool) -> list[dict[str, object]]:
    train_scores = [0.8, 0.7, 0.2, 0.6]
    holdout_scores = [0.8, 0.7, 0.2, 0.6]
    if saturated:
        train_scores = [0.1, 0.9, 0.2, 0.8]
        holdout_scores = [0.1, 0.9, 0.2, 0.8]
    labels = [0, 1, 0, 1]
    rows: list[dict[str, object]] = []
    for offset, split in ((0, "train"), (100, "holdout")):
        scores = train_scores if split == "train" else holdout_scores
        for index, label in enumerate(labels):
            rows.append(
                {
                    "row_index": offset + index,
                    "chrom": "1" if split == "train" else "3",
                    "pos": offset + index + 1,
                    "ref": "A",
                    "alt": "G",
                    "label": label,
                    "split": split,
                    "trait": "",
                    "consequence": "intron_variant",
                    "match_group": "fixture",
                    "maf": 0.1,
                    "ld_score": 1.0,
                    "tss_dist": 100,
                    "borzoi_score": scores[index],
                    "borzoi_score_id": "fixture",
                    "target_kind": "teacher_derived_traitgym_native_borzoi_score",
                }
            )
    return rows
