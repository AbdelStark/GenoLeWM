# SPDX-License-Identifier: Apache-2.0
"""Tests for the GenoLeWM-FX Borzoi score cache tooling."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from geno_lewm.errors import InputError
from tools.research.fx_borzoi_cache import (
    build_cache_package,
    build_cache_rows,
    load_cache_manifest,
    read_cache_rows,
    write_cache_package,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
SOURCE_MANIFEST = REPO_ROOT / "configs" / "fx" / "borzoi_rescue_sources.json"


def test_borzoi_cache_writes_manifest_backed_rows(tmp_path: Path) -> None:
    overlap_report = _write_overlap_report(tmp_path, decision="go_traitgym_native_borzoi")
    output_cache = tmp_path / "cache.parquet"
    output_manifest = tmp_path / "cache-manifest.json"
    output_md = tmp_path / "cache-report.md"

    rows, manifest = build_cache_package(
        source_manifest_path=SOURCE_MANIFEST,
        overlap_report_path=overlap_report,
        output_cache=output_cache,
        generated_at="2026-06-11T12:30:00Z",
        traitgym_records=_fixture_records(),
        score_values=[0.1, 0.9, 0.2, 0.8],
    )
    write_cache_package(
        rows=rows,
        manifest=manifest,
        output_cache=output_cache,
        output_manifest=output_manifest,
        output_md=output_md,
    )

    payload = load_cache_manifest(output_manifest)
    assert payload["row_count"] == 4
    assert payload["cache_artifact"]["rows"] == 4
    assert payload["cache_artifact"]["sha256"].startswith("sha256:")
    assert payload["target_kind"] == "teacher_derived_traitgym_native_borzoi_score"
    assert payload["fipip_exact_join_status"] == "not_run_full_table_not_staged"
    loaded = read_cache_rows(
        output_manifest,
        columns=["chrom", "pos", "split", "borzoi_score", "target_kind"],
    )
    assert loaded[0] == {
        "chrom": "1",
        "pos": 10,
        "split": "train",
        "borzoi_score": 0.1,
        "target_kind": "teacher_derived_traitgym_native_borzoi_score",
    }
    assert {row["split"] for row in loaded} == {"train", "holdout"}
    assert "fipip" not in loaded[0]
    assert "Borzoi score cache report" in output_md.read_text(encoding="utf-8")


def test_borzoi_cache_rejects_failed_overlap_gate(tmp_path: Path) -> None:
    overlap_report = _write_overlap_report(tmp_path, decision="no_go")

    with pytest.raises(InputError, match="requires a passing"):
        build_cache_package(
            source_manifest_path=SOURCE_MANIFEST,
            overlap_report_path=overlap_report,
            output_cache=tmp_path / "cache.parquet",
            traitgym_records=_fixture_records(),
            score_values=[0.1, 0.9, 0.2, 0.8],
        )


def test_borzoi_cache_rows_reject_duplicate_variant_keys() -> None:
    records = _fixture_records()
    records[1] = dict(records[0])

    with pytest.raises(InputError, match="duplicate normalized variant key"):
        build_cache_rows(
            records=records,
            score_values=[0.1, 0.9, 0.2, 0.8],
            holdout_chromosomes=["3", "11"],
            score_id="fixture",
        )


def _write_overlap_report(tmp_path: Path, *, decision: str) -> Path:
    payload = {
        "schema_version": "1.0.0",
        "decision": decision,
        "ok_to_build_cache": decision == "go_traitgym_native_borzoi",
        "traitgym_native_alignment": {"usable_rows": 4},
        "fipip_exact_join": {"status": "not_run_full_table_not_staged"},
        "source_inputs": {
            "traitgym_slice": {"path": "slice.parquet"},
            "traitgym_borzoi_score": {"path": "score.parquet"},
            "fipip_borzoi_source": {"object": "sniff_102_annotations.gz"},
        },
    }
    path = tmp_path / "overlap.json"
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return path


def _fixture_records() -> list[dict[str, object]]:
    return [
        {
            "chrom": "1",
            "pos": 10,
            "ref": "A",
            "alt": "G",
            "label": 0,
            "trait": "",
            "consequence": "intron_variant",
            "match_group": "intron_variant_0",
            "maf": 0.1,
            "ld_score": 1.0,
            "tss_dist": 100,
        },
        {
            "chrom": "2",
            "pos": 20,
            "ref": "C",
            "alt": "T",
            "label": 1,
            "trait": "Height",
            "consequence": "dELS",
            "match_group": "dELS_1",
            "maf": 0.2,
            "ld_score": 2.0,
            "tss_dist": 200,
        },
        {
            "chrom": "3",
            "pos": 30,
            "ref": "A",
            "alt": "C",
            "label": 0,
            "trait": "",
            "consequence": "pELS",
            "match_group": "pELS_0",
            "maf": 0.3,
            "ld_score": 3.0,
            "tss_dist": 300,
        },
        {
            "chrom": "11",
            "pos": 40,
            "ref": "G",
            "alt": "A",
            "label": 1,
            "trait": "Plt",
            "consequence": "PLS",
            "match_group": "PLS_1",
            "maf": 0.4,
            "ld_score": 4.0,
            "tss_dist": 400,
        },
    ]
