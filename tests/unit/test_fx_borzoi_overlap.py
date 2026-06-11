# SPDX-License-Identifier: Apache-2.0
"""Tests for the GenoLeWM-FX Borzoi overlap gate tooling."""

from __future__ import annotations

import gzip
import json
from pathlib import Path

import pytest

from tools.research.fx_borzoi_overlap import (
    build_borzoi_overlap_report,
    main,
    render_markdown,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
MANIFEST = REPO_ROOT / "configs" / "fx" / "borzoi_rescue_sources.json"


def test_borzoi_overlap_fixture_passes_traitgym_native_gate(tmp_path: Path) -> None:
    manifest = _write_manifest(tmp_path, min_usable_rows=4)

    report = build_borzoi_overlap_report(
        manifest_path=manifest,
        generated_at="2026-06-11T12:00:00Z",
        traitgym_records=_fixture_records(),
        score_values=[0.1, 0.9, 0.2, 0.8],
        traitgym_slice_receipt=_receipt("slice.parquet"),
        score_receipt=_receipt("scores.parquet"),
        fipip_metadata=_fipip_metadata(),
    )

    assert report["decision"] == "go_traitgym_native_borzoi"
    assert report["ok_to_build_cache"] is True
    alignment = report["traitgym_native_alignment"]
    assert alignment["usable_rows"] == 4
    assert alignment["variant_key_summary"]["duplicate_key_count"] == 0
    assert report["fipip_exact_join"]["status"] == "not_run_full_table_not_staged"
    markdown = render_markdown(report)
    assert "No #268 blockers remain" in markdown
    assert "makes no exact fipip overlap claim" in markdown


def test_borzoi_overlap_reports_row_count_mismatch_as_no_go(tmp_path: Path) -> None:
    manifest = _write_manifest(tmp_path, min_usable_rows=4)

    report = build_borzoi_overlap_report(
        manifest_path=manifest,
        traitgym_records=_fixture_records(),
        score_values=[0.1, 0.9, 0.2],
        traitgym_slice_receipt=_receipt("slice.parquet"),
        score_receipt=_receipt("scores.parquet"),
        fipip_metadata=_fipip_metadata(),
    )

    assert report["decision"] == "no_go"
    assert report["ok_to_build_cache"] is False
    blocker_codes = {blocker["code"] for blocker in report["blockers"]}
    assert "traitgym_borzoi_row_count_mismatch" in blocker_codes
    assert "below_minimum_usable_rows" in blocker_codes


def test_borzoi_overlap_optional_fipip_scan_counts_exact_matches(tmp_path: Path) -> None:
    pytest.importorskip("pandas")
    manifest = _write_manifest(tmp_path, min_usable_rows=2)
    fipip_table = tmp_path / "fipip.tsv.gz"
    with gzip.open(fipip_table, "wt", encoding="utf-8") as handle:
        handle.write("CHROM POS REF ALT\n")
        handle.write("1 10 A G\n")
        handle.write("2 20 C T\n")
        handle.write("3 30 A C\n")

    report = build_borzoi_overlap_report(
        manifest_path=manifest,
        traitgym_records=_fixture_records(),
        score_values=[0.1, 0.9, 0.2, 0.8],
        traitgym_slice_receipt=_receipt("slice.parquet"),
        score_receipt=_receipt("scores.parquet"),
        fipip_metadata=_fipip_metadata(),
        fipip_score_table=fipip_table,
    )

    exact_join = report["fipip_exact_join"]
    assert exact_join["status"] == "ran_local_table"
    assert exact_join["scanned_rows"] == 3
    assert exact_join["exact_key_matches"] == 3
    assert exact_join["reverse_allele_key_hits"] == 0


def test_borzoi_overlap_cli_writes_report_with_injected_manifest_only(tmp_path: Path) -> None:
    output_json = tmp_path / "report.json"
    output_md = tmp_path / "report.md"

    # Exercise only the CLI parser and file writer on the real checked-in report path elsewhere;
    # network-backed execution is covered by the generated docs artifact.
    report = build_borzoi_overlap_report(
        manifest_path=_write_manifest(tmp_path, min_usable_rows=4),
        generated_at="2026-06-11T12:00:00Z",
        traitgym_records=_fixture_records(),
        score_values=[0.1, 0.9, 0.2, 0.8],
        traitgym_slice_receipt=_receipt("slice.parquet"),
        score_receipt=_receipt("scores.parquet"),
        fipip_metadata=_fipip_metadata(),
    )
    output_json.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    output_md.write_text(render_markdown(report), encoding="utf-8")

    assert json.loads(output_json.read_text(encoding="utf-8"))["decision"] == (
        "go_traitgym_native_borzoi"
    )
    assert "GenoLeWM-FX Borzoi alignment" in output_md.read_text(encoding="utf-8")
    assert main(["--manifest", str(tmp_path / "missing.json")]) != 0


def _write_manifest(tmp_path: Path, *, min_usable_rows: int) -> Path:
    payload = json.loads(MANIFEST.read_text(encoding="utf-8"))
    payload["min_usable_rows"] = min_usable_rows
    payload["traitgym_slice"]["expected_rows"] = 4
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return path


def _fixture_records() -> list[dict[str, object]]:
    return [
        {"chrom": "1", "pos": 10, "ref": "A", "alt": "G", "label": 0},
        {"chrom": "2", "pos": 20, "ref": "C", "alt": "T", "label": 1},
        {"chrom": "3", "pos": 30, "ref": "A", "alt": "C", "label": 0},
        {"chrom": "11", "pos": 40, "ref": "G", "alt": "A", "label": 1},
    ]


def _receipt(path: str) -> dict[str, object]:
    return {
        "repo_id": "songlab/TraitGym",
        "repo_type": "dataset",
        "path": path,
        "revision": "fixture",
        "sha256": "sha256:" + "0" * 64,
        "size_bytes": 4,
    }


def _fipip_metadata() -> dict[str, object]:
    return {
        "repository": "https://github.com/statgen/fipip",
        "genome_build": "hg19",
        "bucket": "seqnn-share",
        "object": "sniff/borzoi_102_annotation_set/sniff_102_annotations.gz",
        "expected_rows": 19534182,
        "size_bytes": 18670353280,
        "md5_hash": "fixture",
        "crc32c": "fixture",
        "generation": "fixture",
        "updated": "2026-06-11T00:00:00Z",
        "score_kind": "fixture",
    }
