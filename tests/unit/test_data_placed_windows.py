"""Tests for placed-window generation from variant shards."""

from __future__ import annotations

import json
from pathlib import Path

from geno_lewm.action import EditSpec
from tools.data.placed_windows import GENERATED_BY, export_placed_variant_windows


def test_export_placed_variant_windows_writes_coordinate_rows(tmp_path: Path) -> None:
    fasta = tmp_path / "reference.fa"
    fasta.write_text(">1 fixture\n" + ("A" * 512) + "\n", encoding="utf-8")
    output = tmp_path / "placed" / "windows.jsonl"
    variants = (
        EditSpec("1", 10, "A", "C"),
        EditSpec("1", 20, "A", "G"),
        EditSpec("1", 30, "A", "T"),
    )

    report = export_placed_variant_windows(
        variants,
        reference_fasta=fasta,
        output=output,
        min_variants_per_window=3,
    )

    assert report.windows_written == 1
    assert report.records_read == 3
    assert report.to_dict()["generated_by"] == GENERATED_BY
    rows = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()]
    assert len(rows) == 1
    row = rows[0]
    assert row["record_id"] == "gnomad:1:1-4096"
    assert row["source"] == "gnomad_common"
    assert row["chrom"] == "1"
    assert row["start_bp"] == 0
    assert row["end_bp"] == 4096
    assert row["variant_count"] == 3
    assert len(row["sequence"]) == 4096


def test_export_placed_variant_windows_skips_sparse_windows(tmp_path: Path) -> None:
    fasta = tmp_path / "reference.fa"
    fasta.write_text(">chr1 fixture\n" + ("A" * 512) + "\n", encoding="utf-8")
    output = tmp_path / "windows.jsonl"

    report = export_placed_variant_windows(
        (EditSpec("1", 10, "A", "C"),),
        reference_fasta=fasta,
        output=output,
        min_variants_per_window=3,
    )

    assert report.windows_written == 0
    assert report.skipped_sparse == 1
    assert output.read_text(encoding="utf-8") == ""
