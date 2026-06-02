"""Unit tests for local gnomAD VCF-to-Parquet preparation."""

from __future__ import annotations

from pathlib import Path

import pytest

from geno_lewm.data import GNOMAD_SCHEMA_VERSION, iter_gnomad_shard, prepare_gnomad_shard


def test_prepare_gnomad_shard_filters_common_pass_variants(tmp_path: Path) -> None:
    pytest.importorskip("pyarrow")
    vcf_path = _write_gnomad_vcf(tmp_path / "gnomad.vcf")

    report = prepare_gnomad_shard(vcf_path, tmp_path, release="v4.1", min_af=0.01)

    assert report.output_path == tmp_path / "gnomad" / "v4.1" / "variants.parquet"
    assert report.records_read == 4
    assert report.allele_records_seen == 5
    assert report.records_written == 2
    assert report.skipped_filter == 1
    assert report.skipped_af == 1
    assert report.skipped_allele == 1
    assert report.size_bytes > 0

    rows = list(iter_gnomad_shard(report.output_path))
    assert [(row.chrom, row.pos, row.ref, row.alt) for row in rows] == [
        ("1", 10, "A", "C"),
        ("MT", 40, "C", "T"),
    ]
    assert rows[0].af_global == pytest.approx(0.02)
    assert rows[0].af_afr == pytest.approx(0.03)
    assert rows[0].schema_version == GNOMAD_SCHEMA_VERSION


def test_prepare_gnomad_shard_is_idempotent_without_overwrite(tmp_path: Path) -> None:
    pytest.importorskip("pyarrow")
    vcf_path = _write_gnomad_vcf(tmp_path / "gnomad.vcf")
    first = prepare_gnomad_shard(vcf_path, tmp_path, release="v4.1", min_af=0.01)

    second = prepare_gnomad_shard(vcf_path, tmp_path, release="v4.1", min_af=0.01)

    assert second.output_path == first.output_path
    assert second.already_exists is True
    assert second.records_read == 0
    assert second.records_written == 2
    assert second.size_bytes == first.size_bytes


def _write_gnomad_vcf(path: Path) -> Path:
    path.write_text(
        "\n".join(
            [
                "##fileformat=VCFv4.2",
                "#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO",
                "chr1\t10\trs1\tA\tC,G\t.\tPASS\tAF=0.02,0.005;AF_afr=0.03,0.001",
                "1\t20\trs2\tA\tT\t.\tLowQual\tAF=0.50",
                "2\t30\trs3\tA\t<DEL>\t.\tPASS\tAF=0.20",
                "chrM\t40\trs4\tC\tT\t.\tPASS\tAF_global=0.02;AF_eas=0.04",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    return path
