"""Unit tests for local ClinVar VCF-to-Parquet preparation."""

from __future__ import annotations

from pathlib import Path

import pytest

from geno_lewm.data import (
    CLINVAR_SCHEMA_VERSION,
    iter_clinvar_shard,
    label_set,
    prepare_clinvar_shard,
)


def test_prepare_clinvar_shard_preserves_vus_but_excludes_from_labels(tmp_path: Path) -> None:
    pytest.importorskip("pyarrow")
    vcf_path = _write_clinvar_vcf(tmp_path / "clinvar.vcf")

    report = prepare_clinvar_shard(vcf_path, tmp_path, release="2026-04-15")

    assert report.output_path == tmp_path / "clinvar" / "2026-04-15" / "variants.parquet"
    assert report.records_read == 6
    assert report.allele_records_seen == 7
    assert report.records_written == 6
    assert report.skipped_allele == 1

    rows = list(iter_clinvar_shard(report.output_path))
    assert [row.clinical_significance for row in rows] == ["P", "VUS", "B", "OTHER", "LP", "LB"]
    assert [row.clinvar_id for row in rows] == [111, 222, 333, 444, 555, 556]
    assert rows[0].gene_symbol == "BRCA1"
    assert rows[0].schema_version == CLINVAR_SCHEMA_VERSION
    assert [row.clinical_significance for row in label_set(rows)] == ["P", "B", "LP", "LB"]


def test_prepare_clinvar_shard_is_idempotent_without_overwrite(tmp_path: Path) -> None:
    pytest.importorskip("pyarrow")
    vcf_path = _write_clinvar_vcf(tmp_path / "clinvar.vcf")
    first = prepare_clinvar_shard(vcf_path, tmp_path, release="2026-04-15")

    second = prepare_clinvar_shard(vcf_path, tmp_path, release="2026-04-15")

    assert second.output_path == first.output_path
    assert second.already_exists is True
    assert second.records_read == 0
    assert second.records_written == 6
    assert second.size_bytes == first.size_bytes


def _write_clinvar_vcf(path: Path) -> Path:
    path.write_text(
        "\n".join(
            [
                "##fileformat=VCFv4.2",
                "#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO",
                (
                    "chr1\t100\t101\tA\tG\t.\t.\t"
                    "CLNSIG=Pathogenic;CLNREVSTAT=criteria_provided;GENEINFO=BRCA1:672;"
                    "ALLELEID=111"
                ),
                (
                    "1\t101\t102\tC\tT\t.\t.\t"
                    "CLNSIG=Uncertain_significance;CLNREVSTAT=reviewed_by_expert_panel;"
                    "GENEINFO=CFTR:1080;ALLELEID=222"
                ),
                "1\t102\t103\tG\tA\t.\t.\tCLNSIG=Benign;GENEINFO=CFTR:1080;ALLELEID=333",
                (
                    "1\t103\t104\tT\tC\t.\t.\t"
                    "CLNSIG=Conflicting_classifications_of_pathogenicity;ALLELEID=444"
                ),
                "1\t104\t105\tA\t<DEL>\t.\t.\tCLNSIG=Pathogenic;ALLELEID=999",
                (
                    "1\t105\trs106\tA\tC,G\t.\t.\t"
                    "CLNSIG=Likely_pathogenic,Likely_benign;GENEINFO=GENE1:1;"
                    "ALLELEID=555,556"
                ),
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    return path
