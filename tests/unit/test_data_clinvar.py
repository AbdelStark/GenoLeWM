"""Unit tests for local ClinVar VCF-to-Parquet preparation."""

from __future__ import annotations

from pathlib import Path

import pytest

import geno_lewm.data.clinvar as clinvar_mod
from geno_lewm.data import (
    CLINVAR_SCHEMA_VERSION,
    ClinvarVariant,
    iter_clinvar_shard,
    iter_clinvar_vcf_variants,
    label_set,
    prepare_clinvar_shard,
)
from geno_lewm.errors import InputError, RuntimeSetupError


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
    assert second.to_dict()["already_exists"] is True


def test_prepare_clinvar_shard_flushes_multiple_small_batches(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pytest.importorskip("pyarrow")
    monkeypatch.setattr(clinvar_mod, "_PARQUET_BATCH_ROWS", 1)
    vcf_path = _write_clinvar_vcf(tmp_path / "clinvar.vcf")

    report = prepare_clinvar_shard(vcf_path, tmp_path, release="2026-04-15")

    assert report.records_written == 6
    assert len(list(iter_clinvar_shard(report.output_path))) == 6


def test_prepare_clinvar_shard_cleans_tmp_file_on_write_error(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pytest.importorskip("pyarrow")
    vcf_path = _write_clinvar_vcf(tmp_path / "clinvar.vcf")

    def fail_write_batch(*_args: object, **_kwargs: object) -> object:
        raise RuntimeError("synthetic parquet write failure")

    monkeypatch.setattr(clinvar_mod, "_write_batch", fail_write_batch)

    with pytest.raises(RuntimeError, match="synthetic parquet write failure"):
        prepare_clinvar_shard(vcf_path, tmp_path, release="2026-04-15")

    target = tmp_path / "clinvar" / "2026-04-15" / "variants.parquet"
    assert not target.exists()
    assert not target.with_name(target.name + ".tmp").exists()


def test_clinvar_pyarrow_dependency_error_is_typed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original_import = clinvar_mod.importlib.import_module

    def fake_import_module(name: str) -> object:
        if name == "pyarrow":
            raise ImportError("missing pyarrow")
        return original_import(name)

    monkeypatch.setattr(clinvar_mod.importlib, "import_module", fake_import_module)

    with pytest.raises(RuntimeSetupError, match="requires pyarrow"):
        list(iter_clinvar_shard(tmp_path / "missing.parquet"))


def test_iter_clinvar_vcf_variants_maps_labels_and_fallback_ids(tmp_path: Path) -> None:
    vcf_path = tmp_path / "clinvar.vcf"
    vcf_path.write_text(
        "\n".join(
            [
                "##fileformat=VCFv4.2",
                "#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO",
                "chr1\t100\t123\tA\tG\t.\t.\tCLNSIG=likely%20pathogenic;CLNREVSTAT=.;GENEINFO=.",
                "1\t101\trs2\tC\tT\t.\t.\tCLNSIG=vus;CLNVID=456",
                "1\t102\trs3\tG\tA\t.\t.\tCLNSIG=likely_benign;ALLELEID=789",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    rows = list(iter_clinvar_vcf_variants(vcf_path))

    assert [row.clinical_significance for row in rows] == ["LP", "VUS", "LB"]
    assert [row.review_status for row in rows] == [".", "unknown", "unknown"]
    assert [row.gene_symbol for row in rows] == [None, None, None]
    assert [row.clinvar_id for row in rows] == [123, 456, 789]

    with pytest.raises(InputError, match="max_allele_len must be a positive integer"):
        list(iter_clinvar_vcf_variants(vcf_path, max_allele_len=False))


def test_iter_clinvar_vcf_variants_requires_numeric_identifier(tmp_path: Path) -> None:
    vcf_path = tmp_path / "clinvar_missing_id.vcf"
    vcf_path.write_text(
        "\n".join(
            [
                "##fileformat=VCFv4.2",
                "#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO",
                "1\t100\trs-not-numeric\tA\tG\t.\t.\tCLNSIG=Pathogenic",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    with pytest.raises(InputError, match="ClinVar row must contain CLNVID"):
        list(iter_clinvar_vcf_variants(vcf_path))


def test_prepare_clinvar_shard_writes_empty_filtered_shard(tmp_path: Path) -> None:
    pytest.importorskip("pyarrow")
    vcf_path = tmp_path / "clinvar_filtered.vcf"
    vcf_path.write_text(
        "\n".join(
            [
                "##fileformat=VCFv4.2",
                "#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO",
                "1\t100\t123\tA\t<DEL>\t.\t.\tCLNSIG=Pathogenic",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    report = prepare_clinvar_shard(vcf_path, tmp_path, release="2026-04-15")

    assert report.records_read == 1
    assert report.allele_records_seen == 1
    assert report.records_written == 0
    assert report.skipped_allele == 1
    assert list(iter_clinvar_shard(report.output_path)) == []


def test_clinvar_reports_and_release_validation(tmp_path: Path) -> None:
    variant = ClinvarVariant(
        chrom="1",
        pos=10,
        ref="A",
        alt="C",
        clinical_significance="P",
        review_status="criteria_provided",
        gene_symbol="BRCA1",
        clinvar_id=123,
    )

    assert variant.to_dict()["schema_version"] == CLINVAR_SCHEMA_VERSION
    with pytest.raises(InputError, match="release must be a non-empty string"):
        prepare_clinvar_shard(tmp_path / "missing.vcf", tmp_path, release="")


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
