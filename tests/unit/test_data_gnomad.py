"""Unit tests for local gnomAD VCF-to-Parquet preparation."""

from __future__ import annotations

from pathlib import Path

import pytest

import geno_lewm.data.gnomad as gnomad_mod
from geno_lewm.data import (
    GNOMAD_SCHEMA_VERSION,
    GnomadVariant,
    iter_gnomad_shard,
    iter_gnomad_vcf_variants,
    prepare_gnomad_shard,
)
from geno_lewm.data.gnomad import _optional_float
from geno_lewm.errors import InputError, RuntimeSetupError
from geno_lewm.provenance import sha256_file


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
    assert report.input_path == vcf_path
    assert report.input_sha256 == sha256_file(vcf_path)
    assert report.output_sha256 == sha256_file(report.output_path)
    assert report.input_size_bytes == vcf_path.stat().st_size
    assert report.size_bytes > 0
    assert report.elapsed_seconds > 0
    assert report.to_dict()["input_sha256"] == report.input_sha256

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
    assert second.input_sha256 == first.input_sha256
    assert second.output_sha256 == first.output_sha256
    assert second.size_bytes == first.size_bytes
    assert second.to_dict()["already_exists"] is True
    assert second.elapsed_seconds >= 0


def test_prepare_gnomad_shard_flushes_multiple_small_batches(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pytest.importorskip("pyarrow")
    monkeypatch.setattr(gnomad_mod, "_PARQUET_BATCH_ROWS", 1)
    vcf_path = _write_gnomad_vcf(tmp_path / "gnomad.vcf")

    report = prepare_gnomad_shard(vcf_path, tmp_path, release="v4.1", min_af=0.01)

    assert report.records_written == 2
    assert len(list(iter_gnomad_shard(report.output_path))) == 2


def test_prepare_gnomad_shard_cleans_tmp_file_on_write_error(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pytest.importorskip("pyarrow")
    vcf_path = _write_gnomad_vcf(tmp_path / "gnomad.vcf")

    def fail_write_batch(*_args: object, **_kwargs: object) -> object:
        raise RuntimeError("synthetic parquet write failure")

    monkeypatch.setattr(gnomad_mod, "_write_batch", fail_write_batch)

    with pytest.raises(RuntimeError, match="synthetic parquet write failure"):
        prepare_gnomad_shard(vcf_path, tmp_path, release="v4.1", min_af=0.01)

    target = tmp_path / "gnomad" / "v4.1" / "variants.parquet"
    assert not target.exists()
    assert not target.with_name(target.name + ".tmp").exists()


def test_gnomad_pyarrow_dependency_error_is_typed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original_import = gnomad_mod.importlib.import_module

    def fake_import_module(name: str) -> object:
        if name == "pyarrow":
            raise ImportError("missing pyarrow")
        return original_import(name)

    monkeypatch.setattr(gnomad_mod.importlib, "import_module", fake_import_module)

    with pytest.raises(RuntimeSetupError, match="requires pyarrow"):
        list(iter_gnomad_shard(tmp_path / "missing.parquet"))


def test_iter_gnomad_vcf_variants_filters_and_validates_inputs(tmp_path: Path) -> None:
    vcf_path = _write_gnomad_vcf(tmp_path / "gnomad.vcf")

    rows = list(iter_gnomad_vcf_variants(vcf_path, min_af=0.01, max_allele_len=1))

    assert [(row.chrom, row.pos, row.ref, row.alt, row.af_global) for row in rows] == [
        ("1", 10, "A", "C", pytest.approx(0.02)),
        ("MT", 40, "C", "T", pytest.approx(0.02)),
    ]
    assert rows[1].af_eas == pytest.approx(0.04)

    with pytest.raises(InputError, match="min_af must be between 0 and 1"):
        list(iter_gnomad_vcf_variants(vcf_path, min_af=True))
    with pytest.raises(InputError, match="min_af must be between 0 and 1"):
        list(iter_gnomad_vcf_variants(vcf_path, min_af=1.5))
    with pytest.raises(InputError, match="max_allele_len must be a positive integer"):
        list(iter_gnomad_vcf_variants(vcf_path, max_allele_len=0))


def test_prepare_gnomad_shard_writes_empty_filtered_shard(tmp_path: Path) -> None:
    pytest.importorskip("pyarrow")
    vcf_path = tmp_path / "rare_or_filtered.vcf"
    vcf_path.write_text(
        "\n".join(
            [
                "##fileformat=VCFv4.2",
                "#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO",
                "1\t10\trs1\tA\tC\t.\tLowQual\tAF=0.5",
                "1\t11\trs2\tA\tG\t.\tPASS\tAF=0.001",
                "1\t12\trs3\tA\t<DEL>\t.\tPASS\tAF=0.5",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    report = prepare_gnomad_shard(vcf_path, tmp_path, release="v4.1", min_af=0.01)

    assert report.records_read == 3
    assert report.allele_records_seen == 3
    assert report.records_written == 0
    assert report.skipped_filter == 1
    assert report.skipped_af == 1
    assert report.skipped_allele == 1
    assert list(iter_gnomad_shard(report.output_path)) == []


def test_gnomad_reports_and_private_float_validation(tmp_path: Path) -> None:
    variant = GnomadVariant(
        chrom="1",
        pos=10,
        ref="A",
        alt="C",
        af_global=0.1,
        af_afr=None,
        af_ami=None,
        af_amr=None,
        af_asj=None,
        af_eas=None,
        af_fin=None,
        af_nfe=None,
        af_oth=None,
        af_sas=None,
        filter="PASS",
    )
    assert variant.to_dict()["schema_version"] == GNOMAD_SCHEMA_VERSION
    assert _optional_float("0.125") == pytest.approx(0.125)
    with pytest.raises(InputError, match="non-float allele frequency"):
        _optional_float(object())
    with pytest.raises(InputError, match="release must be a non-empty string"):
        prepare_gnomad_shard(tmp_path / "missing.vcf", tmp_path, release="")


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
