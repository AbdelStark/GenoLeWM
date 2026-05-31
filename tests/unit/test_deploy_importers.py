"""Unit tests for local-only personal-genome importers."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from geno_lewm.deploy.import_ import (
    convert_23andme_to_vcf,
    convert_ancestry_to_vcf,
    convert_myheritage_to_vcf,
    convert_sequencing_json_to_vcf,
)
from geno_lewm.deploy.import_._common import (
    variant_from_explicit_alleles,
    write_vcf_variants,
)
from geno_lewm.deploy.import_.myheritage import parse_myheritage
from geno_lewm.deploy.import_.sequencing import parse_sequencing_json
from geno_lewm.deploy.runtime import fail_closed_network_guard
from geno_lewm.errors import VcfParseError


def _variant_lines(path: Path) -> list[str]:
    return [
        line
        for line in path.read_text(encoding="utf-8").splitlines()
        if line and not line.startswith("#")
    ]


def test_23andme_converter_round_trips_synthetic_fixture_without_network(
    tmp_path: Path,
) -> None:
    raw = tmp_path / "23andme.txt"
    raw.write_text(
        "# comment\n"
        "rsid\tchromosome\tposition\tgenotype\n"
        "rs1\t1\t100\tAG\n"
        "rs2\t1\t101\tAA\n"
        "rs3\t1\t102\t--\n",
        encoding="utf-8",
    )

    with fail_closed_network_guard():
        summary = convert_23andme_to_vcf(
            raw,
            tmp_path / "out.vcf",
            {("1", 100): "A", ("1", 101): "A", ("1", 102): "C"},
            sample_id="NA0001",
        )

    assert summary.records_written == 1
    assert summary.ref_calls_skipped == 1
    assert summary.no_calls_skipped == 1
    assert _variant_lines(summary.output_path) == [
        "1\t100\trs1\tA\tG\t.\tPASS\tSOURCE=23andMe\tGT\t0/1"
    ]


def test_23andme_converter_requires_local_reference_allele(tmp_path: Path) -> None:
    raw = tmp_path / "23andme.txt"
    raw.write_text("rs1\t1\t100\tAG\n", encoding="utf-8")

    with pytest.raises(VcfParseError, match="reference allele is required"):
        convert_23andme_to_vcf(raw, tmp_path / "out.vcf", {})


def test_ancestry_converter_handles_split_allele_columns_and_chrom_aliases(
    tmp_path: Path,
) -> None:
    raw = tmp_path / "ancestry.txt"
    raw.write_text(
        "# comment\nrsid chromosome position allele1 allele2\nrsX\t23\t200\tC\tT\n",
        encoding="utf-8",
    )

    with fail_closed_network_guard():
        summary = convert_ancestry_to_vcf(raw, tmp_path / "ancestry.vcf", {("X", 200): "C"})

    assert summary.records_written == 1
    assert _variant_lines(summary.output_path) == [
        "X\t200\trsX\tC\tT\t.\tPASS\tSOURCE=AncestryDNA\tGT\t0/1"
    ]


def test_myheritage_converter_accepts_csv_fixture(tmp_path: Path) -> None:
    raw = tmp_path / "myheritage.csv"
    raw.write_text(
        "RSID,CHROMOSOME,POSITION,RESULT\nrs2,chr2,300,TT\n",
        encoding="utf-8",
    )

    with fail_closed_network_guard():
        summary = convert_myheritage_to_vcf(raw, tmp_path / "myheritage.vcf", {("2", 300): "G"})

    assert summary.records_written == 1
    assert _variant_lines(summary.output_path) == [
        "2\t300\trs2\tG\tT\t.\tPASS\tSOURCE=MyHeritage\tGT\t1/1"
    ]


def test_sequencing_json_converter_accepts_vcf_equivalent_variant_rows(
    tmp_path: Path,
) -> None:
    raw = tmp_path / "sequencing.json"
    raw.write_text(
        json.dumps(
            {
                "variants": [
                    {
                        "id": "rs3",
                        "chromosome": "3",
                        "position": 400,
                        "ref": "C",
                        "alt": "T",
                        "genotype": "0/1",
                    },
                    {
                        "rsid": "rs4",
                        "chr": "MT",
                        "pos": "500",
                        "reference": "A",
                        "alternateAlleles": ["C", "G"],
                        "genotype": ["A", "G"],
                    },
                ]
            }
        ),
        encoding="utf-8",
    )

    with fail_closed_network_guard():
        summary = convert_sequencing_json_to_vcf(raw, tmp_path / "sequencing.vcf")

    assert len(parse_sequencing_json(raw)) == 2
    assert summary.records_written == 2
    assert _variant_lines(summary.output_path) == [
        "3\t400\trs3\tC\tT\t.\tPASS\tSOURCE=Sequencing.com\tGT\t0/1",
        "MT\t500\trs4\tA\tC,G\t.\tPASS\tSOURCE=Sequencing.com\tGT\t0/2",
    ]


def test_importers_reject_unsupported_or_malformed_rows(tmp_path: Path) -> None:
    raw = tmp_path / "ancestry.txt"
    raw.write_text("rsBad\t1\t10\tI\tD\n", encoding="utf-8")
    with pytest.raises(VcfParseError, match="unsupported non-SNV allele"):
        convert_ancestry_to_vcf(raw, tmp_path / "bad.vcf", {("1", 10): "A"})

    bad_json = tmp_path / "bad.json"
    bad_json.write_text(json.dumps({"variants": [{"chrom": "1", "pos": 1, "ref": "A"}]}))
    with pytest.raises(VcfParseError, match="missing alternate allele"):
        parse_sequencing_json(bad_json)


def test_array_importers_validate_io_positions_reference_and_sample_id(tmp_path: Path) -> None:
    with pytest.raises(VcfParseError, match="could not read"):
        convert_23andme_to_vcf(tmp_path / "missing.txt", tmp_path / "out.vcf", {})

    bad_pos = tmp_path / "bad-pos.txt"
    bad_pos.write_text("rs1\t1\tNaN\tAG\n", encoding="utf-8")
    with pytest.raises(VcfParseError, match="position must be an integer"):
        convert_23andme_to_vcf(bad_pos, tmp_path / "out.vcf", {})

    negative_pos = tmp_path / "negative-pos.txt"
    negative_pos.write_text("rs1\t1\t0\tAG\n", encoding="utf-8")
    with pytest.raises(VcfParseError, match="position must be >= 1"):
        convert_23andme_to_vcf(negative_pos, tmp_path / "out.vcf", {})

    raw = tmp_path / "valid.txt"
    raw.write_text("rs1\t1\t100\tAG\n", encoding="utf-8")
    with pytest.raises(VcfParseError, match="reference allele positions"):
        convert_23andme_to_vcf(raw, tmp_path / "out.vcf", {("1", 0): "A"})
    with pytest.raises(VcfParseError, match="reference allele contains"):
        convert_23andme_to_vcf(raw, tmp_path / "out.vcf", {("1", 100): "N"})
    with pytest.raises(VcfParseError, match="sample_id"):
        convert_23andme_to_vcf(raw, tmp_path / "out.vcf", {("1", 100): "A"}, sample_id="bad id")


def test_common_vcf_writer_and_explicit_variant_validation(tmp_path: Path) -> None:
    with pytest.raises(VcfParseError, match="at least one alternate"):
        variant_from_explicit_alleles(chrom="1", pos=1, ref="A", alts=(), source="fixture")
    with pytest.raises(VcfParseError, match="must differ"):
        variant_from_explicit_alleles(chrom="1", pos=1, ref="A", alts=("A",), source="fixture")
    with pytest.raises(VcfParseError, match="VCF ID"):
        variant_from_explicit_alleles(
            chrom="1",
            pos=1,
            ref="A",
            alts=("C",),
            variant_id="bad id",
            source="fixture",
        )

    path = write_vcf_variants(
        (
            variant_from_explicit_alleles(
                chrom="chrM",
                pos=2,
                ref="A",
                alts=("G",),
                variant_id="mt",
                genotype="0/1",
                source="fixture",
            ),
            variant_from_explicit_alleles(
                chrom="GL0001",
                pos=1,
                ref="C",
                alts=("T",),
                variant_id="other",
                genotype="1/1",
                source="fixture",
            ),
        ),
        tmp_path / "sorted.vcf",
    )
    assert _variant_lines(path) == [
        "MT\t2\tmt\tA\tG\t.\tPASS\tSOURCE=fixture\tGT\t0/1",
        "GL0001\t1\tother\tC\tT\t.\tPASS\tSOURCE=fixture\tGT\t1/1",
    ]


def test_myheritage_tsv_and_sequencing_container_validation(tmp_path: Path) -> None:
    empty_header = tmp_path / "empty-header.txt"
    empty_header.write_text("# only comments\n", encoding="utf-8")
    assert parse_myheritage(empty_header) == ()

    tsv = tmp_path / "myheritage.tsv"
    tsv.write_text("rsid chromosome position result\nrs2 chr24 300 CT\n", encoding="utf-8")
    summary = convert_myheritage_to_vcf(tsv, tmp_path / "myheritage.vcf", {("Y", 300): "C"})
    assert _variant_lines(summary.output_path) == [
        "Y\t300\trs2\tC\tT\t.\tPASS\tSOURCE=MyHeritage\tGT\t0/1"
    ]

    bad_top = tmp_path / "bad-top.json"
    bad_top.write_text(json.dumps("not an object"), encoding="utf-8")
    with pytest.raises(VcfParseError, match="top level"):
        parse_sequencing_json(bad_top)

    bad_rows = tmp_path / "bad-rows.json"
    bad_rows.write_text(json.dumps({"variants": ["bad"]}), encoding="utf-8")
    with pytest.raises(VcfParseError, match="variant rows must be objects"):
        parse_sequencing_json(bad_rows)

    bad_container = tmp_path / "bad-container.json"
    bad_container.write_text(json.dumps({"metadata": {}}), encoding="utf-8")
    with pytest.raises(VcfParseError, match="must contain"):
        parse_sequencing_json(bad_container)
