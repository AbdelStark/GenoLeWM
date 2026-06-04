"""Unit tests for shared local VCF parsing helpers."""

from __future__ import annotations

import gzip
from pathlib import Path

import pytest

from geno_lewm.data._vcf import (
    info_value_for_alt,
    is_supported_allele,
    iter_vcf_rows,
    normalize_chrom,
    parse_float,
    parse_info,
    parse_int,
)
from geno_lewm.errors import VcfParseError


def test_iter_vcf_rows_parses_gzip_flags_and_skips_empty_alt(tmp_path: Path) -> None:
    path = tmp_path / "variants.vcf.gz"
    with gzip.open(path, "wt", encoding="utf-8") as handle:
        handle.write(
            "\n".join(
                [
                    "##fileformat=VCFv4.2",
                    "#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO",
                    "chrM\t10\trs1\ta\tc,g\t.\tPASS\tAF=0.1,0.2;SOMATIC",
                    "1\t11\trs2\tA\t.\t.\tPASS\tAF=0.3",
                    "",
                ]
            )
            + "\n"
        )

    rows = list(iter_vcf_rows(path))

    assert len(rows) == 1
    row = rows[0]
    assert (row.chrom, row.pos, row.ref, row.alts, row.filter) == (
        "MT",
        10,
        "A",
        ("C", "G"),
        "PASS",
    )
    assert row.info == {"AF": "0.1,0.2", "SOMATIC": True}
    assert info_value_for_alt(row.info, ("AF",), 1) == "0.2"


@pytest.mark.parametrize(
    ("row", "message"),
    [
        ("1\t10\trs1\tA\tC\t.\tPASS", "at least 8 columns"),
        ("1\tnot-int\trs1\tA\tC\t.\tPASS\tAF=0.1", "POS must be an integer"),
        ("1\t0\trs1\tA\tC\t.\tPASS\tAF=0.1", "POS must be >= 1"),
    ],
)
def test_iter_vcf_rows_rejects_malformed_rows(
    tmp_path: Path,
    row: str,
    message: str,
) -> None:
    path = tmp_path / "bad.vcf"
    path.write_text(
        "#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\n" + row + "\n",
        encoding="utf-8",
    )

    with pytest.raises(VcfParseError, match=message):
        list(iter_vcf_rows(path))


def test_iter_vcf_rows_reports_missing_input(tmp_path: Path) -> None:
    with pytest.raises(VcfParseError, match="could not read VCF input"):
        list(iter_vcf_rows(tmp_path / "missing.vcf"))


def test_scalar_and_chrom_helpers_cover_missing_and_malformed_values() -> None:
    assert parse_info("") == {}
    assert parse_info(".") == {}
    assert parse_info("A=1;;FLAG;B=2") == {"A": "1", "FLAG": True, "B": "2"}
    assert info_value_for_alt({"FLAG": True, "AF": "0.1"}, ("FLAG", "AF"), 2) == "0.1"
    assert info_value_for_alt({"AF": "0.1,0.2"}, ("AF",), 4) is None

    assert parse_float(None) is None
    assert parse_float(".") is None
    assert parse_float("nan") is None
    assert parse_float("not-a-float") is None
    assert parse_float("0.25") == 0.25

    assert parse_int(None) is None
    assert parse_int(".") is None
    assert parse_int("42,43") == 42
    assert parse_int("42|43") == 42
    assert parse_int("not-an-int") is None

    assert is_supported_allele("ACGT", max_len=4)
    assert not is_supported_allele("", max_len=4)
    assert not is_supported_allele("ACGT", max_len=3)
    assert not is_supported_allele("ACGN", max_len=4)

    assert normalize_chrom(" chrX ") == "X"
    assert normalize_chrom("chrM") == "MT"
    assert normalize_chrom("GL000207.1") == "GL000207.1"
    with pytest.raises(VcfParseError, match="chromosome must be non-empty"):
        normalize_chrom("  ")
