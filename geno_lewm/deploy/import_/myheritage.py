# SPDX-License-Identifier: Apache-2.0
"""MyHeritage DNA raw genotype importer."""

from __future__ import annotations

import csv
from pathlib import Path

from geno_lewm.deploy.import_._common import (
    ArrayGenotypeCall,
    ReferenceAlleles,
    VcfConversionSummary,
    convert_array_calls_to_vcf,
    parse_pos,
    read_text_lines,
)
from geno_lewm.errors import VcfParseError

__all__ = ["convert_myheritage_to_vcf"]

_PROVIDER = "MyHeritage"


def parse_myheritage(path: str | Path) -> tuple[ArrayGenotypeCall, ...]:
    """Parse common MyHeritage CSV/TSV rows into normalized genotype calls."""
    calls: list[ArrayGenotypeCall] = []
    for line_no, line in read_text_lines(path):
        fields = _split_row(line)
        if _is_header(fields):
            continue
        if len(fields) != 4:
            raise VcfParseError(
                "MyHeritage row must contain rsid, chromosome, position, result",
                details={"line": line_no, "field_count": len(fields)},
            )
        rsid, chrom, pos, genotype = fields
        calls.append(
            ArrayGenotypeCall(
                provider=_PROVIDER,
                rsid=rsid,
                chrom=chrom,
                pos=parse_pos(pos, provider=_PROVIDER, line_no=line_no),
                genotype=genotype,
                line_no=line_no,
            )
        )
    return tuple(calls)


def convert_myheritage_to_vcf(
    input_path: str | Path,
    output_path: str | Path,
    reference_alleles: ReferenceAlleles,
    *,
    sample_id: str = "sample",
) -> VcfConversionSummary:
    """Convert MyHeritage raw genotype CSV/TSV to a local VCF file."""
    return convert_array_calls_to_vcf(
        parse_myheritage(input_path),
        output_path,
        reference_alleles,
        sample_id=sample_id,
    )


def _split_row(line: str) -> list[str]:
    if "," in line:
        return [field.strip() for field in next(csv.reader([line]))]
    return line.split()


def _is_header(fields: list[str]) -> bool:
    if not fields:
        return False
    return fields[0].casefold() in {"rsid", "snp", "snpid"}
