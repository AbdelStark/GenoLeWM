# SPDX-License-Identifier: Apache-2.0
"""23andMe raw genotype importer."""

from __future__ import annotations

from collections.abc import Iterable
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

__all__ = ["convert_23andme_to_vcf"]

_PROVIDER = "23andMe"


def parse_23andme(path: str | Path) -> tuple[ArrayGenotypeCall, ...]:
    """Parse common 23andMe raw-data rows into normalized genotype calls."""
    calls: list[ArrayGenotypeCall] = []
    for line_no, line in read_text_lines(path):
        fields = line.split()
        if _is_header(fields):
            continue
        if len(fields) != 4:
            raise VcfParseError(
                "23andMe row must contain rsid, chromosome, position, genotype",
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


def convert_23andme_to_vcf(
    input_path: str | Path,
    output_path: str | Path,
    reference_alleles: ReferenceAlleles,
    *,
    sample_id: str = "sample",
) -> VcfConversionSummary:
    """Convert 23andMe raw genotype text to a local VCF file."""
    return convert_array_calls_to_vcf(
        parse_23andme(input_path),
        output_path,
        reference_alleles,
        sample_id=sample_id,
    )


def _is_header(fields: Iterable[str]) -> bool:
    first = next(iter(fields), "").casefold()
    return first in {"rsid", "snp", "snpid"}
