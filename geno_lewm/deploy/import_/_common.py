# SPDX-License-Identifier: Apache-2.0
"""Shared helpers for local-only personal-genome importers."""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

from geno_lewm.errors import VcfParseError

_BASES = frozenset("ACGT")
_MISSING_GENOTYPES = frozenset({"", "-", "--", "0", "00", "N", "NN", "NO_CALL"})


@dataclass(frozen=True, slots=True)
class VcfConversionSummary:
    """Summary returned by a local personal-genome conversion."""

    output_path: Path
    records_written: int
    ref_calls_skipped: int
    no_calls_skipped: int


@dataclass(frozen=True, slots=True)
class ArrayGenotypeCall:
    provider: str
    rsid: str
    chrom: str
    pos: int
    genotype: str
    line_no: int


@dataclass(frozen=True, slots=True)
class VcfVariant:
    chrom: str
    pos: int
    id: str
    ref: str
    alts: tuple[str, ...]
    genotype: str
    source: str


ReferenceAlleles = Mapping[tuple[str, int], str]


def read_text_lines(path: str | Path) -> list[tuple[int, str]]:
    src = Path(path)
    try:
        raw = src.read_text(encoding="utf-8-sig")
    except OSError as exc:
        raise VcfParseError(
            "could not read personal-genome input",
            details={"path": str(src), "error": str(exc)},
        ) from exc
    return [
        (line_no, line.strip())
        for line_no, line in enumerate(raw.splitlines(), start=1)
        if line.strip() and not line.lstrip().startswith("#")
    ]


def parse_pos(value: str, *, provider: str, line_no: int) -> int:
    try:
        pos = int(value)
    except ValueError as exc:
        raise VcfParseError(
            "position must be an integer",
            details={"provider": provider, "line": line_no, "value": value},
        ) from exc
    if pos < 1:
        raise VcfParseError(
            "position must be >= 1",
            details={"provider": provider, "line": line_no, "value": value},
        )
    return pos


def normalize_chrom(value: str) -> str:
    chrom = value.strip()
    if not chrom:
        raise VcfParseError("chromosome must be non-empty")
    lowered = chrom.lower()
    if lowered.startswith("chr"):
        chrom = chrom[3:]
    upper = chrom.upper()
    if upper == "M":
        return "MT"
    if upper == "23":
        return "X"
    if upper == "24":
        return "Y"
    if upper in {"25", "26"}:
        return "MT"
    return upper if upper in {"X", "Y", "XY", "MT"} else chrom


def convert_array_calls_to_vcf(
    calls: Iterable[ArrayGenotypeCall],
    output_path: str | Path,
    reference_alleles: ReferenceAlleles,
    *,
    sample_id: str = "sample",
) -> VcfConversionSummary:
    variants: list[VcfVariant] = []
    ref_skipped = 0
    no_call_skipped = 0
    normalized_reference = _normalize_reference_alleles(reference_alleles)

    for call in calls:
        variant = _array_call_to_variant(call, normalized_reference)
        if variant is None:
            genotype = call.genotype.strip().upper()
            if genotype in _MISSING_GENOTYPES:
                no_call_skipped += 1
            else:
                ref_skipped += 1
            continue
        variants.append(variant)

    path = write_vcf_variants(variants, output_path, sample_id=sample_id)
    return VcfConversionSummary(
        output_path=path,
        records_written=len(variants),
        ref_calls_skipped=ref_skipped,
        no_calls_skipped=no_call_skipped,
    )


def write_vcf_variants(
    variants: Iterable[VcfVariant],
    output_path: str | Path,
    *,
    sample_id: str = "sample",
) -> Path:
    path = Path(output_path)
    _validate_sample_id(sample_id)
    rows = sorted(variants, key=lambda row: (_chrom_sort_key(row.chrom), row.pos, row.id))
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "##fileformat=VCFv4.2",
        "##source=GenoLeWM-personal-genome-import",
        '##INFO=<ID=SOURCE,Number=1,Type=String,Description="Local raw-data provider">',
        '##FORMAT=<ID=GT,Number=1,Type=String,Description="Genotype">',
        f"#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\t{sample_id}",
    ]
    lines.extend(_vcf_line(row) for row in rows)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def variant_from_explicit_alleles(
    *,
    chrom: str,
    pos: int,
    ref: str,
    alts: Sequence[str],
    variant_id: str = ".",
    genotype: str = ".",
    source: str,
) -> VcfVariant:
    normalized_ref = _validate_bases(ref, field="ref", source=source)
    normalized_alts = tuple(_validate_bases(alt, field="alt", source=source) for alt in alts)
    if not normalized_alts:
        raise VcfParseError("variant must include at least one alternate allele")
    if normalized_ref in normalized_alts:
        raise VcfParseError(
            "alternate allele must differ from reference",
            details={"chrom": chrom, "pos": pos, "ref": normalized_ref, "alts": normalized_alts},
        )
    return VcfVariant(
        chrom=normalize_chrom(chrom),
        pos=pos,
        id=_clean_vcf_id(variant_id),
        ref=normalized_ref,
        alts=normalized_alts,
        genotype=genotype,
        source=source,
    )


def _array_call_to_variant(
    call: ArrayGenotypeCall,
    reference_alleles: dict[tuple[str, int], str],
) -> VcfVariant | None:
    chrom = normalize_chrom(call.chrom)
    genotype = call.genotype.strip().upper().replace(" ", "")
    if genotype in _MISSING_GENOTYPES:
        return None

    alleles = _genotype_alleles(genotype, provider=call.provider, line_no=call.line_no)
    ref = reference_alleles.get((chrom, call.pos))
    if ref is None:
        raise VcfParseError(
            "reference allele is required for array raw-data conversion",
            details={
                "provider": call.provider,
                "line": call.line_no,
                "rsid": call.rsid,
                "chrom": chrom,
                "pos": call.pos,
            },
            remediation="provide a local reference allele map keyed by (chrom, pos)",
        )

    alts: list[str] = []
    genotype_indexes: list[str] = []
    allele_index = {ref: "0"}
    for allele in alleles:
        if allele not in _BASES:
            raise VcfParseError(
                "array genotype contains unsupported non-SNV allele",
                details={
                    "provider": call.provider,
                    "line": call.line_no,
                    "rsid": call.rsid,
                    "allele": allele,
                },
                remediation="v1 importers support A/C/G/T SNP calls; decompose indels upstream",
            )
        if allele not in allele_index:
            alts.append(allele)
            allele_index[allele] = str(len(alts))
        genotype_indexes.append(allele_index[allele])

    if not alts:
        return None
    separator = "/" if len(genotype_indexes) == 2 else ""
    return VcfVariant(
        chrom=chrom,
        pos=call.pos,
        id=_clean_vcf_id(call.rsid),
        ref=ref,
        alts=tuple(alts),
        genotype=separator.join(genotype_indexes),
        source=call.provider,
    )


def _normalize_reference_alleles(reference_alleles: ReferenceAlleles) -> dict[tuple[str, int], str]:
    normalized: dict[tuple[str, int], str] = {}
    for (chrom, pos), allele in reference_alleles.items():
        if not isinstance(pos, int) or isinstance(pos, bool) or pos < 1:
            raise VcfParseError("reference allele positions must be positive integers")
        normalized[(normalize_chrom(chrom), pos)] = _validate_bases(
            allele,
            field="reference",
            source="reference_alleles",
        )
    return normalized


def _genotype_alleles(genotype: str, *, provider: str, line_no: int) -> tuple[str, ...]:
    if "/" in genotype or "|" in genotype:
        parts = tuple(part for part in genotype.replace("|", "/").split("/") if part)
    else:
        parts = tuple(genotype)
    if len(parts) not in {1, 2}:
        raise VcfParseError(
            "genotype must contain one or two allele calls",
            details={"provider": provider, "line": line_no, "genotype": genotype},
        )
    return parts


def _validate_bases(value: str, *, field: str, source: str) -> str:
    normalized = value.strip().upper()
    if not normalized:
        raise VcfParseError(f"{field} allele must be non-empty", details={"source": source})
    bad = set(normalized) - _BASES
    if bad:
        raise VcfParseError(
            f"{field} allele contains non-ACGT character(s)",
            details={"source": source, "value": value, "bad_chars": sorted(bad)},
        )
    return normalized


def _clean_vcf_id(value: str) -> str:
    if not value or value == ".":
        return "."
    if any(ch.isspace() or ch in ";=," for ch in value):
        raise VcfParseError("VCF ID must not contain whitespace, semicolon, equals, or comma")
    return value


def _validate_sample_id(sample_id: str) -> None:
    if not sample_id or any(ch.isspace() for ch in sample_id):
        raise VcfParseError("sample_id must be non-empty and contain no whitespace")


def _vcf_line(row: VcfVariant) -> str:
    return "\t".join(
        (
            row.chrom,
            str(row.pos),
            row.id,
            row.ref,
            ",".join(row.alts),
            ".",
            "PASS",
            f"SOURCE={row.source}",
            "GT",
            row.genotype,
        )
    )


def _chrom_sort_key(chrom: str) -> tuple[int, int | str]:
    normalized = normalize_chrom(chrom)
    if normalized.isdecimal():
        return (0, int(normalized))
    special = {"X": 23, "Y": 24, "XY": 25, "MT": 26}
    if normalized in special:
        return (0, special[normalized])
    return (1, normalized)
