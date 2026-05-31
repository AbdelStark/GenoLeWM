# SPDX-License-Identifier: Apache-2.0
"""Sequencing.com-style WGS JSON importer."""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from geno_lewm.deploy.import_._common import (
    VcfConversionSummary,
    VcfVariant,
    variant_from_explicit_alleles,
    write_vcf_variants,
)
from geno_lewm.errors import VcfParseError

__all__ = ["convert_sequencing_json_to_vcf"]

_PROVIDER = "Sequencing.com"


def parse_sequencing_json(path: str | Path) -> tuple[VcfVariant, ...]:
    """Parse a Sequencing.com-style JSON variant array into VCF records."""
    src = Path(path)
    try:
        payload = json.loads(src.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise VcfParseError(
            "Sequencing.com JSON is not valid JSON",
            details={"path": str(src), "line": exc.lineno, "column": exc.colno},
        ) from exc
    except OSError as exc:
        raise VcfParseError(
            "could not read Sequencing.com JSON",
            details={"path": str(src), "error": str(exc)},
        ) from exc

    rows = _variant_rows(payload)
    return tuple(_row_to_variant(row, index=index) for index, row in enumerate(rows, start=1))


def convert_sequencing_json_to_vcf(
    input_path: str | Path,
    output_path: str | Path,
    *,
    sample_id: str = "sample",
) -> VcfConversionSummary:
    """Convert Sequencing.com-style WGS JSON to a local VCF file."""
    variants = parse_sequencing_json(input_path)
    path = write_vcf_variants(variants, output_path, sample_id=sample_id)
    return VcfConversionSummary(
        output_path=path,
        records_written=len(variants),
        ref_calls_skipped=0,
        no_calls_skipped=0,
    )


def _variant_rows(payload: object) -> list[Mapping[str, Any]]:
    if isinstance(payload, list):
        rows = payload
    elif isinstance(payload, dict):
        rows = _first_list(payload, ("variants", "records", "data"))
    else:
        raise VcfParseError("Sequencing.com JSON top level must be an object or array")

    output: list[Mapping[str, Any]] = []
    for index, row in enumerate(rows, start=1):
        if not isinstance(row, dict):
            raise VcfParseError(
                "Sequencing.com JSON variant rows must be objects",
                details={"index": index, "type": type(row).__name__},
            )
        output.append(row)
    return output


def _first_list(payload: Mapping[str, Any], names: tuple[str, ...]) -> list[object]:
    for name in names:
        value = payload.get(name)
        if isinstance(value, list):
            return value
    raise VcfParseError(
        "Sequencing.com JSON must contain a variants, records, or data array",
        details={"keys": sorted(str(key) for key in payload)},
    )


def _row_to_variant(row: Mapping[str, Any], *, index: int) -> VcfVariant:
    chrom = _required_str(row, ("chrom", "chromosome", "chr"), index=index)
    pos = _required_int(row, ("pos", "position", "start"), index=index)
    ref = _required_str(row, ("ref", "reference", "referenceAllele"), index=index)
    alts = _alts(row, index=index)
    genotype = _genotype(row, ref=ref, alts=alts, index=index)
    variant_id = _optional_str(row, ("id", "rsid", "variant_id"), default=".")
    return variant_from_explicit_alleles(
        chrom=chrom,
        pos=pos,
        ref=ref,
        alts=alts,
        variant_id=variant_id,
        genotype=genotype,
        source=_PROVIDER,
    )


def _alts(row: Mapping[str, Any], *, index: int) -> tuple[str, ...]:
    value = None
    for name in ("alt", "alts", "alternate", "alternateAllele", "alternateAlleles"):
        if name in row:
            value = row[name]
            break
    if value is None:
        raise VcfParseError("Sequencing.com JSON variant is missing alternate allele")
    if isinstance(value, str):
        parts = tuple(part for part in value.replace("/", ",").split(",") if part)
    elif isinstance(value, list):
        parts = tuple(str(part) for part in value)
    else:
        raise VcfParseError(
            "alternate allele must be a string or list",
            details={"index": index, "type": type(value).__name__},
        )
    if not parts:
        raise VcfParseError("alternate allele list must be non-empty", details={"index": index})
    return parts


def _genotype(row: Mapping[str, Any], *, ref: str, alts: tuple[str, ...], index: int) -> str:
    value = row.get("genotype", row.get("gt", "."))
    if value is None:
        return "."
    if isinstance(value, str):
        text = value.strip().upper()
        if text in {"", "."}:
            return "."
        if "/" in text or "|" in text:
            sep = "|" if "|" in text else "/"
            parts = text.split(sep)
            if all(part.isdecimal() for part in parts):
                max_index = len(alts)
                if all(0 <= int(part) <= max_index for part in parts):
                    return sep.join(parts)
            allele_to_index = {
                ref.upper(): "0",
                **{alt.upper(): str(i) for i, alt in enumerate(alts, start=1)},
            }
            try:
                return sep.join(allele_to_index[part] for part in parts)
            except KeyError as exc:
                raise VcfParseError(
                    "genotype allele is not present in ref/alt alleles",
                    details={"index": index, "genotype": value},
                ) from exc
        if text.isdecimal() and int(text) <= len(alts):
            return text
        if len(text) in {1, 2}:
            allele_to_index = {
                ref.upper(): "0",
                **{alt.upper(): str(i) for i, alt in enumerate(alts, start=1)},
            }
            try:
                return "/".join(allele_to_index[base] for base in text)
            except KeyError as exc:
                raise VcfParseError(
                    "genotype allele is not present in ref/alt alleles",
                    details={"index": index, "genotype": value},
                ) from exc
    if isinstance(value, list):
        allele_to_index = {
            ref.upper(): "0",
            **{alt.upper(): str(i) for i, alt in enumerate(alts, start=1)},
        }
        try:
            return "/".join(allele_to_index[str(base).upper()] for base in value)
        except KeyError as exc:
            raise VcfParseError(
                "genotype allele is not present in ref/alt alleles",
                details={"index": index, "genotype": value},
            ) from exc
    raise VcfParseError(
        "unsupported Sequencing.com genotype shape",
        details={"index": index, "type": type(value).__name__},
    )


def _required_str(row: Mapping[str, Any], names: tuple[str, ...], *, index: int) -> str:
    value = _optional_str(row, names, default="")
    if not value:
        raise VcfParseError(
            "Sequencing.com JSON variant is missing required string field",
            details={"index": index, "fields": list(names)},
        )
    return value


def _optional_str(row: Mapping[str, Any], names: tuple[str, ...], *, default: str) -> str:
    for name in names:
        value = row.get(name)
        if value is not None:
            return str(value).strip()
    return default


def _required_int(row: Mapping[str, Any], names: tuple[str, ...], *, index: int) -> int:
    for name in names:
        value = row.get(name)
        if value is not None:
            try:
                pos = int(value)
            except (TypeError, ValueError) as exc:
                raise VcfParseError(
                    "Sequencing.com JSON position must be an integer",
                    details={"index": index, "field": name, "value": value},
                ) from exc
            if pos < 1:
                raise VcfParseError(
                    "Sequencing.com JSON position must be >= 1",
                    details={"index": index, "field": name, "value": value},
                )
            return pos
    raise VcfParseError(
        "Sequencing.com JSON variant is missing position",
        details={"index": index, "fields": list(names)},
    )
