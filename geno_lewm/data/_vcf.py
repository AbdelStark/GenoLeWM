# SPDX-License-Identifier: Apache-2.0
"""Small VCF helpers shared by local dataset-prep commands."""

from __future__ import annotations

import gzip
import math
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import IO

from geno_lewm.errors import VcfParseError

_ACGT = frozenset("ACGT")


@dataclass(frozen=True, slots=True)
class VcfRow:
    chrom: str
    pos: int
    variant_id: str
    ref: str
    alts: tuple[str, ...]
    filter: str
    info: dict[str, str | bool]
    line_no: int


def iter_vcf_rows(path: str | Path) -> Iterator[VcfRow]:
    """Yield parsed data rows from a local VCF or VCF.gz file."""
    src = Path(path)
    with _open_text(src) as handle:
        for line_no, raw in enumerate(handle, start=1):
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            fields = line.split("\t")
            if len(fields) < 8:
                raise VcfParseError(
                    "VCF row must contain at least 8 columns",
                    details={"path": str(src), "line": line_no, "columns": len(fields)},
                )
            chrom, pos_text, variant_id, ref, alt_field, _qual, filt, info_field = fields[:8]
            try:
                pos = int(pos_text)
            except ValueError as exc:
                raise VcfParseError(
                    "VCF POS must be an integer",
                    details={"path": str(src), "line": line_no, "pos": pos_text},
                ) from exc
            if pos < 1:
                raise VcfParseError(
                    "VCF POS must be >= 1",
                    details={"path": str(src), "line": line_no, "pos": pos},
                )
            alts = tuple(alt for alt in alt_field.upper().split(",") if alt and alt != ".")
            if not alts:
                continue
            yield VcfRow(
                chrom=normalize_chrom(chrom),
                pos=pos,
                variant_id=variant_id,
                ref=ref.upper(),
                alts=alts,
                filter=filt,
                info=parse_info(info_field),
                line_no=line_no,
            )


def parse_info(raw: str) -> dict[str, str | bool]:
    """Parse a VCF INFO field into a mapping."""
    if raw in {"", "."}:
        return {}
    out: dict[str, str | bool] = {}
    for item in raw.split(";"):
        if not item:
            continue
        if "=" not in item:
            out[item] = True
            continue
        key, value = item.split("=", maxsplit=1)
        out[key] = value
    return out


def info_value_for_alt(
    info: dict[str, str | bool],
    keys: tuple[str, ...],
    alt_index: int,
) -> str | None:
    """Return a scalar INFO value aligned to ``alt_index`` when possible."""
    for key in keys:
        raw = info.get(key)
        if raw is None or isinstance(raw, bool):
            continue
        values = raw.split(",")
        if len(values) == 1:
            return values[0]
        if alt_index < len(values):
            return values[alt_index]
    return None


def parse_float(value: str | None) -> float | None:
    """Return a finite float or ``None`` for missing/malformed VCF values."""
    if value is None or value in {"", "."}:
        return None
    try:
        parsed = float(value)
    except ValueError:
        return None
    if not math.isfinite(parsed):
        return None
    return parsed


def parse_int(value: str | None) -> int | None:
    """Return an integer parsed from common VCF scalar/list values."""
    if value is None or value in {"", "."}:
        return None
    first = value.split(",", maxsplit=1)[0].split("|", maxsplit=1)[0]
    try:
        return int(first)
    except ValueError:
        return None


def is_supported_allele(value: str, *, max_len: int) -> bool:
    """Return whether an allele is ACGT-only and within the v1 length cap."""
    return bool(value) and len(value) <= max_len and set(value) <= _ACGT


def normalize_chrom(value: str) -> str:
    """Normalize common chromosome spellings without changing contig names."""
    chrom = value.strip()
    if not chrom:
        raise VcfParseError("chromosome must be non-empty")
    lowered = chrom.lower()
    if lowered.startswith("chr"):
        chrom = chrom[3:]
    upper = chrom.upper()
    if upper == "M":
        return "MT"
    return upper if upper in {"X", "Y", "XY", "MT"} else chrom


@contextmanager
def _open_text(path: Path) -> Iterator[IO[str]]:
    try:
        if path.suffix.lower() in {".bgz", ".gz"}:
            with gzip.open(path, "rt", encoding="utf-8") as handle:
                yield handle
        else:
            with path.open("r", encoding="utf-8") as handle:
                yield handle
    except (OSError, UnicodeError) as exc:
        raise VcfParseError(
            "could not read VCF input",
            details={"path": str(path), "error": str(exc)},
        ) from exc
