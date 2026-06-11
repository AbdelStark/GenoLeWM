# SPDX-License-Identifier: Apache-2.0
"""Fixture data and renderers for the BRCA2 saturation notebook."""

from __future__ import annotations

import hashlib
import html
import math
from dataclasses import dataclass
from typing import Final

BASES: Final = ("A", "C", "G", "T")
BRCA2_FIXTURE_CHROM: Final = "chr13"
BRCA2_FIXTURE_START_BP: Final = 32_316_461
BRCA2_EXON_FIXTURE: Final = "ATGGATTTATCTGCTCTTCGCGTT"
HEATMAP_SHADES: Final = " .:-=+*#%@"


@dataclass(frozen=True, slots=True)
class SaturationRow:
    """One deterministic fixture row for a possible SNV."""

    chrom: str
    pos: int
    offset: int
    ref: str
    alt: str
    codon_index: int
    variant: str
    sigma_calibrated: float
    fixture_function_score: float

    def to_dict(self) -> dict[str, int | float | str]:
        return {
            "chrom": self.chrom,
            "pos": self.pos,
            "offset": self.offset,
            "ref": self.ref,
            "alt": self.alt,
            "codon_index": self.codon_index,
            "variant": self.variant,
            "sigma_calibrated": self.sigma_calibrated,
            "fixture_function_score": self.fixture_function_score,
        }


def enumerate_fixture_saturation(
    *,
    sequence: str = BRCA2_EXON_FIXTURE,
    chrom: str = BRCA2_FIXTURE_CHROM,
    start_bp: int = BRCA2_FIXTURE_START_BP,
) -> tuple[SaturationRow, ...]:
    """Return all single-nucleotide substitutions for the fixture sequence."""
    _validate_sequence(sequence)
    rows: list[SaturationRow] = []
    length = len(sequence)
    for offset, ref in enumerate(sequence):
        for alt in BASES:
            if alt == ref:
                continue
            pos = start_bp + offset
            sigma = _fixture_sigma(offset=offset, length=length, ref=ref, alt=alt)
            rows.append(
                SaturationRow(
                    chrom=chrom,
                    pos=pos,
                    offset=offset,
                    ref=ref,
                    alt=alt,
                    codon_index=offset // 3,
                    variant=f"{chrom}:{pos}:{ref}>{alt}",
                    sigma_calibrated=sigma,
                    fixture_function_score=_fixture_function_score(
                        offset=offset,
                        ref=ref,
                        alt=alt,
                        sigma_calibrated=sigma,
                    ),
                )
            )
    return tuple(rows)


def summarize_rows(rows: tuple[SaturationRow, ...]) -> dict[str, int | float]:
    """Return compact deterministic summary values for notebook output."""
    _require_rows(rows)
    positions = len({row.offset for row in rows})
    return {
        "positions": positions,
        "snvs": len(rows),
        "mean_sigma_calibrated": round(
            sum(row.sigma_calibrated for row in rows) / len(rows),
            3,
        ),
        "fixture_spearman": round(
            spearman(
                tuple(row.sigma_calibrated for row in rows),
                tuple(row.fixture_function_score for row in rows),
            ),
            3,
        ),
    }


def score_matrix(
    rows: tuple[SaturationRow, ...],
    *,
    sequence: str = BRCA2_EXON_FIXTURE,
) -> dict[str, tuple[float | None, ...]]:
    """Return alt-base rows by sequence offset; reference cells are ``None``."""
    _require_rows(rows)
    values: dict[str, list[float | None]] = {base: [None] * len(sequence) for base in BASES}
    for row in rows:
        values[row.alt][row.offset] = row.sigma_calibrated
    return {base: tuple(cells) for base, cells in values.items()}


def render_text_heatmap(
    rows: tuple[SaturationRow, ...],
    *,
    sequence: str = BRCA2_EXON_FIXTURE,
) -> str:
    """Render a compact ASCII heatmap for committed notebook output."""
    matrix = score_matrix(rows, sequence=sequence)
    lines = [
        "ref " + " ".join(sequence),
        "pos " + " ".join(f"{index % 10}" for index in range(len(sequence))),
    ]
    for base in BASES:
        cells = ["x" if value is None else _shade(value) for value in matrix[base]]
        lines.append(f"{base}   " + " ".join(cells))
    return "\n".join(lines)


def render_html_heatmap(
    rows: tuple[SaturationRow, ...],
    *,
    sequence: str = BRCA2_EXON_FIXTURE,
) -> str:
    """Render a notebook-friendly HTML heatmap without plotting dependencies."""
    matrix = score_matrix(rows, sequence=sequence)
    header = "".join(f"<th>{index}</th>" for index in range(len(sequence)))
    ref_row = "".join(f"<td>{html.escape(base)}</td>" for base in sequence)
    body_rows = [
        "<tr><th>ref</th>" + ref_row + "</tr>",
    ]
    for base in BASES:
        cells: list[str] = []
        for value in matrix[base]:
            if value is None:
                cells.append('<td class="ref">ref</td>')
            else:
                cells.append(f'<td style="background:{_color(value)}">{value:.3f}</td>')
        body_rows.append(f"<tr><th>{base}</th>{''.join(cells)}</tr>")
    return (
        "<style>"
        ".geno-brca2-heatmap{border-collapse:collapse;font-family:monospace}"
        ".geno-brca2-heatmap th,.geno-brca2-heatmap td{"
        "border:1px solid #d0d7de;padding:4px;text-align:center}"
        ".geno-brca2-heatmap .ref{background:#f6f8fa;color:#57606a}"
        "</style>"
        '<table class="geno-brca2-heatmap">'
        f"<thead><tr><th>alt</th>{header}</tr></thead>"
        f"<tbody>{''.join(body_rows)}</tbody>"
        "</table>"
    )


def spearman(left: tuple[float, ...], right: tuple[float, ...]) -> float:
    """Return Spearman rank correlation for two equal-length vectors."""
    if len(left) != len(right) or not left:
        raise ValueError("spearman inputs must be non-empty and equal length")
    return _pearson(_ranks(left), _ranks(right))


def _fixture_sigma(*, offset: int, length: int, ref: str, alt: str) -> float:
    progression = offset / max(length - 1, 1)
    alt_weight = {"A": 0.05, "C": 0.15, "G": 0.25, "T": 0.35}[alt]
    transition_bonus = -0.04 if {ref, alt} in ({"A", "G"}, {"C", "T"}) else 0.07
    local = (_stable_fraction(f"{offset}:{ref}>{alt}") - 0.5) * 0.08
    return round(_clamp(0.18 + 0.36 * progression + alt_weight + transition_bonus + local), 3)


def _fixture_function_score(
    *,
    offset: int,
    ref: str,
    alt: str,
    sigma_calibrated: float,
) -> float:
    local = (_stable_fraction(f"function:{offset}:{ref}>{alt}") - 0.5) * 0.10
    codon_term = 0.05 if offset % 3 == 1 else -0.02
    return round(1.05 - (1.65 * sigma_calibrated) + codon_term + local, 3)


def _stable_fraction(key: str) -> float:
    digest = hashlib.sha256(key.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") / float(2**64 - 1)


def _shade(value: float) -> str:
    index = round(_clamp(value) * (len(HEATMAP_SHADES) - 1))
    return HEATMAP_SHADES[index]


def _color(value: float) -> str:
    value = _clamp(value)
    red = 255
    green_blue = round(255 * (1.0 - value))
    return f"rgb({red},{green_blue},{green_blue})"


def _clamp(value: float) -> float:
    return min(0.98, max(0.02, value))


def _ranks(values: tuple[float, ...]) -> tuple[float, ...]:
    indexed = sorted(enumerate(values), key=lambda item: item[1])
    ranks = [0.0] * len(values)
    cursor = 0
    while cursor < len(indexed):
        end = cursor + 1
        while end < len(indexed) and indexed[end][1] == indexed[cursor][1]:
            end += 1
        average_rank = (cursor + 1 + end) / 2
        for original_index, _ in indexed[cursor:end]:
            ranks[original_index] = average_rank
        cursor = end
    return tuple(ranks)


def _pearson(left: tuple[float, ...], right: tuple[float, ...]) -> float:
    left_mean = sum(left) / len(left)
    right_mean = sum(right) / len(right)
    numerator = sum((x - left_mean) * (y - right_mean) for x, y in zip(left, right, strict=True))
    left_norm = math.sqrt(sum((x - left_mean) ** 2 for x in left))
    right_norm = math.sqrt(sum((y - right_mean) ** 2 for y in right))
    if left_norm == 0 or right_norm == 0:
        raise ValueError("spearman inputs must vary")
    return numerator / (left_norm * right_norm)


def _validate_sequence(sequence: str) -> None:
    if not sequence:
        raise ValueError("sequence must be non-empty")
    invalid = sorted(set(sequence) - set(BASES))
    if invalid:
        raise ValueError(f"sequence contains unsupported bases: {''.join(invalid)}")


def _require_rows(rows: tuple[SaturationRow, ...]) -> None:
    if not rows:
        raise ValueError("rows must be non-empty")
