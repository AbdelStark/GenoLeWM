# SPDX-License-Identifier: Apache-2.0
"""Generate placed training windows from variant shards and a reference FASTA."""

from __future__ import annotations

import argparse
import gzip
import json
import sys
from bisect import bisect_right
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Final, TextIO

from geno_lewm.action import EditSpec
from geno_lewm.data import iter_gnomad_shard
from geno_lewm.encoder.windowing import extract_window
from geno_lewm.errors import GenoLeWMError, InputError, exit_code_for
from geno_lewm.provenance import sha256_file

GENERATED_BY: Final = "tools.data.placed_windows"


@dataclass(frozen=True, slots=True)
class PlacedWindowReport:
    """Summary emitted by :func:`export_placed_variant_windows`."""

    output: Path
    reference_fasta: Path
    records_read: int
    windows_written: int
    skipped_missing_contig: int
    skipped_ref_mismatch: int
    skipped_sparse: int
    duplicate_windows: int
    window_bp: int
    min_variants_per_window: int

    def to_dict(self) -> dict[str, object]:
        return {
            "generated_by": GENERATED_BY,
            "output": str(self.output),
            "reference_fasta": str(self.reference_fasta),
            "reference_fasta_sha256": sha256_file(self.reference_fasta),
            "records_read": self.records_read,
            "windows_written": self.windows_written,
            "skipped_missing_contig": self.skipped_missing_contig,
            "skipped_ref_mismatch": self.skipped_ref_mismatch,
            "skipped_sparse": self.skipped_sparse,
            "duplicate_windows": self.duplicate_windows,
            "window_bp": self.window_bp,
            "min_variants_per_window": self.min_variants_per_window,
        }


def export_placed_variant_windows(
    variants: Iterable[EditSpec],
    *,
    reference_fasta: Path,
    output: Path,
    source: str = "gnomad_common",
    variant_source: str = "gnomad",
    window_bp: int = 4096,
    max_windows: int | None = None,
    min_variants_per_window: int = 3,
) -> PlacedWindowReport:
    """Write placed JSONL windows centered on real variants."""
    _require_text("source", source)
    _require_text("variant_source", variant_source)
    _require_positive_int("window_bp", window_bp)
    _require_positive_int("min_variants_per_window", min_variants_per_window)
    if max_windows is not None:
        _require_positive_int("max_windows", max_windows)

    reference = _load_reference_fasta(reference_fasta)
    indexed = _index_variants(variants)
    output.parent.mkdir(parents=True, exist_ok=True)

    records_read = 0
    windows_written = 0
    skipped_missing_contig = 0
    skipped_ref_mismatch = 0
    skipped_sparse = 0
    duplicate_windows = 0
    seen: set[tuple[str, int, int]] = set()

    with output.open("w", encoding="utf-8") as handle:
        for chrom in sorted(indexed):
            contig_name = _contig_name(reference, chrom)
            positions, chrom_variants = indexed[chrom]
            if contig_name is None:
                skipped_missing_contig += len(chrom_variants)
                continue
            sequence = reference[contig_name]
            for variant in chrom_variants:
                records_read += 1
                if not _ref_matches(sequence, variant):
                    skipped_ref_mismatch += 1
                    continue
                window = extract_window(
                    sequence,
                    edit_locus=variant.pos - 1,
                    window_bp=window_bp,
                    assume_canonical=True,
                )
                key = (chrom, window.start_bp, window.end_bp)
                if key in seen:
                    duplicate_windows += 1
                    continue
                variant_count = _variant_count(
                    positions,
                    chrom_variants,
                    window.start_bp,
                    window.end_bp,
                )
                if variant_count < min_variants_per_window:
                    skipped_sparse += 1
                    continue
                seen.add(key)
                record_id = f"{variant_source}:{chrom}:{window.start_bp + 1}-{window.end_bp}"
                handle.write(
                    json.dumps(
                        {
                            "record_id": record_id,
                            "source": source,
                            "variant_source": variant_source,
                            "chrom": chrom,
                            "start_bp": window.start_bp,
                            "end_bp": window.end_bp,
                            "sequence": window.sequence,
                            "variant_count": variant_count,
                        },
                        sort_keys=True,
                    )
                    + "\n"
                )
                windows_written += 1
                if max_windows is not None and windows_written >= max_windows:
                    return PlacedWindowReport(
                        output=output,
                        reference_fasta=reference_fasta,
                        records_read=records_read,
                        windows_written=windows_written,
                        skipped_missing_contig=skipped_missing_contig,
                        skipped_ref_mismatch=skipped_ref_mismatch,
                        skipped_sparse=skipped_sparse,
                        duplicate_windows=duplicate_windows,
                        window_bp=window_bp,
                        min_variants_per_window=min_variants_per_window,
                    )

    return PlacedWindowReport(
        output=output,
        reference_fasta=reference_fasta,
        records_read=records_read,
        windows_written=windows_written,
        skipped_missing_contig=skipped_missing_contig,
        skipped_ref_mismatch=skipped_ref_mismatch,
        skipped_sparse=skipped_sparse,
        duplicate_windows=duplicate_windows,
        window_bp=window_bp,
        min_variants_per_window=min_variants_per_window,
    )


def _index_variants(
    variants: Iterable[EditSpec],
) -> dict[str, tuple[tuple[int, ...], tuple[EditSpec, ...]]]:
    indexed: dict[str, list[EditSpec]] = {}
    for variant in variants:
        if not isinstance(variant, EditSpec):
            raise InputError(
                "variants must contain EditSpec values",
                details={"type": type(variant).__name__},
            )
        indexed.setdefault(variant.chrom, []).append(variant)
    output: dict[str, tuple[tuple[int, ...], tuple[EditSpec, ...]]] = {}
    for chrom, values in indexed.items():
        ordered = tuple(sorted(values, key=_variant_pos))
        output[chrom] = (tuple(variant.pos for variant in ordered), ordered)
    return output


def _variant_count(
    positions: Sequence[int],
    variants: Sequence[EditSpec],
    start_bp: int,
    end_bp: int,
) -> int:
    count = 0
    start = bisect_right(positions, start_bp)
    stop = bisect_right(positions, end_bp)
    for variant in variants[start:stop]:
        if variant.pos - 1 + len(variant.ref) <= end_bp:
            count += 1
    return count


def _variant_pos(value: EditSpec) -> int:
    return value.pos


def _ref_matches(sequence: str, variant: EditSpec) -> bool:
    start = variant.pos - 1
    end = start + len(variant.ref)
    return 0 <= start < len(sequence) and sequence[start:end] == variant.ref


def _load_reference_fasta(path: Path) -> dict[str, str]:
    if not path.is_file():
        raise InputError("reference FASTA is missing", details={"path": str(path)})
    reference: dict[str, str] = {}
    current_name: str | None = None
    chunks: list[str] = []
    with _open_text(path) as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line:
                continue
            if line.startswith(">"):
                if current_name is not None:
                    reference[current_name] = _canonical_fasta_sequence(chunks)
                current_name = line[1:].split()[0]
                chunks = []
            elif current_name is not None:
                chunks.append(line)
            else:
                raise InputError(
                    "FASTA sequence appears before a header", details={"path": str(path)}
                )
    if current_name is not None:
        reference[current_name] = _canonical_fasta_sequence(chunks)
    if not reference:
        raise InputError("reference FASTA contained no contigs", details={"path": str(path)})
    return reference


def _canonical_fasta_sequence(chunks: Sequence[str]) -> str:
    sequence = "".join(chunks).upper()
    bad = sorted(set(sequence) - set("ACGTN"))
    if bad:
        raise InputError("reference FASTA contains unsupported bases", details={"bad_chars": bad})
    return sequence


def _open_text(path: Path) -> TextIO:
    if path.name.endswith(".gz"):
        return gzip.open(path, "rt", encoding="utf-8")
    return path.open("r", encoding="utf-8")


def _contig_name(reference: dict[str, str], chrom: str) -> str | None:
    candidates = (chrom, chrom.removeprefix("chr"), f"chr{chrom.removeprefix('chr')}")
    for candidate in candidates:
        if candidate in reference:
            return candidate
    return None


def _require_text(name: str, value: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise InputError(f"{name} must be a non-empty string")


def _require_positive_int(name: str, value: int) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise InputError(f"{name} must be a positive integer", details={name: value})


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gnomad-shard", type=Path, required=True)
    parser.add_argument("--reference-fasta", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--source", default="gnomad_common")
    parser.add_argument("--variant-source", default="gnomad")
    parser.add_argument("--window-bp", type=int, default=4096)
    parser.add_argument("--max-windows", type=int)
    parser.add_argument("--min-variants-per-window", type=int, default=3)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    try:
        variants = (
            EditSpec(row.chrom, row.pos, row.ref, row.alt)
            for row in iter_gnomad_shard(args.gnomad_shard)
        )
        report = export_placed_variant_windows(
            variants,
            reference_fasta=args.reference_fasta,
            output=args.output,
            source=args.source,
            variant_source=args.variant_source,
            window_bp=args.window_bp,
            max_windows=args.max_windows,
            min_variants_per_window=args.min_variants_per_window,
        )
    except GenoLeWMError as exc:
        sys.stderr.write(f"error: {exc}\n")
        return exit_code_for(exc)
    sys.stdout.write(json.dumps(report.to_dict(), indent=2, sort_keys=True) + "\n")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
