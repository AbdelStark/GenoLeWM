# SPDX-License-Identifier: Apache-2.0
"""Build a held-out ClinVar evaluation set: labels JSONL + a matching VCF.

``geno-lewm-eval`` consumes a label JSONL (``chrom``/``pos``/``ref``/``alt`` +
``clinical_significance``) and requires every labelled variant to be scored,
with no duplicate variant keys. This tool reads a ClinVar VCF, keeps only the
labelled classes (P/LP/B/LB; VUS/OTHER dropped), de-duplicates by variant key,
drops keys whose duplicate rows disagree on the binary label, and writes:

* ``--labels-out``: one JSON object per kept variant for ``geno-lewm-eval``.
* ``--vcf-out``: the same variants as a minimal VCF for ``geno-lewm-score``.

Emitting both from the same rows guarantees the score keys exactly cover the
label keys (no missing/duplicate keys at eval time). Restrict to one chromosome
with ``--chrom`` to keep proof-scale scoring cheap. Run as::

    python -m tools.data.clinvar_eval_set \
        --input-vcf clinvar.vcf.gz --chrom 21 \
        --labels-out eval/clinvar-chr21.labels.jsonl \
        --vcf-out eval/clinvar-chr21.vcf
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Any

from geno_lewm.data import (
    CLINVAR_LABELLED_CLASSES,
    ClinvarVariant,
    iter_clinvar_vcf_variants,
)
from geno_lewm.errors import GenoLeWMError, InputError, exit_code_for

GENERATED_BY = "tools.data.clinvar_eval_set"

_POSITIVE_CLASSES = frozenset({"P", "LP"})
_NEGATIVE_CLASSES = frozenset({"B", "LB"})


def _binary_label(significance: str) -> bool | None:
    if significance in _POSITIVE_CLASSES:
        return True
    if significance in _NEGATIVE_CLASSES:
        return False
    return None


def build_clinvar_eval_set(
    *,
    input_vcf: Path,
    labels_out: Path,
    vcf_out: Path,
    chrom: str | None = None,
    max_allele_len: int = 16,
    fasta: Path | None = None,
    max_variants: int | None = None,
    subset_seed: int = 0,
) -> dict[str, Any]:
    """Write the labels JSONL + matching VCF and return a summary.

    When ``fasta`` is given, drop variants that are not scoreable against that
    reference (off-contig, REF past the contig, or REF disagreeing with the
    FASTA) using the scorer's own extraction logic. A single unscoreable variant
    would otherwise abort the whole VCF at scoring time, so filtering both the
    labels and the VCF here keeps the score keys covering the label keys while
    guaranteeing the downstream scoring pass cannot abort.
    """
    labels_out = Path(labels_out)
    vcf_out = Path(vcf_out)
    by_key: dict[tuple[str, int, str, str], dict[str, Any]] = {}
    seen_labelled = 0
    for variant in iter_clinvar_vcf_variants(input_vcf, max_allele_len=max_allele_len):
        if chrom is not None and variant.chrom != chrom:
            continue
        if variant.clinical_significance not in CLINVAR_LABELLED_CLASSES:
            continue
        binary = _binary_label(variant.clinical_significance)
        if binary is None:  # defensive; labelled classes always map to a binary
            continue
        seen_labelled += 1
        key = (variant.chrom, variant.pos, variant.ref, variant.alt)
        entry = by_key.setdefault(key, {"variant": variant, "binaries": set()})
        entry["binaries"].add(binary)

    kept: list[ClinvarVariant] = []
    conflicting = 0
    for entry in by_key.values():
        binaries: set[bool] = entry["binaries"]
        if len(binaries) != 1:
            conflicting += 1
            continue
        kept.append(entry["variant"])
    kept.sort(key=lambda v: (v.chrom, v.pos, v.ref, v.alt))

    dropped_unscoreable = 0
    if fasta is not None:
        kept, dropped_unscoreable = _filter_scoreable(kept, Path(fasta))

    subsampled_from = 0
    if max_variants is not None and len(kept) > max_variants:
        subsampled_from = len(kept)
        kept = _stratified_subsample(kept, max_variants, subset_seed)

    if not kept:
        raise InputError(
            "no labelled ClinVar variants matched the eval-set filters",
            details={
                "chrom": chrom,
                "labelled_rows_seen": seen_labelled,
                "dropped_unscoreable": dropped_unscoreable,
            },
            remediation="check the ClinVar VCF, --chrom filter, and that the FASTA build matches",
        )

    positives = sum(1 for v in kept if _binary_label(v.clinical_significance))
    negatives = len(kept) - positives

    labels_out.parent.mkdir(parents=True, exist_ok=True)
    with labels_out.open("w", encoding="utf-8") as handle:
        for variant in kept:
            handle.write(
                json.dumps(
                    {
                        "chrom": variant.chrom,
                        "pos": variant.pos,
                        "ref": variant.ref,
                        "alt": variant.alt,
                        "clinical_significance": variant.clinical_significance,
                    },
                    sort_keys=True,
                )
                + "\n"
            )

    vcf_out.parent.mkdir(parents=True, exist_ok=True)
    with vcf_out.open("w", encoding="utf-8") as handle:
        handle.write("##fileformat=VCFv4.2\n")
        handle.write(f"##source={GENERATED_BY}\n")
        handle.write("#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\n")
        for variant in kept:
            handle.write(
                f"{variant.chrom}\t{variant.pos}\t.\t{variant.ref}\t{variant.alt}\t.\t.\t.\n"
            )

    return {
        "generated_by": GENERATED_BY,
        "labels_out": str(labels_out),
        "vcf_out": str(vcf_out),
        "chrom": chrom,
        "variants": len(kept),
        "positives": positives,
        "negatives": negatives,
        "dropped_conflicting": conflicting,
        "dropped_unscoreable": dropped_unscoreable,
        "subsampled_from": subsampled_from,
        "labelled_rows_seen": seen_labelled,
    }


def _stratified_subsample(
    variants: list[ClinvarVariant], max_variants: int, seed: int
) -> list[ClinvarVariant]:
    """Deterministically cap the eval set while keeping both label classes.

    Positives (P/LP) and negatives (B/LB) are sampled proportionally, with at
    least one of each retained so the downstream AUROC always has both classes.
    """
    positives = [v for v in variants if _binary_label(v.clinical_significance)]
    negatives = [v for v in variants if not _binary_label(v.clinical_significance)]
    total = len(variants)
    n_pos = max(1, round(max_variants * len(positives) / total)) if positives else 0
    n_neg = max(1, max_variants - n_pos) if negatives else 0
    n_pos = min(n_pos, len(positives))
    n_neg = min(n_neg, len(negatives))
    rng = random.Random(seed)
    rng.shuffle(positives)
    rng.shuffle(negatives)
    chosen = positives[:n_pos] + negatives[:n_neg]
    chosen.sort(key=lambda v: (v.chrom, v.pos, v.ref, v.alt))
    return chosen


def _filter_scoreable(
    variants: list[ClinvarVariant], fasta: Path
) -> tuple[list[ClinvarVariant], int]:
    """Keep only variants the scorer can extract a window for against ``fasta``.

    Uses the scorer's own FASTA loader + window extraction so the kept set is
    exactly what ``geno_lewm.surprise.score.score_vcf`` will accept (no aborts).
    """
    # Imported here (not at module top) and reused verbatim from the scorer so the
    # scoreability check stays identical to the scoring path. score.py is
    # torch-free at import, so this keeps the tool importable without torch.
    from geno_lewm.action import EditSpec
    from geno_lewm.encoder.windowing import DEFAULT_WINDOW_BP
    from geno_lewm.errors import VcfParseError
    from geno_lewm.surprise.score import _extract_reference_window, _load_reference_fasta

    reference = _load_reference_fasta(fasta)
    kept: list[ClinvarVariant] = []
    dropped = 0
    for variant in variants:
        spec = EditSpec(chrom=variant.chrom, pos=variant.pos, ref=variant.ref, alt=variant.alt)
        try:
            _extract_reference_window(spec, reference, window_bp=DEFAULT_WINDOW_BP)
        except VcfParseError:
            dropped += 1
            continue
        kept.append(variant)
    return kept, dropped


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-vcf", type=Path, required=True)
    parser.add_argument("--labels-out", type=Path, required=True)
    parser.add_argument("--vcf-out", type=Path, required=True)
    parser.add_argument("--chrom", default=None, help="Restrict to one contig (e.g. 21).")
    parser.add_argument("--max-allele-len", type=int, default=16)
    parser.add_argument(
        "--fasta",
        type=Path,
        default=None,
        help="Optional reference FASTA; drop variants not scoreable against it (no scoring abort).",
    )
    parser.add_argument(
        "--max-variants",
        type=int,
        default=None,
        help="Cap the eval set size (class-stratified, deterministic) for proof-scale scoring.",
    )
    parser.add_argument("--subset-seed", type=int, default=0)
    args = parser.parse_args(argv)
    try:
        summary = build_clinvar_eval_set(
            input_vcf=args.input_vcf,
            labels_out=args.labels_out,
            vcf_out=args.vcf_out,
            chrom=args.chrom,
            max_allele_len=args.max_allele_len,
            fasta=args.fasta,
            max_variants=args.max_variants,
            subset_seed=args.subset_seed,
        )
    except GenoLeWMError as exc:
        sys.stderr.write(f"error: {exc}\n")
        if exc.details:
            sys.stderr.write(json.dumps(exc.details, sort_keys=True) + "\n")
        return exit_code_for(exc)
    sys.stdout.write(json.dumps(summary, sort_keys=True) + "\n")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
