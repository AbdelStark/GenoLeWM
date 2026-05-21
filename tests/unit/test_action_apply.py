"""Unit tests for ``geno_lewm.action.apply``."""

from __future__ import annotations

import random
from collections.abc import Sequence

import pytest

from geno_lewm.action import EditType, RelEdit, apply_edit, apply_edits
from geno_lewm.errors import (
    OutOfWindowError,
    OverlappingEditsError,
    WindowMismatchError,
)


def _snv(rel_pos: int, ref: str, alt: str) -> RelEdit:
    return RelEdit(rel_pos=rel_pos, edit_type=EditType.SNV, ref_bases=ref, alt_bases=alt)


def _ins(rel_pos: int, ref: str, alt: str) -> RelEdit:
    return RelEdit(rel_pos=rel_pos, edit_type=EditType.INS, ref_bases=ref, alt_bases=alt)


def _del(rel_pos: int, ref: str, alt: str) -> RelEdit:
    return RelEdit(rel_pos=rel_pos, edit_type=EditType.DEL, ref_bases=ref, alt_bases=alt)


# ---------------------------------------------------------------------------
# Single-edit happy paths.


def test_apply_snv() -> None:
    w = "ACGTACGT"
    out = apply_edit(w, _snv(2, "G", "C"))
    assert out == "ACCTACGT"


def test_apply_insertion() -> None:
    out = apply_edit("ACGT", _ins(1, "C", "CAA"))
    assert out == "ACAAGT"


def test_apply_deletion() -> None:
    out = apply_edit("ACGTACGT", _del(2, "GTAC", "G"))
    assert out == "ACGGT"


def test_apply_mnv() -> None:
    out = apply_edit(
        "ACGT", RelEdit(rel_pos=1, edit_type=EditType.MNV, ref_bases="CG", alt_bases="TA")
    )
    assert out == "ATAT"


# ---------------------------------------------------------------------------
# Validation.


def test_window_mismatch_raises() -> None:
    with pytest.raises(WindowMismatchError):
        apply_edit("ACGT", _snv(1, "G", "A"))  # window[1] is "C", not "G"


def test_out_of_window_raises() -> None:
    with pytest.raises(OutOfWindowError):
        apply_edit("ACGT", _snv(10, "A", "T"))


def test_partial_overlap_with_end_raises() -> None:
    # rel_pos=3, ref_len=4 → end=7 > window_len=4.
    with pytest.raises(OutOfWindowError):
        apply_edit("ACGT", _del(3, "TACG", "T"))


# ---------------------------------------------------------------------------
# preserve_length.


def test_preserve_length_pads_after_deletion_at_left() -> None:
    # Deletion in left half → trim/pad on the right.
    w = "ACGTACGT"  # len 8
    out = apply_edit(w, _del(0, "AC", "A"), preserve_length=True)
    # Edited (before pad): "AGTACGT" (len 7). Pad right with one N.
    assert out == "AGTACGTN"


def test_preserve_length_trims_right_after_insertion_at_left() -> None:
    w = "ACGTACGT"
    out = apply_edit(w, _ins(0, "A", "AGG"), preserve_length=True)
    # Edited: "AGGCGTACGT" (len 10). Trim right back to 8.
    assert out == "AGGCGTAC"


def test_preserve_length_trims_left_after_insertion_at_right() -> None:
    w = "ACGTACGT"  # len 8
    out = apply_edit(w, _ins(7, "T", "TGG"), preserve_length=True)
    # Edited: "ACGTACGTGG" (len 10). Edit on the right half → trim left.
    assert out == "GTACGTGG"


def test_preserve_length_no_change_for_snv() -> None:
    w = "ACGTACGT"
    out = apply_edit(w, _snv(3, "T", "G"), preserve_length=True)
    assert out == "ACGGACGT"
    assert len(out) == len(w)


# ---------------------------------------------------------------------------
# Multi-edit: right-to-left ordering.


def test_apply_edits_right_to_left_preserves_positions() -> None:
    w = "ACGTACGT"
    edits = [
        _snv(0, "A", "T"),
        _ins(4, "A", "AGG"),
        _snv(7, "T", "C"),
    ]
    out = apply_edits(w, edits)
    # Manual right-to-left:
    #   start  "ACGTACGT"
    #   snv7   "ACGTACGC"
    #   ins4   "ACGTAGGCGC"  (insert "GG" after "A")
    #   snv0   "TCGTAGGCGC"
    assert out == "TCGTAGGCGC"


def test_apply_edits_empty_returns_window() -> None:
    assert apply_edits("ACGT", []) == "ACGT"


def test_apply_edits_order_invariant_after_sort() -> None:
    w = "ACGTACGTACGT"
    edits = [
        _snv(0, "A", "T"),
        _snv(5, "C", "G"),
        _snv(10, "G", "A"),
    ]
    out1 = apply_edits(w, edits)
    out2 = apply_edits(w, list(reversed(edits)))
    rng = random.Random(0)
    shuffled = edits.copy()
    rng.shuffle(shuffled)
    out3 = apply_edits(w, shuffled)
    assert out1 == out2 == out3


def test_apply_edits_overlapping_raises() -> None:
    edits = [
        _del(2, "GTA", "G"),  # covers 2..4
        _snv(3, "T", "C"),  # inside 2..4 → overlaps
    ]
    with pytest.raises(OverlappingEditsError):
        apply_edits("ACGTACGT", edits)


def test_apply_edits_adjacent_is_not_overlap() -> None:
    # Edit A covers [2, 4); edit B starts at 4. No overlap.
    edits = [
        _del(2, "GT", "G"),  # covers 2..3
        _snv(4, "A", "C"),
    ]
    out = apply_edits("ACGTACGT", edits)
    # right-to-left:
    #   "ACGTACGT" → snv4 → "ACGTCCGT"
    #   "ACGTCCGT" → del2 → "ACGCCGT"
    assert out == "ACGCCGT"


def test_apply_edits_preserve_length_uses_leftmost_locus() -> None:
    w = "ACGTACGT"  # len 8
    edits = [
        _ins(0, "A", "AGG"),  # left → trim right
        _snv(7, "T", "C"),
    ]
    out = apply_edits(w, edits, preserve_length=True)
    # right-to-left:
    #   snv7  "ACGTACGC"
    #   ins0  "AGGCGTACGC" (len 10)
    # Edit cluster left-most rel_pos = 0 → trim from right back to 8.
    assert out == "AGGCGTAC"
    assert len(out) == len(w)


# ---------------------------------------------------------------------------
# Property tests.


def _rand_bases(rng: random.Random, n: int) -> str:
    return "".join(rng.choice("ACGT") for _ in range(n))


def _rand_window(rng: random.Random, n: int = 64) -> str:
    return _rand_bases(rng, n)


def _rand_compatible_edit(rng: random.Random, window: str) -> RelEdit:
    """Build a valid RelEdit anchored to the supplied window."""
    pos = rng.randint(0, len(window) - 2)
    ref_len = rng.randint(1, min(8, len(window) - pos))
    ref = window[pos : pos + ref_len]
    # Build an alt that is different from ref.
    while True:
        alt_len = rng.randint(1, 8)
        alt = _rand_bases(rng, alt_len)
        if alt != ref:
            break
    et = EditType.SNV
    if ref_len == 1 and alt_len == 1:
        et = EditType.SNV
    elif ref_len == 1:
        et = EditType.INS
    elif alt_len == 1:
        et = EditType.DEL
    elif ref_len == alt_len:
        et = EditType.MNV
    else:
        et = EditType.INDEL
    return RelEdit(rel_pos=pos, edit_type=et, ref_bases=ref, alt_bases=alt)


def test_property_apply_edit_length_formula() -> None:
    """apply_edit length equals len(window) - len(ref) + len(alt)."""
    rng = random.Random(0xFACED)
    for _ in range(500):
        w = _rand_window(rng, n=rng.randint(8, 200))
        e = _rand_compatible_edit(rng, w)
        out = apply_edit(w, e)
        assert len(out) == len(w) - len(e.ref_bases) + len(e.alt_bases)


def _disjoint_edits(rng: random.Random, window: str, n: int) -> Sequence[RelEdit]:
    # Pick `n` non-overlapping intervals inside the window.
    edits: list[RelEdit] = []
    used: list[tuple[int, int]] = []
    attempts = 0
    while len(edits) < n and attempts < 400:
        attempts += 1
        e = _rand_compatible_edit(rng, window)
        s, end = e.rel_pos, e.rel_pos + len(e.ref_bases)
        if any(not (end <= a or s >= b) for a, b in used):
            continue
        edits.append(e)
        used.append((s, end))
    return edits


def test_property_apply_edits_order_invariant_after_sort() -> None:
    rng = random.Random(0xDADA)
    for _ in range(200):
        w = _rand_window(rng, n=rng.randint(40, 160))
        n = rng.randint(2, 5)
        edits = _disjoint_edits(rng, w, n=n)
        if len(edits) < 2:
            continue
        base = apply_edits(w, edits)
        shuffled = list(edits)
        rng.shuffle(shuffled)
        assert apply_edits(w, shuffled) == base
