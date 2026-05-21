"""Unit tests for ``geno_lewm.action.spec``."""

from __future__ import annotations

import pytest

from geno_lewm.action import V1_MAX_LEN, EditSpec, EditType, RelEdit
from geno_lewm.errors import InvalidEditError, OutOfWindowError, UnsupportedEditError

# ---------------------------------------------------------------------------
# Derived edit_type — exhaustive over the six categories.


@pytest.mark.parametrize(
    ("ref", "alt", "expected"),
    [
        ("A", "T", EditType.SNV),
        ("C", "G", EditType.SNV),
        ("A", "AC", EditType.INS),
        ("A", "ACGT", EditType.INS),
        ("AC", "A", EditType.DEL),
        ("ACGT", "A", EditType.DEL),
        ("AT", "TA", EditType.MNV),
        ("ACGT", "TGCA", EditType.MNV),
        ("AC", "ACG", EditType.INDEL),
        ("ACG", "AC", EditType.INDEL),  # symmetric: ref longer
        ("ACGT", "ACG", EditType.INDEL),
    ],
)
def test_edit_type_derived(ref: str, alt: str, expected: EditType) -> None:
    e = EditSpec(chrom="chr1", pos=100, ref=ref, alt=alt)
    assert e.edit_type is expected


# ---------------------------------------------------------------------------
# Validation — InvalidEditError family.


def test_chrom_required() -> None:
    with pytest.raises(InvalidEditError):
        EditSpec(chrom="", pos=1, ref="A", alt="T")


def test_pos_must_be_positive() -> None:
    with pytest.raises(InvalidEditError):
        EditSpec(chrom="chr1", pos=0, ref="A", alt="T")
    with pytest.raises(InvalidEditError):
        EditSpec(chrom="chr1", pos=-3, ref="A", alt="T")


def test_pos_must_be_int() -> None:
    with pytest.raises(InvalidEditError):
        EditSpec(chrom="chr1", pos=1.5, ref="A", alt="T")  # type: ignore[arg-type]
    # bool is an int subclass — explicitly rejected to catch accidental flags.
    with pytest.raises(InvalidEditError):
        EditSpec(chrom="chr1", pos=True, ref="A", alt="T")  # type: ignore[arg-type]


def test_ref_alt_must_be_non_empty() -> None:
    with pytest.raises(InvalidEditError):
        EditSpec(chrom="chr1", pos=1, ref="", alt="T")
    with pytest.raises(InvalidEditError):
        EditSpec(chrom="chr1", pos=1, ref="A", alt="")


@pytest.mark.parametrize("base_chr", ["a", "n", "N", "U", "K", "R", " "])
def test_ref_alt_rejects_non_uppercase_acgt(base_chr: str) -> None:
    with pytest.raises(InvalidEditError):
        EditSpec(chrom="chr1", pos=1, ref=base_chr, alt="A")
    with pytest.raises(InvalidEditError):
        EditSpec(chrom="chr1", pos=1, ref="A", alt=base_chr)


def test_ref_equals_alt_is_rejected() -> None:
    with pytest.raises(InvalidEditError):
        EditSpec(chrom="chr1", pos=1, ref="A", alt="A")
    with pytest.raises(InvalidEditError):
        EditSpec(chrom="chr1", pos=1, ref="ACGT", alt="ACGT")


# ---------------------------------------------------------------------------
# UnsupportedEditError — SV trap.


def test_unsupported_edit_when_ref_too_long() -> None:
    long_ref = "A" * (V1_MAX_LEN + 1)
    with pytest.raises(UnsupportedEditError) as ei:
        EditSpec(chrom="chr1", pos=1, ref=long_ref, alt="T")
    assert ei.value.details["edit_type"] == int(EditType.SV)
    assert ei.value.details["ref_len"] == V1_MAX_LEN + 1


def test_unsupported_edit_when_alt_too_long() -> None:
    long_alt = "G" * (V1_MAX_LEN + 1)
    with pytest.raises(UnsupportedEditError) as ei:
        EditSpec(chrom="chr1", pos=1, ref="A", alt=long_alt)
    assert ei.value.details["edit_type"] == int(EditType.SV)
    assert ei.value.details["alt_len"] == V1_MAX_LEN + 1


def test_exactly_v1_max_len_is_allowed() -> None:
    boundary_ref = "A" * V1_MAX_LEN
    e = EditSpec(chrom="chr1", pos=1, ref=boundary_ref, alt="T")
    # ref=16 / alt=1 → DEL.
    assert e.edit_type is EditType.DEL


# ---------------------------------------------------------------------------
# Frozen contract.


def test_editspec_is_frozen_after_construction() -> None:
    e = EditSpec(chrom="chr1", pos=1, ref="A", alt="T")
    with pytest.raises(AttributeError):
        e.pos = 2  # type: ignore[misc]
    with pytest.raises(AttributeError):
        e.edit_type = EditType.INS  # type: ignore[misc]


def test_editspec_equality_and_hash() -> None:
    a = EditSpec(chrom="chr1", pos=10, ref="A", alt="T")
    b = EditSpec(chrom="chr1", pos=10, ref="A", alt="T")
    c = EditSpec(chrom="chr1", pos=11, ref="A", alt="T")
    assert a == b
    assert hash(a) == hash(b)
    assert a != c
    assert {a, b, c} == {a, c}


# ---------------------------------------------------------------------------
# Window-relative conversion.


def test_relative_to_basic() -> None:
    # window covers 1-based positions 50..200 (inclusive) — 0-based 49..199.
    e = EditSpec(chrom="chr1", pos=100, ref="A", alt="T")
    rel = e.relative_to(window_start_bp=49, window_end_bp=199)
    assert isinstance(rel, RelEdit)
    assert rel.rel_pos == 100 - 1 - 49  # 50
    assert rel.edit_type is EditType.SNV
    assert rel.ref_bases == "A"
    assert rel.alt_bases == "T"


def test_relative_to_left_edge() -> None:
    # Edit at the very first base of the window.
    e = EditSpec(chrom="chr1", pos=50, ref="A", alt="T")
    rel = e.relative_to(window_start_bp=49, window_end_bp=199)
    assert rel.rel_pos == 0


def test_relative_to_right_edge_just_inside() -> None:
    # SNV at the last base of the window: rel_pos = window_len - 1.
    e = EditSpec(chrom="chr1", pos=200, ref="A", alt="T")
    rel = e.relative_to(window_start_bp=49, window_end_bp=199)
    assert rel.rel_pos == 150


def test_relative_to_before_window_raises() -> None:
    e = EditSpec(chrom="chr1", pos=40, ref="A", alt="T")
    with pytest.raises(OutOfWindowError):
        e.relative_to(window_start_bp=49, window_end_bp=199)


def test_relative_to_after_window_raises() -> None:
    e = EditSpec(chrom="chr1", pos=201, ref="A", alt="T")
    with pytest.raises(OutOfWindowError):
        e.relative_to(window_start_bp=49, window_end_bp=199)


def test_relative_to_partial_overlap_for_deletion_raises() -> None:
    # 4-bp deletion starting at pos 199 would need pos 199..202; window
    # ends at 199 (0-based), i.e. 1-based 200. The DEL extends past it.
    e = EditSpec(chrom="chr1", pos=199, ref="ACGT", alt="A")
    with pytest.raises(OutOfWindowError):
        e.relative_to(window_start_bp=49, window_end_bp=199)


def test_relative_to_inverted_window_raises() -> None:
    e = EditSpec(chrom="chr1", pos=100, ref="A", alt="T")
    with pytest.raises(InvalidEditError):
        e.relative_to(window_start_bp=200, window_end_bp=199)


# ---------------------------------------------------------------------------
# RelEdit validation.


def test_reledit_rejects_negative_rel_pos() -> None:
    with pytest.raises(InvalidEditError):
        RelEdit(rel_pos=-1, edit_type=EditType.SNV, ref_bases="A", alt_bases="T")


def test_reledit_accepts_plain_int_edit_type() -> None:
    r = RelEdit(rel_pos=0, edit_type=int(EditType.SNV), alt_bases="T", ref_bases="A")  # type: ignore[arg-type]
    assert r.edit_type is EditType.SNV


def test_reledit_rejects_invalid_edit_type_int() -> None:
    with pytest.raises(InvalidEditError):
        RelEdit(rel_pos=0, edit_type=999, ref_bases="A", alt_bases="T")  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Property test: VCF row → EditSpec round-trip.


def _vcf_row(spec: EditSpec) -> tuple[str, int, str, str]:
    """Project an EditSpec into a 4-tuple modelling a VCF row."""
    return spec.chrom, spec.pos, spec.ref, spec.alt


def test_vcf_round_trip_property() -> None:
    """For 1000 valid edits, ``EditSpec(*_vcf_row(s)) == s``."""
    import random

    rng = random.Random(0xCAB0_0DEF)

    def _rand_bases(rng: random.Random, n: int) -> str:
        return "".join(rng.choice("ACGT") for _ in range(n))

    cases = 0
    for _ in range(2000):
        ref_len = rng.randint(1, V1_MAX_LEN)
        alt_len = rng.randint(1, V1_MAX_LEN)
        ref = _rand_bases(rng, ref_len)
        alt = _rand_bases(rng, alt_len)
        if ref == alt:
            continue  # disallowed
        chrom = rng.choice(["chr1", "chr2", "chrX", "chrM"])
        pos = rng.randint(1, 10_000_000)
        original = EditSpec(chrom=chrom, pos=pos, ref=ref, alt=alt)
        roundtrip = EditSpec(*_vcf_row(original))
        assert roundtrip == original
        assert roundtrip.edit_type is original.edit_type
        cases += 1

    # Sanity: we wanted lots of cases — generation should not have rejected
    # more than ~25% (only ref==alt collisions).
    assert cases >= 1500


# ---------------------------------------------------------------------------
# Public surface.


def test_module_exports() -> None:
    from geno_lewm import action

    for name in ("EditSpec", "EditType", "RelEdit", "V1_MAX_LEN"):
        assert hasattr(action, name)
