"""Unit tests for ``geno_lewm.action.synthetic``."""

from __future__ import annotations

import random
from collections import Counter

import pytest

from geno_lewm.action import (
    DEFAULT_EDGE_MARGIN,
    V1_MAX_LEN,
    EditType,
    indel,
    mnv,
    uniform_snv,
)
from geno_lewm.errors import InputError


def _seeded(seed: int) -> random.Random:
    return random.Random(seed)


def _rand_window(rng: random.Random, n: int = 512) -> str:
    return "".join(rng.choice("ACGT") for _ in range(n))


# ---------------------------------------------------------------------------
# Determinism — same seed, same output.


def test_uniform_snv_is_deterministic() -> None:
    w = "ACGT" * 200
    a = uniform_snv(w, 8, rng=_seeded(7))
    b = uniform_snv(w, 8, rng=_seeded(7))
    assert a == b


def test_indel_is_deterministic() -> None:
    w = "ACGT" * 200
    a = indel(w, 8, rng=_seeded(11))
    b = indel(w, 8, rng=_seeded(11))
    assert a == b


def test_mnv_is_deterministic() -> None:
    w = "ACGT" * 200
    a = mnv(w, 8, rng=_seeded(13))
    b = mnv(w, 8, rng=_seeded(13))
    assert a == b


def test_indel_returns_full_count_on_window_with_n_bases() -> None:
    # Carbon corpus windows can contain N; the sampler must resample over N
    # anchors / N deletion segments and still return exactly n edits (a single
    # dropped slot crashes the data builder for the no-fallback indel source).
    w = ("ACGTNNNNNN" * 60) + "ACGT" * 20  # ~60% N bases
    edits = indel(w, 8, rng=_seeded(5))
    assert len(edits) == 8
    for e in edits:
        # Every emitted edit must be anchored on a real ACGT base.
        assert w[e.rel_pos] in "ACGT"


def test_indel_raises_on_all_n_window() -> None:
    w = "N" * 512
    with pytest.raises(InputError, match="could not sample enough indels"):
        indel(w, 4, rng=_seeded(0))


# ---------------------------------------------------------------------------
# Validation.


def test_negative_n_rejected() -> None:
    w = "ACGT" * 200
    with pytest.raises(InputError):
        uniform_snv(w, -1, rng=_seeded(0))
    with pytest.raises(InputError):
        indel(w, -1, rng=_seeded(0))
    with pytest.raises(InputError):
        mnv(w, -1, rng=_seeded(0))


def test_edge_margin_too_large_rejected() -> None:
    with pytest.raises(InputError):
        uniform_snv("ACGT" * 16, 1, rng=_seeded(0), edge_margin=64)


def test_indel_invalid_length_dist_rejected() -> None:
    w = "ACGT" * 200
    with pytest.raises(InputError):
        indel(w, 1, rng=_seeded(0), length_dist={0: 1.0})  # length 0 invalid
    with pytest.raises(InputError):
        indel(w, 1, rng=_seeded(0), length_dist={20: 1.0})  # > V1_MAX_LEN


def test_indel_invalid_type_mix_rejected() -> None:
    w = "ACGT" * 200
    with pytest.raises(InputError):
        indel(w, 1, rng=_seeded(0), type_mix=(0.0, 0.0))
    with pytest.raises(InputError):
        indel(w, 1, rng=_seeded(0), type_mix=(-1.0, 1.0))


# ---------------------------------------------------------------------------
# Edge-margin property: positions stay inside [edge_margin, len - edge_margin).


def test_property_snv_positions_respect_edge_margin() -> None:
    rng = _seeded(0xC0DE)
    for _ in range(50):
        w = _rand_window(rng, n=rng.randint(256, 1024))
        edits = uniform_snv(w, 32, rng=rng, edge_margin=DEFAULT_EDGE_MARGIN)
        for e in edits:
            assert DEFAULT_EDGE_MARGIN <= e.rel_pos < len(w) - DEFAULT_EDGE_MARGIN


def test_property_indel_positions_respect_edge_margin() -> None:
    rng = _seeded(0xB00B)
    for _ in range(50):
        w = _rand_window(rng, n=rng.randint(256, 1024))
        edits = indel(w, 32, rng=rng)
        for e in edits:
            assert e.rel_pos >= DEFAULT_EDGE_MARGIN
            # For DEL, the ref segment extends right of rel_pos; that
            # extension must not cross the right margin either.
            assert e.rel_pos + len(e.ref_bases) <= len(w) - DEFAULT_EDGE_MARGIN


def test_property_mnv_positions_respect_edge_margin() -> None:
    rng = _seeded(0xBA5E)
    for _ in range(50):
        w = _rand_window(rng, n=rng.randint(256, 1024))
        edits = mnv(w, 16, rng=rng)
        for e in edits:
            assert e.rel_pos >= DEFAULT_EDGE_MARGIN
            assert e.rel_pos + len(e.ref_bases) <= len(w) - DEFAULT_EDGE_MARGIN


# ---------------------------------------------------------------------------
# SNV property: alt is always non-reference.


def test_property_snv_alt_is_non_reference() -> None:
    rng = _seeded(0xFEED)
    for _ in range(100):
        w = _rand_window(rng, n=512)
        edits = uniform_snv(w, 16, rng=rng)
        for e in edits:
            assert e.edit_type is EditType.SNV
            assert e.ref_bases == w[e.rel_pos]
            assert e.alt_bases != e.ref_bases
            assert e.alt_bases in {"A", "C", "G", "T"}


# ---------------------------------------------------------------------------
# Indel length distribution honors length_dist.


def test_indel_length_dist_uniform_mass() -> None:
    # Force length=4 with all probability mass; every drawn indel must
    # have ref_len + alt_len agreeing with that event length.
    rng = _seeded(0)
    w = _rand_window(rng, n=2048)
    edits = indel(w, 200, rng=rng, length_dist={4: 1.0}, type_mix=(0.5, 0.5))
    for e in edits:
        if e.edit_type is EditType.INS:
            assert len(e.alt_bases) - len(e.ref_bases) == 4
        else:
            assert e.edit_type is EditType.DEL
            assert len(e.ref_bases) - len(e.alt_bases) == 4


def test_indel_length_distribution_honored_in_aggregate() -> None:
    # Two-mass distribution: lengths 1 and 8 with 50/50 weights.
    rng = _seeded(0xCAFEFEED)
    w = _rand_window(rng, n=4096)
    edits = indel(w, 400, rng=rng, length_dist={1: 1.0, 8: 1.0}, type_mix=(1.0, 0.0))
    seen: Counter[int] = Counter()
    for e in edits:
        assert e.edit_type is EditType.INS
        seen[len(e.alt_bases) - len(e.ref_bases)] += 1
    # The distribution should split roughly 50/50; allow generous slack.
    assert set(seen) == {1, 8}
    ratio = seen[1] / (seen[1] + seen[8])
    assert 0.35 <= ratio <= 0.65


def test_indel_type_mix_all_insertions() -> None:
    rng = _seeded(0)
    w = _rand_window(rng, n=1024)
    edits = indel(w, 100, rng=rng, type_mix=(1.0, 0.0))
    for e in edits:
        assert e.edit_type is EditType.INS


def test_indel_type_mix_all_deletions() -> None:
    rng = _seeded(0)
    w = _rand_window(rng, n=1024)
    edits = indel(w, 100, rng=rng, type_mix=(0.0, 1.0))
    # Deletions can fall back to INS when the right margin is too close;
    # most should still be DEL with a generous window.
    dels = sum(1 for e in edits if e.edit_type is EditType.DEL)
    assert dels >= 0.85 * len(edits)


# ---------------------------------------------------------------------------
# MNV: every base differs.


def test_property_mnv_alt_differs_at_every_position() -> None:
    rng = _seeded(0xBADA55)
    w = _rand_window(rng, n=2048)
    edits = mnv(w, 64, rng=rng)
    for e in edits:
        assert e.edit_type is EditType.MNV
        assert len(e.ref_bases) == len(e.alt_bases)
        for a, b in zip(e.ref_bases, e.alt_bases, strict=True):
            assert a != b


# ---------------------------------------------------------------------------
# Public exports.


def test_module_exports_samplers() -> None:
    from geno_lewm import action

    for name in ("uniform_snv", "indel", "mnv", "DEFAULT_EDGE_MARGIN"):
        assert hasattr(action, name)
    assert action.DEFAULT_EDGE_MARGIN == DEFAULT_EDGE_MARGIN
    assert DEFAULT_EDGE_MARGIN == 64
    assert V1_MAX_LEN == 16
