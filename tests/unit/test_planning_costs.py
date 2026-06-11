"""Unit tests for planning contract planning cost functions."""

from __future__ import annotations

import math

import pytest

from geno_lewm.action import EditType, RelEdit
from geno_lewm.errors import InputError
from geno_lewm.planning import (
    DEFAULT_TYPE_COSTS,
    bp_cost,
    count_cost,
    custom_cost,
    edit_bp_cost,
    weighted_type_cost,
)


def test_count_cost_counts_candidate_edits() -> None:
    assert count_cost(()) == 0.0
    assert count_cost((_snv(), _ins())) == 2.0


def test_bp_cost_scores_each_edit_shape() -> None:
    edits = (_snv(), _ins(), _del(), _mnv(), _indel())
    assert [edit_bp_cost(edit) for edit in edits] == [1.0, 3.0, 4.0, 3.0, 4.0]
    assert bp_cost(edits) == 15.0


def test_weighted_type_cost_uses_custom_weights() -> None:
    edits = (_snv(), _ins(), _del(), _mnv(), _indel())
    weights = {
        EditType.SNV: 0.5,
        EditType.INS: 2.0,
        EditType.DEL: 3.0,
        EditType.MNV: 4.0,
        EditType.INDEL: 7.0,
    }
    assert weighted_type_cost(edits, weights) == 16.5
    assert weighted_type_cost((_snv(),)) == DEFAULT_TYPE_COSTS[EditType.SNV]


def test_custom_cost_returns_validated_value() -> None:
    edits = (_snv(), _ins())
    assert custom_cost(edits, lambda seq: float(len(seq) * 10)) == 20.0


@pytest.mark.parametrize(
    "value",
    [False, -1.0, math.nan, math.inf],
)
def test_custom_cost_rejects_invalid_values(value: float) -> None:
    with pytest.raises(InputError):
        custom_cost((_snv(),), lambda _seq: value)


def test_weighted_type_cost_rejects_invalid_or_missing_weights() -> None:
    with pytest.raises(InputError):
        weighted_type_cost((_snv(),), {})
    with pytest.raises(InputError):
        weighted_type_cost((_snv(),), {EditType.INS: 1.0})
    with pytest.raises(InputError):
        weighted_type_cost((_snv(),), {EditType.SNV: math.nan})
    with pytest.raises(InputError):
        weighted_type_cost((_snv(),), {EditType.SV: 1.0})


def test_costs_reject_inconsistent_rel_edit_shapes() -> None:
    bad_type = RelEdit(rel_pos=2, edit_type=EditType.SNV, ref_bases="A", alt_bases="AC")
    identity = RelEdit(rel_pos=2, edit_type=EditType.SNV, ref_bases="A", alt_bases="A")

    with pytest.raises(InputError):
        edit_bp_cost(bad_type)
    with pytest.raises(InputError):
        weighted_type_cost((identity,))


def _snv() -> RelEdit:
    return RelEdit(rel_pos=10, edit_type=EditType.SNV, ref_bases="A", alt_bases="C")


def _ins() -> RelEdit:
    return RelEdit(rel_pos=10, edit_type=EditType.INS, ref_bases="A", alt_bases="ACGT")


def _del() -> RelEdit:
    return RelEdit(rel_pos=10, edit_type=EditType.DEL, ref_bases="ACGTA", alt_bases="A")


def _mnv() -> RelEdit:
    return RelEdit(rel_pos=10, edit_type=EditType.MNV, ref_bases="ACG", alt_bases="TCA")


def _indel() -> RelEdit:
    return RelEdit(rel_pos=10, edit_type=EditType.INDEL, ref_bases="AC", alt_bases="AGGT")
