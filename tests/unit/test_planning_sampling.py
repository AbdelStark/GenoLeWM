"""Unit tests for the planning contract planning ``ActionSampler``."""

from __future__ import annotations

import random

import pytest

from geno_lewm.action import DEFAULT_EDGE_MARGIN, V1_MAX_LEN, EditType, RelEdit, apply_edit
from geno_lewm.errors import InputError, WindowMismatchError
from geno_lewm.planning import DEFAULT_ACTION_TYPE_WEIGHTS, ActionSampler
from geno_lewm.training import EditTypeWeight


def test_sampler_is_deterministic_with_seed() -> None:
    window = _window()
    first = ActionSampler(window, seed=123).sample_sequence(16)
    second = ActionSampler(window, seed=123).sample_sequence(16)

    assert first == second


@pytest.mark.parametrize(
    "edit_type",
    [EditType.SNV, EditType.INS, EditType.DEL, EditType.MNV, EditType.INDEL],
)
def test_sampler_returns_valid_rel_edits_inside_window(edit_type: EditType) -> None:
    window = _window()
    sampler = ActionSampler(window, seed=42, length_dist={3: 1.0, 4: 1.0})

    for _ in range(64):
        edit = sampler.sample_edit(edit_type)
        _assert_valid_edit(edit, window, edit_type)
        assert apply_edit(window, edit)


def test_sampler_accepts_rng_without_consuming_seed_contract() -> None:
    window = _window()
    rng = random.Random(7)
    sampler = ActionSampler(window, rng=rng)

    assert sampler.sample_edit() == ActionSampler(window, seed=7).sample_edit()


def test_sampler_rejects_rng_and_seed_together() -> None:
    with pytest.raises(InputError):
        ActionSampler(_window(), seed=1, rng=random.Random(1))


def test_sample_sequences_supports_zero_and_positive_horizon() -> None:
    sampler = ActionSampler(_window(), seed=1)

    assert sampler.sample_sequence(0) == ()
    sequences = sampler.sample_sequences(3, 2)
    assert len(sequences) == 3
    assert all(len(seq) == 2 for seq in sequences)


def test_type_weights_can_force_a_single_edit_type() -> None:
    sampler = ActionSampler(
        _window(),
        seed=5,
        type_weights=(EditTypeWeight(EditType.MNV, 1.0),),
        length_dist={5: 1.0},
    )

    assert {sampler.sample_edit().edit_type for _ in range(20)} == {EditType.MNV}


def test_type_sampler_handles_boundary_draw() -> None:
    class BoundaryRandom(random.Random):
        def __init__(self) -> None:
            super().__init__(0)
            self.calls = 0

        def random(self) -> float:
            self.calls += 1
            if self.calls == 1:
                return 1.0
            return super().random()

    sampler = ActionSampler(
        _window(),
        rng=BoundaryRandom(),
        type_weights=(EditTypeWeight(EditType.SNV, 1.0),),
    )

    assert sampler.sample_edit().edit_type is EditType.SNV


def test_position_weights_sample_from_configured_bins() -> None:
    window = _window()
    sampler = ActionSampler(
        window,
        seed=9,
        edge_margin=16,
        position_bin_bp=8,
        position_weights={2: 1.0},
        type_weights=(EditTypeWeight(EditType.SNV, 1.0),),
    )

    positions = [sampler.sample_edit().rel_pos for _ in range(32)]
    assert min(positions) >= 16 + 2 * 8
    assert max(positions) <= 16 + 3 * 8 - 1


def test_position_weights_sequence_form_samples_configured_bin() -> None:
    sampler = ActionSampler(
        _window(),
        seed=10,
        edge_margin=16,
        position_bin_bp=4,
        position_weights=[0.0, 1.0],
        type_weights=(EditTypeWeight(EditType.SNV, 1.0),),
    )

    positions = [sampler.sample_edit().rel_pos for _ in range(16)]
    assert min(positions) >= 20
    assert max(positions) <= 23


def test_position_weights_outside_window_raise_after_retries() -> None:
    sampler = ActionSampler(
        _window(),
        seed=11,
        position_weights={10_000: 1.0},
        type_weights=(EditTypeWeight(EditType.SNV, 1.0),),
        max_attempts=4,
    )

    with pytest.raises(InputError):
        sampler.sample_edit()


def test_sampler_validation_rejects_invalid_config() -> None:
    with pytest.raises(InputError):
        ActionSampler("ACGT" * 4)
    with pytest.raises(InputError):
        ActionSampler("")
    with pytest.raises(InputError):
        ActionSampler(123)
    with pytest.raises(InputError):
        ActionSampler(_window().lower())
    with pytest.raises(InputError):
        ActionSampler(_window(), type_weights=())
    with pytest.raises(InputError):
        ActionSampler(
            _window(),
            type_weights=(
                EditTypeWeight(EditType.SNV, 1.0),
                EditTypeWeight(EditType.SNV, 2.0),
            ),
        )
    with pytest.raises(InputError):
        ActionSampler(_window(), type_weights=(EditTypeWeight(EditType.SV, 1.0),))
    with pytest.raises(InputError):
        ActionSampler(_window(), position_weights={0: 0.0})
    with pytest.raises(InputError):
        ActionSampler(_window(), position_weights={False: 1.0})
    with pytest.raises(InputError):
        ActionSampler(_window(), position_weights={0: -1.0})
    with pytest.raises(InputError):
        ActionSampler(_window(), position_bin_bp=0)
    with pytest.raises(InputError):
        ActionSampler(_window(), max_attempts=0)
    with pytest.raises(InputError):
        ActionSampler(_window(), seed=1).sample_edit(EditType.SV)
    with pytest.raises(InputError):
        ActionSampler(_window(), seed=1).sample_sequence(-1)
    with pytest.raises(InputError):
        ActionSampler(_window(), seed=1).sample_sequences(-1, 1)
    with pytest.raises(InputError):
        ActionSampler(_window(), seed=1).sample_edit("bad")
    with pytest.raises(InputError):
        ActionSampler(_window(), seed=1, length_dist={"bad": 1.0}).sample_edit(EditType.DEL)
    with pytest.raises(InputError):
        ActionSampler(_window(), seed=1, length_dist={3: 1.0}).sample_edit(EditType.INDEL)
    with pytest.raises(InputError):
        ActionSampler(_window(), seed=1, length_dist={1: 0.0}).sample_edit(EditType.DEL)


@pytest.mark.parametrize(
    "edit_type",
    [EditType.SNV, EditType.INS, EditType.DEL, EditType.MNV, EditType.INDEL],
)
def test_sampler_raises_when_window_has_no_acgt_interior(edit_type: EditType) -> None:
    window = "A" * DEFAULT_EDGE_MARGIN + "N" * 32 + "C" * DEFAULT_EDGE_MARGIN
    sampler = ActionSampler(window, seed=1, max_attempts=8)

    with pytest.raises(InputError):
        sampler.sample_edit(edit_type)


def test_sequence_length_distribution_is_honored() -> None:
    sampler = ActionSampler(_window(), seed=1, length_dist=[0.0, 0.0, 1.0])

    edit = sampler.sample_edit(EditType.DEL)

    assert len(edit.ref_bases) - len(edit.alt_bases) == 3


def test_sampler_retries_lengths_that_do_not_fit_short_interiors() -> None:
    window = "A" * DEFAULT_EDGE_MARGIN + "AC" + "C" * DEFAULT_EDGE_MARGIN
    sampler = ActionSampler(window, seed=2, max_attempts=512)

    edit = sampler.sample_edit(EditType.DEL)

    assert edit.edit_type is EditType.DEL
    assert edit.rel_pos + len(edit.ref_bases) <= len(window) - DEFAULT_EDGE_MARGIN


def test_sampler_exports_public_defaults() -> None:
    from geno_lewm import planning

    assert planning.ActionSampler is ActionSampler
    assert planning.DEFAULT_ACTION_TYPE_WEIGHTS == DEFAULT_ACTION_TYPE_WEIGHTS


def _assert_valid_edit(edit: RelEdit, window: str, edit_type: EditType) -> None:
    assert edit.edit_type is edit_type
    assert edit.rel_pos >= DEFAULT_EDGE_MARGIN
    assert edit.rel_pos + len(edit.ref_bases) <= len(window) - DEFAULT_EDGE_MARGIN
    assert 1 <= len(edit.ref_bases) <= V1_MAX_LEN
    assert 1 <= len(edit.alt_bases) <= V1_MAX_LEN
    assert set(edit.ref_bases) <= set("ACGT")
    assert set(edit.alt_bases) <= set("ACGT")
    assert edit.ref_bases != edit.alt_bases
    assert window[edit.rel_pos : edit.rel_pos + len(edit.ref_bases)] == edit.ref_bases
    try:
        apply_edit(window, edit)
    except WindowMismatchError as exc:  # pragma: no cover - assertion clarity
        raise AssertionError(edit) from exc

    if edit_type is EditType.SNV:
        assert len(edit.ref_bases) == len(edit.alt_bases) == 1
    elif edit_type is EditType.INS:
        assert len(edit.ref_bases) == 1 < len(edit.alt_bases)
    elif edit_type is EditType.DEL:
        assert len(edit.ref_bases) > 1 == len(edit.alt_bases)
    elif edit_type is EditType.MNV:
        assert len(edit.ref_bases) == len(edit.alt_bases) > 1
    elif edit_type is EditType.INDEL:
        assert len(edit.ref_bases) > 1
        assert len(edit.alt_bases) > 1
        assert len(edit.ref_bases) != len(edit.alt_bases)


def _window() -> str:
    return "ACGT" * 256
