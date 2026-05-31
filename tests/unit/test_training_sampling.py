"""Unit tests for RFC-0005 training samplers."""

from __future__ import annotations

import random

import pytest

from geno_lewm.action import EditType
from geno_lewm.errors import InputError
from geno_lewm.training import (
    DEFAULT_EDIT_TYPE_WEIGHTS,
    DEFAULT_ROLLOUT_STEP_MIX,
    EditTypeWeight,
    RolloutStepWeight,
    draw_edit_type_counts,
    draw_rollout_step_counts,
    sample_edit_type,
    sample_rollout_steps,
)


def test_edit_type_sampler_matches_rfc_weights() -> None:
    total = 1_000_000
    counts = draw_edit_type_counts(total, rng=random.Random(42))

    for entry in DEFAULT_EDIT_TYPE_WEIGHTS:
        observed = counts[entry.edit_type] / total
        assert observed == pytest.approx(entry.weight, abs=0.002)


def test_rollout_step_sampler_matches_phase1_mix() -> None:
    total = 1_000_000
    counts = draw_rollout_step_counts(total, rng=random.Random(43))

    for entry in DEFAULT_ROLLOUT_STEP_MIX:
        observed = counts[entry.steps] / total
        assert observed == pytest.approx(entry.weight, abs=0.002)

    assert counts[1] / total == pytest.approx(0.90, abs=0.002)
    assert (counts[2] + counts[3]) / total == pytest.approx(0.10, abs=0.002)


def test_custom_sampling_normalizes_int_edit_types() -> None:
    class BoundaryRandom(random.Random):
        def random(self) -> float:
            return 1.0

    edit_weights = (EditTypeWeight(EditType.SNV, 1.0),)
    rollout_mix = (RolloutStepWeight(4, 1.0),)

    assert sample_edit_type(random.Random(1), weights=edit_weights) is EditType.SNV
    assert sample_edit_type(BoundaryRandom()) is EditType.INDEL
    assert sample_rollout_steps(random.Random(1), mix=rollout_mix) == 4
    assert sample_rollout_steps(BoundaryRandom()) == 3
    assert EditTypeWeight(1, 1.0).edit_type is EditType.INS


def test_sampler_validation_rejects_invalid_distributions() -> None:
    with pytest.raises(InputError):
        draw_edit_type_counts(-1, rng=random.Random(1))
    with pytest.raises(InputError):
        draw_edit_type_counts(1, rng=random.Random(1), weights=())
    with pytest.raises(InputError):
        draw_edit_type_counts(
            1,
            rng=random.Random(1),
            weights=(EditTypeWeight(EditType.SNV, 0.5), EditTypeWeight(EditType.SNV, 0.5)),
        )
    with pytest.raises(InputError):
        draw_edit_type_counts(1, rng=random.Random(1), weights=(EditTypeWeight(EditType.SV, 1.0),))
    with pytest.raises(InputError):
        EditTypeWeight("snv", 1.0)
    with pytest.raises(InputError):
        EditTypeWeight(EditType.SNV, float("nan"))
    with pytest.raises(InputError):
        RolloutStepWeight(0, 1.0)
    with pytest.raises(InputError):
        RolloutStepWeight(1, False)
    with pytest.raises(InputError):
        draw_rollout_step_counts(1, rng=random.Random(1), mix=())
    with pytest.raises(InputError):
        draw_rollout_step_counts(
            1,
            rng=random.Random(1),
            mix=(RolloutStepWeight(1, 0.5), RolloutStepWeight(1, 0.5)),
        )
