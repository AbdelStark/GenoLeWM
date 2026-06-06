"""Unit tests for the RFC-0008 CEM planning core."""

from __future__ import annotations

import math

import pytest

from geno_lewm.action import EditType, RelEdit
from geno_lewm.errors import InputError
from geno_lewm.planning.cem import (
    CandidateEvaluation,
    PlanningConfig,
    cosine_distance,
    l2_distance,
    plan,
    projection_distance,
    region_distance,
)
from geno_lewm.planning.sampling import ActionSampler
from geno_lewm.training import EditTypeWeight


def test_plan_is_deterministic_with_seed_and_finds_best_region() -> None:
    sampler = _snv_sampler()
    config = PlanningConfig(
        horizon=1,
        n_iterations=4,
        n_samples=128,
        n_elite=8,
        seed=7,
        stopping_eps=0.1,
        smoothing=0.0,
    )

    first = plan(_distance_to_position(17), sampler, config=config)
    second = plan(_distance_to_position(17), sampler, config=config)

    assert first.best_edits == second.best_edits
    assert first.best_distance == second.best_distance == 0.0
    assert first.best_predicted_state == second.best_predicted_state == {"best_pos": 17}
    assert first.stopped_reason == "distance_threshold"
    assert first.n_predictor_calls == first.n_evaluations
    assert all(log.n_candidates == config.n_samples for log in first.iterations)


def test_plan_stops_after_patience_without_improvement() -> None:
    result = plan(
        lambda _edits: 1.0,
        _snv_sampler(),
        config=PlanningConfig(
            horizon=1,
            n_iterations=6,
            n_samples=3,
            n_elite=1,
            patience=2,
            seed=3,
            stopping_eps=0.0,
        ),
    )

    assert result.stopped_reason == "patience"
    assert len(result.iterations) == 3
    assert result.n_evaluations == 9


def test_plan_refits_elite_position_bins() -> None:
    result = plan(
        _distance_to_position(5),
        _snv_sampler(),
        config=PlanningConfig(
            horizon=1,
            n_iterations=4,
            n_samples=96,
            n_elite=6,
            seed=11,
            stopping_eps=0.0,
            smoothing=0.0,
        ),
    )

    assert result.iterations[-1].elite_mean_distance <= result.iterations[0].elite_mean_distance
    assert result.best_edits[0].rel_pos == 5


def test_distance_helpers_cover_l2_cosine_region_and_projection() -> None:
    assert l2_distance((0.0, 0.0), (3.0, 4.0)) == 5.0
    assert cosine_distance((1.0, 0.0), (0.0, 1.0)) == 1.0
    assert region_distance((0.0, 2.0, 4.0), (0.0, 5.0, 10.0), (1, 2)) == pytest.approx(
        math.sqrt(45.0)
    )
    assert projection_distance(
        (2.0, 3.0, 5.0),
        (1.0, 4.0, 5.0),
        ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0)),
    ) == pytest.approx(math.sqrt(2.0))


def test_plan_supports_zero_horizon() -> None:
    result = plan(
        lambda edits: CandidateEvaluation(distance=float(len(edits))),
        _snv_sampler(),
        config=PlanningConfig(horizon=0, n_iterations=1, n_samples=1, n_elite=1),
    )

    assert result.best_edits == ()
    assert result.best_distance == 0.0


def test_plan_rejects_invalid_config_and_evaluations() -> None:
    with pytest.raises(InputError):
        PlanningConfig(n_samples=2, n_elite=3)
    with pytest.raises(InputError):
        PlanningConfig(seed=False)
    with pytest.raises(InputError):
        plan(lambda _edits: 0.0, object())
    with pytest.raises(InputError):
        plan(lambda _edits: math.nan, _snv_sampler(), config=PlanningConfig(n_samples=1, n_elite=1))
    with pytest.raises(InputError):
        plan(
            lambda _edits: 0.0,
            _snv_sampler(),
            config=PlanningConfig(n_samples=1, n_elite=1),
            cost_fn=lambda _edits: -1.0,
        )


def test_distance_helpers_reject_invalid_inputs() -> None:
    with pytest.raises(InputError):
        l2_distance((1.0,), (1.0, 2.0))
    with pytest.raises(InputError):
        cosine_distance((0.0,), (1.0,))
    with pytest.raises(InputError):
        region_distance((1.0,), (2.0,), (1,))
    with pytest.raises(InputError):
        projection_distance((1.0, 2.0), (1.0, 2.0), ((1.0,),))


def _distance_to_position(target_pos: int):
    def evaluate(edits: tuple[RelEdit, ...]) -> CandidateEvaluation:
        edit = edits[0]
        return CandidateEvaluation(
            distance=float(abs(edit.rel_pos - target_pos)),
            predicted_state={"best_pos": edit.rel_pos},
        )

    return evaluate


def _snv_sampler() -> ActionSampler:
    return ActionSampler(
        "ACGT" * 16,
        edge_margin=0,
        position_bin_bp=1,
        type_weights=(EditTypeWeight(EditType.SNV, 1.0),),
    )
