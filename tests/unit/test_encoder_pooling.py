"""Unit tests for ``geno_lewm.encoder.pooling``."""

from __future__ import annotations

import math

import pytest

from geno_lewm.encoder import (
    DEFAULT_POOL_RADIUS_TOKENS,
    POOL_CENTERED_MEAN,
    POOL_GLOBAL_MEAN,
    centered_mean,
    global_mean,
    pool_hidden_states,
)
from geno_lewm.errors import InputError


def test_global_mean_pools_all_tokens() -> None:
    hidden = ((1.0, 2.0), (3.0, 4.0), (5.0, 6.0))

    assert global_mean(hidden) == (3.0, 4.0)


def test_centered_mean_pools_inclusive_radius_span() -> None:
    hidden = ((0.0, 0.0), (2.0, 4.0), (4.0, 8.0), (6.0, 12.0), (8.0, 16.0))

    assert centered_mean(hidden, center_token=2, pool_radius=1) == (4.0, 8.0)


def test_centered_mean_clamps_to_available_tokens() -> None:
    hidden = ((1.0, 1.0), (3.0, 5.0), (5.0, 9.0))

    assert centered_mean(hidden, center_token=0, pool_radius=10) == (3.0, 5.0)


def test_pool_hidden_states_uses_centered_mean_by_default() -> None:
    hidden = tuple((float(i), float(i * 2)) for i in range(8))

    result = pool_hidden_states(hidden, edit_locus=18, pool_radius=1, token_bp=6)

    assert result.vector == (3.0, 6.0)
    assert result.pool_type == POOL_CENTERED_MEAN
    assert result.pool_radius == 1
    assert result.center_token == 3
    assert not result.untargeted
    assert result.token_count == 8
    assert result.d_state == 2


def test_pool_hidden_states_without_edit_locus_falls_back_to_global_mean_and_cache_tag() -> None:
    hidden = ((1.0, 2.0), (3.0, 6.0), (5.0, 10.0))

    result = pool_hidden_states(hidden)

    assert result.vector == (3.0, 6.0)
    assert result.pool_type == POOL_GLOBAL_MEAN
    assert result.pool_radius == 0
    assert result.center_token is None
    assert result.untargeted
    assert result.as_cache_fields() == {
        "pool_type": "global_mean",
        "pool_radius": 0,
        "untargeted": True,
    }


def test_pool_hidden_states_global_mean_with_locus_is_targeted_cache_metadata() -> None:
    hidden = ((1.0, 2.0), (3.0, 6.0), (5.0, 10.0))

    result = pool_hidden_states(hidden, edit_locus=6, pool_type=POOL_GLOBAL_MEAN)

    assert result.vector == (3.0, 6.0)
    assert result.as_cache_fields()["untargeted"] is False
    assert result.as_cache_fields()["pool_type"] == POOL_GLOBAL_MEAN


def test_default_pool_radius_matches_rfc() -> None:
    assert DEFAULT_POOL_RADIUS_TOKENS == 256


@pytest.mark.parametrize(
    "hidden",
    [
        (),
        ((1.0, 2.0), (3.0,)),
        ((1.0, math.inf),),
        ((1.0, float("nan")),),
    ],
)
def test_invalid_hidden_states_raise_input_error(hidden: tuple[tuple[float, ...], ...]) -> None:
    with pytest.raises(InputError):
        global_mean(hidden)


@pytest.mark.parametrize("pool_radius", [-1, True])
def test_invalid_pool_radius_raises(pool_radius: int) -> None:
    with pytest.raises(InputError):
        centered_mean(((1.0,),), center_token=0, pool_radius=pool_radius)


def test_invalid_pool_type_raises() -> None:
    with pytest.raises(InputError):
        pool_hidden_states(((1.0,),), edit_locus=0, pool_type="attention")  # type: ignore[arg-type]


def test_edit_locus_outside_hidden_states_raises() -> None:
    with pytest.raises(InputError):
        pool_hidden_states(((1.0,),), edit_locus=6, token_bp=6)
