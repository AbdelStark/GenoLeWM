"""Tests for ``geno_lewm.provenance.commitment``."""

from __future__ import annotations

import pytest

from geno_lewm.action import EditSpec
from geno_lewm.errors import InputError
from geno_lewm.provenance import DtypeConfig, PoolingConfig, compute_input_commitment


def _edit() -> EditSpec:
    return EditSpec(chrom="chr1", pos=100, ref="A", alt="T")


def _pool() -> PoolingConfig:
    return PoolingConfig(state_layer=12, pool_type="centered_mean", pool_radius=24, normalize=True)


def _dtype() -> DtypeConfig:
    return DtypeConfig(encoder_dtype="bf16", predictor_dtype="bf16")


def test_identical_inputs_produce_identical_commitments() -> None:
    w = "ACGT" * 3072
    c1 = compute_input_commitment(w, _edit(), _pool(), _dtype())
    c2 = compute_input_commitment(w, _edit(), _pool(), _dtype())
    assert c1 == c2
    assert c1.startswith("sha256:")


def test_distinct_pool_radius_changes_commitment() -> None:
    w = "ACGT" * 3072
    a = compute_input_commitment(w, _edit(), _pool(), _dtype())
    pool_b = PoolingConfig(
        state_layer=12, pool_type="centered_mean", pool_radius=32, normalize=True
    )
    b = compute_input_commitment(w, _edit(), pool_b, _dtype())
    assert a != b


def test_distinct_state_layer_changes_commitment() -> None:
    w = "ACGT" * 3072
    a = compute_input_commitment(w, _edit(), _pool(), _dtype())
    pool_b = PoolingConfig(state_layer=8, pool_type="centered_mean", pool_radius=24, normalize=True)
    b = compute_input_commitment(w, _edit(), pool_b, _dtype())
    assert a != b


def test_distinct_pool_type_changes_commitment() -> None:
    w = "ACGT" * 3072
    a = compute_input_commitment(w, _edit(), _pool(), _dtype())
    pool_b = PoolingConfig(state_layer=12, pool_type="mean", pool_radius=24, normalize=True)
    b = compute_input_commitment(w, _edit(), pool_b, _dtype())
    assert a != b


def test_distinct_normalize_flag_changes_commitment() -> None:
    w = "ACGT" * 3072
    a = compute_input_commitment(w, _edit(), _pool(), _dtype())
    pool_b = PoolingConfig(
        state_layer=12, pool_type="centered_mean", pool_radius=24, normalize=False
    )
    b = compute_input_commitment(w, _edit(), pool_b, _dtype())
    assert a != b


def test_distinct_encoder_dtype_changes_commitment() -> None:
    w = "ACGT" * 3072
    a = compute_input_commitment(w, _edit(), _pool(), _dtype())
    dt_b = DtypeConfig(encoder_dtype="fp16", predictor_dtype="bf16")
    b = compute_input_commitment(w, _edit(), _pool(), dt_b)
    assert a != b


def test_distinct_predictor_dtype_changes_commitment() -> None:
    w = "ACGT" * 3072
    a = compute_input_commitment(w, _edit(), _pool(), _dtype())
    dt_b = DtypeConfig(encoder_dtype="bf16", predictor_dtype="fp32")
    b = compute_input_commitment(w, _edit(), _pool(), dt_b)
    assert a != b


def test_distinct_edit_alt_changes_commitment() -> None:
    w = "ACGT" * 3072
    a = compute_input_commitment(w, _edit(), _pool(), _dtype())
    other_edit = EditSpec(chrom="chr1", pos=100, ref="A", alt="G")
    b = compute_input_commitment(w, other_edit, _pool(), _dtype())
    assert a != b


def test_distinct_window_changes_commitment() -> None:
    w1 = "ACGT" * 3072
    w2 = "ACGA" + "ACGT" * 3071
    a = compute_input_commitment(w1, _edit(), _pool(), _dtype())
    b = compute_input_commitment(w2, _edit(), _pool(), _dtype())
    assert a != b


def test_empty_window_rejected() -> None:
    with pytest.raises(InputError):
        compute_input_commitment("", _edit(), _pool(), _dtype())


def test_non_string_window_rejected() -> None:
    with pytest.raises(InputError):
        compute_input_commitment(b"ACGT", _edit(), _pool(), _dtype())  # type: ignore[arg-type]


def test_pooling_validation() -> None:
    with pytest.raises(InputError):
        PoolingConfig(state_layer=-1, pool_type="mean", pool_radius=0, normalize=True)
    with pytest.raises(InputError):
        PoolingConfig(state_layer=0, pool_type="", pool_radius=0, normalize=True)
    with pytest.raises(InputError):
        PoolingConfig(state_layer=0, pool_type="mean", pool_radius=-1, normalize=True)


def test_dtype_validation() -> None:
    with pytest.raises(InputError):
        DtypeConfig(encoder_dtype="", predictor_dtype="bf16")
    with pytest.raises(InputError):
        DtypeConfig(encoder_dtype="bf16", predictor_dtype="")
