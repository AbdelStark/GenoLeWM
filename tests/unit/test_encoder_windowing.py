"""Unit and property tests for ``geno_lewm.encoder.windowing``."""

from __future__ import annotations

import hashlib

import pytest
from hypothesis import given, strategies as st

from geno_lewm.encoder import (
    CARBON_DNA_CLOSE_TAG,
    CARBON_DNA_OPEN_TAG,
    CARBON_TOKEN_BP,
    DEFAULT_EDIT_MARGIN_BP,
    SUPPORTED_WINDOW_BP,
    canonicalize_dna,
    extract_window,
    pad_for_carbon_tokenizer,
    window_sha256,
    wrap_dna_for_tokenizer,
)
from geno_lewm.errors import InputError, OutOfWindowError

DNA = st.text(alphabet="ACGTNacgtn", min_size=1, max_size=512)


@pytest.mark.parametrize("window_bp", SUPPORTED_WINDOW_BP)
def test_extract_window_covers_supported_lengths(window_bp: int) -> None:
    source = ("ACGT" * ((window_bp // 4) + 200)).lower()
    edit_locus = len(source) // 2

    window = extract_window(source, edit_locus=edit_locus, window_bp=window_bp)

    assert len(window.sequence) == window_bp
    assert window.window_bp == window_bp
    assert window.end_bp - window.start_bp == window_bp
    assert window.relative_edit_locus == edit_locus - window.start_bp
    assert window.relative_edit_locus == window_bp // 2
    assert window.sequence == window.sequence.upper()


@pytest.mark.parametrize("window_bp", SUPPORTED_WINDOW_BP)
def test_extract_window_right_pads_short_source_with_a(window_bp: int) -> None:
    window = extract_window("acgt", window_bp=window_bp)

    assert window.sequence == "ACGT" + ("A" * (window_bp - 4))
    assert window.pad_right_bp == window_bp - 4
    assert window.start_bp == 0
    assert window.end_bp == window_bp
    assert window.untargeted


def test_extract_window_uses_source_midpoint_when_untargeted() -> None:
    source = "A" * 30_000

    window = extract_window(source)

    assert window.edit_locus is None
    assert window.relative_edit_locus is None
    assert window.start_bp == (len(source) // 2) - (window.window_bp // 2)


@given(DNA)
def test_canonicalize_then_hash_is_stable(sequence: str) -> None:
    canonical = canonicalize_dna(sequence)

    assert window_sha256(sequence) == window_sha256(canonical)
    assert window_sha256(sequence) == hashlib.sha256(canonical.encode("ascii")).digest()


@given(
    st.sampled_from(SUPPORTED_WINDOW_BP),
    st.integers(min_value=0, max_value=2_000),
)
def test_edit_locus_centering_keeps_locus_inside_margin(window_bp: int, offset: int) -> None:
    source_len = window_bp * 3
    source = "ACGT" * ((source_len // 4) + 1)
    lower = DEFAULT_EDIT_MARGIN_BP
    upper = len(source) - DEFAULT_EDIT_MARGIN_BP - 1
    edit_locus = lower + (offset % (upper - lower + 1))

    window = extract_window(source, edit_locus=edit_locus, window_bp=window_bp)

    assert window.relative_edit_locus is not None
    assert (
        DEFAULT_EDIT_MARGIN_BP <= window.relative_edit_locus <= window_bp - DEFAULT_EDIT_MARGIN_BP
    )


def test_tokenizer_wrapper_uppercases_pads_and_wraps() -> None:
    wrapped = wrap_dna_for_tokenizer("acgtn")

    assert wrapped.startswith(CARBON_DNA_OPEN_TAG)
    assert wrapped.endswith(CARBON_DNA_CLOSE_TAG)
    body = wrapped.removeprefix(CARBON_DNA_OPEN_TAG).removesuffix(CARBON_DNA_CLOSE_TAG)
    assert body == "ACGTNA"
    assert len(body) % CARBON_TOKEN_BP == 0


@pytest.mark.parametrize("window_bp", SUPPORTED_WINDOW_BP)
def test_extracted_window_tokenizer_input_is_tagged_and_token_aligned(window_bp: int) -> None:
    window = extract_window("ACGT" * ((window_bp // 4) + 1), window_bp=window_bp)

    wrapped = window.as_tokenizer_input()
    body = wrapped.removeprefix(CARBON_DNA_OPEN_TAG).removesuffix(CARBON_DNA_CLOSE_TAG)

    assert wrapped.startswith(CARBON_DNA_OPEN_TAG)
    assert wrapped.endswith(CARBON_DNA_CLOSE_TAG)
    assert body.startswith(window.sequence)
    assert len(body) >= window_bp
    assert len(body) % CARBON_TOKEN_BP == 0


def test_tokenizer_padding_is_noop_when_aligned() -> None:
    assert pad_for_carbon_tokenizer("acgtac") == "ACGTAC"


def test_invalid_bases_raise_input_error() -> None:
    with pytest.raises(InputError):
        canonicalize_dna("ACGU")


def test_unsupported_window_length_raises() -> None:
    with pytest.raises(InputError):
        extract_window("ACGT" * 100, window_bp=8_192)


def test_out_of_range_edit_locus_raises() -> None:
    with pytest.raises(OutOfWindowError):
        extract_window("ACGT", edit_locus=4, window_bp=4_096)
