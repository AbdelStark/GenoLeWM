"""Unit tests for surprise-scoring contract surprise context stratification."""

from __future__ import annotations

import pytest
from hypothesis import given, strategies as st

from geno_lewm.errors import InputError
from geno_lewm.surprise import (
    DEFAULT_MIN_BUCKET_SIZE,
    ContextLabel,
    backoff_chain,
    classify_context,
    classify_gc_bin,
    classify_region,
    classify_repeat,
    gc_fraction,
    make_bucket_id,
    select_backoff_bucket,
)


@pytest.mark.parametrize(
    ("region", "gc_window", "repeat", "expected_bucket_id"),
    [
        ("missense_variant", "ACGT" * 4, None, "coding_missense|mid|none"),
        (
            "splice_acceptor_variant&intron_variant",
            "AAAAAC",
            "simple_repeat",
            "splice|low|simple",
        ),
        ("promoter_region", "GGGGAT", "LINE/L1", "promoter|high|transposon"),
        (
            "intergenic_variant",
            "NNACGTNN",
            "genomic_superdup",
            "intergenic|mid|segmental_dup",
        ),
    ],
)
def test_bucket_assignment_matches_manual_fixture_set(
    region: str,
    gc_window: str,
    repeat: str | None,
    expected_bucket_id: str,
) -> None:
    label = classify_context(region=region, gc_window=gc_window, repeat=repeat)

    assert label.bucket_id == expected_bucket_id
    assert backoff_chain(label)[0] == expected_bucket_id


def test_context_label_validates_and_exposes_stable_bucket_id() -> None:
    label = ContextLabel("coding_synonymous", "low", "none")

    assert label.as_tuple() == ("coding_synonymous", "low", "none")
    assert label.bucket_id == "coding_synonymous|low|none"
    assert make_bucket_id("coding_synonymous", "low", "none") == label.bucket_id

    with pytest.raises(InputError):
        ContextLabel("coding_missense", "middle", "none")
    with pytest.raises(InputError):
        make_bucket_id("coding_missense", "mid", "unknown_repeat")


def test_region_classifier_uses_specific_annotation_precedence() -> None:
    assert classify_region("intron_variant&splice_donor_variant") == "splice"
    assert classify_region(["stop_gained", "missense_variant"]) == "coding_nonsense"
    assert classify_region("synonymous_variant") == "coding_synonymous"
    assert classify_region("novel_unmapped_annotation") == "other"
    assert classify_region(None) == "other"


def test_repeat_classifier_maps_common_tracks_and_rejects_unknown_terms() -> None:
    assert classify_repeat(None) == "none"
    assert classify_repeat("microsatellite") == "simple"
    assert classify_repeat("low-complexity-region") == "low_complexity"
    assert classify_repeat("SINE/Alu") == "transposon"
    assert classify_repeat("segmental duplication") == "segmental_dup"

    with pytest.raises(InputError):
        classify_repeat("unclassified_repeat_family")


def test_gc_fraction_excludes_unknown_bases() -> None:
    assert gc_fraction("NNGC") == 1.0
    assert gc_fraction("NNAT") == 0.0

    with pytest.raises(InputError):
        gc_fraction("NNNN")


@pytest.mark.parametrize(
    ("sequence", "expected"),
    [
        ("AAAAAC", "low"),
        ("ACGTACGT", "mid"),
        ("GGGGAT", "high"),
    ],
)
def test_gc_bin_defaults_are_deterministic(sequence: str, expected: str) -> None:
    assert classify_gc_bin(sequence) == expected


def test_gc_bin_accepts_fitted_cutpoints() -> None:
    assert classify_gc_bin("ACGT", low_cutoff=0.45, high_cutoff=0.55) == "mid"
    assert classify_gc_bin("ACGT", low_cutoff=0.60, high_cutoff=0.80) == "low"

    with pytest.raises(InputError):
        classify_gc_bin("ACGT", low_cutoff=0.8, high_cutoff=0.2)
    with pytest.raises(InputError):
        classify_gc_bin("ACGT", low_cutoff=-0.1)


def test_backoff_chain_is_stable_for_full_and_parent_buckets() -> None:
    label = ContextLabel("coding_missense", "mid", "none")

    assert label.backoff_chain() == (
        "coding_missense|mid|none",
        "coding_missense|mid",
        "coding_missense",
        "*",
    )
    assert backoff_chain("coding_missense|mid") == (
        "coding_missense|mid",
        "coding_missense",
        "*",
    )
    assert backoff_chain("*") == ("*",)


def test_select_backoff_bucket_uses_first_populated_parent() -> None:
    label = ContextLabel("coding_missense", "mid", "none")
    counts = {
        label.bucket_id: DEFAULT_MIN_BUCKET_SIZE - 1,
        "coding_missense|mid": DEFAULT_MIN_BUCKET_SIZE,
        "coding_missense": DEFAULT_MIN_BUCKET_SIZE * 2,
    }

    assert select_backoff_bucket(label, counts) == "coding_missense|mid"
    assert (
        select_backoff_bucket(label, {label.bucket_id: DEFAULT_MIN_BUCKET_SIZE}) == label.bucket_id
    )
    assert select_backoff_bucket(label, {}) == "*"


@given(
    threshold=st.integers(min_value=1, max_value=5_000),
    parent_extra=st.integers(min_value=0, max_value=5_000),
    region_extra=st.integers(min_value=0, max_value=5_000),
)
def test_sparse_full_bucket_backs_off_to_populated_parent(
    threshold: int,
    parent_extra: int,
    region_extra: int,
) -> None:
    label = ContextLabel("enhancer", "high", "low_complexity")
    counts = {
        label.bucket_id: threshold - 1,
        "enhancer|high": threshold + parent_extra,
        "enhancer": threshold + region_extra,
    }

    assert select_backoff_bucket(label, counts, min_count=threshold) == "enhancer|high"


@pytest.mark.parametrize(
    "bucket_id",
    [
        "coding_missense|middle|none",
        "coding_missense|mid|unknown",
        "coding_missense|mid|none|extra",
        "",
    ],
)
def test_invalid_bucket_ids_are_rejected(bucket_id: str) -> None:
    with pytest.raises(InputError):
        backoff_chain(bucket_id)


def test_bucket_size_validation_rejects_bad_counts() -> None:
    label = ContextLabel("intergenic", "low", "none")

    with pytest.raises(InputError):
        select_backoff_bucket(label, {label.bucket_id: -1})
    with pytest.raises(InputError):
        select_backoff_bucket(label, {label.bucket_id: True})
    with pytest.raises(InputError):
        select_backoff_bucket(label, {"bad|mid|none": 10})
    with pytest.raises(InputError):
        select_backoff_bucket(label, {label.bucket_id: 10}, min_count=0)
