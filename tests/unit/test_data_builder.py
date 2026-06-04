"""Unit tests for the RFC-0006 training tuple builder."""

from __future__ import annotations

import random
from collections import Counter

import pytest

from geno_lewm.action import EditSpec, EditType, RelEdit
from geno_lewm.data import (
    DEFAULT_EDIT_SOURCE_COUNTS,
    SOURCE_CLINVAR,
    SOURCE_GNOMAD_COMMON,
    SOURCE_SYNTHETIC_INDEL,
    SOURCE_SYNTHETIC_SNV,
    EditSourceCount,
    GenoLeWMDataset,
    HoldoutInterval,
    HoldoutPolicy,
    WindowContext,
    build_training_tuples,
    synthetic_indel_provider,
    synthetic_snv_provider,
    variant_provider,
)
from geno_lewm.errors import InputError


def test_builder_maintains_rfc_source_mix_when_sources_are_available() -> None:
    window = _window()
    providers = {
        SOURCE_GNOMAD_COMMON: variant_provider(
            (
                EditSpec("1", 101, "A", "C"),
                EditSpec("1", 105, "A", "G"),
                EditSpec("1", 109, "A", "T"),
            )
        ),
        SOURCE_SYNTHETIC_SNV: _fixed_provider(_snv(20), _snv(24), _snv(28)),
        SOURCE_SYNTHETIC_INDEL: _fixed_provider(_ins(32)),
        SOURCE_CLINVAR: variant_provider((EditSpec("1", 113, "A", "T"),)),
    }

    tuples = build_training_tuples(window, providers, rng=random.Random(7))

    assert len(tuples) == 8
    assert Counter(item.edit_source for item in tuples) == {
        SOURCE_GNOMAD_COMMON: 3,
        SOURCE_SYNTHETIC_SNV: 3,
        SOURCE_SYNTHETIC_INDEL: 1,
        SOURCE_CLINVAR: 1,
    }
    assert all(item.window_id == window.window_id for item in tuples)
    assert all(len(item.target_window) == len(window.sequence) for item in tuples)


def test_clinvar_slot_falls_back_to_synthetic_snv_when_unavailable() -> None:
    window = _window()
    providers = {
        SOURCE_GNOMAD_COMMON: _fixed_provider(_snv(20), _snv(24), _snv(28)),
        SOURCE_SYNTHETIC_SNV: _fixed_provider(_snv(40), _snv(44), _snv(48), _snv(52)),
        SOURCE_SYNTHETIC_INDEL: _fixed_provider(_ins(60)),
        SOURCE_CLINVAR: _fixed_provider(),
    }

    tuples = build_training_tuples(window, providers, rng=random.Random(1))

    assert Counter(item.edit_source for item in tuples) == {
        SOURCE_GNOMAD_COMMON: 3,
        SOURCE_SYNTHETIC_SNV: 4,
        SOURCE_SYNTHETIC_INDEL: 1,
    }


def test_absolute_providers_fall_back_to_synthetic_on_unplaced_windows() -> None:
    # The synthetic Carbon pretraining corpus yields windows with no genome
    # coordinates (chrom is None). Absolute VCF providers cannot place variants
    # there, so both gnomAD and ClinVar slots fall back to synthetic SNVs and the
    # window still produces a full RFC-0006 edit tuple.
    unplaced = WindowContext(
        record_id="carbon-1",
        source="eukaryotic_genes",
        sequence="A" * 256,
        start_bp=0,
        chrom=None,
    )
    providers = {
        SOURCE_GNOMAD_COMMON: variant_provider((EditSpec("1", 101, "A", "C"),)),
        SOURCE_SYNTHETIC_SNV: _fixed_provider(*[_snv(20 + 4 * i) for i in range(7)]),
        SOURCE_SYNTHETIC_INDEL: _fixed_provider(_ins(60)),
        SOURCE_CLINVAR: variant_provider((EditSpec("1", 113, "A", "T"),)),
    }

    tuples = build_training_tuples(unplaced, providers, rng=random.Random(3))

    # 3 gnomAD + 1 ClinVar slots both fall back to synthetic_snv (3 native + 4).
    assert Counter(item.edit_source for item in tuples) == {
        SOURCE_SYNTHETIC_SNV: 7,
        SOURCE_SYNTHETIC_INDEL: 1,
    }
    assert all(item.window_id == unplaced.window_id for item in tuples)


def test_holdout_policy_excludes_chr_window_and_intersecting_edits() -> None:
    providers = {
        SOURCE_SYNTHETIC_SNV: _fixed_provider(_snv(20)),
    }
    mix = (EditSourceCount(SOURCE_SYNTHETIC_SNV, 1),)

    chr21 = _window(chrom="21")
    assert (
        build_training_tuples(
            chr21,
            providers,
            rng=random.Random(1),
            mix=mix,
            holdouts=HoldoutPolicy(holdout_chroms=("21",)),
        )
        == ()
    )

    chr1 = _window(chrom="1")
    assert (
        build_training_tuples(
            chr1,
            providers,
            rng=random.Random(1),
            mix=mix,
            holdouts=HoldoutPolicy(intervals=(HoldoutInterval("1", 118, 122),)),
        )
        == ()
    )

    with pytest.raises(InputError, match="did not produce enough edits"):
        build_training_tuples(
            chr1,
            providers,
            rng=random.Random(1),
            mix=mix,
            holdouts=HoldoutPolicy(edit_keys=("1:121:A:C",)),
        )


def test_variant_provider_filters_by_window_coordinates_and_is_deterministic() -> None:
    window = _window()
    provider = variant_provider(
        (
            EditSpec("1", 101, "A", "C"),
            EditSpec("1", 105, "A", "G"),
            EditSpec("2", 101, "A", "C"),
            EditSpec("1", 10_000, "A", "C"),
        )
    )

    first = provider(window, 2, random.Random(123))
    second = provider(window, 2, random.Random(123))

    assert first == second
    assert {edit.rel_pos for edit in first} == {0, 4}


def test_synthetic_providers_are_deterministic_and_builder_validates_inputs() -> None:
    window = _window(sequence="ACGT" * 80)

    assert synthetic_snv_provider(window, 4, random.Random(5)) == synthetic_snv_provider(
        window, 4, random.Random(5)
    )
    assert synthetic_indel_provider(window, 4, random.Random(6)) == synthetic_indel_provider(
        window, 4, random.Random(6)
    )

    with pytest.raises(InputError, match="missing edit provider"):
        build_training_tuples(
            window,
            {},
            rng=random.Random(1),
            mix=(EditSourceCount(SOURCE_GNOMAD_COMMON, 1),),
        )
    with pytest.raises(InputError, match="duplicate"):
        build_training_tuples(
            window,
            {SOURCE_SYNTHETIC_SNV: _fixed_provider(_snv(20))},
            rng=random.Random(1),
            mix=(
                EditSourceCount(SOURCE_SYNTHETIC_SNV, 1),
                EditSourceCount(SOURCE_SYNTHETIC_SNV, 1),
            ),
        )


def test_default_mix_is_the_rfc_3_3_1_1_allocation() -> None:
    assert [(entry.source, entry.count) for entry in DEFAULT_EDIT_SOURCE_COUNTS] == [
        (SOURCE_GNOMAD_COMMON, 3),
        (SOURCE_SYNTHETIC_SNV, 3),
        (SOURCE_SYNTHETIC_INDEL, 1),
        (SOURCE_CLINVAR, 1),
    ]


def test_genolewm_dataset_streams_tuples_deterministically() -> None:
    windows = (_window(record_id="record-1"), _window(record_id="record-2", start_bp=500))
    providers = {SOURCE_SYNTHETIC_SNV: synthetic_snv_provider}
    mix = (EditSourceCount(SOURCE_SYNTHETIC_SNV, 2),)

    first = list(
        GenoLeWMDataset(
            lambda: iter(windows), providers, seed=11, mix=mix
        ).iter_with_source_windows()
    )
    second = list(
        GenoLeWMDataset(
            lambda: iter(windows), providers, seed=11, mix=mix
        ).iter_with_source_windows()
    )

    assert first == second
    assert len(first) == 4
    assert {item.source_window.record_id for item in first} == {"record-1", "record-2"}
    assert all(
        len(item.training_tuple.target_window) == len(item.source_window.sequence) for item in first
    )


def test_genolewm_dataset_respects_holdouts_and_validates_windows() -> None:
    providers = {SOURCE_SYNTHETIC_SNV: _fixed_provider(_snv(20))}
    mix = (EditSourceCount(SOURCE_SYNTHETIC_SNV, 1),)

    held_out = GenoLeWMDataset(
        (_window(chrom="21"),),
        providers,
        seed=1,
        mix=mix,
        holdouts=HoldoutPolicy(holdout_chroms=("21",)),
    )
    assert list(held_out) == []

    invalid = GenoLeWMDataset((object(),), providers, seed=1, mix=mix)  # type: ignore[arg-type]
    with pytest.raises(InputError, match="WindowContext"):
        list(invalid)


def _window(
    *,
    chrom: str = "1",
    sequence: str | None = None,
    record_id: str = "record-1",
    start_bp: int = 100,
) -> WindowContext:
    return WindowContext(
        record_id=record_id,
        source="eukaryotic_genes",
        sequence=("A" * 256) if sequence is None else sequence,
        start_bp=start_bp,
        chrom=chrom,
    )


def _snv(rel_pos: int, ref: str = "A", alt: str = "C") -> RelEdit:
    return RelEdit(rel_pos=rel_pos, edit_type=EditType.SNV, ref_bases=ref, alt_bases=alt)


def _ins(rel_pos: int) -> RelEdit:
    return RelEdit(rel_pos=rel_pos, edit_type=EditType.INS, ref_bases="A", alt_bases="AC")


def _fixed_provider(*edits: RelEdit):
    def _provider(window: WindowContext, count: int, rng: random.Random) -> tuple[RelEdit, ...]:
        del window, count, rng
        return tuple(edits)

    return _provider
