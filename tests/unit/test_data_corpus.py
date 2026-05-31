"""Unit tests for the Carbon corpus and window sampler."""

from __future__ import annotations

import importlib.util
import random
import sys
import types
from itertools import pairwise

import pytest

from geno_lewm.data import (
    CARBON_SUBMIX,
    DEFAULT_CORPUS_MARGIN_BP,
    DEFAULT_CORPUS_STRIDE_BP,
    CarbonCorpusConfig,
    CarbonRecord,
    CarbonSourceMix,
    draw_source_counts,
    iter_carbon_records,
    iter_record_windows,
    iter_window_starts,
    load_hf_carbon_records,
    normalize_source_label,
    sample_source,
    stable_subset_includes,
)
from geno_lewm.encoder.windowing import DEFAULT_WINDOW_BP, window_sha256
from geno_lewm.errors import InputError, RuntimeSetupError


def test_submix_proportions_match_rfc_on_million_sample_draw() -> None:
    total = 1_000_000
    counts = draw_source_counts(total, rng=random.Random(17))

    for entry in CARBON_SUBMIX:
        observed = counts[entry.source] / total
        assert observed == pytest.approx(entry.fraction, abs=0.002)


def test_source_sampling_normalizes_custom_mix_and_handles_boundary_draw() -> None:
    class BoundaryRandom(random.Random):
        def random(self) -> float:
            return 1.0

    mix = (
        CarbonSourceMix("mRNA transcripts", 0.5),
        CarbonSourceMix("GTDB bacterial genomes", 0.5),
    )

    assert sample_source(random.Random(1), mix=mix) in {"mrna", "gtdb"}
    assert sample_source(BoundaryRandom(), mix=mix) == "gtdb"

    with pytest.raises(InputError):
        draw_source_counts(1, rng=random.Random(1), mix=())
    with pytest.raises(InputError):
        draw_source_counts(
            1,
            rng=random.Random(1),
            mix=(CarbonSourceMix("mrna", 0.5), CarbonSourceMix("mRNA transcripts", 0.5)),
        )


def test_window_starts_respect_margin_and_stride() -> None:
    sequence_length = DEFAULT_WINDOW_BP + (2 * DEFAULT_CORPUS_MARGIN_BP)
    sequence_length += 2 * DEFAULT_CORPUS_STRIDE_BP

    starts = list(iter_window_starts(sequence_length))

    assert starts == [256, 8448, 16640]
    assert all((right - left) == DEFAULT_CORPUS_STRIDE_BP for left, right in pairwise(starts))
    assert starts[0] >= DEFAULT_CORPUS_MARGIN_BP
    assert starts[-1] + DEFAULT_WINDOW_BP <= sequence_length - DEFAULT_CORPUS_MARGIN_BP


def test_window_starts_skip_short_sequences() -> None:
    too_short = DEFAULT_WINDOW_BP + (2 * DEFAULT_CORPUS_MARGIN_BP) - 1

    assert list(iter_window_starts(too_short)) == []


def test_random_phase_keeps_stride_and_bounds() -> None:
    sequence_length = DEFAULT_WINDOW_BP + (2 * DEFAULT_CORPUS_MARGIN_BP)
    sequence_length += 3 * DEFAULT_CORPUS_STRIDE_BP

    starts = list(iter_window_starts(sequence_length, rng=random.Random(2)))

    assert starts
    assert all((right - left) == DEFAULT_CORPUS_STRIDE_BP for left, right in pairwise(starts))
    assert all(start >= DEFAULT_CORPUS_MARGIN_BP for start in starts)
    assert all(
        start + DEFAULT_WINDOW_BP <= sequence_length - DEFAULT_CORPUS_MARGIN_BP for start in starts
    )


def test_iter_carbon_records_normalizes_source_sequence_and_subset() -> None:
    rows = [
        {"id": "r1", "source": "mRNA transcripts", "sequence": "acgtn"},
        {"id": "r2", "source": "GTDB bacterial genomes", "sequence": "aaaaa"},
    ]

    records = list(iter_carbon_records(rows, subset_fraction=1.0))

    assert records == [
        CarbonRecord(record_id="r1", source="mrna", sequence="ACGTN"),
        CarbonRecord(record_id="r2", source="gtdb", sequence="AAAAA"),
    ]
    assert stable_subset_includes("r1", fraction=1.0)


def test_iter_record_windows_extracts_canonical_slices_and_hashes() -> None:
    prefix = "C" * DEFAULT_CORPUS_MARGIN_BP
    first = "A" * DEFAULT_WINDOW_BP
    second = "G" * DEFAULT_CORPUS_STRIDE_BP
    suffix = "T" * DEFAULT_CORPUS_MARGIN_BP
    record = CarbonRecord(
        record_id="record-1",
        source="mrna",
        sequence=prefix + first + second + suffix,
    )

    windows = list(iter_record_windows(record))

    assert len(windows) == 2
    assert windows[0].start_bp == DEFAULT_CORPUS_MARGIN_BP
    assert windows[0].sequence == first
    assert windows[0].window_bp == DEFAULT_WINDOW_BP
    assert windows[0].window_id == window_sha256(first).hex()
    assert windows[1].start_bp == DEFAULT_CORPUS_MARGIN_BP + DEFAULT_CORPUS_STRIDE_BP


def test_iter_carbon_records_filters_subset_and_falls_back_to_hash_id() -> None:
    dropped_id = next(
        str(candidate)
        for candidate in range(100)
        if not stable_subset_includes(str(candidate), fraction=0.5)
    )

    assert (
        list(
            iter_carbon_records(
                [{"id": dropped_id, "source": "mrna", "sequence": "AAAA"}],
                subset_fraction=0.5,
            )
        )
        == []
    )
    records = list(iter_carbon_records([{"source": "gtdb", "sequence": "CCCC"}]))

    assert records == [
        CarbonRecord(record_id=window_sha256("CCCC").hex()[:16], source="gtdb", sequence="CCCC")
    ]


def test_source_label_validation_and_config_validation() -> None:
    assert normalize_source_label("splice-enriched mRNA") == "splice_mrna"
    assert CarbonCorpusConfig(dataset_config="default").dataset_config == "default"

    with pytest.raises(InputError):
        normalize_source_label("unknown")
    with pytest.raises(InputError):
        normalize_source_label("")
    with pytest.raises(InputError):
        CarbonCorpusConfig(subset_fraction=0.0)
    with pytest.raises(InputError):
        CarbonCorpusConfig(dataset_config="")
    with pytest.raises(InputError):
        CarbonCorpusConfig(window_bp=0)
    with pytest.raises(InputError):
        CarbonCorpusConfig(margin_bp=-1)
    with pytest.raises(InputError):
        stable_subset_includes("r1", fraction=True)
    with pytest.raises(InputError):
        stable_subset_includes("r1", fraction=float("nan"))
    with pytest.raises(InputError):
        stable_subset_includes("r1", fraction=1.0, seed=-1)
    with pytest.raises(InputError):
        list(iter_carbon_records([{"source": "mrna", "sequence": "ACGTX"}]))
    with pytest.raises(InputError):
        list(iter_carbon_records([{"source": "mrna"}]))


def test_hf_loader_reports_missing_datasets_runtime() -> None:
    if importlib.util.find_spec("datasets") is not None:
        pytest.skip("datasets is installed in this environment")

    with pytest.raises(RuntimeSetupError):
        next(load_hf_carbon_records())

    with pytest.raises(RuntimeSetupError):
        next(load_hf_carbon_records(CarbonCorpusConfig(subset_fraction=1.0)))


def test_hf_loader_uses_configured_dataset_and_fields(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[tuple[tuple[str, ...], dict[str, object]]] = []

    def fake_load_dataset(*args: str, **kwargs: object) -> list[dict[str, str]]:
        calls.append((args, kwargs))
        if args == ("HuggingFaceBio/carbon-pretraining-corpus",):
            return [{"source": "mRNA transcripts", "sequence": "acgt"}]
        return [{"name": "row-1", "bucket": "mRNA transcripts", "seq": "acgt"}]

    monkeypatch.setitem(
        sys.modules,
        "datasets",
        types.SimpleNamespace(load_dataset=fake_load_dataset),
    )

    default_records = list(load_hf_carbon_records(CarbonCorpusConfig(subset_fraction=1.0)))
    configured_records = list(
        load_hf_carbon_records(
            CarbonCorpusConfig(
                dataset_id="local/carbon",
                dataset_config="v1",
                split="validation",
                streaming=False,
                subset_fraction=1.0,
                sequence_field="seq",
                source_field="bucket",
                source_id_field="name",
            )
        )
    )

    assert calls[0] == (
        ("HuggingFaceBio/carbon-pretraining-corpus",),
        {"split": "train", "streaming": True},
    )
    assert default_records == [
        CarbonRecord(record_id=window_sha256("ACGT").hex()[:16], source="mrna", sequence="ACGT")
    ]
    assert calls[1] == (
        ("local/carbon", "v1"),
        {"split": "validation", "streaming": False},
    )
    assert configured_records == [CarbonRecord(record_id="row-1", source="mrna", sequence="ACGT")]
