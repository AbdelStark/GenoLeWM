"""Unit tests for the edit-response spectroscopy research tool.

These tests never load Carbon: a deterministic fake encoder returns known
per-token states so the tool's window extraction, radius sweep, geometry,
Parquet schema, resume, skip, and summary behaviour can be asserted exactly.
"""

from __future__ import annotations

import importlib
import json
import math
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import pytest

from geno_lewm.encoder.pooling import pool_hidden_states
from tools.research import edit_response_spectroscopy as ers
from tools.research.edit_response_spectroscopy import EncodedTokenStates

_CONTIG = "chr1"
_WINDOW_BP = 4096


class FakeStateEncoder:
    """Deterministic stand-in for ``CarbonStateEncoder.encode_token_states``.

    Returns five tokens per window whose per-token variation (``t * t``)
    makes the global-mean pool (radius 0) differ from the centred-content
    pool (radius > 0). The state depends on the window's C/G counts so a
    single-base edit produces a non-trivial displacement.
    """

    N_TOKENS = 5
    CONTENT = (1, 4)
    CENTER = 2

    def __init__(self) -> None:
        self.batch_sizes: list[int] = []

    def states_for(self, window: str, edit_locus: int | None) -> EncodedTokenStates:
        c = float(window.count("C"))
        g = float(window.count("G"))
        rows = tuple(
            (1.0 + c + float(t * t), 2.0 + g + float(t), 3.0 + c - float(t))
            for t in range(self.N_TOKENS)
        )
        center = None if edit_locus is None else self.CENTER
        return EncodedTokenStates(
            rows=rows,
            center_token=center,
            content_token_bounds=self.CONTENT,
        )

    def encode_token_states(
        self,
        windows: Any,
        edit_loci: Any,
    ) -> tuple[EncodedTokenStates, ...]:
        self.batch_sizes.append(len(windows))
        return tuple(
            self.states_for(window, locus) for window, locus in zip(windows, edit_loci, strict=True)
        )


def _reference_sequences() -> dict[str, str]:
    return {_CONTIG: "A" * 120}


def _variants() -> list[ers.VariantRecord]:
    return [
        ers.VariantRecord(
            variant_id="chr1:10:A:C",
            chrom=_CONTIG,
            pos=10,
            ref="A",
            alt="C",
            label="pathogenic",
            label_group="clinvar",
            continuous_score=0.9,
            region="exon",
            gene="BRCA2",
        ),
        ers.VariantRecord(
            variant_id="chr1:50:A:G",
            chrom=_CONTIG,
            pos=50,
            ref="A",
            alt="G",
            label="benign",
            label_group="gnomad",
            continuous_score=0.1,
        ),
        ers.VariantRecord(
            variant_id="chr1:90:A:C",
            chrom=_CONTIG,
            pos=90,
            ref="A",
            alt="C",
            label="pathogenic",
            label_group="clinvar",
        ),
    ]


def _skip_variant() -> ers.VariantRecord:
    # REF says T but the contig is A at pos 15 -> FASTA extraction fails.
    return ers.VariantRecord(
        variant_id="chr1:15:T:C",
        chrom=_CONTIG,
        pos=15,
        ref="T",
        alt="C",
    )


def _expected_vector(
    states: EncodedTokenStates,
    rel_pos: int,
    pool_radius: int,
) -> tuple[float, ...]:
    if pool_radius == 0:
        return pool_hidden_states(states.rows, pool_type="global_mean", pool_radius=0).vector
    return pool_hidden_states(
        states.rows,
        edit_locus=rel_pos,
        center_token=states.center_token,
        content_token_bounds=states.content_token_bounds,
        pool_type="centered_mean",
        pool_radius=pool_radius,
    ).vector


# ---------------------------------------------------------------------------
# edit_geometry


def test_edit_geometry_known_vectors() -> None:
    identical = ers.edit_geometry((3.0, 4.0, 0.0), (3.0, 4.0, 0.0))
    assert identical.d_state == 3
    assert identical.norm_s_ref == pytest.approx(5.0)
    assert identical.cos_ref_alt == pytest.approx(1.0)
    assert identical.l2_delta == pytest.approx(0.0)
    assert identical.rel_delta == pytest.approx(0.0)

    rotated = ers.edit_geometry((3.0, 4.0, 0.0), (0.0, 4.0, 3.0))
    assert rotated.norm_s_ref == pytest.approx(5.0)
    assert rotated.norm_s_alt == pytest.approx(5.0)
    assert rotated.cos_ref_alt == pytest.approx(16.0 / 25.0)
    assert rotated.l2_delta == pytest.approx(math.sqrt(18.0))
    assert rotated.rel_delta == pytest.approx(math.sqrt(18.0) / 5.0)


def test_edit_geometry_rejects_zero_norm() -> None:
    with pytest.raises(ers.InputError, match="zero L2 norm"):
        ers.edit_geometry((0.0, 0.0), (1.0, 1.0))


def test_edit_geometry_rejects_width_mismatch() -> None:
    with pytest.raises(ers.InputError, match="share a width"):
        ers.edit_geometry((1.0, 2.0), (1.0, 2.0, 3.0))


# ---------------------------------------------------------------------------
# load_variants


def test_load_variants_parses_required_and_optional(tmp_path: Path) -> None:
    path = tmp_path / "variants.jsonl"
    path.write_text(
        "\n".join(
            [
                json.dumps({"chrom": "chr1", "pos": 10, "ref": "a", "alt": "c", "gene": "BRCA2"}),
                "# a comment line",
                json.dumps(
                    {
                        "chrom": "chr1",
                        "pos": 50,
                        "ref": "A",
                        "alt": "G",
                        "variant_id": "custom-id",
                        "continuous_score": 0.25,
                    }
                ),
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    records = ers.load_variants(path)

    assert len(records) == 2
    assert records[0].variant_id == "chr1:10:A:C"  # derived, uppercased
    assert records[0].ref == "A"
    assert records[0].gene == "BRCA2"
    assert records[1].variant_id == "custom-id"
    assert records[1].continuous_score == pytest.approx(0.25)


def test_load_variants_rejects_missing_field(tmp_path: Path) -> None:
    path = tmp_path / "bad.jsonl"
    path.write_text(json.dumps({"chrom": "chr1", "pos": 10, "ref": "A"}) + "\n", encoding="utf-8")

    with pytest.raises(ers.InputError, match="missing a required field"):
        ers.load_variants(path)


def test_load_variants_rejects_malformed_json(tmp_path: Path) -> None:
    path = tmp_path / "bad.jsonl"
    path.write_text("{not json}\n", encoding="utf-8")

    with pytest.raises(ers.InputError, match="not valid JSON"):
        ers.load_variants(path)


# ---------------------------------------------------------------------------
# run_spectroscopy


def test_run_spectroscopy_rows_counts_and_geometry() -> None:
    variants = _variants()
    variants.insert(1, _skip_variant())
    reference = _reference_sequences()
    encoder = FakeStateEncoder()
    radii = (0, 8, 64)

    run = ers.run_spectroscopy(
        variants,
        reference,
        encoder,
        window_bp=_WINDOW_BP,
        pool_radii=radii,
        batch_size=2,
    )

    assert run.n_input == 4
    assert run.processed_variant_ids == ("chr1:10:A:C", "chr1:50:A:G", "chr1:90:A:C")
    assert len(run.skips) == 1
    assert run.skips[0].variant_id == "chr1:15:T:C"
    assert run.skips[0].code == "INPUT.VCF_PARSE"
    assert len(run.rows) == 9  # 3 variants x 3 radii

    # Independently recompute the geometry for one processed variant.
    prepared = ers.prepare_variant(_variants()[0], reference, window_bp=_WINDOW_BP)
    ref_states = encoder.states_for(prepared.reference_window, prepared.rel_pos)
    alt_states = encoder.states_for(prepared.edited_window, prepared.rel_pos)
    rows_v1 = {row["pool_radius"]: row for row in run.rows if row["variant_id"] == "chr1:10:A:C"}
    for pool_radius in radii:
        expected_ref = _expected_vector(ref_states, prepared.rel_pos, pool_radius)
        expected_alt = _expected_vector(alt_states, prepared.rel_pos, pool_radius)
        expected = ers.edit_geometry(expected_ref, expected_alt)
        row = rows_v1[pool_radius]
        assert row["cos_ref_alt"] == pytest.approx(expected.cos_ref_alt)
        assert row["l2_delta"] == pytest.approx(expected.l2_delta)
        assert row["rel_delta"] == pytest.approx(expected.l2_delta / expected.norm_s_ref)
        assert row["norm_s_ref"] == pytest.approx(expected.norm_s_ref)
        assert row["s_ref"] == [pytest.approx(v) for v in expected_ref]
        assert row["s_alt"] == [pytest.approx(v) for v in expected_alt]


def test_run_spectroscopy_radius_zero_uses_global_mean() -> None:
    reference = _reference_sequences()
    encoder = FakeStateEncoder()
    run = ers.run_spectroscopy(
        _variants(),
        reference,
        encoder,
        window_bp=_WINDOW_BP,
        pool_radii=(0, 8),
        batch_size=8,
    )

    rows = {(row["variant_id"], row["pool_radius"]): row for row in run.rows}
    global_row = rows[("chr1:10:A:C", 0)]
    centered_row = rows[("chr1:10:A:C", 8)]

    assert global_row["pool_type"] == "global_mean"
    assert centered_row["pool_type"] == "centered_mean"

    prepared = ers.prepare_variant(_variants()[0], reference, window_bp=_WINDOW_BP)
    ref_states = encoder.states_for(prepared.reference_window, prepared.rel_pos)
    global_expected = pool_hidden_states(
        ref_states.rows, pool_type="global_mean", pool_radius=0
    ).vector
    centered_expected = pool_hidden_states(
        ref_states.rows,
        edit_locus=prepared.rel_pos,
        center_token=ref_states.center_token,
        content_token_bounds=ref_states.content_token_bounds,
        pool_type="centered_mean",
        pool_radius=8,
    ).vector

    assert global_row["s_ref"] == [pytest.approx(v) for v in global_expected]
    assert centered_row["s_ref"] == [pytest.approx(v) for v in centered_expected]
    # global mean over five tokens differs from the centred content-token mean.
    assert global_row["s_ref"] != [pytest.approx(v) for v in centered_expected]


def test_run_spectroscopy_batches_one_forward_per_chunk() -> None:
    encoder = FakeStateEncoder()
    ers.run_spectroscopy(
        _variants(),
        _reference_sequences(),
        encoder,
        window_bp=_WINDOW_BP,
        pool_radii=(0, 8, 64, 256),
        batch_size=2,
    )
    # 3 prepared variants in chunks of 2 -> two forward calls, one per chunk,
    # each carrying 2 windows per variant. Radius count does not add forwards.
    assert encoder.batch_sizes == [4, 2]


def test_run_spectroscopy_resume_skips_done_ids() -> None:
    reference = _reference_sequences()
    encoder = FakeStateEncoder()
    run = ers.run_spectroscopy(
        _variants(),
        reference,
        encoder,
        window_bp=_WINDOW_BP,
        pool_radii=(0, 8),
        resume_variant_ids={"chr1:10:A:C", "chr1:50:A:G"},
    )

    assert run.processed_variant_ids == ("chr1:90:A:C",)
    assert {row["variant_id"] for row in run.rows} == {"chr1:90:A:C"}


# ---------------------------------------------------------------------------
# Parquet I/O


def _read_table(path: Path) -> Any:
    pq = cast(Any, importlib.import_module("pyarrow.parquet"))
    return pq.read_table(path)


def test_write_embeddings_parquet_schema(tmp_path: Path) -> None:
    encoder = FakeStateEncoder()
    run = ers.run_spectroscopy(
        _variants(),
        _reference_sequences(),
        encoder,
        window_bp=_WINDOW_BP,
        pool_radii=(0, 8, 64),
    )
    out = tmp_path / "embeddings.parquet"
    ers.write_embeddings_parquet(out, run.rows)

    pa = cast(Any, importlib.import_module("pyarrow"))
    table = _read_table(out)
    assert list(table.schema.names) == list(ers.EMBEDDING_COLUMNS)
    assert table.num_rows == 9
    s_ref_type = table.schema.field("s_ref").type
    assert pa.types.is_list(s_ref_type)
    assert pa.types.is_float32(s_ref_type.value_type)  # raw pooled vectors are fp32
    assert pa.types.is_float64(table.schema.field("norm_s_ref").type)
    assert pa.types.is_int64(table.schema.field("pool_radius").type)


def test_read_done_variant_ids_roundtrip(tmp_path: Path) -> None:
    encoder = FakeStateEncoder()
    run = ers.run_spectroscopy(
        _variants(),
        _reference_sequences(),
        encoder,
        window_bp=_WINDOW_BP,
        pool_radii=(0,),
    )
    out = tmp_path / "embeddings.parquet"
    ers.write_embeddings_parquet(out, run.rows)

    assert ers.read_done_variant_ids(out) == {
        "chr1:10:A:C",
        "chr1:50:A:G",
        "chr1:90:A:C",
    }
    assert ers.read_done_variant_ids(tmp_path / "missing.parquet") == frozenset()


def test_write_embeddings_parquet_resume_appends(tmp_path: Path) -> None:
    reference = _reference_sequences()
    out = tmp_path / "embeddings.parquet"

    first = ers.run_spectroscopy(
        _variants()[:2],
        reference,
        FakeStateEncoder(),
        window_bp=_WINDOW_BP,
        pool_radii=(0, 8),
    )
    ers.write_embeddings_parquet(out, first.rows)
    done = ers.read_done_variant_ids(out)

    second = ers.run_spectroscopy(
        _variants(),
        reference,
        FakeStateEncoder(),
        window_bp=_WINDOW_BP,
        pool_radii=(0, 8),
        resume_variant_ids=done,
    )
    assert second.processed_variant_ids == ("chr1:90:A:C",)
    ers.write_embeddings_parquet(out, second.rows, resume=True)

    table = _read_table(out)
    ids = set(table.column("variant_id").to_pylist())
    assert ids == {"chr1:10:A:C", "chr1:50:A:G", "chr1:90:A:C"}
    assert table.num_rows == 6  # 3 variants x 2 radii


# ---------------------------------------------------------------------------
# Summary


def test_build_summary_aggregates(tmp_path: Path) -> None:
    reference = _reference_sequences()
    encoder = FakeStateEncoder()
    radii = (0, 8)
    run = ers.run_spectroscopy(
        _variants(),
        reference,
        encoder,
        window_bp=_WINDOW_BP,
        pool_radii=radii,
    )
    scalar_rows = [
        {
            "variant_id": row["variant_id"],
            "pool_radius": row["pool_radius"],
            "pool_type": row["pool_type"],
            "label_group": row["label_group"],
            "cos_ref_alt": row["cos_ref_alt"],
            "rel_delta": row["rel_delta"],
            "norm_s_ref": row["norm_s_ref"],
        }
        for row in run.rows
    ]

    summary = ers.build_summary(
        scalar_rows,
        run.skips,
        config={"window_bp": _WINDOW_BP, "pool_radii": list(radii)},
        provenance={"git_commit": "deadbeef", "encoder": {"model_id": "fake"}},
        n_input=run.n_input,
    )

    assert summary["schema_version"] == ers.SCHEMA_VERSION
    assert summary["generated_by"] == ers.GENERATED_BY
    assert summary["config"]["window_bp"] == _WINDOW_BP
    assert summary["provenance"]["git_commit"] == "deadbeef"
    assert summary["counts"] == {
        "n_input": 3,
        "n_variants": 3,
        "n_rows": 6,
        "n_skipped": 0,
    }

    per_radius = {entry["pool_radius"]: entry for entry in summary["per_pool_radius"]}
    assert set(per_radius) == {0, 8}
    radius_zero = per_radius[0]
    assert radius_zero["pool_type"] == "global_mean"
    assert radius_zero["n_variants"] == 3
    assert radius_zero["n_by_label_group"] == {"clinvar": 2, "gnomad": 1}

    # Independently compute the mean cosine at radius 0.
    cos_values = [row["cos_ref_alt"] for row in scalar_rows if row["pool_radius"] == 0]
    assert radius_zero["cos_ref_alt"]["mean"] == pytest.approx(sum(cos_values) / len(cos_values))
    assert radius_zero["cos_ref_alt"]["median"] is not None
    assert radius_zero["cos_ref_alt"]["std"] is not None


def test_build_summary_records_skip_reasons() -> None:
    skips = (
        ers.SkipRecord("chr1:15:T:C", "INPUT.VCF_PARSE", "reference FASTA bases do not match"),
        ers.SkipRecord("chr1:20:A:T", "INPUT.OUT_OF_WINDOW", "edit falls outside the window"),
    )
    summary = ers.build_summary(
        [],
        skips,
        config={},
        provenance={},
        n_input=2,
    )
    assert summary["counts"] == {"n_input": 2, "n_variants": 0, "n_rows": 0, "n_skipped": 2}
    assert summary["skips"]["by_code"] == {
        "INPUT.OUT_OF_WINDOW": 1,
        "INPUT.VCF_PARSE": 1,
    }
    assert summary["skips"]["records"][0]["variant_id"] == "chr1:15:T:C"


# ---------------------------------------------------------------------------
# Validation


def test_run_spectroscopy_rejects_unsupported_window_bp() -> None:
    with pytest.raises(ers.InputError, match="unsupported window length"):
        ers.run_spectroscopy(
            _variants(),
            _reference_sequences(),
            FakeStateEncoder(),
            window_bp=1024,
            pool_radii=(0, 8),
        )


def test_run_spectroscopy_rejects_negative_radius() -> None:
    with pytest.raises(ers.InputError, match="non-negative"):
        ers.run_spectroscopy(
            _variants(),
            _reference_sequences(),
            FakeStateEncoder(),
            window_bp=_WINDOW_BP,
            pool_radii=(-1,),
        )


# ---------------------------------------------------------------------------
# CarbonTokenStateEncoder <-> CarbonStateEncoder equivalence.
#
# The R1 dataset is published and the paper cites its numbers, so the tool's
# token-state path must reproduce the frozen encoder's pooled states exactly.
# These fakes are Carbon-shaped (6bp/token, one <dna> control pair, right
# padding, layered hidden states) so both paths exercise the real tokenize ->
# layout -> forward -> pool code in geno_lewm.encoder.carbon.


_KMER_BASES = "ACGT"


def _kmer_index(kmer: str) -> int:
    index = 0
    for base in kmer:
        index = index * 4 + _KMER_BASES.index(base)
    return index


class _CarbonShapedTokenizer:
    """Minimal Carbon-shaped tokenizer: 6bp k-mers wrapped in a control pair."""

    k = 6
    dna_start_id = 151_669
    dna_vocab_size = 4_107
    dna_begin_token_id = 151_669
    dna_end_token_id = 151_670
    oov_token_id = 151_671
    pad_token_id = 151_643

    def __call__(
        self,
        texts: list[str],
        *,
        return_tensors: str,
        padding: bool,
        add_special_tokens: bool,
    ) -> dict[str, list[list[int]]]:
        assert return_tensors == "pt"
        assert padding is True
        assert add_special_tokens is False
        rows: list[list[int]] = []
        for text in texts:
            dna = text.removeprefix("<dna>").removesuffix("</dna>")
            n_content = (len(dna) + self.k - 1) // self.k
            ids = [self.dna_begin_token_id]
            for i in range(n_content):
                kmer = dna[i * self.k : (i + 1) * self.k].ljust(self.k, "A")
                ids.append(self.dna_start_id + 3 + _kmer_index(kmer))
            ids.append(self.dna_end_token_id)
            rows.append(ids)
        width = max(len(row) for row in rows)
        return {
            "input_ids": [row + [self.pad_token_id] * (width - len(row)) for row in rows],
            "attention_mask": [[1] * len(row) + [0] * (width - len(row)) for row in rows],
        }


class _CarbonShapedModel:
    """Deterministic layered hidden states keyed on token id and position."""

    config = SimpleNamespace(hidden_size=3)
    n_layers = 3

    def __init__(self) -> None:
        self.forward_calls = 0

    def eval(self) -> _CarbonShapedModel:
        return self

    def __call__(
        self,
        *,
        input_ids: list[list[int]],
        attention_mask: list[list[int]],
        output_hidden_states: bool,
    ) -> object:
        assert output_hidden_states is True
        self.forward_calls += 1
        return SimpleNamespace(
            hidden_states=tuple(
                tuple(self._rows(ids, layer) for ids in input_ids) for layer in range(self.n_layers)
            )
        )

    @staticmethod
    def _rows(ids: list[int], layer: int) -> tuple[tuple[float, ...], ...]:
        return tuple(
            (
                float(token_id % 97) + float(pos) + 100.0 * layer,
                float(token_id % 89) - 2.0 * float(pos) + 10.0 * layer,
                float(pos * pos) - float(token_id % 83) + float(layer),
            )
            for pos, token_id in enumerate(ids)
        )


#: Two windows of different length so the batch is genuinely right-padded.
_EQUIV_WINDOWS = ["ACGTAC" * 6, "TTGGCA" * 6 + "AC"]
_EQUIV_LOCI: list[int | None] = [7, 20]
_EQUIV_STATE_LAYER = -1


def _carbon_encoder(*, pool_type: str, pool_radius: int) -> Any:
    from geno_lewm.encoder.carbon import CarbonStateEncoder

    return CarbonStateEncoder(
        "HuggingFaceBio/Carbon-500M",
        "main@deadbeef",
        model=_CarbonShapedModel(),
        tokenizer=_CarbonShapedTokenizer(),
        state_layer=_EQUIV_STATE_LAYER,
        pool_type=pool_type,
        pool_radius=pool_radius,
        normalize=False,
    )


@pytest.mark.parametrize("pool_radius", [1, 2, 64])
def test_token_state_encoder_pools_identically_to_encode_batch(pool_radius: int) -> None:
    """Centred pooling over the tool's token states == the encoder's own pooling.

    Exact equality, not allclose: both paths feed the same float rows through
    the same ``pool_hidden_states``, so any difference would mean the token
    states or the pooling anchors drifted -- which would silently invalidate the
    published R1 table.
    """
    batched = _carbon_encoder(pool_type="centered_mean", pool_radius=pool_radius).encode_batch(
        _EQUIV_WINDOWS, _EQUIV_LOCI
    )

    token_encoder = ers.CarbonTokenStateEncoder(
        "HuggingFaceBio/Carbon-500M",
        "main@deadbeef",
        state_layer=_EQUIV_STATE_LAYER,
        encoder=_carbon_encoder(pool_type="centered_mean", pool_radius=0),
    )
    states = token_encoder.encode_token_states(_EQUIV_WINDOWS, _EQUIV_LOCI)
    manual = tuple(
        pool_hidden_states(
            item.rows,
            edit_locus=locus,
            center_token=item.center_token,
            content_token_bounds=item.content_token_bounds,
            pool_type="centered_mean",
            pool_radius=pool_radius,
        ).vector
        for item, locus in zip(states, _EQUIV_LOCI, strict=True)
    )

    assert manual == batched


def test_token_state_encoder_global_mean_matches_encode_batch() -> None:
    """Radius 0 (the tool's global-mean rung) also reproduces the encoder.

    The encoder is passed ``None`` loci where the tool is passed real ones:
    ``encode_batch`` always derives a ``center_token`` from a non-None locus and
    ``pool_hidden_states`` rejects that under ``global_mean``, so ``None`` is the
    only way to ask the encoder for a global mean. The tool's radius-0 rung
    ignores the locus instead. Equality here is therefore the point, not an
    accident: it shows the locus cannot leak into the global-mean rung.
    """
    batched = _carbon_encoder(pool_type="global_mean", pool_radius=0).encode_batch(
        _EQUIV_WINDOWS, [None, None]
    )

    token_encoder = ers.CarbonTokenStateEncoder(
        "HuggingFaceBio/Carbon-500M",
        "main@deadbeef",
        state_layer=_EQUIV_STATE_LAYER,
        encoder=_carbon_encoder(pool_type="centered_mean", pool_radius=0),
    )
    states = token_encoder.encode_token_states(_EQUIV_WINDOWS, _EQUIV_LOCI)
    manual = tuple(
        ers._pool_state(item, edit_locus=cast(int, locus), pool_radius=0)
        for item, locus in zip(states, _EQUIV_LOCI, strict=True)
    )

    assert manual == batched


def test_token_state_encoder_serves_every_radius_from_one_forward() -> None:
    """The whole point: one forward, many radii."""
    encoder = _carbon_encoder(pool_type="centered_mean", pool_radius=0)
    model = cast(_CarbonShapedModel, encoder.model)
    token_encoder = ers.CarbonTokenStateEncoder(
        "HuggingFaceBio/Carbon-500M",
        "main@deadbeef",
        state_layer=_EQUIV_STATE_LAYER,
        encoder=encoder,
    )

    (item, _alt) = token_encoder.encode_token_states(_EQUIV_WINDOWS, _EQUIV_LOCI)
    pooled = [ers._pool_state(item, edit_locus=7, pool_radius=radius) for radius in (0, 1, 2, 64)]

    assert model.forward_calls == 1
    assert token_encoder.d_state == 3
    # Radius 0 pools the control tokens too, so it must differ from radius 1.
    assert pooled[0] != pooled[1]


def test_token_state_encoder_excludes_padding_rows() -> None:
    """Active rows only: the shorter window must not carry the batch's padding."""
    token_encoder = ers.CarbonTokenStateEncoder(
        "HuggingFaceBio/Carbon-500M",
        "main@deadbeef",
        state_layer=_EQUIV_STATE_LAYER,
        encoder=_carbon_encoder(pool_type="centered_mean", pool_radius=0),
    )

    short, long = token_encoder.encode_token_states(_EQUIV_WINDOWS, _EQUIV_LOCI)

    # 36bp -> 6 content tokens + 2 control; 38bp -> 7 content tokens + 2.
    assert len(short.rows) == 8
    assert len(long.rows) == 9
    assert short.content_token_bounds == (1, 7)
    assert long.content_token_bounds == (1, 8)
    assert short.center_token == 1 + 7 // 6
    assert long.center_token == 1 + 20 // 6


def test_token_state_encoder_d_state_is_none_before_any_forward() -> None:
    token_encoder = ers.CarbonTokenStateEncoder(
        "HuggingFaceBio/Carbon-500M",
        "main@deadbeef",
        encoder=_carbon_encoder(pool_type="centered_mean", pool_radius=0),
    )

    assert token_encoder.d_state is None


def test_token_state_encoder_rejects_mismatched_lengths() -> None:
    token_encoder = ers.CarbonTokenStateEncoder(
        "HuggingFaceBio/Carbon-500M",
        "main@deadbeef",
        encoder=_carbon_encoder(pool_type="centered_mean", pool_radius=0),
    )

    with pytest.raises(ers.InputError, match="same length"):
        token_encoder.encode_token_states(["ACGTAC"], [0, 1])
