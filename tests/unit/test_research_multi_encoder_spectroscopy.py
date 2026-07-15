"""Tests for the R6 multi-encoder edit-response spectroscopy tool.

The scientific risk this module guards is the locus -> token mapping. Every
genomic encoder frames DNA differently (Nucleotide Transformer packs 6-mers
behind a ``<cls>``; HyenaDNA is char-level with a trailing ``[SEP]``), and a
mapping that is merely *plausible* pools the wrong tokens and yields
confident nonsense. So the mapping is pinned here against token layouts
recorded from the real tokenizers, using fake tokenizer/model objects so the
suite never touches the network.

The module is skipped cleanly when NumPy / PyArrow are unavailable, matching
the repository's optional-dependency test convention.
"""

from __future__ import annotations

import json
from itertools import pairwise
from pathlib import Path
from typing import Any, ClassVar

import pytest

from geno_lewm.errors import InputError
from tools.research.multi_encoder_spectroscopy import (
    EMBEDDING_COLUMNS,
    ENCODER_REGISTRY,
    EncoderSpec,
    HuggingFaceStateEncoder,
    derive_token_layout,
    main,
    run_multi_encoder_spectroscopy,
)

np = pytest.importorskip("numpy")
pytest.importorskip("pyarrow")

_NT_SPECIALS = ("<unk>", "<pad>", "<cls>", "<mask>")
_HYENA_SPECIALS = ("[BOS]", "[SEP]", "[UNK]", "[PAD]", "[CLS]", "[MASK]")


# ---------------------------------------------------------------------------
# Fake tokenizers reproducing the real token layouts (verified against the
# actual hub tokenizers; see the tool docstring for the recorded outputs).


class _FakeNTTokenizer:
    """Mimic ``EsmTokenizer`` for nucleotide-transformer-v2.

    Real behaviour, verified on the hub tokenizer: ``<cls>`` is prepended,
    the sequence is consumed greedily as 6-mers from the left, and any
    residue that cannot form a known 6-mer is emitted as single-nucleotide
    tokens. A 6-mer containing ``N`` is not in the vocabulary, so it degrades
    to single characters and *re-frames* the 6-mer grid after it.
    """

    is_fast = False
    all_special_tokens: ClassVar[tuple[str, ...]] = _NT_SPECIALS
    unk_token = "<unk>"
    pad_token_id = 1

    def tokenize_window(self, sequence: str) -> list[str]:
        tokens = ["<cls>"]
        i = 0
        while i < len(sequence):
            chunk = sequence[i : i + 6]
            if len(chunk) == 6 and "N" not in chunk:
                tokens.append(chunk)
                i += 6
                continue
            # Emit single characters until the frame can resync on a clean 6-mer.
            tokens.append(sequence[i])
            i += 1
        return tokens


class _FakeHyenaTokenizer:
    """Mimic ``HyenaDNATokenizer``: char-level, no CLS, trailing ``[SEP]``."""

    is_fast = False
    all_special_tokens: ClassVar[tuple[str, ...]] = _HYENA_SPECIALS
    unk_token = "[UNK]"
    pad_token_id = 4

    def tokenize_window(self, sequence: str) -> list[str]:
        return [*list(sequence), "[SEP]"]


# ---------------------------------------------------------------------------
# Locus -> token mapping: the highest-risk contract.


def _layout(tokenizer: Any, sequence: str) -> Any:
    return derive_token_layout(
        tokenizer.tokenize_window(sequence),
        special_tokens=frozenset(tokenizer.all_special_tokens),
        sequence_bp=len(sequence),
    )


class TestNucleotideTransformerLayout:
    def test_clean_sequence_maps_locus_through_the_cls_offset(self) -> None:
        seq = "ACGTAC" * 4  # 24bp -> <cls> + 4 six-mers
        layout = _layout(_FakeNTTokenizer(), seq)
        assert layout.content_bounds == (1, 5)
        # bp 0..5 -> token 1, bp 6..11 -> token 2, ... (the <cls> shifts by one)
        for locus, expected in [(0, 1), (5, 1), (6, 2), (11, 2), (12, 3), (23, 4)]:
            assert layout.center_token(locus) == expected

    def test_n_base_reframes_the_six_mer_grid(self) -> None:
        # Verified on the real tokenizer: "ACGTACACGNACACGTACACGTAC" ->
        # ['<cls>', 'ACGTAC', 'A', 'C', 'G', 'N', 'ACACGT', 'ACACGT', 'A', 'C']
        seq = "ACGTAC" + "ACGNAC" + "ACGTAC" * 2
        layout = _layout(_FakeNTTokenizer(), seq)
        # A naive locus//6 + 1 mapping would send bp 12 to token 3; the real
        # frame re-anchors after the N, so bp 12 lives inside token 6.
        assert layout.center_token(12) != 12 // 6 + 1
        assert layout.center_token(12) == 6
        assert layout.center_token(9) == 5  # the N itself

    def test_spans_tile_the_sequence_exactly(self) -> None:
        seq = "ACGTACACGNACACGTACACGTAC"
        layout = _layout(_FakeNTTokenizer(), seq)
        covered = [span for span in layout.spans if span is not None]
        assert covered[0][0] == 0
        assert covered[-1][1] == len(seq)
        for left, right in pairwise(covered):
            assert left[1] == right[0]


class TestHyenaDNALayout:
    def test_char_level_maps_locus_to_itself(self) -> None:
        seq = "ACGTACGT"
        layout = _layout(_FakeHyenaTokenizer(), seq)
        # No CLS: bp i is token i. The trailing [SEP] is excluded from content.
        assert layout.content_bounds == (0, 8)
        for locus in range(len(seq)):
            assert layout.center_token(locus) == locus

    def test_n_base_does_not_shift_the_frame(self) -> None:
        seq = "ACGNACGT"
        layout = _layout(_FakeHyenaTokenizer(), seq)
        for locus in range(len(seq)):
            assert layout.center_token(locus) == locus


class TestLayoutFailsClosed:
    def test_unconsumed_bases_are_rejected(self) -> None:
        # A tokenizer whose tokens do not tile the sequence must not be
        # silently trusted: that is the silent-garbage failure mode.
        with pytest.raises(InputError, match="does not tile"):
            derive_token_layout(
                ("<cls>", "ACGTAC"), special_tokens=frozenset({"<cls>"}), sequence_bp=24
            )

    def test_unk_token_is_rejected(self) -> None:
        # <unk> consumes an unknown number of bases, so the mapping is
        # unverifiable and the variant must be skipped rather than guessed.
        with pytest.raises(InputError, match="does not tile"):
            derive_token_layout(
                ("<cls>", "<unk>"),
                special_tokens=frozenset({"<cls>", "<unk>"}),
                sequence_bp=24,
            )

    def test_non_contiguous_content_is_rejected(self) -> None:
        with pytest.raises(InputError, match="contiguous"):
            derive_token_layout(
                ("ACG", "<cls>", "TAC"),
                special_tokens=frozenset({"<cls>"}),
                sequence_bp=6,
            )

    def test_out_of_range_locus_is_rejected(self) -> None:
        layout = _layout(_FakeHyenaTokenizer(), "ACGT")
        with pytest.raises(InputError, match="outside"):
            layout.center_token(4)


# ---------------------------------------------------------------------------
# Fake model: hidden states are a deterministic function of token ids so the
# pooling/geometry path is exercised without loading a real encoder.

_D_STATE = 8


class _FakeModel:
    """Return hidden states with a known, position-dependent structure."""

    def __init__(self, n_layers: int = 3) -> None:
        self._n_layers = n_layers
        self.seen_batch_shapes: list[tuple[int, ...]] = []

    def eval(self) -> _FakeModel:
        return self

    def to(self, device: str) -> _FakeModel:  # pragma: no cover - device is a no-op
        return self

    def __call__(self, **kwargs: Any) -> Any:
        input_ids = kwargs["input_ids"]
        self.seen_batch_shapes.append(tuple(input_ids.shape))
        torch = pytest.importorskip("torch")
        layers = []
        for layer in range(self._n_layers):
            base = input_ids.to(torch.float32).unsqueeze(-1)
            offsets = torch.arange(_D_STATE, dtype=torch.float32)
            rows = base + offsets + float(layer)
            layers.append(rows)
        return type("Out", (), {"hidden_states": tuple(layers)})()


class _FakeTokenizerAdapter(_FakeHyenaTokenizer):
    """Char-level tokenizer exposing the HF call surface the tool needs."""

    def __call__(self, sequences: Any, **kwargs: Any) -> Any:
        torch = pytest.importorskip("torch")
        vocab = {"A": 7, "C": 8, "G": 9, "T": 10, "N": 11, "[SEP]": 1}
        rows = [[vocab[t] for t in self.tokenize_window(s)] for s in sequences]
        return {"input_ids": torch.tensor(rows, dtype=torch.long)}

    def convert_ids_to_tokens(self, ids: Any) -> list[str]:
        inv = {7: "A", 8: "C", 9: "G", 10: "T", 11: "N", 1: "[SEP]"}
        return [inv[int(i)] for i in ids]


def _spec() -> EncoderSpec:
    return EncoderSpec(
        encoder_id="fake-char",
        model_id="fake/char",
        revision="0" * 40,
        auto_class="auto_model",
        trust_remote_code=False,
        state_layer=-1,
    )


def _reference() -> dict[str, str]:
    # Deterministic 16384bp contig so a 4096bp window resolves around pos.
    return {"1": "ACGT" * 4096}


def _variants_jsonl(tmp_path: Path) -> Path:
    rows = [
        {
            "chrom": "1",
            "pos": 8193,
            "ref": "A",
            "alt": "G",
            "label": "clinvar_path",
            "label_group": "clinvar_path",
            "variant_id": "v1",
        },
        {
            "chrom": "1",
            "pos": 8197,
            "ref": "A",
            "alt": "T",
            "label": "clinvar_benign",
            "label_group": "clinvar_benign",
            "variant_id": "v2",
        },
        # Non-SNV: must be counted and dropped, not silently encoded.
        {
            "chrom": "1",
            "pos": 8201,
            "ref": "AC",
            "alt": "A",
            "label": "clinvar_path",
            "label_group": "clinvar_path",
            "variant_id": "v3-indel",
        },
    ]
    path = tmp_path / "variants.jsonl"
    path.write_text("\n".join(json.dumps(r, sort_keys=True) for r in rows) + "\n")
    return path


class TestRunMultiEncoderSpectroscopy:
    def _encoder(self) -> HuggingFaceStateEncoder:
        pytest.importorskip("torch")
        return HuggingFaceStateEncoder(
            _spec(),
            model=_FakeModel(),
            tokenizer=_FakeTokenizerAdapter(),
        )

    def test_emits_the_carbon_schema_plus_encoder_id(self, tmp_path: Path) -> None:
        from tools.research.edit_response_spectroscopy import load_variants

        variants = load_variants(_variants_jsonl(tmp_path))
        run = run_multi_encoder_spectroscopy(
            variants,
            _reference(),
            self._encoder(),
            encoder_id="fake-char",
            window_bp=4096,
            pool_radii=(0, 8),
        )
        assert run.n_non_snv == 1
        assert {row["variant_id"] for row in run.rows} == {"v1", "v2"}
        for row in run.rows:
            assert set(row) == set(EMBEDDING_COLUMNS)
            assert row["encoder_id"] == "fake-char"
            assert len(row["s_ref"]) == _D_STATE
            assert row["d_state"] == _D_STATE

    def test_pool_radius_zero_is_a_global_mean_over_all_tokens(self, tmp_path: Path) -> None:
        from tools.research.edit_response_spectroscopy import load_variants

        variants = load_variants(_variants_jsonl(tmp_path))
        run = run_multi_encoder_spectroscopy(
            variants,
            _reference(),
            self._encoder(),
            encoder_id="fake-char",
            window_bp=4096,
            pool_radii=(0,),
        )
        row = next(r for r in run.rows if r["variant_id"] == "v1")
        assert row["pool_type"] == "global_mean"
        # The fake model's states are id + offset + layer, so the global mean
        # is analytically the mean token id plus the offset and layer.
        assert row["l2_delta"] > 0.0

    def test_single_base_edit_moves_only_the_edited_token_at_small_radius(
        self, tmp_path: Path
    ) -> None:
        from tools.research.edit_response_spectroscopy import load_variants

        variants = load_variants(_variants_jsonl(tmp_path))
        run = run_multi_encoder_spectroscopy(
            variants,
            _reference(),
            self._encoder(),
            encoder_id="fake-char",
            window_bp=4096,
            pool_radii=(0, 8),
        )
        rows = {r["pool_radius"]: r for r in run.rows if r["variant_id"] == "v1"}
        # A tighter pool concentrates the single-token change, so the relative
        # displacement must exceed the whole-window mean's.
        assert rows[8]["rel_delta"] > rows[0]["rel_delta"]

    def test_cli_writes_parquet_and_summary(self, tmp_path: Path, monkeypatch: Any) -> None:
        pq = pytest.importorskip("pyarrow.parquet")
        variants = _variants_jsonl(tmp_path)
        fasta = tmp_path / "ref.fa"
        fasta.write_text(">1\n" + "ACGT" * 4096 + "\n")
        emb = tmp_path / "emb.parquet"
        summary = tmp_path / "summary.json"

        monkeypatch.setattr(
            "tools.research.multi_encoder_spectroscopy._build_encoder",
            lambda spec, args: self._encoder(),
        )
        code = main(
            [
                "--variants",
                str(variants),
                "--reference-fasta",
                str(fasta),
                "--out-embeddings",
                str(emb),
                "--out-summary",
                str(summary),
                "--encoder",
                "hyenadna-medium-450k",
                "--window-bp",
                "4096",
                "--pool-radii",
                "0,8",
            ]
        )
        assert code == 0
        table = pq.read_table(emb)
        assert set(table.column_names) == set(EMBEDDING_COLUMNS)
        payload = json.loads(summary.read_text())
        assert payload["counts"]["n_non_snv"] == 1
        assert payload["provenance"]["encoders"][0]["encoder_id"] == "hyenadna-medium-450k"
        assert payload["encoder_id"] == "hyenadna-medium-450k"


class TestLoadHFComponents:
    """Cover the model-loading recipe with stub modules (never the network)."""

    def _stub_transformers(self, monkeypatch: Any) -> dict[str, Any]:
        import sys
        import types

        seen: dict[str, Any] = {}

        class _Loader:
            def __init__(self, name: str) -> None:
                self._name = name

            def from_pretrained(self, model_id: str, **kwargs: Any) -> str:
                seen[self._name] = {"model_id": model_id, **kwargs}
                return f"{self._name}:{model_id}"

        stub = types.ModuleType("transformers")
        stub.AutoTokenizer = _Loader("tokenizer")  # type: ignore[attr-defined]
        stub.AutoModel = _Loader("auto_model")  # type: ignore[attr-defined]
        stub.AutoModelForMaskedLM = _Loader("masked_lm")  # type: ignore[attr-defined]
        monkeypatch.setitem(sys.modules, "transformers", stub)
        return seen

    def test_masked_lm_spec_loads_via_masked_lm_class(self, monkeypatch: Any) -> None:
        pytest.importorskip("torch")
        from tools.research.multi_encoder_spectroscopy import (
            ENCODER_REGISTRY,
            _load_hf_components,
        )

        seen = self._stub_transformers(monkeypatch)
        spec = ENCODER_REGISTRY["nt-v2-100m-multi"]
        model, tokenizer = _load_hf_components(spec, dtype="fp32")
        assert model.startswith("masked_lm:")
        assert tokenizer.startswith("tokenizer:")
        # The pinned revision and trust_remote_code must reach the hub call.
        assert seen["masked_lm"]["revision"] == spec.revision
        assert seen["masked_lm"]["trust_remote_code"] is True

    def test_auto_model_spec_loads_via_auto_model_class(self, monkeypatch: Any) -> None:
        pytest.importorskip("torch")
        from tools.research.multi_encoder_spectroscopy import (
            ENCODER_REGISTRY,
            _load_hf_components,
        )

        self._stub_transformers(monkeypatch)
        spec = ENCODER_REGISTRY["hyenadna-medium-450k"]
        model, _tokenizer = _load_hf_components(spec, dtype="bf16")
        assert model.startswith("auto_model:")

    def test_unsupported_dtype_is_rejected(self) -> None:
        from tools.research.multi_encoder_spectroscopy import (
            ENCODER_REGISTRY,
            _load_hf_components,
        )

        with pytest.raises(InputError, match="dtype"):
            _load_hf_components(ENCODER_REGISTRY["hyenadna-medium-450k"], dtype="int4")

    def test_unsupported_auto_class_is_rejected(self, monkeypatch: Any) -> None:
        pytest.importorskip("torch")
        from tools.research.multi_encoder_spectroscopy import _load_hf_components

        self._stub_transformers(monkeypatch)
        spec = EncoderSpec(
            encoder_id="x",
            model_id="x",
            revision=None,
            auto_class="not_a_class",
            trust_remote_code=False,
            state_layer=-1,
        )
        with pytest.raises(InputError, match="auto_class"):
            _load_hf_components(spec, dtype="fp32")


class TestFailureIsolation:
    def test_one_bad_window_does_not_abort_the_batch(self, tmp_path: Path) -> None:
        pytest.importorskip("torch")
        from tools.research.edit_response_spectroscopy import load_variants

        class _PickyEncoder(HuggingFaceStateEncoder):
            """Reject any batch containing v2's window, then succeed alone."""

            def encode_token_states(self, windows: Any, edit_loci: Any) -> Any:
                if len(windows) > 2:
                    raise InputError("batch encode failed")
                return super().encode_token_states(windows, edit_loci)

        encoder = _PickyEncoder(_spec(), model=_FakeModel(), tokenizer=_FakeTokenizerAdapter())
        variants = load_variants(_variants_jsonl(tmp_path))
        run = run_multi_encoder_spectroscopy(
            variants,
            _reference(),
            encoder,
            encoder_id="fake-char",
            window_bp=4096,
            pool_radii=(0,),
            batch_size=8,
        )
        # The batch forward failed, so the retry path encoded each variant
        # alone and both still landed.
        assert {row["variant_id"] for row in run.rows} == {"v1", "v2"}

    def test_encoder_returning_wrong_count_is_skipped(self, tmp_path: Path) -> None:
        pytest.importorskip("torch")
        from tools.research.edit_response_spectroscopy import load_variants

        class _ShortEncoder(HuggingFaceStateEncoder):
            def encode_token_states(self, windows: Any, edit_loci: Any) -> Any:
                return super().encode_token_states(windows, edit_loci)[:-1]

        encoder = _ShortEncoder(_spec(), model=_FakeModel(), tokenizer=_FakeTokenizerAdapter())
        variants = load_variants(_variants_jsonl(tmp_path))
        run = run_multi_encoder_spectroscopy(
            variants,
            _reference(),
            encoder,
            encoder_id="fake-char",
            window_bp=4096,
            pool_radii=(0,),
            batch_size=8,
        )
        assert run.rows == ()
        assert len(run.skips) == 2

    def test_mismatched_windows_and_loci_are_rejected(self) -> None:
        pytest.importorskip("torch")
        encoder = HuggingFaceStateEncoder(
            _spec(), model=_FakeModel(), tokenizer=_FakeTokenizerAdapter()
        )
        with pytest.raises(InputError, match="same length"):
            encoder.encode_token_states(["ACGT", "ACGT"], [0])

    def test_pool_radii_must_be_non_negative_ints(self, tmp_path: Path) -> None:
        pytest.importorskip("torch")
        from tools.research.edit_response_spectroscopy import load_variants

        encoder = HuggingFaceStateEncoder(
            _spec(), model=_FakeModel(), tokenizer=_FakeTokenizerAdapter()
        )
        variants = load_variants(_variants_jsonl(tmp_path))
        with pytest.raises(InputError, match="pool_radii"):
            run_multi_encoder_spectroscopy(
                variants,
                _reference(),
                encoder,
                encoder_id="fake-char",
                window_bp=4096,
                pool_radii=(-1,),
            )
        with pytest.raises(InputError, match="pool_radii"):
            run_multi_encoder_spectroscopy(
                variants,
                _reference(),
                encoder,
                encoder_id="fake-char",
                window_bp=4096,
                pool_radii=(),
            )


class TestResumeAndFailClosed:
    def test_unknown_encoder_id_is_rejected(self, tmp_path: Path) -> None:
        # A typo must not silently fall back to some default checkpoint and
        # mislabel the resulting table.
        code = main(
            [
                "--variants",
                str(_variants_jsonl(tmp_path)),
                "--reference-fasta",
                str(tmp_path / "nope.fa"),
                "--out-embeddings",
                str(tmp_path / "e.parquet"),
                "--out-summary",
                str(tmp_path / "s.json"),
                "--encoder",
                "not-a-real-encoder",
            ]
        )
        assert code != 0

    def test_resume_skips_variants_already_in_the_table(self, tmp_path: Path) -> None:
        pytest.importorskip("torch")
        from tools.research.edit_response_spectroscopy import load_variants
        from tools.research.multi_encoder_spectroscopy import (
            read_done_variant_ids,
            write_embeddings_parquet,
        )

        variants = load_variants(_variants_jsonl(tmp_path))
        encoder = HuggingFaceStateEncoder(
            _spec(), model=_FakeModel(), tokenizer=_FakeTokenizerAdapter()
        )
        first = run_multi_encoder_spectroscopy(
            variants,
            _reference(),
            encoder,
            encoder_id="fake-char",
            window_bp=4096,
            pool_radii=(0,),
        )
        out = tmp_path / "emb.parquet"
        write_embeddings_parquet(out, first.rows)
        assert read_done_variant_ids(out) == {"v1", "v2"}

        second = run_multi_encoder_spectroscopy(
            variants,
            _reference(),
            encoder,
            encoder_id="fake-char",
            window_bp=4096,
            pool_radii=(0,),
            resume_variant_ids=read_done_variant_ids(out),
        )
        assert second.rows == ()

    def test_read_done_variant_ids_on_missing_file_is_empty(self, tmp_path: Path) -> None:
        from tools.research.multi_encoder_spectroscopy import read_done_variant_ids

        assert read_done_variant_ids(tmp_path / "absent.parquet") == frozenset()

    def test_unsupported_window_is_rejected(self, tmp_path: Path) -> None:
        pytest.importorskip("torch")
        from tools.research.edit_response_spectroscopy import load_variants

        variants = load_variants(_variants_jsonl(tmp_path))
        encoder = HuggingFaceStateEncoder(
            _spec(), model=_FakeModel(), tokenizer=_FakeTokenizerAdapter()
        )
        with pytest.raises(InputError, match="window"):
            run_multi_encoder_spectroscopy(
                variants, _reference(), encoder, encoder_id="fake-char", window_bp=777
            )

    def test_state_layer_out_of_range_is_rejected(self) -> None:
        pytest.importorskip("torch")
        from tools.research.multi_encoder_spectroscopy import _resolve_state_layer

        assert _resolve_state_layer(-1, 3) == 2
        assert _resolve_state_layer(0, 3) == 0
        with pytest.raises(InputError, match="state_layer"):
            _resolve_state_layer(9, 3)


class TestEncoderRegistry:
    def test_default_encoders_are_pinned(self) -> None:
        # Unpinned revisions make the run irreproducible; the registry must
        # carry an explicit commit for every default encoder.
        for spec in ENCODER_REGISTRY.values():
            assert spec.revision is not None
            assert len(spec.revision) == 40

    def test_nucleotide_transformer_uses_masked_lm(self) -> None:
        # NT v2's auto_map has no AutoModel entry, so AutoModel silently falls
        # back to stock ESM and shape-mismatches on the gated MLP. Only the
        # masked-LM class reaches the repo's own modeling code.
        spec = ENCODER_REGISTRY["nt-v2-100m-multi"]
        assert spec.auto_class == "masked_lm"
        assert spec.trust_remote_code is True

    def test_hyenadna_uses_auto_model(self) -> None:
        spec = ENCODER_REGISTRY["hyenadna-medium-450k"]
        assert spec.auto_class == "auto_model"
        assert spec.trust_remote_code is True
