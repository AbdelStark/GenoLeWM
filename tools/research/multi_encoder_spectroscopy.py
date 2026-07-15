# SPDX-License-Identifier: Apache-2.0
"""Replicate the edit-response geometry (R1) across non-Carbon genomic encoders.

R1 measured the frozen Carbon-500M encoder and found that a single-base edit
displaces the pooled state ``delta = s(alt) - s(ref)`` informatively (ClinVar
pathogenic-vs-benign AUROC ~0.92) while that displacement is *not* predictable
from ``(pooled reference state, action)``. On one model that is an anecdote.
This tool re-measures the same geometry on other genomic foundation models so
the claim can be stated about the model class rather than one checkpoint.

Output contract
---------------
The emitted Parquet is byte-compatible with
:mod:`tools.research.edit_response_spectroscopy`: identical column names,
order, types and semantics, plus one added ``encoder_id`` column identifying
the producing model. ``tools/research/edit_response_analysis.py`` selects the
columns it needs and documents that extra columns are ignored, so it runs on
this table unchanged. Where a design choice was ambiguous this module always
picks the option that keeps the schema identical to the Carbon tool: geometry
is computed by importing :func:`edit_response_spectroscopy.edit_geometry`
rather than reimplementing it, and ``pool_radius`` stays measured in *tokens*.

One encoder per invocation
--------------------------
``--encoder`` is deliberately singular and each run writes exactly one table.
This is a correctness constraint, not ergonomics: ``edit_response_analysis``
buckets rows by ``pool_radius`` alone and has no notion of ``encoder_id``, so a
table holding two encoders would silently blend their rows — and those rows are
not commensurable (NT is 512-dimensional, HyenaDNA 256, and both reuse the same
``variant_id``). Keeping one encoder per file makes that mixture structurally
impossible while ``encoder_id`` still identifies the producer once the
per-encoder tables are concatenated for cross-encoder analysis.

Why not extend ``CarbonStateEncoder``
-------------------------------------
That class is Carbon-specific: it hard-codes the 6bp-per-token frame via
``CarbonDNATokenizer``, wraps sequences in ``<dna>`` control tokens, and
hard-rejects LoRA. None of that generalises. This module instead drives plain
``AutoTokenizer`` / ``AutoModel`` with ``output_hidden_states=True`` and
derives the token frame from the tokenizer itself.

Locus -> token mapping (the load-bearing part)
---------------------------------------------
Pooling the wrong tokens produces plausible garbage, so the mapping is derived,
never assumed. Both target tokenizers are *slow* (``is_fast == False``), so
``return_offsets_mapping`` is unavailable and the offsets must be reconstructed.
The derivation used here exploits a property both tokenizers share: every
non-special token's *string is literally the DNA it consumed*. So walking the
token list and accumulating ``len(token)`` yields each token's exact base-pair
span, and the reconstruction is then *verified* by asserting the spans tile the
window exactly (:func:`derive_token_layout`). Anything that fails to tile — most
importantly an ``<unk>`` token, whose consumed length is unknowable — fails
closed and the variant is skipped rather than mismapped.

This matters because the obvious arithmetic is wrong. Recorded from the real
``InstaDeepAI/nucleotide-transformer-v2-100m-multi-species`` tokenizer::

    "ACGTACACGNACACGTACACGTAC"
    -> ['<cls>', 'ACGTAC', 'A', 'C', 'G', 'N', 'ACACGT', 'ACACGT', 'A', 'C']

NT packs 6-mers greedily from the left behind a ``<cls>``, but a 6-mer holding
an ``N`` is out of vocabulary, degrades to single-nucleotide tokens, and
*re-anchors the 6-mer grid after it*. A ``locus // 6 + 1`` mapping silently
pools the wrong tokens on every window containing an ``N`` — and reference
windows do contain ``N``. Recorded from ``LongSafari/hyenadna-medium-450k-seqlen-hf``::

    "ACGNAC" -> ['A', 'C', 'G', 'N', 'A', 'C', '[SEP]']

HyenaDNA is char-level with no ``<cls>`` and a trailing ``[SEP]``, so bp ``i``
is token ``i``. The two layouts differ in offset, in tokens-per-base, and in
their response to ``N``, yet the tile-and-verify derivation covers both without
special-casing, which is why one code path is justified here.

Batching
--------
Windows are bucketed by token count and encoded without padding. HyenaDNA's
tokenizer emits no ``attention_mask`` and pads on the *left*, so a padded batch
would both contaminate the states and shift every token index; NT pads right
with a mask. Bucketing sidesteps both hazards for every encoder at once.
Pooling uses a NumPy fast path: the Carbon tool's pure-Python pooling made R1
compute-bound.
"""

from __future__ import annotations

import argparse
import importlib
import json
import subprocess
import sys
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final, cast

from geno_lewm.encoder.pooling import POOL_CENTERED_MEAN, POOL_GLOBAL_MEAN
from geno_lewm.encoder.windowing import DEFAULT_WINDOW_BP, SUPPORTED_WINDOW_BP
from geno_lewm.errors import GenoLeWMError, InputError, RuntimeSetupError, exit_code_for
from geno_lewm.surprise.score import _load_reference_fasta
from tools.research.edit_response_spectroscopy import (
    EMBEDDING_COLUMNS as _CARBON_EMBEDDING_COLUMNS,
    PreparedVariant,
    SkipRecord,
    VariantRecord,
    edit_geometry,
    load_variants,
    prepare_variant,
)

__all__ = [
    "DEFAULT_ENCODER_IDS",
    "DEFAULT_POOL_RADII",
    "EMBEDDING_COLUMNS",
    "ENCODER_REGISTRY",
    "GENERATED_BY",
    "SCHEMA_VERSION",
    "EncoderSpec",
    "HuggingFaceStateEncoder",
    "MultiEncoderRun",
    "TokenLayout",
    "build_summary",
    "derive_token_layout",
    "main",
    "read_done_variant_ids",
    "run_multi_encoder_spectroscopy",
    "write_embeddings_parquet",
]

SCHEMA_VERSION: Final = "1.0.0"
GENERATED_BY: Final = "tools.research.multi_encoder_spectroscopy"

#: The Carbon table's columns plus ``encoder_id``. The Carbon prefix is kept in
#: order so a reader that positionally trusts the R1 schema still lines up.
EMBEDDING_COLUMNS: Final[tuple[str, ...]] = (*_CARBON_EMBEDDING_COLUMNS, "encoder_id")

#: Radius grid swept for every window, in *tokens* (matching the R1 schema).
#: NOTE: a token is 6bp for NT and 1bp for HyenaDNA, so the same radius spans
#: very different physical distances across encoders; the job passes per-encoder
#: radii when a bp-equivalent comparison is wanted.
DEFAULT_POOL_RADII: Final[tuple[int, ...]] = (0, 8, 64, 256)
DEFAULT_BATCH_SIZE: Final = 16
DEFAULT_DTYPE: Final = "fp32"

_AUTO_MODEL: Final = "auto_model"
_MASKED_LM: Final = "masked_lm"
_SUPPORTED_AUTO_CLASSES: Final = (_AUTO_MODEL, _MASKED_LM)
_SUPPORTED_DTYPES: Final = frozenset({"bf16", "fp16", "fp32"})


@dataclass(frozen=True, slots=True)
class EncoderSpec:
    """Identity and load recipe for one genomic encoder."""

    encoder_id: str
    model_id: str
    revision: str | None
    auto_class: str
    trust_remote_code: bool
    state_layer: int


#: Pinned default encoders. ``auto_class`` is not cosmetic:
#: nucleotide-transformer-v2's ``auto_map`` declares no ``AutoModel`` entry, so
#: ``AutoModel.from_pretrained`` silently falls back to stock ESM and dies on a
#: gated-MLP shape mismatch (4096 vs 2048); only ``AutoModelForMaskedLM`` reaches
#: InstaDeep's own ``modeling_esm``. Its hidden states are the encoder's.
ENCODER_REGISTRY: Final[Mapping[str, EncoderSpec]] = {
    "nt-v2-100m-multi": EncoderSpec(
        encoder_id="nt-v2-100m-multi",
        model_id="InstaDeepAI/nucleotide-transformer-v2-100m-multi-species",
        revision="f34324c6fde36a4f635f0f1f06cac5d25acd6798",
        auto_class=_MASKED_LM,
        trust_remote_code=True,
        state_layer=-1,
    ),
    "hyenadna-medium-450k": EncoderSpec(
        encoder_id="hyenadna-medium-450k",
        model_id="LongSafari/hyenadna-medium-450k-seqlen-hf",
        revision="42dedd4d374eac0fb8168549e546a3472fbd27ae",
        auto_class=_AUTO_MODEL,
        trust_remote_code=True,
        state_layer=-1,
    ),
}

DEFAULT_ENCODER_IDS: Final[tuple[str, ...]] = tuple(ENCODER_REGISTRY)


@dataclass(frozen=True, slots=True)
class TokenLayout:
    """Per-token base-pair spans for one tokenized window.

    ``spans[i]`` is the half-open bp span consumed by token ``i``, or ``None``
    for a special token (which consumes no DNA). ``content_bounds`` is the
    half-open token range holding DNA.
    """

    spans: tuple[tuple[int, int] | None, ...]
    content_bounds: tuple[int, int]

    def center_token(self, locus: int) -> int:
        """Return the index of the token containing base-pair ``locus``."""
        for index, span in enumerate(self.spans):
            if span is not None and span[0] <= locus < span[1]:
                return index
        raise InputError(
            "edit locus falls outside the tokenized window content",
            details={"locus": locus, "content_bounds": list(self.content_bounds)},
        )


@dataclass(frozen=True, slots=True)
class EncodedWindow:
    """Token states for one window plus the anchors needed to pool them."""

    rows: Any  # numpy ndarray (n_tokens, d_state)
    center_token: int
    content_bounds: tuple[int, int]


@dataclass(frozen=True, slots=True)
class MultiEncoderRun:
    """The outcome of one spectroscopy pass over a variant set."""

    rows: tuple[dict[str, Any], ...]
    skips: tuple[SkipRecord, ...]
    n_input: int
    n_non_snv: int


# ---------------------------------------------------------------------------
# Locus -> token mapping


def derive_token_layout(
    tokens: Sequence[str],
    *,
    special_tokens: frozenset[str],
    sequence_bp: int,
) -> TokenLayout:
    """Reconstruct each token's base-pair span and verify it tiles the window.

    Both target tokenizers are slow, so no offset mapping exists. They do share
    the property that a non-special token's string is exactly the DNA it
    consumed, so spans accumulate from token lengths. The reconstruction is
    then checked against ``sequence_bp``: if the spans do not tile the window
    the mapping is unverifiable — an ``<unk>`` swallows an unknown number of
    bases — and we fail closed rather than pool the wrong tokens.
    """
    spans: list[tuple[int, int] | None] = []
    cursor = 0
    content_indices: list[int] = []
    for index, token in enumerate(tokens):
        if token in special_tokens:
            spans.append(None)
            continue
        spans.append((cursor, cursor + len(token)))
        cursor += len(token)
        content_indices.append(index)

    if cursor != sequence_bp:
        raise InputError(
            "tokenization does not tile the window; locus mapping is unverifiable",
            details={"consumed_bp": cursor, "sequence_bp": sequence_bp},
            remediation="ensure the window is canonical ACGTN so no <unk> token is emitted",
        )
    if not content_indices:
        raise InputError(
            "tokenization produced no DNA content tokens",
            details={"n_tokens": len(tokens)},
        )
    start, end = content_indices[0], content_indices[-1] + 1
    if len(content_indices) != end - start:
        raise InputError(
            "DNA content tokens must be contiguous",
            details={"start": start, "end": end, "n_content": len(content_indices)},
        )
    return TokenLayout(spans=tuple(spans), content_bounds=(start, end))


# ---------------------------------------------------------------------------
# Generic HuggingFace encoder


class HuggingFaceStateEncoder:
    """Encode DNA windows to per-token hidden states with a stock HF model.

    Deliberately does not reuse :class:`CarbonStateEncoder`, whose 6bp frame and
    ``<dna>`` control tokens are Carbon-specific. Tests inject ``model`` and
    ``tokenizer`` so the suite never downloads anything.
    """

    def __init__(
        self,
        spec: EncoderSpec,
        *,
        dtype: str = DEFAULT_DTYPE,
        device: str | None = None,
        model: Any | None = None,
        tokenizer: Any | None = None,
    ) -> None:
        self.spec = spec
        self._torch = cast(Any, importlib.import_module("torch"))
        if model is None or tokenizer is None:
            model, tokenizer = _load_hf_components(spec, dtype=dtype)
        self._model = model.eval()
        self._tokenizer = tokenizer
        self._device = device
        if device is not None:
            self._model = self._model.to(device)
        self._specials = frozenset(getattr(tokenizer, "all_special_tokens", ()) or ())
        self._d_state: int | None = None

    @property
    def d_state(self) -> int | None:
        """Return the hidden width once a forward has run."""
        return self._d_state

    def _tokenize(self, window: str) -> tuple[list[int], TokenLayout]:
        encoded = self._tokenizer([window])
        ids = [int(value) for value in _first_row(encoded["input_ids"])]
        tokens = list(self._tokenizer.convert_ids_to_tokens(ids))
        layout = derive_token_layout(
            tokens,
            special_tokens=self._specials,
            sequence_bp=len(window),
        )
        return ids, layout

    def encode_token_states(
        self,
        windows: Sequence[str],
        edit_loci: Sequence[int],
    ) -> tuple[EncodedWindow, ...]:
        """Encode windows to token states, bucketing equal-length inputs.

        Bucketing by token count means no padding is ever emitted, which keeps
        HyenaDNA (no attention mask, left padding) and NT (right padding, mask)
        on the same correct path.
        """
        if len(windows) != len(edit_loci):
            raise InputError(
                "windows and edit_loci must have the same length",
                details={"windows": len(windows), "edit_loci": len(edit_loci)},
            )
        tokenized = [self._tokenize(window) for window in windows]
        buckets: dict[int, list[int]] = {}
        for index, (ids, _layout) in enumerate(tokenized):
            buckets.setdefault(len(ids), []).append(index)

        out: list[EncodedWindow | None] = [None] * len(windows)
        for indices in buckets.values():
            batch_ids = [tokenized[i][0] for i in indices]
            states = self._forward(batch_ids)
            for row, index in enumerate(indices):
                layout = tokenized[index][1]
                out[index] = EncodedWindow(
                    rows=states[row],
                    center_token=layout.center_token(edit_loci[index]),
                    content_bounds=layout.content_bounds,
                )
        return tuple(cast("list[EncodedWindow]", out))

    def _forward(self, batch_ids: Sequence[Sequence[int]]) -> Any:
        torch = self._torch
        input_ids = torch.tensor(list(batch_ids), dtype=torch.long)
        if self._device is not None:
            input_ids = input_ids.to(self._device)
        kwargs: dict[str, Any] = {"input_ids": input_ids, "output_hidden_states": True}
        with torch.no_grad():
            output = self._model(**kwargs)
        hidden = output.hidden_states
        layer = _resolve_state_layer(self.spec.state_layer, len(hidden))
        states = hidden[layer]
        array = states.to(torch.float32).cpu().numpy()
        self._d_state = int(array.shape[-1])
        return array


def _first_row(value: Any) -> Sequence[Any]:
    row = value[0]
    return cast("Sequence[Any]", row)


def _resolve_state_layer(state_layer: int, n_hidden: int) -> int:
    """Resolve a possibly-negative hidden-state index, failing closed.

    ``hidden_states[0]`` is the embedding output and ``hidden_states[i]`` the
    i-th layer, so a fixed index is not portable across architectures with
    different depths — hence the default of ``-1`` (final layer).
    """
    layer = state_layer + n_hidden if state_layer < 0 else state_layer
    if layer < 0 or layer >= n_hidden:
        raise InputError(
            "state_layer falls outside the model's hidden states",
            details={"state_layer": state_layer, "n_hidden_states": n_hidden},
        )
    return layer


def _load_hf_components(spec: EncoderSpec, *, dtype: str) -> tuple[Any, Any]:
    if dtype not in _SUPPORTED_DTYPES:
        raise InputError(
            "unsupported dtype",
            details={"dtype": dtype, "supported": sorted(_SUPPORTED_DTYPES)},
        )
    try:
        transformers = cast(Any, importlib.import_module("transformers"))
        torch = cast(Any, importlib.import_module("torch"))
    except ImportError as exc:
        raise RuntimeSetupError(
            "multi-encoder spectroscopy requires Hugging Face Transformers and torch",
            remediation="install geno-lewm[train] or inject model=... and tokenizer=...",
        ) from exc

    if spec.auto_class not in _SUPPORTED_AUTO_CLASSES:
        raise InputError(
            "unsupported auto_class",
            details={"auto_class": spec.auto_class, "supported": list(_SUPPORTED_AUTO_CLASSES)},
        )
    class_name = "AutoModel" if spec.auto_class == _AUTO_MODEL else "AutoModelForMaskedLM"
    model_cls = getattr(transformers, class_name, None)
    if model_cls is None:
        raise RuntimeSetupError(f"transformers must expose {class_name}")

    torch_dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[dtype]
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        spec.model_id,
        revision=spec.revision,
        trust_remote_code=spec.trust_remote_code,
    )
    model = model_cls.from_pretrained(
        spec.model_id,
        revision=spec.revision,
        trust_remote_code=spec.trust_remote_code,
        torch_dtype=torch_dtype,
    )
    return model, tokenizer


# ---------------------------------------------------------------------------
# Pooling (NumPy fast path)


def _pool_rows(window: EncodedWindow, *, pool_radius: int) -> Any:
    """Pool token states, matching ``pool_hidden_states`` semantics exactly.

    Radius ``0`` is a global mean over *every* row including special tokens,
    which is what the Carbon tool's ``pool_radius=0`` path does; positive radii
    mean-pool the content-clamped span ``center ± radius``.
    """
    rows = window.rows
    if pool_radius == 0:
        return rows.mean(axis=0)
    content_start, content_end = window.content_bounds
    start = max(content_start, window.center_token - pool_radius)
    end = min(content_end, window.center_token + pool_radius + 1)
    return rows[start:end].mean(axis=0)


def _pool_type_for_radius(pool_radius: int) -> str:
    return POOL_GLOBAL_MEAN if pool_radius == 0 else POOL_CENTERED_MEAN


# ---------------------------------------------------------------------------
# Core measurement


def run_multi_encoder_spectroscopy(
    variants: Sequence[VariantRecord],
    reference_sequences: Mapping[str, str],
    encoder: Any,
    *,
    encoder_id: str,
    window_bp: int = DEFAULT_WINDOW_BP,
    pool_radii: Sequence[int] = DEFAULT_POOL_RADII,
    batch_size: int = DEFAULT_BATCH_SIZE,
    resume_variant_ids: Iterable[str] = (),
) -> MultiEncoderRun:
    """Measure edit-response geometry over ``variants`` for one encoder.

    SNV-only: multi-base alleles change the window's length under a
    length-preserving edit and would confound the geometry, so they are counted
    and dropped explicitly rather than silently encoded.
    """
    radii = _validate_pool_radii(pool_radii)
    _validate_window_bp(window_bp)
    _require_positive_int("batch_size", batch_size)
    already_done = frozenset(resume_variant_ids)

    prepared: list[PreparedVariant] = []
    skips: list[SkipRecord] = []
    n_non_snv = 0
    for record in variants:
        if record.variant_id in already_done:
            continue
        if len(record.ref) != 1 or len(record.alt) != 1:
            n_non_snv += 1
            continue
        try:
            prepared.append(prepare_variant(record, reference_sequences, window_bp=window_bp))
        except GenoLeWMError as exc:
            skips.append(_skip_from_error(record.variant_id, exc))

    rows: list[dict[str, Any]] = []
    for group in _chunk(prepared, batch_size):
        encoded, group_skips = _encode_group(encoder, group)
        skips.extend(group_skips)
        for item, ref_states, alt_states in encoded:
            try:
                rows.extend(
                    _rows_for_variant(
                        item,
                        ref_states,
                        alt_states,
                        radii=radii,
                        encoder_id=encoder_id,
                    )
                )
            except GenoLeWMError as exc:
                skips.append(_skip_from_error(item.record.variant_id, exc))

    return MultiEncoderRun(
        rows=tuple(rows),
        skips=tuple(skips),
        n_input=len(variants),
        n_non_snv=n_non_snv,
    )


def _encode_group(
    encoder: Any,
    group: Sequence[PreparedVariant],
) -> tuple[list[tuple[PreparedVariant, EncodedWindow, EncodedWindow]], list[SkipRecord]]:
    """Encode a batch, retrying one-by-one so a bad window cannot abort it."""
    if not group:
        return [], []
    try:
        states = _encode_windows(encoder, group)
    except GenoLeWMError:
        return _encode_group_individually(encoder, group)
    return list(zip(group, states[0::2], states[1::2], strict=True)), []


def _encode_group_individually(
    encoder: Any,
    group: Sequence[PreparedVariant],
) -> tuple[list[tuple[PreparedVariant, EncodedWindow, EncodedWindow]], list[SkipRecord]]:
    encoded: list[tuple[PreparedVariant, EncodedWindow, EncodedWindow]] = []
    skips: list[SkipRecord] = []
    for item in group:
        try:
            single = _encode_windows(encoder, [item])
        except GenoLeWMError as exc:
            skips.append(_skip_from_error(item.record.variant_id, exc))
            continue
        encoded.append((item, single[0], single[1]))
    return encoded, skips


def _encode_windows(
    encoder: Any,
    group: Sequence[PreparedVariant],
) -> tuple[EncodedWindow, ...]:
    windows: list[str] = []
    edit_loci: list[int] = []
    for item in group:
        windows.append(item.reference_window)
        windows.append(item.edited_window)
        edit_loci.append(item.rel_pos)
        edit_loci.append(item.rel_pos)
    states = tuple(encoder.encode_token_states(windows, edit_loci))
    if len(states) != len(windows):
        raise InputError(
            "encoder returned an unexpected number of token-state items",
            details={"expected": len(windows), "observed": len(states)},
        )
    return states


def _rows_for_variant(
    item: PreparedVariant,
    ref_states: EncodedWindow,
    alt_states: EncodedWindow,
    *,
    radii: Sequence[int],
    encoder_id: str,
) -> list[dict[str, Any]]:
    record = item.record
    rows: list[dict[str, Any]] = []
    for pool_radius in radii:
        s_ref = _pool_rows(ref_states, pool_radius=pool_radius)
        s_alt = _pool_rows(alt_states, pool_radius=pool_radius)
        # Geometry is imported from the Carbon tool, not reimplemented, so the
        # two tables' scalar columns are guaranteed to mean the same thing.
        geometry = edit_geometry(
            [float(value) for value in s_ref],
            [float(value) for value in s_alt],
        )
        rows.append(
            {
                "variant_id": record.variant_id,
                "chrom": record.chrom,
                "pos": record.pos,
                "ref": record.ref,
                "alt": record.alt,
                "label": record.label,
                "label_group": record.label_group,
                "continuous_score": record.continuous_score,
                "region": record.region,
                "gene": record.gene,
                "pool_radius": pool_radius,
                "pool_type": _pool_type_for_radius(pool_radius),
                "d_state": geometry.d_state,
                "norm_s_ref": geometry.norm_s_ref,
                "norm_s_alt": geometry.norm_s_alt,
                "cos_ref_alt": geometry.cos_ref_alt,
                "l2_delta": geometry.l2_delta,
                "rel_delta": geometry.rel_delta,
                "s_ref": [float(value) for value in s_ref],
                "s_alt": [float(value) for value in s_alt],
                "encoder_id": encoder_id,
            }
        )
    return rows


def _skip_from_error(variant_id: str, exc: GenoLeWMError) -> SkipRecord:
    return SkipRecord(variant_id=variant_id, code=exc.code, message=exc.message or str(exc))


# ---------------------------------------------------------------------------
# Parquet I/O


def _parquet_schema(pa: Any) -> Any:
    """Return the R1 schema plus ``encoder_id``."""
    return pa.schema(
        [
            ("variant_id", pa.string()),
            ("chrom", pa.string()),
            ("pos", pa.int64()),
            ("ref", pa.string()),
            ("alt", pa.string()),
            ("label", pa.string()),
            ("label_group", pa.string()),
            ("continuous_score", pa.float64()),
            ("region", pa.string()),
            ("gene", pa.string()),
            ("pool_radius", pa.int64()),
            ("pool_type", pa.string()),
            ("d_state", pa.int64()),
            ("norm_s_ref", pa.float64()),
            ("norm_s_alt", pa.float64()),
            ("cos_ref_alt", pa.float64()),
            ("l2_delta", pa.float64()),
            ("rel_delta", pa.float64()),
            ("s_ref", pa.list_(pa.float32())),
            ("s_alt", pa.list_(pa.float32())),
            ("encoder_id", pa.string()),
        ]
    )


def write_embeddings_parquet(
    path: str | Path,
    rows: Sequence[Mapping[str, Any]],
    *,
    resume: bool = False,
) -> Path:
    """Write ``rows`` to a Parquet table (appending when ``resume`` is set)."""
    pa = cast(Any, importlib.import_module("pyarrow"))
    pq = cast(Any, importlib.import_module("pyarrow.parquet"))
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    columns = {name: [row[name] for row in rows] for name in EMBEDDING_COLUMNS}
    new_table = pa.table(columns, schema=_parquet_schema(pa))
    if resume and out.is_file():
        existing = pq.read_table(out).cast(_parquet_schema(pa))
        table = pa.concat_tables([existing, new_table])
    else:
        table = new_table
    pq.write_table(table, out)
    return out


def read_done_variant_ids(path: str | Path) -> frozenset[str]:
    """Return the ``variant_id`` set already present in a Parquet table.

    Each table holds exactly one encoder, so ``variant_id`` alone is a
    sufficient resume key.
    """
    out = Path(path)
    if not out.is_file():
        return frozenset()
    pq = cast(Any, importlib.import_module("pyarrow.parquet"))
    table = pq.read_table(out, columns=["variant_id"])
    return frozenset(str(value) for value in table.column("variant_id").to_pylist())


# ---------------------------------------------------------------------------
# Summary


def build_summary(
    run: MultiEncoderRun,
    *,
    encoder_id: str,
    config: Mapping[str, Any],
    provenance: Mapping[str, Any],
) -> dict[str, Any]:
    """Build the provenance + per-radius aggregate summary payload."""
    return {
        "schema_version": SCHEMA_VERSION,
        "generated_by": GENERATED_BY,
        "encoder_id": encoder_id,
        "config": dict(config),
        "provenance": dict(provenance),
        "counts": {
            "n_input": run.n_input,
            "n_non_snv": run.n_non_snv,
            "n_variants": len({str(row["variant_id"]) for row in run.rows}),
            "n_rows": len(run.rows),
            "n_skipped": len(run.skips),
        },
        "skips": {
            "by_code": _count(skip.code for skip in run.skips),
            "records": [skip.as_dict() for skip in run.skips],
        },
        "per_pool_radius": _per_radius(run.rows),
    }


def _per_radius(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    import statistics

    by_radius: dict[int, list[Mapping[str, Any]]] = {}
    for row in rows:
        by_radius.setdefault(int(row["pool_radius"]), []).append(row)
    out: list[dict[str, Any]] = []
    for pool_radius in sorted(by_radius):
        radius_rows = by_radius[pool_radius]
        entry: dict[str, Any] = {
            "pool_radius": pool_radius,
            "pool_type": _pool_type_for_radius(pool_radius),
            "n_variants": len(radius_rows),
            "d_state": int(radius_rows[0]["d_state"]),
        }
        for field in ("cos_ref_alt", "rel_delta", "norm_s_ref"):
            values = [float(row[field]) for row in radius_rows]
            entry[field] = {
                "mean": statistics.fmean(values),
                "median": statistics.median(values),
                "std": statistics.pstdev(values),
            }
        out.append(entry)
    return out


def _count(values: Iterable[str]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for value in values:
        counts[value] = counts.get(value, 0) + 1
    return dict(sorted(counts.items()))


# ---------------------------------------------------------------------------
# Validation helpers


def _validate_pool_radii(pool_radii: Sequence[int]) -> tuple[int, ...]:
    if not pool_radii:
        raise InputError("pool_radii must contain at least one radius")
    for value in pool_radii:
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise InputError(
                "pool_radii must be non-negative integers",
                details={"pool_radii": list(pool_radii)},
            )
    return tuple(sorted(dict.fromkeys(pool_radii)))


def _validate_window_bp(window_bp: int) -> int:
    if isinstance(window_bp, bool) or not isinstance(window_bp, int):
        raise InputError("window_bp must be an integer", details={"window_bp": window_bp})
    if window_bp not in SUPPORTED_WINDOW_BP:
        raise InputError(
            "unsupported window length; reuse a scorer-supported width",
            details={"window_bp": window_bp, "supported": list(SUPPORTED_WINDOW_BP)},
            remediation="pass one of the supported window widths so windows match the scorer",
        )
    return window_bp


def _require_positive_int(name: str, value: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise InputError(f"{name} must be a positive integer", details={name: value})
    return value


def _chunk(items: Sequence[PreparedVariant], size: int) -> Iterable[Sequence[PreparedVariant]]:
    for start in range(0, len(items), size):
        yield items[start : start + size]


# ---------------------------------------------------------------------------
# CLI


def _git_commit() -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
            timeout=10,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return result.stdout.strip() or None


def _parse_radii_csv(raw: str) -> tuple[int, ...]:
    parts = [chunk.strip() for chunk in raw.split(",") if chunk.strip()]
    if not parts:
        raise InputError("--pool-radii must list at least one integer")
    try:
        radii = [int(part) for part in parts]
    except ValueError as exc:
        raise InputError(
            "--pool-radii must be comma-separated integers", details={"raw": raw}
        ) from exc
    return _validate_pool_radii(radii)


def _resolve_encoder(encoder_id: str) -> EncoderSpec:
    """Resolve one registry id, failing closed on a typo.

    A typo must not silently fall back to some default checkpoint: that would
    label a table with an encoder that never produced it.
    """
    spec = ENCODER_REGISTRY.get(encoder_id)
    if spec is None:
        raise InputError(
            "unknown encoder id",
            details={"encoder": encoder_id, "known": sorted(ENCODER_REGISTRY)},
            remediation="pass one of the registered encoder ids",
        )
    return spec


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Multi-encoder edit-response spectroscopy.")
    parser.add_argument("--variants", type=Path, required=True)
    parser.add_argument("--reference-fasta", type=Path, required=True)
    parser.add_argument("--out-embeddings", type=Path, required=True)
    parser.add_argument("--out-summary", type=Path, required=True)
    parser.add_argument("--encoder", type=str, required=True)
    parser.add_argument("--window-bp", type=int, default=DEFAULT_WINDOW_BP)
    parser.add_argument(
        "--pool-radii", type=str, default=",".join(str(r) for r in DEFAULT_POOL_RADII)
    )
    parser.add_argument("--state-layer", type=int, default=None)
    parser.add_argument(
        "--dtype", type=str, default=DEFAULT_DTYPE, choices=sorted(_SUPPORTED_DTYPES)
    )
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--resume", action="store_true")
    return parser


def _build_encoder(spec: EncoderSpec, args: argparse.Namespace) -> HuggingFaceStateEncoder:
    return HuggingFaceStateEncoder(spec, dtype=args.dtype, device=args.device)


def main(argv: list[str] | None = None) -> int:
    """CLI entry point: run spectroscopy for one encoder and write the outputs."""
    parser = _parser()
    args = parser.parse_args(argv)
    try:
        radii = _parse_radii_csv(args.pool_radii)
        _validate_window_bp(args.window_bp)
        _require_positive_int("batch_size", args.batch_size)
        spec = _resolve_encoder(args.encoder)
        if args.state_layer is not None:
            spec = EncoderSpec(
                encoder_id=spec.encoder_id,
                model_id=spec.model_id,
                revision=spec.revision,
                auto_class=spec.auto_class,
                trust_remote_code=spec.trust_remote_code,
                state_layer=args.state_layer,
            )
        variants = load_variants(args.variants)
        if args.limit is not None:
            _require_positive_int("limit", args.limit)
            variants = variants[: args.limit]
        reference_sequences = _load_reference_fasta(args.reference_fasta)

        done = read_done_variant_ids(args.out_embeddings) if args.resume else frozenset()
        encoder = _build_encoder(spec, args)
        run = run_multi_encoder_spectroscopy(
            variants,
            reference_sequences,
            encoder,
            encoder_id=spec.encoder_id,
            window_bp=args.window_bp,
            pool_radii=radii,
            batch_size=args.batch_size,
            resume_variant_ids=done,
        )
        write_embeddings_parquet(args.out_embeddings, run.rows, resume=args.resume)
        config = {
            "window_bp": args.window_bp,
            "pool_radii": list(radii),
            "dtype": args.dtype,
            "batch_size": args.batch_size,
            "limit": args.limit,
            "resume": bool(args.resume),
            "reference_fasta": args.reference_fasta.name,
            "variants": args.variants.name,
        }
        provenance = {
            "git_commit": _git_commit(),
            "encoders": [
                {
                    "encoder_id": spec.encoder_id,
                    "model_id": spec.model_id,
                    "revision": spec.revision,
                    "auto_class": spec.auto_class,
                    "state_layer": spec.state_layer,
                    "dtype": args.dtype,
                    "d_state": getattr(encoder, "d_state", None),
                    "normalize": False,
                    "pool_family": "raw_token_states",
                }
            ],
        }
        summary = build_summary(
            run,
            encoder_id=spec.encoder_id,
            config=config,
            provenance=provenance,
        )
        _write_summary(args.out_summary, summary)
    except GenoLeWMError as exc:
        sys.stderr.write(f"error: {exc}\n")
        if exc.details:
            sys.stderr.write(json.dumps(exc.details, sort_keys=True) + "\n")
        return exit_code_for(exc)
    sys.stdout.write(
        json.dumps(
            {
                "out_embeddings": str(args.out_embeddings),
                "out_summary": str(args.out_summary),
                "encoder_id": spec.encoder_id,
                "n_rows": len(run.rows),
                "n_skipped": len(run.skips),
            },
            sort_keys=True,
        )
        + "\n"
    )
    return 0


def _write_summary(path: str | Path, summary: Mapping[str, Any]) -> Path:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return out


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
