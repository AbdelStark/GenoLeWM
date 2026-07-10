# SPDX-License-Identifier: Apache-2.0
"""Carbon state encoder wrapper.

The heavy runtime dependencies remain optional. Callers can inject a
tokenizer/model pair for tests or already-loaded runtimes; otherwise the
wrapper loads from Hugging Face Transformers with ``local_files_only`` by
default so constructing the encoder does not hide a network download.
"""

from __future__ import annotations

import hashlib
import importlib
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, cast

from geno_lewm._inference import torch_inference_context
from geno_lewm.encoder._dna_tokenizer import CarbonDNATokenizer
from geno_lewm.encoder._identity import encoder_runtime_hash, encoder_weights_hash
from geno_lewm.encoder._normalization import l2_normalize_state
from geno_lewm.encoder.pooling import (
    DEFAULT_POOL_RADIUS_TOKENS,
    POOL_CENTERED_MEAN,
    POOL_GLOBAL_MEAN,
    pool_hidden_states,
)
from geno_lewm.encoder.windowing import CARBON_TOKEN_BP, canonicalize_dna, wrap_dna_for_tokenizer
from geno_lewm.errors import InputError, RuntimeSetupError

__all__ = ["CarbonStateEncoder"]

_SUPPORTED_DTYPES = frozenset({"bf16", "fp16", "fp32"})
_PoolType = Literal["centered_mean", "global_mean"]


class CarbonStateEncoder:
    """Encode DNA windows with Carbon hidden states plus deterministic pooling."""

    def __init__(
        self,
        model_id: str,
        revision: str,
        *,
        dtype: str = "bf16",
        state_layer: int = -1,
        pool_type: str = POOL_CENTERED_MEAN,
        pool_radius: int = DEFAULT_POOL_RADIUS_TOKENS,
        normalize: bool = True,
        lora_config: object | None = None,
        model: object | None = None,
        tokenizer: object | None = None,
        encoder_hash: bytes | str | None = None,
        local_files_only: bool = True,
        trust_remote_code: bool = False,
        device: str | None = None,
    ) -> None:
        if not model_id:
            raise InputError("model_id must be non-empty")
        if not revision:
            raise InputError("revision must be non-empty")
        if dtype not in _SUPPORTED_DTYPES:
            raise InputError(
                "unsupported encoder dtype",
                details={"dtype": dtype, "supported": sorted(_SUPPORTED_DTYPES)},
            )
        if not isinstance(state_layer, int) or isinstance(state_layer, bool):
            raise InputError(
                "state_layer must be an integer",
                details={"state_layer": state_layer, "type": type(state_layer).__name__},
            )
        if pool_type not in {POOL_CENTERED_MEAN, POOL_GLOBAL_MEAN}:
            raise InputError(
                "unsupported pool_type",
                details={
                    "pool_type": pool_type,
                    "supported": [POOL_CENTERED_MEAN, POOL_GLOBAL_MEAN],
                },
            )
        if not isinstance(pool_radius, int) or isinstance(pool_radius, bool) or pool_radius < 0:
            raise InputError(
                "pool_radius must be a non-negative integer",
                details={"pool_radius": pool_radius, "type": type(pool_radius).__name__},
            )
        if pool_type == POOL_GLOBAL_MEAN and pool_radius != 0:
            raise InputError(
                "global_mean pooling requires pool_radius=0",
                details={"pool_type": pool_type, "pool_radius": pool_radius},
            )
        if not isinstance(normalize, bool):
            raise InputError(
                "normalize must be bool",
                details={"type": type(normalize).__name__},
            )
        if lora_config is not None:
            raise RuntimeSetupError(
                "Carbon LoRA adapters are not supported by CarbonStateEncoder yet",
                remediation="merge LoRA adapters before loading or track the Phase 2 adapter issue",
            )
        if (model is None) != (tokenizer is None):
            raise InputError(
                "model and tokenizer must be supplied together",
                details={"model": model is not None, "tokenizer": tokenizer is not None},
            )

        self.model_id = model_id
        self.revision = revision
        self.dtype = dtype
        self.state_layer = state_layer
        self.pool_type = cast(_PoolType, pool_type)
        self.pool_radius = pool_radius
        self.normalize = normalize
        self.local_files_only = local_files_only
        self.trust_remote_code = trust_remote_code
        self.device = _resolve_device(device)
        self._encoder_hash = _coerce_encoder_hash(encoder_hash)
        self._d_state: int | None = None

        if model is None or tokenizer is None:
            if self._encoder_hash is not None:
                _verify_local_encoder_weights(model_id, expected_hash=self._encoder_hash)
            tokenizer, model = _load_transformers_components(
                model_id=model_id,
                revision=revision,
                dtype=dtype,
                local_files_only=local_files_only,
                trust_remote_code=trust_remote_code,
            )
        self.tokenizer = tokenizer
        self.model = model
        self._parameter_count, self._trainable_parameter_count = _freeze_module_parameters(
            self.model
        )
        _eval_if_available(self.model)
        _move_module_to_device(self.model, self.device)
        config = getattr(self.model, "config", None)
        hidden_size = getattr(config, "hidden_size", None)
        if isinstance(hidden_size, int) and not isinstance(hidden_size, bool) and hidden_size > 0:
            self._d_state = hidden_size

    def encode(self, window: str, edit_locus: int | None = None) -> tuple[float, ...]:
        """Encode and pool one DNA window."""
        return self.encode_batch([window], [edit_locus])[0]

    def encode_batch(
        self,
        windows: Sequence[str],
        edit_loci: Sequence[int | None],
    ) -> tuple[tuple[float, ...], ...]:
        """Encode and pool a batch of DNA windows."""
        if not isinstance(windows, Sequence) or isinstance(windows, str | bytes):
            raise InputError(
                "windows must be a sequence of DNA strings",
                details={"type": type(windows).__name__},
            )
        if not isinstance(edit_loci, Sequence) or isinstance(edit_loci, str | bytes):
            raise InputError(
                "edit_loci must be a sequence of int or None values",
                details={"type": type(edit_loci).__name__},
            )
        if len(windows) != len(edit_loci):
            raise InputError(
                "windows and edit_loci must have the same length",
                details={"windows": len(windows), "edit_loci": len(edit_loci)},
            )
        if not windows:
            raise InputError("windows must contain at least one sequence")

        normalized = tuple(canonicalize_dna(window) for window in windows)
        wrapped = [wrap_dna_for_tokenizer(window) for window in normalized]
        tokenized = _tokenize(self.tokenizer, wrapped)
        layouts = _resolve_dna_token_layouts(
            self.tokenizer,
            tokenized,
            sequences=normalized,
        )
        tokenized = _move_inputs_to_device(tokenized, self.device)
        with torch_inference_context():
            output = _call_model(self.model, tokenized)
        rows_by_item = _hidden_rows_by_item(output, state_layer=self.state_layer)
        if len(rows_by_item) != len(windows):
            raise InputError(
                "encoder output batch size does not match input windows",
                details={"expected": len(windows), "observed": len(rows_by_item)},
            )

        pooled_rows: list[tuple[float, ...]] = []
        for rows, edit_locus, layout, sequence in zip(
            rows_by_item,
            edit_loci,
            layouts,
            normalized,
            strict=True,
        ):
            if len(rows) != layout.padded_token_count:
                raise InputError(
                    "encoder hidden-state length does not match tokenized input",
                    details={
                        "hidden_tokens": len(rows),
                        "tokenized_tokens": layout.padded_token_count,
                    },
                )
            center_token = layout.center_token(edit_locus, sequence_bp=len(sequence))
            pooled_rows.append(
                pool_hidden_states(
                    rows[: layout.active_token_count],
                    edit_locus=edit_locus,
                    center_token=center_token,
                    content_token_bounds=(
                        layout.dna_content_start,
                        layout.dna_content_start + layout.dna_content_count,
                    ),
                    pool_type=self.pool_type,
                    pool_radius=self.pool_radius,
                ).vector
            )
        pooled = tuple(pooled_rows)
        encoded = (
            tuple(
                l2_normalize_state(vector, item_index=index) for index, vector in enumerate(pooled)
            )
            if self.normalize
            else pooled
        )
        if encoded:
            self._d_state = len(encoded[0])
        return encoded

    def pooling_identity(
        self,
        window: str,
        edit_locus: int | None,
    ) -> tuple[str, int, int | None]:
        """Resolve the exact cache pooling identity from Carbon token IDs."""
        sequence = canonicalize_dna(window)
        tokenized = _tokenize(self.tokenizer, [wrap_dna_for_tokenizer(sequence)])
        layout = _resolve_dna_token_layouts(
            self.tokenizer,
            tokenized,
            sequences=(sequence,),
        )[0]
        center_token = layout.center_token(edit_locus, sequence_bp=len(sequence))
        if edit_locus is None:
            return POOL_GLOBAL_MEAN, 0, None
        if self.pool_type == POOL_GLOBAL_MEAN:
            return POOL_GLOBAL_MEAN, 0, None
        return POOL_CENTERED_MEAN, self.pool_radius, center_token

    @property
    def encoder_hash(self) -> bytes:
        """Return the configured encoder hash bytes."""
        if self._encoder_hash is None:
            raise RuntimeSetupError(
                "encoder_hash is not available",
                remediation="pass encoder_hash from the model manifest when constructing CarbonStateEncoder",
            )
        return self._encoder_hash

    @property
    def d_state(self) -> int:
        """Return the pooled state width when known."""
        if self._d_state is None:
            raise RuntimeSetupError(
                "d_state is not known until the encoder has produced at least one state",
            )
        return self._d_state

    @property
    def parameter_count(self) -> int:
        """Return the number of parameters exposed by the encoder module."""
        return self._parameter_count

    @property
    def trainable_parameter_count(self) -> int:
        """Return zero after the frozen-encoder contract is enforced."""
        return self._trainable_parameter_count


@dataclass(frozen=True, slots=True)
class _DNATokenLayout:
    active_token_count: int
    padded_token_count: int
    dna_content_start: int
    dna_content_count: int
    token_bp: int

    def center_token(self, edit_locus: int | None, *, sequence_bp: int) -> int | None:
        if edit_locus is None:
            return None
        if isinstance(edit_locus, bool) or not isinstance(edit_locus, int):
            raise InputError(
                "edit_locus must be an integer offset",
                details={"edit_locus": edit_locus, "type": type(edit_locus).__name__},
            )
        if edit_locus < 0 or edit_locus >= sequence_bp:
            raise InputError(
                "edit_locus falls outside the encoded DNA window",
                details={"edit_locus": edit_locus, "sequence_bp": sequence_bp},
            )
        content_offset = edit_locus // self.token_bp
        if content_offset >= self.dna_content_count:
            raise InputError(
                "edit_locus maps outside the tokenized DNA content",
                details={
                    "edit_locus": edit_locus,
                    "token_bp": self.token_bp,
                    "dna_content_tokens": self.dna_content_count,
                },
            )
        return self.dna_content_start + content_offset


def _load_transformers_components(
    *,
    model_id: str,
    revision: str,
    dtype: str,
    local_files_only: bool,
    trust_remote_code: bool,
) -> tuple[object, object]:
    try:
        transformers = importlib.import_module("transformers")
    except ImportError as exc:
        raise RuntimeSetupError(
            "CarbonStateEncoder requires Hugging Face Transformers",
            remediation="install geno-lewm[train] or pass injected model=... and tokenizer=...",
        ) from exc

    try:
        torch = importlib.import_module("torch")
    except ImportError:
        torch = None

    model_cls = getattr(transformers, "AutoModel", None)
    if model_cls is None:
        raise RuntimeSetupError("transformers must expose AutoModel")

    model_dir = _resolve_runtime_directory(
        model_id=model_id,
        revision=revision,
        local_files_only=local_files_only,
    )
    tokenizer = CarbonDNATokenizer.from_model_dir(model_dir)
    model_kwargs: dict[str, object] = {
        "local_files_only": True,
        "trust_remote_code": trust_remote_code,
    }
    torch_dtype = _torch_dtype(torch, dtype)
    if torch_dtype is not None:
        model_kwargs["torch_dtype"] = torch_dtype
    model = model_cls.from_pretrained(str(model_dir), **model_kwargs)
    return tokenizer, model


def _resolve_runtime_directory(
    *,
    model_id: str,
    revision: str,
    local_files_only: bool,
) -> Path:
    local_path = Path(model_id).expanduser()
    if local_path.is_dir():
        return local_path.resolve()
    try:
        hub = importlib.import_module("huggingface_hub")
    except ImportError as exc:
        raise RuntimeSetupError(
            "resolving a Carbon repository ID requires huggingface_hub",
            remediation="install geno-lewm[train] or pass a local Carbon runtime directory",
        ) from exc
    snapshot_download = getattr(hub, "snapshot_download", None)
    if not callable(snapshot_download):
        raise RuntimeSetupError("huggingface_hub must expose snapshot_download")
    try:
        snapshot = snapshot_download(
            repo_id=model_id,
            revision=revision,
            local_files_only=local_files_only,
        )
    except Exception as exc:
        raise RuntimeSetupError(
            "Carbon runtime snapshot could not be resolved",
            details={
                "model_id": model_id,
                "revision": revision,
                "local_files_only": local_files_only,
            },
            remediation="mount or cache the complete pinned Carbon runtime package",
        ) from exc
    resolved = Path(snapshot)
    if not resolved.is_dir():
        raise RuntimeSetupError(
            "resolved Carbon runtime snapshot is not a directory",
            details={"path": str(resolved)},
        )
    return resolved.resolve()


def _resolve_dna_token_layouts(
    tokenizer: object,
    tokenized: Mapping[str, object],
    *,
    sequences: Sequence[str],
) -> tuple[_DNATokenLayout, ...]:
    input_ids = _integer_matrix(tokenized.get("input_ids"), field="input_ids")
    attention_mask = _integer_matrix(
        tokenized.get("attention_mask"),
        field="attention_mask",
    )
    if len(input_ids) != len(sequences) or len(attention_mask) != len(sequences):
        raise InputError(
            "tokenizer batch size does not match encoded DNA windows",
            details={
                "sequences": len(sequences),
                "input_ids": len(input_ids),
                "attention_mask": len(attention_mask),
            },
        )
    begin_id = _tokenizer_nonnegative_int(tokenizer, "dna_begin_token_id")
    end_id = _tokenizer_nonnegative_int(tokenizer, "dna_end_token_id")
    oov_id = _tokenizer_nonnegative_int(tokenizer, "oov_token_id")
    pad_id = _tokenizer_nonnegative_int(tokenizer, "pad_token_id")
    token_bp = _tokenizer_nonnegative_int(tokenizer, "k", positive=True)
    dna_start_id = _tokenizer_nonnegative_int(tokenizer, "dna_start_id")
    dna_vocab_size = _tokenizer_nonnegative_int(tokenizer, "dna_vocab_size", positive=True)
    if len({begin_id, end_id, oov_id}) != 3:
        raise InputError("Carbon tokenizer DNA control-token IDs must be distinct")
    if (begin_id, end_id, oov_id) != (
        dna_start_id,
        dna_start_id + 1,
        dna_start_id + 2,
    ):
        raise InputError(
            "Carbon tokenizer DNA control-token IDs are not contiguous",
            details={
                "dna_start_id": dna_start_id,
                "begin_id": begin_id,
                "end_id": end_id,
                "oov_id": oov_id,
            },
        )
    if token_bp != CARBON_TOKEN_BP:
        raise InputError(
            "Carbon tokenizer k does not match the encoder contract",
            details={"observed": token_bp, "expected": CARBON_TOKEN_BP},
        )
    first_kmer_id = dna_start_id + 3
    last_kmer_id = first_kmer_id + (4**token_bp) - 1
    if last_kmer_id >= dna_start_id + dna_vocab_size:
        raise InputError("Carbon tokenizer DNA vocabulary cannot represent every k-mer")

    layouts: list[_DNATokenLayout] = []
    for item_index, (ids, mask, sequence) in enumerate(
        zip(input_ids, attention_mask, sequences, strict=True)
    ):
        if len(ids) != len(mask) or not ids:
            raise InputError(
                "tokenizer input_ids and attention_mask rows must have equal non-zero length",
                details={"item_index": item_index, "ids": len(ids), "mask": len(mask)},
            )
        if any(value not in (0, 1) for value in mask):
            raise InputError(
                "Carbon attention_mask must contain only zero or one",
                details={"item_index": item_index},
            )
        active_count = sum(mask)
        if mask != ([1] * active_count) + ([0] * (len(mask) - active_count)):
            raise InputError(
                "Carbon tokenizer must use contiguous right padding",
                details={"item_index": item_index},
            )
        if any(token_id != pad_id for token_id in ids[active_count:]):
            raise InputError(
                "masked Carbon tokenizer positions must contain the declared pad token",
                details={"item_index": item_index},
            )
        active_ids = ids[:active_count]
        begin_positions = [
            index for index, token_id in enumerate(active_ids) if token_id == begin_id
        ]
        end_positions = [index for index, token_id in enumerate(active_ids) if token_id == end_id]
        if len(begin_positions) != 1 or len(end_positions) != 1:
            raise InputError(
                "tokenized Carbon window must contain exactly one DNA control-token pair",
                details={
                    "item_index": item_index,
                    "dna_begin_count": len(begin_positions),
                    "dna_end_count": len(end_positions),
                },
            )
        begin = begin_positions[0]
        end = end_positions[0]
        expected_content_count = (len(sequence) + token_bp - 1) // token_bp
        if begin != 0 or end != active_count - 1 or end - begin - 1 != expected_content_count:
            raise InputError(
                "tokenized Carbon window has an ambiguous DNA/control-token layout",
                details={
                    "item_index": item_index,
                    "begin_token": begin,
                    "end_token": end,
                    "active_tokens": active_count,
                    "expected_dna_tokens": expected_content_count,
                },
            )
        content_ids = active_ids[begin + 1 : end]
        invalid_content = [
            token_id
            for token_id in content_ids
            if token_id != oov_id and not first_kmer_id <= token_id <= last_kmer_id
        ]
        if invalid_content:
            raise InputError(
                "tokenized Carbon DNA span contains non-DNA token IDs",
                details={"item_index": item_index, "token_ids": sorted(set(invalid_content))},
            )
        layouts.append(
            _DNATokenLayout(
                active_token_count=active_count,
                padded_token_count=len(ids),
                dna_content_start=begin + 1,
                dna_content_count=expected_content_count,
                token_bp=token_bp,
            )
        )
    return tuple(layouts)


def _integer_matrix(value: object, *, field: str) -> list[list[int]]:
    materialized = _materialize(value)
    if not isinstance(materialized, Sequence) or isinstance(materialized, str | bytes):
        raise InputError(
            f"tokenizer output {field} must be a batch matrix",
            details={"type": type(materialized).__name__},
        )
    matrix: list[list[int]] = []
    for item_index, raw_row in enumerate(cast(Sequence[object], materialized)):
        if not isinstance(raw_row, Sequence) or isinstance(raw_row, str | bytes):
            raise InputError(
                f"tokenizer output {field} rows must be sequences",
                details={"item_index": item_index},
            )
        row: list[int] = []
        for token_index, raw_value in enumerate(cast(Sequence[object], raw_row)):
            if isinstance(raw_value, bool) or not isinstance(raw_value, int):
                raise InputError(
                    f"tokenizer output {field} values must be integers",
                    details={
                        "item_index": item_index,
                        "token_index": token_index,
                        "type": type(raw_value).__name__,
                    },
                )
            row.append(raw_value)
        matrix.append(row)
    return matrix


def _tokenizer_nonnegative_int(
    tokenizer: object,
    name: str,
    *,
    positive: bool = False,
) -> int:
    value = getattr(tokenizer, name, None)
    minimum = 1 if positive else 0
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise InputError(
            f"Carbon tokenizer must expose a valid {name}",
            details={"value": value, "type": type(value).__name__},
        )
    return value


def _torch_dtype(torch: object | None, dtype: str) -> object | None:
    if torch is None:
        return None
    if dtype == "bf16":
        return getattr(torch, "bfloat16", None)
    if dtype == "fp16":
        return getattr(torch, "float16", None)
    if dtype == "fp32":
        return getattr(torch, "float32", None)
    return None


def _eval_if_available(model: object) -> None:
    eval_method = getattr(model, "eval", None)
    if callable(eval_method):
        eval_method()


def _freeze_module_parameters(model: object) -> tuple[int, int]:
    """Disable gradients on a torch-like module and verify the frozen contract."""
    parameters_method = getattr(model, "parameters", None)
    if not callable(parameters_method):
        return 0, 0
    try:
        parameters = tuple(parameters_method())
    except Exception as exc:
        raise RuntimeSetupError("failed to enumerate Carbon encoder parameters") from exc

    total = 0
    trainable = 0
    trainable_indices: list[int] = []
    for index, parameter in enumerate(parameters):
        numel_method = getattr(parameter, "numel", None)
        count = 0
        if callable(numel_method):
            count = numel_method()
            if isinstance(count, bool) or not isinstance(count, int) or count < 0:
                raise RuntimeSetupError(
                    "Carbon encoder parameter numel() must return a non-negative integer",
                    details={"parameter_index": index},
                )
            total += count
        requires_grad_method = getattr(parameter, "requires_grad_", None)
        if callable(requires_grad_method):
            requires_grad_method(False)
        elif hasattr(parameter, "requires_grad"):
            try:
                cast(Any, parameter).requires_grad = False
            except (AttributeError, TypeError) as exc:
                raise RuntimeSetupError(
                    "failed to freeze a Carbon encoder parameter",
                    details={"parameter_index": index},
                ) from exc
        if getattr(parameter, "requires_grad", False) is True:
            trainable += count
            trainable_indices.append(index)
    if trainable_indices:
        raise RuntimeSetupError(
            "Carbon encoder parameters must be frozen",
            details={
                "trainable_parameter_count": trainable,
                "trainable_parameter_indices": trainable_indices,
            },
        )
    return total, trainable


def _resolve_device(device: str | None) -> str:
    """Resolve the encoder device, defaulting to CUDA when available.

    ``None`` or ``"auto"`` selects ``"cuda"`` if a GPU is present (e.g. on a
    Hugging Face Jobs GPU flavor) and ``"cpu"`` otherwise; an explicit value is
    used verbatim. Resolution never imports torch eagerly, so the encoder stays
    importable without the ``[train]`` extra.
    """
    if device is not None and device != "auto":
        return device
    try:
        torch = importlib.import_module("torch")

        if bool(torch.cuda.is_available()):
            return "cuda"
    except Exception:  # pragma: no cover - depends on optional torch/accelerator
        return "cpu"
    return "cpu"


def _move_module_to_device(model: object, device: str) -> None:
    """Move a real torch module to ``device``; no-op for CPU or test fakes."""
    if device == "cpu":
        return
    to = getattr(model, "to", None)
    if callable(to):
        to(device)


def _move_inputs_to_device(tokenized: Mapping[str, object], device: str) -> Mapping[str, object]:
    """Move tokenizer output tensors to ``device``; no-op for CPU or fakes."""
    if device == "cpu":
        return tokenized
    moved: dict[str, object] = {}
    for key, value in tokenized.items():
        to = getattr(value, "to", None)
        moved[key] = to(device) if callable(to) else value
    return moved


def _tokenize(tokenizer: object, wrapped_windows: Sequence[str]) -> Mapping[str, object]:
    if not callable(tokenizer):
        raise InputError(
            "tokenizer must be callable",
            details={"type": type(tokenizer).__name__},
        )
    call = cast(Callable[..., object], tokenizer)
    payload = call(
        list(wrapped_windows),
        return_tensors="pt",
        padding=True,
        add_special_tokens=False,
    )
    if not isinstance(payload, Mapping):
        raise InputError(
            "tokenizer must return a mapping",
            details={"type": type(payload).__name__},
        )
    return payload


def _call_model(model: object, tokenized: Mapping[str, object]) -> object:
    if not callable(model):
        raise InputError(
            "model must be callable",
            details={"type": type(model).__name__},
        )
    call = cast(Callable[..., object], model)
    return call(**dict(tokenized), output_hidden_states=True)


def _hidden_rows_by_item(
    output: object, *, state_layer: int
) -> tuple[tuple[tuple[float, ...], ...], ...]:
    hidden_states = getattr(output, "hidden_states", None)
    if hidden_states is not None:
        if not isinstance(hidden_states, Sequence) or isinstance(hidden_states, bytes | str):
            raise InputError(
                "model output hidden_states must be a sequence",
                details={"type": type(hidden_states).__name__},
            )
        try:
            selected = hidden_states[state_layer]
        except IndexError as exc:
            raise InputError(
                "state_layer is outside model hidden_states",
                details={"state_layer": state_layer, "layers": len(hidden_states)},
            ) from exc
    else:
        selected = getattr(output, "last_hidden_state", None)
        if selected is None:
            raise InputError("model output must expose hidden_states or last_hidden_state")

    batch = _materialize(selected)
    if not isinstance(batch, Sequence) or isinstance(batch, bytes | str):
        raise InputError(
            "selected hidden states must be a batched sequence",
            details={"type": type(batch).__name__},
        )

    rows_by_item: list[tuple[tuple[float, ...], ...]] = []
    for item in cast(Sequence[object], batch):
        rows = _coerce_item_rows(item)
        rows_by_item.append(rows)
    return tuple(rows_by_item)


def _coerce_item_rows(item: object) -> tuple[tuple[float, ...], ...]:
    rows = _materialize(item)
    if not isinstance(rows, Sequence) or isinstance(rows, bytes | str):
        raise InputError(
            "hidden state batch item must be a token sequence",
            details={"type": type(rows).__name__},
        )
    out: list[tuple[float, ...]] = []
    for row in cast(Sequence[object], rows):
        values = _materialize(row)
        if not isinstance(values, Sequence) or isinstance(values, bytes | str):
            raise InputError(
                "hidden state row must be a numeric sequence",
                details={"type": type(values).__name__},
            )
        out.append(tuple(_coerce_float(value) for value in cast(Sequence[object], values)))
    return tuple(out)


def _coerce_float(value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise InputError(
            "hidden state values must be numeric",
            details={"type": type(value).__name__, "value": repr(value)},
        )
    return float(value)


def _materialize(value: object) -> object:
    out = value
    for attr in ("detach", "cpu"):
        method = getattr(out, attr, None)
        if callable(method):
            out = method()
    tolist = getattr(out, "tolist", None)
    if callable(tolist):
        out = tolist()
    return out


def _coerce_encoder_hash(value: bytes | str | None) -> bytes | None:
    if value is None:
        return None
    if isinstance(value, bytes):
        if len(value) != hashlib.sha256().digest_size:
            raise InputError(
                "encoder_hash bytes must be 32 bytes",
                details={"len": len(value)},
            )
        return value
    if isinstance(value, str):
        raw = value.removeprefix("sha256:")
        try:
            out = bytes.fromhex(raw)
        except ValueError as exc:
            raise InputError("encoder_hash string must be hex or sha256:<hex>") from exc
        if len(out) != hashlib.sha256().digest_size:
            raise InputError(
                "encoder_hash string must encode 32 bytes",
                details={"len": len(out)},
            )
        return out
    raise InputError(
        "encoder_hash must be bytes, str, or None",
        details={"type": type(value).__name__},
    )


def _verify_local_encoder_weights(model_id: str, *, expected_hash: bytes) -> None:
    model_dir = Path(model_id).expanduser()
    if not model_dir.is_dir():
        raise RuntimeSetupError(
            "manifest-backed Carbon loading requires a local model directory",
            details={"model_id": model_id},
            remediation="mount the pinned Carbon checkpoint and set encoder.model_id to that path",
        )
    expected = f"sha256:{expected_hash.hex()}"
    observed_weights = encoder_weights_hash(model_dir)
    if observed_weights == expected:
        return
    try:
        observed_runtime = encoder_runtime_hash(model_dir)
    except InputError as exc:
        raise RuntimeSetupError(
            "local Carbon runtime does not match the committed encoder hash",
            details={
                "model_dir": str(model_dir),
                "expected": expected,
                "observed_weights": observed_weights,
                "runtime_identity_error": str(exc),
            },
            remediation="mount the exact encoder runtime committed by the model manifest",
        ) from exc
    if observed_runtime != expected:
        raise RuntimeSetupError(
            "local Carbon runtime does not match the committed encoder hash",
            details={
                "model_dir": str(model_dir),
                "expected": expected,
                "observed_weights": observed_weights,
                "observed_runtime": observed_runtime,
            },
            remediation="mount the exact encoder runtime committed by the model manifest",
        )
