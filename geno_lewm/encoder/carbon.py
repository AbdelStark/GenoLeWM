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
from typing import Literal, cast

from geno_lewm.encoder.pooling import (
    DEFAULT_POOL_RADIUS_TOKENS,
    POOL_CENTERED_MEAN,
    POOL_GLOBAL_MEAN,
    pool_hidden_states,
)
from geno_lewm.encoder.windowing import canonicalize_dna, wrap_dna_for_tokenizer
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
            tokenizer, model = _load_transformers_components(
                model_id=model_id,
                revision=revision,
                dtype=dtype,
                local_files_only=local_files_only,
                trust_remote_code=trust_remote_code,
            )
        self.tokenizer = tokenizer
        self.model = model
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
        tokenized = _move_inputs_to_device(tokenized, self.device)
        output = _call_model(self.model, tokenized)
        rows_by_item = _hidden_rows_by_item(output, state_layer=self.state_layer)
        if len(rows_by_item) != len(windows):
            raise InputError(
                "encoder output batch size does not match input windows",
                details={"expected": len(windows), "observed": len(rows_by_item)},
            )

        encoded = tuple(
            pool_hidden_states(
                rows,
                edit_locus=edit_locus,
                pool_type=self.pool_type,
                pool_radius=self.pool_radius,
            ).vector
            for rows, edit_locus in zip(rows_by_item, edit_loci, strict=True)
        )
        if encoded:
            self._d_state = len(encoded[0])
        return encoded

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

    tokenizer_cls = getattr(transformers, "AutoTokenizer", None)
    model_cls = getattr(transformers, "AutoModel", None)
    if tokenizer_cls is None or model_cls is None:
        raise RuntimeSetupError("transformers must expose AutoTokenizer and AutoModel")

    kwargs = {
        "revision": revision,
        "local_files_only": local_files_only,
        "trust_remote_code": trust_remote_code,
    }
    tokenizer = tokenizer_cls.from_pretrained(model_id, **kwargs)
    model_kwargs = dict(kwargs)
    torch_dtype = _torch_dtype(torch, dtype)
    if torch_dtype is not None:
        model_kwargs["torch_dtype"] = torch_dtype
    model = model_cls.from_pretrained(model_id, **model_kwargs)
    return tokenizer, model


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
        import torch  # type: ignore[import-not-found]

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
    payload = call(list(wrapped_windows), return_tensors="pt", padding=True)
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
