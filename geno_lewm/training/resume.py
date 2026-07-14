# SPDX-License-Identifier: Apache-2.0
"""Closed state helpers for fresh-process Carbon training continuation."""

from __future__ import annotations

import importlib
import os
import random
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from geno_lewm.errors import InputError, RuntimeSetupError
from geno_lewm.provenance import canonical_json_sha256, sha256_bytes

__all__: list[str] = []

CHECKPOINT_SCHEMA_VERSION = "geno-lewm.carbon-resume-checkpoint.v2"
_RNG_KEYS = frozenset({"python", "numpy", "torch_cpu", "torch_cuda"})
_CHECKPOINT_KEYS = frozenset(
    {
        "schema_version",
        "source",
        "training_contract",
        "identities",
        "progress",
        "states",
        "trainer_state",
        "state_digests",
        "rng_state",
        "rng_state_digests",
        "metric_history",
        "payload_digest",
    }
)
_PROGRESS_KEYS = frozenset(
    {
        "steps_completed",
        "samples_consumed",
        "consumed_window_ids",
        "consumed_order_digest",
        "collapse_alert_count",
    }
)
_STATE_KEYS = frozenset({"predictor", "action_encoder", "optimizer"})


def capture_rng_state() -> dict[str, object]:
    """Capture every global RNG domain used by the training process."""
    numpy, torch = _runtime_modules()
    numpy_state = numpy.random.get_state()
    cuda_available = bool(torch.cuda.is_available())
    return {
        "python": _jsonable(random.getstate()),
        "numpy": {
            "algorithm": str(numpy_state[0]),
            "keys": [int(value) for value in numpy_state[1].tolist()],
            "position": int(numpy_state[2]),
            "has_gauss": int(numpy_state[3]),
            "cached_gaussian": float(numpy_state[4]),
        },
        "torch_cpu": torch.get_rng_state().cpu(),
        "torch_cuda": {
            "available": cuda_available,
            "device_count": int(torch.cuda.device_count()) if cuda_available else 0,
            "states": [state.cpu() for state in torch.cuda.get_rng_state_all()]
            if cuda_available
            else [],
        },
    }


def restore_rng_state(state: Mapping[str, object]) -> None:
    """Restore a closed RNG payload and reject device-domain drift."""
    if set(state) != _RNG_KEYS:
        raise InputError(
            "resume checkpoint RNG state set is incomplete",
            details={
                "missing": sorted(_RNG_KEYS - set(state)),
                "unexpected": sorted(set(state) - _RNG_KEYS),
            },
        )
    numpy, torch = _runtime_modules()
    numpy_state = _mapping(state["numpy"], "rng.numpy")
    if set(numpy_state) != {
        "algorithm",
        "keys",
        "position",
        "has_gauss",
        "cached_gaussian",
    }:
        raise InputError("resume checkpoint NumPy RNG state is not closed")
    cuda_state = _mapping(state["torch_cuda"], "rng.torch_cuda")
    if set(cuda_state) != {"available", "device_count", "states"}:
        raise InputError("resume checkpoint CUDA RNG state is not closed")
    keys = numpy_state.get("keys")
    cuda_states = cuda_state.get("states")
    if not isinstance(keys, list) or not isinstance(cuda_states, list):
        raise InputError("resume checkpoint RNG arrays must be lists")
    available = bool(torch.cuda.is_available())
    if cuda_state.get("available") is not available:
        raise InputError("CUDA RNG availability changed across checkpoint resume")
    expected_devices = int(torch.cuda.device_count()) if available else 0
    if cuda_state.get("device_count") != expected_devices or len(cuda_states) != expected_devices:
        raise InputError("CUDA RNG device count changed across checkpoint resume")
    python_state = _tuplify(state["python"])
    if not isinstance(python_state, tuple):
        raise InputError("resume checkpoint Python RNG state must resolve to a tuple")
    cached_gaussian = numpy_state.get("cached_gaussian")
    if isinstance(cached_gaussian, bool) or not isinstance(cached_gaussian, int | float):
        raise InputError("resume checkpoint NumPy cached Gaussian must be numeric")
    try:
        random.setstate(python_state)
        numpy.random.set_state(
            (
                _text(numpy_state, "algorithm"),
                numpy.asarray(keys, dtype=numpy.uint32),
                _integer(numpy_state, "position"),
                _integer(numpy_state, "has_gauss"),
                float(cached_gaussian),
            )
        )
        torch.set_rng_state(state["torch_cpu"])
        if available:
            torch.cuda.set_rng_state_all(cuda_states)
    except (TypeError, ValueError, RuntimeError) as exc:
        raise InputError("resume checkpoint RNG states could not be restored") from exc


def write_resume_checkpoint(
    path: Path,
    *,
    source: Mapping[str, object],
    training_contract: Mapping[str, object],
    identities: Mapping[str, object],
    progress: Mapping[str, object],
    states: Mapping[str, object],
    trainer_state: Mapping[str, object],
    rng_state: Mapping[str, object],
    metric_history: list[dict[str, object]],
) -> dict[str, object]:
    """Atomically write one closed checkpoint and return its in-memory payload."""
    if set(states) != _STATE_KEYS:
        raise InputError("resume checkpoint trainer-state set is incomplete")
    if set(rng_state) != _RNG_KEYS:
        raise InputError("resume checkpoint RNG state set is incomplete")
    consumed = progress.get("consumed_window_ids")
    if not isinstance(consumed, list) or any(not isinstance(item, str) for item in consumed):
        raise InputError("resume checkpoint consumed window order must be a list of strings")
    normalized_progress = dict(progress)
    normalized_progress["consumed_order_digest"] = canonical_json_sha256(consumed)
    if set(normalized_progress) != _PROGRESS_KEYS:
        raise InputError("resume checkpoint progress fields do not match the closed contract")
    _validate_progress(normalized_progress)
    payload: dict[str, object] = {
        "schema_version": CHECKPOINT_SCHEMA_VERSION,
        "source": dict(source),
        "training_contract": dict(training_contract),
        "identities": dict(identities),
        "progress": normalized_progress,
        "states": dict(states),
        "trainer_state": dict(trainer_state),
        "state_digests": {
            **{name: state_digest(value) for name, value in states.items()},
            "trainer": state_digest(trainer_state),
        },
        "rng_state": dict(rng_state),
        "rng_state_digests": {name: state_digest(value) for name, value in rng_state.items()},
        "metric_history": [dict(row) for row in metric_history],
    }
    payload["payload_digest"] = _payload_digest(payload)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    _, torch = _runtime_modules()
    try:
        with temporary.open("wb") as stream:
            torch.save(payload, stream)
            stream.flush()
            os.fsync(stream.fileno())
        temporary.replace(path)
        _fsync_directory(path.parent)
    finally:
        temporary.unlink(missing_ok=True)
    return payload


def load_resume_checkpoint(path: Path) -> dict[str, Any]:
    """Safely load and validate a closed production checkpoint."""
    if not path.is_file():
        raise InputError("resume checkpoint is missing", details={"path": str(path)})
    _, torch = _runtime_modules()
    try:
        payload = torch.load(path, map_location="cpu", weights_only=True)
    except Exception as exc:
        raise InputError(
            "resume checkpoint could not be loaded safely",
            details={"path": str(path), "error": str(exc)},
        ) from exc
    if not isinstance(payload, dict):
        raise InputError("resume checkpoint must contain a mapping")
    keys = set(payload)
    if keys != _CHECKPOINT_KEYS:
        raise InputError(
            "resume checkpoint fields do not match the closed contract",
            details={
                "missing": sorted(_CHECKPOINT_KEYS - keys),
                "unexpected": sorted(keys - _CHECKPOINT_KEYS),
            },
        )
    if payload.get("schema_version") != CHECKPOINT_SCHEMA_VERSION:
        raise InputError("resume checkpoint schema version is unsupported")
    if payload.get("payload_digest") != _payload_digest(payload):
        raise InputError("resume checkpoint payload digest does not match its contents")
    states = _mapping(payload.get("states"), "checkpoint.states")
    if set(states) != _STATE_KEYS:
        raise InputError("resume checkpoint trainer-state set is incomplete")
    trainer_state = _mapping(payload.get("trainer_state"), "checkpoint.trainer_state")
    expected_state_digests = {
        **{name: state_digest(value) for name, value in states.items()},
        "trainer": state_digest(trainer_state),
    }
    if payload.get("state_digests") != expected_state_digests:
        raise InputError("resume checkpoint trainer-state digests do not match")
    rng_state = _mapping(payload.get("rng_state"), "checkpoint.rng_state")
    if set(rng_state) != _RNG_KEYS:
        raise InputError("resume checkpoint RNG state set is incomplete")
    if payload.get("rng_state_digests") != {
        name: state_digest(value) for name, value in rng_state.items()
    }:
        raise InputError("resume checkpoint RNG-state digests do not match")
    progress = _mapping(payload.get("progress"), "checkpoint.progress")
    if set(progress) != _PROGRESS_KEYS:
        raise InputError("resume checkpoint progress fields do not match the closed contract")
    _validate_progress(progress)
    return payload


def state_digest(value: object) -> str:
    """Return a canonical digest for nested primitive and tensor state."""
    return canonical_json_sha256(_digest_view(value))


def _payload_digest(payload: Mapping[str, object]) -> str:
    return state_digest({key: value for key, value in payload.items() if key != "payload_digest"})


def _validate_progress(progress: Mapping[str, object]) -> None:
    steps_completed = progress.get("steps_completed")
    if isinstance(steps_completed, bool) or not isinstance(steps_completed, int):
        raise InputError("resume checkpoint completed-step cursor must be an integer")
    if steps_completed <= 0:
        raise InputError("resume checkpoint completed-step cursor must be positive")
    samples_consumed = progress.get("samples_consumed")
    if (
        isinstance(samples_consumed, bool)
        or not isinstance(samples_consumed, int)
        or samples_consumed <= 0
    ):
        raise InputError("resume checkpoint sample cursor must be a positive integer")
    collapse_alert_count = progress.get("collapse_alert_count")
    if (
        isinstance(collapse_alert_count, bool)
        or not isinstance(collapse_alert_count, int)
        or collapse_alert_count < 0
    ):
        raise InputError("resume checkpoint collapse-alert count must be non-negative")
    consumed = progress.get("consumed_window_ids")
    if (
        not isinstance(consumed, list)
        or any(not isinstance(item, str) for item in consumed)
        or len(consumed) != samples_consumed
    ):
        raise InputError("resume checkpoint consumed sample order does not match its cursor")
    if canonical_json_sha256(consumed) != progress.get("consumed_order_digest"):
        raise InputError("resume checkpoint consumed sample order digest does not match")


def _digest_view(value: object) -> object:
    _, torch = _runtime_modules()
    if isinstance(value, torch.Tensor):
        tensor = value.detach().cpu().contiguous()
        raw = tensor.reshape(-1).view(torch.uint8).numpy().tobytes()
        return {
            "type": "tensor",
            "dtype": str(tensor.dtype),
            "shape": list(tensor.shape),
            "content": sha256_bytes(raw),
        }
    if isinstance(value, Mapping):
        return {
            "type": "mapping",
            "items": [
                [_digest_key(key), _digest_view(item)]
                for key, item in sorted(value.items(), key=lambda pair: _digest_key(pair[0]))
            ],
        }
    if isinstance(value, tuple):
        return {"type": "tuple", "items": [_digest_view(item) for item in value]}
    if isinstance(value, list):
        return {"type": "list", "items": [_digest_view(item) for item in value]}
    if value is None or isinstance(value, str | int | float | bool):
        return {"type": type(value).__name__, "value": value}
    raise InputError(
        "resume checkpoint state contains an unsupported value",
        details={"type": type(value).__name__},
    )


def _digest_key(value: object) -> str:
    if isinstance(value, bool) or not isinstance(value, str | int):
        raise InputError("resume checkpoint mapping keys must be strings or integers")
    return f"{type(value).__name__}:{value}"


def _fsync_directory(path: Path) -> None:
    if os.name == "nt":  # pragma: no cover - Windows does not open directories this way.
        return
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _runtime_modules() -> tuple[Any, Any]:
    try:
        numpy = importlib.import_module("numpy")
        torch = importlib.import_module("torch")
    except ImportError as exc:  # pragma: no cover - guarded by the train extra.
        raise RuntimeSetupError(
            "Carbon checkpoint RNG state requires NumPy and PyTorch",
            remediation="install geno-lewm[train]",
        ) from exc
    return numpy, torch


def _mapping(value: object, label: str) -> dict[str, object]:
    if not isinstance(value, dict):
        raise InputError(f"{label} must be a mapping")
    return value


def _text(value: Mapping[str, object], key: str) -> str:
    item = value.get(key)
    if not isinstance(item, str) or not item:
        raise InputError(f"{key} must be non-empty text")
    return item


def _integer(value: Mapping[str, object], key: str) -> int:
    item = value.get(key)
    if isinstance(item, bool) or not isinstance(item, int):
        raise InputError(f"{key} must be an integer")
    return item


def _jsonable(value: object) -> object:
    if isinstance(value, tuple):
        return [_jsonable(item) for item in value]
    if isinstance(value, list):
        return [_jsonable(item) for item in value]
    if value is None or isinstance(value, str | int | float | bool):
        return value
    raise InputError("Python RNG state contains an unsupported value")


def _tuplify(value: object) -> object:
    if isinstance(value, list):
        return tuple(_tuplify(item) for item in value)
    return value
