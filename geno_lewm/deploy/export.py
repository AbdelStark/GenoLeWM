# SPDX-License-Identifier: Apache-2.0
"""Export a trained checkpoint to deployable safetensors weights (RFC-0018 §3.3).

Phase 1 converts the training-produced ``predictor_checkpoint.pt`` into the
``predictor.safetensors`` + ``action_encoder.safetensors`` artifacts that the
deploy runtime (:mod:`geno_lewm.deploy.runtime`) loads with ``strict=True``,
plus an ``export_report.json`` recording artifact identities. The ONNX / Core
ML / GGUF targets and int8/int4 quantization land later (#67–#70); this is the
minimal serialize step that unblocks packaging, scoring, eval, and the demo.
"""

from __future__ import annotations

import importlib
import json
from pathlib import Path
from typing import Any, Final

from geno_lewm.errors import ExportFormatError, InputError, RuntimeSetupError
from geno_lewm.provenance import sha256_file

__all__ = [
    "ACTION_ENCODER_ARTIFACT",
    "CHECKPOINT_NAME",
    "EXPORT_REPORT_NAME",
    "GENERATED_BY",
    "PREDICTOR_ARTIFACT",
    "SCHEMA_VERSION",
    "export_checkpoint",
]

SCHEMA_VERSION: Final = "1.0.0"
GENERATED_BY: Final = "geno_lewm.deploy.export"
CHECKPOINT_NAME: Final = "predictor_checkpoint.pt"
PREDICTOR_ARTIFACT: Final = "predictor.safetensors"
ACTION_ENCODER_ARTIFACT: Final = "action_encoder.safetensors"
EXPORT_REPORT_NAME: Final = "export_report.json"

_STATE_COMPONENTS: Final = (
    ("predictor", PREDICTOR_ARTIFACT),
    ("action_encoder", ACTION_ENCODER_ARTIFACT),
)


def export_checkpoint(
    checkpoint_path: Path,
    output_dir: Path,
    *,
    overwrite: bool = False,
) -> dict[str, Any]:
    """Convert a training checkpoint into deploy-ready safetensors artifacts.

    Reads the ``torch.save`` checkpoint at ``checkpoint_path``, writes the
    predictor and action-encoder ``state_dict`` tensors as safetensors files
    under ``output_dir``, and emits ``export_report.json``. Returns the report.
    """
    checkpoint_path = Path(checkpoint_path)
    output_dir = Path(output_dir)
    if not checkpoint_path.is_file():
        raise InputError(
            "checkpoint file does not exist",
            details={"path": str(checkpoint_path)},
        )
    _prepare_output_dir(output_dir, overwrite=overwrite)

    payload = _load_checkpoint(checkpoint_path)
    artifacts: list[dict[str, Any]] = []
    for key, artifact_name in _STATE_COMPONENTS:
        state_dict = _require_state_dict(payload, key)
        destination = output_dir / artifact_name
        _save_safetensors(state_dict, destination)
        artifacts.append(
            {
                "component": key,
                "file": artifact_name,
                "sha256": sha256_file(destination),
                "size_bytes": destination.stat().st_size,
                "tensors": len(state_dict),
            }
        )

    report: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "generated_by": GENERATED_BY,
        "format": "safetensors",
        "checkpoint": {
            "file": checkpoint_path.name,
            "sha256": sha256_file(checkpoint_path),
            "size_bytes": checkpoint_path.stat().st_size,
            "schema_version": payload.get("schema_version"),
            "run_id": payload.get("run_id"),
            "dataset_snapshot_id": payload.get("dataset_snapshot_id"),
            "steps_completed": payload.get("steps_completed"),
        },
        "artifacts": artifacts,
    }
    report_path = output_dir / EXPORT_REPORT_NAME
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return report


def _prepare_output_dir(output_dir: Path, *, overwrite: bool) -> None:
    if output_dir.exists() and not output_dir.is_dir():
        raise InputError(
            "output path is not a directory",
            details={"path": str(output_dir)},
        )
    for _, artifact_name in _STATE_COMPONENTS:
        existing = output_dir / artifact_name
        if existing.exists() and not overwrite:
            raise InputError(
                "export artifact already exists; pass overwrite to replace it",
                details={"path": str(existing)},
            )
    output_dir.mkdir(parents=True, exist_ok=True)


def _require_state_dict(payload: dict[str, Any], key: str) -> dict[str, Any]:
    state_dict = payload.get(key)
    if not isinstance(state_dict, dict) or not state_dict:
        raise ExportFormatError(
            f"checkpoint is missing a non-empty '{key}' state_dict",
            details={"component": key},
        )
    return state_dict


def _load_checkpoint(path: Path) -> dict[str, Any]:
    torch = _import_torch()
    try:
        # Our own training checkpoints store a config mapping alongside the
        # tensors, so weights_only=False is required to deserialize them.
        payload = torch.load(str(path), map_location="cpu", weights_only=False)
    except Exception as exc:  # surface any torch load failure uniformly
        raise ExportFormatError(
            "could not load training checkpoint",
            details={"path": str(path), "error": str(exc)},
        ) from exc
    if not isinstance(payload, dict):
        raise ExportFormatError(
            "training checkpoint must be a mapping",
            details={"path": str(path), "type": type(payload).__name__},
        )
    return payload


def _save_safetensors(state_dict: dict[str, Any], destination: Path) -> None:
    save_file = _import_safetensors_save()
    tensors = {name: _contiguous_cpu(name, tensor) for name, tensor in state_dict.items()}
    try:
        save_file(tensors, str(destination))
    except Exception as exc:  # surface any safetensors write failure uniformly
        raise ExportFormatError(
            "could not write safetensors artifact",
            details={"path": str(destination), "error": str(exc)},
        ) from exc


def _contiguous_cpu(name: str, tensor: Any) -> Any:
    detach = getattr(tensor, "detach", None)
    cpu = getattr(tensor, "cpu", None)
    contiguous = getattr(tensor, "contiguous", None)
    if not callable(detach) or not callable(cpu) or not callable(contiguous):
        raise ExportFormatError(
            "state_dict entry is not a tensor",
            details={"key": name, "type": type(tensor).__name__},
        )
    return tensor.detach().cpu().contiguous()


def _import_torch() -> Any:
    try:
        return importlib.import_module("torch")
    except ImportError as exc:
        raise RuntimeSetupError(
            "exporting a checkpoint requires PyTorch",
            remediation="install geno-lewm[train]",
        ) from exc


def _import_safetensors_save() -> Any:
    try:
        module = importlib.import_module("safetensors.torch")
    except ImportError as exc:
        raise RuntimeSetupError(
            "exporting safetensors requires the safetensors package",
            remediation="install geno-lewm[train]",
        ) from exc
    save_file = getattr(module, "save_file", None)
    if not callable(save_file):
        raise RuntimeSetupError("safetensors.torch.save_file is unavailable")
    return save_file
