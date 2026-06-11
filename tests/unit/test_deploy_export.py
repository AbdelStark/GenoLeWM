# SPDX-License-Identifier: Apache-2.0
"""Tests for ``geno_lewm.deploy.export`` (checkpoint -> deploy safetensors)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from geno_lewm.deploy.export import (
    ACTION_ENCODER_ARTIFACT,
    EXPORT_REPORT_NAME,
    PREDICTOR_ARTIFACT,
    export_checkpoint,
)
from geno_lewm.errors import ExportFormatError, InputError


def _write_checkpoint(path: Path, **extra: Any) -> tuple[Any, Any]:
    import torch

    predictor = torch.nn.Linear(4, 8)
    action_encoder = torch.nn.Linear(2, 4)
    payload: dict[str, Any] = {
        "schema_version": "1.0.0",
        "run_id": "run-test",
        "dataset_snapshot_id": "geno-lewm-data-v0.1.0-r1",
        "steps_completed": 3,
        "predictor": predictor.state_dict(),
        "action_encoder": action_encoder.state_dict(),
        "optimizer": {},
    }
    payload.update(extra)
    torch.save(payload, str(path))
    return predictor, action_encoder


def test_export_missing_checkpoint_raises_without_torch(tmp_path: Path) -> None:
    # Existence is validated before torch is imported, so this runs torch-less.
    with pytest.raises(InputError, match="does not exist"):
        export_checkpoint(tmp_path / "nope.pt", tmp_path / "model")


def test_export_rejects_non_directory_output(tmp_path: Path) -> None:
    checkpoint = tmp_path / "predictor_checkpoint.pt"
    checkpoint.write_bytes(b"not-a-real-checkpoint")
    not_a_dir = tmp_path / "file.txt"
    not_a_dir.write_text("x", encoding="utf-8")
    with pytest.raises(InputError, match="not a directory"):
        export_checkpoint(checkpoint, not_a_dir)


def test_export_refuses_existing_artifact_without_overwrite(tmp_path: Path) -> None:
    checkpoint = tmp_path / "predictor_checkpoint.pt"
    checkpoint.write_bytes(b"placeholder")
    out = tmp_path / "model"
    out.mkdir()
    (out / PREDICTOR_ARTIFACT).write_bytes(b"existing")
    with pytest.raises(InputError, match="already exists"):
        export_checkpoint(checkpoint, out)


def test_export_checkpoint_writes_artifacts_and_report(tmp_path: Path) -> None:
    pytest.importorskip("torch")
    pytest.importorskip("safetensors")
    checkpoint = tmp_path / "predictor_checkpoint.pt"
    _write_checkpoint(checkpoint)
    out = tmp_path / "model"

    report = export_checkpoint(checkpoint, out)

    assert (out / PREDICTOR_ARTIFACT).is_file()
    assert (out / ACTION_ENCODER_ARTIFACT).is_file()
    assert (out / EXPORT_REPORT_NAME).is_file()
    assert report["format"] == "safetensors"
    assert report["checkpoint"]["run_id"] == "run-test"
    assert report["checkpoint"]["dataset_snapshot_id"] == "geno-lewm-data-v0.1.0-r1"
    components = {artifact["component"]: artifact for artifact in report["artifacts"]}
    assert set(components) == {"predictor", "action_encoder"}
    for artifact in report["artifacts"]:
        assert artifact["sha256"].startswith("sha256:")
        assert artifact["size_bytes"] > 0
        assert artifact["tensors"] >= 1
    assert json.loads((out / EXPORT_REPORT_NAME).read_text(encoding="utf-8")) == report


def test_export_checkpoint_is_reproducible_across_runs(tmp_path: Path) -> None:
    pytest.importorskip("torch")
    pytest.importorskip("safetensors")
    checkpoint = tmp_path / "predictor_checkpoint.pt"
    _write_checkpoint(checkpoint)

    first = export_checkpoint(checkpoint, tmp_path / "first")
    second = export_checkpoint(checkpoint, tmp_path / "second")

    assert first == second


def test_export_checkpoint_canonicalizes_state_dict_order(tmp_path: Path) -> None:
    pytest.importorskip("torch")
    pytest.importorskip("safetensors")
    import torch

    checkpoint = tmp_path / "predictor_checkpoint.pt"
    predictor, action_encoder = _write_checkpoint(checkpoint)
    reordered_checkpoint = tmp_path / "predictor_checkpoint_reordered.pt"
    torch.save(
        {
            "schema_version": "1.0.0",
            "run_id": "run-test",
            "dataset_snapshot_id": "geno-lewm-data-v0.1.0-r1",
            "steps_completed": 3,
            "predictor": dict(reversed(list(predictor.state_dict().items()))),
            "action_encoder": dict(reversed(list(action_encoder.state_dict().items()))),
            "optimizer": {},
        },
        str(reordered_checkpoint),
    )

    original = export_checkpoint(checkpoint, tmp_path / "original")
    reordered = export_checkpoint(reordered_checkpoint, tmp_path / "reordered")

    assert _artifact_hashes(original) == _artifact_hashes(reordered)


def test_exported_safetensors_reload_strict(tmp_path: Path) -> None:
    pytest.importorskip("torch")
    import torch
    from safetensors.torch import load_file

    checkpoint = tmp_path / "predictor_checkpoint.pt"
    predictor, _ = _write_checkpoint(checkpoint)
    out = tmp_path / "model"
    export_checkpoint(checkpoint, out)

    fresh = torch.nn.Linear(4, 8)
    fresh.load_state_dict(load_file(str(out / PREDICTOR_ARTIFACT)), strict=True)
    for key, value in predictor.state_dict().items():
        assert torch.equal(fresh.state_dict()[key], value)


def test_export_overwrite_replaces_artifacts(tmp_path: Path) -> None:
    pytest.importorskip("torch")
    checkpoint = tmp_path / "predictor_checkpoint.pt"
    _write_checkpoint(checkpoint)
    out = tmp_path / "model"
    export_checkpoint(checkpoint, out)
    # Second run without overwrite is refused; with overwrite it succeeds.
    with pytest.raises(InputError, match="already exists"):
        export_checkpoint(checkpoint, out)
    report = export_checkpoint(checkpoint, out, overwrite=True)
    assert report["artifacts"]


def test_export_rejects_checkpoint_missing_predictor_state(tmp_path: Path) -> None:
    pytest.importorskip("torch")
    import torch

    checkpoint = tmp_path / "predictor_checkpoint.pt"
    torch.save(
        {"schema_version": "1.0.0", "action_encoder": torch.nn.Linear(2, 4).state_dict()},
        str(checkpoint),
    )
    out = tmp_path / "model"
    with pytest.raises(ExportFormatError, match="predictor"):
        export_checkpoint(checkpoint, out)


def test_export_rejects_unreadable_checkpoint(tmp_path: Path) -> None:
    pytest.importorskip("torch")
    checkpoint = tmp_path / "predictor_checkpoint.pt"
    checkpoint.write_bytes(b"not a torch archive")
    out = tmp_path / "model"
    with pytest.raises(ExportFormatError, match="could not load training checkpoint"):
        export_checkpoint(checkpoint, out)


def _artifact_hashes(report: dict[str, Any]) -> dict[str, str]:
    return {str(artifact["component"]): str(artifact["sha256"]) for artifact in report["artifacts"]}
