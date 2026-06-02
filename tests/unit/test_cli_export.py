# SPDX-License-Identifier: Apache-2.0
"""Tests for the ``geno-lewm-export`` CLI."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from geno_lewm.cli import _dispatch
from geno_lewm.cli.export import app


def test_export_cli_requires_checkpoint(tmp_path: Path) -> None:
    rc = _dispatch.run_app(
        app,
        argv=["--output-dir", str(tmp_path / "model"), "--no-banner", "--quiet"],
    )
    assert rc != 0


def test_export_cli_requires_output_dir(tmp_path: Path) -> None:
    checkpoint = tmp_path / "predictor_checkpoint.pt"
    checkpoint.write_bytes(b"placeholder")
    rc = _dispatch.run_app(
        app,
        argv=["--checkpoint", str(checkpoint), "--no-banner", "--quiet"],
    )
    assert rc != 0


def test_export_cli_writes_artifacts(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    pytest.importorskip("torch")
    import torch

    checkpoint = tmp_path / "predictor_checkpoint.pt"
    torch.save(
        {
            "schema_version": "1.0.0",
            "run_id": "cli-run",
            "predictor": torch.nn.Linear(4, 8).state_dict(),
            "action_encoder": torch.nn.Linear(2, 4).state_dict(),
        },
        str(checkpoint),
    )
    out = tmp_path / "model"
    rc = _dispatch.run_app(
        app,
        argv=["--checkpoint", str(checkpoint), "--output-dir", str(out), "--no-banner", "--quiet"],
    )
    assert rc == 0
    assert (out / "predictor.safetensors").is_file()
    assert (out / "action_encoder.safetensors").is_file()
    payload = json.loads(capsys.readouterr().out.strip().splitlines()[-1])
    assert payload["format"] == "safetensors"
    assert payload["checkpoint"]["run_id"] == "cli-run"
