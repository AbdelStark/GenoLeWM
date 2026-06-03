"""Unit tests for the deploy manifest authoring tool."""

from __future__ import annotations

from pathlib import Path

import pytest

from geno_lewm.errors import InputError
from geno_lewm.provenance import load_manifest, sha256_file
from tools.release.author_manifest import author_manifest

CONFIG = Path("configs/first_experiment/train-carbon-500m-snv.yaml")


def _setup_model_dir(tmp_path: Path) -> tuple[Path, Path]:
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    (model_dir / "predictor.safetensors").write_bytes(b"predictor-weights")
    (model_dir / "action_encoder.safetensors").write_bytes(b"action-weights")
    encoder_dir = tmp_path / "carbon"
    encoder_dir.mkdir()
    (encoder_dir / "model.safetensors").write_bytes(b"carbon-weights")
    return model_dir, encoder_dir


def _author(model_dir: Path, encoder_dir: Path, **overrides: object) -> dict[str, object]:
    kwargs: dict[str, object] = {
        "model_dir": model_dir,
        "training_config": CONFIG,
        "encoder_weights": encoder_dir,
        "model_name": "geno-lewm",
        "model_version": "0.1.0",
        "release_id": "geno-lewm-v0.1.0-r1",
        "dataset_snapshot": "geno-lewm-data-v0.1.0-r1",
    }
    kwargs.update(overrides)
    return author_manifest(**kwargs)  # type: ignore[arg-type]


def test_authors_preliminary_manifest_with_placeholder_evidence(tmp_path: Path) -> None:
    model_dir, encoder_dir = _setup_model_dir(tmp_path)

    summary = _author(model_dir, encoder_dir, allow_missing_evidence=True)

    manifest = load_manifest(model_dir / "manifest.json")
    assert manifest.release_id == "geno-lewm-v0.1.0-r1"
    assert manifest.model_name == "geno-lewm"
    assert manifest.encoder.id == "HuggingFaceBio/Carbon-500M"
    assert manifest.encoder.revision == "5d31d59b3c845b288a13aedb1358934196852eec"
    assert manifest.encoder.hash == sha256_file(encoder_dir / "model.safetensors")
    assert manifest.predictor.hash == sha256_file(model_dir / "predictor.safetensors")
    assert manifest.predictor.dtype == "bf16"
    assert manifest.action_encoder.hash == sha256_file(model_dir / "action_encoder.safetensors")
    # The training config travels with the package and is committed by hash.
    assert manifest.training.config_file == "training_config.yaml"
    assert (model_dir / "training_config.yaml").is_file()
    assert manifest.training.hash == sha256_file(model_dir / "training_config.yaml")
    assert manifest.training.data_snapshot == {"snapshot": "geno-lewm-data-v0.1.0-r1"}
    # Calibration + eval evidence are placeholders until those artifacts exist.
    assert summary["placeholder_evidence"] == ["calibration", "eval"]
    assert manifest.model_id().startswith("sha256:")
    assert summary["model_id"] == manifest.model_id()


def test_commits_real_evidence_hashes_when_present(tmp_path: Path) -> None:
    model_dir, encoder_dir = _setup_model_dir(tmp_path)
    (model_dir / "calibration.parquet").write_bytes(b"calibration-table")
    (model_dir / "eval_report.md").write_text("# eval\n", encoding="utf-8")

    summary = _author(model_dir, encoder_dir, allow_missing_evidence=False)

    manifest = load_manifest(model_dir / "manifest.json")
    assert manifest.calibration.hash == sha256_file(model_dir / "calibration.parquet")
    assert manifest.eval.hash == sha256_file(model_dir / "eval_report.md")
    assert summary["placeholder_evidence"] == []


def test_missing_evidence_without_flag_raises(tmp_path: Path) -> None:
    model_dir, encoder_dir = _setup_model_dir(tmp_path)

    with pytest.raises(InputError, match="calibration evidence is missing"):
        _author(model_dir, encoder_dir, allow_missing_evidence=False)


def test_missing_predictor_artifact_raises(tmp_path: Path) -> None:
    model_dir, encoder_dir = _setup_model_dir(tmp_path)
    (model_dir / "predictor.safetensors").unlink()

    with pytest.raises(InputError, match="predictor artifact is missing"):
        _author(model_dir, encoder_dir, allow_missing_evidence=True)


def test_encoder_weights_must_exist(tmp_path: Path) -> None:
    model_dir, _ = _setup_model_dir(tmp_path)

    with pytest.raises(InputError, match="encoder weights"):
        _author(model_dir, tmp_path / "missing", allow_missing_evidence=True)
