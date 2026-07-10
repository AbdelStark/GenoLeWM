"""Unit tests for the deploy manifest authoring tool."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

import geno_lewm.encoder._identity as identity_module
from geno_lewm.encoder._identity import encoder_runtime_hash
from geno_lewm.errors import InputError
from geno_lewm.provenance import load_manifest, sha256_file
from tools.release.author_manifest import author_manifest, encoder_weights_hash

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
    assert manifest.encoder.id == "/carbon"
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


def test_sharded_encoder_identity_commits_every_referenced_shard(tmp_path: Path) -> None:
    model_dir = tmp_path / "carbon-sharded"
    model_dir.mkdir()
    (model_dir / "model-00001-of-00002.safetensors").write_bytes(b"shard-one")
    second = model_dir / "model-00002-of-00002.safetensors"
    second.write_bytes(b"shard-two")
    (model_dir / "model.safetensors.index.json").write_text(
        json.dumps(
            {
                "weight_map": {
                    "layer.0": "model-00001-of-00002.safetensors",
                    "layer.1": "model-00002-of-00002.safetensors",
                }
            }
        ),
        encoding="utf-8",
    )

    first_hash = encoder_weights_hash(model_dir)
    second.write_bytes(b"changed-shard-two")

    assert encoder_weights_hash(model_dir) != first_hash


def test_corrected_encoder_runtime_identity_commits_tokenizer_code(tmp_path: Path) -> None:
    model_dir = tmp_path / "carbon-runtime"
    model_dir.mkdir()
    (model_dir / "model.safetensors").write_bytes(b"weights")
    (model_dir / "config.json").write_text("{}\n", encoding="utf-8")
    (model_dir / "tokenizer_config.json").write_text("{}\n", encoding="utf-8")
    (model_dir / "dna_config.json").write_text("{}\n", encoding="utf-8")
    tokenizer = model_dir / "tokenizer.py"
    tokenizer.write_text("# v1\n", encoding="utf-8")

    first_hash = encoder_runtime_hash(model_dir)
    tokenizer.write_text("# v2\n", encoding="utf-8")

    assert encoder_runtime_hash(model_dir) != first_hash


def test_corrected_encoder_runtime_identity_commits_local_encoder_implementation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model_dir = tmp_path / "carbon-runtime"
    model_dir.mkdir()
    (model_dir / "model.safetensors").write_bytes(b"weights")
    for name in ("config.json", "tokenizer_config.json", "dna_config.json"):
        (model_dir / name).write_text("{}\n", encoding="utf-8")
    (model_dir / "tokenizer.py").write_text("# upstream tokenizer\n", encoding="utf-8")
    tokenizer_implementation = tmp_path / "_dna_tokenizer.py"
    tokenizer_implementation.write_text("# tokenizer implementation v1\n", encoding="utf-8")
    pooling_implementation = tmp_path / "pooling.py"
    pooling_implementation.write_text("# pooling implementation v1\n", encoding="utf-8")
    monkeypatch.setattr(
        identity_module,
        "_ENCODER_IMPLEMENTATION_FILES",
        (
            ("geno_lewm/encoder/_dna_tokenizer.py", tokenizer_implementation),
            ("geno_lewm/encoder/pooling.py", pooling_implementation),
        ),
    )

    first_hash = encoder_runtime_hash(model_dir)
    pooling_implementation.write_text("# pooling implementation v2\n", encoding="utf-8")

    assert encoder_runtime_hash(model_dir) != first_hash


def test_encoder_weights_identity_matches_transformers_safetensors_precedence(
    tmp_path: Path,
) -> None:
    model_dir = tmp_path / "carbon-mixed-weights"
    model_dir.mkdir()
    shard = model_dir / "model-00001-of-00001.safetensors"
    shard.write_bytes(b"safetensors-shard")
    (model_dir / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": {"layer.0": shard.name}}),
        encoding="utf-8",
    )
    (model_dir / "pytorch_model.bin").write_bytes(b"unused-pytorch-monolith")

    first_hash = encoder_weights_hash(model_dir)
    (model_dir / "pytorch_model.bin").write_bytes(b"changed-but-still-unused")

    assert encoder_weights_hash(model_dir) == first_hash
    shard.write_bytes(b"changed-loaded-safetensors-shard")
    assert encoder_weights_hash(model_dir) != first_hash


def test_encoder_identity_rejects_missing_or_incomplete_runtime_paths(tmp_path: Path) -> None:
    with pytest.raises(InputError, match="path does not exist"):
        identity_module.encoder_weights_hash(tmp_path / "missing")

    empty_dir = tmp_path / "empty"
    empty_dir.mkdir()
    with pytest.raises(InputError, match="no recognized weight file"):
        identity_module.encoder_weights_hash(empty_dir)
    with pytest.raises(InputError, match="missing required identity files"):
        encoder_runtime_hash(empty_dir)

    weight_file = tmp_path / "model.safetensors"
    weight_file.write_bytes(b"weights")
    assert identity_module.encoder_weights_hash(weight_file) == sha256_file(weight_file)
    with pytest.raises(InputError, match="must be a directory"):
        encoder_runtime_hash(weight_file)


@pytest.mark.parametrize(
    ("payload", "message"),
    [
        ("{", "not readable JSON"),
        ("[]", "must be a JSON object"),
        ("{}", "non-empty weight_map"),
        (json.dumps({"weight_map": {}}), "non-empty weight_map"),
        (json.dumps({"weight_map": {"layer.0": ""}}), "non-empty shard paths"),
    ],
)
def test_sharded_encoder_identity_rejects_invalid_index_contract(
    tmp_path: Path,
    payload: str,
    message: str,
) -> None:
    index = tmp_path / "model.safetensors.index.json"
    index.write_text(payload, encoding="utf-8")

    with pytest.raises(InputError, match=message):
        identity_module.encoder_weights_hash(index)


def test_sharded_encoder_identity_rejects_escape_and_missing_shards(tmp_path: Path) -> None:
    index = tmp_path / "model.safetensors.index.json"
    index.write_text(
        json.dumps({"weight_map": {"layer.0": "../outside.safetensors"}}),
        encoding="utf-8",
    )
    with pytest.raises(InputError, match="must stay beside"):
        identity_module.encoder_weights_hash(index)

    index.write_text(
        json.dumps({"weight_map": {"layer.0": "missing.safetensors"}}),
        encoding="utf-8",
    )
    with pytest.raises(InputError, match="shard is missing"):
        identity_module.encoder_weights_hash(index)


def test_encoder_identity_dispatch_rejects_unknown_state_contract(tmp_path: Path) -> None:
    with pytest.raises(InputError, match="unsupported encoder state contract"):
        identity_module.encoder_identity_hash(
            tmp_path,
            state_contract_version="future_contract",
        )
