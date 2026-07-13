"""Tests for release rollout-state row generation."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from geno_lewm._artifact_sources import ROLLOUT_STATES_GENERATED_BY
from geno_lewm.errors import InputError
from geno_lewm.provenance import (
    Manifest,
    ManifestArtifact,
    ManifestEncoder,
    ManifestTraining,
    sha256_bytes,
    sha256_file,
    write_manifest,
)
from tools.release import rollout_state_rows


def test_rollout_state_rows_generate_ranked_jsonl_and_report(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model_dir = _write_model_dir(tmp_path / "model")
    examples = tmp_path / "eval" / "rollout_state_examples.jsonl"
    output = tmp_path / "eval" / "rollout_states.jsonl"
    report = tmp_path / "eval" / "rollout_state_rows_report.json"
    examples.parent.mkdir()
    examples.write_text(json.dumps(_example_row()) + "\n", encoding="utf-8")

    monkeypatch.setattr(
        rollout_state_rows,
        "_predictor_fn_from_model_dir",
        lambda _model_dir: lambda _example: (0.9, 0.1),
    )
    payload = rollout_state_rows.write_rollout_state_artifacts(
        examples_jsonl=examples,
        model_dir=model_dir,
        artifact_root=tmp_path,
        output_jsonl=output,
        output_report=report,
        command=("python", "-m", "tools.release.rollout_state_rows"),
    )

    row = json.loads(output.read_text(encoding="utf-8"))
    assert row["generated_by"] == ROLLOUT_STATES_GENERATED_BY
    assert row["id"] == "phased-k2-a"
    assert row["k"] == 2
    assert row["target_rank"] == 1
    assert row["baseline_target_rank"] == 2
    assert row["predicted_state"] == [0.9, 0.1]
    assert payload["generated_by"] == "tools.release.rollout_state_rows"
    assert payload["rows"] == 1
    assert payload["splits"] == ["rollout_phased_haplotypes"]
    assert payload["horizons"] == [2]
    assert payload["state_contract"] == {
        "version": "legacy_raw_v1",
        "encoder_hash": sha256_bytes(b"predictor").removeprefix("sha256:"),
        "state_layer": 20,
        "pool_type": "centered_mean",
        "pool_radius": 8,
        "dtype": "bf16",
        "normalize": False,
        "d_state": 2,
        "cache_schema_version": "3.0.0",
        "cached_state_value_contract": "raw_pooled_v1",
        "validated_against_examples": True,
    }
    assert payload["inputs"]["model_manifest"]["path"] == "model/manifest.json"
    assert payload["inputs"]["examples_jsonl"]["path"] == "eval/rollout_state_examples.jsonl"
    assert payload["outputs"]["rollout_states_jsonl"]["path"] == "eval/rollout_states.jsonl"
    assert payload["outputs"]["rollout_states_jsonl"]["sha256"] == sha256_file(output)


def test_rollout_state_examples_reject_stale_target_candidate(tmp_path: Path) -> None:
    examples = tmp_path / "rollout_state_examples.jsonl"
    row = _example_row()
    candidates = row["candidates"]
    assert isinstance(candidates, list)
    target = candidates[0]
    assert isinstance(target, dict)
    target["state"] = [0.5, 0.5]
    examples.write_text(json.dumps(row) + "\n", encoding="utf-8")

    with pytest.raises(InputError, match="target candidate state must match target_state"):
        rollout_state_rows.load_rollout_state_examples(examples)


def test_rollout_state_examples_reject_stale_schema_version(tmp_path: Path) -> None:
    examples = tmp_path / "rollout_state_examples.jsonl"
    row = _example_row()
    row["schema_version"] = "0.9.0"
    examples.write_text(json.dumps(row) + "\n", encoding="utf-8")

    with pytest.raises(InputError, match="unsupported rollout-state example schema_version"):
        rollout_state_rows.load_rollout_state_examples(examples)


def test_rollout_state_rows_reject_prediction_dimension_mismatch(tmp_path: Path) -> None:
    examples = tmp_path / "rollout_state_examples.jsonl"
    examples.write_text(json.dumps(_example_row()) + "\n", encoding="utf-8")
    parsed = rollout_state_rows.load_rollout_state_examples(examples)

    with pytest.raises(InputError, match="share the same dimension"):
        rollout_state_rows.generate_rollout_state_rows(
            parsed,
            predictor_fn=lambda example: (1.0, 0.0, 0.0),
        )


@pytest.mark.parametrize(
    ("normalized_examples", "normalized_model"),
    [(True, False), (False, True)],
)
def test_rollout_state_rows_reject_state_contract_mismatch(
    tmp_path: Path,
    normalized_examples: bool,
    normalized_model: bool,
) -> None:
    model_dir = _write_model_dir(tmp_path / "model", normalized_contract=normalized_model)
    examples = tmp_path / "rollout_state_examples.jsonl"
    examples.write_text(
        json.dumps(_example_row(normalize=normalized_examples)) + "\n",
        encoding="utf-8",
    )

    with pytest.raises(InputError, match="representation does not match model state contract"):
        rollout_state_rows.write_rollout_state_artifacts(
            examples_jsonl=examples,
            model_dir=model_dir,
            artifact_root=tmp_path,
            output_jsonl=tmp_path / "rollout_states.jsonl",
            output_report=tmp_path / "report.json",
        )


def test_rollout_state_rows_rejects_encoder_setting_mismatch(tmp_path: Path) -> None:
    model_dir = _write_model_dir(tmp_path / "model")
    examples = tmp_path / "rollout_state_examples.jsonl"
    examples.write_text(json.dumps(_example_row(state_layer=-1)) + "\n", encoding="utf-8")

    with pytest.raises(InputError, match="representation does not match model state contract"):
        rollout_state_rows.write_rollout_state_artifacts(
            examples_jsonl=examples,
            model_dir=model_dir,
            artifact_root=tmp_path,
            output_jsonl=tmp_path / "rollout_states.jsonl",
            output_report=tmp_path / "report.json",
        )


def test_rollout_state_examples_reject_invalid_edit_type(tmp_path: Path) -> None:
    examples = tmp_path / "rollout_state_examples.jsonl"
    row = _example_row()
    edits = row["edits"]
    assert isinstance(edits, list)
    edit = edits[0]
    assert isinstance(edit, dict)
    edit["edit_type"] = 99
    examples.write_text(json.dumps(row) + "\n", encoding="utf-8")

    with pytest.raises(InputError, match="edit_type is not a supported EditType"):
        rollout_state_rows.load_rollout_state_examples(examples)


def test_normalized_rollout_examples_require_unit_norm_values(tmp_path: Path) -> None:
    examples = tmp_path / "rollout_state_examples.jsonl"
    row = _example_row(normalize=True)
    row["source_state"] = [3.0, 4.0]
    candidates = row["candidates"]
    assert isinstance(candidates, list)
    source_candidate = candidates[1]
    assert isinstance(source_candidate, dict)
    source_candidate["state"] = [3.0, 4.0]
    examples.write_text(json.dumps(row) + "\n", encoding="utf-8")

    with pytest.raises(InputError, match="must be unit norm"):
        rollout_state_rows.load_rollout_state_examples(examples)


def test_normalized_rollout_rows_reject_non_unit_prediction(tmp_path: Path) -> None:
    examples = tmp_path / "rollout_state_examples.jsonl"
    examples.write_text(json.dumps(_example_row(normalize=True)) + "\n", encoding="utf-8")
    parsed = rollout_state_rows.load_rollout_state_examples(examples)

    with pytest.raises(InputError, match="predicted_state"):
        rollout_state_rows.generate_rollout_state_rows(
            parsed,
            predictor_fn=lambda example: (0.9, 0.1),
        )


def test_rollout_state_rows_reject_model_state_width_mismatch(tmp_path: Path) -> None:
    model_dir = _write_model_dir(tmp_path / "model")
    examples = tmp_path / "rollout_state_examples.jsonl"
    row = _example_row()
    row["source_state"] = [0.0, 1.0, 0.0]
    row["target_state"] = [1.0, 0.0, 0.0]
    candidates = row["candidates"]
    assert isinstance(candidates, list)
    candidates[0]["state"] = [1.0, 0.0, 0.0]
    candidates[1]["state"] = [0.0, 1.0, 0.0]
    examples.write_text(json.dumps(row) + "\n", encoding="utf-8")

    with pytest.raises(InputError, match="representation does not match model state contract"):
        rollout_state_rows.write_rollout_state_artifacts(
            examples_jsonl=examples,
            model_dir=model_dir,
            artifact_root=tmp_path,
            output_jsonl=tmp_path / "rollout_states.jsonl",
            output_report=tmp_path / "report.json",
        )


def _example_row(
    *,
    normalize: bool | None = None,
    state_layer: int = 20,
) -> dict[str, object]:
    source_key = _state_key("1", state_layer=state_layer)
    target_key = _state_key("2", state_layer=state_layer)
    row: dict[str, object] = {
        "schema_version": "1.2.0",
        "generated_by": "tools.release.rollout_state_examples",
        "cache_schema_version": "3.0.0",
        "cached_state_value_contract": "raw_pooled_v1",
        "materialized_state_contract": ("l2_normalized_v2" if normalize else "legacy_raw_v1"),
        "id": "phased-k2-a",
        "split": "rollout_phased_haplotypes",
        "normalize": bool(normalize),
        "source_state": [0.0, 1.0],
        "source_state_key": source_key,
        "target_state": [1.0, 0.0],
        "target_state_key": target_key,
        "target_candidate_id": "target",
        "edits": [
            {"rel_pos": 3, "edit_type": 0, "ref_bases": "A", "alt_bases": "C"},
            {"rel_pos": 7, "edit_type": 0, "ref_bases": "G", "alt_bases": "T"},
        ],
        "candidates": [
            {"id": "target", "state": [1.0, 0.0], "state_key": target_key},
            {
                "id": "source-like",
                "state": [0.0, 1.0],
                "state_key": _state_key("3", state_layer=state_layer),
            },
        ],
    }
    return row


def _state_key(seed: str, *, state_layer: int) -> dict[str, object]:
    return {
        "window_hash": seed * 64,
        "encoder_hash": sha256_bytes(b"predictor").removeprefix("sha256:"),
        "state_layer": state_layer,
        "pool_type": "centered_mean",
        "pool_radius": 8,
        "center_token": 0,
        "dtype": "bf16",
    }


def _write_model_dir(root: Path, *, normalized_contract: bool = False) -> Path:
    root.mkdir(parents=True)
    (root / "predictor.safetensors").write_bytes(b"predictor")
    (root / "action_encoder.safetensors").write_bytes(b"action")
    (root / "calibration.parquet").write_bytes(b"calibration")
    (root / "eval_metrics.json").write_text("{}", encoding="utf-8")
    config = (
        "schema_version: 1.1.0\n"
        "predictor:\n  d_state: 2\n"
        "encoder:\n  normalize: true\n  state_contract_version: legacy_raw_v1\n"
    )
    if normalized_contract:
        config = config.replace("legacy_raw_v1", "l2_normalized_v2")
    (root / "train_config.yaml").write_text(config, encoding="utf-8")
    manifest = Manifest(
        schema_version="1.0.0",
        model_name="GenoLeWM",
        model_version="0.2.0",
        release_id="geno-lewm-v0.2.0-r1",
        encoder=ManifestEncoder(
            id="carbon-500m",
            revision="main",
            hash=sha256_file(root / "predictor.safetensors"),
        ),
        predictor=ManifestArtifact(
            file="predictor.safetensors",
            hash=sha256_file(root / "predictor.safetensors"),
        ),
        action_encoder=ManifestArtifact(
            file="action_encoder.safetensors",
            hash=sha256_file(root / "action_encoder.safetensors"),
        ),
        calibration=ManifestArtifact(
            file="calibration.parquet",
            hash=sha256_file(root / "calibration.parquet"),
        ),
        training=ManifestTraining(
            config_file="train_config.yaml",
            hash=sha256_file(root / "train_config.yaml"),
            data_snapshot={"id": "geno-lewm-data-v0.2.0-r1"},
        ),
        eval=ManifestArtifact(
            file="eval_metrics.json",
            hash=sha256_file(root / "eval_metrics.json"),
        ),
    )
    write_manifest(manifest, root / "manifest.json")
    return root
