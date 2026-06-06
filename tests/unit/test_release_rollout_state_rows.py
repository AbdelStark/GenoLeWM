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
    sha256_file,
    write_manifest,
)
from tools.release import rollout_state_rows


def test_rollout_state_rows_generate_ranked_jsonl_and_report(tmp_path: Path) -> None:
    model_dir = _write_model_dir(tmp_path / "model")
    examples = tmp_path / "eval" / "rollout_state_examples.jsonl"
    output = tmp_path / "eval" / "rollout_states.jsonl"
    report = tmp_path / "eval" / "rollout_state_rows_report.json"
    examples.parent.mkdir()
    examples.write_text(json.dumps(_example_row()) + "\n", encoding="utf-8")

    payload = rollout_state_rows.write_rollout_state_artifacts(
        examples_jsonl=examples,
        model_dir=model_dir,
        artifact_root=tmp_path,
        output_jsonl=output,
        output_report=report,
        command=("python", "-m", "tools.release.rollout_state_rows"),
        predictor_fn=lambda example: (0.9, 0.1),
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


def test_rollout_state_rows_reject_prediction_dimension_mismatch(tmp_path: Path) -> None:
    examples = tmp_path / "rollout_state_examples.jsonl"
    examples.write_text(json.dumps(_example_row()) + "\n", encoding="utf-8")
    parsed = rollout_state_rows.load_rollout_state_examples(examples)

    with pytest.raises(InputError, match="share the same dimension"):
        rollout_state_rows.generate_rollout_state_rows(
            parsed,
            predictor_fn=lambda example: (1.0, 0.0, 0.0),
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


def _example_row() -> dict[str, object]:
    return {
        "generated_by": "tools.release.rollout_state_examples",
        "id": "phased-k2-a",
        "split": "rollout_phased_haplotypes",
        "source_state": [0.0, 1.0],
        "target_state": [1.0, 0.0],
        "target_candidate_id": "target",
        "edits": [
            {"rel_pos": 3, "edit_type": 0, "ref_bases": "A", "alt_bases": "C"},
            {"rel_pos": 7, "edit_type": 0, "ref_bases": "G", "alt_bases": "T"},
        ],
        "candidates": [
            {"id": "target", "state": [1.0, 0.0]},
            {"id": "source-like", "state": [0.0, 1.0]},
        ],
    }


def _write_model_dir(root: Path) -> Path:
    root.mkdir(parents=True)
    (root / "predictor.safetensors").write_bytes(b"predictor")
    (root / "action_encoder.safetensors").write_bytes(b"action")
    (root / "calibration.parquet").write_bytes(b"calibration")
    (root / "eval_metrics.json").write_text("{}", encoding="utf-8")
    (root / "train_config.yaml").write_text("predictor: {}\n", encoding="utf-8")
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
