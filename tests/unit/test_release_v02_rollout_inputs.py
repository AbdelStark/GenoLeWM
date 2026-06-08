"""Tests for v0.2 rollout input generation."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from geno_lewm.encoder import read_embedding
from geno_lewm.errors import InputError
from geno_lewm.provenance import (
    Manifest,
    ManifestArtifact,
    ManifestEncoder,
    ManifestTraining,
    write_manifest,
)
from tools.release import rollout_state_examples, v02_rollout_inputs

pa = pytest.importorskip("pyarrow")
pq = pytest.importorskip("pyarrow.parquet")


def test_v02_rollout_inputs_write_cache_specs_and_report(tmp_path: Path) -> None:
    placed = tmp_path / "dataset" / "placed" / "gnomad-common-windows.jsonl"
    gnomad = tmp_path / "dataset" / "gnomad" / "variants.parquet"
    manifest = tmp_path / "model" / "manifest.json"
    carbon_dir = tmp_path / "carbon"
    cache_dir = tmp_path / "cache" / "window_embeddings"
    phased_spec = tmp_path / "eval" / "rollout_phased_haplotypes.specs.jsonl"
    synthetic_spec = tmp_path / "eval" / "rollout_synthetic_edit_chains.specs.jsonl"
    report = tmp_path / "eval" / "v02_rollout_inputs_report.json"

    placed.parent.mkdir(parents=True)
    sequence = "ACGTACGTACGTACGTACGT"
    placed.write_text(
        "\n".join(
            json.dumps(row)
            for row in [
                {
                    "chrom": "22",
                    "start_bp": 0,
                    "end_bp": len(sequence),
                    "sequence": sequence,
                    "record_id": "gnomad:22:0-20",
                },
                {
                    "chrom": "22",
                    "start_bp": 100,
                    "end_bp": 100 + len(sequence),
                    "sequence": "TGCATGCATGCATGCATGCA",
                    "record_id": "gnomad:22:100-120",
                },
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    gnomad.parent.mkdir(parents=True)
    pq.write_table(
        pa.Table.from_pylist(
            [
                {"chrom": "22", "pos": 2, "ref": "C", "alt": "A", "filter": "PASS"},
                {"chrom": "22", "pos": 4, "ref": "T", "alt": "A", "filter": "PASS"},
            ]
        ),
        gnomad,
    )
    _write_manifest(manifest)
    carbon_dir.mkdir()
    (carbon_dir / "config.json").write_text("{}", encoding="utf-8")

    payload = v02_rollout_inputs.write_v02_rollout_inputs(
        artifact_root=tmp_path,
        placed_windows_jsonl=placed,
        gnomad_variants_parquet=gnomad,
        model_manifest=manifest,
        carbon_model_dir=carbon_dir,
        cache_dir=cache_dir,
        phased_spec_jsonl=phased_spec,
        synthetic_spec_jsonl=synthetic_spec,
        output_report=report,
        examples_per_split=1,
        phased_horizon=2,
        synthetic_horizon=3,
        candidate_count=3,
        batch_size=2,
        dtype="fp32",
        encoder=_FakeEncoder(),
    )

    assert payload["ok"] is True
    assert payload["splits"]["rollout_phased_haplotypes"]["rows"] == 1
    assert payload["splits"]["rollout_synthetic_edit_chains"]["rows"] == 1
    specs = rollout_state_examples.load_rollout_state_example_specs(phased_spec)
    assert len(specs) == 1
    assert len(specs[0].candidates) == 3
    assert read_embedding(cache_dir, specs[0].source_state_key) is not None


def test_v02_rollout_inputs_rejects_horizon_above_model_action_limit(tmp_path: Path) -> None:
    placed = tmp_path / "dataset" / "placed" / "gnomad-common-windows.jsonl"
    gnomad = tmp_path / "dataset" / "gnomad" / "variants.parquet"
    manifest = tmp_path / "model" / "manifest.json"
    carbon_dir = tmp_path / "carbon"
    cache_dir = tmp_path / "cache" / "window_embeddings"
    phased_spec = tmp_path / "eval" / "rollout_phased_haplotypes.specs.jsonl"
    synthetic_spec = tmp_path / "eval" / "rollout_synthetic_edit_chains.specs.jsonl"
    report = tmp_path / "eval" / "v02_rollout_inputs_report.json"

    placed.parent.mkdir(parents=True)
    placed.write_text(
        json.dumps(
            {
                "chrom": "22",
                "start_bp": 0,
                "end_bp": 20,
                "sequence": "ACGTACGTACGTACGTACGT",
                "record_id": "gnomad:22:0-20",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    gnomad.parent.mkdir(parents=True)
    pq.write_table(
        pa.Table.from_pylist([{"chrom": "22", "pos": 2, "ref": "C", "alt": "A", "filter": "PASS"}]),
        gnomad,
    )
    _write_manifest(manifest, action_max_len=4)
    carbon_dir.mkdir()

    with pytest.raises(InputError, match="rollout horizon exceeds model action max_len"):
        v02_rollout_inputs.write_v02_rollout_inputs(
            artifact_root=tmp_path,
            placed_windows_jsonl=placed,
            gnomad_variants_parquet=gnomad,
            model_manifest=manifest,
            carbon_model_dir=carbon_dir,
            cache_dir=cache_dir,
            phased_spec_jsonl=phased_spec,
            synthetic_spec_jsonl=synthetic_spec,
            output_report=report,
            examples_per_split=1,
            phased_horizon=2,
            synthetic_horizon=5,
            candidate_count=3,
            batch_size=2,
            dtype="fp32",
            encoder=_FakeEncoder(),
        )


class _FakeEncoder:
    def encode_batch(
        self,
        windows: list[str],
        edit_loci: list[int],
    ) -> tuple[tuple[float, ...], ...]:
        return tuple(
            (
                float(len(window)),
                float(edit_locus),
                float(sum(ord(char) for char in window) % 97),
            )
            for window, edit_locus in zip(windows, edit_loci, strict=True)
        )


def _write_manifest(path: Path, *, action_max_len: int = 16) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    (path.parent / "training_config.effective.yaml").write_text(
        f"action:\n  max_len: {action_max_len}\n",
        encoding="utf-8",
    )
    write_manifest(
        Manifest(
            schema_version="1.0.0",
            model_name="test",
            model_version="0.0.0",
            release_id="test-release",
            encoder=ManifestEncoder(
                id="/carbon",
                revision="test-revision",
                hash="sha256:" + "1" * 64,
            ),
            predictor=ManifestArtifact(file="predictor.safetensors", hash="sha256:" + "2" * 64),
            action_encoder=ManifestArtifact(
                file="action_encoder.safetensors",
                hash="sha256:" + "3" * 64,
            ),
            calibration=ManifestArtifact(file="calibration.parquet", hash="sha256:" + "4" * 64),
            training=ManifestTraining(
                config_file="training_config.effective.yaml",
                hash="sha256:" + "5" * 64,
                data_snapshot={"id": "test-dataset"},
            ),
            eval=ManifestArtifact(file="eval_metrics.json", hash="sha256:" + "6" * 64),
        ),
        path,
    )
