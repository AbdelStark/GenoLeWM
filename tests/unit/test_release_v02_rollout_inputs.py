"""Tests for v0.2 rollout input generation."""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from geno_lewm.encoder import read_embedding
from geno_lewm.encoder._identity import encoder_identity_hash
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
from tools.release import rollout_state_examples, v02_rollout_inputs

pa = pytest.importorskip("pyarrow")
pq = pytest.importorskip("pyarrow.parquet")


@pytest.mark.skipif(
    os.name == "nt",
    reason="cache-backed rollout inputs require secure POSIX dirfd primitives",
)
def test_v02_rollout_inputs_write_cache_specs_and_report(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
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
    _write_carbon_runtime(carbon_dir)
    runtime_hash = encoder_identity_hash(
        carbon_dir,
        state_contract_version="l2_normalized_v2",
    )
    _write_manifest(
        manifest,
        encoder_hash=runtime_hash,
    )

    monkeypatch.setattr(
        v02_rollout_inputs,
        "CarbonStateEncoder",
        lambda *_args, **_kwargs: _FakeEncoder(encoder_hash=runtime_hash),
    )
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
    )

    assert payload["ok"] is True
    assert payload["splits"]["rollout_phased_haplotypes"]["rows"] == 1
    assert payload["splits"]["rollout_synthetic_edit_chains"]["rows"] == 1
    assert payload["settings"]["cache_representation"] == "raw_pooled"
    assert payload["settings"]["normalize"] is True
    assert payload["settings"]["state_contract_version"] == "l2_normalized_v2"
    assert payload["settings"]["state_layer"] == 20
    assert payload["settings"]["pool_type"] == "centered_mean"
    assert payload["settings"]["pool_radius"] == 8
    assert payload["settings"]["dtype"] == "bf16"
    specs = rollout_state_examples.load_rollout_state_example_specs(phased_spec)
    assert len(specs) == 1
    assert specs[0].normalize is True
    assert len(specs[0].candidates) == 3
    cached = read_embedding(cache_dir, specs[0].source_state_key)
    assert cached is not None
    assert sum(value * value for value in cached) != pytest.approx(1.0)
    materialized = rollout_state_examples.generate_rollout_state_examples(
        specs,
        cache_dir=cache_dir,
    )
    source_state = materialized[0]["source_state"]
    assert isinstance(source_state, list)
    assert sum(value * value for value in source_state) == pytest.approx(1.0)


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
        )


def test_v02_rollout_inputs_rejects_normalization_override_mismatch(tmp_path: Path) -> None:
    artifact_root = tmp_path / "artifact"
    manifest_path = artifact_root / "manifest.json"
    _write_manifest(manifest_path)

    with pytest.raises(InputError, match="must match the model state contract"):
        v02_rollout_inputs.write_v02_rollout_inputs(
            artifact_root=artifact_root,
            placed_windows_jsonl=tmp_path / "placed.jsonl",
            gnomad_variants_parquet=tmp_path / "variants.parquet",
            model_manifest=manifest_path,
            carbon_model_dir=tmp_path / "carbon",
            phased_spec_jsonl=tmp_path / "phased-spec.jsonl",
            synthetic_spec_jsonl=tmp_path / "synthetic-spec.jsonl",
            cache_dir=tmp_path / "cache",
            output_report=tmp_path / "report.json",
            normalize=False,
        )


def test_v02_rollout_inputs_rejects_mismatched_carbon_weights(tmp_path: Path) -> None:
    artifact_root = tmp_path / "artifact"
    manifest_path = artifact_root / "manifest.json"
    carbon_dir = tmp_path / "carbon"
    _write_carbon_runtime(carbon_dir)
    expected_hash = encoder_identity_hash(
        carbon_dir,
        state_contract_version="l2_normalized_v2",
    )
    (carbon_dir / "model.safetensors").write_bytes(b"wrong-carbon-weights")
    _write_manifest(manifest_path, encoder_hash=expected_hash)

    with pytest.raises(InputError, match="Carbon weights do not match"):
        v02_rollout_inputs.write_v02_rollout_inputs(
            artifact_root=artifact_root,
            placed_windows_jsonl=tmp_path / "placed.jsonl",
            gnomad_variants_parquet=tmp_path / "variants.parquet",
            model_manifest=manifest_path,
            carbon_model_dir=carbon_dir,
            phased_spec_jsonl=tmp_path / "phased-spec.jsonl",
            synthetic_spec_jsonl=tmp_path / "synthetic-spec.jsonl",
            cache_dir=tmp_path / "cache",
            output_report=tmp_path / "report.json",
        )


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("state_layer", -1),
        ("pool_type", "global_mean"),
        ("pool_radius", 256),
        ("dtype", "fp32"),
    ],
)
def test_v02_rollout_inputs_rejects_encoder_setting_mismatch(
    tmp_path: Path,
    field: str,
    value: object,
) -> None:
    artifact_root = tmp_path / "artifact"
    manifest_path = artifact_root / "manifest.json"
    _write_manifest(manifest_path)

    with pytest.raises(InputError, match="must match the model encoder state contract"):
        v02_rollout_inputs.write_v02_rollout_inputs(
            artifact_root=artifact_root,
            placed_windows_jsonl=tmp_path / "placed.jsonl",
            gnomad_variants_parquet=tmp_path / "variants.parquet",
            model_manifest=manifest_path,
            carbon_model_dir=tmp_path / "carbon",
            phased_spec_jsonl=tmp_path / "phased-spec.jsonl",
            synthetic_spec_jsonl=tmp_path / "synthetic-spec.jsonl",
            cache_dir=tmp_path / "cache",
            output_report=tmp_path / "report.json",
            **{field: value},
        )


def test_rollout_cache_generation_rejects_normalized_encoder() -> None:
    with pytest.raises(InputError, match="normalize=False"):
        v02_rollout_inputs.encode_example_states(
            (),
            encoder=_NormalizedFakeEncoder(),
            encoder_hash=bytes.fromhex("1" * 64),
            state_layer=20,
            pool_type="centered_mean",
            pool_radius=8,
            dtype="fp32",
            batch_size=1,
        )


class _FakeEncoder:
    normalize = False
    state_layer = 20
    pool_type = "centered_mean"
    pool_radius = 8
    dtype = "bf16"

    def __init__(self, *, encoder_hash: str | None = None) -> None:
        identity = encoder_hash or sha256_bytes(b"carbon-weights")
        self.encoder_hash = bytes.fromhex(identity.removeprefix("sha256:"))

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

    def pooling_identity(self, window: str, edit_locus: int) -> tuple[str, int, int]:
        del window
        return self.pool_type, self.pool_radius, 1 + (edit_locus // 6)


class _NormalizedFakeEncoder(_FakeEncoder):
    normalize = True


def _write_manifest(
    path: Path,
    *,
    action_max_len: int = 16,
    encoder_hash: str | None = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    config_path = path.parent / "training_config.effective.yaml"
    config_path.write_text(
        (
            "encoder:\n"
            "  normalize: true\n"
            "  state_contract_version: l2_normalized_v2\n"
            "predictor:\n"
            "  d_state: 3\n"
            f"action:\n  max_len: {action_max_len}\n"
        ),
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
                hash=encoder_hash or sha256_bytes(b"carbon-weights"),
            ),
            predictor=ManifestArtifact(file="predictor.safetensors", hash="sha256:" + "2" * 64),
            action_encoder=ManifestArtifact(
                file="action_encoder.safetensors",
                hash="sha256:" + "3" * 64,
            ),
            calibration=ManifestArtifact(file="calibration.parquet", hash="sha256:" + "4" * 64),
            training=ManifestTraining(
                config_file="training_config.effective.yaml",
                hash=sha256_file(config_path),
                data_snapshot={"id": "test-dataset"},
            ),
            eval=ManifestArtifact(file="eval_metrics.json", hash="sha256:" + "6" * 64),
        ),
        path,
    )


def _write_carbon_runtime(path: Path) -> None:
    path.mkdir()
    (path / "model.safetensors").write_bytes(b"carbon-weights")
    (path / "config.json").write_text("{}\n", encoding="utf-8")
    (path / "tokenizer_config.json").write_text("{}\n", encoding="utf-8")
    (path / "tokenizer.py").write_text("# pinned tokenizer\n", encoding="utf-8")
    (path / "dna_config.json").write_text("{}\n", encoding="utf-8")
