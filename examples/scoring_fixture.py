# SPDX-License-Identifier: Apache-2.0
"""Small deterministic scorer fixture for executable tutorial notebooks.

The fixture uses the real runtime/scoring/receipt code paths with tiny
in-memory model components. It is not a learned model and should not be used as
model-quality evidence.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

from geno_lewm.action import EditSpec, RelEdit
from geno_lewm.deploy import GenoLeWMRuntime
from geno_lewm.encoder._normalization import l2_normalize_state
from geno_lewm.provenance import (
    SCHEMA_VERSION,
    DtypeConfig,
    Manifest,
    ManifestArtifact,
    ManifestEncoder,
    ManifestTraining,
    PoolingConfig,
    sha256_bytes,
    sha256_file,
    write_manifest,
)
from geno_lewm.surprise import CalibrationBucket, CalibrationTable

REFERENCE_WINDOW = "ACGT"
REFERENCE_FASTA_SEQUENCE = "ACGT" * 3072
CLINVAR_LIKE_VARIANT = EditSpec(chrom="1", pos=1, ref="A", alt="T")

POOLING_CONFIG = PoolingConfig(
    state_layer=20,
    pool_type="centered_mean",
    pool_radius=8,
    normalize=True,
)
DTYPE_CONFIG = DtypeConfig(encoder_dtype="bf16", predictor_dtype="bf16")


class NucleotideFrequencyEncoder:
    """Encode a DNA window as normalized A/C/G/T frequencies."""

    alphabet = "ACGT"

    def encode(self, window: str, *, edit_locus: int | None = None) -> tuple[float, ...]:
        del edit_locus
        denom = float(len(window))
        frequencies = tuple(window.count(base) / denom for base in self.alphabet)
        return l2_normalize_state(frequencies)


class RelativeEditActionEncoder:
    """Encode the first relative edit as a compact numeric tuple."""

    position_scale = 12_288.0

    def encode(self, edits: Sequence[RelEdit]) -> tuple[float, ...]:
        edit = edits[0]
        return (
            edit.rel_pos / self.position_scale,
            float(edit.edit_type),
            float(len(edit.ref_bases)),
            float(len(edit.alt_bases)),
        )


class EchoPredictor:
    """Return the source-state vector unchanged."""

    bias = 0.0

    def predict(self, state: Sequence[float], action: object) -> tuple[float, ...]:
        del action
        return tuple(float(value) + self.bias for value in state)


def fixture_calibration() -> CalibrationTable:
    """Return a deterministic calibration table for the tutorial fixture."""

    return CalibrationTable(
        buckets=(
            CalibrationBucket(
                bucket_id="other|mid|none",
                n_calibration=1_000,
                cdf=(0.0, 0.7, 1.0),
                sigma_grid=(0.0, 0.5, 1.0),
            ),
            CalibrationBucket(
                bucket_id="*",
                n_calibration=1_000,
                cdf=(0.0, 0.5, 1.0),
                sigma_grid=(0.0, 0.5, 1.0),
            ),
        )
    )


def make_fixture_model_dir(root: Path) -> Path:
    """Create a manifest-backed local model directory for notebook smoke runs."""

    root.mkdir(parents=True, exist_ok=True)
    (root / "predictor.safetensors").write_bytes(b"notebook predictor fixture\n")
    (root / "action_encoder.safetensors").write_bytes(b"notebook action fixture\n")
    (root / "calibration.parquet").write_bytes(b"notebook calibration fixture\n")
    (root / "train_config.yaml").write_text(
        "seed: 0\n"
        "schema_version: 1.1.0\n"
        "encoder:\n"
        "  normalize: true\n"
        "  state_contract_version: l2_normalized_v2\n",
        encoding="utf-8",
    )
    (root / "eval_report.md").write_text(
        "# Fixture evaluation\n\nSynthetic tutorial fixture only.\n",
        encoding="utf-8",
    )
    manifest = Manifest(
        schema_version=SCHEMA_VERSION,
        model_name="geno-lewm",
        model_version="0.2.1-notebook-fixture",
        release_id="geno-lewm-notebook-fixture-r1",
        encoder=ManifestEncoder(
            id="NucleotideFrequencyEncoder",
            revision="notebook-fixture",
            hash=sha256_bytes(b"notebook encoder fixture\n"),
        ),
        predictor=ManifestArtifact(
            file="predictor.safetensors",
            hash=sha256_file(root / "predictor.safetensors"),
            dtype=DTYPE_CONFIG.predictor_dtype,
        ),
        action_encoder=ManifestArtifact(
            file="action_encoder.safetensors",
            hash=sha256_file(root / "action_encoder.safetensors"),
            dtype=DTYPE_CONFIG.predictor_dtype,
        ),
        calibration=ManifestArtifact(
            file="calibration.parquet",
            hash=sha256_file(root / "calibration.parquet"),
            version="fixture-v1",
        ),
        training=ManifestTraining(
            config_file="train_config.yaml",
            hash=sha256_file(root / "train_config.yaml"),
            data_snapshot={"fixture": "notebook"},
        ),
        eval=ManifestArtifact(
            file="eval_report.md",
            hash=sha256_file(root / "eval_report.md"),
        ),
    )
    write_manifest(manifest, root / "manifest.json")
    return root


def build_fixture_runtime(model_dir: Path) -> GenoLeWMRuntime:
    """Return a runtime using the tutorial fixture components."""

    return GenoLeWMRuntime(
        model_dir,
        backend="cpu",
        encoder=NucleotideFrequencyEncoder(),
        action_encoder=RelativeEditActionEncoder(),
        predictor=EchoPredictor(),
        calibration=fixture_calibration(),
    )


def write_fixture_vcf(path: Path) -> Path:
    """Write a one-row VCF matching :data:`CLINVAR_LIKE_VARIANT`."""

    path.write_text(
        "##fileformat=VCFv4.2\n"
        "#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\n"
        "1\t1\t.\tA\tT\t.\tPASS\tCLNSIG=Pathogenic\n",
        encoding="utf-8",
    )
    return path


def write_fixture_fasta(path: Path) -> Path:
    """Write a local FASTA covering the tutorial variant."""

    path.write_text(f">1\n{REFERENCE_FASTA_SEQUENCE}\n", encoding="utf-8")
    return path


def verify_cli_args(receipt_path: Path, manifest_path: Path, window: str) -> list[str]:
    """Return ``geno-lewm-verify`` args for a fixture score receipt."""

    variant = CLINVAR_LIKE_VARIANT
    return [
        str(receipt_path),
        "--manifest",
        str(manifest_path),
        "--input-window",
        window,
        "--edit-chrom",
        variant.chrom,
        "--edit-pos",
        str(variant.pos),
        "--edit-ref",
        variant.ref,
        "--edit-alt",
        variant.alt,
        "--state-layer",
        str(POOLING_CONFIG.state_layer),
        "--pool-type",
        POOLING_CONFIG.pool_type,
        "--pool-radius",
        str(POOLING_CONFIG.pool_radius),
        "--normalize",
        "--encoder-dtype",
        DTYPE_CONFIG.encoder_dtype,
        "--predictor-dtype",
        DTYPE_CONFIG.predictor_dtype,
    ]
