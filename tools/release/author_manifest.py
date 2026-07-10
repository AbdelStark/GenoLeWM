# SPDX-License-Identifier: Apache-2.0
"""Author a deploy ``manifest.json`` for a trained GenoLeWM model directory.

Bridges the export step (``predictor.safetensors`` + ``action_encoder.safetensors``
from :mod:`geno_lewm.deploy.export`) and the deploy runtime / calibration /
model-package steps, all of which require an artifact-provenance contract ``manifest.json``. The
manifest commits, by SHA-256: the Carbon encoder identity, the predictor and
action-encoder artifacts, the training config + data snapshot, and the
calibration + eval evidence.

Calibration and eval artifacts are produced *after* the model can be loaded
(calibration needs the model; eval needs scores), so ``--allow-missing-evidence``
writes format-valid placeholder hashes for any not-yet-existing calibration/eval
file. Re-run *without* the flag once those artifacts exist to commit their real
hashes for the published, verifiable package. Run as::

    python -m tools.release.author_manifest \
        --model-dir ARTIFACTS/model \
        --training-config configs/first_experiment/train-carbon-500m-snv.yaml \
        --encoder-weights /carbon \
        --model-name geno-lewm --model-version 0.1.0 \
        --release-id geno-lewm-v0.1.0-r1 \
        --dataset-snapshot geno-lewm-data-v0.1.0-r1 \
        --allow-missing-evidence
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path
from typing import Any

from geno_lewm.config import load_config
from geno_lewm.encoder._identity import (
    encoder_identity_hash,
    encoder_weights_hash as _encoder_weights_hash,
)
from geno_lewm.errors import GenoLeWMError, InputError, exit_code_for
from geno_lewm.provenance import (
    SCHEMA_VERSION,
    Manifest,
    ManifestArtifact,
    ManifestEncoder,
    ManifestTraining,
    sha256_file,
    write_manifest,
)

GENERATED_BY = "tools.release.author_manifest"

DEFAULT_PREDICTOR_FILE = "predictor.safetensors"
DEFAULT_ACTION_ENCODER_FILE = "action_encoder.safetensors"
DEFAULT_CALIBRATION_FILE = "calibration.parquet"
DEFAULT_EVAL_FILE = "eval_report.md"
DEFAULT_CONFIG_NAME = "training_config.yaml"

# Format-valid sentinel committed for evidence (calibration/eval) that does not
# exist yet. It can never match a real file's hash, so the downstream
# model-package verifier still fails until the real artifact is committed.
_PLACEHOLDER_HASH = "sha256:" + ("0" * 64)


def encoder_weights_hash(encoder_weights: Path) -> str:
    """Compatibility wrapper for the shared encoder weight identity helper."""
    return _encoder_weights_hash(encoder_weights)


def author_manifest(
    *,
    model_dir: Path,
    training_config: Path,
    encoder_weights: Path,
    model_name: str,
    model_version: str,
    release_id: str,
    dataset_snapshot: str,
    predictor_file: str = DEFAULT_PREDICTOR_FILE,
    action_encoder_file: str = DEFAULT_ACTION_ENCODER_FILE,
    calibration_file: str = DEFAULT_CALIBRATION_FILE,
    eval_file: str = DEFAULT_EVAL_FILE,
    config_name: str = DEFAULT_CONFIG_NAME,
    allow_missing_evidence: bool = False,
) -> dict[str, Any]:
    """Write ``model_dir/manifest.json`` for a trained model directory."""
    model_dir = Path(model_dir)
    if not model_dir.is_dir():
        raise InputError(
            "model_dir must be an existing directory", details={"path": str(model_dir)}
        )
    for name in (model_name, model_version, release_id, dataset_snapshot):
        if not name:
            raise InputError("model_name, model_version, release_id, dataset_snapshot are required")

    # Copy the training config into the model dir so the runtime/calibration
    # commitment config travels with the package, then validate it.
    config_dest = model_dir / config_name
    _copy_into(training_config, config_dest)
    cfg = load_config(config_dest)

    encoder = ManifestEncoder(
        id=cfg.encoder.model_id,
        revision=cfg.encoder.revision,
        hash=encoder_identity_hash(
            Path(encoder_weights),
            state_contract_version=cfg.encoder.state_contract_version,
        ),
    )
    predictor = ManifestArtifact(
        file=predictor_file,
        hash=_required_artifact_hash(model_dir, predictor_file, "predictor"),
        dtype=cfg.predictor.dtype,
    )
    action_encoder = ManifestArtifact(
        file=action_encoder_file,
        hash=_required_artifact_hash(model_dir, action_encoder_file, "action_encoder"),
    )
    calibration = ManifestArtifact(
        file=calibration_file,
        hash=_evidence_hash(model_dir, calibration_file, allow_missing_evidence, "calibration"),
    )
    evaluation = ManifestArtifact(
        file=eval_file,
        hash=_evidence_hash(model_dir, eval_file, allow_missing_evidence, "eval"),
    )
    training = ManifestTraining(
        config_file=config_name,
        hash=sha256_file(config_dest),
        data_snapshot={"snapshot": dataset_snapshot},
    )

    manifest = Manifest(
        schema_version=SCHEMA_VERSION,
        model_name=model_name,
        model_version=model_version,
        release_id=release_id,
        encoder=encoder,
        predictor=predictor,
        action_encoder=action_encoder,
        calibration=calibration,
        training=training,
        eval=evaluation,
    )
    manifest_path = model_dir / "manifest.json"
    write_manifest(manifest, manifest_path)

    placeholders = sorted(
        name
        for name, artifact in (("calibration", calibration), ("eval", evaluation))
        if artifact.hash == _PLACEHOLDER_HASH
    )
    return {
        "generated_by": GENERATED_BY,
        "manifest": str(manifest_path),
        "model_id": manifest.model_id(),
        "release_id": release_id,
        "placeholder_evidence": placeholders,
    }


def _copy_into(source: Path, dest: Path) -> None:
    source = Path(source)
    if not source.is_file():
        raise InputError("file is required", details={"path": str(source)})
    if source.resolve() != dest.resolve():
        shutil.copy2(source, dest)


def _required_artifact_hash(model_dir: Path, file_name: str, label: str) -> str:
    path = model_dir / file_name
    if not path.is_file():
        raise InputError(
            f"{label} artifact is missing",
            details={"path": str(path)},
            remediation="export the checkpoint before authoring the manifest",
        )
    return sha256_file(path)


def _evidence_hash(model_dir: Path, file_name: str, allow_missing: bool, label: str) -> str:
    path = model_dir / file_name
    if path.is_file():
        return sha256_file(path)
    if allow_missing:
        return _PLACEHOLDER_HASH
    raise InputError(
        f"{label} evidence is missing",
        details={"path": str(path)},
        remediation=(
            "build the calibration/eval artifacts first, or pass "
            "--allow-missing-evidence to author a preliminary manifest"
        ),
    )


def main(argv: list[str] | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    try:
        summary = author_manifest(
            model_dir=args.model_dir,
            training_config=args.training_config,
            encoder_weights=args.encoder_weights,
            model_name=args.model_name,
            model_version=args.model_version,
            release_id=args.release_id,
            dataset_snapshot=args.dataset_snapshot,
            predictor_file=args.predictor_file,
            action_encoder_file=args.action_encoder_file,
            calibration_file=args.calibration_file,
            eval_file=args.eval_file,
            config_name=args.config_name,
            allow_missing_evidence=args.allow_missing_evidence,
        )
    except GenoLeWMError as exc:
        sys.stderr.write(f"error: {exc}\n")
        if exc.details:
            sys.stderr.write(json.dumps(exc.details, sort_keys=True) + "\n")
        return exit_code_for(exc)
    sys.stdout.write(json.dumps(summary, sort_keys=True) + "\n")
    return 0


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--training-config", type=Path, required=True)
    parser.add_argument(
        "--encoder-weights",
        type=Path,
        required=True,
        help="Carbon model directory or weight file used for the encoder hash.",
    )
    parser.add_argument("--model-name", required=True)
    parser.add_argument("--model-version", required=True)
    parser.add_argument("--release-id", required=True)
    parser.add_argument("--dataset-snapshot", required=True)
    parser.add_argument("--predictor-file", default=DEFAULT_PREDICTOR_FILE)
    parser.add_argument("--action-encoder-file", default=DEFAULT_ACTION_ENCODER_FILE)
    parser.add_argument("--calibration-file", default=DEFAULT_CALIBRATION_FILE)
    parser.add_argument("--eval-file", default=DEFAULT_EVAL_FILE)
    parser.add_argument("--config-name", default=DEFAULT_CONFIG_NAME)
    parser.add_argument(
        "--allow-missing-evidence",
        action="store_true",
        help="Write placeholder hashes for not-yet-existing calibration/eval files.",
    )
    return parser


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
