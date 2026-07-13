# SPDX-License-Identifier: Apache-2.0
"""Tests for the correction-control model evidence manifest."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest

from geno_lewm.errors import InputError
from geno_lewm.provenance import sha256_file

CONFIG = Path("configs/correction_control/train-carbon-500m-snv-l2-smoke-v1.yaml")
COMMIT_SHA = "a" * 40
RUN_ID = "correction-control-l2-p1-smoke-v1"
SNAPSHOT_ID = "geno-lewm-data-correction-control-l2-p1-proof-v1"
RUNTIME_HASH = "sha256:71d27acc26bc809d850e9cd8cf558762c5bd4c1d611e2778c1614c0a8be77b38"
WEIGHTS_HASH = "sha256:e257506988203fdb8bb46976ee81c97e24f29073754bbff70137c7704dbadaa8"


def test_authors_non_release_manifest_from_coherent_proof_evidence(tmp_path: Path) -> None:
    from tools.research.correction_control_model_manifest import (
        CorrectionControlModelManifestRequest,
        author_correction_control_model_manifest,
    )

    request = _coherent_request(tmp_path, CorrectionControlModelManifestRequest)

    manifest = author_correction_control_model_manifest(request)

    written = json.loads(request.output_json.read_text(encoding="utf-8"))
    assert written == manifest
    assert manifest["artifact_kind"] == "correction_control_engineering_evidence"
    assert manifest["release_status"] == "non_release"
    assert manifest["model_quality_evaluated"] is False
    assert manifest["run"] == {
        "commit_sha": COMMIT_SHA,
        "run_id": RUN_ID,
        "steps_completed": 50,
    }
    assert manifest["encoder"]["runtime_hash"] == RUNTIME_HASH
    assert manifest["encoder"]["weights_hash"] == WEIGHTS_HASH
    assert manifest["encoder"]["revision"] == "5d31d59b3c845b288a13aedb1358934196852eec"
    assert manifest["encoder"]["state_contract_version"] == "l2_normalized_v2"
    assert manifest["training"]["config"]["sha256"] == sha256_file(request.training_config)
    assert manifest["training"]["checkpoint"]["sha256"] == sha256_file(request.checkpoint)
    assert manifest["dataset"]["snapshot_id"] == SNAPSHOT_ID
    assert manifest["dataset"]["manifest"]["sha256"] == sha256_file(request.dataset_manifest_json)
    assert manifest["export"]["predictor"]["sha256"] == sha256_file(
        request.model_dir / "predictor.safetensors"
    )
    assert manifest["export"]["action_encoder"]["sha256"] == sha256_file(
        request.model_dir / "action_encoder.safetensors"
    )
    assert "calibration" not in manifest
    assert "eval" not in manifest


def test_rejects_non_boolean_green_postflight_status(tmp_path: Path) -> None:
    from tools.research.correction_control_model_manifest import (
        CorrectionControlModelManifestRequest,
        author_correction_control_model_manifest,
    )

    request = _coherent_request(tmp_path, CorrectionControlModelManifestRequest)
    postflight = json.loads(request.correction_control_postflight_json.read_text(encoding="utf-8"))
    postflight["ok"] = 1
    _write_json(request.correction_control_postflight_json, postflight)

    with pytest.raises(InputError, match=r"correction_control_postflight\.ok"):
        author_correction_control_model_manifest(request)


def test_rejects_unknown_postflight_schema(tmp_path: Path) -> None:
    from tools.research.correction_control_model_manifest import (
        CorrectionControlModelManifestRequest,
        author_correction_control_model_manifest,
    )

    request = _coherent_request(tmp_path, CorrectionControlModelManifestRequest)
    postflight = json.loads(request.correction_control_postflight_json.read_text(encoding="utf-8"))
    postflight["schema_version"] = "2.0.0"
    _write_json(request.correction_control_postflight_json, postflight)

    with pytest.raises(InputError, match=r"correction_control_postflight\.schema_version"):
        author_correction_control_model_manifest(request)


def test_validator_rejects_manifest_claim_drift(tmp_path: Path) -> None:
    from tools.research.correction_control_model_manifest import (
        CorrectionControlModelManifestRequest,
        author_correction_control_model_manifest,
        validate_correction_control_model_manifest,
    )

    request = _coherent_request(tmp_path, CorrectionControlModelManifestRequest)
    author_correction_control_model_manifest(request)
    manifest = json.loads(request.output_json.read_text(encoding="utf-8"))
    manifest["release_status"] = "release"
    _write_json(request.output_json, manifest)

    with pytest.raises(InputError, match="does not match its bound evidence"):
        validate_correction_control_model_manifest(request)


def test_rejects_exported_predictor_hash_drift(tmp_path: Path) -> None:
    from tools.research.correction_control_model_manifest import (
        CorrectionControlModelManifestRequest,
        author_correction_control_model_manifest,
    )

    request = _coherent_request(tmp_path, CorrectionControlModelManifestRequest)
    (request.model_dir / "predictor.safetensors").write_bytes(b"tampered-predictor")

    with pytest.raises(InputError, match=r"export_report\.artifacts\.predictor\.sha256"):
        author_correction_control_model_manifest(request)


def test_cli_authors_then_validates_manifest_and_writes_receipt(tmp_path: Path) -> None:
    from tools.research.correction_control_model_manifest import (
        CorrectionControlModelManifestRequest,
        main,
    )

    request = _coherent_request(tmp_path, CorrectionControlModelManifestRequest)
    shared = _cli_args(request)

    assert main(["author", *shared]) == 0
    validation_report = request.model_dir / "manifest_validation.json"
    assert (
        main(
            [
                "validate",
                *shared,
                "--validation-report-json",
                str(validation_report),
            ]
        )
        == 0
    )

    receipt = json.loads(validation_report.read_text(encoding="utf-8"))
    assert receipt["ok"] is True
    assert receipt["artifact_kind"] == "correction_control_engineering_evidence"
    assert receipt["release_status"] == "non_release"
    assert receipt["model_quality_evaluated"] is False
    assert receipt["manifest"]["sha256"] == sha256_file(request.output_json)


def _coherent_request(tmp_path: Path, request_type: type[object]) -> object:
    run_dir = tmp_path / "run"
    evidence_dir = run_dir / "correction_control"
    model_dir = tmp_path / "model"
    evidence_dir.mkdir(parents=True)
    model_dir.mkdir()

    training_config = run_dir / "training_config.effective.yaml"
    shutil.copy2(CONFIG, training_config)
    checkpoint = run_dir / "predictor_checkpoint.pt"
    checkpoint.write_bytes(b"checkpoint")
    predictor = model_dir / "predictor.safetensors"
    predictor.write_bytes(b"predictor")
    action_encoder = model_dir / "action_encoder.safetensors"
    action_encoder.write_bytes(b"action-encoder")

    dataset_manifest = run_dir / "dataset_manifest.json"
    _write_json(
        dataset_manifest,
        {
            "schema_version": "1.0.0",
            "generated_by": "tools.release.dataset_package",
            "snapshot_id": SNAPSHOT_ID,
        },
    )
    state_audit = evidence_dir / "state_contract_audit.json"
    _write_json(
        state_audit,
        {
            "schema_version": "1.0.0",
            "generated_by": "tools.research.state_contract_audit",
            "ok": True,
            "blockers": [],
            "commit_sha": COMMIT_SHA,
            "encoder": {
                "revision": "5d31d59b3c845b288a13aedb1358934196852eec",
                "runtime_hash": RUNTIME_HASH,
                "expected_runtime_hash": RUNTIME_HASH,
                "runtime_identity_verified": True,
                "weights_hash": WEIGHTS_HASH,
                "expected_weights_hash": WEIGHTS_HASH,
                "weights_identity_verified": True,
                "normalized_state_contract": "l2_normalized_v2",
            },
        },
    )
    training_run = run_dir / "training_run.json"
    _write_json(
        training_run,
        {
            "schema_version": "1.0.0",
            "generated_by": "tools.release.training_run",
            "status": "completed",
            "commit_sha": COMMIT_SHA,
            "run_id": RUN_ID,
            "dataset_snapshot_id": SNAPSHOT_ID,
            "artifact_identities": {
                "training_config": _identity(training_config, "training_config.effective.yaml"),
                "dataset_manifest": _identity(dataset_manifest, "dataset_manifest.json"),
                "checkpoint_files": [_identity(checkpoint, "predictor_checkpoint.pt")],
            },
        },
    )
    postflight = evidence_dir / "correction_control_postflight.json"
    _write_json(
        postflight,
        {
            "schema_version": "1.0.0",
            "generated_by": "tools.research.correction_control_postflight",
            "ok": True,
            "blockers": [],
            "expected": {
                "commit_sha": COMMIT_SHA,
                "run_id": RUN_ID,
                "dataset_snapshot_id": SNAPSHOT_ID,
                "steps_completed": 50,
                "encoder_runtime_hash": RUNTIME_HASH,
                "state_contract_version": "l2_normalized_v2",
            },
            "artifacts": {
                "training_run": _identity(training_run, "training_run.json"),
                "training_config": _identity(training_config, "training_config.effective.yaml"),
                "checkpoint": _identity(checkpoint, "predictor_checkpoint.pt"),
                "state_contract_audit": _identity(state_audit, "state_contract_audit.json"),
                "dataset_manifest": _identity(dataset_manifest, "dataset_manifest.json"),
            },
        },
    )
    export_report = model_dir / "export_report.json"
    _write_json(
        export_report,
        {
            "schema_version": "1.0.0",
            "generated_by": "geno_lewm.deploy.export",
            "format": "safetensors",
            "checkpoint": {
                **_export_identity(checkpoint, "predictor_checkpoint.pt"),
                "schema_version": "1.0.0",
                "run_id": RUN_ID,
                "dataset_snapshot_id": SNAPSHOT_ID,
                "steps_completed": 50,
            },
            "artifacts": [
                {
                    "component": "predictor",
                    **_export_identity(predictor, "predictor.safetensors"),
                    "tensors": 2,
                },
                {
                    "component": "action_encoder",
                    **_export_identity(action_encoder, "action_encoder.safetensors"),
                    "tensors": 1,
                },
            ],
        },
    )

    return request_type(
        model_dir=model_dir,
        training_run_json=training_run,
        training_config=training_config,
        checkpoint=checkpoint,
        state_contract_audit_json=state_audit,
        dataset_manifest_json=dataset_manifest,
        correction_control_postflight_json=postflight,
        export_report_json=export_report,
        output_json=model_dir / "manifest.json",
    )


def _identity(path: Path, reported_path: str) -> dict[str, object]:
    return {
        "path": reported_path,
        "sha256": sha256_file(path),
        "size_bytes": path.stat().st_size,
    }


def _export_identity(path: Path, reported_file: str) -> dict[str, object]:
    return {
        "file": reported_file,
        "sha256": sha256_file(path),
        "size_bytes": path.stat().st_size,
    }


def _cli_args(request: object) -> list[str]:
    return [
        "--model-dir",
        str(request.model_dir),
        "--training-run-json",
        str(request.training_run_json),
        "--training-config",
        str(request.training_config),
        "--checkpoint",
        str(request.checkpoint),
        "--state-contract-audit-json",
        str(request.state_contract_audit_json),
        "--dataset-manifest-json",
        str(request.dataset_manifest_json),
        "--correction-control-postflight-json",
        str(request.correction_control_postflight_json),
        "--export-report-json",
        str(request.export_report_json),
        "--manifest-json",
        str(request.output_json),
    ]


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
