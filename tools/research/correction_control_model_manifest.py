# SPDX-License-Identifier: Apache-2.0
"""Author and validate a non-release correction-control model manifest.

The correction-control proof exports model weights before calibration or model-quality
evaluation exists.  The deploy/release ``geno_lewm.provenance.Manifest`` therefore cannot
represent this artifact without placeholder evidence.  This dedicated contract instead
binds the exact engineering-proof inputs and exports while stating the narrower claim
boundary explicitly.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Final

from geno_lewm.config import load_config
from geno_lewm.deploy.export import (
    ACTION_ENCODER_ARTIFACT,
    EXPORT_REPORT_NAME,
    GENERATED_BY as EXPORT_GENERATED_BY,
    PREDICTOR_ARTIFACT,
    SCHEMA_VERSION as EXPORT_SCHEMA_VERSION,
)
from geno_lewm.errors import GenoLeWMError, InputError, exit_code_for
from geno_lewm.provenance import sha256_file
from geno_lewm.provenance.hashing import looks_like_sha256
from tools.release.dataset_package import (
    GENERATED_BY as DATASET_MANIFEST_GENERATED_BY,
    SCHEMA_VERSION as DATASET_MANIFEST_SCHEMA_VERSION,
)
from tools.release.training_run import (
    GENERATED_BY as TRAINING_RUN_GENERATED_BY,
    SCHEMA_VERSION as TRAINING_RUN_SCHEMA_VERSION,
)
from tools.research.correction_control_postflight import (
    EXPECTED_STEPS,
    GENERATED_BY as POSTFLIGHT_GENERATED_BY,
    SCHEMA_VERSION as POSTFLIGHT_SCHEMA_VERSION,
)
from tools.research.correction_control_preflight import EXPECTED_RUN_ID, EXPECTED_SNAPSHOT_ID
from tools.research.state_contract_audit import (
    DEFAULT_CARBON_REVISION,
    DEFAULT_CARBON_RUNTIME_HASH,
    DEFAULT_CARBON_WEIGHTS_HASH,
    GENERATED_BY as STATE_AUDIT_GENERATED_BY,
    SCHEMA_VERSION as STATE_AUDIT_SCHEMA_VERSION,
)

SCHEMA_VERSION: Final = "1.0.0"
GENERATED_BY: Final = "tools.research.correction_control_model_manifest"
VALIDATION_GENERATED_BY: Final = f"{GENERATED_BY}.validate"
ARTIFACT_KIND: Final = "correction_control_engineering_evidence"
RELEASE_STATUS: Final = "non_release"
MANIFEST_NAME: Final = "manifest.json"
VALIDATION_REPORT_NAME: Final = "manifest_validation.json"
_COMMIT_RE: Final = re.compile(r"^[0-9a-f]{40}$")
_CLAIM_BOUNDARY: Final = (
    "This manifest binds one 50-step correction-control engineering proof and its exported "
    "weights. It is a non-release artifact without calibration or model-quality evaluation "
    "evidence; it does not establish convergence, repeat-run reproducibility, predictive "
    "quality, benchmark performance, dataset representativeness, or clinical utility."
)


@dataclass(frozen=True, slots=True)
class CorrectionControlModelManifestRequest:
    """Files that define one completed correction-control model export."""

    model_dir: Path
    training_run_json: Path
    training_config: Path
    checkpoint: Path
    state_contract_audit_json: Path
    dataset_manifest_json: Path
    correction_control_postflight_json: Path
    export_report_json: Path
    output_json: Path


def build_correction_control_model_manifest(
    request: CorrectionControlModelManifestRequest,
) -> dict[str, object]:
    """Build a manifest after validating every bound source and export."""
    _validate_layout(request)
    training_run = _load_json_object(request.training_run_json, "training_run")
    audit = _load_json_object(request.state_contract_audit_json, "state_contract_audit")
    dataset_manifest = _load_json_object(request.dataset_manifest_json, "dataset_manifest")
    postflight = _load_json_object(
        request.correction_control_postflight_json,
        "correction_control_postflight",
    )
    export_report = _load_json_object(request.export_report_json, "export_report")
    config = load_config(request.training_config)

    commit_sha = _required_text(training_run, "commit_sha", "training_run")
    if not _COMMIT_RE.fullmatch(commit_sha):
        raise InputError("training_run.commit_sha must be an exact lowercase 40-hex SHA")
    _expect(training_run, "schema_version", TRAINING_RUN_SCHEMA_VERSION, "training_run")
    _expect(training_run, "generated_by", TRAINING_RUN_GENERATED_BY, "training_run")
    _expect(training_run, "status", "completed", "training_run")
    _expect(training_run, "run_id", EXPECTED_RUN_ID, "training_run")
    _expect(training_run, "dataset_snapshot_id", EXPECTED_SNAPSHOT_ID, "training_run")

    _validate_config(config)
    training_identities = _required_mapping(
        training_run.get("artifact_identities"),
        "training_run.artifact_identities",
    )
    _verify_declared_file_identity(
        training_identities.get("training_config"),
        request.training_config,
        "training_config.effective.yaml",
        "training_run.artifact_identities.training_config",
    )
    _verify_declared_file_identity(
        training_identities.get("dataset_manifest"),
        request.dataset_manifest_json,
        "dataset_manifest.json",
        "training_run.artifact_identities.dataset_manifest",
    )
    checkpoint_files = _required_sequence(
        training_identities.get("checkpoint_files"),
        "training_run.artifact_identities.checkpoint_files",
    )
    if len(checkpoint_files) != 1:
        raise InputError("training_run must bind exactly one correction-control checkpoint")
    _verify_declared_file_identity(
        checkpoint_files[0],
        request.checkpoint,
        "predictor_checkpoint.pt",
        "training_run.artifact_identities.checkpoint_files[0]",
    )

    _validate_dataset_manifest(dataset_manifest)
    _validate_state_audit(audit, commit_sha=commit_sha)
    _validate_postflight(postflight, request=request, commit_sha=commit_sha)
    export_artifacts = _validate_export_report(export_report, request=request)

    predictor_path = request.model_dir / PREDICTOR_ARTIFACT
    action_encoder_path = request.model_dir / ACTION_ENCODER_ARTIFACT
    return {
        "schema_version": SCHEMA_VERSION,
        "generated_by": GENERATED_BY,
        "artifact_kind": ARTIFACT_KIND,
        "release_status": RELEASE_STATUS,
        "model_quality_evaluated": False,
        "run": {
            "commit_sha": commit_sha,
            "run_id": EXPECTED_RUN_ID,
            "steps_completed": EXPECTED_STEPS,
        },
        "encoder": {
            "model_id": config.encoder.model_id,
            "revision": config.encoder.revision,
            "dtype": config.encoder.dtype,
            "state_contract_version": config.encoder.state_contract_version,
            "runtime_hash": DEFAULT_CARBON_RUNTIME_HASH,
            "weights_hash": DEFAULT_CARBON_WEIGHTS_HASH,
            "state_contract_audit": _file_identity(
                request.state_contract_audit_json,
                "run/correction_control/state_contract_audit.json",
            ),
        },
        "training": {
            "config": _file_identity(
                request.training_config,
                "run/training_config.effective.yaml",
            ),
            "checkpoint": _file_identity(
                request.checkpoint,
                "run/predictor_checkpoint.pt",
            ),
        },
        "dataset": {
            "snapshot_id": EXPECTED_SNAPSHOT_ID,
            "manifest": _file_identity(
                request.dataset_manifest_json,
                "run/dataset_manifest.json",
            ),
        },
        "export": {
            "format": "safetensors",
            "report": _file_identity(request.export_report_json, f"model/{EXPORT_REPORT_NAME}"),
            "predictor": _file_identity(predictor_path, f"model/{PREDICTOR_ARTIFACT}"),
            "action_encoder": _file_identity(
                action_encoder_path,
                f"model/{ACTION_ENCODER_ARTIFACT}",
            ),
            "components": sorted(export_artifacts),
        },
        "evidence": {
            "training_run": _file_identity(request.training_run_json, "run/training_run.json"),
            "correction_control_postflight": _file_identity(
                request.correction_control_postflight_json,
                "run/correction_control/correction_control_postflight.json",
            ),
        },
        "claim_boundary": _CLAIM_BOUNDARY,
    }


def author_correction_control_model_manifest(
    request: CorrectionControlModelManifestRequest,
) -> dict[str, object]:
    """Validate sources and write the run-specific ``model/manifest.json``."""
    if request.output_json.exists():
        raise InputError(
            "correction-control model manifest already exists",
            details={"path": str(request.output_json)},
        )
    manifest = build_correction_control_model_manifest(request)
    request.output_json.parent.mkdir(parents=True, exist_ok=True)
    request.output_json.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return manifest


def validate_correction_control_model_manifest(
    request: CorrectionControlModelManifestRequest,
) -> dict[str, object]:
    """Rebuild the expected contract and validate the on-disk manifest exactly."""
    observed = _load_json_object(request.output_json, "manifest")
    expected = build_correction_control_model_manifest(request)
    if observed != expected:
        raise InputError(
            "correction-control model manifest does not match its bound evidence",
            details={"path": str(request.output_json)},
        )
    return {
        "schema_version": SCHEMA_VERSION,
        "generated_by": VALIDATION_GENERATED_BY,
        "ok": True,
        "manifest": _file_identity(request.output_json, "model/manifest.json"),
        "artifact_kind": ARTIFACT_KIND,
        "release_status": RELEASE_STATUS,
        "model_quality_evaluated": False,
        "claim_boundary": _CLAIM_BOUNDARY,
    }


def write_validation_report(report: Mapping[str, object], output_json: Path) -> Path:
    """Write a successful validation receipt without replacing existing evidence."""
    if output_json.exists():
        raise InputError(
            "correction-control manifest validation report already exists",
            details={"path": str(output_json)},
        )
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(
        json.dumps(dict(report), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return output_json


def _validate_layout(request: CorrectionControlModelManifestRequest) -> None:
    model_dir = request.model_dir.resolve()
    if not request.model_dir.is_dir():
        raise InputError("model_dir must be an existing directory")
    if request.output_json.resolve().parent != model_dir:
        raise InputError("correction-control manifest must be written inside model_dir")
    if request.output_json.name != MANIFEST_NAME:
        raise InputError(f"correction-control manifest must be named {MANIFEST_NAME}")
    if request.export_report_json.resolve() != (model_dir / EXPORT_REPORT_NAME):
        raise InputError("export_report_json must be model_dir/export_report.json")


def _validate_config(config: object) -> None:
    # ``load_config`` returns the public typed config; keeping this helper free of
    # file-format details makes the manifest contract depend on resolved semantics.
    run_id = getattr(config, "run_id", None)
    encoder = getattr(config, "encoder", None)
    training = getattr(config, "training", None)
    if run_id != EXPECTED_RUN_ID:
        raise InputError("training config run_id is not the correction-control run")
    expected_encoder = {
        "model_id": "/carbon",
        "revision": DEFAULT_CARBON_REVISION,
        "state_contract_version": "l2_normalized_v2",
        "normalize": True,
    }
    for field, expected in expected_encoder.items():
        observed = getattr(encoder, field, None)
        if observed != expected:
            raise InputError(
                f"training config encoder.{field} drifted",
                details={"expected": expected, "observed": observed},
            )
    if getattr(training, "max_steps", None) != EXPECTED_STEPS:
        raise InputError("training config max_steps is not the 50-step proof contract")


def _validate_dataset_manifest(payload: Mapping[str, object]) -> None:
    _expect(payload, "schema_version", DATASET_MANIFEST_SCHEMA_VERSION, "dataset_manifest")
    _expect(payload, "generated_by", DATASET_MANIFEST_GENERATED_BY, "dataset_manifest")
    _expect(payload, "snapshot_id", EXPECTED_SNAPSHOT_ID, "dataset_manifest")


def _validate_state_audit(payload: Mapping[str, object], *, commit_sha: str) -> None:
    _expect(payload, "schema_version", STATE_AUDIT_SCHEMA_VERSION, "state_contract_audit")
    _expect(payload, "generated_by", STATE_AUDIT_GENERATED_BY, "state_contract_audit")
    _expect(payload, "ok", True, "state_contract_audit")
    _expect(payload, "blockers", [], "state_contract_audit")
    _expect(payload, "commit_sha", commit_sha, "state_contract_audit")
    encoder = _required_mapping(payload.get("encoder"), "state_contract_audit.encoder")
    expected = {
        "revision": DEFAULT_CARBON_REVISION,
        "runtime_hash": DEFAULT_CARBON_RUNTIME_HASH,
        "expected_runtime_hash": DEFAULT_CARBON_RUNTIME_HASH,
        "runtime_identity_verified": True,
        "weights_hash": DEFAULT_CARBON_WEIGHTS_HASH,
        "expected_weights_hash": DEFAULT_CARBON_WEIGHTS_HASH,
        "weights_identity_verified": True,
        "normalized_state_contract": "l2_normalized_v2",
    }
    for key, value in expected.items():
        _expect(encoder, key, value, "state_contract_audit.encoder")


def _validate_postflight(
    payload: Mapping[str, object],
    *,
    request: CorrectionControlModelManifestRequest,
    commit_sha: str,
) -> None:
    _expect(
        payload,
        "schema_version",
        POSTFLIGHT_SCHEMA_VERSION,
        "correction_control_postflight",
    )
    _expect(payload, "generated_by", POSTFLIGHT_GENERATED_BY, "correction_control_postflight")
    _expect(payload, "ok", True, "correction_control_postflight")
    _expect(payload, "blockers", [], "correction_control_postflight")
    expected = _required_mapping(payload.get("expected"), "correction_control_postflight.expected")
    for key, value in {
        "commit_sha": commit_sha,
        "run_id": EXPECTED_RUN_ID,
        "dataset_snapshot_id": EXPECTED_SNAPSHOT_ID,
        "steps_completed": EXPECTED_STEPS,
        "encoder_runtime_hash": DEFAULT_CARBON_RUNTIME_HASH,
        "state_contract_version": "l2_normalized_v2",
    }.items():
        _expect(expected, key, value, "correction_control_postflight.expected")

    artifacts = _required_mapping(
        payload.get("artifacts"),
        "correction_control_postflight.artifacts",
    )
    for key, path, reported_path in (
        ("training_run", request.training_run_json, "training_run.json"),
        ("training_config", request.training_config, "training_config.effective.yaml"),
        ("checkpoint", request.checkpoint, "predictor_checkpoint.pt"),
        ("state_contract_audit", request.state_contract_audit_json, "state_contract_audit.json"),
        ("dataset_manifest", request.dataset_manifest_json, "dataset_manifest.json"),
    ):
        _verify_declared_file_identity(
            artifacts.get(key),
            path,
            reported_path,
            f"correction_control_postflight.artifacts.{key}",
        )


def _validate_export_report(
    payload: Mapping[str, object],
    *,
    request: CorrectionControlModelManifestRequest,
) -> frozenset[str]:
    _expect(payload, "schema_version", EXPORT_SCHEMA_VERSION, "export_report")
    _expect(payload, "generated_by", EXPORT_GENERATED_BY, "export_report")
    _expect(payload, "format", "safetensors", "export_report")
    checkpoint = _required_mapping(payload.get("checkpoint"), "export_report.checkpoint")
    _verify_export_identity(
        checkpoint,
        request.checkpoint,
        "predictor_checkpoint.pt",
        "export_report.checkpoint",
    )
    for key, value in {
        "run_id": EXPECTED_RUN_ID,
        "dataset_snapshot_id": EXPECTED_SNAPSHOT_ID,
        "steps_completed": EXPECTED_STEPS,
    }.items():
        _expect(checkpoint, key, value, "export_report.checkpoint")

    declared_artifacts = _required_sequence(payload.get("artifacts"), "export_report.artifacts")
    by_component: dict[str, Mapping[str, object]] = {}
    for index, artifact in enumerate(declared_artifacts):
        row = _required_mapping(artifact, f"export_report.artifacts[{index}]")
        component = _required_text(row, "component", f"export_report.artifacts[{index}]")
        if component in by_component:
            raise InputError(f"export_report contains duplicate component {component}")
        by_component[component] = row
    expected_components = {
        "predictor": (request.model_dir / PREDICTOR_ARTIFACT, PREDICTOR_ARTIFACT),
        "action_encoder": (
            request.model_dir / ACTION_ENCODER_ARTIFACT,
            ACTION_ENCODER_ARTIFACT,
        ),
    }
    if set(by_component) != set(expected_components):
        raise InputError(
            "export_report must contain exactly predictor and action_encoder",
            details={"observed": sorted(by_component)},
        )
    for component, (path, reported_file) in expected_components.items():
        _verify_export_identity(
            by_component[component],
            path,
            reported_file,
            f"export_report.artifacts.{component}",
        )
    return frozenset(by_component)


def _file_identity(path: Path, reported_path: str) -> dict[str, object]:
    if not path.is_file():
        raise InputError(
            "bound correction-control artifact is missing", details={"path": str(path)}
        )
    return {
        "path": reported_path,
        "sha256": sha256_file(path),
        "size_bytes": path.stat().st_size,
    }


def _verify_declared_file_identity(
    value: object,
    path: Path,
    reported_path: str,
    label: str,
) -> None:
    declared = _required_mapping(value, label)
    _expect(declared, "path", reported_path, label)
    if "exists" in declared:
        _expect(declared, "exists", True, label)
    _verify_hash_and_size(declared, path, label)


def _verify_export_identity(
    value: Mapping[str, object],
    path: Path,
    reported_file: str,
    label: str,
) -> None:
    _expect(value, "file", reported_file, label)
    _verify_hash_and_size(value, path, label)


def _verify_hash_and_size(value: Mapping[str, object], path: Path, label: str) -> None:
    observed = _file_identity(path, path.name)
    declared_hash = value.get("sha256")
    if not isinstance(declared_hash, str) or not looks_like_sha256(declared_hash):
        raise InputError(f"{label}.sha256 must be a sha256 identity")
    for key in ("sha256", "size_bytes"):
        expected = observed[key]
        if value.get(key) != expected:
            raise InputError(
                f"{label}.{key} does not match the bound file",
                details={"expected": expected, "observed": value.get(key)},
            )


def _load_json_object(path: Path, label: str) -> dict[str, object]:
    try:
        payload: object = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise InputError(f"{label} is not readable JSON", details={"path": str(path)}) from exc
    if not isinstance(payload, dict):
        raise InputError(f"{label} must be a JSON object")
    if not all(isinstance(key, str) for key in payload):
        raise InputError(f"{label} keys must be strings")
    return payload


def _required_mapping(value: object, label: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping) or not all(isinstance(key, str) for key in value):
        raise InputError(f"{label} must be an object")
    return value


def _required_sequence(value: object, label: str) -> Sequence[object]:
    if not isinstance(value, list):
        raise InputError(f"{label} must be a list")
    return value


def _required_text(payload: Mapping[str, object], key: str, label: str) -> str:
    value = payload.get(key)
    if not isinstance(value, str) or not value:
        raise InputError(f"{label}.{key} must be non-empty text")
    return value


def _expect(payload: Mapping[str, object], key: str, expected: object, label: str) -> None:
    observed = payload.get(key)
    if type(observed) is not type(expected) or observed != expected:
        raise InputError(
            f"{label}.{key} does not match the correction-control contract",
            details={"expected": expected, "observed": observed},
        )


def _request_from_args(args: argparse.Namespace) -> CorrectionControlModelManifestRequest:
    return CorrectionControlModelManifestRequest(
        model_dir=args.model_dir,
        training_run_json=args.training_run_json,
        training_config=args.training_config,
        checkpoint=args.checkpoint,
        state_contract_audit_json=args.state_contract_audit_json,
        dataset_manifest_json=args.dataset_manifest_json,
        correction_control_postflight_json=args.correction_control_postflight_json,
        export_report_json=args.export_report_json,
        output_json=args.manifest_json,
    )


def _add_shared_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--training-run-json", type=Path, required=True)
    parser.add_argument("--training-config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--state-contract-audit-json", type=Path, required=True)
    parser.add_argument("--dataset-manifest-json", type=Path, required=True)
    parser.add_argument("--correction-control-postflight-json", type=Path, required=True)
    parser.add_argument("--export-report-json", type=Path, required=True)
    parser.add_argument("--manifest-json", type=Path, required=True)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    author = commands.add_parser("author", help="author model/manifest.json")
    _add_shared_arguments(author)
    validate = commands.add_parser("validate", help="validate model/manifest.json")
    _add_shared_arguments(validate)
    validate.add_argument("--validation-report-json", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    request = _request_from_args(args)
    try:
        if args.command == "author":
            manifest = author_correction_control_model_manifest(request)
            summary: Mapping[str, object] = {
                "ok": True,
                "manifest": _file_identity(request.output_json, "model/manifest.json"),
                "artifact_kind": manifest["artifact_kind"],
                "release_status": manifest["release_status"],
            }
        else:
            summary = validate_correction_control_model_manifest(request)
            write_validation_report(summary, args.validation_report_json)
    except GenoLeWMError as exc:
        sys.stderr.write(f"error: {exc}\n")
        if exc.details:
            sys.stderr.write(json.dumps(exc.details, sort_keys=True) + "\n")
        return exit_code_for(exc)
    sys.stdout.write(json.dumps(dict(summary), sort_keys=True) + "\n")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
