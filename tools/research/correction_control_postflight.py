# SPDX-License-Identifier: Apache-2.0
"""Validate evidence from the corrected 50-step Phase-1 smoke run.

This postflight is intentionally narrower than a benchmark or reproducibility
gate. It verifies that one clean-machine training launch completed with the
expected pinned identities, configuration, and finite optimizer history.
"""

from __future__ import annotations

import argparse
import importlib
import json
import math
import re
import sys
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Final

from geno_lewm.config import GenoLeWMConfig, config_to_dict, load_config
from geno_lewm.errors import GenoLeWMError
from geno_lewm.provenance import sha256_file
from geno_lewm.provenance.hashing import looks_like_sha256
from geno_lewm.training.preflight import (
    GENERATED_BY as TRAINING_PREFLIGHT_GENERATED_BY,
    SCHEMA_VERSION as TRAINING_PREFLIGHT_SCHEMA_VERSION,
)
from tools.data.tuple_throughput import (
    GENERATED_BY as TUPLE_THROUGHPUT_GENERATED_BY,
    SCHEMA_VERSION as TUPLE_THROUGHPUT_SCHEMA_VERSION,
)
from tools.release.dataset_package import (
    GENERATED_BY as DATASET_MANIFEST_GENERATED_BY,
    SCHEMA_VERSION as DATASET_MANIFEST_SCHEMA_VERSION,
)
from tools.release.dataset_snapshot import (
    GENERATED_BY as DATASET_SNAPSHOT_GENERATED_BY,
    INPUT_CHECK_REPORT_NAME as DATASET_INPUT_CHECK_REPORT_NAME,
    REPORT_NAME as DATASET_SNAPSHOT_REPORT_NAME,
)
from tools.research.correction_control_preflight import (
    EXPECTED_CARBON_CONFIG,
    EXPECTED_CARBON_MODEL_DIR,
    EXPECTED_CARBON_SOURCE,
    EXPECTED_CLINVAR_LINES,
    EXPECTED_CLINVAR_URL,
    EXPECTED_CONFIG_PATH,
    EXPECTED_CONTAINER_IMAGE,
    EXPECTED_GNOMAD_LINES,
    EXPECTED_GNOMAD_URL,
    EXPECTED_HOLDOUT_CHROM,
    EXPECTED_MAX_WINDOWS,
    EXPECTED_OPTIMIZER_LR,
    EXPECTED_RUN_ID,
    EXPECTED_SNAPSHOT_ID,
    EXPECTED_SNAPSHOT_PATH,
    EXPECTED_SOURCE_INTEGRITY,
    EXPECTED_TUPLE_THROUGHPUT_SAMPLES,
    EXPECTED_WINDOW_BP,
    GENERATED_BY as JOB_CONTRACT_PREFLIGHT_GENERATED_BY,
    SCHEMA_VERSION as JOB_CONTRACT_PREFLIGHT_SCHEMA_VERSION,
)
from tools.research.state_contract_audit import (
    DEFAULT_CARBON_REVISION,
    DEFAULT_CARBON_RUNTIME_HASH,
    DEFAULT_CARBON_WEIGHTS_HASH,
    GENERATED_BY as STATE_CONTRACT_AUDIT_GENERATED_BY,
    SCHEMA_VERSION as STATE_CONTRACT_AUDIT_SCHEMA_VERSION,
)

SCHEMA_VERSION: Final = "1.0.0"
GENERATED_BY: Final = "tools.research.correction_control_postflight"
EXPECTED_STEPS: Final = 50
EXPECTED_SAMPLE_COUNT: Final = 400
EXPECTED_D_STATE: Final = 1024
EXPECTED_BATCH_SIZE: Final = 8
EXPECTED_SEED: Final = 104729
EXPECTED_SEEDS: Final[dict[str, int]] = {
    "data": EXPECTED_SEED,
    "predictor": EXPECTED_SEED + 1,
    "lora": EXPECTED_SEED + 2,
}
EXPECTED_CORPUS_ID: Final = "HuggingFaceBio/carbon-pretraining-corpus"
EXPECTED_CORPUS_REVISION: Final = "cb4c13a78102933b3a6ac65734d326f7b431d9b7"
EXPECTED_TUPLE_THROUGHPUT_SEED: Final = 0
EXPECTED_MIN_TUPLES_PER_SECOND: Final = 5000.0
EXPECTED_MIN_CUDA_MEMORY_BYTES: Final = 120 * 1024**3
SOURCE_IDENTITY_SCHEMA_VERSION: Final = "1.0.0"
SOURCE_IDENTITY_GENERATED_BY: Final = "tools.jobs.proof_run.source_identity"
_EXPECTED_MANIFEST_FILES: Final[dict[str, tuple[str, int | None]]] = {
    "carbon/source-mix-windows.jsonl": ("train_carbon", EXPECTED_MAX_WINDOWS),
    "gnomad/v4.1/variants.parquet": ("train_gnomad_common", None),
    "clinvar/2026-04-15/variants.parquet": ("eval_clinvar", None),
}
_EXPECTED_SNAPSHOT_SOURCES: Final[dict[str, tuple[str, str]]] = {
    "carbon/source-mix-windows.jsonl": (
        "carbon",
        "inputs/carbon/source-mix-windows.jsonl",
    ),
    "gnomad/v4.1/variants.parquet": (
        "gnomad",
        "inputs/gnomad/gnomad-v4.1-snv.vcf.gz",
    ),
    "clinvar/2026-04-15/variants.parquet": (
        "clinvar",
        "inputs/clinvar/clinvar-2026-04-15-snv.vcf.gz",
    ),
}
_COMMIT_RE: Final = re.compile(r"^[0-9a-f]{40}$")
_RUN_NAME_RE: Final = re.compile(
    r"^geno-lewm-l2-p1-smoke-(?P<commit>[0-9a-f]{12})-50-r(?P<attempt>[1-9][0-9]*)$"
)
_CLAIM_BOUNDARY: Final = (
    "This postflight verifies source and artifact identity, configuration coherence, and finite "
    "completion for one 50-step Phase-1 smoke run. The tuple-throughput evidence is only a "
    "generic mixed-provider builder gate, not proof of training-source parity. This receipt does "
    "not establish convergence, repeat-run determinism, predictive quality, benchmark "
    "performance, dataset representativeness, or clinical utility."
)

CheckpointLoader = Callable[[Path], Mapping[str, Any]]


@dataclass(frozen=True, slots=True)
class CorrectionControlPostflightRequest:
    """Paths and immutable identities for one correction-control smoke run."""

    training_run_json: Path
    metrics_json: Path
    training_config: Path
    checkpoint: Path
    state_contract_audit_json: Path
    job_contract_preflight_json: Path
    source_identity_report_json: Path
    dataset_manifest_json: Path
    dataset_snapshot_report_json: Path
    training_preflight_report_json: Path
    tuple_throughput_report_json: Path
    expected_commit_sha: str
    expected_run_id: str
    expected_dataset_snapshot_id: str
    output_json: Path


def build_correction_control_postflight_report(
    request: CorrectionControlPostflightRequest,
    *,
    checkpoint_loader: CheckpointLoader | None = None,
    generated_at: str | None = None,
) -> dict[str, object]:
    """Return a machine-readable, fail-closed postflight report."""
    blockers: list[dict[str, object]] = []
    _validate_expected_identity(request, blockers)

    training_run = _load_json_object(request.training_run_json, "training_run", blockers)
    metrics = _load_json_object(request.metrics_json, "metrics", blockers)
    audit = _load_json_object(
        request.state_contract_audit_json,
        "state_contract_audit",
        blockers,
    )
    job_contract_preflight = _load_json_object(
        request.job_contract_preflight_json,
        "job_contract_preflight",
        blockers,
    )
    source_identity = _load_json_object(
        request.source_identity_report_json,
        "source_identity_report",
        blockers,
    )
    dataset_manifest = _load_json_object(
        request.dataset_manifest_json,
        "dataset_manifest",
        blockers,
    )
    dataset_snapshot_report = _load_json_object(
        request.dataset_snapshot_report_json,
        "dataset_snapshot_report",
        blockers,
    )
    training_preflight = _load_json_object(
        request.training_preflight_report_json,
        "training_preflight_report",
        blockers,
    )
    tuple_throughput = _load_json_object(
        request.tuple_throughput_report_json,
        "tuple_throughput_report",
        blockers,
    )
    config = _load_training_config(request.training_config, blockers)
    checkpoint = _load_checkpoint(
        request.checkpoint,
        blockers,
        loader=checkpoint_loader or _load_torch_checkpoint,
    )

    if training_run is not None:
        _validate_training_run(training_run, request, blockers)
        _validate_declared_artifacts(training_run, request, blockers)
    if metrics is not None:
        _validate_metrics(metrics, request, blockers)
    if config is not None:
        _validate_config(config, request, blockers)
    if checkpoint is not None:
        _validate_checkpoint(checkpoint, request, blockers)
    if audit is not None:
        _validate_state_contract_audit(audit, request, blockers)
    if job_contract_preflight is not None:
        _validate_job_contract_preflight(job_contract_preflight, request, blockers)
    if source_identity is not None:
        _validate_source_identity_report(source_identity, request, blockers)
    if dataset_manifest is not None:
        _validate_dataset_manifest(dataset_manifest, request, blockers)
    if dataset_snapshot_report is not None:
        _validate_dataset_snapshot_report(dataset_snapshot_report, request, blockers)
    if training_preflight is not None:
        _validate_training_preflight_report(training_preflight, request, blockers)
    if tuple_throughput is not None:
        _validate_tuple_throughput_report(tuple_throughput, request, blockers)
    if metrics is not None and checkpoint is not None:
        _validate_metrics_checkpoint_coherence(metrics, checkpoint, blockers)
    if config is not None and checkpoint is not None:
        _validate_config_checkpoint_coherence(config, checkpoint, blockers)
    if audit is not None and checkpoint is not None:
        _validate_audit_checkpoint_coherence(audit, checkpoint, blockers)
    if job_contract_preflight is not None and source_identity is not None:
        _validate_job_source_coherence(job_contract_preflight, source_identity, blockers)
    if job_contract_preflight is not None and training_preflight is not None:
        _validate_job_training_preflight_coherence(
            job_contract_preflight,
            training_preflight,
            blockers,
        )
    if config is not None and training_preflight is not None:
        _validate_effective_config_training_preflight_coherence(
            config,
            training_preflight,
            blockers,
        )
    if dataset_manifest is not None and training_preflight is not None:
        _validate_manifest_training_preflight_coherence(
            dataset_manifest,
            training_preflight,
            request,
            blockers,
        )
    if dataset_snapshot_report is not None and training_preflight is not None:
        _validate_snapshot_training_preflight_coherence(
            dataset_snapshot_report,
            training_preflight,
            blockers,
        )
    if job_contract_preflight is not None and dataset_snapshot_report is not None:
        _validate_job_snapshot_coherence(
            job_contract_preflight,
            dataset_snapshot_report,
            blockers,
        )
    if source_identity is not None and dataset_manifest is not None:
        _validate_source_manifest_coherence(
            source_identity,
            dataset_manifest,
            blockers,
        )
    if source_identity is not None and dataset_snapshot_report is not None:
        _validate_source_snapshot_coherence(
            source_identity,
            dataset_snapshot_report,
            blockers,
        )
    if dataset_snapshot_report is not None and dataset_manifest is not None:
        _validate_snapshot_manifest_coherence(
            dataset_snapshot_report,
            dataset_manifest,
            blockers,
        )
    if dataset_manifest is not None and tuple_throughput is not None:
        _validate_manifest_tuple_throughput_coherence(
            dataset_manifest,
            tuple_throughput,
            request,
            blockers,
        )

    return {
        "schema_version": SCHEMA_VERSION,
        "generated_by": GENERATED_BY,
        "generated_at": generated_at or _utc_now(),
        "ok": not blockers,
        "expected": {
            "commit_sha": request.expected_commit_sha,
            "run_id": request.expected_run_id,
            "dataset_snapshot_id": request.expected_dataset_snapshot_id,
            "steps_completed": EXPECTED_STEPS,
            "sample_count": EXPECTED_SAMPLE_COUNT,
            "phase": "phase1",
            "state_contract_version": "l2_normalized_v2",
            "encoder_runtime_hash": DEFAULT_CARBON_RUNTIME_HASH,
        },
        "artifacts": {
            "training_run": _artifact_summary(request.training_run_json),
            "metrics": _artifact_summary(request.metrics_json),
            "training_config": _artifact_summary(request.training_config),
            "checkpoint": _artifact_summary(request.checkpoint),
            "state_contract_audit": _artifact_summary(request.state_contract_audit_json),
            "job_contract_preflight": _artifact_summary(request.job_contract_preflight_json),
            "source_identity_report": _artifact_summary(request.source_identity_report_json),
            "dataset_manifest": _artifact_summary(request.dataset_manifest_json),
            "dataset_snapshot_report": _artifact_summary(request.dataset_snapshot_report_json),
            "training_preflight_report": _artifact_summary(request.training_preflight_report_json),
            "tuple_throughput_report": _artifact_summary(request.tuple_throughput_report_json),
        },
        "blockers": blockers,
        "claim_boundary": _CLAIM_BOUNDARY,
    }


def main(argv: list[str] | None = None) -> int:
    """Validate the smoke-run artifacts and write the postflight receipt."""
    args = _parser().parse_args(argv)
    request = CorrectionControlPostflightRequest(
        training_run_json=args.training_run_json,
        metrics_json=args.metrics_json,
        training_config=args.training_config,
        checkpoint=args.checkpoint,
        state_contract_audit_json=args.state_contract_audit_json,
        job_contract_preflight_json=args.job_contract_preflight_json,
        source_identity_report_json=args.source_identity_report_json,
        dataset_manifest_json=args.dataset_manifest_json,
        dataset_snapshot_report_json=args.dataset_snapshot_report_json,
        training_preflight_report_json=args.training_preflight_report_json,
        tuple_throughput_report_json=args.tuple_throughput_report_json,
        expected_commit_sha=args.expected_commit_sha,
        expected_run_id=args.expected_run_id,
        expected_dataset_snapshot_id=args.expected_dataset_snapshot_id,
        output_json=args.output_json,
    )
    report = build_correction_control_postflight_report(request)
    try:
        request.output_json.parent.mkdir(parents=True, exist_ok=True)
        request.output_json.write_text(
            json.dumps(report, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    except OSError as exc:
        sys.stderr.write(f"error: failed to write postflight report: {exc}\n")
        return 2
    sys.stdout.write(json.dumps(report, sort_keys=True) + "\n")
    return 0 if report["ok"] is True else 1


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Validate corrected 50-step Phase-1 smoke-run evidence.",
    )
    parser.add_argument("--training-run-json", type=Path, required=True)
    parser.add_argument("--metrics-json", type=Path, required=True)
    parser.add_argument("--training-config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--state-contract-audit-json", type=Path, required=True)
    parser.add_argument("--job-contract-preflight-json", type=Path, required=True)
    parser.add_argument("--source-identity-report-json", type=Path, required=True)
    parser.add_argument("--dataset-manifest-json", type=Path, required=True)
    parser.add_argument("--dataset-snapshot-report-json", type=Path, required=True)
    parser.add_argument("--training-preflight-report-json", type=Path, required=True)
    parser.add_argument("--tuple-throughput-report-json", type=Path, required=True)
    parser.add_argument("--expected-commit-sha", required=True)
    parser.add_argument("--expected-run-id", required=True)
    parser.add_argument("--expected-dataset-snapshot-id", required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    return parser


def _validate_expected_identity(
    request: CorrectionControlPostflightRequest,
    blockers: list[dict[str, object]],
) -> None:
    if _COMMIT_RE.fullmatch(request.expected_commit_sha) is None:
        _block(
            blockers,
            "expected.commit_sha_invalid",
            "expected.commit_sha",
            "expected commit SHA must be exactly 40 lowercase hexadecimal characters",
            observed=request.expected_commit_sha,
        )
    for field, value in (
        ("run_id", request.expected_run_id),
        ("dataset_snapshot_id", request.expected_dataset_snapshot_id),
    ):
        if not value.strip():
            _block(
                blockers,
                f"expected.{field}_invalid",
                f"expected.{field}",
                f"expected {field} must be non-empty",
                observed=value,
            )
    if request.expected_run_id != EXPECTED_RUN_ID:
        _block(
            blockers,
            "expected.run_id_mismatch",
            "expected.run_id",
            "run ID must name the checked correction-control configuration",
            expected=EXPECTED_RUN_ID,
            observed=request.expected_run_id,
        )
    if request.expected_dataset_snapshot_id != EXPECTED_SNAPSHOT_ID:
        _block(
            blockers,
            "expected.dataset_snapshot_id_mismatch",
            "expected.dataset_snapshot_id",
            "dataset snapshot ID must name the checked correction-control snapshot",
            expected=EXPECTED_SNAPSHOT_ID,
            observed=request.expected_dataset_snapshot_id,
        )


def _load_json_object(
    path: Path,
    label: str,
    blockers: list[dict[str, object]],
) -> dict[str, Any] | None:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except OSError as exc:
        _block(blockers, f"{label}.unreadable", label, "artifact is unreadable", observed=str(exc))
        return None
    except json.JSONDecodeError as exc:
        _block(
            blockers,
            f"{label}.invalid_json",
            label,
            "artifact is not valid JSON",
            observed={"line": exc.lineno, "column": exc.colno},
        )
        return None
    if not isinstance(payload, dict):
        _block(blockers, f"{label}.invalid_type", label, "artifact must be a JSON object")
        return None
    return payload


def _load_training_config(
    path: Path,
    blockers: list[dict[str, object]],
) -> GenoLeWMConfig | None:
    try:
        return load_config(path)
    except (GenoLeWMError, OSError) as exc:
        _block(
            blockers,
            "training_config.invalid",
            "training_config",
            "resolved training config is invalid",
            observed=str(exc),
        )
        return None


def _load_checkpoint(
    path: Path,
    blockers: list[dict[str, object]],
    *,
    loader: CheckpointLoader,
) -> Mapping[str, Any] | None:
    try:
        payload = loader(path)
    except Exception as exc:
        _block(
            blockers,
            "checkpoint.unreadable",
            "checkpoint",
            "checkpoint could not be loaded safely",
            observed=f"{type(exc).__name__}: {exc}",
        )
        return None
    return payload


def _load_torch_checkpoint(path: Path) -> Mapping[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    torch = importlib.import_module("torch")
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(payload, Mapping):
        raise TypeError("checkpoint payload must be a mapping")
    return payload


def _validate_training_run(
    payload: Mapping[str, Any],
    request: CorrectionControlPostflightRequest,
    blockers: list[dict[str, object]],
) -> None:
    _expect(payload, "run_id", request.expected_run_id, "training_run.run_id", blockers)
    _expect(
        payload,
        "dataset_snapshot_id",
        request.expected_dataset_snapshot_id,
        "training_run.dataset_snapshot_id",
        blockers,
    )
    _expect(
        payload,
        "commit_sha",
        request.expected_commit_sha,
        "training_run.commit_sha",
        blockers,
    )
    _expect(payload, "status", "completed", "training_run.status", blockers)
    _expect(
        payload, "generated_by", "tools.release.training_run", "training_run.generated_by", blockers
    )
    _expect(payload, "resumed_from_step", 0, "training_run.resumed_from_step", blockers)
    _expect(payload, "resume_checkpoint", None, "training_run.resume_checkpoint", blockers)


def _validate_declared_artifacts(
    payload: Mapping[str, Any],
    request: CorrectionControlPostflightRequest,
    blockers: list[dict[str, object]],
) -> None:
    root = request.training_run_json.parent
    declarations = (
        ("training_config", request.training_config),
        ("metrics", request.metrics_json),
        ("dataset_manifest", request.dataset_manifest_json),
        ("training_preflight_report", request.training_preflight_report_json),
    )
    for field, supplied in declarations:
        declared = payload.get(field)
        if not isinstance(declared, str) or not declared:
            _block(
                blockers,
                f"training_run.{field}_declaration_invalid",
                f"training_run.{field}",
                "artifact declaration must be a non-empty relative path",
                observed=declared,
            )
            continue
        _validate_declared_path(root, declared, supplied, f"training_run.{field}", blockers)

    checkpoint_files = payload.get("checkpoint_files")
    if (
        not isinstance(checkpoint_files, Sequence)
        or isinstance(checkpoint_files, str | bytes)
        or len(checkpoint_files) != 1
        or not isinstance(checkpoint_files[0], str)
    ):
        _block(
            blockers,
            "training_run.checkpoint_declaration_invalid",
            "training_run.checkpoint_files",
            "exactly one checkpoint path must be declared",
            observed=checkpoint_files,
        )
    else:
        _validate_declared_path(
            root,
            checkpoint_files[0],
            request.checkpoint,
            "training_run.checkpoint_files[0]",
            blockers,
        )

    identities = payload.get("artifact_identities")
    if not isinstance(identities, Mapping):
        _block(
            blockers,
            "training_run.artifact_identities_invalid",
            "training_run.artifact_identities",
            "artifact identities must be an object",
        )
        return
    _validate_file_identity(
        identities.get("training_config"),
        request.training_config,
        "training_run.artifact_identities.training_config",
        blockers,
    )
    _validate_file_identity(
        identities.get("metrics"),
        request.metrics_json,
        "training_run.artifact_identities.metrics",
        blockers,
    )
    _validate_file_identity(
        identities.get("dataset_manifest"),
        request.dataset_manifest_json,
        "training_run.artifact_identities.dataset_manifest",
        blockers,
    )
    _validate_file_identity(
        identities.get("training_preflight_report"),
        request.training_preflight_report_json,
        "training_run.artifact_identities.training_preflight_report",
        blockers,
    )
    checkpoint_identities = identities.get("checkpoint_files")
    if (
        not isinstance(checkpoint_identities, Sequence)
        or isinstance(checkpoint_identities, str | bytes)
        or len(checkpoint_identities) != 1
    ):
        _block(
            blockers,
            "training_run.checkpoint_identity_invalid",
            "training_run.artifact_identities.checkpoint_files",
            "exactly one checkpoint identity must be recorded",
            observed=checkpoint_identities,
        )
    else:
        _validate_file_identity(
            checkpoint_identities[0],
            request.checkpoint,
            "training_run.artifact_identities.checkpoint_files[0]",
            blockers,
        )


def _validate_declared_path(
    root: Path,
    declared: str,
    supplied: Path,
    field: str,
    blockers: list[dict[str, object]],
) -> None:
    relative = Path(declared)
    if relative.is_absolute() or ".." in relative.parts:
        _block(
            blockers,
            f"{field}.nonportable",
            field,
            "declared artifact path must remain relative to the run directory",
            observed=declared,
        )
        return
    if (root / relative).resolve() != supplied.resolve():
        _block(
            blockers,
            f"{field}.mismatch",
            field,
            "declared artifact path does not match the supplied artifact",
            expected=str(supplied.resolve()),
            observed=str((root / relative).resolve()),
        )


def _validate_file_identity(
    raw: object,
    path: Path,
    field: str,
    blockers: list[dict[str, object]],
) -> None:
    if not isinstance(raw, Mapping):
        _block(blockers, f"{field}.invalid", field, "artifact identity must be an object")
        return
    if not path.is_file():
        _block(
            blockers, f"{field}.missing", field, "identity target is missing", observed=str(path)
        )
        return
    _expect(raw, "path", path.name, f"{field}.path", blockers)
    _expect(raw, "sha256", sha256_file(path), f"{field}.sha256", blockers)
    _expect(raw, "size_bytes", path.stat().st_size, f"{field}.size_bytes", blockers)


def _validate_metrics(
    payload: Mapping[str, Any],
    request: CorrectionControlPostflightRequest,
    blockers: list[dict[str, object]],
) -> None:
    _expect(payload, "run_id", request.expected_run_id, "metrics.run_id", blockers)
    _expect(
        payload,
        "dataset_snapshot_id",
        request.expected_dataset_snapshot_id,
        "metrics.dataset_snapshot_id",
        blockers,
    )
    _expect(payload, "steps_completed", EXPECTED_STEPS, "metrics.steps_completed", blockers)
    _expect(payload, "sample_count", EXPECTED_SAMPLE_COUNT, "metrics.sample_count", blockers)
    _expect(
        payload, "new_sample_count", EXPECTED_SAMPLE_COUNT, "metrics.new_sample_count", blockers
    )
    _expect(payload, "resumed_from_step", 0, "metrics.resumed_from_step", blockers)
    _expect(payload, "resume_checkpoint", None, "metrics.resume_checkpoint", blockers)
    root_loss = _expect_finite(payload.get("train_loss"), "metrics.train_loss", blockers)

    metrics = payload.get("metrics")
    if not isinstance(metrics, Mapping):
        _block(blockers, "metrics.metrics_invalid", "metrics.metrics", "metrics must be an object")
    else:
        nested_loss = _expect_finite(
            metrics.get("train_loss"), "metrics.metrics.train_loss", blockers
        )
        _expect(
            metrics, "sample_count", EXPECTED_SAMPLE_COUNT, "metrics.metrics.sample_count", blockers
        )
        _expect(
            metrics,
            "new_sample_count",
            EXPECTED_SAMPLE_COUNT,
            "metrics.metrics.new_sample_count",
            blockers,
        )
        _expect(metrics, "resumed_from_step", 0, "metrics.metrics.resumed_from_step", blockers)
        _expect(metrics, "nan_loss_count", 0, "metrics.metrics.nan_loss_count", blockers)
        _expect(
            metrics,
            "collapse_alert_count",
            0,
            "metrics.metrics.collapse_alert_count",
            blockers,
        )
        if root_loss is not None and nested_loss is not None and root_loss != nested_loss:
            _block(
                blockers,
                "metrics.train_loss_incoherent",
                "metrics.metrics.train_loss",
                "root and nested train loss must match",
                expected=root_loss,
                observed=nested_loss,
            )

    history = payload.get("history")
    if not isinstance(history, list):
        _block(blockers, "metrics.history_invalid", "metrics.history", "history must be a list")
        return
    if len(history) != EXPECTED_STEPS:
        _block(
            blockers,
            "metrics.history_length_mismatch",
            "metrics.history",
            "history must contain exactly one row per optimizer step",
            expected=EXPECTED_STEPS,
            observed=len(history),
        )
    final_history_loss: float | None = None
    for index, row in enumerate(history):
        path = f"metrics.history[{index}]"
        if not isinstance(row, Mapping):
            _block(blockers, "metrics.history_row_invalid", path, "history row must be an object")
            continue
        _expect(row, "step", index + 1, f"{path}.step", blockers)
        _expect(row, "action_count", EXPECTED_BATCH_SIZE, f"{path}.action_count", blockers)
        for field in ("loss", "pred_loss", "kl_reg", "lr_multiplier", "pred_var_per_dim"):
            value = _expect_finite(row.get(field), f"{path}.{field}", blockers)
            if field == "loss" and value is not None:
                final_history_loss = value
    if root_loss is not None and final_history_loss is not None and root_loss != final_history_loss:
        _block(
            blockers,
            "metrics.final_loss_incoherent",
            "metrics.history[-1].loss",
            "final history loss must match train_loss",
            expected=root_loss,
            observed=final_history_loss,
        )


def _validate_config(
    config: GenoLeWMConfig,
    request: CorrectionControlPostflightRequest,
    blockers: list[dict[str, object]],
    *,
    prefix: str = "training_config",
) -> None:
    values = (
        ("run_id", config.run_id, request.expected_run_id),
        ("seed", config.seed, EXPECTED_SEED),
        ("phase", config.phase, "phase1"),
        ("deterministic", config.deterministic, True),
        ("schema_version", config.schema_version, "1.1.0"),
        ("training.max_steps", config.training.max_steps, EXPECTED_STEPS),
        ("training.collapse_log_every_steps", config.training.collapse_log_every_steps, 10),
        ("optimizer.warmup_steps", config.optimizer.warmup_steps, 10),
        ("optimizer.schedule", config.optimizer.schedule, "wsd"),
        ("data.corpus_id", config.data.corpus_id, EXPECTED_CORPUS_ID),
        ("data.corpus_revision", config.data.corpus_revision, EXPECTED_CORPUS_REVISION),
        ("data.batch_size", config.data.batch_size, EXPECTED_BATCH_SIZE),
        ("data.num_workers", config.data.num_workers, 0),
        ("data.shuffle_buffer", config.data.shuffle_buffer, 0),
        ("predictor.architecture", config.predictor.architecture, "cross_attention"),
        ("predictor.n_layers", config.predictor.n_layers, 6),
        ("predictor.n_heads", config.predictor.n_heads, 8),
        ("predictor.d_state", config.predictor.d_state, EXPECTED_D_STATE),
        ("predictor.d_action", config.predictor.d_action, 64),
        ("predictor.dtype", config.predictor.dtype, "fp32"),
        ("action.d_action", config.action.d_action, 64),
        ("action.max_len", config.action.max_len, 16),
        ("action.sub_encoders", config.action.sub_encoders, ("snv",)),
        ("optimizer.name", config.optimizer.name, "adamw"),
        ("optimizer.lr", config.optimizer.lr, EXPECTED_OPTIMIZER_LR),
        ("optimizer.beta1", config.optimizer.beta1, 0.9),
        ("optimizer.beta2", config.optimizer.beta2, 0.95),
        ("optimizer.weight_decay", config.optimizer.weight_decay, 0.1),
        ("optimizer.grad_clip", config.optimizer.grad_clip, 1.0),
        (
            "eval.benchmarks",
            config.eval.benchmarks,
            ("clinvar_coding", "clinvar_noncoding", "rollout"),
        ),
        ("eval.smoke_variants", config.eval.smoke_variants, 100),
        ("observability.log_level", config.observability.log_level, "info"),
        ("observability.redaction_strict", config.observability.redaction_strict, True),
        ("observability.wandb_project", config.observability.wandb_project, None),
        ("runtime.backend", config.runtime.backend, "torch"),
        ("runtime.device", config.runtime.device, "cuda"),
        ("encoder.model_id", config.encoder.model_id, "/carbon"),
        ("encoder.revision", config.encoder.revision, DEFAULT_CARBON_REVISION),
        ("encoder.dtype", config.encoder.dtype, "bf16"),
        ("encoder.normalize", config.encoder.normalize, True),
        (
            "encoder.state_contract_version",
            config.encoder.state_contract_version,
            "l2_normalized_v2",
        ),
        ("encoder.trust_remote_code", config.encoder.trust_remote_code, False),
        ("encoder.state_layer", config.encoder.state_layer, 20),
        ("encoder.pool_type", config.encoder.pool_type, "centered_mean"),
        ("encoder.pool_radius", config.encoder.pool_radius, 8),
    )
    for field, observed, expected in values:
        if not _equal(observed, expected):
            _block(
                blockers,
                f"{prefix}.{field}_mismatch",
                f"{prefix}.{field}",
                "training config does not match the correction-control contract",
                expected=expected,
                observed=observed,
            )


def _validate_checkpoint(
    payload: Mapping[str, Any],
    request: CorrectionControlPostflightRequest,
    blockers: list[dict[str, object]],
) -> None:
    _expect(payload, "schema_version", "1.0.0", "checkpoint.schema_version", blockers)
    _expect(payload, "run_id", request.expected_run_id, "checkpoint.run_id", blockers)
    _expect(
        payload,
        "dataset_snapshot_id",
        request.expected_dataset_snapshot_id,
        "checkpoint.dataset_snapshot_id",
        blockers,
    )
    _expect(payload, "steps_completed", EXPECTED_STEPS, "checkpoint.steps_completed", blockers)
    _expect(payload, "seeds", EXPECTED_SEEDS, "checkpoint.seeds", blockers)
    _validate_checkpoint_state(payload, blockers)
    config = payload.get("config")
    if not isinstance(config, Mapping):
        _block(
            blockers, "checkpoint.config_invalid", "checkpoint.config", "config must be an object"
        )
        return
    expected = {
        "run_id": request.expected_run_id,
        "seed": EXPECTED_SEED,
        "deterministic": True,
        "data.batch_size": EXPECTED_BATCH_SIZE,
        "predictor.d_state": EXPECTED_D_STATE,
        "predictor.dtype": "fp32",
        "action.sub_encoders": ["snv"],
        "encoder.normalize": True,
        "encoder.state_contract_version": "l2_normalized_v2",
        "encoder.effective_normalize": True,
        "encoder.identity_hash": DEFAULT_CARBON_RUNTIME_HASH,
        "encoder.revision": DEFAULT_CARBON_REVISION,
        "encoder.dtype": "bf16",
        "encoder.state_layer": 20,
        "encoder.pool_type": "centered_mean",
        "encoder.pool_radius": 8,
    }
    for field, value in expected.items():
        _expect(config, field, value, f"checkpoint.config.{field}", blockers)


def _validate_checkpoint_state(
    payload: Mapping[str, Any],
    blockers: list[dict[str, object]],
) -> None:
    for field in ("predictor", "action_encoder"):
        path = f"checkpoint.{field}"
        state = payload.get(field)
        if not isinstance(state, Mapping) or not state:
            _block(
                blockers,
                f"{path}_empty",
                path,
                "checkpoint model state must be a non-empty mapping",
            )
            continue
        if _validate_exportable_model_state(state, path, blockers) == 0:
            _block(
                blockers,
                f"{path}_no_numeric_state",
                path,
                "checkpoint model state must contain exportable tensor values",
            )

    optimizer = payload.get("optimizer")
    if not isinstance(optimizer, Mapping) or not optimizer:
        _block(
            blockers,
            "checkpoint.optimizer_empty",
            "checkpoint.optimizer",
            "checkpoint optimizer state must be a non-empty mapping",
        )
        return
    state = optimizer.get("state")
    if not isinstance(state, Mapping) or not state:
        _block(
            blockers,
            "checkpoint.optimizer.state_empty",
            "checkpoint.optimizer.state",
            "AdamW checkpoint state must contain updated parameter entries",
        )
    else:
        _validate_finite_state_tree(state, "checkpoint.optimizer.state", blockers)
        for parameter, parameter_state in state.items():
            path = f"checkpoint.optimizer.state[{parameter!s}].step"
            if not isinstance(parameter_state, Mapping):
                _block(
                    blockers,
                    "checkpoint.optimizer.parameter_state_invalid",
                    f"checkpoint.optimizer.state[{parameter!s}]",
                    "optimizer parameter state must be an object",
                )
                continue
            observed_step = _scalar_number(parameter_state.get("step"))
            if observed_step != float(EXPECTED_STEPS):
                _block(
                    blockers,
                    "checkpoint.optimizer.step_mismatch",
                    path,
                    "every materialized AdamW parameter state must reach step 50",
                    expected=EXPECTED_STEPS,
                    observed=observed_step,
                )
    param_groups = optimizer.get("param_groups")
    if (
        not isinstance(param_groups, Sequence)
        or isinstance(param_groups, str | bytes)
        or not param_groups
    ):
        _block(
            blockers,
            "checkpoint.optimizer.param_groups_invalid",
            "checkpoint.optimizer.param_groups",
            "optimizer checkpoint must contain parameter groups",
        )
        return
    for index, group in enumerate(param_groups):
        params = group.get("params") if isinstance(group, Mapping) else None
        if not isinstance(params, Sequence) or isinstance(params, str | bytes) or not params:
            _block(
                blockers,
                "checkpoint.optimizer.param_group_empty",
                f"checkpoint.optimizer.param_groups[{index}].params",
                "every optimizer parameter group must contain parameters",
            )


def _validate_exportable_model_state(
    state: Mapping[str, Any],
    path: str,
    blockers: list[dict[str, object]],
) -> int:
    tensor_count = 0
    for name, value in state.items():
        value_path = f"{path}.{name!s}"
        tensor_methods = (
            "detach",
            "cpu",
            "contiguous",
            "numel",
            "isfinite",
            "is_floating_point",
        )
        if any(not callable(getattr(value, method, None)) for method in tensor_methods):
            _block(
                blockers,
                f"{value_path}_not_tensor",
                value_path,
                "model state values must be tensors accepted by the deploy exporter",
                observed=type(value).__name__,
            )
            continue
        try:
            is_floating = bool(value.is_floating_point())
        except (RuntimeError, TypeError, ValueError) as exc:
            _block(
                blockers,
                f"{value_path}_dtype_unreadable",
                value_path,
                "model tensor dtype could not be inspected",
                observed=f"{type(exc).__name__}: {exc}",
            )
            continue
        if is_floating and str(getattr(value, "dtype", None)) != "torch.float32":
            _block(
                blockers,
                f"{value_path}_dtype_mismatch",
                value_path,
                "correction-control trainable state must match the declared FP32 dtype",
                expected="torch.float32",
                observed=str(getattr(value, "dtype", None)),
            )
        _validate_finite_state_tree(value, value_path, blockers)
        tensor_count += 1
    return tensor_count


def _validate_finite_state_tree(
    value: object,
    path: str,
    blockers: list[dict[str, object]],
) -> int:
    if isinstance(value, Mapping):
        return sum(
            _validate_finite_state_tree(item, f"{path}.{key!s}", blockers)
            for key, item in value.items()
        )
    if isinstance(value, Sequence) and not isinstance(value, str | bytes):
        return sum(
            _validate_finite_state_tree(item, f"{path}[{index}]", blockers)
            for index, item in enumerate(value)
        )
    if isinstance(value, int | float) and not isinstance(value, bool):
        if not math.isfinite(float(value)):
            _block(
                blockers,
                f"{path}_nonfinite",
                path,
                "checkpoint numeric state must be finite",
                observed=str(value),
            )
        return 1
    numel = getattr(value, "numel", None)
    isfinite = getattr(value, "isfinite", None)
    if not callable(numel) or not callable(isfinite):
        return 0
    try:
        element_count = int(numel())
        finite_result = isfinite()
        finite = bool(finite_result.all().item())
    except (AttributeError, RuntimeError, TypeError, ValueError) as exc:
        _block(
            blockers,
            f"{path}_unreadable",
            path,
            "checkpoint tensor state could not be inspected",
            observed=f"{type(exc).__name__}: {exc}",
        )
        return 0
    if element_count <= 0:
        _block(
            blockers,
            f"{path}_empty",
            path,
            "checkpoint tensors must be non-empty",
        )
    elif not finite:
        _block(
            blockers,
            f"{path}_nonfinite",
            path,
            "checkpoint tensor state must be finite",
        )
    return 1


def _scalar_number(value: object) -> float | None:
    if isinstance(value, int | float) and not isinstance(value, bool):
        number = float(value)
        return number if math.isfinite(number) else None
    numel = getattr(value, "numel", None)
    item = getattr(value, "item", None)
    if not callable(numel) or not callable(item):
        return None
    try:
        if int(numel()) != 1:
            return None
        number = float(item())
    except (RuntimeError, TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _validate_state_contract_audit(
    payload: Mapping[str, Any],
    request: CorrectionControlPostflightRequest,
    blockers: list[dict[str, object]],
) -> None:
    _expect(
        payload,
        "schema_version",
        STATE_CONTRACT_AUDIT_SCHEMA_VERSION,
        "state_contract_audit.schema_version",
        blockers,
    )
    _expect(
        payload,
        "generated_by",
        STATE_CONTRACT_AUDIT_GENERATED_BY,
        "state_contract_audit.generated_by",
        blockers,
    )
    _expect(payload, "ok", True, "state_contract_audit.ok", blockers)
    _expect(
        payload,
        "commit_sha",
        request.expected_commit_sha,
        "state_contract_audit.commit_sha",
        blockers,
    )
    _expect(payload, "blockers", [], "state_contract_audit.blockers", blockers)
    encoder = payload.get("encoder")
    if not isinstance(encoder, Mapping):
        _block(
            blockers,
            "state_contract_audit.encoder_invalid",
            "state_contract_audit.encoder",
            "encoder audit must be an object",
        )
        return
    expected = {
        "revision": DEFAULT_CARBON_REVISION,
        "weights_hash": DEFAULT_CARBON_WEIGHTS_HASH,
        "expected_weights_hash": DEFAULT_CARBON_WEIGHTS_HASH,
        "weights_identity_verified": True,
        "runtime_hash": DEFAULT_CARBON_RUNTIME_HASH,
        "expected_runtime_hash": DEFAULT_CARBON_RUNTIME_HASH,
        "runtime_identity_verified": True,
        "parameters_frozen": True,
        "expected_d_state": EXPECTED_D_STATE,
        "window_bp": 4096,
        "state_layer": 20,
        "pool_type": "centered_mean",
        "pool_radius": 8,
        "pooling_identity_verified": True,
        "dtype": "bf16",
        "normalized_state_contract": "l2_normalized_v2",
    }
    for field, value in expected.items():
        _expect(encoder, field, value, f"state_contract_audit.encoder.{field}", blockers)
    runtime_hash = encoder.get("runtime_hash")
    if not isinstance(runtime_hash, str) or not looks_like_sha256(runtime_hash):
        _block(
            blockers,
            "state_contract_audit.encoder.runtime_hash_invalid",
            "state_contract_audit.encoder.runtime_hash",
            "runtime hash must be a canonical SHA-256 identity",
            observed=runtime_hash,
        )
    rows = payload.get("rows")
    if not isinstance(rows, list) or not rows:
        _block(
            blockers,
            "state_contract_audit.rows_invalid",
            "state_contract_audit.rows",
            "audit rows must be non-empty",
        )
    elif any(not isinstance(row, Mapping) or row.get("ok") is not True for row in rows):
        _block(
            blockers,
            "state_contract_audit.rows_failed",
            "state_contract_audit.rows",
            "every state-contract audit row must pass",
        )
    runtime = _require_mapping(
        payload.get("runtime"),
        "state_contract_audit.runtime",
        blockers,
    )
    if runtime is not None:
        _expect_fields(
            runtime,
            {
                "device": "cuda",
                "hf_hub_offline": True,
                "transformers_offline": True,
                "cuda_available": True,
            },
            "state_contract_audit.runtime",
            blockers,
        )
        device_name = runtime.get("cuda_device_name")
        if not isinstance(device_name, str) or "H200" not in device_name:
            _block(
                blockers,
                "state_contract_audit.runtime.cuda_device_name_mismatch",
                "state_contract_audit.runtime.cuda_device_name",
                "state-contract evidence must come from the H200 correction-control worker",
                expected="a CUDA device name containing H200",
                observed=device_name,
            )


def _validate_job_contract_preflight(
    payload: Mapping[str, Any],
    request: CorrectionControlPostflightRequest,
    blockers: list[dict[str, object]],
) -> None:
    prefix = "job_contract_preflight"
    _expect_fields(
        payload,
        {
            "schema_version": JOB_CONTRACT_PREFLIGHT_SCHEMA_VERSION,
            "generated_by": JOB_CONTRACT_PREFLIGHT_GENERATED_BY,
            "ok": True,
            "issues": [],
        },
        prefix,
        blockers,
    )
    repository = _require_mapping(payload.get("repository"), f"{prefix}.repository", blockers)
    if repository is not None:
        _expect_fields(
            repository,
            {
                "root": ".",
                "expected_commit_sha": request.expected_commit_sha,
                "observed_commit_sha": request.expected_commit_sha,
                "observed_git_root": ".",
                "worktree_clean": True,
                "dirty_paths": [],
            },
            f"{prefix}.repository",
            blockers,
        )

    job = _require_mapping(payload.get("job"), f"{prefix}.job", blockers)
    if job is not None:
        expected_job = {
            "steps": EXPECTED_STEPS,
            "max_windows": EXPECTED_MAX_WINDOWS,
            "clinvar_lines": EXPECTED_CLINVAR_LINES,
            "gnomad_lines": EXPECTED_GNOMAD_LINES,
            "tuple_throughput_samples": EXPECTED_TUPLE_THROUGHPUT_SAMPLES,
            "window_bp": EXPECTED_WINDOW_BP,
            "holdout_chrom": EXPECTED_HOLDOUT_CHROM,
            "carbon_model_dir": EXPECTED_CARBON_MODEL_DIR,
            "carbon_config": EXPECTED_CARBON_CONFIG,
            "carbon_source": EXPECTED_CARBON_SOURCE,
            "corpus_revision": EXPECTED_CORPUS_REVISION,
            "container_image": EXPECTED_CONTAINER_IMAGE,
        }
        _expect_fields(job, expected_job, f"{prefix}.job", blockers)
        run_attempt = job.get("run_attempt")
        if not _is_positive_int(run_attempt):
            _block(
                blockers,
                f"{prefix}.job.run_attempt_invalid",
                f"{prefix}.job.run_attempt",
                "run attempt must be a positive integer",
                observed=run_attempt,
            )
        run_name = job.get("run_name")
        match = _RUN_NAME_RE.fullmatch(run_name) if isinstance(run_name, str) else None
        if match is None or match.group("commit") != request.expected_commit_sha[:12]:
            _block(
                blockers,
                f"{prefix}.job.run_name_invalid",
                f"{prefix}.job.run_name",
                "run name must identify the exact correction commit and positive attempt",
                observed=run_name,
            )
        elif _is_positive_int(run_attempt) and int(match.group("attempt")) != run_attempt:
            _block(
                blockers,
                f"{prefix}.job.run_attempt_incoherent",
                f"{prefix}.job.run_attempt",
                "run attempt must match the attempt encoded in run_name",
                expected=int(match.group("attempt")),
                observed=run_attempt,
            )
        sources = _require_mapping(job.get("sources"), f"{prefix}.job.sources", blockers)
        if sources is not None:
            _expect_fields(
                sources,
                {"clinvar_url": EXPECTED_CLINVAR_URL, "gnomad_url": EXPECTED_GNOMAD_URL},
                f"{prefix}.job.sources",
                blockers,
            )

    config = _require_mapping(payload.get("config"), f"{prefix}.config", blockers)
    if config is not None:
        _expect_fields(
            config,
            {
                "path": EXPECTED_CONFIG_PATH.as_posix(),
                "exists": True,
                "run_id": request.expected_run_id,
                "schema_version": "1.1.0",
            },
            f"{prefix}.config",
            blockers,
        )
        _validate_reported_identity(config, f"{prefix}.config", blockers)
    snapshot = _require_mapping(payload.get("snapshot"), f"{prefix}.snapshot", blockers)
    if snapshot is not None:
        _expect_fields(
            snapshot,
            {
                "path": EXPECTED_SNAPSHOT_PATH.as_posix(),
                "exists": True,
                "snapshot_id": request.expected_dataset_snapshot_id,
                "schema_version": "1.0.0",
            },
            f"{prefix}.snapshot",
            blockers,
        )
        _validate_reported_identity(snapshot, f"{prefix}.snapshot", blockers)


def _validate_source_identity_report(
    payload: Mapping[str, Any],
    request: CorrectionControlPostflightRequest,
    blockers: list[dict[str, object]],
) -> None:
    prefix = "source_identity_report"
    _expect_fields(
        payload,
        {
            "schema_version": SOURCE_IDENTITY_SCHEMA_VERSION,
            "generated_by": SOURCE_IDENTITY_GENERATED_BY,
            "ok": True,
            "commit_sha": request.expected_commit_sha,
            "dataset_snapshot_id": request.expected_dataset_snapshot_id,
        },
        prefix,
        blockers,
    )
    run_name = payload.get("run_name")
    match = _RUN_NAME_RE.fullmatch(run_name) if isinstance(run_name, str) else None
    if match is None or match.group("commit") != request.expected_commit_sha[:12]:
        _block(
            blockers,
            f"{prefix}.run_name_invalid",
            f"{prefix}.run_name",
            "source receipt run name must identify the exact correction commit",
            observed=run_name,
        )

    training_contract = _require_mapping(
        payload.get("training_contract"),
        f"{prefix}.training_contract",
        blockers,
    )
    if training_contract is not None:
        _expect_fields(
            training_contract,
            {
                "active_window_source": "carbon",
                "window_bp": EXPECTED_WINDOW_BP,
                "action_sub_encoders": ["snv"],
                "actions_per_window": EXPECTED_BATCH_SIZE,
                "absolute_variant_fallback": "synthetic_snv",
            },
            f"{prefix}.training_contract",
            blockers,
        )

    sources = _require_mapping(payload.get("sources"), f"{prefix}.sources", blockers)
    if sources is None:
        return
    if set(sources) != {"carbon_corpus", "clinvar", "gnomad"}:
        _block(
            blockers,
            f"{prefix}.sources_keys_mismatch",
            f"{prefix}.sources",
            "source receipt must contain exactly Carbon, ClinVar, and gnomAD",
            expected=["carbon_corpus", "clinvar", "gnomad"],
            observed=sorted(str(key) for key in sources),
        )
    carbon = _require_mapping(
        sources.get("carbon_corpus"), f"{prefix}.sources.carbon_corpus", blockers
    )
    if carbon is not None:
        _expect_fields(
            carbon,
            {
                "revision": EXPECTED_CORPUS_REVISION,
                "dataset_config": EXPECTED_CARBON_CONFIG,
                "default_source": EXPECTED_CARBON_SOURCE,
                "windows": EXPECTED_MAX_WINDOWS,
            },
            f"{prefix}.sources.carbon_corpus",
            blockers,
        )
        _validate_reported_identity(
            carbon.get("artifact"),
            f"{prefix}.sources.carbon_corpus.artifact",
            blockers,
            expected_path="source-mix-windows.jsonl",
        )
    clinvar = _require_mapping(sources.get("clinvar"), f"{prefix}.sources.clinvar", blockers)
    if clinvar is not None:
        _expect_fields(
            clinvar,
            {
                "url": EXPECTED_CLINVAR_URL,
                "md5": EXPECTED_SOURCE_INTEGRITY["clinvar_md5"],
                "subset_lines": EXPECTED_CLINVAR_LINES,
            },
            f"{prefix}.sources.clinvar",
            blockers,
        )
        _validate_reported_identity(
            clinvar.get("archive"),
            f"{prefix}.sources.clinvar.archive",
            blockers,
            expected_path="clinvar_20260415.vcf.gz",
        )
        _validate_reported_identity(
            clinvar.get("filtered_artifact"),
            f"{prefix}.sources.clinvar.filtered_artifact",
            blockers,
            expected_path="clinvar-2026-04-15-snv.vcf.gz",
        )
    gnomad = _require_mapping(sources.get("gnomad"), f"{prefix}.sources.gnomad", blockers)
    if gnomad is not None:
        _expect_fields(
            gnomad,
            {
                "url": EXPECTED_GNOMAD_URL,
                "generation": EXPECTED_SOURCE_INTEGRITY["gnomad_generation"],
                "md5": EXPECTED_SOURCE_INTEGRITY["gnomad_md5"],
                "size_bytes": EXPECTED_SOURCE_INTEGRITY["gnomad_size_bytes"],
                "subset_lines": EXPECTED_GNOMAD_LINES,
            },
            f"{prefix}.sources.gnomad",
            blockers,
        )
        _validate_reported_identity(
            gnomad.get("subset_artifact"),
            f"{prefix}.sources.gnomad.subset_artifact",
            blockers,
            expected_path="gnomad-v4.1-snv.vcf.gz",
        )


def _validate_dataset_manifest(
    payload: Mapping[str, Any],
    request: CorrectionControlPostflightRequest,
    blockers: list[dict[str, object]],
) -> None:
    prefix = "dataset_manifest"
    _expect_fields(
        payload,
        {
            "schema_version": DATASET_MANIFEST_SCHEMA_VERSION,
            "generated_by": DATASET_MANIFEST_GENERATED_BY,
            "snapshot_id": request.expected_dataset_snapshot_id,
        },
        prefix,
        blockers,
    )
    _validate_manifest_sources(payload.get("sources"), blockers)
    file_index = _manifest_file_index(payload.get("files"), prefix, blockers)
    if file_index is None:
        return
    if set(file_index) != set(_EXPECTED_MANIFEST_FILES):
        _block(
            blockers,
            f"{prefix}.file_set_mismatch",
            f"{prefix}.files",
            "manifest must contain exactly Carbon, gnomAD, and ClinVar files with no placed file",
            expected=sorted(_EXPECTED_MANIFEST_FILES),
            observed=sorted(file_index),
        )
    for path, (expected_split, expected_records) in _EXPECTED_MANIFEST_FILES.items():
        row = file_index.get(path)
        if row is None:
            continue
        _expect(row, "split", expected_split, f"{prefix}.files[{path}].split", blockers)
        _validate_reported_identity(
            row,
            f"{prefix}.files[{path}]",
            blockers,
            expected_path=path,
        )
        records = row.get("records")
        if not _is_positive_int(records):
            _block(
                blockers,
                f"{prefix}.files[{path}].records_invalid",
                f"{prefix}.files[{path}].records",
                "manifest file record count must be a positive integer",
                observed=records,
            )
        elif expected_records is not None and records != expected_records:
            _block(
                blockers,
                f"{prefix}.files[{path}].records_mismatch",
                f"{prefix}.files[{path}].records",
                "active Carbon window count must match the correction-control contract",
                expected=expected_records,
                observed=records,
            )

    splits = _require_mapping(payload.get("splits"), f"{prefix}.splits", blockers)
    if splits is None:
        return
    expected_splits = {value[0] for value in _EXPECTED_MANIFEST_FILES.values()}
    if set(splits) != expected_splits:
        _block(
            blockers,
            f"{prefix}.split_set_mismatch",
            f"{prefix}.splits",
            "manifest split set must match the three correction-control files",
            expected=sorted(expected_splits),
            observed=sorted(str(key) for key in splits),
        )
    for path, (split_name, _expected_records) in _EXPECTED_MANIFEST_FILES.items():
        row = file_index.get(path)
        split = splits.get(split_name)
        if row is None or not isinstance(split, Mapping):
            continue
        _expect_cross(
            row.get("records"),
            split.get("records"),
            f"{prefix}.files[{path}].records",
            f"{prefix}.splits.{split_name}.records",
            blockers,
        )


def _validate_dataset_snapshot_report(
    payload: Mapping[str, Any],
    request: CorrectionControlPostflightRequest,
    blockers: list[dict[str, object]],
) -> None:
    prefix = "dataset_snapshot_report"
    _expect_fields(
        payload,
        {
            "schema_version": DATASET_MANIFEST_SCHEMA_VERSION,
            "generated_by": DATASET_SNAPSHOT_GENERATED_BY,
            "snapshot_id": request.expected_dataset_snapshot_id,
            "report_path": DATASET_SNAPSHOT_REPORT_NAME,
            "input_check_path": DATASET_INPUT_CHECK_REPORT_NAME,
            "metadata_path": "dataset_package.json",
        },
        prefix,
        blockers,
    )
    snapshot_spec = _require_mapping(
        payload.get("snapshot_spec"),
        f"{prefix}.snapshot_spec",
        blockers,
    )
    if snapshot_spec is not None:
        _validate_reported_identity(
            snapshot_spec,
            f"{prefix}.snapshot_spec",
            blockers,
            expected_path="dataset-snapshot-snv.json",
        )
    input_check = _require_mapping(
        payload.get("input_check"),
        f"{prefix}.input_check",
        blockers,
    )
    if input_check is not None:
        _validate_reported_identity(
            input_check,
            f"{prefix}.input_check",
            blockers,
            expected_path=DATASET_INPUT_CHECK_REPORT_NAME,
        )
    package = _require_mapping(payload.get("package"), f"{prefix}.package", blockers)
    if package is not None:
        manifest = package.get("manifest")
        _validate_file_identity(
            manifest,
            request.dataset_manifest_json,
            f"{prefix}.package.manifest",
            blockers,
        )

    file_index = _manifest_file_index(payload.get("files"), prefix, blockers)
    if file_index is None:
        return
    if set(file_index) != set(_EXPECTED_SNAPSHOT_SOURCES):
        _block(
            blockers,
            f"{prefix}.file_set_mismatch",
            f"{prefix}.files",
            "snapshot report must describe exactly the three correction-control inputs",
            expected=sorted(_EXPECTED_SNAPSHOT_SOURCES),
            observed=sorted(file_index),
        )
    for path, (kind, source_path) in _EXPECTED_SNAPSHOT_SOURCES.items():
        row = file_index.get(path)
        if row is None:
            continue
        expected_split, expected_records = _EXPECTED_MANIFEST_FILES[path]
        _expect_fields(
            row,
            {"source_path": source_path, "split": expected_split},
            f"{prefix}.files[{path}]",
            blockers,
        )
        _validate_reported_identity(
            row,
            f"{prefix}.files[{path}]",
            blockers,
            expected_path=path,
        )
        source_identity = {
            "path": source_path,
            "sha256": row.get("source_sha256"),
            "size_bytes": row.get("source_size_bytes"),
        }
        _validate_reported_identity(
            source_identity,
            f"{prefix}.files[{path}].source",
            blockers,
            expected_path=source_path,
        )
        records = row.get("records")
        if not _is_positive_int(records):
            _block(
                blockers,
                f"{prefix}.files[{path}].records_invalid",
                f"{prefix}.files[{path}].records",
                f"{kind} snapshot records must be a positive integer",
                observed=records,
            )
        elif expected_records is not None and records != expected_records:
            _block(
                blockers,
                f"{prefix}.files[{path}].records_mismatch",
                f"{prefix}.files[{path}].records",
                "active Carbon window count must match the correction-control contract",
                expected=expected_records,
                observed=records,
            )


def _validate_training_preflight_report(
    payload: Mapping[str, Any],
    request: CorrectionControlPostflightRequest,
    blockers: list[dict[str, object]],
) -> None:
    prefix = "training_preflight_report"
    _expect_fields(
        payload,
        {
            "schema_version": TRAINING_PREFLIGHT_SCHEMA_VERSION,
            "generated_by": TRAINING_PREFLIGHT_GENERATED_BY,
            "ok": True,
            "dataset_snapshot_id": request.expected_dataset_snapshot_id,
            "issues": [],
        },
        prefix,
        blockers,
    )
    training_config = _require_mapping(
        payload.get("training_config"),
        f"{prefix}.training_config",
        blockers,
    )
    if training_config is not None:
        _validate_file_identity(
            training_config,
            request.training_config,
            f"{prefix}.training_config",
            blockers,
        )
        resolved = training_config.get("resolved")
        if not isinstance(resolved, Mapping):
            _block(
                blockers,
                f"{prefix}.training_config.resolved_invalid",
                f"{prefix}.training_config.resolved",
                "training preflight must preserve the resolved config object",
            )
        else:
            try:
                resolved_config = load_config(dict(resolved))
            except GenoLeWMError as exc:
                _block(
                    blockers,
                    f"{prefix}.training_config.resolved_invalid",
                    f"{prefix}.training_config.resolved",
                    "training preflight resolved config is invalid",
                    observed=exc.message or str(exc),
                )
            else:
                _validate_config(
                    resolved_config,
                    request,
                    blockers,
                    prefix=f"{prefix}.training_config.resolved",
                )

    dataset = _require_mapping(payload.get("dataset"), f"{prefix}.dataset", blockers)
    if dataset is not None:
        _expect(
            dataset,
            "snapshot_id",
            request.expected_dataset_snapshot_id,
            f"{prefix}.dataset.snapshot_id",
            blockers,
        )
        _manifest_file_index(dataset.get("files"), f"{prefix}.dataset", blockers)
        core_files = _require_mapping(
            dataset.get("core_files"),
            f"{prefix}.dataset.core_files",
            blockers,
        )
        if core_files is not None:
            _validate_file_identity(
                core_files.get("dataset_manifest.json"),
                request.dataset_manifest_json,
                f"{prefix}.dataset.core_files.dataset_manifest",
                blockers,
            )
            _validate_file_identity(
                core_files.get(DATASET_SNAPSHOT_REPORT_NAME),
                request.dataset_snapshot_report_json,
                f"{prefix}.dataset.core_files.{DATASET_SNAPSHOT_REPORT_NAME}",
                blockers,
            )
    accelerator = _require_mapping(
        payload.get("accelerator"),
        f"{prefix}.accelerator",
        blockers,
    )
    if accelerator is not None:
        _expect_fields(
            accelerator,
            {"requested_device": "cuda", "required": True, "available": True},
            f"{prefix}.accelerator",
            blockers,
        )
        _expect(
            accelerator,
            "min_memory_bytes",
            EXPECTED_MIN_CUDA_MEMORY_BYTES,
            f"{prefix}.accelerator.min_memory_bytes",
            blockers,
        )
        device_count = accelerator.get("device_count")
        if not _is_positive_int(device_count):
            _block(
                blockers,
                f"{prefix}.accelerator.device_count_invalid",
                f"{prefix}.accelerator.device_count",
                "training preflight must observe at least one CUDA device",
                observed=device_count,
            )
        device_name = accelerator.get("device_name")
        if not isinstance(device_name, str) or "H200" not in device_name:
            _block(
                blockers,
                f"{prefix}.accelerator.device_name_mismatch",
                f"{prefix}.accelerator.device_name",
                "training preflight must observe the H200 correction-control worker",
                expected="a CUDA device name containing H200",
                observed=device_name,
            )
        total_memory = accelerator.get("total_memory_bytes")
        if not _is_positive_int(total_memory):
            _block(
                blockers,
                f"{prefix}.accelerator.total_memory_bytes_invalid",
                f"{prefix}.accelerator.total_memory_bytes",
                "observed CUDA memory must be a positive integer",
                observed=total_memory,
            )
        elif isinstance(total_memory, int) and total_memory < EXPECTED_MIN_CUDA_MEMORY_BYTES:
            _block(
                blockers,
                f"{prefix}.accelerator.total_memory_bytes_below_minimum",
                f"{prefix}.accelerator.total_memory_bytes",
                "observed CUDA memory must satisfy the immutable H200 gate",
                expected=f">={EXPECTED_MIN_CUDA_MEMORY_BYTES}",
                observed=total_memory,
            )


def _validate_tuple_throughput_report(
    payload: Mapping[str, Any],
    request: CorrectionControlPostflightRequest,
    blockers: list[dict[str, object]],
) -> None:
    prefix = "tuple_throughput_report"
    _expect_fields(
        payload,
        {
            "schema_version": TUPLE_THROUGHPUT_SCHEMA_VERSION,
            "generated_by": TUPLE_THROUGHPUT_GENERATED_BY,
            "dataset_snapshot_id": request.expected_dataset_snapshot_id,
            "seed": EXPECTED_TUPLE_THROUGHPUT_SEED,
            "requested_samples": EXPECTED_TUPLE_THROUGHPUT_SAMPLES,
            "samples": EXPECTED_TUPLE_THROUGHPUT_SAMPLES,
            "windows": EXPECTED_MAX_WINDOWS,
            "min_tuples_per_second": EXPECTED_MIN_TUPLES_PER_SECOND,
            "passed_min_tuples_per_second": True,
        },
        prefix,
        blockers,
    )
    _validate_file_identity(
        payload.get("dataset_manifest"),
        request.dataset_manifest_json,
        f"{prefix}.dataset_manifest",
        blockers,
    )
    for field in ("gnomad_edits", "clinvar_edits"):
        value = payload.get(field)
        if not _is_positive_int(value):
            _block(
                blockers,
                f"{prefix}.{field}_invalid",
                f"{prefix}.{field}",
                "tuple-builder input edit count must be a positive integer",
                observed=value,
            )
    elapsed = _expect_positive_finite(
        payload.get("elapsed_seconds"),
        f"{prefix}.elapsed_seconds",
        blockers,
    )
    rate = _expect_positive_finite(
        payload.get("tuples_per_second"),
        f"{prefix}.tuples_per_second",
        blockers,
    )
    if elapsed is not None and rate is not None:
        observed_samples = rate * elapsed
        if not math.isclose(
            observed_samples,
            EXPECTED_TUPLE_THROUGHPUT_SAMPLES,
            rel_tol=1e-6,
            abs_tol=1e-6,
        ):
            _block(
                blockers,
                f"{prefix}.rate_incoherent",
                f"{prefix}.tuples_per_second",
                "tuple rate and elapsed time must reconstruct the requested sample count",
                expected=EXPECTED_TUPLE_THROUGHPUT_SAMPLES,
                observed=observed_samples,
            )
    if rate is not None and rate < EXPECTED_MIN_TUPLES_PER_SECOND:
        _block(
            blockers,
            f"{prefix}.rate_below_minimum",
            f"{prefix}.tuples_per_second",
            "tuple-builder rate is below the declared gate",
            expected=EXPECTED_MIN_TUPLES_PER_SECOND,
            observed=rate,
        )


def _validate_job_source_coherence(
    job_report: Mapping[str, Any],
    source_report: Mapping[str, Any],
    blockers: list[dict[str, object]],
) -> None:
    job = job_report.get("job")
    if not isinstance(job, Mapping):
        return
    _expect_cross(
        job.get("run_name"),
        source_report.get("run_name"),
        "job_contract_preflight.job.run_name",
        "source_identity_report.run_name",
        blockers,
    )
    sources = source_report.get("sources")
    if not isinstance(sources, Mapping):
        return
    carbon = sources.get("carbon_corpus")
    clinvar = sources.get("clinvar")
    gnomad = sources.get("gnomad")
    for job_field, source_row, source_field in (
        ("max_windows", carbon, "windows"),
        ("clinvar_lines", clinvar, "subset_lines"),
        ("gnomad_lines", gnomad, "subset_lines"),
    ):
        if isinstance(source_row, Mapping):
            _expect_cross(
                job.get(job_field),
                source_row.get(source_field),
                f"job_contract_preflight.job.{job_field}",
                f"source_identity_report.sources.{source_field}",
                blockers,
            )


def _validate_job_training_preflight_coherence(
    job_report: Mapping[str, Any],
    training_preflight: Mapping[str, Any],
    blockers: list[dict[str, object]],
) -> None:
    job_config = job_report.get("config")
    training_config = training_preflight.get("training_config")
    if not isinstance(job_config, Mapping) or not isinstance(training_config, Mapping):
        return
    resolved = training_config.get("resolved")
    if not isinstance(resolved, Mapping):
        return
    for field in ("run_id", "schema_version"):
        _expect_cross(
            job_config.get(field),
            resolved.get(field),
            f"job_contract_preflight.config.{field}",
            f"training_preflight_report.training_config.resolved.{field}",
            blockers,
        )


def _validate_effective_config_training_preflight_coherence(
    config: GenoLeWMConfig,
    training_preflight: Mapping[str, Any],
    blockers: list[dict[str, object]],
) -> None:
    training_config = training_preflight.get("training_config")
    if not isinstance(training_config, Mapping):
        return
    resolved = training_config.get("resolved")
    if not isinstance(resolved, Mapping):
        return
    _expect_cross(
        config_to_dict(config),
        dict(resolved),
        "training_config.effective",
        "training_preflight_report.training_config.resolved",
        blockers,
    )


def _validate_snapshot_training_preflight_coherence(
    snapshot_report: Mapping[str, Any],
    training_preflight: Mapping[str, Any],
    blockers: list[dict[str, object]],
) -> None:
    dataset = training_preflight.get("dataset")
    if not isinstance(dataset, Mapping):
        return
    core_files = dataset.get("core_files")
    if not isinstance(core_files, Mapping):
        return
    snapshot_input_check = snapshot_report.get("input_check")
    preflight_input_check = core_files.get(DATASET_INPUT_CHECK_REPORT_NAME)
    if isinstance(snapshot_input_check, Mapping) and isinstance(preflight_input_check, Mapping):
        for field in ("path", "sha256", "size_bytes"):
            _expect_cross(
                snapshot_input_check.get(field),
                preflight_input_check.get(field),
                f"dataset_snapshot_report.input_check.{field}",
                (
                    "training_preflight_report.dataset.core_files."
                    f"{DATASET_INPUT_CHECK_REPORT_NAME}.{field}"
                ),
                blockers,
            )


def _validate_job_snapshot_coherence(
    job_report: Mapping[str, Any],
    snapshot_report: Mapping[str, Any],
    blockers: list[dict[str, object]],
) -> None:
    job_snapshot = job_report.get("snapshot")
    snapshot_spec = snapshot_report.get("snapshot_spec")
    if not isinstance(job_snapshot, Mapping) or not isinstance(snapshot_spec, Mapping):
        return
    for field in ("sha256", "size_bytes"):
        _expect_cross(
            job_snapshot.get(field),
            snapshot_spec.get(field),
            f"job_contract_preflight.snapshot.{field}",
            f"dataset_snapshot_report.snapshot_spec.{field}",
            blockers,
        )


def _validate_source_snapshot_coherence(
    source_report: Mapping[str, Any],
    snapshot_report: Mapping[str, Any],
    blockers: list[dict[str, object]],
) -> None:
    sources = source_report.get("sources")
    snapshot_files = _manifest_file_index(
        snapshot_report.get("files"),
        "dataset_snapshot_report",
        [],
    )
    if not isinstance(sources, Mapping) or snapshot_files is None:
        return
    source_artifacts = {
        "carbon/source-mix-windows.jsonl": (
            sources.get("carbon_corpus"),
            "artifact",
        ),
        "gnomad/v4.1/variants.parquet": (sources.get("gnomad"), "subset_artifact"),
        "clinvar/2026-04-15/variants.parquet": (
            sources.get("clinvar"),
            "filtered_artifact",
        ),
    }
    for path, (source_row, artifact_field) in source_artifacts.items():
        snapshot_row = snapshot_files.get(path)
        if not isinstance(source_row, Mapping) or snapshot_row is None:
            continue
        artifact = source_row.get(artifact_field)
        if not isinstance(artifact, Mapping):
            continue
        for source_field, snapshot_field in (
            ("sha256", "source_sha256"),
            ("size_bytes", "source_size_bytes"),
        ):
            _expect_cross(
                artifact.get(source_field),
                snapshot_row.get(snapshot_field),
                f"source_identity_report.sources.{artifact_field}.{source_field}",
                f"dataset_snapshot_report.files[{path}].{snapshot_field}",
                blockers,
            )


def _validate_snapshot_manifest_coherence(
    snapshot_report: Mapping[str, Any],
    manifest: Mapping[str, Any],
    blockers: list[dict[str, object]],
) -> None:
    snapshot_files = _manifest_file_index(
        snapshot_report.get("files"),
        "dataset_snapshot_report",
        [],
    )
    manifest_files = _manifest_file_index(manifest.get("files"), "dataset_manifest", [])
    if snapshot_files is None or manifest_files is None:
        return
    for path in sorted(set(snapshot_files) & set(manifest_files)):
        for field in ("sha256", "size_bytes", "split", "records"):
            _expect_cross(
                snapshot_files[path].get(field),
                manifest_files[path].get(field),
                f"dataset_snapshot_report.files[{path}].{field}",
                f"dataset_manifest.files[{path}].{field}",
                blockers,
            )


def _validate_manifest_training_preflight_coherence(
    manifest: Mapping[str, Any],
    training_preflight: Mapping[str, Any],
    request: CorrectionControlPostflightRequest,
    blockers: list[dict[str, object]],
) -> None:
    dataset = training_preflight.get("dataset")
    if not isinstance(dataset, Mapping):
        return
    manifest_files = _manifest_file_index(manifest.get("files"), "dataset_manifest", [])
    preflight_files = _manifest_file_index(
        dataset.get("files"),
        "training_preflight_report.dataset",
        [],
    )
    if manifest_files is None or preflight_files is None:
        return
    if set(preflight_files) != set(_EXPECTED_MANIFEST_FILES):
        _block(
            blockers,
            "training_preflight_report.dataset.file_set_mismatch",
            "training_preflight_report.dataset.files",
            "training preflight must inspect exactly the active three-file manifest",
            expected=sorted(_EXPECTED_MANIFEST_FILES),
            observed=sorted(preflight_files),
        )
    for path in sorted(set(manifest_files) & set(preflight_files)):
        manifest_row = manifest_files[path]
        preflight_row = preflight_files[path]
        for field in ("sha256", "size_bytes", "split", "records"):
            _expect_cross(
                manifest_row.get(field),
                preflight_row.get(field),
                f"dataset_manifest.files[{path}].{field}",
                f"training_preflight_report.dataset.files[{path}].{field}",
                blockers,
            )
    _expect_cross(
        request.expected_dataset_snapshot_id,
        dataset.get("snapshot_id"),
        "expected.dataset_snapshot_id",
        "training_preflight_report.dataset.snapshot_id",
        blockers,
    )


def _validate_source_manifest_coherence(
    source_report: Mapping[str, Any],
    manifest: Mapping[str, Any],
    blockers: list[dict[str, object]],
) -> None:
    _expect_cross(
        source_report.get("dataset_snapshot_id"),
        manifest.get("snapshot_id"),
        "source_identity_report.dataset_snapshot_id",
        "dataset_manifest.snapshot_id",
        blockers,
    )
    sources = source_report.get("sources")
    manifest_files = _manifest_file_index(manifest.get("files"), "dataset_manifest", [])
    if not isinstance(sources, Mapping) or manifest_files is None:
        return
    carbon = sources.get("carbon_corpus")
    carbon_row = manifest_files.get("carbon/source-mix-windows.jsonl")
    if not isinstance(carbon, Mapping) or carbon_row is None:
        return
    artifact = carbon.get("artifact")
    if not isinstance(artifact, Mapping):
        return
    for field in ("sha256", "size_bytes"):
        _expect_cross(
            artifact.get(field),
            carbon_row.get(field),
            f"source_identity_report.sources.carbon_corpus.artifact.{field}",
            f"dataset_manifest.files[carbon/source-mix-windows.jsonl].{field}",
            blockers,
        )


def _validate_manifest_tuple_throughput_coherence(
    manifest: Mapping[str, Any],
    tuple_report: Mapping[str, Any],
    request: CorrectionControlPostflightRequest,
    blockers: list[dict[str, object]],
) -> None:
    identity = tuple_report.get("dataset_manifest")
    if not isinstance(identity, Mapping):
        return
    _expect_cross(
        manifest.get("snapshot_id"),
        tuple_report.get("dataset_snapshot_id"),
        "dataset_manifest.snapshot_id",
        "tuple_throughput_report.dataset_snapshot_id",
        blockers,
    )
    _expect_cross(
        sha256_file(request.dataset_manifest_json),
        identity.get("sha256"),
        "dataset_manifest.file.sha256",
        "tuple_throughput_report.dataset_manifest.sha256",
        blockers,
    )


def _validate_metrics_checkpoint_coherence(
    metrics: Mapping[str, Any],
    checkpoint: Mapping[str, Any],
    blockers: list[dict[str, object]],
) -> None:
    for field in ("run_id", "dataset_snapshot_id", "steps_completed"):
        _expect_cross(
            metrics.get(field),
            checkpoint.get(field),
            f"metrics.{field}",
            f"checkpoint.{field}",
            blockers,
        )


def _validate_config_checkpoint_coherence(
    config: GenoLeWMConfig,
    checkpoint: Mapping[str, Any],
    blockers: list[dict[str, object]],
) -> None:
    checkpoint_config = checkpoint.get("config")
    if not isinstance(checkpoint_config, Mapping):
        return
    fields = {
        "run_id": config.run_id,
        "seed": config.seed,
        "deterministic": config.deterministic,
        "data.batch_size": config.data.batch_size,
        "predictor.d_state": config.predictor.d_state,
        "predictor.dtype": config.predictor.dtype,
        "action.sub_encoders": list(config.action.sub_encoders),
        "encoder.normalize": config.encoder.normalize,
        "encoder.state_contract_version": config.encoder.state_contract_version,
        "encoder.revision": config.encoder.revision,
        "encoder.dtype": config.encoder.dtype,
        "encoder.state_layer": config.encoder.state_layer,
        "encoder.pool_type": config.encoder.pool_type,
        "encoder.pool_radius": config.encoder.pool_radius,
    }
    for field, value in fields.items():
        _expect_cross(
            value,
            checkpoint_config.get(field),
            f"training_config.{field}",
            f"checkpoint.config.{field}",
            blockers,
        )


def _validate_audit_checkpoint_coherence(
    audit: Mapping[str, Any],
    checkpoint: Mapping[str, Any],
    blockers: list[dict[str, object]],
) -> None:
    encoder = audit.get("encoder")
    checkpoint_config = checkpoint.get("config")
    if not isinstance(encoder, Mapping) or not isinstance(checkpoint_config, Mapping):
        return
    _expect_cross(
        encoder.get("runtime_hash"),
        checkpoint_config.get("encoder.identity_hash"),
        "state_contract_audit.encoder.runtime_hash",
        "checkpoint.config.encoder.identity_hash",
        blockers,
    )
    for audit_field, checkpoint_field in (
        ("revision", "encoder.revision"),
        ("state_layer", "encoder.state_layer"),
        ("pool_type", "encoder.pool_type"),
        ("pool_radius", "encoder.pool_radius"),
    ):
        _expect_cross(
            encoder.get(audit_field),
            checkpoint_config.get(checkpoint_field),
            f"state_contract_audit.encoder.{audit_field}",
            f"checkpoint.config.{checkpoint_field}",
            blockers,
        )


def _validate_manifest_sources(
    raw: object,
    blockers: list[dict[str, object]],
) -> None:
    prefix = "dataset_manifest.sources"
    if not isinstance(raw, list) or any(not isinstance(row, Mapping) for row in raw):
        _block(
            blockers,
            f"{prefix}_invalid",
            prefix,
            "manifest sources must be a list of source objects",
            observed=raw,
        )
        return
    by_name: dict[str, Mapping[str, Any]] = {}
    duplicates: list[str] = []
    for row in raw:
        assert isinstance(row, Mapping)
        name = row.get("name")
        if not isinstance(name, str):
            _block(
                blockers,
                f"{prefix}.name_invalid",
                f"{prefix}.name",
                "manifest source name must be a string",
                observed=name,
            )
            continue
        if name in by_name:
            duplicates.append(name)
        by_name[name] = row
    expected = {
        "Carbon pretraining corpus": {
            "revision": EXPECTED_CORPUS_REVISION,
            "url": "https://huggingface.co/datasets/HuggingFaceBio/carbon-pretraining-corpus",
        },
        "gnomAD": {
            "revision": "v4.1 chr22 generation 1713312296186865",
            "url": EXPECTED_GNOMAD_URL,
        },
        "ClinVar": {
            "revision": "2026-04-15 md5:e63b5c3a046010c098cc70e81bebaa8d",
            "url": EXPECTED_CLINVAR_URL,
        },
    }
    if set(by_name) != set(expected) or duplicates:
        _block(
            blockers,
            f"{prefix}_mismatch",
            prefix,
            "manifest sources must be exactly the three pinned correction-control sources",
            expected=sorted(expected),
            observed={"names": sorted(by_name), "duplicates": sorted(duplicates)},
        )
    for name, fields in expected.items():
        row = by_name.get(name)
        if row is not None:
            _expect_fields(row, fields, f"{prefix}[{name}]", blockers)


def _manifest_file_index(
    raw: object,
    prefix: str,
    blockers: list[dict[str, object]],
) -> dict[str, Mapping[str, Any]] | None:
    if not isinstance(raw, list) or not raw or any(not isinstance(row, Mapping) for row in raw):
        _block(
            blockers,
            f"{prefix}.files_invalid",
            f"{prefix}.files",
            "files must be a non-empty list of objects",
        )
        return None
    index: dict[str, Mapping[str, Any]] = {}
    duplicates: list[str] = []
    for row in raw:
        assert isinstance(row, Mapping)
        path = row.get("path")
        if not isinstance(path, str) or not _is_public_relative_path(path):
            _block(
                blockers,
                f"{prefix}.file_path_invalid",
                f"{prefix}.files.path",
                "file path must be a public relative path",
                observed=path,
            )
            continue
        if path in index:
            duplicates.append(path)
        index[path] = row
    if duplicates:
        _block(
            blockers,
            f"{prefix}.file_paths_duplicate",
            f"{prefix}.files",
            "file paths must be unique",
            observed=sorted(duplicates),
        )
    return index


def _require_mapping(
    raw: object,
    path: str,
    blockers: list[dict[str, object]],
) -> Mapping[str, Any] | None:
    if not isinstance(raw, Mapping):
        _block(blockers, f"{path}_invalid", path, "artifact field must be an object")
        return None
    return raw


def _validate_reported_identity(
    raw: object,
    field: str,
    blockers: list[dict[str, object]],
    *,
    expected_path: str | None = None,
) -> None:
    if not isinstance(raw, Mapping):
        _block(blockers, f"{field}.invalid", field, "artifact identity must be an object")
        return
    path = raw.get("path")
    if not isinstance(path, str) or not _is_public_relative_path(path):
        _block(
            blockers,
            f"{field}.path_invalid",
            f"{field}.path",
            "artifact path must be a public relative path",
            observed=path,
        )
    elif expected_path is not None and path != expected_path:
        _block(
            blockers,
            f"{field}.path_mismatch",
            f"{field}.path",
            "artifact path does not match the correction-control contract",
            expected=expected_path,
            observed=path,
        )
    digest = raw.get("sha256")
    if not isinstance(digest, str) or not looks_like_sha256(digest):
        _block(
            blockers,
            f"{field}.sha256_invalid",
            f"{field}.sha256",
            "artifact SHA-256 must be canonical",
            observed=digest,
        )
    size_bytes = raw.get("size_bytes")
    if not _is_positive_int(size_bytes):
        _block(
            blockers,
            f"{field}.size_bytes_invalid",
            f"{field}.size_bytes",
            "artifact size must be a positive integer",
            observed=size_bytes,
        )


def _expect_fields(
    payload: Mapping[str, Any],
    expected: Mapping[str, object],
    prefix: str,
    blockers: list[dict[str, object]],
) -> None:
    for field, value in expected.items():
        _expect(payload, field, value, f"{prefix}.{field}", blockers)


def _expect_positive_finite(
    value: object,
    path: str,
    blockers: list[dict[str, object]],
) -> float | None:
    number = _expect_finite(value, path, blockers)
    if number is not None and number <= 0:
        _block(
            blockers,
            f"{path}_nonpositive",
            path,
            "value must be positive",
            observed=number,
        )
        return None
    return number


def _is_positive_int(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value > 0


def _is_public_relative_path(value: str) -> bool:
    path = Path(value)
    return bool(value.strip()) and not path.is_absolute() and ".." not in path.parts


def _expect(
    payload: Mapping[str, Any],
    key: str,
    expected: object,
    path: str,
    blockers: list[dict[str, object]],
) -> None:
    observed = payload.get(key)
    if not _equal(observed, expected):
        _block(
            blockers,
            f"{path}_mismatch",
            path,
            "artifact field does not match the correction-control contract",
            expected=expected,
            observed=observed,
        )


def _expect_cross(
    left: object,
    right: object,
    left_path: str,
    right_path: str,
    blockers: list[dict[str, object]],
) -> None:
    if not _equal(left, right):
        _block(
            blockers,
            "artifacts.coherence_mismatch",
            right_path,
            "artifact fields disagree",
            expected={"path": left_path, "value": left},
            observed={"path": right_path, "value": right},
        )


def _expect_finite(
    value: object,
    path: str,
    blockers: list[dict[str, object]],
) -> float | None:
    if isinstance(value, bool) or not isinstance(value, int | float):
        _block(blockers, f"{path}_invalid", path, "value must be a finite number", observed=value)
        return None
    number = float(value)
    if not math.isfinite(number):
        _block(blockers, f"{path}_nonfinite", path, "value must be finite", observed=str(value))
        return None
    return number


def _equal(observed: object, expected: object) -> bool:
    if isinstance(expected, bool):
        return observed is expected
    if isinstance(expected, int):
        return not isinstance(observed, bool) and isinstance(observed, int) and observed == expected
    if expected is None:
        return observed is None
    return observed == expected


def _artifact_summary(path: Path) -> dict[str, object]:
    summary: dict[str, object] = {"path": path.name, "exists": path.is_file()}
    if path.is_file():
        try:
            summary.update({"sha256": sha256_file(path), "size_bytes": path.stat().st_size})
        except OSError as exc:
            summary["identity_error"] = str(exc)
    return summary


def _block(
    blockers: list[dict[str, object]],
    code: str,
    path: str,
    message: str,
    *,
    expected: object | None = None,
    observed: object | None = None,
) -> None:
    blocker: dict[str, object] = {"code": code, "path": path, "message": message}
    if expected is not None:
        blocker["expected"] = expected
    if observed is not None:
        blocker["observed"] = observed
    blockers.append(blocker)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
