# SPDX-License-Identifier: Apache-2.0
"""Single-process Carbon-backed training launcher.

This module owns the real training orchestration boundary used by
``geno-lewm-train --carbon-train``. It remains optional-runtime code:
imports are lightweight, while execution requires a ``geno-lewm[train]``
environment with local Carbon model files and a packaged dataset.
"""

from __future__ import annotations

import importlib
import json
import math
import platform
import shutil
import time
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from geno_lewm.action import ActionEncoder, EditSpec, EditType
from geno_lewm.config import GenoLeWMConfig, write_resolved_config
from geno_lewm.config._state_contract import encoder_uses_normalized_states
from geno_lewm.data import (
    DEFAULT_EDIT_SOURCE_COUNTS,
    DEFAULT_SOURCE_FALLBACKS,
    SOURCE_CLINVAR,
    SOURCE_GNOMAD_COMMON,
    SOURCE_SYNTHETIC_INDEL,
    SOURCE_SYNTHETIC_SNV,
    EditSourceCount,
    GenoLeWMDataset,
    TrainingDatasetItem,
    WindowContext,
    iter_clinvar_shard,
    iter_gnomad_shard,
    synthetic_indel_provider,
    synthetic_snv_provider,
    variant_provider,
)
from geno_lewm.encoder import CarbonStateEncoder
from geno_lewm.encoder._identity import encoder_identity_hash
from geno_lewm.errors import InputError, RuntimeSetupError
from geno_lewm.observability import get_logger
from geno_lewm.predictor import build_predictor
from geno_lewm.provenance import sha256_file
from geno_lewm.training._phase_contract import require_executable_training_phase
from geno_lewm.training.preflight import REPORT_NAME, TrainingPreflightReport
from geno_lewm.training.trainer import (
    TorchTrainer,
    TrainerSeeds,
    build_adamw_optimizer,
    configure_torch_reproducibility,
    encode_training_batch,
)

__all__ = [
    "CARBON_CHECKPOINT_NAME",
    "CARBON_LOG_NAME",
    "CARBON_METRICS_NAME",
    "CARBON_TRAINING_METADATA_NAME",
    "CarbonTrainingReport",
    "run_carbon_training",
]

CARBON_CHECKPOINT_NAME = "predictor_checkpoint.pt"
CARBON_LOG_NAME = "train.log"
CARBON_METRICS_NAME = "metrics.json"
CARBON_TRAINING_METADATA_NAME = "training_run.json"
_SCHEMA_VERSION = "1.0.0"
_TRAINING_RUN_PACKAGE_GENERATED_BY = "tools.release.training_run"
_RESOLVED_CONFIG_NAME = "training_config.effective.yaml"
_ALL_V1_SUB_ENCODERS = ("snv", "ins", "del", "mnv")
_SNV_ONLY_EDIT_SOURCE_COUNTS = (
    EditSourceCount(SOURCE_GNOMAD_COMMON, 3),
    EditSourceCount(SOURCE_SYNTHETIC_SNV, 4),
    EditSourceCount(SOURCE_CLINVAR, 1),
)


@dataclass(frozen=True, slots=True)
class CarbonTrainingReport:
    """Summary emitted by the real Carbon-backed trainer."""

    run_id: str
    run_dir: Path
    dataset_snapshot_id: str
    steps_requested: int
    steps_completed: int
    resumed_from_step: int
    sample_count: int
    final_loss: float
    checkpoint_path: Path
    resume_checkpoint_path: Path | None
    metrics_path: Path
    log_path: Path
    config_path: Path
    preflight_path: Path | None
    training_metadata_path: Path

    def to_dict(self) -> dict[str, object]:
        return {
            "run_id": self.run_id,
            "run_dir": str(self.run_dir),
            "dataset_snapshot_id": self.dataset_snapshot_id,
            "steps_requested": self.steps_requested,
            "steps_completed": self.steps_completed,
            "resumed_from_step": self.resumed_from_step,
            "sample_count": self.sample_count,
            "final_loss": self.final_loss,
            "checkpoint_path": str(self.checkpoint_path),
            "resume_checkpoint_path": (
                None if self.resume_checkpoint_path is None else str(self.resume_checkpoint_path)
            ),
            "metrics_path": str(self.metrics_path),
            "log_path": str(self.log_path),
            "config_path": str(self.config_path),
            "preflight_path": None if self.preflight_path is None else str(self.preflight_path),
            "training_metadata_path": str(self.training_metadata_path),
        }


@dataclass(frozen=True, slots=True)
class _ResumeCheckpoint:
    path: Path
    steps_completed: int
    payload: dict[str, Any]


def run_carbon_training(
    *,
    config: GenoLeWMConfig,
    dataset_dir: Path,
    carbon_model_dir: Path,
    run_dir: Path,
    steps: int,
    command: str,
    commit_sha: str,
    package_version: str,
    preflight_report: TrainingPreflightReport | None = None,
    resume_from: Path | None = None,
) -> CarbonTrainingReport:
    """Run a single-process Carbon-backed training job."""
    require_executable_training_phase(config, boundary="run_carbon_training")
    _require_positive_int("steps", steps)
    _require_positive_int("data.batch_size", config.data.batch_size)
    run_dir.mkdir(parents=True, exist_ok=True)
    config_path = write_resolved_config(config, run_dir / _RESOLVED_CONFIG_NAME)
    log_path = run_dir / CARBON_LOG_NAME
    metrics_path = run_dir / CARBON_METRICS_NAME
    checkpoint_path = run_dir / CARBON_CHECKPOINT_NAME
    metadata_path = run_dir / CARBON_TRAINING_METADATA_NAME
    run_dataset_manifest_path = run_dir / "dataset_manifest.json"
    preflight_path = run_dir / REPORT_NAME if (run_dir / REPORT_NAME).is_file() else None

    dataset_manifest = _load_dataset_manifest(dataset_dir)
    shutil.copy2(dataset_dir / "dataset_manifest.json", run_dataset_manifest_path)
    dataset_snapshot_id = _required_text(dataset_manifest, "snapshot_id")
    dataset_files = _dataset_files(dataset_manifest)
    windows = tuple(_load_windows(dataset_dir, dataset_files))
    if not windows:
        raise InputError("Carbon training requires at least one source window")
    gnomad_edits = tuple(_load_gnomad_edits(dataset_dir, dataset_files))
    clinvar_edits = tuple(_load_clinvar_edits(dataset_dir, dataset_files))
    if not gnomad_edits:
        raise InputError("Carbon training requires at least one gnomAD edit")
    carbon_identity_hash = _carbon_identity_hash(
        carbon_model_dir,
        state_contract_version=config.encoder.state_contract_version,
    )

    device = _training_device(config)
    seeds = TrainerSeeds.from_base_seed(config.seed)
    determinism = configure_torch_reproducibility(
        seed=seeds.predictor, deterministic=config.deterministic
    )
    providers, edit_source_counts = _training_edit_contract(
        config,
        gnomad_edits=gnomad_edits,
        clinvar_edits=clinvar_edits,
    )
    iterator = _repeat_training_items(
        windows,
        providers,
        seed=seeds.data,
        fallback_sources=_dataset_fallback_sources(windows),
        mix=edit_source_counts,
    )
    resumed_from_step = 0
    resume_checkpoint: _ResumeCheckpoint | None = None
    if resume_from is not None:
        resume_checkpoint = _validate_resume_checkpoint_payload(
            _load_torch_checkpoint(resume_from),
            path=resume_from,
            config=config,
            dataset_snapshot_id=dataset_snapshot_id,
            seeds=seeds,
            target_steps=steps,
            encoder_identity_hash=carbon_identity_hash,
        )
        resumed_from_step = resume_checkpoint.steps_completed
        _skip_training_items(
            iterator,
            item_count=resumed_from_step * config.data.batch_size,
        )

    encoder = CarbonStateEncoder(
        str(carbon_model_dir),
        config.encoder.revision,
        dtype=config.encoder.dtype,
        state_layer=config.encoder.state_layer,
        pool_type=config.encoder.pool_type,
        pool_radius=config.encoder.pool_radius,
        normalize=encoder_uses_normalized_states(config.encoder),
        encoder_hash=carbon_identity_hash,
        device=device,
        local_files_only=True,
        trust_remote_code=config.encoder.trust_remote_code,
    )
    training_started_at = time.perf_counter()
    first_items = _next_batch(iterator, config.data.batch_size)
    first_batch = _encode_items(encoder, first_items, device=device)
    observed_d_state = int(first_batch.state.shape[1])
    if observed_d_state != config.predictor.d_state:
        raise InputError(
            "predictor.d_state must match the Carbon encoder state width",
            details={"predictor.d_state": config.predictor.d_state, "observed": observed_d_state},
            remediation="set predictor.d_state to the encoder output width in the training config",
        )

    action_encoder = _move_trainable_to_device(
        ActionEncoder(d_action=config.action.d_action),
        device,
        label="action_encoder",
    )
    predictor = _move_trainable_to_device(build_predictor(config), device, label="predictor")
    optimizer = build_adamw_optimizer(
        predictor=predictor, action_encoder=action_encoder, config=config
    )
    if resume_checkpoint is not None:
        _restore_resume_checkpoint(
            resume_checkpoint.payload,
            predictor=predictor,
            action_encoder=action_encoder,
            optimizer=optimizer,
        )
    trainer = TorchTrainer(
        predictor=predictor,
        action_encoder=action_encoder,
        optimizer=optimizer,
        config=config,
        total_steps=steps,
    )
    progress_every = max(1, int(config.training.collapse_log_every_steps))

    step_results = []
    collapse_alert_count = 0
    sample_count = resumed_from_step * config.data.batch_size
    log_mode = "a" if resumed_from_step else "w"
    progress_logger = get_logger("training", run_id=config.run_id, log_dir=run_dir)
    with log_path.open(log_mode, encoding="utf-8") as log:
        if resumed_from_step:
            log.write(
                json.dumps(
                    {
                        "event": "train.resume",
                        "run_id": config.run_id,
                        "resume_from": None
                        if resume_from is None
                        else _public_resume_path(resume_from),
                        "resumed_from_step": resumed_from_step,
                        "target_steps": steps,
                    },
                    sort_keys=True,
                )
                + "\n"
            )
        else:
            log.write(json.dumps({"event": "train.start", "run_id": config.run_id}) + "\n")
        current_batch = first_batch
        first_step = resumed_from_step + 1
        for step in range(first_step, steps + 1):
            if step > first_step:
                current_batch = _encode_items(
                    encoder,
                    _next_batch(iterator, config.data.batch_size),
                    device=device,
                )
            result = trainer.train_step(current_batch, step=step)
            step_results.append(result)
            collapse_alerts = _last_collapse_alerts(trainer)
            collapse_alert_count += len(collapse_alerts)
            sample_count += len(current_batch.window_ids)
            log.write(json.dumps({"event": "train.step", **result.to_dict()}) + "\n")
            if step in (first_step, steps) or step % progress_every == 0:
                progress_logger.info(
                    "training.metric",
                    step=step,
                    name="sample_count",
                    value=sample_count,
                    unit="samples",
                    kind="counter",
                )
                progress_logger.info(
                    "training.metric",
                    step=step,
                    name="loss",
                    value=result.loss,
                    unit="unitless",
                    kind="gauge",
                )
                progress_logger.info(
                    "training.metric",
                    step=step,
                    name="pred_var_per_dim",
                    value=result.pred_var_per_dim,
                    unit="unitless",
                    kind="gauge",
                )
                _write_checkpoint(
                    checkpoint_path,
                    predictor=predictor,
                    action_encoder=action_encoder,
                    optimizer=optimizer,
                    config=config,
                    dataset_snapshot_id=dataset_snapshot_id,
                    steps=step,
                    seeds=seeds,
                    encoder_identity_hash=carbon_identity_hash,
                )
            for alert in collapse_alerts:
                log.write(
                    json.dumps(
                        {"event": "training.collapse.alert", "step": step, **alert},
                        sort_keys=True,
                    )
                    + "\n"
                )
        log.write(json.dumps({"event": "train.end", "steps_completed": steps}) + "\n")
    elapsed_seconds = max(time.perf_counter() - training_started_at, 1e-9)

    final = step_results[-1]
    _write_metrics(
        metrics_path,
        config=config,
        steps=steps,
        resumed_from_step=resumed_from_step,
        sample_count=sample_count,
        final_loss=final.loss,
        step_results=step_results,
        elapsed_seconds=elapsed_seconds,
        collapse_alert_count=collapse_alert_count,
        dataset_snapshot_id=dataset_snapshot_id,
        resume_checkpoint_path=resume_from,
    )
    _write_checkpoint(
        checkpoint_path,
        predictor=predictor,
        action_encoder=action_encoder,
        optimizer=optimizer,
        config=config,
        dataset_snapshot_id=dataset_snapshot_id,
        steps=steps,
        seeds=seeds,
        encoder_identity_hash=carbon_identity_hash,
    )
    _write_training_metadata(
        metadata_path,
        config=config,
        command=command,
        commit_sha=commit_sha,
        package_version=package_version,
        dataset_snapshot_id=dataset_snapshot_id,
        seeds=seeds,
        determinism=determinism.to_dict(),
        artifacts={
            "training_config": config_path.name,
            "metrics": metrics_path.name,
            "logs": [log_path.name],
            "checkpoint_files": [checkpoint_path.name],
            "dataset_manifest": run_dataset_manifest_path.name,
        },
        preflight_report=preflight_report,
        final_loss=final.loss,
        sample_count=sample_count,
        resumed_from_step=resumed_from_step,
        resume_checkpoint_path=resume_from,
    )
    return CarbonTrainingReport(
        run_id=config.run_id,
        run_dir=run_dir,
        dataset_snapshot_id=dataset_snapshot_id,
        steps_requested=steps,
        steps_completed=steps,
        resumed_from_step=resumed_from_step,
        sample_count=sample_count,
        final_loss=final.loss,
        checkpoint_path=checkpoint_path,
        resume_checkpoint_path=resume_from,
        metrics_path=metrics_path,
        log_path=log_path,
        config_path=config_path,
        preflight_path=preflight_path,
        training_metadata_path=metadata_path,
    )


def _skip_training_items(
    iterator: Iterator[TrainingDatasetItem],
    *,
    item_count: int,
) -> None:
    for index in range(item_count):
        try:
            next(iterator)
        except StopIteration as exc:
            raise InputError(
                "training dataset exhausted while advancing to resume checkpoint",
                details={"items_to_skip": item_count, "items_skipped": index},
                remediation="resume with the same dataset snapshot and training config",
            ) from exc


def _repeat_training_items(
    windows: Sequence[WindowContext],
    providers: Mapping[str, Any],
    *,
    seed: int,
    fallback_sources: Mapping[str, str],
    mix: Sequence[EditSourceCount] = DEFAULT_EDIT_SOURCE_COUNTS,
) -> Iterator[TrainingDatasetItem]:
    """Yield deterministic repeated passes over a finite release dataset."""
    epoch = 0
    while True:
        dataset = GenoLeWMDataset(
            windows,
            providers,
            seed=seed + epoch,
            fallback_sources=fallback_sources,
            mix=mix,
        )
        produced = 0
        for item in dataset.iter_with_source_windows():
            produced += 1
            yield item
        if produced == 0:
            raise InputError(
                "training dataset epoch produced no usable tuples",
                details={"epoch": epoch, "window_count": len(windows)},
                remediation=(
                    "provide placed windows with matching edit shards or restore explicit "
                    "fallback sources for the active release dataset"
                ),
            )
        epoch += 1


def _training_edit_contract(
    config: GenoLeWMConfig,
    *,
    gnomad_edits: Sequence[EditSpec],
    clinvar_edits: Sequence[EditSpec],
) -> tuple[dict[str, Any], tuple[EditSourceCount, ...]]:
    """Resolve the configured edit surface into providers and source counts."""
    sub_encoders = config.action.sub_encoders
    if sub_encoders == ("snv",):
        gnomad_snv = tuple(edit for edit in gnomad_edits if edit.edit_type is EditType.SNV)
        clinvar_snv = tuple(edit for edit in clinvar_edits if edit.edit_type is EditType.SNV)
        if not gnomad_snv:
            raise InputError("SNV-only training requires at least one gnomAD SNV")
        return (
            {
                SOURCE_GNOMAD_COMMON: variant_provider(gnomad_snv),
                SOURCE_SYNTHETIC_SNV: synthetic_snv_provider,
                SOURCE_CLINVAR: variant_provider(clinvar_snv),
            },
            _SNV_ONLY_EDIT_SOURCE_COUNTS,
        )
    if sub_encoders != _ALL_V1_SUB_ENCODERS:
        raise InputError(
            "real training does not support the configured action.sub_encoders subset",
            details={
                "observed": list(sub_encoders),
                "supported": [["snv"], list(_ALL_V1_SUB_ENCODERS)],
            },
            remediation="use SNV-only or the complete V1 edit surface",
        )
    return (
        {
            SOURCE_GNOMAD_COMMON: variant_provider(gnomad_edits),
            SOURCE_SYNTHETIC_SNV: synthetic_snv_provider,
            SOURCE_SYNTHETIC_INDEL: synthetic_indel_provider,
            SOURCE_CLINVAR: variant_provider(clinvar_edits),
        },
        DEFAULT_EDIT_SOURCE_COUNTS,
    )


def _next_batch(
    iterator: Iterator[TrainingDatasetItem],
    batch_size: int,
) -> tuple[TrainingDatasetItem, ...]:
    items: list[TrainingDatasetItem] = []
    while len(items) < batch_size:
        try:
            items.append(next(iterator))
        except StopIteration as exc:
            raise InputError(
                "training dataset exhausted before requested steps completed",
                details={"batch_size": batch_size, "items_collected": len(items)},
                remediation="provide more source windows or reduce --steps / data.batch_size",
            ) from exc
    return tuple(items)


def _training_device(config: GenoLeWMConfig) -> str:
    return config.runtime.device


def _move_trainable_to_device(component: object, device: str, *, label: str) -> object:
    if device == "cpu":
        return component
    to = getattr(component, "to", None)
    if not callable(to):
        raise RuntimeSetupError(
            "training component does not support accelerator placement",
            details={"component": label, "device": device},
        )
    try:
        moved = to(device)
    except Exception as exc:
        raise RuntimeSetupError(
            "failed to move training component to accelerator",
            details={"component": label, "device": device, "error": str(exc)},
        ) from exc
    return component if moved is None else moved


def _encode_items(
    encoder: CarbonStateEncoder,
    items: Sequence[TrainingDatasetItem],
    *,
    device: str,
) -> Any:
    tuples = tuple(item.training_tuple for item in items)
    source_windows = {item.source_window.window_id: item.source_window.sequence for item in items}
    return encode_training_batch(
        encoder=encoder,
        tuples=tuples,
        source_windows=source_windows,
        device=device,
    )


def _load_dataset_manifest(dataset_dir: Path) -> dict[str, Any]:
    path = dataset_dir / "dataset_manifest.json"
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise InputError("failed to read dataset manifest", details={"path": str(path)}) from exc
    except json.JSONDecodeError as exc:
        raise InputError(
            "dataset manifest JSON is invalid",
            details={"path": str(path), "line": exc.lineno, "column": exc.colno},
        ) from exc
    if not isinstance(payload, dict):
        raise InputError("dataset manifest must be a JSON object")
    return payload


def _dataset_files(manifest: dict[str, Any]) -> tuple[dict[str, Any], ...]:
    raw = manifest.get("files")
    if not isinstance(raw, list) or not raw:
        raise InputError("dataset manifest files must be a non-empty list")
    files: list[dict[str, Any]] = []
    for item in raw:
        if not isinstance(item, dict):
            raise InputError("dataset manifest file entries must be objects")
        files.append(dict(item))
    return tuple(files)


def _load_windows(dataset_dir: Path, files: Sequence[dict[str, Any]]) -> Iterator[WindowContext]:
    placed_paths = _window_jsonl_paths(files, prefix="placed/")
    path_texts = placed_paths or _window_jsonl_paths(files, prefix="carbon/")
    require_chrom = bool(placed_paths)
    for path_text in path_texts:
        path = _safe_dataset_path(dataset_dir, path_text)
        with path.open(encoding="utf-8") as handle:
            for line_no, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                payload = _json_object_line(line, path=path, line_no=line_no)
                chrom = _optional_text(payload, "chrom")
                if require_chrom and chrom is None:
                    raise InputError(
                        "placed training window rows must include chrom",
                        details={"path": str(path), "line": line_no},
                    )
                yield WindowContext(
                    record_id=_required_text(payload, "record_id"),
                    source=_required_text(payload, "source"),
                    sequence=_required_text(payload, "sequence"),
                    start_bp=_optional_int(payload, "start_bp", default=0),
                    chrom=chrom,
                )


def _window_jsonl_paths(files: Sequence[dict[str, Any]], *, prefix: str) -> tuple[str, ...]:
    paths: list[str] = []
    for item in files:
        path_text = _required_text(item, "path")
        if path_text.startswith(prefix) and path_text.endswith(".jsonl"):
            paths.append(path_text)
    return tuple(paths)


def _dataset_fallback_sources(windows: Sequence[WindowContext]) -> dict[str, str]:
    """Return source fallbacks for a release dataset's active window stream."""
    if windows and all(window.chrom is not None for window in windows):
        return {SOURCE_CLINVAR: SOURCE_SYNTHETIC_SNV}
    return dict(DEFAULT_SOURCE_FALLBACKS)


def _load_gnomad_edits(dataset_dir: Path, files: Sequence[dict[str, Any]]) -> Iterator[EditSpec]:
    for path in _variant_shard_paths(dataset_dir, files, prefix="gnomad/"):
        for row in iter_gnomad_shard(path):
            yield EditSpec(row.chrom, row.pos, row.ref, row.alt)


def _load_clinvar_edits(dataset_dir: Path, files: Sequence[dict[str, Any]]) -> Iterator[EditSpec]:
    for path in _variant_shard_paths(dataset_dir, files, prefix="clinvar/"):
        for row in iter_clinvar_shard(path):
            yield EditSpec(row.chrom, row.pos, row.ref, row.alt)


def _variant_shard_paths(
    dataset_dir: Path,
    files: Sequence[dict[str, Any]],
    *,
    prefix: str,
) -> Iterator[Path]:
    for item in files:
        path_text = _required_text(item, "path")
        if path_text.startswith(prefix) and path_text.endswith(".parquet"):
            yield _safe_dataset_path(dataset_dir, path_text)


def _safe_dataset_path(dataset_dir: Path, relative: str) -> Path:
    path = Path(relative)
    if path.is_absolute() or ".." in path.parts:
        raise InputError("dataset manifest paths must stay inside dataset_dir")
    target = (dataset_dir / path).resolve()
    root = dataset_dir.resolve()
    if target != root and root not in target.parents:
        raise InputError("dataset manifest paths must stay inside dataset_dir")
    if not target.is_file():
        raise InputError("dataset manifest file is missing", details={"path": str(target)})
    return target


def _carbon_identity_hash(carbon_model_dir: Path, *, state_contract_version: str) -> str:
    return encoder_identity_hash(
        carbon_model_dir,
        state_contract_version=state_contract_version,
    )


def _json_object_line(line: str, *, path: Path, line_no: int) -> dict[str, Any]:
    try:
        payload = json.loads(line)
    except json.JSONDecodeError as exc:
        raise InputError(
            "Carbon window JSONL row is invalid",
            details={"path": str(path), "line": line_no, "column": exc.colno},
        ) from exc
    if not isinstance(payload, dict):
        raise InputError(
            "Carbon window JSONL rows must be JSON objects",
            details={"path": str(path), "line": line_no},
        )
    return payload


def _write_metrics(
    path: Path,
    *,
    config: GenoLeWMConfig,
    steps: int,
    resumed_from_step: int,
    sample_count: int,
    final_loss: float,
    step_results: Sequence[Any],
    elapsed_seconds: float,
    collapse_alert_count: int,
    dataset_snapshot_id: str,
    resume_checkpoint_path: Path | None,
) -> None:
    new_sample_count = sample_count - (resumed_from_step * config.data.batch_size)
    samples_per_second = new_sample_count / max(elapsed_seconds, 1e-9)
    payload = {
        "schema_version": _SCHEMA_VERSION,
        "run_id": config.run_id,
        "dataset_snapshot_id": dataset_snapshot_id,
        "steps_completed": steps,
        "resumed_from_step": resumed_from_step,
        "resume_checkpoint": None
        if resume_checkpoint_path is None
        else _public_resume_path(resume_checkpoint_path),
        "sample_count": sample_count,
        "new_sample_count": new_sample_count,
        "elapsed_seconds": elapsed_seconds,
        "samples_per_second": samples_per_second,
        "train_loss": final_loss,
        "metrics": {
            "train_loss": final_loss,
            "sample_count": sample_count,
            "new_sample_count": new_sample_count,
            "elapsed_seconds": {"value": elapsed_seconds, "unit": "s"},
            "samples_per_second": {
                "value": samples_per_second,
                "unit": "samples/s",
            },
            "resumed_from_step": resumed_from_step,
            "nan_loss_count": _nan_loss_count(step_results),
            "collapse_var_min": {"value": _collapse_var_min(step_results)},
            "collapse_alert_count": collapse_alert_count,
        },
        "history": [result.to_dict() for result in step_results],
    }
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _nan_loss_count(step_results: Sequence[Any]) -> int:
    """Count optimizer steps whose loss was non-finite (NaN/inf)."""
    return sum(1 for result in step_results if not math.isfinite(float(result.loss)))


def _collapse_var_min(step_results: Sequence[Any]) -> float:
    """Minimum observed prediction variance-per-dim across steps (collapse floor).

    Values near zero indicate representation collapse (training contract). Returns
    ``0.0`` when no steps ran.
    """
    variances = [float(result.pred_var_per_dim) for result in step_results]
    return min(variances) if variances else 0.0


def _last_collapse_alerts(trainer: TorchTrainer) -> tuple[dict[str, object], ...]:
    alerts = getattr(trainer, "last_collapse_alerts", ())
    if not isinstance(alerts, tuple):
        return ()
    return alerts


def _write_checkpoint(
    path: Path,
    *,
    predictor: object,
    action_encoder: object,
    optimizer: object,
    config: GenoLeWMConfig,
    dataset_snapshot_id: str,
    steps: int,
    seeds: TrainerSeeds,
    encoder_identity_hash: str,
) -> None:
    try:
        torch = importlib.import_module("torch")
    except ImportError as exc:  # pragma: no cover - guarded by trainer runtime.
        raise RuntimeSetupError("Carbon training checkpointing requires PyTorch") from exc
    torch.save(
        {
            "schema_version": _SCHEMA_VERSION,
            "run_id": config.run_id,
            "dataset_snapshot_id": dataset_snapshot_id,
            "steps_completed": steps,
            "seeds": seeds.to_dict(),
            "predictor": _state_dict(predictor),
            "action_encoder": _state_dict(action_encoder),
            "optimizer": _state_dict(optimizer),
            "config": {
                "run_id": config.run_id,
                "seed": config.seed,
                "deterministic": config.deterministic,
                "data.batch_size": config.data.batch_size,
                "predictor.d_state": config.predictor.d_state,
                "predictor.dtype": config.predictor.dtype,
                "action.d_action": config.action.d_action,
                "action.sub_encoders": list(config.action.sub_encoders),
                "encoder.normalize": config.encoder.normalize,
                "encoder.state_contract_version": config.encoder.state_contract_version,
                "encoder.effective_normalize": encoder_uses_normalized_states(config.encoder),
                "encoder.identity_hash": encoder_identity_hash,
                "encoder.revision": config.encoder.revision,
                "encoder.dtype": config.encoder.dtype,
                "encoder.state_layer": config.encoder.state_layer,
                "encoder.pool_type": config.encoder.pool_type,
                "encoder.pool_radius": config.encoder.pool_radius,
            },
        },
        path,
    )


def _load_torch_checkpoint(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise InputError("resume checkpoint is missing", details={"path": str(path)})
    try:
        torch = importlib.import_module("torch")
    except ImportError as exc:  # pragma: no cover - guarded by trainer runtime.
        raise RuntimeSetupError("Carbon training resume requires PyTorch") from exc
    try:
        payload = torch.load(path, map_location="cpu")
    except Exception as exc:
        raise InputError(
            "failed to load resume checkpoint",
            details={"path": str(path), "error": str(exc)},
        ) from exc
    if not isinstance(payload, dict):
        raise InputError("resume checkpoint must be a mapping", details={"path": str(path)})
    return payload


def _validate_resume_checkpoint_payload(
    payload: dict[str, Any],
    *,
    path: Path,
    config: GenoLeWMConfig,
    dataset_snapshot_id: str,
    seeds: TrainerSeeds,
    target_steps: int,
    encoder_identity_hash: str,
) -> _ResumeCheckpoint:
    if payload.get("schema_version") != _SCHEMA_VERSION:
        raise InputError(
            "resume checkpoint schema version is unsupported",
            details={
                "path": str(path),
                "expected": _SCHEMA_VERSION,
                "observed": payload.get("schema_version"),
            },
        )
    if payload.get("run_id") != config.run_id:
        raise InputError(
            "resume checkpoint run_id does not match training config",
            details={
                "path": str(path),
                "checkpoint": payload.get("run_id"),
                "config": config.run_id,
            },
        )
    if payload.get("dataset_snapshot_id") != dataset_snapshot_id:
        raise InputError(
            "resume checkpoint dataset_snapshot_id does not match dataset package",
            details={
                "path": str(path),
                "checkpoint": payload.get("dataset_snapshot_id"),
                "dataset": dataset_snapshot_id,
            },
        )
    steps_completed = payload.get("steps_completed")
    if isinstance(steps_completed, bool) or not isinstance(steps_completed, int):
        raise InputError(
            "resume checkpoint steps_completed must be an integer",
            details={"path": str(path), "steps_completed": steps_completed},
        )
    if steps_completed <= 0:
        raise InputError(
            "resume checkpoint must have completed at least one step",
            details={"path": str(path), "steps_completed": steps_completed},
        )
    if steps_completed >= target_steps:
        raise InputError(
            "resume checkpoint is already at or beyond requested --steps",
            details={
                "path": str(path),
                "checkpoint_step": steps_completed,
                "target_steps": target_steps,
            },
        )
    checkpoint_seeds = payload.get("seeds")
    if checkpoint_seeds != seeds.to_dict():
        raise InputError(
            "resume checkpoint seed split does not match training config",
            details={"path": str(path)},
        )
    checkpoint_config = payload.get("config")
    if not isinstance(checkpoint_config, dict):
        raise InputError("resume checkpoint config must be a mapping", details={"path": str(path)})
    _validate_resume_state_contract(
        checkpoint_config,
        path=path,
        config=config,
        encoder_identity_hash=encoder_identity_hash,
    )
    expected_config = {
        "run_id": config.run_id,
        "seed": config.seed,
        "deterministic": config.deterministic,
        "data.batch_size": config.data.batch_size,
        "predictor.d_state": config.predictor.d_state,
        "predictor.dtype": config.predictor.dtype,
        "action.d_action": config.action.d_action,
        "action.sub_encoders": list(config.action.sub_encoders),
    }
    for key, expected in expected_config.items():
        if checkpoint_config.get(key) != expected:
            raise InputError(
                "resume checkpoint config does not match training config",
                details={
                    "path": str(path),
                    "field": key,
                    "checkpoint": checkpoint_config.get(key),
                    "config": expected,
                },
            )
    for key in ("predictor", "action_encoder", "optimizer"):
        if key not in payload:
            raise InputError(
                "resume checkpoint is missing trainer state",
                details={"path": str(path), "state": key},
            )
    return _ResumeCheckpoint(path=path, steps_completed=steps_completed, payload=payload)


def _validate_resume_state_contract(
    checkpoint_config: dict[str, Any],
    *,
    path: Path,
    config: GenoLeWMConfig,
    encoder_identity_hash: str,
) -> None:
    keys = {
        "encoder.normalize",
        "encoder.state_contract_version",
        "encoder.effective_normalize",
    }
    present = keys.intersection(checkpoint_config)
    if present != keys:
        raise InputError(
            "resume checkpoint is missing a complete encoder state contract",
            details={"path": str(path), "present": sorted(present), "required": sorted(keys)},
        )

    expected_version = config.encoder.state_contract_version
    expected_effective = encoder_uses_normalized_states(config.encoder)
    observed_version = checkpoint_config["encoder.state_contract_version"]
    observed_effective = checkpoint_config["encoder.effective_normalize"]
    observed_configured = checkpoint_config["encoder.normalize"]
    if not isinstance(observed_configured, bool) or not isinstance(observed_effective, bool):
        raise InputError(
            "resume checkpoint encoder normalization fields must be boolean",
            details={"path": str(path)},
        )
    if observed_configured != config.encoder.normalize:
        raise InputError(
            "resume checkpoint config does not match training config",
            details={
                "path": str(path),
                "field": "encoder.normalize",
                "checkpoint": observed_configured,
                "config": config.encoder.normalize,
            },
        )

    if observed_version != expected_version or observed_effective != expected_effective:
        raise InputError(
            "resume checkpoint encoder state contract does not match training config",
            details={
                "path": str(path),
                "checkpoint_version": observed_version,
                "checkpoint_effective_normalize": observed_effective,
                "config_version": expected_version,
                "config_effective_normalize": expected_effective,
            },
        )

    identity = {
        "encoder.identity_hash": encoder_identity_hash,
        "encoder.revision": config.encoder.revision,
        "encoder.dtype": config.encoder.dtype,
        "encoder.state_layer": config.encoder.state_layer,
        "encoder.pool_type": config.encoder.pool_type,
        "encoder.pool_radius": config.encoder.pool_radius,
    }
    identity_keys = set(identity)
    identity_present = identity_keys.intersection(checkpoint_config)
    if identity_present != identity_keys:
        raise InputError(
            "resume checkpoint is missing a complete encoder representation identity",
            details={
                "path": str(path),
                "present": sorted(identity_present),
                "required": sorted(identity_keys),
            },
        )
    for field, expected in identity.items():
        observed = checkpoint_config.get(field)
        if observed != expected:
            raise InputError(
                "resume checkpoint encoder representation does not match training config",
                details={
                    "path": str(path),
                    "field": field,
                    "checkpoint": observed,
                    "config": expected,
                },
            )


def _restore_resume_checkpoint(
    payload: dict[str, Any],
    *,
    predictor: object,
    action_encoder: object,
    optimizer: object,
) -> None:
    _load_state_dict(predictor, payload["predictor"], "predictor")
    _load_state_dict(action_encoder, payload["action_encoder"], "action_encoder")
    _load_state_dict(optimizer, payload["optimizer"], "optimizer")


def _load_state_dict(target: object, state: object, label: str) -> None:
    load_state_dict = getattr(target, "load_state_dict", None)
    if not callable(load_state_dict):
        raise InputError(f"{label} does not expose load_state_dict")
    try:
        load_state_dict(state)
    except Exception as exc:
        raise InputError(
            "failed to restore resume checkpoint state",
            details={"component": label, "error": str(exc)},
        ) from exc


def _write_training_metadata(
    path: Path,
    *,
    config: GenoLeWMConfig,
    command: str,
    commit_sha: str,
    package_version: str,
    dataset_snapshot_id: str,
    seeds: TrainerSeeds,
    determinism: dict[str, object],
    artifacts: dict[str, object],
    preflight_report: TrainingPreflightReport | None,
    final_loss: float,
    sample_count: int,
    resumed_from_step: int,
    resume_checkpoint_path: Path | None,
) -> None:
    artifact_identities = _training_artifact_identities(path.parent, artifacts)
    payload = {
        "schema_version": _SCHEMA_VERSION,
        "run_id": config.run_id,
        "generated_by": _TRAINING_RUN_PACKAGE_GENERATED_BY,
        "command": command,
        "commit_sha": commit_sha,
        "package_version": package_version,
        "dataset_snapshot_id": dataset_snapshot_id,
        "dataset_manifest": artifacts["dataset_manifest"],
        "training_config": artifacts["training_config"],
        "metrics": artifacts["metrics"],
        "logs": artifacts["logs"],
        "checkpoint_files": artifacts["checkpoint_files"],
        "artifact_identities": artifact_identities,
        "status": "completed",
        "hardware": _hardware_notes(),
        "runtime": _runtime_notes(preflight_report),
        "seeds": {"base": config.seed, **seeds.to_dict()},
        "determinism": json.dumps(determinism, sort_keys=True),
        "monitoring": {
            "collapse_monitoring": True,
            "collapse_log_every_steps": config.training.collapse_log_every_steps,
            "nan_monitoring": True,
        },
        "resumed_from_step": resumed_from_step,
        "resume_checkpoint": None
        if resume_checkpoint_path is None
        else _public_resume_path(resume_checkpoint_path),
        "result_summary": (
            f"Completed Carbon-backed training launch for {sample_count} samples"
            f"{_resume_summary_suffix(resumed_from_step)}; "
            f"final training loss {final_loss:.6g}."
        ),
        "limitations": [
            "This run supports research iteration only until the paired evaluation report is published.",
            "Model-quality claims require the release evaluation report and terminal demo evidence.",
        ],
    }
    if preflight_report is not None:
        payload["training_preflight_report"] = REPORT_NAME
        artifact_identities["training_preflight_report"] = _file_identity(
            path.parent / REPORT_NAME,
            label=REPORT_NAME,
        )
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _training_artifact_identities(
    root: Path,
    artifacts: Mapping[str, object],
) -> dict[str, object]:
    dataset_manifest = _artifact_name(artifacts, "dataset_manifest")
    training_config = _artifact_name(artifacts, "training_config")
    metrics = _artifact_name(artifacts, "metrics")
    return {
        "dataset_manifest": _file_identity(root / dataset_manifest, label=dataset_manifest),
        "training_config": _file_identity(root / training_config, label=training_config),
        "metrics": _file_identity(root / metrics, label=metrics),
        "logs": [
            _file_identity(root / log_name, label=log_name)
            for log_name in _artifact_names(artifacts, "logs")
        ],
        "checkpoint_files": [
            _file_identity(root / checkpoint_name, label=checkpoint_name)
            for checkpoint_name in _artifact_names(artifacts, "checkpoint_files")
        ],
    }


def _file_identity(path: Path, *, label: str) -> dict[str, object]:
    return {
        "path": label,
        "sha256": sha256_file(path),
        "size_bytes": path.stat().st_size,
    }


def _artifact_name(artifacts: Mapping[str, object], key: str) -> str:
    value = artifacts.get(key)
    if not isinstance(value, str) or not value:
        raise InputError(
            "training metadata artifact must be a non-empty string",
            details={"key": key, "type": type(value).__name__},
        )
    return value


def _artifact_names(artifacts: Mapping[str, object], key: str) -> tuple[str, ...]:
    value = artifacts.get(key)
    if (
        not isinstance(value, Sequence)
        or isinstance(value, str | bytes)
        or any(not isinstance(item, str) or not item for item in value)
    ):
        raise InputError(
            "training metadata artifact list must contain non-empty strings",
            details={"key": key, "type": type(value).__name__},
        )
    return tuple(value)


def _resume_summary_suffix(resumed_from_step: int) -> str:
    return "" if resumed_from_step == 0 else f" resumed from step {resumed_from_step}"


def _public_resume_path(path: Path) -> str:
    if path.is_absolute():
        return path.name
    if ".." in path.parts or not path.parts:
        return path.name
    return path.as_posix()


def _state_dict(value: object) -> object:
    state_dict = getattr(value, "state_dict", None)
    if not callable(state_dict):
        raise InputError("training component does not expose state_dict")
    return state_dict()


def _hardware_notes() -> list[str]:
    return [f"{platform.system()} {platform.machine()}", f"Python {platform.python_version()}"]


def _runtime_notes(preflight_report: TrainingPreflightReport | None) -> list[str]:
    notes = ["GenoLeWM Carbon-backed torch trainer."]
    if preflight_report is not None:
        notes.extend(
            f"{probe.import_name}=={probe.version or 'unavailable'}"
            for probe in preflight_report.dependencies
        )
    return notes


def _required_text(payload: dict[str, Any], key: str) -> str:
    value = payload.get(key)
    if not isinstance(value, str) or not value:
        raise InputError(f"{key} must be a non-empty string")
    return value


def _optional_text(payload: dict[str, Any], key: str) -> str | None:
    value = payload.get(key)
    if value is None:
        return None
    if not isinstance(value, str) or not value:
        raise InputError(f"{key} must be a non-empty string when present")
    return value


def _optional_int(payload: dict[str, Any], key: str, *, default: int) -> int:
    value = payload.get(key, default)
    if isinstance(value, bool) or not isinstance(value, int):
        raise InputError(f"{key} must be an integer")
    return value


def _require_positive_int(name: str, value: int) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise InputError(
            f"{name} must be a positive integer",
            details={"field": name, "value": value, "type": type(value).__name__},
        )
