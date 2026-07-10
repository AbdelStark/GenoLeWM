"""Tests for the real Carbon-backed training launcher boundary."""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import pytest

import geno_lewm.training.real as real_module
from geno_lewm.action import EditSpec
from geno_lewm.config import load_config
from geno_lewm.config._state_contract import encoder_uses_normalized_states
from geno_lewm.data import (
    SOURCE_CLINVAR,
    SOURCE_GNOMAD_COMMON,
    SOURCE_SYNTHETIC_INDEL,
    SOURCE_SYNTHETIC_SNV,
    WindowContext,
    synthetic_indel_provider,
    synthetic_snv_provider,
)
from geno_lewm.errors import InputError, RuntimeSetupError
from geno_lewm.provenance import sha256_file
from geno_lewm.training.preflight import REPORT_NAME, AcceleratorProbe, TrainingPreflightReport
from geno_lewm.training.real import (
    _collapse_var_min,
    _dataset_fallback_sources,
    _dataset_files,
    _load_dataset_manifest,
    _load_windows,
    _move_trainable_to_device,
    _nan_loss_count,
    _next_batch,
    _repeat_training_items,
    _training_device,
    _validate_resume_checkpoint_payload,
    _write_checkpoint,
    _write_metrics,
    _write_training_metadata,
)
from geno_lewm.training.trainer import TorchTrainerStepResult, TrainerSeeds
from tests.unit.test_training_preflight import _write_release_dataset, _write_training_config

_ENCODER_WEIGHTS_HASH = "sha256:" + ("a" * 64)


def _step_result(*, loss: float, var: float, step: int = 1) -> TorchTrainerStepResult:
    return TorchTrainerStepResult(
        step=step,
        lr_multiplier=1.0,
        loss=loss,
        pred_loss=loss,
        kl_reg=0.0,
        action_count=1,
        pred_var_per_dim=var,
    )


def test_nan_loss_count_counts_nonfinite_losses() -> None:
    results = [
        _step_result(loss=0.5, var=1.0),
        _step_result(loss=float("nan"), var=1.0),
        _step_result(loss=float("inf"), var=1.0),
        _step_result(loss=0.3, var=1.0),
    ]
    assert _nan_loss_count(results) == 2


def test_run_carbon_training_rejects_phase2_before_runtime_setup(tmp_path: Path) -> None:
    config = load_config(_write_training_config(tmp_path))
    phase2_config = replace(config, phase="phase2")

    with pytest.raises(RuntimeSetupError, match="graph-preserving trainable encoder-adapter"):
        real_module.run_carbon_training(
            config=phase2_config,
            dataset_dir=tmp_path / "missing-dataset",
            carbon_model_dir=tmp_path / "missing-carbon",
            run_dir=tmp_path / "run",
            steps=1,
            command="geno-lewm-train --carbon-train",
            commit_sha="a" * 40,
            package_version="0.2.1",
        )


def test_collapse_var_min_returns_minimum_and_handles_empty() -> None:
    results = [
        _step_result(loss=0.5, var=0.8),
        _step_result(loss=0.4, var=0.2),
        _step_result(loss=0.3, var=0.9),
    ]
    assert _collapse_var_min(results) == pytest.approx(0.2)
    assert _collapse_var_min([]) == 0.0


def test_write_metrics_emits_real_nan_and_collapse_floor(tmp_path: Path) -> None:
    config = load_config(_write_training_config(tmp_path))
    results = [
        _step_result(loss=0.5, var=0.8, step=1),
        TorchTrainerStepResult(
            step=2,
            lr_multiplier=1.0,
            loss=float("nan"),
            pred_loss=float("nan"),
            kl_reg=0.0,
            action_count=1,
            pred_var_per_dim=0.2,
        ),
        _step_result(loss=0.3, var=0.9, step=3),
    ]
    path = tmp_path / "metrics.json"
    _write_metrics(
        path,
        config=config,
        steps=3,
        resumed_from_step=0,
        sample_count=24,
        final_loss=0.3,
        step_results=results,
        elapsed_seconds=3.0,
        collapse_alert_count=1,
        dataset_snapshot_id="geno-lewm-data-v0.1.0-r1",
        resume_checkpoint_path=None,
    )
    payload = json.loads(path.read_text(encoding="utf-8"))
    metrics = payload["metrics"]
    assert metrics["nan_loss_count"] == 1
    assert metrics["collapse_var_min"]["value"] == pytest.approx(0.2)
    assert metrics["collapse_alert_count"] == 1
    assert payload["elapsed_seconds"] == pytest.approx(3.0)
    assert payload["samples_per_second"] == pytest.approx(8.0)
    assert metrics["elapsed_seconds"]["value"] == pytest.approx(3.0)
    assert metrics["samples_per_second"]["value"] == pytest.approx(8.0)


def test_write_training_metadata_records_artifact_identities(tmp_path: Path) -> None:
    config = load_config(_write_training_config(tmp_path))
    seeds = TrainerSeeds.from_base_seed(config.seed)
    artifacts = {
        "dataset_manifest": "dataset_manifest.json",
        "training_config": "training_config.effective.yaml",
        "metrics": "metrics.json",
        "logs": ["train.log"],
        "checkpoint_files": ["predictor_checkpoint.pt"],
    }
    for name in (
        "dataset_manifest.json",
        "training_config.effective.yaml",
        "metrics.json",
        "train.log",
        "predictor_checkpoint.pt",
        REPORT_NAME,
    ):
        (tmp_path / name).write_text(f"{name}\n", encoding="utf-8")

    metadata_path = tmp_path / "training_run.json"
    _write_training_metadata(
        metadata_path,
        config=config,
        command="geno-lewm-train --carbon-train",
        commit_sha="abcdef123456",
        package_version="0.1.0.dev0",
        dataset_snapshot_id="geno-lewm-data-v0.1.0-r1",
        seeds=seeds,
        determinism={"deterministic": True},
        artifacts=artifacts,
        preflight_report=_preflight_report(),
        final_loss=0.5,
        sample_count=4,
        resumed_from_step=0,
        resume_checkpoint_path=None,
    )

    identities = json.loads(metadata_path.read_text(encoding="utf-8"))["artifact_identities"]
    assert identities["dataset_manifest"] == _identity(tmp_path, "dataset_manifest.json")
    assert identities["training_config"] == _identity(tmp_path, "training_config.effective.yaml")
    assert identities["metrics"] == _identity(tmp_path, "metrics.json")
    assert identities["logs"] == [_identity(tmp_path, "train.log")]
    assert identities["checkpoint_files"] == [_identity(tmp_path, "predictor_checkpoint.pt")]
    assert identities["training_preflight_report"] == _identity(tmp_path, REPORT_NAME)


def test_training_device_uses_runtime_config_device(tmp_path: Path) -> None:
    config = load_config(_write_training_config(tmp_path))

    assert _training_device(config) == "cuda"


def test_release_training_loader_prefers_placed_windows(tmp_path: Path) -> None:
    pytest.importorskip("pyarrow")
    dataset_dir = _write_release_dataset(tmp_path)
    files = _dataset_files(_load_dataset_manifest(dataset_dir))

    windows = tuple(_load_windows(dataset_dir, files))

    assert len(windows) == 1
    assert windows[0].chrom == "1"
    assert windows[0].source == "gnomad_common"
    assert _dataset_fallback_sources(windows) == {"clinvar": "synthetic_snv"}


def test_repeat_training_items_cycles_finite_release_snapshot() -> None:
    windows = (
        WindowContext(
            record_id="placed-1",
            source="gnomad_common",
            sequence="ACGT" * 64,
            chrom="1",
            start_bp=100,
        ),
    )
    providers = {
        SOURCE_GNOMAD_COMMON: synthetic_snv_provider,
        SOURCE_SYNTHETIC_SNV: synthetic_snv_provider,
        SOURCE_SYNTHETIC_INDEL: synthetic_indel_provider,
        SOURCE_CLINVAR: lambda window, count, rng: (),
    }

    iterator = _repeat_training_items(
        windows,
        providers,
        seed=11,
        fallback_sources=_dataset_fallback_sources(windows),
    )

    first_epoch = _next_batch(iterator, 8)
    next_epoch = _next_batch(iterator, 1)

    assert {item.source_window.record_id for item in first_epoch} == {"placed-1"}
    assert next_epoch[0].source_window.record_id == "placed-1"


def test_repeat_training_items_rejects_empty_epoch() -> None:
    windows: tuple[WindowContext, ...] = ()
    providers = {
        SOURCE_GNOMAD_COMMON: lambda window, count, rng: (),
        SOURCE_SYNTHETIC_SNV: lambda window, count, rng: (),
        SOURCE_SYNTHETIC_INDEL: lambda window, count, rng: (),
        SOURCE_CLINVAR: lambda window, count, rng: (),
    }

    iterator = _repeat_training_items(
        windows,
        providers,
        seed=11,
        fallback_sources={},
    )

    with pytest.raises(InputError, match="epoch produced no usable tuples"):
        next(iterator)


def test_move_trainable_to_device_invokes_module_to_for_accelerator() -> None:
    module = _DeviceModule()

    moved = _move_trainable_to_device(module, "cuda", label="predictor")

    assert moved is module
    assert module.devices == ["cuda"]


def test_move_trainable_to_device_leaves_cpu_module_in_place() -> None:
    module = _DeviceModule()

    moved = _move_trainable_to_device(module, "cpu", label="predictor")

    assert moved is module
    assert module.devices == []


def test_carbon_training_resolves_contract_aware_encoder_identity_before_runtime_setup(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = load_config(_write_training_config(tmp_path))
    dataset_dir = tmp_path / "dataset"
    dataset_dir.mkdir()
    (dataset_dir / "dataset_manifest.json").write_text("{}\n", encoding="utf-8")
    carbon_dir = tmp_path / "carbon"
    carbon_dir.mkdir()
    window = WindowContext(
        record_id="placed-1",
        source=SOURCE_GNOMAD_COMMON,
        sequence="ACGT" * 16,
        chrom="1",
    )
    observed: list[tuple[Path, str]] = []

    monkeypatch.setattr(
        real_module,
        "_load_dataset_manifest",
        lambda _dataset_dir: {"snapshot_id": "fixture", "files": []},
    )
    monkeypatch.setattr(real_module, "_dataset_files", lambda _manifest: ())
    monkeypatch.setattr(real_module, "_load_windows", lambda *_args: iter((window,)))
    monkeypatch.setattr(
        real_module,
        "_load_gnomad_edits",
        lambda *_args: iter((EditSpec(chrom="1", pos=1, ref="A", alt="T"),)),
    )
    monkeypatch.setattr(real_module, "_load_clinvar_edits", lambda *_args: iter(()))

    def fake_encoder_identity_hash(model_dir: Path, *, state_contract_version: str) -> str:
        observed.append((model_dir, state_contract_version))
        return _ENCODER_WEIGHTS_HASH

    def stop_after_identity(_config: object) -> str:
        raise RuntimeError("runtime setup reached")

    monkeypatch.setattr(real_module, "encoder_identity_hash", fake_encoder_identity_hash)
    monkeypatch.setattr(real_module, "_training_device", stop_after_identity)

    with pytest.raises(RuntimeError, match="runtime setup reached"):
        real_module.run_carbon_training(
            config=config,
            dataset_dir=dataset_dir,
            carbon_model_dir=carbon_dir,
            run_dir=tmp_path / "run",
            steps=1,
            command="geno-lewm-train --carbon-train",
            commit_sha="abcdef123456",
            package_version="0.1.0.dev0",
        )

    assert observed == [(carbon_dir, config.encoder.state_contract_version)]


def test_validate_resume_checkpoint_accepts_matching_identity(tmp_path: Path) -> None:
    config = load_config(_write_training_config(tmp_path))
    seeds = TrainerSeeds.from_base_seed(config.seed)

    checkpoint = _validate_resume_checkpoint_payload(
        _resume_payload(config, seeds, steps_completed=3),
        path=tmp_path / "predictor_checkpoint.pt",
        config=config,
        dataset_snapshot_id="geno-lewm-data-v0.1.0-r1",
        seeds=seeds,
        target_steps=5,
        encoder_identity_hash=_ENCODER_WEIGHTS_HASH,
    )

    assert checkpoint.steps_completed == 3


@pytest.mark.parametrize(
    ("field", "value", "delete", "message"),
    [
        ("encoder.effective_normalize", "true", False, "must be boolean"),
        ("encoder.normalize", False, False, "config does not match"),
        (
            "encoder.identity_hash",
            None,
            True,
            "missing a complete encoder representation identity",
        ),
    ],
)
def test_validate_resume_checkpoint_rejects_malformed_encoder_contract(
    tmp_path: Path,
    field: str,
    value: object,
    delete: bool,
    message: str,
) -> None:
    config = load_config(_write_training_config(tmp_path))
    seeds = TrainerSeeds.from_base_seed(config.seed)
    payload = _resume_payload(config, seeds, steps_completed=3)
    checkpoint_config = payload["config"]
    assert isinstance(checkpoint_config, dict)
    if delete:
        del checkpoint_config[field]
    else:
        checkpoint_config[field] = value

    with pytest.raises(InputError, match=message):
        _validate_resume_checkpoint_payload(
            payload,
            path=tmp_path / "predictor_checkpoint.pt",
            config=config,
            dataset_snapshot_id="geno-lewm-data-v0.1.0-r1",
            seeds=seeds,
            target_steps=5,
            encoder_identity_hash=_ENCODER_WEIGHTS_HASH,
        )


def test_validate_resume_checkpoint_rejects_legacy_payload_for_normalized_lineage(
    tmp_path: Path,
) -> None:
    legacy_config = load_config(_write_training_config(tmp_path))
    config = replace(
        legacy_config,
        encoder=replace(
            legacy_config.encoder,
            state_contract_version="l2_normalized_v2",
        ),
    )
    seeds = TrainerSeeds.from_base_seed(config.seed)

    payload = _resume_payload(config, seeds, steps_completed=3)
    payload_config = payload["config"]
    assert isinstance(payload_config, dict)
    payload_config["encoder.state_contract_version"] = "legacy_raw_v1"
    payload_config["encoder.effective_normalize"] = False

    with pytest.raises(InputError, match="state contract does not match"):
        _validate_resume_checkpoint_payload(
            payload,
            path=tmp_path / "predictor_checkpoint.pt",
            config=config,
            dataset_snapshot_id="geno-lewm-data-v0.1.0-r1",
            seeds=seeds,
            target_steps=5,
            encoder_identity_hash=_ENCODER_WEIGHTS_HASH,
        )


def test_legacy_resume_rejects_missing_encoder_identity(tmp_path: Path) -> None:
    config = load_config(_write_training_config(tmp_path))
    seeds = TrainerSeeds.from_base_seed(config.seed)
    payload = _resume_payload(config, seeds, steps_completed=3)
    payload_config = payload["config"]
    assert isinstance(payload_config, dict)
    for key in tuple(payload_config):
        if key.startswith("encoder."):
            del payload_config[key]

    with pytest.raises(InputError, match="complete encoder state contract"):
        _validate_resume_checkpoint_payload(
            payload,
            path=tmp_path / "predictor_checkpoint.pt",
            config=config,
            dataset_snapshot_id="geno-lewm-data-v0.1.0-r1",
            seeds=seeds,
            target_steps=5,
            encoder_identity_hash=_ENCODER_WEIGHTS_HASH,
        )


def test_validate_resume_checkpoint_rejects_dataset_mismatch(tmp_path: Path) -> None:
    config = load_config(_write_training_config(tmp_path))
    seeds = TrainerSeeds.from_base_seed(config.seed)
    payload = _resume_payload(config, seeds, steps_completed=3)
    payload["dataset_snapshot_id"] = "geno-lewm-data-old"

    with pytest.raises(InputError, match="dataset_snapshot_id"):
        _validate_resume_checkpoint_payload(
            payload,
            path=tmp_path / "predictor_checkpoint.pt",
            config=config,
            dataset_snapshot_id="geno-lewm-data-v0.1.0-r1",
            seeds=seeds,
            target_steps=5,
            encoder_identity_hash=_ENCODER_WEIGHTS_HASH,
        )


def test_validate_resume_checkpoint_rejects_config_mismatch(tmp_path: Path) -> None:
    config = load_config(_write_training_config(tmp_path))
    seeds = TrainerSeeds.from_base_seed(config.seed)
    payload = _resume_payload(config, seeds, steps_completed=3)
    payload["config"]["data.batch_size"] = config.data.batch_size + 1

    with pytest.raises(InputError, match="config does not match"):
        _validate_resume_checkpoint_payload(
            payload,
            path=tmp_path / "predictor_checkpoint.pt",
            config=config,
            dataset_snapshot_id="geno-lewm-data-v0.1.0-r1",
            seeds=seeds,
            target_steps=5,
            encoder_identity_hash=_ENCODER_WEIGHTS_HASH,
        )


def test_normalized_resume_rejects_encoder_identity_mismatch(tmp_path: Path) -> None:
    legacy_config = load_config(_write_training_config(tmp_path))
    config = replace(
        legacy_config,
        encoder=replace(
            legacy_config.encoder,
            state_contract_version="l2_normalized_v2",
        ),
    )
    seeds = TrainerSeeds.from_base_seed(config.seed)
    payload = _resume_payload(config, seeds, steps_completed=3)
    payload_config = payload["config"]
    assert isinstance(payload_config, dict)
    payload_config.update(
        {
            "encoder.normalize": True,
            "encoder.state_contract_version": "l2_normalized_v2",
            "encoder.effective_normalize": True,
            "encoder.identity_hash": "sha256:" + ("b" * 64),
            "encoder.revision": config.encoder.revision,
            "encoder.dtype": config.encoder.dtype,
            "encoder.state_layer": config.encoder.state_layer,
            "encoder.pool_type": config.encoder.pool_type,
            "encoder.pool_radius": config.encoder.pool_radius,
        }
    )

    with pytest.raises(InputError, match="encoder representation does not match"):
        _validate_resume_checkpoint_payload(
            payload,
            path=tmp_path / "predictor_checkpoint.pt",
            config=config,
            dataset_snapshot_id="geno-lewm-data-v0.1.0-r1",
            seeds=seeds,
            target_steps=5,
            encoder_identity_hash=_ENCODER_WEIGHTS_HASH,
        )


def test_validate_resume_checkpoint_rejects_finished_checkpoint(tmp_path: Path) -> None:
    config = load_config(_write_training_config(tmp_path))
    seeds = TrainerSeeds.from_base_seed(config.seed)

    with pytest.raises(InputError, match="already at or beyond"):
        _validate_resume_checkpoint_payload(
            _resume_payload(config, seeds, steps_completed=5),
            path=tmp_path / "predictor_checkpoint.pt",
            config=config,
            dataset_snapshot_id="geno-lewm-data-v0.1.0-r1",
            seeds=seeds,
            target_steps=5,
            encoder_identity_hash=_ENCODER_WEIGHTS_HASH,
        )


def test_write_checkpoint_emits_resume_compatible_payload(tmp_path: Path) -> None:
    torch = pytest.importorskip("torch")
    config = load_config(_write_training_config(tmp_path))
    seeds = TrainerSeeds.from_base_seed(config.seed)
    predictor = torch.nn.Linear(2, 2)
    action_encoder = torch.nn.Linear(2, 2)
    optimizer = torch.optim.AdamW(list(predictor.parameters()) + list(action_encoder.parameters()))
    path = tmp_path / "predictor_checkpoint.pt"

    _write_checkpoint(
        path,
        predictor=predictor,
        action_encoder=action_encoder,
        optimizer=optimizer,
        config=config,
        dataset_snapshot_id="geno-lewm-data-v0.1.0-r1",
        steps=7,
        seeds=seeds,
        encoder_identity_hash=_ENCODER_WEIGHTS_HASH,
    )

    payload = torch.load(path, map_location="cpu")
    checkpoint = _validate_resume_checkpoint_payload(
        payload,
        path=path,
        config=config,
        dataset_snapshot_id="geno-lewm-data-v0.1.0-r1",
        seeds=seeds,
        target_steps=8,
        encoder_identity_hash=_ENCODER_WEIGHTS_HASH,
    )

    assert checkpoint.steps_completed == 7
    assert payload["predictor"]
    assert payload["action_encoder"]
    assert payload["optimizer"]
    assert payload["config"]["encoder.normalize"] is True
    assert payload["config"]["encoder.state_contract_version"] == "legacy_raw_v1"
    assert payload["config"]["encoder.effective_normalize"] is False
    assert payload["config"]["encoder.identity_hash"] == _ENCODER_WEIGHTS_HASH
    assert payload["config"]["encoder.state_layer"] == config.encoder.state_layer


def _resume_payload(config, seeds: TrainerSeeds, *, steps_completed: int) -> dict[str, object]:
    return {
        "schema_version": "1.0.0",
        "run_id": config.run_id,
        "dataset_snapshot_id": "geno-lewm-data-v0.1.0-r1",
        "steps_completed": steps_completed,
        "seeds": seeds.to_dict(),
        "config": {
            "run_id": config.run_id,
            "seed": config.seed,
            "deterministic": config.deterministic,
            "data.batch_size": config.data.batch_size,
            "predictor.d_state": config.predictor.d_state,
            "action.d_action": config.action.d_action,
            "encoder.normalize": config.encoder.normalize,
            "encoder.state_contract_version": config.encoder.state_contract_version,
            "encoder.effective_normalize": encoder_uses_normalized_states(config.encoder),
            "encoder.identity_hash": _ENCODER_WEIGHTS_HASH,
            "encoder.revision": config.encoder.revision,
            "encoder.dtype": config.encoder.dtype,
            "encoder.state_layer": config.encoder.state_layer,
            "encoder.pool_type": config.encoder.pool_type,
            "encoder.pool_radius": config.encoder.pool_radius,
        },
        "predictor": {},
        "action_encoder": {},
        "optimizer": {},
    }


class _DeviceModule:
    def __init__(self) -> None:
        self.devices: list[str] = []

    def to(self, device: str) -> _DeviceModule:
        self.devices.append(device)
        return self


def _identity(root: Path, name: str) -> dict[str, object]:
    path = root / name
    return {
        "path": name,
        "sha256": sha256_file(path),
        "size_bytes": path.stat().st_size,
    }


def _preflight_report() -> TrainingPreflightReport:
    return TrainingPreflightReport(
        schema_version="1.0.0",
        generated_by="test",
        generated_at="2026-06-11T00:00:00Z",
        ok=True,
        dataset_snapshot_id="geno-lewm-data-v0.1.0-r1",
        training_config={},
        run_dir={},
        dataset={},
        carbon={},
        accelerator=AcceleratorProbe(
            requested_device=None,
            required=False,
            available=True,
            device_count=0,
            device_name=None,
            total_memory_bytes=None,
            min_memory_bytes=0,
            reason="not required",
        ),
        dependencies=(),
        issues=(),
    )
