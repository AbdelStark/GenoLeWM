"""Tests for the real Carbon-backed training launcher boundary."""

from __future__ import annotations

import json
import random
import threading
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest

import geno_lewm._atomic as atomic_module
import geno_lewm.training.real as real_module
from geno_lewm._source_provenance import SourceProvenance
from geno_lewm.action import EditSpec, EditType, RelEdit
from geno_lewm.config import config_to_dict, load_config
from geno_lewm.data import (
    MEMBERSHIP_STORE_SCHEMA_VERSION,
    SOURCE_CLINVAR,
    SOURCE_GNOMAD_COMMON,
    SOURCE_SYNTHETIC_INDEL,
    SOURCE_SYNTHETIC_SNV,
    HoldoutPolicy,
    WindowContext,
    synthetic_indel_provider,
    synthetic_snv_provider,
)
from geno_lewm.errors import InputError, RuntimeSetupError
from geno_lewm.provenance import canonical_json_sha256, sha256_file
from geno_lewm.training._data_stream import PreparedTrainingStream
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
    _skip_training_items,
    _training_device,
    _training_edit_contract,
    _validate_resume_checkpoint_payload,
    _write_checkpoint,
    _write_metrics,
    _write_training_metadata,
)
from geno_lewm.training.resume import (
    CHECKPOINT_SCHEMA_VERSION,
    capture_rng_state,
    load_resume_checkpoint,
    write_resume_checkpoint,
)
from geno_lewm.training.trainer import (
    TorchTrainerBatch,
    TorchTrainerStepResult,
    TrainerSeeds,
    wsd_lr_multiplier,
)
from tests.unit.test_training_preflight import _write_release_dataset, _write_training_config

_ENCODER_WEIGHTS_HASH = "sha256:" + ("a" * 64)
_MEMBERSHIP_BINDING = {
    "path": "evidence/membership-store",
    "artifact_id": "geno-lewm-v03-membership-fixture-r1",
    "content_identity": "sha256:" + ("b" * 64),
    "physical_identity": "sha256:" + ("c" * 64),
    "rowset_sha256": "sha256:" + ("d" * 64),
}
_REPORT_BINDING = {
    "path": "evidence/membership-split-evidence.json",
    "schema_path": "contract/membership-split-evidence.schema.json",
    "artifact_id": "geno-lewm-v03-membership-splits-fixture-r1",
    "schema_version": "geno-lewm.membership-split-evidence.v1",
}

requires_secure_atomic_publication = pytest.mark.skipif(
    not atomic_module.supports_secure_atomic_publication(),
    reason=(
        "secure production checkpoint/report publication requires POSIX "
        "anchored directory operations"
    ),
)


@pytest.fixture(autouse=True)
def _stable_package_source(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        real_module,
        "resolve_package_source",
        lambda **_kwargs: SourceProvenance(
            commit_sha="a" * 40,
            tree_sha="b" * 40,
            package_version="0.2.1",
        ),
    )


def _membership_dataset_binding() -> dict[str, object]:
    return {
        "membership_store": dict(_MEMBERSHIP_BINDING),
        "report": dict(_REPORT_BINDING),
    }


def _membership_runtime_binding() -> dict[str, object]:
    holdout_policy = {
        "schema_version": MEMBERSHIP_STORE_SCHEMA_VERSION,
        "membership_content_identity": _MEMBERSHIP_BINDING["content_identity"],
        "excluded_chromosomes": ["20", "21"],
        "selection": "chromosome_roles",
        "lookup": "lookup.sqlite",
    }
    return {
        **_membership_dataset_binding(),
        "holdout_policy": holdout_policy,
        "holdout_policy_identity": canonical_json_sha256(holdout_policy),
    }


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
            source_tree="b" * 40,
            package_version="0.2.1",
        )


def test_run_carbon_training_rejects_unresolved_source_sentinel_before_writes(
    tmp_path: Path,
) -> None:
    config = load_config(_write_training_config(tmp_path))
    run_dir = tmp_path / "run"

    with pytest.raises(InputError, match="full lowercase 40-character Git SHA"):
        real_module.run_carbon_training(
            config=config,
            dataset_dir=tmp_path / "missing-dataset",
            carbon_model_dir=tmp_path / "missing-carbon",
            run_dir=run_dir,
            steps=1,
            command="geno-lewm-train --carbon-train",
            commit_sha="0000000",
            source_tree="0000000",
            package_version="0.2.1",
        )

    assert not run_dir.exists()


def test_run_carbon_training_rejects_forged_full_source_pair_before_writes(
    tmp_path: Path,
) -> None:
    config = load_config(_write_training_config(tmp_path))
    run_dir = tmp_path / "run"

    with pytest.raises(InputError, match="does not match the imported package"):
        real_module.run_carbon_training(
            config=config,
            dataset_dir=tmp_path / "missing-dataset",
            carbon_model_dir=tmp_path / "missing-carbon",
            run_dir=run_dir,
            steps=1,
            command="geno-lewm-train --carbon-train",
            commit_sha="c" * 40,
            source_tree="d" * 40,
            package_version="0.2.1",
        )

    assert not run_dir.exists()


def test_run_carbon_training_rejects_wrong_imported_package_version_before_writes(
    tmp_path: Path,
) -> None:
    config = load_config(_write_training_config(tmp_path))
    run_dir = tmp_path / "run"

    with pytest.raises(InputError, match="does not match the imported package"):
        real_module.run_carbon_training(
            config=config,
            dataset_dir=tmp_path / "missing-dataset",
            carbon_model_dir=tmp_path / "missing-carbon",
            run_dir=run_dir,
            steps=1,
            command="geno-lewm-train --carbon-train",
            commit_sha="a" * 40,
            source_tree="b" * 40,
            package_version="0.3.0",
        )

    assert not run_dir.exists()


def test_run_carbon_training_rejects_unsupported_atomic_boundary_before_writes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = load_config(_write_training_config(tmp_path))
    run_dir = tmp_path / "run"
    monkeypatch.setattr(
        atomic_module,
        "_supports_anchored_directory_operations",
        lambda: False,
    )

    with pytest.raises(InputError, match="requires anchored directory operations"):
        real_module.run_carbon_training(
            config=config,
            dataset_dir=tmp_path / "missing-dataset",
            carbon_model_dir=tmp_path / "missing-carbon",
            run_dir=run_dir,
            steps=1,
            command="geno-lewm-train --carbon-train",
            commit_sha="a" * 40,
            source_tree="b" * 40,
            package_version="0.2.1",
        )

    assert not run_dir.exists()


@requires_secure_atomic_publication
def test_run_carbon_training_rejects_concurrent_same_run_writer(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = load_config(_write_training_config(tmp_path))
    run_dir = tmp_path / "run"
    entered = threading.Event()
    release = threading.Event()
    owner_errors: list[BaseException] = []

    def blocking_manifest(_dataset_dir: Path) -> dict[str, object]:
        if threading.current_thread().name == "training-run-owner":
            entered.set()
            assert release.wait(timeout=5)
            raise InputError("owner stopped after concurrency check")
        raise InputError("concurrent writer reached dataset loading")

    def owner() -> None:
        try:
            real_module.run_carbon_training(
                config=config,
                dataset_dir=tmp_path / "dataset",
                carbon_model_dir=tmp_path / "carbon",
                run_dir=run_dir,
                steps=1,
                command="geno-lewm-train --carbon-train",
                commit_sha="a" * 40,
                source_tree="b" * 40,
                package_version="0.2.1",
            )
        except BaseException as exc:  # pragma: no cover - asserted below.
            owner_errors.append(exc)

    monkeypatch.setattr(real_module, "_load_dataset_manifest", blocking_manifest)
    thread = threading.Thread(target=owner, name="training-run-owner")
    thread.start()
    assert entered.wait(timeout=5)
    try:
        with pytest.raises(InputError, match="already active"):
            real_module.run_carbon_training(
                config=config,
                dataset_dir=tmp_path / "dataset",
                carbon_model_dir=tmp_path / "carbon",
                run_dir=run_dir,
                steps=1,
                command="geno-lewm-train --carbon-train",
                commit_sha="a" * 40,
                source_tree="b" * 40,
                package_version="0.2.1",
            )
    finally:
        release.set()
        thread.join(timeout=5)

    assert not thread.is_alive()
    assert len(owner_errors) == 1
    assert "owner stopped" in str(owner_errors[0])


def test_run_carbon_training_stops_at_k_under_the_n_step_horizon(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    torch = pytest.importorskip("torch")
    config = load_config(_write_training_config(tmp_path))
    config = replace(
        config,
        predictor=replace(config.predictor, d_state=4, dtype="fp32"),
        data=replace(config.data, batch_size=1),
        runtime=replace(config.runtime, device="cpu"),
    )
    dataset_dir = tmp_path / "dataset"
    dataset_dir.mkdir()
    (dataset_dir / "dataset_manifest.json").write_text(
        json.dumps({"schema_version": "1.0.0", "snapshot_id": "fixture", "files": [{}]}),
        encoding="utf-8",
    )
    carbon_dir = tmp_path / "carbon"
    carbon_dir.mkdir()

    window = WindowContext(
        record_id="record-1",
        source="fixture",
        sequence="ACGT" * 16,
    )
    monkeypatch.setattr(real_module, "_load_windows", lambda *_args, **_kwargs: iter((window,)))
    monkeypatch.setattr(
        real_module,
        "_load_gnomad_edits",
        lambda *_args, **_kwargs: iter((EditSpec(chrom="1", pos=1, ref="A", alt="T"),)),
    )
    monkeypatch.setattr(real_module, "_load_clinvar_edits", lambda *_args, **_kwargs: iter(()))
    monkeypatch.setattr(
        real_module, "_carbon_identity_hash", lambda *_args, **_kwargs: _ENCODER_WEIGHTS_HASH
    )
    monkeypatch.setattr(real_module, "_training_edit_contract", lambda *_args, **_kwargs: ({}, ()))

    class Stream:
        def iter_repeated(self):
            return iter([object()] * 20)

    monkeypatch.setattr(
        real_module.PreparedTrainingStream,
        "from_components",
        staticmethod(lambda **_kwargs: Stream()),
    )

    class Encoder:
        def __init__(self, *_args, **_kwargs) -> None:
            pass

    monkeypatch.setattr(real_module, "CarbonStateEncoder", Encoder)
    monkeypatch.setattr(
        real_module,
        "_encode_items",
        lambda *_args, **_kwargs: SimpleNamespace(
            state=torch.zeros((1, 4)),
            window_ids=("window-1",),
        ),
    )

    class Stateful:
        def state_dict(self) -> dict[str, object]:
            return {"value": torch.tensor([1.0])}

        def load_state_dict(self, _state: object) -> None:
            pass

    monkeypatch.setattr(real_module, "ActionEncoder", lambda **_kwargs: Stateful())
    monkeypatch.setattr(real_module, "build_predictor", lambda _config: Stateful())
    monkeypatch.setattr(real_module, "build_adamw_optimizer", lambda **_kwargs: Stateful())

    class Trainer:
        def __init__(self, **kwargs: object) -> None:
            self.total_steps = kwargs["total_steps"]
            self.last_collapse_alerts: tuple[dict[str, object], ...] = ()

        def train_step(self, _batch: object, *, step: int) -> TorchTrainerStepResult:
            return _step_result(loss=float(step), var=1.0, step=step)

        def state_dict(self) -> dict[str, object]:
            return {
                "schema_version": "fixture-trainer-state",
                "total_steps": self.total_steps,
            }

    monkeypatch.setattr(real_module, "TorchTrainer", Trainer)
    monkeypatch.setattr(
        real_module,
        "configure_torch_reproducibility",
        lambda **_kwargs: SimpleNamespace(to_dict=lambda: {"deterministic": True}),
    )

    report = real_module.run_carbon_training(
        config=config,
        dataset_dir=dataset_dir,
        carbon_model_dir=carbon_dir,
        run_dir=tmp_path / "run",
        steps=5,
        stop_after_step=2,
        command="geno-lewm-train --carbon-train --steps 5 --stop-after-step 2",
        commit_sha="a" * 40,
        source_tree="b" * 40,
        package_version="0.2.1",
    )

    assert report.steps_requested == 5
    assert report.steps_completed == 2
    checkpoint = load_resume_checkpoint(report.checkpoint_path)
    assert checkpoint["schema_version"] == CHECKPOINT_SCHEMA_VERSION
    assert checkpoint["progress"]["steps_completed"] == 2
    assert checkpoint["training_contract"]["target_steps"] == 5
    assert [row["step"] for row in checkpoint["metric_history"]] == [1, 2]
    assert set(checkpoint["rng_state"]) == {"python", "numpy", "torch_cpu", "torch_cuda"}


def test_real_training_restores_cumulative_state_before_k_plus_one_and_is_bit_equal(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    numpy = pytest.importorskip("numpy")
    torch = pytest.importorskip("torch")
    config = load_config(_write_training_config(tmp_path))
    config = replace(
        config,
        encoder=replace(
            config.encoder,
            dtype="fp32",
            normalize=True,
            state_contract_version="l2_normalized_v2",
        ),
        predictor=replace(
            config.predictor,
            n_layers=1,
            n_heads=1,
            d_state=4,
            d_action=3,
            dtype="fp32",
        ),
        action=replace(config.action, d_action=3, sub_encoders=("snv",)),
        training=replace(config.training, max_steps=4, collapse_log_every_steps=1),
        optimizer=replace(config.optimizer, warmup_steps=1, lr=1.0e-3),
        data=replace(config.data, batch_size=2, num_workers=0, shuffle_buffer=0),
        runtime=replace(config.runtime, device="cpu"),
        deterministic=True,
    )
    dataset_dir = tmp_path / "dataset"
    dataset_dir.mkdir()
    (dataset_dir / "dataset_manifest.json").write_text(
        json.dumps({"schema_version": "1.0.0", "snapshot_id": "fixture", "files": [{}]}),
        encoding="utf-8",
    )
    carbon_dir = tmp_path / "carbon"
    carbon_dir.mkdir()
    window = WindowContext(record_id="record-1", source="fixture", sequence="ACGT" * 16)
    monkeypatch.setattr(real_module, "_load_windows", lambda *_args, **_kwargs: iter((window,)))
    monkeypatch.setattr(
        real_module,
        "_load_gnomad_edits",
        lambda *_args, **_kwargs: iter((EditSpec(chrom="1", pos=1, ref="A", alt="T"),)),
    )
    monkeypatch.setattr(real_module, "_load_clinvar_edits", lambda *_args, **_kwargs: iter(()))
    monkeypatch.setattr(
        real_module, "_carbon_identity_hash", lambda *_args, **_kwargs: _ENCODER_WEIGHTS_HASH
    )
    monkeypatch.setattr(real_module, "_training_edit_contract", lambda *_args, **_kwargs: ({}, ()))

    class Stream:
        def iter_repeated(self):
            return iter(
                SimpleNamespace(
                    value=value,
                    training_tuple=SimpleNamespace(window_id=f"window-{value}"),
                )
                for value in range(1, 100)
            )

    monkeypatch.setattr(
        real_module.PreparedTrainingStream,
        "from_components",
        staticmethod(lambda **_kwargs: Stream()),
    )

    class Encoder:
        def __init__(self, *_args, **_kwargs) -> None:
            pass

    class SyntheticActionEncoder(torch.nn.Module):
        def __init__(self, *, d_action: int) -> None:
            super().__init__()
            self.projection = torch.nn.Linear(4, d_action)

        def forward(self, edits):
            rows = []
            for row in edits:
                edit = row[0]
                jitter = random.random() + float(numpy.random.random())
                rows.append([[float(edit.rel_pos), float(int(edit.edit_type)), 1.0, jitter * 0.01]])
            return self.projection(torch.tensor(rows, dtype=torch.float32))

    class SyntheticPredictor(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.action = torch.nn.Linear(3, 4)
            self.dropout = torch.nn.Dropout(p=0.25)
            self.output = torch.nn.Linear(4, 4)

        def forward(self, state, actions, action_mask):
            hidden = state.unsqueeze(1) + self.action(actions)
            return self.output(self.dropout(hidden)).masked_fill(~action_mask.unsqueeze(-1), 0.0)

    def encoded_batch(_encoder, items, *, device):
        del device
        values = [int(item.value) for item in items]
        state = torch.tensor(
            [[value / 10.0, 0.2, 0.3, 0.4] for value in values],
            dtype=torch.float32,
        )
        target = (torch.roll(state, shifts=1, dims=1) + 0.01).unsqueeze(1)
        edits = tuple((RelEdit(value % 16, EditType.SNV, "A", "T"),) for value in values)
        return TorchTrainerBatch(
            state=state,
            target=target,
            rel_edits=edits,
            action_mask=torch.ones((len(values), 1), dtype=torch.bool),
            window_ids=tuple(f"window-{value}" for value in values),
        )

    monkeypatch.setattr(real_module, "CarbonStateEncoder", Encoder)
    monkeypatch.setattr(real_module, "ActionEncoder", SyntheticActionEncoder)
    monkeypatch.setattr(real_module, "build_predictor", lambda _config: SyntheticPredictor())
    monkeypatch.setattr(real_module, "_encode_items", encoded_batch)

    common = {
        "config": config,
        "dataset_dir": dataset_dir,
        "carbon_model_dir": carbon_dir,
        "steps": 4,
        "command": "geno-lewm-train --carbon-train --steps 4",
        "commit_sha": "a" * 40,
        "source_tree": "b" * 40,
        "package_version": "0.2.1",
    }
    full = real_module.run_carbon_training(run_dir=tmp_path / "full", **common)
    prefix = real_module.run_carbon_training(
        run_dir=tmp_path / "prefix",
        stop_after_step=2,
        **common,
    )
    resume_events: list[str] = []
    original_checkpoint_loader = real_module._load_torch_checkpoint
    original_next_batch = real_module._next_batch

    class TrackedWindowId(str):
        def __str__(self) -> str:
            resume_events.append("restore-cumulative-progress")
            return super().__str__()

    def tracked_checkpoint_loader(path: Path) -> dict[str, object]:
        payload = original_checkpoint_loader(path)
        progress = payload["progress"]
        assert isinstance(progress, dict)
        consumed = progress["consumed_window_ids"]
        assert isinstance(consumed, list)
        progress["consumed_window_ids"] = [TrackedWindowId(item) for item in consumed]
        return payload

    def tracked_next_batch(iterator, batch_size: int):
        resume_events.append("fetch-k-plus-one")
        return original_next_batch(iterator, batch_size)

    monkeypatch.setattr(real_module, "_load_torch_checkpoint", tracked_checkpoint_loader)
    monkeypatch.setattr(real_module, "_next_batch", tracked_next_batch)
    resumed = real_module.run_carbon_training(
        run_dir=tmp_path / "resumed",
        resume_from=prefix.checkpoint_path,
        **common,
    )

    assert resume_events.index("restore-cumulative-progress") < resume_events.index(
        "fetch-k-plus-one"
    )

    full_checkpoint = load_resume_checkpoint(full.checkpoint_path)
    resumed_checkpoint = load_resume_checkpoint(resumed.checkpoint_path)
    assert full_checkpoint["state_digests"] == resumed_checkpoint["state_digests"]
    assert full_checkpoint["rng_state_digests"] == resumed_checkpoint["rng_state_digests"]
    assert full_checkpoint["metric_history"] == resumed_checkpoint["metric_history"]
    assert full_checkpoint["progress"] == resumed_checkpoint["progress"]
    full_metrics = json.loads(full.metrics_path.read_text(encoding="utf-8"))
    resumed_metrics = json.loads(resumed.metrics_path.read_text(encoding="utf-8"))
    assert full_metrics["history"] == resumed_metrics["history"]
    assert full_metrics["metrics"]["nan_loss_count"] == resumed_metrics["metrics"]["nan_loss_count"]
    assert (
        full_metrics["metrics"]["collapse_var_min"]
        == resumed_metrics["metrics"]["collapse_var_min"]
    )
    assert [row["lr_multiplier"] for row in resumed_metrics["history"]] == [
        row["lr_multiplier"] for row in full_metrics["history"]
    ]


def test_skip_training_items_rejects_replayed_window_order_drift() -> None:
    items = iter(
        SimpleNamespace(training_tuple=SimpleNamespace(window_id=window_id))
        for window_id in ("window-1", "window-drift")
    )

    with pytest.raises(InputError, match="sample order does not match"):
        _skip_training_items(
            items,
            item_count=2,
            expected_window_ids=("window-1", "window-2"),
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
    membership_identity = _membership_runtime_binding()
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
        membership_identity=membership_identity,
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
    assert payload["membership_and_split_evidence"] == membership_identity


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
    membership_identity = _membership_runtime_binding()
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
        target_steps=8,
        steps_completed=3,
        resumed_from_step=0,
        resume_checkpoint_path=None,
        membership_identity=membership_identity,
    )

    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    identities = metadata["artifact_identities"]
    assert identities["dataset_manifest"] == _identity(tmp_path, "dataset_manifest.json")
    assert identities["training_config"] == _identity(tmp_path, "training_config.effective.yaml")
    assert identities["metrics"] == _identity(tmp_path, "metrics.json")
    assert identities["logs"] == [_identity(tmp_path, "train.log")]
    assert identities["checkpoint_files"] == [_identity(tmp_path, "predictor_checkpoint.pt")]
    assert identities["training_preflight_report"] == _identity(tmp_path, REPORT_NAME)
    assert metadata["schema_version"] == "1.1.0"
    assert metadata["membership_and_split_evidence"] == membership_identity
    assert metadata["status"] == "stopped_early"
    assert metadata["target_steps"] == 8
    assert metadata["steps_completed"] == 3


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


def test_release_training_loader_uses_carbon_when_snapshot_has_no_placed_windows(
    tmp_path: Path,
) -> None:
    carbon_path = tmp_path / "carbon" / "source-mix-windows.jsonl"
    carbon_path.parent.mkdir(parents=True)
    carbon_path.write_text(
        json.dumps(
            {
                "record_id": "carbon-1",
                "source": "eukaryotic_genes",
                "sequence": "ACGT" * 64,
            }
        )
        + "\n",
        encoding="utf-8",
    )

    windows = tuple(
        _load_windows(
            tmp_path,
            ({"path": "carbon/source-mix-windows.jsonl"},),
        )
    )

    assert len(windows) == 1
    assert windows[0].record_id == "carbon-1"
    assert windows[0].chrom is None
    assert _dataset_fallback_sources(windows) == {
        "gnomad_common": "synthetic_snv",
        "clinvar": "synthetic_snv",
    }


def test_schema_1_1_window_loader_reads_only_split_data(tmp_path: Path) -> None:
    placed = tmp_path / "placed"
    placed.mkdir()
    row = {
        "record_id": "train-window",
        "source": SOURCE_GNOMAD_COMMON,
        "sequence": "ACGT" * 64,
        "chrom": "1",
    }
    (placed / "train.jsonl").write_text(json.dumps(row) + "\n", encoding="utf-8")
    (placed / "labels.jsonl").write_text("not-json\n", encoding="utf-8")

    windows = tuple(
        _load_windows(
            tmp_path,
            (
                {"path": "placed/train.jsonl", "artifact_role": "split_data"},
                {"path": "placed/labels.jsonl", "artifact_role": "split_companion"},
            ),
            schema_version="1.1.0",
        )
    )

    assert [window.record_id for window in windows] == ["train-window"]


def test_schema_1_1_variant_loader_reads_only_split_data(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    gnomad = tmp_path / "gnomad"
    gnomad.mkdir()
    (gnomad / "train.parquet").write_bytes(b"train")
    (gnomad / "labels.parquet").write_bytes(b"companion")
    observed: list[Path] = []

    def fake_iter(path: Path):
        observed.append(path)
        return iter(())

    monkeypatch.setattr(real_module, "iter_gnomad_shard", fake_iter)

    assert (
        tuple(
            real_module._load_gnomad_edits(
                tmp_path,
                (
                    {"path": "gnomad/train.parquet", "artifact_role": "split_data"},
                    {"path": "gnomad/labels.parquet", "artifact_role": "split_companion"},
                ),
                schema_version="1.1.0",
            )
        )
        == ()
    )
    assert observed == [(gnomad / "train.parquet").resolve()]


def test_membership_holdout_policy_opens_once_and_closes_on_success(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store_dir = tmp_path / "evidence" / "membership-store"
    store_dir.mkdir(parents=True)
    events: list[object] = []

    class FakeStore:
        manifest = SimpleNamespace(
            **{key: value for key, value in _MEMBERSHIP_BINDING.items() if key != "path"}
        )

        @classmethod
        def open(cls, path: Path, *, verify: bool):
            events.append(("open", path, verify))
            return cls()

        def close(self) -> None:
            events.append("close")

    class FakePolicy:
        def __init__(self, store: object) -> None:
            events.append(("policy", store))

    monkeypatch.setattr(real_module, "MembershipStore", FakeStore)
    monkeypatch.setattr(real_module, "MembershipStoreHoldoutPolicy", FakePolicy)
    manifest = {
        "schema_version": "1.1.0",
        "membership_and_split_evidence": _membership_dataset_binding(),
    }

    with real_module._membership_holdout_policy(tmp_path, manifest) as policy:
        assert isinstance(policy, FakePolicy)
        assert [event[0] for event in events if isinstance(event, tuple)].count("open") == 1

    assert events[-1] == "close"


def test_membership_runtime_identity_carries_full_dataset_binding_and_policy() -> None:
    store = object.__new__(real_module.MembershipStore)
    object.__setattr__(
        store,
        "manifest",
        SimpleNamespace(
            chromosome_roles=SimpleNamespace(validation=("20",), evaluation=("21",)),
            content_identity=_MEMBERSHIP_BINDING["content_identity"],
        ),
    )
    policy = real_module.MembershipStoreHoldoutPolicy(store)
    dataset_binding = _membership_dataset_binding()

    identity = real_module._membership_runtime_identity(
        {
            "schema_version": "1.1.0",
            "membership_and_split_evidence": dataset_binding,
        },
        policy,
    )

    assert identity == {
        **dataset_binding,
        "holdout_policy": policy.to_dict(),
        "holdout_policy_identity": policy.identity(),
    }


def test_membership_holdout_policy_rejects_identity_drift_and_closes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store_dir = tmp_path / "evidence" / "membership-store"
    store_dir.mkdir(parents=True)
    events: list[str] = []

    class FakeStore:
        manifest = SimpleNamespace(
            **{
                **{key: value for key, value in _MEMBERSHIP_BINDING.items() if key != "path"},
                "rowset_sha256": "sha256:" + ("e" * 64),
            }
        )

        @classmethod
        def open(cls, path: Path, *, verify: bool):
            del path, verify
            events.append("open")
            return cls()

        def close(self) -> None:
            events.append("close")

    monkeypatch.setattr(real_module, "MembershipStore", FakeStore)
    manifest = {
        "schema_version": "1.1.0",
        "membership_and_split_evidence": _membership_dataset_binding(),
    }

    with pytest.raises(InputError, match="identity does not match"):
        with real_module._membership_holdout_policy(tmp_path, manifest):
            pytest.fail("identity drift must fail before yielding")

    assert events == ["open", "close"]


def test_legacy_manifest_does_not_open_membership_store(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class ForbiddenStore:
        @classmethod
        def open(cls, path: Path, *, verify: bool):
            del path, verify
            pytest.fail("legacy manifests must not open a membership store")

    monkeypatch.setattr(real_module, "MembershipStore", ForbiddenStore)

    with real_module._membership_holdout_policy(
        tmp_path,
        {"schema_version": "1.0.0"},
    ) as policy:
        assert policy is None


def test_membership_evidence_section_requires_membership_store() -> None:
    with pytest.raises(InputError, match="membership_store is required"):
        real_module._membership_store_binding(
            {"membership_and_split_evidence": {"report": {"path": "evidence/report.json"}}}
        )


def test_membership_evidence_section_requires_report() -> None:
    with pytest.raises(InputError, match="report is required"):
        real_module._membership_store_binding(
            {"membership_and_split_evidence": {"membership_store": _MEMBERSHIP_BINDING}}
        )


def test_prepared_training_stream_cycles_finite_release_snapshot() -> None:
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

    stream = PreparedTrainingStream.from_components(
        dataset_snapshot_id="fixture",
        schema_version="1.0.0",
        windows=windows,
        providers=providers,
        seed=11,
        fallback_sources=_dataset_fallback_sources(windows),
        mix=real_module.DEFAULT_EDIT_SOURCE_COUNTS,
        holdouts=None,
        membership_identity=None,
    )
    iterator = stream.iter_repeated()

    first_epoch = _next_batch(iterator, 8)
    next_epoch = _next_batch(iterator, 1)

    assert {item.source_window.record_id for item in first_epoch} == {"placed-1"}
    assert next_epoch[0].source_window.record_id == "placed-1"


def test_prepared_training_stream_applies_holdouts_to_every_epoch() -> None:
    windows = (
        WindowContext(
            record_id="train-window",
            source=SOURCE_GNOMAD_COMMON,
            sequence="ACGT" * 64,
            chrom="1",
            start_bp=100,
        ),
        WindowContext(
            record_id="evaluation-window",
            source=SOURCE_GNOMAD_COMMON,
            sequence="ACGT" * 64,
            chrom="21",
            start_bp=100,
        ),
    )
    providers = {
        SOURCE_GNOMAD_COMMON: synthetic_snv_provider,
        SOURCE_SYNTHETIC_SNV: synthetic_snv_provider,
        SOURCE_SYNTHETIC_INDEL: synthetic_indel_provider,
        SOURCE_CLINVAR: lambda window, count, rng: (),
    }

    stream = PreparedTrainingStream.from_components(
        dataset_snapshot_id="fixture",
        schema_version="1.0.0",
        windows=windows,
        providers=providers,
        seed=11,
        fallback_sources=_dataset_fallback_sources(windows),
        mix=real_module.DEFAULT_EDIT_SOURCE_COUNTS,
        holdouts=HoldoutPolicy(holdout_chroms=("21",)),
        membership_identity=None,
    )
    iterator = stream.iter_repeated()

    first_epoch = _next_batch(iterator, 8)
    second_epoch = _next_batch(iterator, 8)

    assert {item.source_window.record_id for item in first_epoch} == {"train-window"}
    assert {item.source_window.record_id for item in second_epoch} == {"train-window"}


def test_snv_only_training_contract_filters_indels_and_preserves_eight_actions(
    tmp_path: Path,
) -> None:
    config = load_config(_write_training_config(tmp_path))
    window = WindowContext(
        record_id="carbon-1",
        source="eukaryotic_genes",
        sequence="ACGT" * 64,
    )
    providers, mix = _training_edit_contract(
        config,
        gnomad_edits=(
            EditSpec(chrom="1", pos=1, ref="A", alt="T"),
            EditSpec(chrom="1", pos=2, ref="C", alt="CA"),
        ),
        clinvar_edits=(EditSpec(chrom="1", pos=3, ref="G", alt="A"),),
    )

    assert [(entry.source, entry.count) for entry in mix] == [
        (SOURCE_GNOMAD_COMMON, 3),
        (SOURCE_SYNTHETIC_SNV, 4),
        (SOURCE_CLINVAR, 1),
    ]
    stream = PreparedTrainingStream.from_components(
        dataset_snapshot_id="fixture",
        schema_version="1.0.0",
        windows=(window,),
        providers=providers,
        seed=11,
        fallback_sources=_dataset_fallback_sources((window,)),
        mix=mix,
        holdouts=None,
        membership_identity=None,
    )
    iterator = stream.iter_repeated()
    items = _next_batch(iterator, 8)

    assert len(items) == 8
    assert all(item.training_tuple.rel_edits[0].edit_type is EditType.SNV for item in items)


def test_snv_only_training_accepts_a_sparse_acgt_carbon_window(tmp_path: Path) -> None:
    config = load_config(_write_training_config(tmp_path))
    sequence = ["N"] * 4096
    sequence[2048] = "A"
    window = WindowContext(
        record_id="sparse-carbon",
        source="eukaryotic_genes",
        sequence="".join(sequence),
    )
    providers, mix = _training_edit_contract(
        config,
        gnomad_edits=(EditSpec(chrom="1", pos=1, ref="A", alt="T"),),
        clinvar_edits=(EditSpec(chrom="1", pos=3, ref="G", alt="A"),),
    )

    stream = PreparedTrainingStream.from_components(
        dataset_snapshot_id="fixture",
        schema_version="1.0.0",
        windows=(window,),
        providers=providers,
        seed=11,
        fallback_sources=_dataset_fallback_sources((window,)),
        mix=mix,
        holdouts=None,
        membership_identity=None,
    )
    iterator = stream.iter_repeated()
    items = _next_batch(iterator, 8)

    assert len(items) == 8
    assert {item.training_tuple.rel_edits[0].rel_pos for item in items} == {2048}


def test_training_edit_contract_rejects_unimplemented_action_subset(tmp_path: Path) -> None:
    config = load_config(_write_training_config(tmp_path))
    unsupported = replace(config, action=replace(config.action, sub_encoders=("snv", "ins")))

    with pytest.raises(InputError, match="does not support"):
        _training_edit_contract(
            unsupported,
            gnomad_edits=(EditSpec(chrom="1", pos=1, ref="A", alt="T"),),
            clinvar_edits=(),
        )


def test_prepared_training_stream_rejects_empty_window_set() -> None:
    windows: tuple[WindowContext, ...] = ()
    providers = {
        SOURCE_GNOMAD_COMMON: lambda window, count, rng: (),
        SOURCE_SYNTHETIC_SNV: lambda window, count, rng: (),
        SOURCE_SYNTHETIC_INDEL: lambda window, count, rng: (),
        SOURCE_CLINVAR: lambda window, count, rng: (),
    }

    with pytest.raises(InputError, match="contains no usable source windows"):
        PreparedTrainingStream.from_components(
            dataset_snapshot_id="fixture",
            schema_version="1.0.0",
            windows=windows,
            providers=providers,
            seed=11,
            fallback_sources={},
            mix=real_module.DEFAULT_EDIT_SOURCE_COUNTS,
            holdouts=None,
            membership_identity=None,
        )


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


@requires_secure_atomic_publication
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
    prepared: list[dict[str, object]] = []

    monkeypatch.setattr(
        real_module,
        "_load_dataset_manifest",
        lambda _dataset_dir: {
            "schema_version": "1.0.0",
            "snapshot_id": "fixture",
            "files": [],
        },
    )
    monkeypatch.setattr(real_module, "_dataset_files", lambda _manifest: ())
    monkeypatch.setattr(real_module, "_load_windows", lambda *_args, **_kwargs: iter((window,)))
    monkeypatch.setattr(
        real_module,
        "_load_gnomad_edits",
        lambda *_args, **_kwargs: iter((EditSpec(chrom="1", pos=1, ref="A", alt="T"),)),
    )
    monkeypatch.setattr(real_module, "_load_clinvar_edits", lambda *_args, **_kwargs: iter(()))

    def fake_encoder_identity_hash(model_dir: Path, *, state_contract_version: str) -> str:
        observed.append((model_dir, state_contract_version))
        return _ENCODER_WEIGHTS_HASH

    def resolve_edit_contract(
        observed_config: object,
        *,
        gnomad_edits: object,
        clinvar_edits: object,
    ) -> tuple[dict[str, object], tuple[object, ...]]:
        assert observed_config is config
        assert tuple(gnomad_edits) == (EditSpec(chrom="1", pos=1, ref="A", alt="T"),)
        assert tuple(clinvar_edits) == ()
        return {}, ()

    class StopStream:
        def iter_repeated(self) -> None:
            raise RuntimeError("prepared iterator reached")

    def capture_prepared_stream(**kwargs: object) -> StopStream:
        prepared.append(dict(kwargs))
        return StopStream()

    monkeypatch.setattr(real_module, "encoder_identity_hash", fake_encoder_identity_hash)
    monkeypatch.setattr(real_module, "_training_device", lambda _config: "cpu")
    monkeypatch.setattr(real_module, "configure_torch_reproducibility", lambda **_kwargs: object())
    monkeypatch.setattr(real_module, "_training_edit_contract", resolve_edit_contract)
    monkeypatch.setattr(
        real_module.PreparedTrainingStream,
        "from_components",
        staticmethod(capture_prepared_stream),
    )

    with pytest.raises(RuntimeError, match="prepared iterator reached"):
        real_module.run_carbon_training(
            config=config,
            dataset_dir=dataset_dir,
            carbon_model_dir=carbon_dir,
            run_dir=tmp_path / "run",
            steps=1,
            command="geno-lewm-train --carbon-train",
            commit_sha="a" * 40,
            source_tree="b" * 40,
            package_version="0.2.1",
        )

    assert observed == [(carbon_dir, config.encoder.state_contract_version)]
    assert prepared[0]["dataset_snapshot_id"] == "fixture"
    assert prepared[0]["schema_version"] == "1.0.0"
    assert prepared[0]["membership_identity"] is None


_RESUME_DATASET_MANIFEST = {
    "schema_version": "1.0.0",
    "snapshot_id": "geno-lewm-data-v0.1.0-r1",
    "files": [],
}
_RESUME_COMMIT = "a" * 40
_RESUME_TREE = "b" * 40


def test_validate_resume_checkpoint_accepts_matching_closed_identity(tmp_path: Path) -> None:
    config = load_config(_write_training_config(tmp_path))
    seeds = TrainerSeeds.from_base_seed(config.seed)
    payload = _resume_payload(tmp_path, config, seeds, steps_completed=3)

    checkpoint = _validate_resume_checkpoint_payload(
        payload,
        path=tmp_path / "predictor_checkpoint.pt",
        **_resume_validation_args(config, seeds),
    )

    assert checkpoint.steps_completed == 3


@pytest.mark.parametrize("identity", ["commit", "tree"])
def test_validate_resume_checkpoint_rejects_source_identity_drift(
    tmp_path: Path,
    identity: str,
) -> None:
    config = load_config(_write_training_config(tmp_path))
    seeds = TrainerSeeds.from_base_seed(config.seed)
    source = {"commit_sha": _RESUME_COMMIT, "tree_sha": _RESUME_TREE}
    source[f"{identity}_sha"] = "f" * 40
    payload = _resume_payload(tmp_path, config, seeds, steps_completed=3, source=source)

    with pytest.raises(InputError, match="source identity"):
        _validate_resume_checkpoint_payload(
            payload,
            path=tmp_path / "predictor_checkpoint.pt",
            **_resume_validation_args(config, seeds),
        )


def test_validate_resume_checkpoint_rejects_cross_package_version(
    tmp_path: Path,
) -> None:
    config = load_config(_write_training_config(tmp_path))
    seeds = TrainerSeeds.from_base_seed(config.seed)
    payload = _resume_payload(
        tmp_path,
        config,
        seeds,
        steps_completed=3,
        source={
            "commit_sha": _RESUME_COMMIT,
            "tree_sha": _RESUME_TREE,
            "package_version": "0.2.0",
        },
    )
    validation = _resume_validation_args(config, seeds)
    validation["package_version"] = "0.3.0"

    with pytest.raises(InputError, match="source identity"):
        _validate_resume_checkpoint_payload(
            payload,
            path=tmp_path / "predictor_checkpoint.pt",
            **validation,
        )


def test_validate_resume_checkpoint_rejects_full_config_mismatch(tmp_path: Path) -> None:
    config = load_config(_write_training_config(tmp_path))
    seeds = TrainerSeeds.from_base_seed(config.seed)
    contract = _resume_training_contract(config, seeds, target_steps=5)
    contract_config = contract["config"]
    assert isinstance(contract_config, dict)
    contract_config["seed"] = config.seed + 1
    payload = _resume_payload(
        tmp_path,
        config,
        seeds,
        steps_completed=3,
        training_contract=contract,
    )

    with pytest.raises(InputError, match="training contract"):
        _validate_resume_checkpoint_payload(
            payload,
            path=tmp_path / "predictor_checkpoint.pt",
            **_resume_validation_args(config, seeds),
        )


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (lambda row: row.update({"unexpected": 1}), "metric row fields"),
        (lambda row: row.update({"lr_multiplier": 0.123}), "learning-rate schedule"),
    ],
)
def test_validate_resume_checkpoint_rejects_open_or_wrong_lr_metric_history(
    tmp_path: Path,
    mutation,
    message: str,
) -> None:
    config = load_config(_write_training_config(tmp_path))
    seeds = TrainerSeeds.from_base_seed(config.seed)
    payload = _resume_payload(tmp_path, config, seeds, steps_completed=3)
    history = payload["metric_history"]
    assert isinstance(history, list)
    row = history[0]
    assert isinstance(row, dict)
    row.update(
        {
            "lr_multiplier": 1.0,
            "loss": 0.5,
            "pred_loss": 0.5,
            "kl_reg": 0.0,
            "action_count": 1,
            "pred_var_per_dim": 0.2,
        }
    )
    mutation(row)

    with pytest.raises(InputError, match=message):
        _validate_resume_checkpoint_payload(
            payload,
            path=tmp_path / "predictor_checkpoint.pt",
            **_resume_validation_args(config, seeds),
        )


@pytest.mark.parametrize("identity", ["dataset_manifest", "encoder", "membership_and_split"])
def test_validate_resume_checkpoint_rejects_data_or_encoder_identity_drift(
    tmp_path: Path,
    identity: str,
) -> None:
    config = load_config(_write_training_config(tmp_path))
    seeds = TrainerSeeds.from_base_seed(config.seed)
    membership = _membership_runtime_binding()
    identities = _resume_identities(membership)
    identities[identity] = "sha256:" + ("f" * 64)
    payload = _resume_payload(
        tmp_path,
        config,
        seeds,
        steps_completed=3,
        identities=identities,
    )

    with pytest.raises(InputError, match="data or encoder identities"):
        _validate_resume_checkpoint_payload(
            payload,
            path=tmp_path / "predictor_checkpoint.pt",
            **_resume_validation_args(config, seeds, membership_identity=membership),
        )


def test_validate_resume_checkpoint_rejects_finished_checkpoint(tmp_path: Path) -> None:
    config = load_config(_write_training_config(tmp_path))
    seeds = TrainerSeeds.from_base_seed(config.seed)
    payload = _resume_payload(tmp_path, config, seeds, steps_completed=5)

    with pytest.raises(InputError, match="already at or beyond"):
        _validate_resume_checkpoint_payload(
            payload,
            path=tmp_path / "predictor_checkpoint.pt",
            **_resume_validation_args(config, seeds),
        )


def test_write_checkpoint_emits_resume_compatible_closed_payload(tmp_path: Path) -> None:
    torch = pytest.importorskip("torch")
    config = load_config(_write_training_config(tmp_path))
    seeds = TrainerSeeds.from_base_seed(config.seed)
    predictor = torch.nn.Linear(2, 2)
    action_encoder = torch.nn.Linear(2, 2)
    optimizer = torch.optim.AdamW(list(predictor.parameters()) + list(action_encoder.parameters()))
    trainer = SimpleNamespace(state_dict=lambda: {"schema_version": "fixture", "total_steps": 8})
    path = tmp_path / "predictor_checkpoint.pt"
    membership_identity = _membership_runtime_binding()
    consumed = [f"window-{index}" for index in range(7 * config.data.batch_size)]
    history = [
        {
            **_step_result(loss=0.5, var=0.2, step=step).to_dict(),
            "lr_multiplier": wsd_lr_multiplier(
                step,
                total_steps=8,
                warmup_steps=config.optimizer.warmup_steps,
                schedule=config.optimizer.schedule,
            ),
        }
        for step in range(1, 8)
    ]

    _write_checkpoint(
        path,
        predictor=predictor,
        action_encoder=action_encoder,
        optimizer=optimizer,
        trainer=trainer,
        config=config,
        dataset_snapshot_id="geno-lewm-data-v0.1.0-r1",
        dataset_manifest=_RESUME_DATASET_MANIFEST,
        steps=7,
        target_steps=8,
        seeds=seeds,
        encoder_identity_hash=_ENCODER_WEIGHTS_HASH,
        commit_sha=_RESUME_COMMIT,
        source_tree=_RESUME_TREE,
        package_version="0.2.1",
        consumed_window_ids=consumed,
        metric_history=history,
        collapse_alert_count=1,
        membership_identity=membership_identity,
    )

    payload = load_resume_checkpoint(path)
    checkpoint = _validate_resume_checkpoint_payload(
        payload,
        path=path,
        **_resume_validation_args(
            config,
            seeds,
            target_steps=8,
            membership_identity=membership_identity,
        ),
    )

    assert checkpoint.steps_completed == 7
    assert set(payload["states"]) == {"predictor", "action_encoder", "optimizer"}
    assert set(payload["state_digests"]) == {
        "predictor",
        "action_encoder",
        "optimizer",
        "trainer",
    }
    assert set(payload["rng_state"]) == {"python", "numpy", "torch_cpu", "torch_cuda"}
    assert payload["training_contract"]["config"] == config_to_dict(config)
    assert payload["progress"]["consumed_window_ids"] == consumed


def _resume_payload(
    tmp_path: Path,
    config,
    seeds: TrainerSeeds,
    *,
    steps_completed: int,
    source: dict[str, object] | None = None,
    training_contract: dict[str, object] | None = None,
    identities: dict[str, object] | None = None,
) -> dict[str, object]:
    torch = pytest.importorskip("torch")
    consumed = [f"window-{index}" for index in range(steps_completed * config.data.batch_size)]
    path = tmp_path / "fixture-resume-checkpoint.pt"
    write_resume_checkpoint(
        path,
        source=source
        or {
            "commit_sha": _RESUME_COMMIT,
            "tree_sha": _RESUME_TREE,
            "package_version": "0.2.1",
        },
        training_contract=training_contract or _resume_training_contract(config, seeds),
        identities=identities or _resume_identities(),
        progress={
            "steps_completed": steps_completed,
            "samples_consumed": len(consumed),
            "consumed_window_ids": consumed,
            "collapse_alert_count": 0,
        },
        states={
            "predictor": {"weight": torch.tensor([1.0])},
            "action_encoder": {"weight": torch.tensor([2.0])},
            "optimizer": {"state": {}, "param_groups": []},
        },
        trainer_state={"schema_version": "fixture", "total_steps": 5},
        rng_state=capture_rng_state(),
        metric_history=[
            {
                **_step_result(loss=0.5, var=0.2, step=step).to_dict(),
                "lr_multiplier": wsd_lr_multiplier(
                    step,
                    total_steps=5,
                    warmup_steps=config.optimizer.warmup_steps,
                    schedule=config.optimizer.schedule,
                ),
            }
            for step in range(1, steps_completed + 1)
        ],
    )
    return load_resume_checkpoint(path)


def _resume_training_contract(
    config,
    seeds: TrainerSeeds,
    *,
    target_steps: int = 5,
) -> dict[str, object]:
    return {
        "target_steps": target_steps,
        "batch_size": config.data.batch_size,
        "config": config_to_dict(config),
        "seeds": seeds.to_dict(),
    }


def _resume_identities(
    membership_identity: dict[str, object] | None = None,
) -> dict[str, object]:
    return {
        "dataset_snapshot_id": "geno-lewm-data-v0.1.0-r1",
        "dataset_manifest": canonical_json_sha256(_RESUME_DATASET_MANIFEST),
        "encoder": _ENCODER_WEIGHTS_HASH,
        "membership_and_split": (
            None if membership_identity is None else canonical_json_sha256(membership_identity)
        ),
    }


def _resume_validation_args(
    config,
    seeds: TrainerSeeds,
    *,
    target_steps: int = 5,
    membership_identity: dict[str, object] | None = None,
) -> dict[str, object]:
    return {
        "config": config,
        "dataset_snapshot_id": "geno-lewm-data-v0.1.0-r1",
        "seeds": seeds,
        "target_steps": target_steps,
        "encoder_identity_hash": _ENCODER_WEIGHTS_HASH,
        "dataset_manifest": _RESUME_DATASET_MANIFEST,
        "commit_sha": _RESUME_COMMIT,
        "source_tree": _RESUME_TREE,
        "package_version": "0.2.1",
        "membership_identity": membership_identity,
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
