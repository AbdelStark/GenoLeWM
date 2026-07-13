"""Tests for the correction-control training postflight receipt."""

from __future__ import annotations

import json
import math
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest

from geno_lewm.config import config_to_dict, load_config, write_resolved_config
from geno_lewm.provenance import sha256_file
from tools.research import correction_control_postflight as postflight
from tools.research.state_contract_audit import (
    DEFAULT_CARBON_REVISION,
    DEFAULT_CARBON_RUNTIME_HASH,
    DEFAULT_CARBON_WEIGHTS_HASH,
    GENERATED_BY as STATE_CONTRACT_AUDIT_GENERATED_BY,
    SCHEMA_VERSION as STATE_CONTRACT_AUDIT_SCHEMA_VERSION,
)

_COMMIT = "a" * 40
_RUN_ID = "correction-control-l2-p1-smoke-v1"
_SNAPSHOT_ID = "geno-lewm-data-correction-control-l2-p1-proof-v1"
_RUN_NAME = f"geno-lewm-l2-p1-smoke-{_COMMIT[:12]}-50-r2"


class _ScalarStub:
    def __init__(self, value: bool) -> None:
        self._value = value

    def item(self) -> bool:
        return self._value


class _FiniteMaskStub:
    def __init__(self, values: list[float]) -> None:
        self._values = values

    def all(self) -> _ScalarStub:
        return _ScalarStub(all(math.isfinite(value) for value in self._values))


class _TensorStub:
    """Minimal tensor protocol used by export and postflight contract tests."""

    def __init__(self, values: list[float], *, dtype: str = "torch.float32") -> None:
        self._values = values
        self.dtype = dtype

    def __setitem__(self, index: int, value: float) -> None:
        self._values[index] = value

    def detach(self) -> _TensorStub:
        return self

    def cpu(self) -> _TensorStub:
        return self

    def contiguous(self) -> _TensorStub:
        return self

    def numel(self) -> int:
        return len(self._values)

    def isfinite(self) -> _FiniteMaskStub:
        return _FiniteMaskStub(self._values)

    def is_floating_point(self) -> bool:
        return True


def test_correction_control_postflight_accepts_coherent_smoke_receipt(tmp_path: Path) -> None:
    request, checkpoint = _valid_request(tmp_path)

    report = postflight.build_correction_control_postflight_report(
        request,
        checkpoint_loader=lambda _path: checkpoint,
        generated_at="2026-07-10T00:00:00Z",
    )

    assert report["ok"] is True
    assert report["blockers"] == []
    assert report["expected"] == {
        "commit_sha": _COMMIT,
        "run_id": _RUN_ID,
        "dataset_snapshot_id": _SNAPSHOT_ID,
        "steps_completed": 50,
        "sample_count": 400,
        "phase": "phase1",
        "state_contract_version": "l2_normalized_v2",
        "encoder_runtime_hash": DEFAULT_CARBON_RUNTIME_HASH,
    }
    claim_boundary = report["claim_boundary"]
    assert isinstance(claim_boundary, str)
    assert "does not establish convergence" in claim_boundary
    assert "not proof of training-source parity" in claim_boundary
    assert str(tmp_path) not in json.dumps(report, sort_keys=True)
    job_preflight = _read_json(request.job_contract_preflight_json)
    training_preflight = _read_json(request.training_preflight_report_json)
    assert job_preflight["config"]["path"].startswith("configs/correction_control/")
    assert training_preflight["training_config"]["path"] == request.training_config.name
    assert job_preflight["config"]["sha256"] != training_preflight["training_config"]["sha256"]
    evidence_paths = {
        "job_contract_preflight": request.job_contract_preflight_json,
        "source_identity_report": request.source_identity_report_json,
        "dataset_manifest": request.dataset_manifest_json,
        "dataset_snapshot_report": request.dataset_snapshot_report_json,
        "training_preflight_report": request.training_preflight_report_json,
        "tuple_throughput_report": request.tuple_throughput_report_json,
    }
    for name, path in evidence_paths.items():
        identity = report["artifacts"][name]
        assert identity["exists"] is True
        assert identity["sha256"] == sha256_file(path)


def test_correction_control_postflight_rejects_bad_counts_and_nonfinite_history(
    tmp_path: Path,
) -> None:
    request, checkpoint = _valid_request(tmp_path)
    metrics = _read_json(request.metrics_json)
    metrics["steps_completed"] = 49
    metrics["sample_count"] = 399
    metrics["metrics"]["nan_loss_count"] = 1
    metrics["metrics"]["collapse_alert_count"] = 1
    metrics["history"][12]["loss"] = float("nan")
    metrics["history"] = metrics["history"][:-1]
    _write_json(request.metrics_json, metrics)
    _refresh_identity(request.training_run_json, "metrics", request.metrics_json)

    report = postflight.build_correction_control_postflight_report(
        request,
        checkpoint_loader=lambda _path: checkpoint,
    )

    assert report["ok"] is False
    codes = _blocker_codes(report)
    assert {
        "metrics.steps_completed_mismatch",
        "metrics.sample_count_mismatch",
        "metrics.metrics.nan_loss_count_mismatch",
        "metrics.metrics.collapse_alert_count_mismatch",
        "metrics.history[12].loss_nonfinite",
        "metrics.history_length_mismatch",
        "artifacts.coherence_mismatch",
    } <= codes


def test_correction_control_postflight_rejects_config_and_runtime_identity_drift(
    tmp_path: Path,
) -> None:
    request, checkpoint = _valid_request(tmp_path)
    config_text = request.training_config.read_text(encoding="utf-8")
    request.training_config.write_text(
        config_text.replace("phase: phase1", "phase: phase2"),
        encoding="utf-8",
    )
    _refresh_identity(request.training_run_json, "training_config", request.training_config)
    checkpoint_config = checkpoint["config"]
    checkpoint_config["encoder.identity_hash"] = "sha256:" + ("0" * 64)
    audit = _read_json(request.state_contract_audit_json)
    audit["commit_sha"] = "b" * 40
    audit["encoder"]["runtime_hash"] = "sha256:" + ("1" * 64)
    _write_json(request.state_contract_audit_json, audit)

    report = postflight.build_correction_control_postflight_report(
        request,
        checkpoint_loader=lambda _path: checkpoint,
    )

    assert report["ok"] is False
    codes = _blocker_codes(report)
    assert "training_config.phase_mismatch" in codes
    assert "checkpoint.config.encoder.identity_hash_mismatch" in codes
    assert "state_contract_audit.commit_sha_mismatch" in codes
    assert "state_contract_audit.encoder.runtime_hash_mismatch" in codes
    assert "artifacts.coherence_mismatch" in codes


def test_correction_control_postflight_rejects_artifact_hash_drift(tmp_path: Path) -> None:
    request, checkpoint = _valid_request(tmp_path)
    request.checkpoint.write_bytes(b"modified after training metadata was written")

    report = postflight.build_correction_control_postflight_report(
        request,
        checkpoint_loader=lambda _path: checkpoint,
    )

    assert report["ok"] is False
    assert "training_run.artifact_identities.checkpoint_files[0].sha256_mismatch" in _blocker_codes(
        report
    )


def test_correction_control_postflight_rejects_source_to_snapshot_hash_drift(
    tmp_path: Path,
) -> None:
    request, checkpoint = _valid_request(tmp_path)
    snapshot = _read_json(request.dataset_snapshot_report_json)
    snapshot["files"][1]["source_sha256"] = "sha256:" + ("f" * 64)
    _write_json(request.dataset_snapshot_report_json, snapshot)
    training_preflight = _read_json(request.training_preflight_report_json)
    training_preflight["dataset"]["core_files"][request.dataset_snapshot_report_json.name] = (
        _identity(request.dataset_snapshot_report_json)
    )
    _write_json(request.training_preflight_report_json, training_preflight)
    _refresh_identity(
        request.training_run_json,
        "training_preflight_report",
        request.training_preflight_report_json,
    )

    report = postflight.build_correction_control_postflight_report(
        request,
        checkpoint_loader=lambda _path: checkpoint,
    )

    assert report["ok"] is False
    assert "artifacts.coherence_mismatch" in _blocker_codes(report)


def test_correction_control_postflight_rejects_corrupt_final_optimizer_state(
    tmp_path: Path,
) -> None:
    request, checkpoint = _valid_request(tmp_path)
    checkpoint["predictor"]["weight"][0] = float("nan")
    checkpoint["action_encoder"] = {"weight": _TensorStub([0.25], dtype="torch.float16")}
    checkpoint["optimizer"]["state"]["0"]["step"] = 49
    metrics = _read_json(request.metrics_json)
    metrics["history"][0]["action_count"] = 7
    _write_json(request.metrics_json, metrics)
    _refresh_identity(request.training_run_json, "metrics", request.metrics_json)

    report = postflight.build_correction_control_postflight_report(
        request,
        checkpoint_loader=lambda _path: checkpoint,
    )

    assert report["ok"] is False
    codes = _blocker_codes(report)
    assert "checkpoint.predictor.weight_nonfinite" in codes
    assert "checkpoint.action_encoder.weight_dtype_mismatch" in codes
    assert "checkpoint.optimizer.step_mismatch" in codes
    assert "metrics.history[0].action_count_mismatch" in codes


def test_correction_control_postflight_rejects_non_tensor_model_state(tmp_path: Path) -> None:
    request, checkpoint = _valid_request(tmp_path)
    checkpoint["predictor"] = {"weight": [1.0, -0.5]}

    report = postflight.build_correction_control_postflight_report(
        request,
        checkpoint_loader=lambda _path: checkpoint,
    )

    assert report["ok"] is False
    assert "checkpoint.predictor.weight_not_tensor" in _blocker_codes(report)


def test_correction_control_postflight_cli_writes_failure_receipt(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    request, checkpoint = _valid_request(tmp_path)
    metrics = _read_json(request.metrics_json)
    metrics["metrics"]["nan_loss_count"] = 2
    _write_json(request.metrics_json, metrics)
    _refresh_identity(request.training_run_json, "metrics", request.metrics_json)
    monkeypatch.setattr(postflight, "_load_torch_checkpoint", lambda _path: checkpoint)

    result = postflight.main(_argv(request))

    assert result == 1
    receipt = _read_json(request.output_json)
    assert receipt["ok"] is False
    assert "metrics.metrics.nan_loss_count_mismatch" in _blocker_codes(receipt)


def test_correction_control_postflight_reports_malformed_inputs(tmp_path: Path) -> None:
    request, _checkpoint = _valid_request(tmp_path)
    request.training_run_json.unlink()
    request.metrics_json.write_text("{not-json\n", encoding="utf-8")
    request.training_config.write_text("- not-an-object\n", encoding="utf-8")
    request.state_contract_audit_json.write_text("[]\n", encoding="utf-8")
    request = replace(
        request,
        expected_commit_sha="ABC",
        expected_run_id=" ",
        expected_dataset_snapshot_id="",
    )

    def fail_checkpoint_load(_path: Path) -> dict[str, Any]:
        raise RuntimeError("fixture checkpoint load failure")

    report = postflight.build_correction_control_postflight_report(
        request,
        checkpoint_loader=fail_checkpoint_load,
    )

    assert report["ok"] is False
    assert {
        "expected.commit_sha_invalid",
        "expected.run_id_invalid",
        "expected.dataset_snapshot_id_invalid",
        "training_run.unreadable",
        "metrics.invalid_json",
        "training_config.invalid",
        "state_contract_audit.invalid_type",
        "checkpoint.unreadable",
    } <= _blocker_codes(report)


def test_correction_control_postflight_rejects_malformed_artifact_contracts(
    tmp_path: Path,
) -> None:
    request, checkpoint = _valid_request(tmp_path)
    training_run = _read_json(request.training_run_json)
    training_run["training_config"] = ""
    training_run["metrics"] = "../metrics.json"
    training_run["checkpoint_files"] = []
    training_run["artifact_identities"] = {
        "training_config": [],
        "metrics": {},
        "checkpoint_files": [],
    }
    _write_json(request.training_run_json, training_run)
    metrics = _read_json(request.metrics_json)
    metrics["train_loss"] = "not-numeric"
    metrics["metrics"] = []
    metrics["history"] = "not-a-list"
    _write_json(request.metrics_json, metrics)
    checkpoint["config"] = []
    audit = _read_json(request.state_contract_audit_json)
    audit["encoder"] = []
    _write_json(request.state_contract_audit_json, audit)

    report = postflight.build_correction_control_postflight_report(
        request,
        checkpoint_loader=lambda _path: checkpoint,
    )

    assert report["ok"] is False
    assert {
        "training_run.training_config_declaration_invalid",
        "training_run.metrics.nonportable",
        "training_run.checkpoint_declaration_invalid",
        "training_run.artifact_identities.training_config.invalid",
        "training_run.checkpoint_identity_invalid",
        "metrics.train_loss_invalid",
        "metrics.metrics_invalid",
        "metrics.history_invalid",
        "checkpoint.config_invalid",
        "state_contract_audit.encoder_invalid",
    } <= _blocker_codes(report)


def test_correction_control_postflight_rejects_failed_audit_rows(tmp_path: Path) -> None:
    request, checkpoint = _valid_request(tmp_path)
    audit = _read_json(request.state_contract_audit_json)
    audit["ok"] = False
    audit["blockers"] = [{"code": "normalization_failed"}]
    audit["encoder"]["runtime_hash"] = "not-a-sha256"
    audit["rows"] = [{"index": 0, "ok": False}]
    _write_json(request.state_contract_audit_json, audit)

    report = postflight.build_correction_control_postflight_report(
        request,
        checkpoint_loader=lambda _path: checkpoint,
    )

    assert report["ok"] is False
    assert {
        "state_contract_audit.ok_mismatch",
        "state_contract_audit.blockers_mismatch",
        "state_contract_audit.encoder.runtime_hash_invalid",
        "state_contract_audit.rows_failed",
    } <= _blocker_codes(report)


def test_correction_control_postflight_loads_checkpoint_with_weights_only(
    tmp_path: Path,
) -> None:
    torch = pytest.importorskip("torch")
    path = tmp_path / "checkpoint.pt"
    torch.save({"state": {"weight": torch.ones(2)}}, path)

    payload = postflight._load_torch_checkpoint(path)

    assert payload["state"]["weight"].tolist() == [1.0, 1.0]


def test_correction_control_postflight_accepts_real_torch_checkpoint(tmp_path: Path) -> None:
    torch = pytest.importorskip("torch")
    request, checkpoint = _valid_request(tmp_path)
    checkpoint["predictor"] = {"weight": torch.tensor([1.0, -0.5])}
    checkpoint["action_encoder"] = {"weight": torch.tensor([0.25, 0.75])}
    optimizer_state = checkpoint["optimizer"]["state"]["0"]
    optimizer_state["step"] = torch.tensor(50.0)
    optimizer_state["exp_avg"] = torch.tensor([0.1, 0.2])
    optimizer_state["exp_avg_sq"] = torch.tensor([0.01, 0.04])
    torch.save(checkpoint, request.checkpoint)
    training_run = _read_json(request.training_run_json)
    training_run["artifact_identities"]["checkpoint_files"] = [_identity(request.checkpoint)]
    _write_json(request.training_run_json, training_run)

    report = postflight.build_correction_control_postflight_report(request)

    assert report["ok"] is True
    assert report["blockers"] == []


def _valid_request(
    tmp_path: Path,
) -> tuple[postflight.CorrectionControlPostflightRequest, dict[str, Any]]:
    source_config_path = tmp_path / "train-carbon-500m-snv-l2-smoke-v1.yaml"
    source_config_path.write_text(
        f"""\
run_id: {_RUN_ID}
seed: 104729
phase: phase1
deterministic: true
schema_version: 1.1.0
encoder:
  model_id: /carbon
  revision: {DEFAULT_CARBON_REVISION}
  dtype: bf16
  state_layer: 20
  pool_type: centered_mean
  pool_radius: 8
  normalize: true
  state_contract_version: l2_normalized_v2
  trust_remote_code: false
predictor:
  d_state: 1024
  dtype: fp32
training:
  max_steps: 50
  collapse_log_every_steps: 10
data:
  corpus_id: HuggingFaceBio/carbon-pretraining-corpus
  corpus_revision: cb4c13a78102933b3a6ac65734d326f7b431d9b7
  batch_size: 8
  num_workers: 0
  shuffle_buffer: 0
action:
  sub_encoders:
    - snv
optimizer:
  lr: 3.0e-5
  warmup_steps: 10
  schedule: wsd
eval:
  smoke_variants: 100
runtime:
  backend: torch
  device: cuda
""",
        encoding="utf-8",
    )
    config_path = tmp_path / "training_config.effective.yaml"
    write_resolved_config(load_config(source_config_path), config_path)
    metrics_path = tmp_path / "metrics.json"
    history = [
        {
            "step": step,
            "lr_multiplier": min(step / 10.0, 1.0),
            "loss": 1.0 / step,
            "pred_loss": 1.0 / step,
            "kl_reg": 0.0,
            "action_count": 8,
            "pred_var_per_dim": 0.25,
        }
        for step in range(1, 51)
    ]
    metrics = {
        "schema_version": "1.0.0",
        "run_id": _RUN_ID,
        "dataset_snapshot_id": _SNAPSHOT_ID,
        "steps_completed": 50,
        "resumed_from_step": 0,
        "resume_checkpoint": None,
        "sample_count": 400,
        "new_sample_count": 400,
        "elapsed_seconds": 12.0,
        "samples_per_second": 400 / 12,
        "train_loss": history[-1]["loss"],
        "metrics": {
            "train_loss": history[-1]["loss"],
            "sample_count": 400,
            "new_sample_count": 400,
            "resumed_from_step": 0,
            "nan_loss_count": 0,
            "collapse_var_min": {"value": 0.25},
            "collapse_alert_count": 0,
        },
        "history": history,
    }
    _write_json(metrics_path, metrics)

    checkpoint_path = tmp_path / "predictor_checkpoint.pt"
    checkpoint_path.write_bytes(b"fixture checkpoint identity")
    checkpoint = {
        "schema_version": "1.0.0",
        "run_id": _RUN_ID,
        "dataset_snapshot_id": _SNAPSHOT_ID,
        "steps_completed": 50,
        "seeds": {"data": 104729, "predictor": 104730, "lora": 104731},
        "predictor": {"weight": _TensorStub([1.0, -0.5])},
        "action_encoder": {"weight": _TensorStub([0.25, 0.75])},
        "optimizer": {
            "state": {
                "0": {
                    "step": 50,
                    "exp_avg": [0.1, 0.2],
                    "exp_avg_sq": [0.01, 0.04],
                }
            },
            "param_groups": [{"params": [0]}],
        },
        "config": {
            "run_id": _RUN_ID,
            "seed": 104729,
            "deterministic": True,
            "data.batch_size": 8,
            "predictor.d_state": 1024,
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
        },
    }

    audit_path = tmp_path / "state_contract_audit.json"
    audit = {
        "schema_version": STATE_CONTRACT_AUDIT_SCHEMA_VERSION,
        "generated_by": STATE_CONTRACT_AUDIT_GENERATED_BY,
        "generated_at": "2026-07-10T00:00:00Z",
        "ok": True,
        "commit_sha": _COMMIT,
        "encoder": {
            "revision": DEFAULT_CARBON_REVISION,
            "weights_hash": DEFAULT_CARBON_WEIGHTS_HASH,
            "expected_weights_hash": DEFAULT_CARBON_WEIGHTS_HASH,
            "weights_identity_verified": True,
            "runtime_hash": DEFAULT_CARBON_RUNTIME_HASH,
            "expected_runtime_hash": DEFAULT_CARBON_RUNTIME_HASH,
            "runtime_identity_verified": True,
            "parameters_frozen": True,
            "expected_d_state": 1024,
            "window_bp": 4096,
            "state_layer": 20,
            "pool_type": "centered_mean",
            "pool_radius": 8,
            "pooling_identity_verified": True,
            "dtype": "bf16",
            "normalized_state_contract": "l2_normalized_v2",
        },
        "runtime": {
            "device": "cuda",
            "hf_hub_offline": True,
            "transformers_offline": True,
            "cuda_available": True,
            "cuda_device_name": "NVIDIA H200",
        },
        "rows": [{"index": 0, "ok": True}],
        "blockers": [],
    }
    _write_json(audit_path, audit)

    carbon_identity = _reported_identity("carbon/source-mix-windows.jsonl", "1", 4096)
    gnomad_identity = _reported_identity("gnomad/v4.1/variants.parquet", "2", 2048)
    clinvar_identity = _reported_identity("clinvar/2026-04-15/variants.parquet", "3", 1024)
    manifest_path = tmp_path / "dataset_manifest.json"
    manifest = {
        "schema_version": "1.0.0",
        "generated_by": "tools.release.dataset_package",
        "generated_at": "2026-07-10T00:00:00Z",
        "snapshot_id": _SNAPSHOT_ID,
        "sources": [
            {
                "name": "Carbon pretraining corpus",
                "revision": "cb4c13a78102933b3a6ac65734d326f7b431d9b7",
                "url": "https://huggingface.co/datasets/HuggingFaceBio/carbon-pretraining-corpus",
            },
            {
                "name": "gnomAD",
                "revision": "v4.1 chr22 generation 1713312296186865",
                "url": (
                    "https://storage.googleapis.com/gcp-public-data--gnomad/release/4.1/"
                    "vcf/exomes/gnomad.exomes.v4.1.sites.chr22.vcf.bgz?"
                    "generation=1713312296186865"
                ),
            },
            {
                "name": "ClinVar",
                "revision": "2026-04-15 md5:e63b5c3a046010c098cc70e81bebaa8d",
                "url": (
                    "https://ftp.ncbi.nlm.nih.gov/pub/clinvar/vcf_GRCh38/"
                    "archive_2.0/2026/clinvar_20260415.vcf.gz"
                ),
            },
        ],
        "splits": {
            "train_carbon": {"records": 512},
            "train_gnomad_common": {"records": 37},
            "eval_clinvar": {"records": 19},
        },
        "files": [
            {**carbon_identity, "split": "train_carbon", "records": 512},
            {**gnomad_identity, "split": "train_gnomad_common", "records": 37},
            {**clinvar_identity, "split": "eval_clinvar", "records": 19},
        ],
    }
    _write_json(manifest_path, manifest)

    dataset_input_check_path = tmp_path / "dataset_input_check_report.json"
    _write_json(
        dataset_input_check_path,
        {
            "schema_version": "1.0.0",
            "generated_by": "tools.release.dataset_snapshot.check_inputs",
            "ok": True,
            "snapshot_id": _SNAPSHOT_ID,
        },
    )
    snapshot_spec_identity = _reported_identity("dataset-snapshot-snv.json", "4", 8192)
    snapshot_report_path = tmp_path / "dataset_snapshot_report.json"
    snapshot_report = {
        "schema_version": "1.0.0",
        "generated_by": "tools.release.dataset_snapshot",
        "generated_at": "2026-07-10T00:00:00Z",
        "snapshot_id": _SNAPSHOT_ID,
        "report_path": snapshot_report_path.name,
        "snapshot_spec": snapshot_spec_identity,
        "input_check_path": dataset_input_check_path.name,
        "input_check": _identity(dataset_input_check_path),
        "metadata_path": "dataset_package.json",
        "package": {"manifest": _identity(manifest_path)},
        "files": [
            {
                **carbon_identity,
                "kind": "carbon",
                "source_path": "inputs/carbon/source-mix-windows.jsonl",
                "source_sha256": carbon_identity["sha256"],
                "source_size_bytes": carbon_identity["size_bytes"],
                "split": "train_carbon",
                "records": 512,
            },
            {
                **gnomad_identity,
                "kind": "gnomad",
                "source_path": "inputs/gnomad/gnomad-v4.1-snv.vcf.gz",
                "source_sha256": "sha256:" + ("7" * 64),
                "source_size_bytes": 100_000,
                "split": "train_gnomad_common",
                "records": 37,
            },
            {
                **clinvar_identity,
                "kind": "clinvar",
                "source_path": "inputs/clinvar/clinvar-2026-04-15-snv.vcf.gz",
                "source_sha256": "sha256:" + ("6" * 64),
                "source_size_bytes": 90_000,
                "split": "eval_clinvar",
                "records": 19,
            },
        ],
    }
    _write_json(snapshot_report_path, snapshot_report)

    job_preflight_path = tmp_path / "job_contract_preflight.json"
    config_identity = _identity(config_path)
    source_config_identity = _identity(source_config_path)
    source_config_identity["path"] = (
        "configs/correction_control/train-carbon-500m-snv-l2-smoke-v1.yaml"
    )
    job_preflight = {
        "schema_version": "1.0.0",
        "generated_by": "tools.research.correction_control_preflight",
        "generated_at": "2026-07-10T00:00:00Z",
        "ok": True,
        "repository": {
            "root": ".",
            "expected_commit_sha": _COMMIT,
            "observed_commit_sha": _COMMIT,
            "observed_git_root": ".",
            "worktree_clean": True,
            "dirty_paths": [],
        },
        "job": {
            "run_name": _RUN_NAME,
            "run_attempt": 2,
            "steps": 50,
            "max_windows": 512,
            "clinvar_lines": 60_000,
            "gnomad_lines": 60_000,
            "tuple_throughput_samples": 400,
            "window_bp": 4096,
            "holdout_chrom": 22,
            "carbon_model_dir": "/carbon",
            "carbon_config": "eukaryote_generator_10B_subset",
            "carbon_source": "eukaryotic_genes",
            "corpus_revision": "cb4c13a78102933b3a6ac65734d326f7b431d9b7",
            "container_image": (
                "ghcr.io/astral-sh/uv@sha256:"
                "35b0aa516fbcf6f18624919cfc38fa02ab3458e0ffcd3c03e932051b37f315db"
            ),
            "sources": {
                "clinvar_url": (
                    "https://ftp.ncbi.nlm.nih.gov/pub/clinvar/vcf_GRCh38/"
                    "archive_2.0/2026/clinvar_20260415.vcf.gz"
                ),
                "gnomad_url": (
                    "https://storage.googleapis.com/gcp-public-data--gnomad/release/4.1/"
                    "vcf/exomes/gnomad.exomes.v4.1.sites.chr22.vcf.bgz?"
                    "generation=1713312296186865"
                ),
            },
        },
        "config": {
            **source_config_identity,
            "exists": True,
            "run_id": _RUN_ID,
            "schema_version": "1.1.0",
        },
        "snapshot": {
            **snapshot_spec_identity,
            "path": "configs/correction_control/dataset-snapshot-snv-l2-smoke-v1.json",
            "exists": True,
            "snapshot_id": _SNAPSHOT_ID,
            "schema_version": "1.0.0",
        },
        "issues": [],
    }
    _write_json(job_preflight_path, job_preflight)

    source_identity_path = tmp_path / "source_identity_report.json"
    source_identity = {
        "schema_version": "1.0.0",
        "generated_by": "tools.jobs.proof_run.source_identity",
        "generated_at": "2026-07-10T00:00:00Z",
        "ok": True,
        "commit_sha": _COMMIT,
        "run_name": _RUN_NAME,
        "dataset_snapshot_id": _SNAPSHOT_ID,
        "training_contract": {
            "active_window_source": "carbon",
            "window_bp": 4096,
            "action_sub_encoders": ["snv"],
            "actions_per_window": 8,
            "absolute_variant_fallback": "synthetic_snv",
        },
        "sources": {
            "carbon_corpus": {
                "revision": "cb4c13a78102933b3a6ac65734d326f7b431d9b7",
                "dataset_config": "eukaryote_generator_10B_subset",
                "default_source": "eukaryotic_genes",
                "windows": 512,
                "artifact": {**carbon_identity, "path": "source-mix-windows.jsonl"},
            },
            "clinvar": {
                "url": (
                    "https://ftp.ncbi.nlm.nih.gov/pub/clinvar/vcf_GRCh38/"
                    "archive_2.0/2026/clinvar_20260415.vcf.gz"
                ),
                "md5": "e63b5c3a046010c098cc70e81bebaa8d",
                "subset_lines": 60_000,
                "archive": _reported_identity("clinvar_20260415.vcf.gz", "5", 190_691_010),
                "filtered_artifact": _reported_identity(
                    "clinvar-2026-04-15-snv.vcf.gz", "6", 90_000
                ),
            },
            "gnomad": {
                "url": (
                    "https://storage.googleapis.com/gcp-public-data--gnomad/release/4.1/"
                    "vcf/exomes/gnomad.exomes.v4.1.sites.chr22.vcf.bgz?"
                    "generation=1713312296186865"
                ),
                "generation": "1713312296186865",
                "md5": "dcf191563e69054a71bd4dc77862799a",
                "size_bytes": 5_060_347_554,
                "subset_lines": 60_000,
                "subset_artifact": _reported_identity("gnomad-v4.1-snv.vcf.gz", "7", 100_000),
            },
        },
    }
    _write_json(source_identity_path, source_identity)

    training_preflight_path = tmp_path / "training_preflight_report.json"
    training_preflight = {
        "schema_version": "1.0.0",
        "generated_by": "geno_lewm.training.preflight",
        "generated_at": "2026-07-10T00:00:00Z",
        "ok": True,
        "dataset_snapshot_id": _SNAPSHOT_ID,
        "training_config": {
            **config_identity,
            "resolved": config_to_dict(load_config(config_path)),
        },
        "dataset": {
            "snapshot_id": _SNAPSHOT_ID,
            "core_files": {
                "dataset_manifest.json": _identity(manifest_path),
                dataset_input_check_path.name: _identity(dataset_input_check_path),
                snapshot_report_path.name: _identity(snapshot_report_path),
            },
            "files": manifest["files"],
            "splits": manifest["splits"],
        },
        "accelerator": {
            "requested_device": "cuda",
            "required": True,
            "available": True,
            "device_count": 1,
            "device_name": "NVIDIA H200",
            "total_memory_bytes": 141 * 1024**3,
            "min_memory_bytes": 120 * 1024**3,
        },
        "issues": [],
    }
    _write_json(training_preflight_path, training_preflight)

    tuple_throughput_path = tmp_path / "tuple_throughput_report.json"
    tuple_throughput = {
        "schema_version": "1.0.0",
        "generated_by": "tools.data.tuple_throughput",
        "dataset_snapshot_id": _SNAPSHOT_ID,
        "dataset_manifest": _identity(manifest_path),
        "seed": 0,
        "requested_samples": 400,
        "samples": 400,
        "elapsed_seconds": 0.05,
        "tuples_per_second": 8000.0,
        "windows": 512,
        "gnomad_edits": 37,
        "clinvar_edits": 19,
        "min_tuples_per_second": 5000.0,
        "passed_min_tuples_per_second": True,
    }
    _write_json(tuple_throughput_path, tuple_throughput)

    training_run_path = tmp_path / "training_run.json"
    training_run = {
        "schema_version": "1.0.0",
        "run_id": _RUN_ID,
        "generated_by": "tools.release.training_run",
        "commit_sha": _COMMIT,
        "dataset_snapshot_id": _SNAPSHOT_ID,
        "dataset_manifest": manifest_path.name,
        "training_config": config_path.name,
        "metrics": metrics_path.name,
        "training_preflight_report": training_preflight_path.name,
        "checkpoint_files": [checkpoint_path.name],
        "artifact_identities": {
            "training_config": _identity(config_path),
            "metrics": _identity(metrics_path),
            "dataset_manifest": _identity(manifest_path),
            "training_preflight_report": _identity(training_preflight_path),
            "checkpoint_files": [_identity(checkpoint_path)],
        },
        "status": "completed",
        "resumed_from_step": 0,
        "resume_checkpoint": None,
    }
    _write_json(training_run_path, training_run)

    request = postflight.CorrectionControlPostflightRequest(
        training_run_json=training_run_path,
        metrics_json=metrics_path,
        training_config=config_path,
        checkpoint=checkpoint_path,
        state_contract_audit_json=audit_path,
        job_contract_preflight_json=job_preflight_path,
        source_identity_report_json=source_identity_path,
        dataset_manifest_json=manifest_path,
        dataset_snapshot_report_json=snapshot_report_path,
        training_preflight_report_json=training_preflight_path,
        tuple_throughput_report_json=tuple_throughput_path,
        expected_commit_sha=_COMMIT,
        expected_run_id=_RUN_ID,
        expected_dataset_snapshot_id=_SNAPSHOT_ID,
        output_json=tmp_path / "correction_control_postflight.json",
    )
    return request, checkpoint


def _refresh_identity(training_run_path: Path, field: str, artifact: Path) -> None:
    payload = _read_json(training_run_path)
    payload["artifact_identities"][field] = _identity(artifact)
    _write_json(training_run_path, payload)


def _identity(path: Path) -> dict[str, object]:
    return {
        "path": path.name,
        "sha256": sha256_file(path),
        "size_bytes": path.stat().st_size,
    }


def _reported_identity(path: str, digest_char: str, size_bytes: int) -> dict[str, object]:
    return {
        "path": path,
        "sha256": "sha256:" + (digest_char * 64),
        "size_bytes": size_bytes,
    }


def _write_json(path: Path, payload: object) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert isinstance(payload, dict)
    return payload


def _blocker_codes(report: dict[str, Any]) -> set[str]:
    return {blocker["code"] for blocker in report["blockers"]}


def _argv(request: postflight.CorrectionControlPostflightRequest) -> list[str]:
    return [
        "--training-run-json",
        str(request.training_run_json),
        "--metrics-json",
        str(request.metrics_json),
        "--training-config",
        str(request.training_config),
        "--checkpoint",
        str(request.checkpoint),
        "--state-contract-audit-json",
        str(request.state_contract_audit_json),
        "--job-contract-preflight-json",
        str(request.job_contract_preflight_json),
        "--source-identity-report-json",
        str(request.source_identity_report_json),
        "--dataset-manifest-json",
        str(request.dataset_manifest_json),
        "--dataset-snapshot-report-json",
        str(request.dataset_snapshot_report_json),
        "--training-preflight-report-json",
        str(request.training_preflight_report_json),
        "--tuple-throughput-report-json",
        str(request.tuple_throughput_report_json),
        "--expected-commit-sha",
        request.expected_commit_sha,
        "--expected-run-id",
        request.expected_run_id,
        "--expected-dataset-snapshot-id",
        request.expected_dataset_snapshot_id,
        "--output-json",
        str(request.output_json),
    ]
