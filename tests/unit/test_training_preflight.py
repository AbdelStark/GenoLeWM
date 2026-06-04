"""Tests for Carbon-backed training preflight evidence."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from geno_lewm.config import load_config
from geno_lewm.provenance import sha256_file
from geno_lewm.training.preflight import (
    REPORT_NAME,
    DependencyProbe,
    TrainingPreflightRequest,
    build_training_preflight_report,
    write_training_preflight_report,
)
from tests.unit.test_release_dataset_snapshot import _write_spec
from tools.release.dataset_snapshot import build_dataset_snapshot


def test_training_preflight_accepts_packaged_dataset_and_local_carbon(tmp_path: Path) -> None:
    pytest.importorskip("pyarrow")
    dataset_dir = _write_release_dataset(tmp_path)
    carbon_dir = _write_carbon_model_dir(tmp_path)
    config = _write_training_config(tmp_path)

    report = build_training_preflight_report(
        TrainingPreflightRequest(
            dataset_dir=dataset_dir,
            carbon_model_dir=carbon_dir,
            training_config=config,
            run_dir=tmp_path / "run",
        ),
        generated_at="2026-06-01T12:00:00Z",
        dependency_probe=_available_dependency,
    )

    assert report.ok is True
    assert report.dataset_snapshot_id == "geno-lewm-data-v0.1.0-r1"
    assert report.carbon["artifacts"]["config"] is not None
    assert report.training_config["sha256"]
    assert isinstance(report.training_config["resolved"], dict)
    assert report.training_config["resolved"]["run_id"] == "first-snv-test"
    assert str(tmp_path) not in json.dumps(report.to_dict(), sort_keys=True)
    assert report.run_dir["preflight_report_path"] == REPORT_NAME
    assert report.dataset["core_files"]["dataset_package.json"] is not None
    assert report.dataset["core_files"]["dataset_input_check_report.json"] is not None
    assert report.dataset["core_files"]["dataset_snapshot_report.json"] is not None
    assert not report.issues


def test_training_preflight_writes_default_report_path(tmp_path: Path) -> None:
    pytest.importorskip("pyarrow")
    dataset_dir = _write_release_dataset(tmp_path)
    carbon_dir = _write_carbon_model_dir(tmp_path)
    config = _write_training_config(tmp_path)
    run_dir = tmp_path / "run"

    report = write_training_preflight_report(
        TrainingPreflightRequest(
            dataset_dir=dataset_dir,
            carbon_model_dir=carbon_dir,
            training_config=config,
            run_dir=run_dir,
        ),
        dependency_probe=_available_dependency,
    )

    assert report.ok is True
    payload = json.loads((run_dir / REPORT_NAME).read_text(encoding="utf-8"))
    assert payload["ok"] is True


def test_training_preflight_reports_missing_dependencies_and_carbon_files(tmp_path: Path) -> None:
    pytest.importorskip("pyarrow")
    dataset_dir = _write_release_dataset(tmp_path)
    carbon_dir = tmp_path / "carbon"
    carbon_dir.mkdir()
    config = _write_training_config(tmp_path)

    report = build_training_preflight_report(
        TrainingPreflightRequest(
            dataset_dir=dataset_dir,
            carbon_model_dir=carbon_dir,
            training_config=config,
            run_dir=tmp_path / "run",
        ),
        dependency_probe=_missing_dependency,
    )

    codes = _codes(report)
    assert report.ok is False
    assert "carbon.config_missing" in codes
    assert "carbon.tokenizer_missing" in codes
    assert "carbon.weights_missing" in codes
    assert "training.dependency_unavailable" in codes


def test_training_preflight_rejects_fixture_dataset_by_default(tmp_path: Path) -> None:
    pytest.importorskip("pyarrow")
    dataset_dir = _write_release_dataset(tmp_path, snapshot_id="geno-lewm-fixture-snapshot")
    carbon_dir = _write_carbon_model_dir(tmp_path)
    config = _write_training_config(tmp_path)

    report = build_training_preflight_report(
        TrainingPreflightRequest(
            dataset_dir=dataset_dir,
            carbon_model_dir=carbon_dir,
            training_config=config,
            run_dir=tmp_path / "run",
        ),
        dependency_probe=_available_dependency,
    )

    assert report.ok is False
    assert "dataset.fixture_snapshot" in _codes(report)


def test_training_preflight_requires_release_dataset_evidence(tmp_path: Path) -> None:
    pytest.importorskip("pyarrow")
    dataset_dir = _write_release_dataset(tmp_path)
    (dataset_dir / "dataset_package.json").unlink()
    (dataset_dir / "dataset_snapshot_report.json").unlink()
    carbon_dir = _write_carbon_model_dir(tmp_path)
    config = _write_training_config(tmp_path)

    report = build_training_preflight_report(
        TrainingPreflightRequest(
            dataset_dir=dataset_dir,
            carbon_model_dir=carbon_dir,
            training_config=config,
            run_dir=tmp_path / "run",
        ),
        dependency_probe=_available_dependency,
    )

    codes = _codes(report)
    assert report.ok is False
    assert "dataset.dataset_package.json.missing" in codes
    assert "dataset.dataset_snapshot_report.json.missing" in codes


def test_training_preflight_rejects_stale_dataset_input_check_report(
    tmp_path: Path,
) -> None:
    pytest.importorskip("pyarrow")
    dataset_dir = _write_release_dataset(tmp_path)
    input_check_path = dataset_dir / "dataset_input_check_report.json"
    input_check = json.loads(input_check_path.read_text(encoding="utf-8"))
    input_check["inputs"][0]["staged_path"] = "carbon/stale-window.jsonl"
    input_check_path.write_text(
        json.dumps(input_check, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    snapshot_report_path = dataset_dir / "dataset_snapshot_report.json"
    snapshot_report = json.loads(snapshot_report_path.read_text(encoding="utf-8"))
    snapshot_report["input_check"] = _dataset_artifact_identity(
        dataset_dir, "dataset_input_check_report.json"
    )
    snapshot_report_path.write_text(
        json.dumps(snapshot_report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    _refresh_dataset_sha256sums(dataset_dir)
    carbon_dir = _write_carbon_model_dir(tmp_path)
    config = _write_training_config(tmp_path)

    report = build_training_preflight_report(
        TrainingPreflightRequest(
            dataset_dir=dataset_dir,
            carbon_model_dir=carbon_dir,
            training_config=config,
            run_dir=tmp_path / "run",
        ),
        dependency_probe=_available_dependency,
    )

    codes = _codes(report)
    assert report.ok is False
    assert "dataset.input_check.stale" in codes
    assert "dataset.dataset_input_check_report.json.checksum_mismatch" not in codes
    assert "dataset.snapshot_report.input_check.stale" not in codes


def test_training_preflight_rejects_unknown_training_config_keys(tmp_path: Path) -> None:
    pytest.importorskip("pyarrow")
    dataset_dir = _write_release_dataset(tmp_path)
    carbon_dir = _write_carbon_model_dir(tmp_path)
    config = _write_training_config(tmp_path)
    config.write_text(
        config.read_text(encoding="utf-8") + "\ntraining:\n  unsupported: 100\n",
        encoding="utf-8",
    )

    report = build_training_preflight_report(
        TrainingPreflightRequest(
            dataset_dir=dataset_dir,
            carbon_model_dir=carbon_dir,
            training_config=config,
            run_dir=tmp_path / "run",
        ),
        dependency_probe=_available_dependency,
    )

    assert report.ok is False
    assert "training_config.schema_invalid" in _codes(report)


def test_training_preflight_rejects_wsd_warmup_without_decay_horizon(
    tmp_path: Path,
) -> None:
    pytest.importorskip("pyarrow")
    dataset_dir = _write_release_dataset(tmp_path)
    carbon_dir = _write_carbon_model_dir(tmp_path)
    config = _write_training_config(tmp_path)
    text = config.read_text(encoding="utf-8")
    text = text.replace("  max_steps: 2", "  max_steps: 10")
    text = text.replace("  warmup_steps: 0", "  warmup_steps: 10")
    config.write_text(text, encoding="utf-8")

    report = build_training_preflight_report(
        TrainingPreflightRequest(
            dataset_dir=dataset_dir,
            carbon_model_dir=carbon_dir,
            training_config=config,
            run_dir=tmp_path / "run",
        ),
        dependency_probe=_available_dependency,
    )

    assert report.ok is False
    assert "training_config.training.max_steps_wsd_warmup" in _codes(report)


def test_first_experiment_configs_are_checked_schema_configs() -> None:
    root = Path(__file__).resolve().parents[2]
    train_cfg = load_config(root / "configs/first_experiment/train-carbon-500m-snv.yaml")
    eval_cfg = load_config(root / "configs/first_experiment/eval-clinvar-snv.yaml")

    assert train_cfg.run_id == "first-snv-carbon-500m-r1"
    assert train_cfg.action.sub_encoders == ("snv",)
    assert train_cfg.deterministic is True
    assert train_cfg.training.max_steps == 20000
    assert train_cfg.optimizer.warmup_steps < train_cfg.training.max_steps
    assert eval_cfg.run_id == "first-snv-clinvar-eval-r1"
    assert "clinvar_coding" in eval_cfg.eval.benchmarks


def _write_release_dataset(root: Path, *, snapshot_id: str = "geno-lewm-data-v0.1.0-r1") -> Path:
    spec_path = _write_spec(root)
    if snapshot_id != "geno-lewm-data-v0.1.0-r1":
        payload = json.loads(spec_path.read_text(encoding="utf-8"))
        payload["snapshot_id"] = snapshot_id
        spec_path.write_text(json.dumps(payload), encoding="utf-8")
    dataset_dir = root / "dataset"
    build_dataset_snapshot(spec_path, dataset_dir)
    return dataset_dir


def _write_carbon_model_dir(root: Path) -> Path:
    carbon_dir = root / "carbon-model"
    carbon_dir.mkdir()
    (carbon_dir / "config.json").write_text('{"model_type":"carbon"}\n', encoding="utf-8")
    (carbon_dir / "tokenizer.json").write_text('{"version":"1.0"}\n', encoding="utf-8")
    (carbon_dir / "model.safetensors").write_bytes(b"not-real-weights-for-preflight\n")
    return carbon_dir


def _write_training_config(root: Path) -> Path:
    path = root / "train.real.yaml"
    path.write_text(
        "\n".join(
            [
                "run_id: first-snv-test",
                "seed: 104729",
                "phase: phase1",
                "deterministic: true",
                "schema_version: 1.0.0",
                "encoder:",
                "  model_id: /local/carbon",
                "  revision: main",
                "  dtype: bf16",
                "  state_layer: 20",
                "  pool_type: centered_mean",
                "  pool_radius: 8",
                "  normalize: true",
                "data:",
                "  batch_size: 8",
                "  corpus_id: HuggingFaceBio/carbon-pretraining-corpus",
                "  corpus_revision: main",
                "  num_workers: 0",
                "  shuffle_buffer: 4096",
                "predictor:",
                "  architecture: cross_attention",
                "  n_layers: 6",
                "  n_heads: 8",
                "  d_state: 512",
                "  d_action: 64",
                "  dtype: bf16",
                "action:",
                "  d_action: 64",
                "  max_len: 16",
                "  sub_encoders:",
                "    - snv",
                "training:",
                "  max_steps: 2",
                "  collapse_log_every_steps: 1",
                "optimizer:",
                "  name: adamw",
                "  lr: 3.0e-4",
                "  beta1: 0.9",
                "  beta2: 0.95",
                "  weight_decay: 0.1",
                "  grad_clip: 1.0",
                "  warmup_steps: 0",
                "  schedule: wsd",
                "eval:",
                "  benchmarks:",
                "    - clinvar_coding",
                "    - clinvar_noncoding",
                "    - rollout",
                "  smoke_variants: 1000",
                "observability:",
                "  log_level: info",
                "  redaction_strict: true",
                "  wandb_project: null",
                "runtime:",
                "  backend: torch",
                "  device: cpu",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    return path


def _available_dependency(import_name: str, required: bool) -> DependencyProbe:
    return DependencyProbe(
        import_name=import_name,
        package=import_name.split(".", 1)[0],
        required=required,
        available=True,
        version="1.0.0",
        reason="available in test",
    )


def _missing_dependency(import_name: str, required: bool) -> DependencyProbe:
    return DependencyProbe(
        import_name=import_name,
        package=import_name.split(".", 1)[0],
        required=required,
        available=False,
        version=None,
        reason="missing in test",
    )


def _dataset_artifact_identity(root: Path, relative: str) -> dict[str, object]:
    path = root / relative
    return {
        "path": relative,
        "sha256": sha256_file(path),
        "size_bytes": path.stat().st_size,
    }


def _refresh_dataset_sha256sums(root: Path) -> None:
    path = root / "SHA256SUMS"
    lines = []
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        if not raw_line.strip():
            continue
        _, relative = raw_line.split(maxsplit=1)
        digest = sha256_file(root / relative).removeprefix("sha256:")
        lines.append(f"{digest}  {relative}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _codes(report) -> set[str]:
    return {issue.code for issue in report.issues}
