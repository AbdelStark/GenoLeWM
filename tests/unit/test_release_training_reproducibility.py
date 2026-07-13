"""Tests for deterministic training reproducibility evidence reports."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from geno_lewm.provenance import sha256_file
from geno_lewm.training.preflight import (
    GENERATED_BY as TRAINING_PREFLIGHT_GENERATED_BY,
    REPORT_NAME as TRAINING_PREFLIGHT_REPORT_NAME,
    SCHEMA_VERSION as TRAINING_PREFLIGHT_SCHEMA_VERSION,
)
from tools.release.training_reproducibility import (
    GENERATED_BY,
    REPORT_NAME,
    build_training_reproducibility_report,
    main,
)
from tools.release.training_run import (
    GENERATED_BY as TRAINING_RUN_GENERATED_BY,
    REQUIRED_PREFLIGHT_DATASET_CORE_FILES,
    build_training_run_package,
)


def test_reproducibility_report_accepts_matching_deterministic_runs(
    tmp_path: Path,
) -> None:
    baseline_a = _write_run(tmp_path / "baseline-a", deterministic=False, throughput=100.0)
    baseline_b = _write_run(tmp_path / "baseline-b", deterministic=False, throughput=99.0)
    det_a = _write_run(tmp_path / "det-a", deterministic=True, throughput=92.0)
    det_b = _write_run(tmp_path / "det-b", deterministic=True, throughput=90.0)

    report = build_training_reproducibility_report(
        baseline_run_a=baseline_a,
        deterministic_run_a=det_a,
        deterministic_run_b=det_b,
        baseline_run_b=baseline_b,
    )

    assert report.ok is True
    payload = report.to_dict()
    assert payload["generated_by"] == GENERATED_BY
    assert [run.label for run in report.runs] == [
        "baseline_a",
        "deterministic_a",
        "deterministic_b",
        "baseline_b",
    ]
    assert list(report.deterministic_pair.matched_artifacts) == [
        "checkpoint:action_encoder.safetensors",
        "checkpoint:predictor.safetensors",
        "dataset_manifest:dataset_manifest.json",
        "training_config:train_config.yaml",
    ]
    assert payload["throughput"]["baseline_spread_definition"] == "(max-min)/max"
    assert payload["throughput"]["deterministic_spread_definition"] == "(max-min)/max"
    assert payload["throughput"]["drop_definition"] == (
        "max(0,(baseline_max-det_min)/baseline_max)"
    )
    assert report.throughput.baseline_samples_per_second == {
        "baseline_a": 100.0,
        "baseline_b": 99.0,
    }
    assert report.throughput.baseline_max_samples_per_second == 100.0
    assert report.throughput.baseline_spread_fraction == pytest.approx(0.01)
    assert report.throughput.drop_fraction == pytest.approx(0.10)


def test_reproducibility_report_blocks_checkpoint_mismatch(tmp_path: Path) -> None:
    baseline_a, baseline_b = _write_baseline_pair(tmp_path)
    det_a = _write_run(tmp_path / "det-a", deterministic=True, throughput=90.0)
    det_b = _write_run(
        tmp_path / "det-b",
        deterministic=True,
        throughput=90.0,
        predictor_bytes=b"changed-checkpoint",
    )

    report = build_training_reproducibility_report(
        baseline_run_a=baseline_a,
        deterministic_run_a=det_a,
        deterministic_run_b=det_b,
        baseline_run_b=baseline_b,
    )

    assert report.ok is False
    assert {blocker.code for blocker in report.deterministic_pair.blockers} == {
        "deterministic_pair.artifact_mismatch"
    }


def test_reproducibility_report_blocks_missing_baseline(tmp_path: Path) -> None:
    det_a = _write_run(tmp_path / "det-a", deterministic=True, throughput=90.0)
    det_b = _write_run(tmp_path / "det-b", deterministic=True, throughput=90.0)

    report = build_training_reproducibility_report(
        deterministic_run_a=det_a,
        deterministic_run_b=det_b,
    )

    assert report.ok is False
    assert [blocker.code for blocker in report.throughput.blockers] == [
        "throughput.missing_baseline_runs"
    ]


def test_reproducibility_report_requires_independent_arm_directories(tmp_path: Path) -> None:
    baseline = _write_run(tmp_path / "baseline", deterministic=False, throughput=100.0)
    det_a = _write_run(tmp_path / "det-a", deterministic=True, throughput=90.0)
    det_b = _write_run(tmp_path / "det-b", deterministic=True, throughput=90.0)

    report = build_training_reproducibility_report(
        baseline_run_a=baseline,
        deterministic_run_a=det_a,
        deterministic_run_b=det_b,
        baseline_run_b=baseline,
    )

    assert report.ok is False
    assert "run_contract.run_directory_reused" in {
        blocker.code for blocker in report.run_contract.blockers
    }


def test_reproducibility_report_blocks_missing_throughput_metric(
    tmp_path: Path,
) -> None:
    baseline_a = _write_run(tmp_path / "baseline-a", deterministic=False, throughput=None)
    baseline_b = _write_run(tmp_path / "baseline-b", deterministic=False, throughput=100.0)
    det_a = _write_run(tmp_path / "det-a", deterministic=True, throughput=90.0)
    det_b = _write_run(tmp_path / "det-b", deterministic=True, throughput=90.0)

    report = build_training_reproducibility_report(
        baseline_run_a=baseline_a,
        deterministic_run_a=det_a,
        deterministic_run_b=det_b,
        baseline_run_b=baseline_b,
    )

    assert report.ok is False
    assert "throughput.missing_baseline_rate" in {
        blocker.code for blocker in report.throughput.blockers
    }


def test_reproducibility_report_rejects_baseline_config_drift(tmp_path: Path) -> None:
    baseline_a = _write_run(
        tmp_path / "baseline-a",
        deterministic=False,
        throughput=100.0,
        config_overrides={"optimizer": {"lr": 1.0e-3}},
    )
    det_a = _write_run(tmp_path / "det-a", deterministic=True, throughput=92.0)
    det_b = _write_run(tmp_path / "det-b", deterministic=True, throughput=90.0)
    baseline_b = _write_run(tmp_path / "baseline-b", deterministic=False, throughput=100.0)

    report = build_training_reproducibility_report(
        baseline_run_a=baseline_a,
        deterministic_run_a=det_a,
        deterministic_run_b=det_b,
        baseline_run_b=baseline_b,
        require_preflight=True,
    )

    assert report.ok is False
    assert "throughput.baseline_config_mismatch" in {
        blocker.code for blocker in report.throughput.blockers
    }


def test_reproducibility_report_rejects_baseline_dataset_manifest_drift(
    tmp_path: Path,
) -> None:
    baseline_a = _write_run(
        tmp_path / "baseline-a",
        deterministic=False,
        throughput=100.0,
        dataset_marker="different-bytes-same-snapshot-id",
    )
    det_a = _write_run(tmp_path / "det-a", deterministic=True, throughput=92.0)
    det_b = _write_run(tmp_path / "det-b", deterministic=True, throughput=90.0)
    baseline_b = _write_run(tmp_path / "baseline-b", deterministic=False, throughput=100.0)

    report = build_training_reproducibility_report(
        baseline_run_a=baseline_a,
        deterministic_run_a=det_a,
        deterministic_run_b=det_b,
        baseline_run_b=baseline_b,
        require_preflight=True,
    )

    assert report.ok is False
    assert "throughput.dataset_manifest_mismatch" in {
        blocker.code for blocker in report.throughput.blockers
    }


def test_reproducibility_report_rejects_internally_inconsistent_throughput(
    tmp_path: Path,
) -> None:
    baseline_a = _write_run(
        tmp_path / "baseline-a",
        deterministic=False,
        throughput=100.0,
        elapsed_seconds=80.0,
    )
    det_a = _write_run(tmp_path / "det-a", deterministic=True, throughput=92.0)
    det_b = _write_run(tmp_path / "det-b", deterministic=True, throughput=90.0)
    baseline_b = _write_run(tmp_path / "baseline-b", deterministic=False, throughput=100.0)

    report = build_training_reproducibility_report(
        baseline_run_a=baseline_a,
        deterministic_run_a=det_a,
        deterministic_run_b=det_b,
        baseline_run_b=baseline_b,
        require_preflight=True,
    )

    assert report.ok is False
    assert "run_contract.throughput_metric_inconsistent" in {
        blocker.code for blocker in report.run_contract.blockers
    }


def test_reproducibility_report_requires_500_steps_for_every_arm(tmp_path: Path) -> None:
    baseline_a = _write_run(
        tmp_path / "baseline-a",
        deterministic=False,
        throughput=100.0,
        steps_completed=499,
    )
    det_a = _write_run(tmp_path / "det-a", deterministic=True, throughput=92.0)
    det_b = _write_run(tmp_path / "det-b", deterministic=True, throughput=90.0)
    baseline_b = _write_run(tmp_path / "baseline-b", deterministic=False, throughput=100.0)

    report = build_training_reproducibility_report(
        baseline_run_a=baseline_a,
        deterministic_run_a=det_a,
        deterministic_run_b=det_b,
        baseline_run_b=baseline_b,
        require_preflight=True,
    )

    assert report.ok is False
    assert "run_contract.steps_completed_mismatch" in {blocker.code for blocker in report.blockers}


@pytest.mark.parametrize(
    ("field", "kwargs", "code"),
    [
        ("sample_count", {"sample_count": 3_992}, "run_contract.sample_count_mismatch"),
        (
            "new_sample_count",
            {"new_sample_count": 3_992},
            "run_contract.new_sample_count_mismatch",
        ),
    ],
)
def test_reproducibility_report_requires_exact_fresh_sample_counts(
    tmp_path: Path,
    field: str,
    kwargs: dict[str, int],
    code: str,
) -> None:
    baseline_a = _write_run(
        tmp_path / "baseline-a",
        deterministic=False,
        throughput=100.0,
        **kwargs,
    )
    det_a = _write_run(tmp_path / "det-a", deterministic=True, throughput=92.0)
    det_b = _write_run(tmp_path / "det-b", deterministic=True, throughput=90.0)
    baseline_b = _write_run(tmp_path / "baseline-b", deterministic=False, throughput=100.0)

    report = build_training_reproducibility_report(
        baseline_run_a=baseline_a,
        deterministic_run_a=det_a,
        deterministic_run_b=det_b,
        baseline_run_b=baseline_b,
        require_preflight=True,
    )

    assert report.ok is False, field
    assert code in {blocker.code for blocker in report.blockers}


def test_reproducibility_report_rejects_resumed_arm(tmp_path: Path) -> None:
    baseline_a, baseline_b = _write_baseline_pair(tmp_path)
    det_a = _write_run(
        tmp_path / "det-a",
        deterministic=True,
        throughput=92.0,
        resumed_from_step=10,
        resume_checkpoint="predictor_checkpoint.pt",
    )
    det_b = _write_run(tmp_path / "det-b", deterministic=True, throughput=90.0)

    report = build_training_reproducibility_report(
        baseline_run_a=baseline_a,
        deterministic_run_a=det_a,
        deterministic_run_b=det_b,
        baseline_run_b=baseline_b,
        require_preflight=True,
    )

    assert report.ok is False
    assert {
        "run_contract.resumed_from_step_nonzero",
        "run_contract.resume_checkpoint_present",
    } <= {blocker.code for blocker in report.blockers}


@pytest.mark.parametrize(
    ("kwargs", "code"),
    [
        ({"nan_loss_count": 1}, "run_contract.nan_loss_detected"),
        ({"collapse_alert_count": 1}, "run_contract.collapse_alert_detected"),
    ],
)
def test_reproducibility_report_rejects_unsafe_arm_metrics(
    tmp_path: Path,
    kwargs: dict[str, int],
    code: str,
) -> None:
    baseline_a, baseline_b = _write_baseline_pair(tmp_path)
    det_a = _write_run(tmp_path / "det-a", deterministic=True, throughput=92.0)
    det_b = _write_run(
        tmp_path / "det-b",
        deterministic=True,
        throughput=90.0,
        **kwargs,
    )

    report = build_training_reproducibility_report(
        baseline_run_a=baseline_a,
        deterministic_run_a=det_a,
        deterministic_run_b=det_b,
        baseline_run_b=baseline_b,
        require_preflight=True,
    )

    assert report.ok is False
    assert code in {blocker.code for blocker in report.blockers}


@pytest.mark.parametrize(
    ("baseline_kwargs", "det_kwargs", "code"),
    [
        (
            {"cublas_workspace_config": ":4096:8"},
            {},
            "run_contract.baseline_cublas_workspace_present",
        ),
        (
            {"include_cublas_workspace_field": False},
            {},
            "run_contract.baseline_cublas_workspace_present",
        ),
        (
            {"torch_deterministic_algorithms": True},
            {},
            "run_contract.baseline_algorithms_enabled",
        ),
        (
            {},
            {"cublas_workspace_config": ":16:8"},
            "run_contract.deterministic_cublas_workspace_mismatch",
        ),
    ],
)
def test_reproducibility_report_enforces_per_arm_torch_controls(
    tmp_path: Path,
    baseline_kwargs: dict[str, object],
    det_kwargs: dict[str, object],
    code: str,
) -> None:
    baseline_a = _write_run(
        tmp_path / "baseline-a",
        deterministic=False,
        throughput=100.0,
        **baseline_kwargs,
    )
    det_a = _write_run(
        tmp_path / "det-a",
        deterministic=True,
        throughput=92.0,
        **det_kwargs,
    )
    det_b = _write_run(tmp_path / "det-b", deterministic=True, throughput=90.0)
    baseline_b = _write_run(tmp_path / "baseline-b", deterministic=False, throughput=100.0)

    report = build_training_reproducibility_report(
        baseline_run_a=baseline_a,
        deterministic_run_a=det_a,
        deterministic_run_b=det_b,
        baseline_run_b=baseline_b,
        require_preflight=True,
    )

    assert report.ok is False
    assert code in {blocker.code for blocker in report.blockers}


def test_reproducibility_report_requires_h200_for_every_arm(tmp_path: Path) -> None:
    baseline_a = _write_run(
        tmp_path / "baseline-a",
        deterministic=False,
        throughput=100.0,
        device_name="NVIDIA A100-SXM4-80GB",
    )
    det_a = _write_run(tmp_path / "det-a", deterministic=True, throughput=92.0)
    det_b = _write_run(tmp_path / "det-b", deterministic=True, throughput=90.0)
    baseline_b = _write_run(tmp_path / "baseline-b", deterministic=False, throughput=100.0)

    report = build_training_reproducibility_report(
        baseline_run_a=baseline_a,
        deterministic_run_a=det_a,
        deterministic_run_b=det_b,
        baseline_run_b=baseline_b,
        require_preflight=True,
    )

    assert report.ok is False
    assert "run_contract.not_h200" in {blocker.code for blocker in report.blockers}


def test_reproducibility_report_requires_available_h200(tmp_path: Path) -> None:
    baseline_a = _write_run(
        tmp_path / "baseline-a",
        deterministic=False,
        throughput=100.0,
        device_available=False,
    )
    det_a = _write_run(tmp_path / "det-a", deterministic=True, throughput=92.0)
    det_b = _write_run(tmp_path / "det-b", deterministic=True, throughput=90.0)
    baseline_b = _write_run(tmp_path / "baseline-b", deterministic=False, throughput=100.0)

    report = build_training_reproducibility_report(
        baseline_run_a=baseline_a,
        deterministic_run_a=det_a,
        deterministic_run_b=det_b,
        baseline_run_b=baseline_b,
        require_preflight=True,
    )

    assert report.ok is False
    assert "run_contract.not_h200" in {blocker.code for blocker in report.blockers}


def test_reproducibility_report_rejects_runtime_drift_between_arms(tmp_path: Path) -> None:
    baseline_a, baseline_b = _write_baseline_pair(tmp_path)
    det_a = _write_run(tmp_path / "det-a", deterministic=True, throughput=92.0)
    det_b = _write_run(
        tmp_path / "det-b",
        deterministic=True,
        throughput=90.0,
        torch_version="2.7.0+cu126",
    )

    report = build_training_reproducibility_report(
        baseline_run_a=baseline_a,
        deterministic_run_a=det_a,
        deterministic_run_b=det_b,
        baseline_run_b=baseline_b,
        require_preflight=True,
    )

    assert report.ok is False
    assert "run_contract.runtime_identity_mismatch" in {blocker.code for blocker in report.blockers}


def test_reproducibility_report_marks_noisy_deterministic_repeats_inconclusive(
    tmp_path: Path,
) -> None:
    baseline_a, baseline_b = _write_baseline_pair(tmp_path)
    det_a = _write_run(tmp_path / "det-a", deterministic=True, throughput=92.0)
    det_b = _write_run(tmp_path / "det-b", deterministic=True, throughput=85.0)

    report = build_training_reproducibility_report(
        baseline_run_a=baseline_a,
        deterministic_run_a=det_a,
        deterministic_run_b=det_b,
        baseline_run_b=baseline_b,
        require_preflight=True,
    )

    assert report.ok is False
    assert report.throughput.status == "inconclusive"
    assert report.throughput.threshold_evaluated is False
    assert report.throughput.deterministic_spread_fraction == pytest.approx(7 / 92)
    codes = {blocker.code for blocker in report.throughput.blockers}
    assert codes == {"throughput.deterministic_repeat_spread_inconclusive"}


def test_reproducibility_report_marks_noisy_baseline_repeats_inconclusive(
    tmp_path: Path,
) -> None:
    baseline_a, baseline_b = _write_baseline_pair(
        tmp_path,
        throughput_a=100.0,
        throughput_b=93.0,
    )
    det_a = _write_run(tmp_path / "det-a", deterministic=True, throughput=90.0)
    det_b = _write_run(tmp_path / "det-b", deterministic=True, throughput=89.0)

    report = build_training_reproducibility_report(
        baseline_run_a=baseline_a,
        deterministic_run_a=det_a,
        deterministic_run_b=det_b,
        baseline_run_b=baseline_b,
        require_preflight=True,
    )

    assert report.ok is False
    assert report.throughput.status == "inconclusive"
    assert report.throughput.threshold_evaluated is False
    assert report.throughput.baseline_spread_fraction == pytest.approx(0.07)
    codes = {blocker.code for blocker in report.throughput.blockers}
    assert codes == {"throughput.baseline_repeat_spread_inconclusive"}


@pytest.mark.parametrize(
    ("deterministic_rate", "expected_ok", "expected_status"),
    [(85.0, True, "pass"), (84.0, False, "fail")],
)
def test_reproducibility_report_applies_inclusive_fifteen_percent_drop_limit(
    tmp_path: Path,
    deterministic_rate: float,
    expected_ok: bool,
    expected_status: str,
) -> None:
    baseline_a, baseline_b = _write_baseline_pair(
        tmp_path,
        throughput_a=99.0,
        throughput_b=100.0,
    )
    det_a = _write_run(
        tmp_path / "det-a",
        deterministic=True,
        throughput=deterministic_rate,
    )
    det_b = _write_run(
        tmp_path / "det-b",
        deterministic=True,
        throughput=deterministic_rate,
    )

    report = build_training_reproducibility_report(
        baseline_run_a=baseline_a,
        deterministic_run_a=det_a,
        deterministic_run_b=det_b,
        baseline_run_b=baseline_b,
        require_preflight=True,
    )

    assert report.ok is expected_ok
    assert report.throughput.status == expected_status
    assert report.throughput.threshold_evaluated is True


def test_reproducibility_cli_writes_report(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    baseline_a, baseline_b = _write_baseline_pair(tmp_path)
    det_a = _write_run(tmp_path / "det-a", deterministic=True, throughput=92.0)
    det_b = _write_run(tmp_path / "det-b", deterministic=True, throughput=90.0)
    output = tmp_path / REPORT_NAME

    rc = main(
        [
            "--baseline-run-a",
            str(baseline_a),
            "--deterministic-run-a",
            str(det_a),
            "--deterministic-run-b",
            str(det_b),
            "--baseline-run-b",
            str(baseline_b),
            "--output",
            str(output),
        ]
    )
    captured = capsys.readouterr()

    assert rc == 0
    assert output.is_file()
    assert json.loads(captured.out)["ok"] is True
    assert json.loads(output.read_text(encoding="utf-8"))["ok"] is True


def _write_baseline_pair(
    root: Path,
    *,
    throughput_a: float = 100.0,
    throughput_b: float = 100.0,
) -> tuple[Path, Path]:
    return (
        _write_run(root / "baseline-a", deterministic=False, throughput=throughput_a),
        _write_run(root / "baseline-b", deterministic=False, throughput=throughput_b),
    )


def _write_run(
    root: Path,
    *,
    deterministic: bool,
    throughput: float | None,
    predictor_bytes: bytes = b"predictor",
    steps_completed: int = 500,
    sample_count: int = 4_000,
    new_sample_count: int = 4_000,
    resumed_from_step: int = 0,
    resume_checkpoint: str | None = None,
    nan_loss_count: int = 0,
    collapse_alert_count: int = 0,
    cublas_workspace_config: str | None = None,
    include_cublas_workspace_field: bool = True,
    torch_deterministic_algorithms: bool | None = None,
    device_name: str = "NVIDIA H200",
    device_available: bool = True,
    torch_version: str = "2.6.0+cu124",
    config_overrides: dict[str, object] | None = None,
    elapsed_seconds: float | None = None,
    dataset_marker: str | None = None,
) -> Path:
    root.mkdir(parents=True)
    dataset_manifest = {
        "schema_version": "1.0.0",
        "snapshot_id": "geno-lewm-data-v0.1.0-r1",
    }
    if dataset_marker is not None:
        dataset_manifest["marker"] = dataset_marker
    (root / "dataset_manifest.json").write_text(
        json.dumps(dataset_manifest, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    config: dict[str, object] = {
        "run_id": "training-reproducibility-h200-nddn-500-v2",
        "seed": 7,
        "deterministic": deterministic,
        "training": {"max_steps": 500, "collapse_log_every_steps": 10},
        "data": {"batch_size": 8, "num_workers": 0, "shuffle_buffer": 0},
    }
    if config_overrides:
        config.update(config_overrides)
    import yaml

    (root / "train_config.yaml").write_text(
        yaml.safe_dump(config, sort_keys=True),
        encoding="utf-8",
    )
    metric_values: dict[str, object] = {
        "train_loss": 0.42,
        "collapse_var_min": {"value": 0.11},
        "nan_loss_count": nan_loss_count,
        "collapse_alert_count": collapse_alert_count,
        "new_sample_count": new_sample_count,
        "resumed_from_step": resumed_from_step,
    }
    if throughput is not None:
        metric_values["samples_per_second"] = throughput
        if elapsed_seconds is None:
            elapsed_seconds = new_sample_count / throughput
    if elapsed_seconds is not None:
        metric_values["elapsed_seconds"] = {"value": elapsed_seconds, "unit": "s"}
    metrics: dict[str, object] = {
        "steps_completed": steps_completed,
        "resumed_from_step": resumed_from_step,
        "resume_checkpoint": resume_checkpoint,
        "sample_count": sample_count,
        "new_sample_count": new_sample_count,
        "elapsed_seconds": elapsed_seconds,
        "metrics": metric_values,
    }
    (root / "metrics.json").write_text(
        json.dumps(metrics, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (root / "train.log").write_text(
        "step=1 loss=0.42 collapse_var_min=0.11 nan_loss=false\n",
        encoding="utf-8",
    )
    (root / "predictor.safetensors").write_bytes(predictor_bytes)
    (root / "action_encoder.safetensors").write_bytes(b"action")
    _write_preflight(
        root,
        device_name=device_name,
        device_available=device_available,
        torch_version=torch_version,
    )
    metadata_path = root / "training_run.json"
    deterministic_algorithms = (
        deterministic if torch_deterministic_algorithms is None else torch_deterministic_algorithms
    )
    workspace = (
        ":4096:8" if deterministic and cublas_workspace_config is None else cublas_workspace_config
    )
    determinism_payload: dict[str, object] = {
        "deterministic": deterministic,
        "torch_deterministic_algorithms": deterministic_algorithms,
    }
    if include_cublas_workspace_field:
        determinism_payload["cublas_workspace_config"] = workspace
    metadata_path.write_text(
        json.dumps(
            {
                "schema_version": "1.0.0",
                "run_id": "training-reproducibility-h200-nddn-500-v2",
                "generated_by": TRAINING_RUN_GENERATED_BY,
                "generated_at": "2026-06-10T00:00:00Z",
                "command": "uv run geno-lewm-train --carbon-train --steps 500",
                "commit_sha": "abcdef1234567890",
                "package_version": "0.2.1",
                "dataset_snapshot_id": "geno-lewm-data-v0.1.0-r1",
                "dataset_manifest": "dataset_manifest.json",
                "training_config": "train_config.yaml",
                "metrics": "metrics.json",
                "training_preflight_report": TRAINING_PREFLIGHT_REPORT_NAME,
                "logs": ["train.log"],
                "checkpoint_files": [
                    "predictor.safetensors",
                    "action_encoder.safetensors",
                ],
                "status": "completed",
                "hardware": ["Linux x86_64", "Python 3.13.5"],
                "runtime": [
                    "GenoLeWM Carbon-backed torch trainer.",
                    f"torch=={torch_version}",
                ],
                "seeds": {"base": 7, "data": 7, "predictor": 8, "lora": 9},
                "determinism": json.dumps(determinism_payload, sort_keys=True),
                "monitoring": {"collapse_monitoring": True, "nan_monitoring": True},
                "result_summary": "Completed reproducibility fixture archive.",
                "limitations": [
                    "Fixture archive used only to test reproducibility tooling.",
                ],
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    build_training_run_package(root, metadata_path)
    return root


def _write_preflight(
    root: Path,
    *,
    device_name: str,
    device_available: bool,
    torch_version: str,
) -> None:
    config_path = root / "train_config.yaml"
    core_files = {
        relative: {
            "path": relative,
            "sha256": "sha256:" + ("5" * 64),
            "size_bytes": 128,
        }
        for relative in REQUIRED_PREFLIGHT_DATASET_CORE_FILES
    }
    payload = {
        "schema_version": TRAINING_PREFLIGHT_SCHEMA_VERSION,
        "generated_by": TRAINING_PREFLIGHT_GENERATED_BY,
        "generated_at": "2026-07-13T00:00:00Z",
        "ok": True,
        "dataset_snapshot_id": "geno-lewm-data-v0.1.0-r1",
        "training_config": {
            "path": "train_config.yaml",
            "sha256": sha256_file(config_path),
            "size_bytes": config_path.stat().st_size,
            "resolved": {"runtime": {"device": "cuda"}},
        },
        "run_dir": {"path": root.name, "exists": True},
        "dataset": {
            "path": "dataset",
            "snapshot_id": "geno-lewm-data-v0.1.0-r1",
            "core_files": core_files,
            "files": [
                {
                    "path": "carbon/source-mix-windows.jsonl",
                    "sha256": "sha256:" + ("4" * 64),
                    "size_bytes": 128,
                }
            ],
        },
        "carbon": {
            "path": "carbon",
            "local_files_only": True,
            "artifacts": {},
        },
        "accelerator": {
            "requested_device": "cuda",
            "required": True,
            "available": device_available,
            "device_count": 1,
            "device_name": device_name,
            "total_memory_bytes": 150_000_000_000,
            "min_memory_bytes": 120_000_000_000,
            "reason": "cuda accelerator satisfies the training preflight requirement",
            "issue_code": None,
        },
        "dependencies": [
            {
                "import_name": "torch",
                "package": "torch",
                "required": True,
                "available": True,
                "version": torch_version,
                "reason": "available",
            }
        ],
        "issues": [],
    }
    (root / TRAINING_PREFLIGHT_REPORT_NAME).write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
