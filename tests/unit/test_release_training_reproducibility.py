"""Tests for deterministic training reproducibility evidence reports."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from tools.release.training_reproducibility import (
    GENERATED_BY,
    REPORT_NAME,
    build_training_reproducibility_report,
    main,
)
from tools.release.training_run import (
    GENERATED_BY as TRAINING_RUN_GENERATED_BY,
    build_training_run_package,
)


def test_reproducibility_report_accepts_matching_deterministic_runs(
    tmp_path: Path,
) -> None:
    baseline = _write_run(tmp_path / "baseline", deterministic=False, throughput=100.0)
    det_a = _write_run(tmp_path / "det-a", deterministic=True, throughput=92.0)
    det_b = _write_run(tmp_path / "det-b", deterministic=True, throughput=90.0)

    report = build_training_reproducibility_report(
        baseline_run_dir=baseline,
        deterministic_run_a=det_a,
        deterministic_run_b=det_b,
    )

    assert report.ok is True
    payload = report.to_dict()
    assert payload["generated_by"] == GENERATED_BY
    assert list(report.deterministic_pair.matched_artifacts) == [
        "checkpoint:action_encoder.safetensors",
        "checkpoint:predictor.safetensors",
        "dataset_manifest:dataset_manifest.json",
        "training_config:train_config.yaml",
    ]
    assert report.throughput.drop_fraction == pytest.approx(0.10)


def test_reproducibility_report_blocks_checkpoint_mismatch(tmp_path: Path) -> None:
    baseline = _write_run(tmp_path / "baseline", deterministic=False, throughput=100.0)
    det_a = _write_run(tmp_path / "det-a", deterministic=True, throughput=90.0)
    det_b = _write_run(
        tmp_path / "det-b",
        deterministic=True,
        throughput=90.0,
        predictor_bytes=b"changed-checkpoint",
    )

    report = build_training_reproducibility_report(
        baseline_run_dir=baseline,
        deterministic_run_a=det_a,
        deterministic_run_b=det_b,
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
        "throughput.missing_baseline_run"
    ]


def test_reproducibility_report_blocks_missing_throughput_metric(
    tmp_path: Path,
) -> None:
    baseline = _write_run(tmp_path / "baseline", deterministic=False, throughput=None)
    det_a = _write_run(tmp_path / "det-a", deterministic=True, throughput=90.0)
    det_b = _write_run(tmp_path / "det-b", deterministic=True, throughput=90.0)

    report = build_training_reproducibility_report(
        baseline_run_dir=baseline,
        deterministic_run_a=det_a,
        deterministic_run_b=det_b,
    )

    assert report.ok is False
    assert "throughput.missing_baseline_rate" in {
        blocker.code for blocker in report.throughput.blockers
    }


def test_reproducibility_cli_writes_report(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    baseline = _write_run(tmp_path / "baseline", deterministic=False, throughput=100.0)
    det_a = _write_run(tmp_path / "det-a", deterministic=True, throughput=92.0)
    det_b = _write_run(tmp_path / "det-b", deterministic=True, throughput=90.0)
    output = tmp_path / REPORT_NAME

    rc = main(
        [
            "--baseline-run-dir",
            str(baseline),
            "--deterministic-run-a",
            str(det_a),
            "--deterministic-run-b",
            str(det_b),
            "--output",
            str(output),
        ]
    )
    captured = capsys.readouterr()

    assert rc == 0
    assert output.is_file()
    assert json.loads(captured.out)["ok"] is True
    assert json.loads(output.read_text(encoding="utf-8"))["ok"] is True


def _write_run(
    root: Path,
    *,
    deterministic: bool,
    throughput: float | None,
    predictor_bytes: bytes = b"predictor",
) -> Path:
    root.mkdir(parents=True)
    (root / "dataset_manifest.json").write_text(
        json.dumps(
            {
                "schema_version": "1.0.0",
                "snapshot_id": "geno-lewm-data-v0.1.0-r1",
            },
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    (root / "train_config.yaml").write_text(
        f"seed: 7\ndeterministic: {str(deterministic).lower()}\nbatch_size: 2\n",
        encoding="utf-8",
    )
    metric_values: dict[str, object] = {
        "train_loss": 0.42,
        "collapse_var_min": {"value": 0.11},
    }
    if throughput is not None:
        metric_values["samples_per_second"] = throughput
    metrics: dict[str, object] = {
        "sample_count": 128,
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
    metadata_path = root / "training_run.json"
    metadata_path.write_text(
        json.dumps(
            {
                "schema_version": "1.0.0",
                "run_id": "geno-lewm-repro-seed7",
                "generated_by": TRAINING_RUN_GENERATED_BY,
                "generated_at": "2026-06-10T00:00:00Z",
                "command": "uv run geno-lewm-train --carbon-train --steps 50",
                "commit_sha": "abcdef1234567890",
                "package_version": "0.2.1",
                "dataset_snapshot_id": "geno-lewm-data-v0.1.0-r1",
                "dataset_manifest": "dataset_manifest.json",
                "training_config": "train_config.yaml",
                "metrics": "metrics.json",
                "logs": ["train.log"],
                "checkpoint_files": [
                    "predictor.safetensors",
                    "action_encoder.safetensors",
                ],
                "status": "completed",
                "hardware": ["Supported torch backend fixture."],
                "runtime": ["Python 3.13; GenoLeWM 0.2.1; torch fixture."],
                "seeds": {"base": 7, "data": 7, "predictor": 8, "lora": 9},
                "determinism": json.dumps(
                    {
                        "deterministic": deterministic,
                        "torch_deterministic_algorithms": deterministic,
                    },
                    sort_keys=True,
                ),
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
