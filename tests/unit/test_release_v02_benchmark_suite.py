"""Tests for the v0.2 benchmark-suite runner."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest

from geno_lewm.errors import InputError
from geno_lewm.provenance import sha256_bytes
from tools.release import v02_benchmark_suite


def test_build_suite_steps_plans_score_baseline_eval_rollout_and_readiness(
    tmp_path: Path,
) -> None:
    manifest = _write_manifest(tmp_path)

    steps = v02_benchmark_suite.build_suite_steps(manifest)

    assert [step.step_id for step in steps] == [
        "clinvar_coding.score",
        "clinvar_coding.carbon_baseline",
        "clinvar_coding.eval",
        "rollout_phased_haplotypes.rollout",
        "aggregate.eval_all",
        "readiness.v02",
    ]
    commands = {step.step_id: step.command for step in steps}
    assert commands["clinvar_coding.score"][:3] == (
        "geno-lewm-score",
        "--quiet",
        "--no-banner",
    )
    assert "--model-dir" in commands["clinvar_coding.score"]
    assert "model" in commands["clinvar_coding.score"]
    assert commands["clinvar_coding.carbon_baseline"][0] == "geno-lewm-carbon-baseline"
    assert "--baseline-name" in commands["clinvar_coding.eval"]
    assert "carbon_zero_shot" in commands["clinvar_coding.eval"]
    assert commands["rollout_phased_haplotypes.rollout"][0] == "geno-lewm-rollout"
    assert commands["aggregate.eval_all"].count("--metrics-json") == 2
    assert "--require-v02-vep-metrics" in commands["aggregate.eval_all"]
    assert "--require-v02-rollout-metrics" in commands["aggregate.eval_all"]
    assert commands["readiness.v02"][:3] == (
        "python",
        "-m",
        "tools.release.v02_benchmark_readiness",
    )
    assert "--require-ok" in commands["readiness.v02"]


def test_write_suite_report_plan_only_is_not_evidence(tmp_path: Path) -> None:
    manifest = _write_manifest(tmp_path)
    output = tmp_path / "suite_report.json"

    report = v02_benchmark_suite.write_suite_report(
        manifest_path=manifest,
        output_report=output,
    )

    assert report["ok"] is False
    assert report["status"] == "planned"
    assert report["negative_findings"] == [
        "The benchmark suite was planned but not executed; this is not measured evidence.",
    ]
    written = json.loads(output.read_text(encoding="utf-8"))
    assert written["generated_by"] == "tools.release.v02_benchmark_suite"
    assert all(step["status"] == "planned" for step in written["steps"])


def test_write_suite_report_execute_runs_steps_in_manifest_directory(tmp_path: Path) -> None:
    manifest = _write_manifest(tmp_path)
    output = tmp_path / "suite_report.json"
    calls: list[tuple[list[str], Path]] = []

    def fake_runner(
        args: list[str],
        *,
        cwd: Path,
        text: bool,
        capture_output: bool,
        check: bool,
    ) -> subprocess.CompletedProcess[str]:
        calls.append((args, cwd))
        return subprocess.CompletedProcess(args, 0, stdout="ok", stderr="")

    report = v02_benchmark_suite.write_suite_report(
        manifest_path=manifest,
        output_report=output,
        execute=True,
        runner=fake_runner,
    )

    assert report["ok"] is True
    assert report["status"] == "pass"
    assert len(calls) == 6
    assert all(cwd == tmp_path for _, cwd in calls)
    assert all(step["status"] == "pass" for step in report["steps"])


def test_write_suite_report_execute_stops_after_first_failure(tmp_path: Path) -> None:
    manifest = _write_manifest(tmp_path)
    output = tmp_path / "suite_report.json"
    call_count = 0

    def fake_runner(
        args: list[str],
        *,
        cwd: Path,
        text: bool,
        capture_output: bool,
        check: bool,
    ) -> subprocess.CompletedProcess[str]:
        nonlocal call_count
        call_count += 1
        return subprocess.CompletedProcess(args, 2, stdout="", stderr="failed")

    report = v02_benchmark_suite.write_suite_report(
        manifest_path=manifest,
        output_report=output,
        execute=True,
        runner=fake_runner,
    )

    assert report["ok"] is False
    assert report["status"] == "failed"
    assert call_count == 1
    statuses = [step["status"] for step in report["steps"]]
    assert statuses[0] == "failed"
    assert statuses[1:] == ["not_run"] * 5


def test_suite_manifest_rejects_nonportable_paths(tmp_path: Path) -> None:
    manifest = _manifest_payload()
    artifacts = manifest["artifacts"]
    assert isinstance(artifacts, dict)
    artifacts["model_dir"] = str(tmp_path / "model")
    path = tmp_path / "suite.json"
    path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(InputError, match=r"artifacts\.model_dir must be package-relative"):
        v02_benchmark_suite.build_suite_steps(path)


def test_main_writes_plan_report(tmp_path: Path) -> None:
    manifest = _write_manifest(tmp_path)
    output = tmp_path / "suite_report.json"

    rc = v02_benchmark_suite.main(
        [
            "--manifest",
            str(manifest),
            "--output-report",
            str(output),
        ]
    )

    assert rc == 0
    assert json.loads(output.read_text(encoding="utf-8"))["status"] == "planned"


def _write_manifest(tmp_path: Path) -> Path:
    path = tmp_path / "suite.json"
    path.write_text(json.dumps(_manifest_payload()), encoding="utf-8")
    return path


def _manifest_payload() -> dict[str, object]:
    return {
        "schema_version": "1.0.0",
        "artifact_root": ".",
        "identity": {
            "model_id": sha256_bytes(b"model"),
            "model_release": "geno-lewm-v0.2.0-rc1",
            "dataset_snapshot": "geno-lewm-data-v0.2.0-rc1",
            "commit": "abcdef1234567890",
            "hardware": "NVIDIA H200",
        },
        "artifacts": {
            "model_dir": "model",
            "checkpoint": "model/predictor.safetensors",
            "config": "model/train_config.yaml",
            "dataset_manifest": "dataset/dataset_manifest.json",
            "efficiency_report": "model/efficiency_report.json",
        },
        "benchmarks": [
            {
                "id": "clinvar_coding",
                "kind": "vep",
                "split": "clinvar_coding",
                "vcf": "benchmarks/clinvar_coding.vcf",
                "fasta": "benchmark_inputs/ref.fa",
                "labels_jsonl": "eval/clinvar_coding.labels.jsonl",
                "scores_jsonl": "eval/clinvar_coding.scores.jsonl",
                "metrics_json": "eval/clinvar_coding.metrics.json",
                "score": {"enabled": True, "batch_size": 32, "backend": "cpu"},
                "carbon_baseline": {
                    "enabled": True,
                    "carbon_model_dir": "carbon/500m",
                    "scores_jsonl": "eval/clinvar_coding.carbon_zero_shot_scores.jsonl",
                    "metadata_json": "eval/clinvar_coding.carbon_zero_shot_summary.json",
                    "logp_cache_jsonl": "cache/carbon_zero_shot_logp.jsonl",
                    "carbon_revision": "main",
                    "dtype": "bf16",
                },
            },
            {
                "id": "rollout_phased_haplotypes",
                "kind": "rollout",
                "split": "rollout_phased_haplotypes",
                "states_jsonl": "eval/rollout_phased_haplotypes.states.jsonl",
                "metrics_json": "eval/rollout_phased_haplotypes.metrics.json",
                "recall_k": 10,
            },
        ],
        "aggregate": {
            "metrics_json": "model/eval_metrics.json",
            "report_md": "model/eval_report.md",
            "require_v02_vep_metrics": True,
            "require_v02_rollout_metrics": True,
        },
        "readiness": {
            "output_json": "model/v0.2_benchmark_readiness_report.json",
            "rollout_speed_report": "bench/rollout.ar_speed.json",
            "require_ok": True,
            "require_release_inputs": True,
        },
    }
