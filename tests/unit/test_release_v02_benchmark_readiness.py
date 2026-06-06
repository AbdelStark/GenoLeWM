"""Tests for the v0.2 benchmark-readiness report generator."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from geno_lewm.errors import InputError
from geno_lewm.provenance import sha256_bytes
from tools.release import v02_benchmark_readiness


def test_readiness_report_marks_missing_and_failed_rows(tmp_path: Path) -> None:
    metrics = tmp_path / "clinvar_coding.metrics.json"
    rollout = tmp_path / "rollout.ar_speed.json"
    metrics.write_text(
        json.dumps(
            _metrics_payload(
                [
                    *_binary_metrics("clinvar_coding", baseline=True),
                ]
            )
        ),
        encoding="utf-8",
    )
    rollout.write_text(
        json.dumps(
            {
                "schema_version": "1.0.0",
                "generated_by": "bench.rollout",
                "ok": False,
                "commit": "abcdef1234567890",
                "command": ["python", "-m", "bench.rollout", "--k", "5"],
                "rows": [
                    {
                        "horizon": 5,
                        "measured_speedup": 1.5,
                        "target_speedup": 2.0,
                        "target_met": False,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    report = v02_benchmark_readiness.build_readiness_report(
        metrics_json=(metrics,),
        rollout_speed_report=rollout,
        command=("python", "-m", "tools.release.v02_benchmark_readiness"),
    )

    assert report["ok"] is False
    rows = _rows_by_id(report)
    assert rows["clinvar_coding"]["status"] == "pass"
    assert rows["clinvar_noncoding"]["status"] == "missing"
    assert rows["clinvar_noncoding"]["baseline_observed"] is False
    assert rows["ar_rollout_speed"]["status"] == "failed"
    assert rows["ar_rollout_speed"]["command"] == ["python", "-m", "bench.rollout", "--k", "5"]
    assert "clinvar_noncoding" in report["missing_or_failed_benchmarks"]
    assert "ar_rollout_speed" in report["missing_or_failed_benchmarks"]
    assert report["negative_findings"]
    assert report["inputs"]["metrics_json"][0]["sha256"].startswith("sha256:")


def test_readiness_report_can_pass_with_all_required_artifacts(tmp_path: Path) -> None:
    metrics = tmp_path / "eval_metrics.json"
    rollout = tmp_path / "rollout.ar_speed.json"
    efficiency = tmp_path / "efficiency_report.json"
    metrics.write_text(
        json.dumps(
            _metrics_payload(
                [
                    *_binary_metrics("clinvar_coding", baseline=True),
                    *_binary_metrics("clinvar_noncoding", baseline=True),
                    _metric("brca2", "spearman_rho", 0.61, baseline=True),
                    _metric("traitgym_mendelian", "spearman_rho", 0.44, baseline=True),
                    _metric("rollout_phased_haplotypes", "cosine_similarity_mean", 0.91),
                    _metric("rollout_phased_haplotypes", "l2_distance_mean", 0.12),
                    _metric("rollout_phased_haplotypes", "recall_at_k", 0.66),
                    _metric("rollout_synthetic_edit_chains", "cosine_similarity_mean", 0.89),
                    _metric("rollout_synthetic_edit_chains", "l2_distance_mean", 0.15),
                    _metric("rollout_synthetic_edit_chains", "recall_at_k", 0.63),
                ]
            )
        ),
        encoding="utf-8",
    )
    rollout.write_text(
        json.dumps(
            {
                "schema_version": "1.0.0",
                "generated_by": "bench.rollout",
                "ok": True,
                "commit": "abcdef1234567890",
                "command": ["python", "-m", "bench.rollout", "--k", "5", "--k", "20"],
                "rows": [
                    {
                        "horizon": 5,
                        "measured_speedup": 2.1,
                        "target_speedup": 2.0,
                        "target_met": True,
                    },
                    {
                        "horizon": 20,
                        "measured_speedup": 5.2,
                        "target_speedup": 5.0,
                        "target_met": True,
                    },
                ],
            }
        ),
        encoding="utf-8",
    )
    efficiency.write_text(json.dumps(_efficiency_payload()), encoding="utf-8")

    report = v02_benchmark_readiness.build_readiness_report(
        metrics_json=(metrics,),
        rollout_speed_report=rollout,
        efficiency_report=efficiency,
    )

    assert report["ok"] is True
    assert report["missing_or_failed_benchmarks"] == []
    rows = _rows_by_id(report)
    assert all(row["status"] == "pass" for row in rows.values())
    assert rows["inference_efficiency"]["observed_metrics"] == [
        "batched_throughput_variants_per_s",
        "peak_memory_bytes",
        "single_variant_latency_ms",
    ]


def test_release_inputs_row_passes_for_release_shaped_artifacts(tmp_path: Path) -> None:
    metrics = tmp_path / "eval_metrics.json"
    rollout = tmp_path / "rollout.ar_speed.json"
    efficiency = tmp_path / "efficiency_report.json"
    metrics.write_text(
        json.dumps(
            _metrics_payload(
                [
                    *_binary_metrics("clinvar_coding", baseline=True),
                    *_binary_metrics("clinvar_noncoding", baseline=True),
                    _metric("brca2", "spearman_rho", 0.61, baseline=True),
                    _metric("traitgym_mendelian", "spearman_rho", 0.44, baseline=True),
                    _metric("rollout_phased_haplotypes", "cosine_similarity_mean", 0.91),
                    _metric("rollout_phased_haplotypes", "l2_distance_mean", 0.12),
                    _metric("rollout_phased_haplotypes", "recall_at_k", 0.66),
                    _metric("rollout_synthetic_edit_chains", "cosine_similarity_mean", 0.89),
                    _metric("rollout_synthetic_edit_chains", "l2_distance_mean", 0.15),
                    _metric("rollout_synthetic_edit_chains", "recall_at_k", 0.63),
                ],
                release_ready=True,
            )
        ),
        encoding="utf-8",
    )
    rollout.write_text(json.dumps(_passing_rollout_payload()), encoding="utf-8")
    efficiency.write_text(
        json.dumps(_efficiency_payload(release_ready=True)),
        encoding="utf-8",
    )

    report = v02_benchmark_readiness.build_readiness_report(
        metrics_json=(metrics,),
        rollout_speed_report=rollout,
        efficiency_report=efficiency,
        require_release_inputs=True,
    )

    assert report["ok"] is True
    assert report["release_inputs_required"] is True
    rows = _rows_by_id(report)
    assert rows["release_inputs"]["status"] == "pass"
    assert rows["release_inputs"]["findings"] == []


def test_release_inputs_row_rejects_fixture_like_readiness_payloads(
    tmp_path: Path,
) -> None:
    metrics = tmp_path / "eval_metrics.json"
    rollout = tmp_path / "rollout.ar_speed.json"
    efficiency = tmp_path / "efficiency_report.json"
    metrics.write_text(
        json.dumps(
            _metrics_payload(
                [
                    *_binary_metrics("clinvar_coding", baseline=True),
                    *_binary_metrics("clinvar_noncoding", baseline=True),
                    _metric("brca2", "spearman_rho", 0.61, baseline=True),
                    _metric("traitgym_mendelian", "spearman_rho", 0.44, baseline=True),
                    _metric("rollout_phased_haplotypes", "cosine_similarity_mean", 0.91),
                    _metric("rollout_phased_haplotypes", "l2_distance_mean", 0.12),
                    _metric("rollout_phased_haplotypes", "recall_at_k", 0.66),
                    _metric("rollout_synthetic_edit_chains", "cosine_similarity_mean", 0.89),
                    _metric("rollout_synthetic_edit_chains", "l2_distance_mean", 0.15),
                    _metric("rollout_synthetic_edit_chains", "recall_at_k", 0.63),
                ]
            )
        ),
        encoding="utf-8",
    )
    rollout.write_text(json.dumps(_passing_rollout_payload()), encoding="utf-8")
    efficiency.write_text(json.dumps(_efficiency_payload()), encoding="utf-8")

    report = v02_benchmark_readiness.build_readiness_report(
        metrics_json=(metrics,),
        rollout_speed_report=rollout,
        efficiency_report=efficiency,
        require_release_inputs=True,
    )

    rows = _rows_by_id(report)
    assert report["ok"] is False
    assert rows["release_inputs"]["status"] == "failed"
    assert "release_inputs" in report["missing_or_failed_benchmarks"]
    findings = "\n".join(rows["release_inputs"]["findings"])
    assert "readiness evidence" in findings
    assert "scores+labels or metrics_input_*" in findings


def test_release_inputs_row_accepts_aggregate_metrics_input_artifacts(
    tmp_path: Path,
) -> None:
    metrics = tmp_path / "eval_metrics.json"
    rollout = tmp_path / "rollout.ar_speed.json"
    efficiency = tmp_path / "efficiency_report.json"
    metrics.write_text(
        json.dumps(
            _metrics_payload(
                [
                    *_binary_metrics("clinvar_coding", baseline=True),
                ],
                release_ready=True,
                aggregate_inputs=True,
            )
        ),
        encoding="utf-8",
    )
    rollout.write_text(json.dumps(_passing_rollout_payload()), encoding="utf-8")
    efficiency.write_text(
        json.dumps(_efficiency_payload(release_ready=True)),
        encoding="utf-8",
    )

    report = v02_benchmark_readiness.build_readiness_report(
        metrics_json=(metrics,),
        rollout_speed_report=rollout,
        efficiency_report=efficiency,
        require_release_inputs=True,
    )

    rows = _rows_by_id(report)
    assert rows["release_inputs"]["status"] == "pass"


def test_rollout_speed_requires_declared_horizons(tmp_path: Path) -> None:
    rollout = tmp_path / "rollout.ar_speed.json"
    rollout.write_text(
        json.dumps(
            {
                "schema_version": "1.0.0",
                "generated_by": "bench.rollout",
                "ok": True,
                "commit": "abcdef1",
                "command": ["python", "-m", "bench.rollout", "--k", "5"],
                "rows": [
                    {
                        "horizon": 5,
                        "measured_speedup": 2.5,
                        "target_speedup": 2.0,
                        "target_met": True,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    report = v02_benchmark_readiness.build_readiness_report(rollout_speed_report=rollout)

    rows = _rows_by_id(report)
    assert rows["ar_rollout_speed"]["status"] == "incomplete"
    assert rows["ar_rollout_speed"]["missing_metrics"] == ["k20_speedup"]


def test_readiness_rejects_rollout_commit_drift(tmp_path: Path) -> None:
    metrics = tmp_path / "eval_metrics.json"
    rollout = tmp_path / "rollout.ar_speed.json"
    metrics.write_text(
        json.dumps(_metrics_payload([*_binary_metrics("clinvar_coding", baseline=True)])),
        encoding="utf-8",
    )
    rollout.write_text(
        json.dumps(
            {
                "schema_version": "1.0.0",
                "generated_by": "bench.rollout",
                "ok": True,
                "commit": "fffffff",
                "command": ["python", "-m", "bench.rollout", "--k", "5", "--k", "20"],
                "rows": [
                    {
                        "horizon": 5,
                        "measured_speedup": 2.5,
                        "target_speedup": 2.0,
                        "target_met": True,
                    },
                    {
                        "horizon": 20,
                        "measured_speedup": 5.5,
                        "target_speedup": 5.0,
                        "target_met": True,
                    },
                ],
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(InputError, match="commit does not match"):
        v02_benchmark_readiness.build_readiness_report(
            metrics_json=(metrics,),
            rollout_speed_report=rollout,
        )


def test_readiness_rejects_rollout_without_command(tmp_path: Path) -> None:
    rollout = tmp_path / "rollout.ar_speed.json"
    rollout.write_text(
        json.dumps(
            {
                "schema_version": "1.0.0",
                "generated_by": "bench.rollout",
                "ok": True,
                "commit": "abcdef1",
                "rows": [],
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(InputError, match="command"):
        v02_benchmark_readiness.build_readiness_report(rollout_speed_report=rollout)


def test_main_writes_report_and_require_ok_returns_nonzero(tmp_path: Path) -> None:
    output = tmp_path / "v0.2_benchmark_readiness_report.json"

    rc = v02_benchmark_readiness.main(["--output", str(output), "--require-ok"])

    assert rc == 1
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["generated_by"] == "tools.release.v02_benchmark_readiness"
    assert payload["ok"] is False
    assert payload["release_inputs_required"] is True
    assert payload["command"][-1] == "--require-ok"
    assert "release_inputs" in payload["missing_or_failed_benchmarks"]


def _rows_by_id(report: dict[str, object]) -> dict[str, dict[str, object]]:
    rows = report["benchmark_rows"]
    assert isinstance(rows, list)
    return {str(row["benchmark_id"]): row for row in rows if isinstance(row, dict)}


def _metrics_payload(
    metrics: list[dict[str, object]],
    *,
    release_ready: bool = False,
    aggregate_inputs: bool = False,
) -> dict[str, object]:
    artifacts = {
        "baseline_scores": "eval/carbon_zero_shot_scores.jsonl",
        "checkpoint": "model/predictor.safetensors",
        "config": "model/train_config.yaml",
        "dataset_manifest": "dataset/dataset_manifest.json",
        "efficiency_report": "model/efficiency_report.json",
        "eval_config": "eval_config.effective.yaml",
        "scores": "eval/scores.jsonl",
    }
    if aggregate_inputs:
        artifacts = {
            "checkpoint": "model/predictor.safetensors",
            "config": "model/train_config.yaml",
            "dataset_manifest": "dataset/dataset_manifest.json",
            "efficiency_report": "model/efficiency_report.json",
            "eval_config": "eval_config.effective.yaml",
            "input_1.baseline_scores": "eval/carbon_zero_shot_scores.jsonl",
            "input_1.labels": "eval/labels.jsonl",
            "input_1.scores": "eval/scores.jsonl",
            "metrics_input_1": "eval/clinvar_coding.metrics.json",
        }
    if release_ready and not aggregate_inputs:
        artifacts["labels"] = "eval/labels.jsonl"
    return {
        "schema_version": "1.0.0",
        "generated_by": "geno-lewm-eval-all",
        "generated_at": "2026-06-06T00:00:00Z",
        "model_id": sha256_bytes(b"model"),
        "model_release": ("geno-lewm-v0.2.0-r1" if release_ready else "geno-lewm-v0.2.0-readiness"),
        "dataset_snapshot": (
            "geno-lewm-data-v0.2.0-r1" if release_ready else "geno-lewm-data-v0.2.0-readiness"
        ),
        "commit": "abcdef1234567890",
        "hardware": "Apple M3 Max CPU" if release_ready else "readiness test CPU",
        "metrics": metrics,
        "artifacts": artifacts,
        "limitations": (
            ["Single local hardware profile; no cross-platform timing claim is made."]
            if release_ready
            else ["Readiness tests use synthetic measured rows."]
        ),
        "negative_findings": ["No clinical utility is measured by this benchmark report."],
        "conclusions": [
            (
                f"The {metric['name']} metric value {metric['value']:.6g} on "
                f"{metric['split']} was evaluated from measured artifacts"
                + (
                    f" with delta {metric['delta_vs_baseline']:.6g} versus carbon_zero_shot."
                    if "delta_vs_baseline" in metric
                    else "."
                )
            )
            for metric in metrics
        ],
    }


def _binary_metrics(split: str, *, baseline: bool) -> list[dict[str, object]]:
    return [
        _metric(split, "auroc", 0.73, baseline=baseline),
        _metric(split, "average_precision", 0.71, baseline=baseline),
        _metric(split, "balanced_accuracy", 0.69, baseline=baseline),
        _metric(split, "accuracy", 0.68, baseline=baseline),
    ]


def _metric(
    split: str,
    name: str,
    value: float,
    *,
    baseline: bool = False,
) -> dict[str, object]:
    metric = {
        "name": name,
        "value": value,
        "split": split,
        "unit": "score",
        "higher_is_better": True,
        "ci_low": max(0.0, value - 0.05),
        "ci_high": min(1.0, value + 0.05),
        "n": 100,
        "notes": f"measured {split} {name}",
    }
    if baseline:
        variant_hash = sha256_bytes(f"{split}-{name}-variant-keys".encode())
        metric.update(
            {
                "baseline": "carbon_zero_shot",
                "baseline_value": value - 0.01,
                "delta_vs_baseline": 0.01,
                "evaluated_variant_keys_sha256": variant_hash,
                "baseline_evaluated_variant_keys_sha256": variant_hash,
            }
        )
    return metric


def _passing_rollout_payload() -> dict[str, object]:
    return {
        "schema_version": "1.0.0",
        "generated_by": "bench.rollout",
        "ok": True,
        "commit": "abcdef1234567890",
        "command": ["python", "-m", "bench.rollout", "--k", "5", "--k", "20"],
        "rows": [
            {
                "horizon": 5,
                "measured_speedup": 2.1,
                "target_speedup": 2.0,
                "target_met": True,
            },
            {
                "horizon": 20,
                "measured_speedup": 5.2,
                "target_speedup": 5.0,
                "target_met": True,
            },
        ],
    }


def _efficiency_payload(*, release_ready: bool = False) -> dict[str, object]:
    return {
        "schema_version": "1.0.0",
        "generated_by": "tools.release.efficiency_report",
        "generated_at": "2026-06-06T00:00:00Z",
        "model_id": sha256_bytes(b"model"),
        "model_release": ("geno-lewm-v0.2.0-r1" if release_ready else "geno-lewm-v0.2.0-readiness"),
        "dataset_snapshot": (
            "geno-lewm-data-v0.2.0-r1" if release_ready else "geno-lewm-data-v0.2.0-readiness"
        ),
        "commit": "abcdef1234567890",
        "command": ["geno-lewm-score", "--variant", "1:10:A:T"],
        "hardware": "Apple M3 Max CPU" if release_ready else "readiness test CPU",
        "runtime": "Python 3.13; backend=cpu",
        "warmup_batches": 1,
        "samples": 3,
        "measurements": {
            "single_variant_latency_ms": 10.0,
            "batched_throughput_variants_per_s": 50.0,
            "peak_memory_bytes": 123456,
        },
        "inputs": {
            "model_manifest": {
                "path": "model/manifest.json",
                "sha256": sha256_bytes(b"manifest"),
                "size_bytes": 10,
            }
        },
        "limitations": ["Single local hardware profile."],
    }
