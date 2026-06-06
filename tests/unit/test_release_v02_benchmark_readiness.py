"""Tests for the v0.2 benchmark-readiness report generator."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from geno_lewm.errors import InputError
from geno_lewm.provenance import sha256_bytes
from tools.release import rollout_speed_scope, v02_benchmark_readiness


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


def test_readiness_report_input_identities_are_public_safe_for_absolute_paths(
    tmp_path: Path,
) -> None:
    metrics = tmp_path / "eval_metrics.json"
    rollout = tmp_path / "rollout.ar_speed.json"
    efficiency = tmp_path / "efficiency_report.json"
    output = tmp_path / "v0.2_benchmark_readiness_report.json"
    metrics.write_text(
        json.dumps(_metrics_payload([*_binary_metrics("clinvar_coding", baseline=True)])),
        encoding="utf-8",
    )
    rollout.write_text(json.dumps(_passing_rollout_payload()), encoding="utf-8")
    efficiency.write_text(json.dumps(_efficiency_payload()), encoding="utf-8")

    report = v02_benchmark_readiness.build_readiness_report(
        metrics_json=(metrics.resolve(),),
        rollout_speed_report=rollout.resolve(),
        efficiency_report=efficiency.resolve(),
        command=(
            "python",
            "-m",
            "tools.release.v02_benchmark_readiness",
            "--metrics-json",
            str(metrics.resolve()),
            "--rollout-speed-report",
            str(rollout.resolve()),
            "--efficiency-report",
            str(efficiency.resolve()),
            "--output",
            str(output.resolve()),
        ),
    )

    assert report["inputs"]["metrics_json"][0]["path"] == "eval_metrics.json"
    assert report["inputs"]["rollout_speed_report"]["path"] == "rollout.ar_speed.json"
    assert report["inputs"]["efficiency_report"]["path"] == "efficiency_report.json"
    assert report["command"] == [
        "python",
        "-m",
        "tools.release.v02_benchmark_readiness",
        "--metrics-json",
        "eval_metrics.json",
        "--rollout-speed-report",
        "rollout.ar_speed.json",
        "--efficiency-report",
        "efficiency_report.json",
        "--output",
        "v0.2_benchmark_readiness_report.json",
    ]
    assert str(tmp_path) not in json.dumps(report, sort_keys=True)


def test_readiness_report_sanitizes_nested_rollout_speed_command_paths(
    tmp_path: Path,
) -> None:
    rollout = tmp_path / "rollout.ar_speed.json"
    payload = _passing_rollout_payload()
    payload["command"] = [
        "python",
        "-m",
        "bench.rollout",
        "--output-json",
        str(rollout.resolve()),
        "--out-dir",
        str((tmp_path / "bench-results").resolve()),
        str((tmp_path / "extra.json").resolve()),
        "--k",
        "5",
    ]
    rollout.write_text(json.dumps(payload), encoding="utf-8")

    report = v02_benchmark_readiness.build_readiness_report(
        rollout_speed_report=rollout.resolve(),
    )

    command = _rows_by_id(report)["ar_rollout_speed"]["command"]
    assert command == [
        "python",
        "-m",
        "bench.rollout",
        "--output-json",
        "rollout.ar_speed.json",
        "--out-dir",
        "bench-results",
        "extra.json",
        "--k",
        "5",
    ]
    assert str(tmp_path) not in json.dumps(command, sort_keys=True)


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
    assert rows["clinvar_coding"]["observed_values"]["auroc"] == 0.73
    assert rows["clinvar_coding"]["delta_vs_baseline"]["auroc"] == 0.01
    assert rows["clinvar_coding"]["confidence_intervals"]["auroc"]["ci_low"] == pytest.approx(0.68)
    assert rows["clinvar_coding"]["confidence_intervals"]["auroc"]["ci_high"] == pytest.approx(0.78)
    assert rows["clinvar_coding"]["evaluated_variant_key_identities"]["auroc"].startswith("sha256:")
    assert rows["inference_efficiency"]["observed_metrics"] == [
        "batched_throughput_variants_per_s",
        "peak_memory_bytes",
        "single_variant_latency_ms",
    ]
    conclusions = "\n".join(str(item) for item in report["metric_conclusions"])
    assert "clinvar_coding passed" in conclusions
    assert "auroc=0.73" in conclusions
    assert "Baseline deltas: accuracy=0.01" in conclusions


def test_readiness_can_pass_with_accepted_rollout_speed_rescope(tmp_path: Path) -> None:
    metrics = tmp_path / "eval_metrics.json"
    rollout = tmp_path / "rollout.ar_speed.json"
    scope = tmp_path / "rollout_speed_scope.json"
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
    rollout.write_text(json.dumps(_failing_rollout_payload()), encoding="utf-8")
    rollout_speed_scope.write_scope_report(
        rollout_speed_report=rollout,
        output=scope,
        accepted_by="maintainer",
        accepted_at="2026-06-06T12:00:00Z",
        decision_url="https://github.com/AbdelStark/GenoLeWM/issues/42#issuecomment-1",
        rationale="The current recurrent rollout benchmark missed the RFC-0004 speed target.",
        replacement_target="Report measured rollout speed until #42 accepts a new target.",
        command=_scope_report_command(rollout, scope),
    )
    efficiency.write_text(json.dumps(_efficiency_payload()), encoding="utf-8")

    report = v02_benchmark_readiness.build_readiness_report(
        metrics_json=(metrics,),
        rollout_speed_report=rollout,
        rollout_speed_scope_report=scope,
        efficiency_report=efficiency,
    )

    assert report["ok"] is True
    assert report["missing_or_failed_benchmarks"] == []
    rows = _rows_by_id(report)
    assert rows["ar_rollout_speed"]["status"] == "rescoped"
    assert rows["ar_rollout_speed"]["failed_targets"]
    assert rows["ar_rollout_speed"]["scope_decision"]["decision"] == (
        "rescope_rfc0004_speed_target"
    )
    assert rows["ar_rollout_speed"]["scope_decision"]["generated_at"].endswith("Z")
    assert report["scope_decisions"] == [
        {
            "benchmark_id": "ar_rollout_speed",
            "decision": "rescope_rfc0004_speed_target",
            "generated_at": rows["ar_rollout_speed"]["scope_decision"]["generated_at"],
            "accepted_at": "2026-06-06T12:00:00Z",
            "decision_url": "https://github.com/AbdelStark/GenoLeWM/issues/42#issuecomment-1",
            "issue_refs": ["#42", "#197"],
        }
    ]
    assert "not passing speed evidence" in "\n".join(report["negative_findings"])


def test_readiness_rejects_stale_rollout_speed_rescope(tmp_path: Path) -> None:
    rollout = tmp_path / "rollout.ar_speed.json"
    scope = tmp_path / "rollout_speed_scope.json"
    rollout.write_text(json.dumps(_failing_rollout_payload()), encoding="utf-8")
    rollout_speed_scope.write_scope_report(
        rollout_speed_report=rollout,
        output=scope,
        accepted_by="maintainer",
        accepted_at="2026-06-06T12:00:00Z",
        decision_url="https://github.com/AbdelStark/GenoLeWM/issues/42#issuecomment-1",
        rationale="The current recurrent rollout benchmark missed the RFC-0004 speed target.",
        replacement_target="Report measured rollout speed until #42 accepts a new target.",
        command=_scope_report_command(rollout, scope),
    )
    payload = _failing_rollout_payload()
    payload["rows"][0]["measured_speedup"] = 1.7
    rollout.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(InputError, match="does not match rollout speed report"):
        v02_benchmark_readiness.build_readiness_report(
            rollout_speed_report=rollout,
            rollout_speed_scope_report=scope,
        )


def test_readiness_rejects_scope_report_with_stale_rollout_speed_path(
    tmp_path: Path,
) -> None:
    rollout = tmp_path / "rollout.ar_speed.json"
    scope = tmp_path / "rollout_speed_scope.json"
    rollout.write_text(json.dumps(_failing_rollout_payload()), encoding="utf-8")
    rollout_speed_scope.write_scope_report(
        rollout_speed_report=rollout,
        output=scope,
        accepted_by="maintainer",
        accepted_at="2026-06-06T12:00:00Z",
        decision_url="https://github.com/AbdelStark/GenoLeWM/issues/42#issuecomment-1",
        rationale="The current recurrent rollout benchmark missed the RFC-0004 speed target.",
        replacement_target="Report measured rollout speed until #42 accepts a new target.",
        command=_scope_report_command(rollout, scope),
    )
    scope_payload = json.loads(scope.read_text(encoding="utf-8"))
    scope_payload["rollout_speed_report"]["path"] = "stale-rollout.ar_speed.json"
    scope.write_text(json.dumps(scope_payload), encoding="utf-8")

    with pytest.raises(InputError, match="does not match rollout speed report"):
        v02_benchmark_readiness.build_readiness_report(
            rollout_speed_report=rollout,
            rollout_speed_scope_report=scope,
        )


def test_readiness_rejects_scope_report_with_absolute_scope_command_paths(
    tmp_path: Path,
) -> None:
    rollout = tmp_path / "rollout.ar_speed.json"
    scope = tmp_path / "rollout_speed_scope.json"
    rollout.write_text(json.dumps(_failing_rollout_payload()), encoding="utf-8")
    rollout_speed_scope.write_scope_report(
        rollout_speed_report=rollout,
        output=scope,
        accepted_by="maintainer",
        accepted_at="2026-06-06T12:00:00Z",
        decision_url="https://github.com/AbdelStark/GenoLeWM/issues/42#issuecomment-1",
        rationale="The current recurrent rollout benchmark missed the RFC-0004 speed target.",
        replacement_target="Report measured rollout speed until #42 accepts a new target.",
        command=_scope_report_command(rollout, scope),
    )
    scope_payload = json.loads(scope.read_text(encoding="utf-8"))
    scope_payload["command"] = [
        "python",
        "-m",
        "tools.release.rollout_speed_scope",
        "--rollout-speed-report",
        str(rollout.resolve()),
        "--output",
        str(scope.resolve()),
    ]
    scope.write_text(json.dumps(scope_payload), encoding="utf-8")

    with pytest.raises(InputError, match="scope report command must be public-safe"):
        v02_benchmark_readiness.build_readiness_report(
            rollout_speed_report=rollout,
            rollout_speed_scope_report=scope,
        )


def test_readiness_rejects_scope_report_with_stale_rollout_summary_command(
    tmp_path: Path,
) -> None:
    rollout = tmp_path / "rollout.ar_speed.json"
    scope = tmp_path / "rollout_speed_scope.json"
    payload = _failing_rollout_payload()
    payload["command"] = [
        "python",
        "-m",
        "bench.rollout",
        "--output-json",
        str(rollout.resolve()),
        "--out-dir",
        str((tmp_path / "bench").resolve()),
        "--k",
        "5",
        "--k",
        "20",
    ]
    rollout.write_text(json.dumps(payload), encoding="utf-8")
    rollout_speed_scope.write_scope_report(
        rollout_speed_report=rollout,
        output=scope,
        accepted_by="maintainer",
        accepted_at="2026-06-06T12:00:00Z",
        decision_url="https://github.com/AbdelStark/GenoLeWM/issues/42#issuecomment-1",
        rationale="The current recurrent rollout benchmark missed the RFC-0004 speed target.",
        replacement_target="Report measured rollout speed until #42 accepts a new target.",
        command=_scope_report_command(rollout, scope),
    )
    scope_payload = json.loads(scope.read_text(encoding="utf-8"))
    scope_payload["rollout_speed_summary"]["command"] = payload["command"]
    scope.write_text(json.dumps(scope_payload), encoding="utf-8")

    with pytest.raises(InputError, match="scope report summary is stale"):
        v02_benchmark_readiness.build_readiness_report(
            rollout_speed_report=rollout,
            rollout_speed_scope_report=scope,
        )


def test_readiness_rejects_scope_report_with_malformed_accepted_at(
    tmp_path: Path,
) -> None:
    rollout = tmp_path / "rollout.ar_speed.json"
    scope = tmp_path / "rollout_speed_scope.json"
    rollout.write_text(json.dumps(_failing_rollout_payload()), encoding="utf-8")
    rollout_speed_scope.write_scope_report(
        rollout_speed_report=rollout,
        output=scope,
        accepted_by="maintainer",
        accepted_at="2026-06-06T12:00:00Z",
        decision_url="https://github.com/AbdelStark/GenoLeWM/issues/42#issuecomment-1",
        rationale="The current recurrent rollout benchmark missed the RFC-0004 speed target.",
        replacement_target="Report measured rollout speed until #42 accepts a new target.",
        command=_scope_report_command(rollout, scope),
    )
    scope_payload = json.loads(scope.read_text(encoding="utf-8"))
    scope_payload["accepted_at"] = "2026-06-06 12:00:00"
    scope.write_text(json.dumps(scope_payload), encoding="utf-8")

    with pytest.raises(InputError, match="accepted_at must be a UTC timestamp"):
        v02_benchmark_readiness.build_readiness_report(
            rollout_speed_report=rollout,
            rollout_speed_scope_report=scope,
        )


def test_readiness_rejects_scope_report_with_malformed_decision_url(
    tmp_path: Path,
) -> None:
    rollout = tmp_path / "rollout.ar_speed.json"
    scope = tmp_path / "rollout_speed_scope.json"
    rollout.write_text(json.dumps(_failing_rollout_payload()), encoding="utf-8")
    rollout_speed_scope.write_scope_report(
        rollout_speed_report=rollout,
        output=scope,
        accepted_by="maintainer",
        accepted_at="2026-06-06T12:00:00Z",
        decision_url="https://github.com/AbdelStark/GenoLeWM/issues/42#issuecomment-1",
        rationale="The current recurrent rollout benchmark missed the RFC-0004 speed target.",
        replacement_target="Report measured rollout speed until #42 accepts a new target.",
        command=_scope_report_command(rollout, scope),
    )
    scope_payload = json.loads(scope.read_text(encoding="utf-8"))
    scope_payload["decision_url"] = "github.com/AbdelStark/GenoLeWM/issues/42"
    scope.write_text(json.dumps(scope_payload), encoding="utf-8")

    with pytest.raises(InputError, match="decision_url must be an HTTP\\(S\\) URL"):
        v02_benchmark_readiness.build_readiness_report(
            rollout_speed_report=rollout,
            rollout_speed_scope_report=scope,
        )


def test_readiness_rejects_scope_report_with_malformed_issue_refs(
    tmp_path: Path,
) -> None:
    rollout = tmp_path / "rollout.ar_speed.json"
    scope = tmp_path / "rollout_speed_scope.json"
    rollout.write_text(json.dumps(_failing_rollout_payload()), encoding="utf-8")
    rollout_speed_scope.write_scope_report(
        rollout_speed_report=rollout,
        output=scope,
        accepted_by="maintainer",
        accepted_at="2026-06-06T12:00:00Z",
        decision_url="https://github.com/AbdelStark/GenoLeWM/issues/42#issuecomment-1",
        rationale="The current recurrent rollout benchmark missed the RFC-0004 speed target.",
        replacement_target="Report measured rollout speed until #42 accepts a new target.",
        command=_scope_report_command(rollout, scope),
    )
    scope_payload = json.loads(scope.read_text(encoding="utf-8"))
    scope_payload["issue_refs"] = ["#42", "#197", "not-an-issue-ref", 56]
    scope.write_text(json.dumps(scope_payload), encoding="utf-8")

    with pytest.raises(InputError, match="issue_refs must be GitHub issue refs"):
        v02_benchmark_readiness.build_readiness_report(
            rollout_speed_report=rollout,
            rollout_speed_scope_report=scope,
        )


def test_readiness_rejects_scope_report_with_weakened_negative_findings(
    tmp_path: Path,
) -> None:
    rollout = tmp_path / "rollout.ar_speed.json"
    scope = tmp_path / "rollout_speed_scope.json"
    rollout.write_text(json.dumps(_failing_rollout_payload()), encoding="utf-8")
    rollout_speed_scope.write_scope_report(
        rollout_speed_report=rollout,
        output=scope,
        accepted_by="maintainer",
        accepted_at="2026-06-06T12:00:00Z",
        decision_url="https://github.com/AbdelStark/GenoLeWM/issues/42#issuecomment-1",
        rationale="The current recurrent rollout benchmark missed the RFC-0004 speed target.",
        replacement_target="Report measured rollout speed until #42 accepts a new target.",
        command=_scope_report_command(rollout, scope),
    )
    scope_payload = json.loads(scope.read_text(encoding="utf-8"))
    scope_payload["negative_findings"] = ["The target was not met."]
    scope.write_text(json.dumps(scope_payload), encoding="utf-8")

    with pytest.raises(InputError, match="negative_findings must preserve"):
        v02_benchmark_readiness.build_readiness_report(
            rollout_speed_report=rollout,
            rollout_speed_scope_report=scope,
        )


def test_readiness_rejects_scope_report_with_weakened_claim_boundary(
    tmp_path: Path,
) -> None:
    rollout = tmp_path / "rollout.ar_speed.json"
    scope = tmp_path / "rollout_speed_scope.json"
    rollout.write_text(json.dumps(_failing_rollout_payload()), encoding="utf-8")
    rollout_speed_scope.write_scope_report(
        rollout_speed_report=rollout,
        output=scope,
        accepted_by="maintainer",
        accepted_at="2026-06-06T12:00:00Z",
        decision_url="https://github.com/AbdelStark/GenoLeWM/issues/42#issuecomment-1",
        rationale="The current recurrent rollout benchmark missed the RFC-0004 speed target.",
        replacement_target="Report measured rollout speed until #42 accepts a new target.",
        command=_scope_report_command(rollout, scope),
    )
    scope_payload = json.loads(scope.read_text(encoding="utf-8"))
    scope_payload["claim_boundary"] = "This report records an accepted scope decision."
    scope.write_text(json.dumps(scope_payload), encoding="utf-8")

    with pytest.raises(InputError, match="claim_boundary must preserve"):
        v02_benchmark_readiness.build_readiness_report(
            rollout_speed_report=rollout,
            rollout_speed_scope_report=scope,
        )


def test_readiness_rejects_scope_report_with_malformed_generated_at(
    tmp_path: Path,
) -> None:
    rollout = tmp_path / "rollout.ar_speed.json"
    scope = tmp_path / "rollout_speed_scope.json"
    rollout.write_text(json.dumps(_failing_rollout_payload()), encoding="utf-8")
    rollout_speed_scope.write_scope_report(
        rollout_speed_report=rollout,
        output=scope,
        accepted_by="maintainer",
        accepted_at="2026-06-06T12:00:00Z",
        decision_url="https://github.com/AbdelStark/GenoLeWM/issues/42#issuecomment-1",
        rationale="The current recurrent rollout benchmark missed the RFC-0004 speed target.",
        replacement_target="Report measured rollout speed until #42 accepts a new target.",
        command=_scope_report_command(rollout, scope),
    )
    scope_payload = json.loads(scope.read_text(encoding="utf-8"))
    scope_payload["generated_at"] = "2026-06-06 12:00:00"
    scope.write_text(json.dumps(scope_payload), encoding="utf-8")

    with pytest.raises(InputError, match="generated_at must be a UTC timestamp"):
        v02_benchmark_readiness.build_readiness_report(
            rollout_speed_report=rollout,
            rollout_speed_scope_report=scope,
        )


def test_vep_rows_require_confidence_intervals(tmp_path: Path) -> None:
    metrics = tmp_path / "eval_metrics.json"
    rollout = tmp_path / "rollout.ar_speed.json"
    efficiency = tmp_path / "efficiency_report.json"
    point_estimate = _metric("clinvar_coding", "auroc", 0.73, baseline=True)
    point_estimate.pop("ci_low")
    point_estimate.pop("ci_high")
    metrics.write_text(
        json.dumps(
            _metrics_payload(
                [
                    point_estimate,
                    _metric("clinvar_coding", "average_precision", 0.71, baseline=True),
                    _metric("clinvar_coding", "balanced_accuracy", 0.69, baseline=True),
                    _metric("clinvar_coding", "accuracy", 0.68, baseline=True),
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
    )

    assert report["ok"] is False
    rows = _rows_by_id(report)
    assert rows["clinvar_coding"]["status"] == "incomplete"
    assert rows["clinvar_coding"]["observed_values"]["auroc"] == 0.73
    assert rows["clinvar_coding"]["confidence_intervals_required"] is True
    assert rows["clinvar_coding"]["missing_confidence_intervals"] == ["auroc"]
    assert "auroc" not in rows["clinvar_coding"]["confidence_intervals"]
    assert "clinvar_coding" in report["missing_or_failed_benchmarks"]


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


def test_release_inputs_row_accepts_rollout_state_artifacts(tmp_path: Path) -> None:
    metrics = tmp_path / "rollout_metrics.json"
    payload = _metrics_payload(
        [_metric("rollout_phased_haplotypes", "cosine_similarity_mean", 0.91, baseline=True)],
        release_ready=True,
    )
    artifacts = payload["artifacts"]
    assert isinstance(artifacts, dict)
    artifacts.pop("baseline_scores")
    artifacts.pop("scores")
    artifacts.pop("labels")
    artifacts["rollout_states"] = "eval/rollout_states.jsonl"
    artifacts["baseline_rollout_states"] = "eval/rollout_states.jsonl"
    artifacts["rollout_state_examples_report"] = "eval/rollout_state_examples_report.json"
    artifacts["rollout_state_rows_report"] = "eval/rollout_state_rows_report.json"
    metrics.write_text(json.dumps(payload), encoding="utf-8")

    report = v02_benchmark_readiness.build_readiness_report(
        metrics_json=(metrics,),
        require_release_inputs=True,
    )

    rows = _rows_by_id(report)
    findings = rows["release_inputs"]["findings"]
    assert "a bench.rollout speed report is required" in findings
    assert "an efficiency_report.json artifact is required" in findings
    assert not any("scores+labels" in str(finding) for finding in findings)
    assert not any("baseline artifact" in str(finding) for finding in findings)
    assert not any("rollout generation provenance" in str(finding) for finding in findings)


def test_release_inputs_row_rejects_rollout_state_artifacts_without_generation_reports(
    tmp_path: Path,
) -> None:
    metrics = tmp_path / "rollout_metrics.json"
    payload = _metrics_payload(
        [_metric("rollout_phased_haplotypes", "cosine_similarity_mean", 0.91, baseline=True)],
        release_ready=True,
    )
    artifacts = payload["artifacts"]
    assert isinstance(artifacts, dict)
    artifacts.pop("baseline_scores")
    artifacts.pop("scores")
    artifacts.pop("labels")
    artifacts["rollout_states"] = "eval/rollout_states.jsonl"
    artifacts["baseline_rollout_states"] = "eval/rollout_states.jsonl"
    artifacts.pop("rollout_state_examples_report")
    artifacts.pop("rollout_state_rows_report")
    metrics.write_text(json.dumps(payload), encoding="utf-8")

    report = v02_benchmark_readiness.build_readiness_report(
        metrics_json=(metrics,),
        require_release_inputs=True,
    )

    rows = _rows_by_id(report)
    findings = "\n".join(rows["release_inputs"]["findings"])
    assert "rollout generation provenance" in findings
    assert "rollout_state_examples_report" in findings
    assert "rollout_state_rows_report" in findings


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
    assert str(tmp_path) not in json.dumps(payload["command"], sort_keys=True)
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
    if any(
        str(metric.get("split")) in {"rollout_phased_haplotypes", "rollout_synthetic_edit_chains"}
        for metric in metrics
    ):
        artifacts["rollout_states"] = "eval/rollout_states.jsonl"
        artifacts["baseline_rollout_states"] = "eval/rollout_states.jsonl"
        artifacts["rollout_state_examples_report"] = "eval/rollout_state_examples_report.json"
        artifacts["rollout_state_rows_report"] = "eval/rollout_state_rows_report.json"
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


def _failing_rollout_payload() -> dict[str, object]:
    return {
        "schema_version": "1.0.0",
        "generated_by": "bench.rollout",
        "ok": False,
        "commit": "abcdef1234567890",
        "command": ["python", "-m", "bench.rollout", "--k", "5", "--k", "20"],
        "rows": [
            {
                "horizon": 5,
                "measured_speedup": 1.8,
                "target_speedup": 2.0,
                "target_met": False,
            },
            {
                "horizon": 20,
                "measured_speedup": 1.9,
                "target_speedup": 5.0,
                "target_met": False,
            },
        ],
    }


def _scope_report_command(rollout: Path, scope: Path) -> tuple[str, ...]:
    return (
        "python",
        "-m",
        "tools.release.rollout_speed_scope",
        "--rollout-speed-report",
        str(rollout),
        "--output",
        str(scope),
    )


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
