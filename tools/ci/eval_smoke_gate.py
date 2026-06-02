# SPDX-License-Identifier: Apache-2.0
"""Hosted fixture-backed eval smoke regression gate.

This gate exercises the public ``geno-lewm-eval`` and
``geno-lewm-eval-all`` CLI boundaries on generated score/label JSONL
artifacts. It is not first-experiment evidence: it exists to catch
evaluation plumbing regressions on hosted CI without private datasets,
released checkpoints, or accelerator access.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Final

from geno_lewm._artifact_sources import CARBON_ZERO_SHOT_GENERATED_BY, SCORE_JSONL_GENERATED_BY
from geno_lewm.cli import _dispatch, eval as eval_cli, eval_all as eval_all_cli
from tools.release.eval_report import EvalReportInput, load_report_input

GENERATED_BY: Final = "tools.ci.eval_smoke_gate"
DEFAULT_MIN_AUROC: Final = 0.95
DEFAULT_MIN_AVERAGE_PRECISION: Final = 0.95
DEFAULT_MIN_BALANCED_ACCURACY: Final = 0.95
DEFAULT_MIN_AUROC_DELTA: Final = 0.05
_MODEL_ID: Final = "sha256:" + "c" * 64


@dataclass(frozen=True, slots=True)
class EvalSmokeThresholds:
    """Regression thresholds enforced by the hosted eval smoke gate."""

    min_auroc: float = DEFAULT_MIN_AUROC
    min_average_precision: float = DEFAULT_MIN_AVERAGE_PRECISION
    min_balanced_accuracy: float = DEFAULT_MIN_BALANCED_ACCURACY
    min_auroc_delta_vs_baseline: float = DEFAULT_MIN_AUROC_DELTA


def run_eval_smoke_gate(
    *,
    work_dir: Path,
    summary_json: Path | None = None,
    thresholds: EvalSmokeThresholds | None = None,
) -> dict[str, object]:
    """Run the generated fixture eval smoke gate and write its summary JSON."""
    active_thresholds = thresholds or EvalSmokeThresholds()
    _prepare_work_dir(work_dir)
    release_dir = work_dir / "release"
    eval_dir = release_dir / "eval"
    model_dir = release_dir / "model"
    dataset_dir = release_dir / "dataset"
    for directory in (eval_dir, model_dir, dataset_dir):
        directory.mkdir(parents=True, exist_ok=True)

    scores = eval_dir / "clinvar_smoke.scores.jsonl"
    labels = eval_dir / "clinvar_smoke.labels.jsonl"
    baseline_scores = eval_dir / "clinvar_smoke.carbon_zero_shot.jsonl"
    _write_fixture_score_artifacts(scores, labels, baseline_scores)
    checkpoint, config, dataset_manifest, efficiency_report = _write_core_artifacts(
        model_dir,
        dataset_dir,
    )

    metrics_json = eval_dir / "clinvar_smoke.metrics.json"
    aggregate_metrics_json = eval_dir / "eval_metrics.json"
    eval_report = eval_dir / "eval_report.md"
    eval_rc = _dispatch.run_app(
        eval_cli.app,
        argv=[
            "--quiet",
            "--no-banner",
            "--scores-jsonl",
            str(scores),
            "--labels-jsonl",
            str(labels),
            "--baseline-scores-jsonl",
            str(baseline_scores),
            "--baseline-name",
            "carbon_zero_shot",
            "--baseline-score-field",
            "carbon_zero_shot_score",
            "--output-metrics",
            str(metrics_json),
            "--artifact-root",
            str(release_dir),
            "--model-id",
            _MODEL_ID,
            "--model-release",
            "geno-lewm-eval-smoke-v0",
            "--dataset-snapshot",
            "geno-lewm-eval-smoke-fixture-v0",
            "--commit",
            "evalsmokefixture",
            "--hardware",
            "hosted CPU eval smoke fixture",
            "--checkpoint",
            str(checkpoint),
            "--config-artifact",
            str(config),
            "--dataset-manifest",
            str(dataset_manifest),
            "--efficiency-report",
            str(efficiency_report),
            "--split",
            "clinvar_smoke_fixture",
            "--bootstrap-resamples",
            "25",
            "--bootstrap-seed",
            "123",
        ],
    )
    if eval_rc != 0:
        raise RuntimeError(f"geno-lewm-eval smoke command failed with exit code {eval_rc}")

    eval_all_rc = _dispatch.run_app(
        eval_all_cli.app,
        argv=[
            "--quiet",
            "--no-banner",
            "--metrics-json",
            str(metrics_json),
            "--output-metrics",
            str(aggregate_metrics_json),
            "--output-report",
            str(eval_report),
            "--run-id",
            "eval-smoke",
            "--deterministic",
            "--seed",
            "123",
        ],
    )
    if eval_all_rc != 0:
        raise RuntimeError(f"geno-lewm-eval-all smoke command failed with exit code {eval_all_rc}")

    report = load_report_input(aggregate_metrics_json)
    observed = _observed_metrics(report)
    regressions = _threshold_regressions(observed, active_thresholds)
    output = {
        "schema_version": "1.0.0",
        "generated_by": GENERATED_BY,
        "ok": not regressions,
        "thresholds": {
            "min_auroc": active_thresholds.min_auroc,
            "min_average_precision": active_thresholds.min_average_precision,
            "min_balanced_accuracy": active_thresholds.min_balanced_accuracy,
            "min_auroc_delta_vs_baseline": active_thresholds.min_auroc_delta_vs_baseline,
        },
        "observed": observed,
        "regressions": regressions,
        "artifacts": {
            "scores": _relative(scores, work_dir),
            "labels": _relative(labels, work_dir),
            "baseline_scores": _relative(baseline_scores, work_dir),
            "metrics_json": _relative(metrics_json, work_dir),
            "aggregate_metrics_json": _relative(aggregate_metrics_json, work_dir),
            "eval_report": _relative(eval_report, work_dir),
            "eval_config": _relative(eval_dir / "eval_config.effective.yaml", work_dir),
        },
        "real_model_path": {
            "status": "not_attempted",
            "reason": (
                "Hosted eval smoke uses generated public fixture artifacts only. "
                "Real checkpoint, dataset, rollout, and paper-eval evidence remain tracked by "
                "#53, #57, #101, #163, #164, and #165."
            ),
        },
        "claim_boundary": (
            "Fixture smoke regression gate only; this is not first-experiment or model evidence."
        ),
    }
    destination = summary_json or (work_dir / "eval_smoke_summary.json")
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return output


def main(argv: list[str] | None = None) -> int:
    """CLI entrypoint for hosted CI."""
    parser = _parser()
    args = parser.parse_args(argv)
    thresholds = EvalSmokeThresholds(
        min_auroc=args.min_auroc,
        min_average_precision=args.min_average_precision,
        min_balanced_accuracy=args.min_balanced_accuracy,
        min_auroc_delta_vs_baseline=args.min_auroc_delta_vs_baseline,
    )
    try:
        summary = run_eval_smoke_gate(
            work_dir=args.work_dir,
            summary_json=args.summary_json,
            thresholds=thresholds,
        )
    except Exception as exc:
        print(f"eval_smoke_gate: error: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(summary, sort_keys=True))
    return 0 if summary["ok"] else 1


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--work-dir", type=Path, default=Path(".eval-smoke"))
    parser.add_argument("--summary-json", type=Path)
    parser.add_argument("--min-auroc", type=float, default=DEFAULT_MIN_AUROC)
    parser.add_argument(
        "--min-average-precision",
        type=float,
        default=DEFAULT_MIN_AVERAGE_PRECISION,
    )
    parser.add_argument(
        "--min-balanced-accuracy",
        type=float,
        default=DEFAULT_MIN_BALANCED_ACCURACY,
    )
    parser.add_argument(
        "--min-auroc-delta-vs-baseline",
        type=float,
        default=DEFAULT_MIN_AUROC_DELTA,
    )
    return parser


def _prepare_work_dir(work_dir: Path) -> None:
    if work_dir.exists():
        shutil.rmtree(work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)


def _write_fixture_score_artifacts(scores: Path, labels: Path, baseline_scores: Path) -> None:
    rows = [
        ("1", 10, "A", "T", "P", 0.95, 0.65),
        ("1", 20, "C", "G", "LP", 0.90, 0.55),
        ("1", 30, "G", "A", "P", 0.85, 0.45),
        ("1", 40, "T", "C", "LP", 0.80, 0.35),
        ("1", 50, "A", "C", "P", 0.75, 0.25),
        ("1", 60, "G", "T", "B", 0.05, 0.60),
        ("1", 70, "C", "A", "LB", 0.10, 0.50),
        ("1", 80, "T", "G", "B", 0.15, 0.40),
        ("1", 90, "A", "G", "LB", 0.20, 0.30),
        ("1", 100, "C", "T", "B", 0.25, 0.20),
    ]
    _write_jsonl(
        scores,
        [
            {
                "chrom": chrom,
                "pos": pos,
                "ref": ref,
                "alt": alt,
                "sigma_calibrated": score,
                "generated_by": SCORE_JSONL_GENERATED_BY,
            }
            for chrom, pos, ref, alt, _label, score, _baseline in rows
        ],
    )
    _write_jsonl(
        labels,
        [
            {
                "chrom": chrom,
                "pos": pos,
                "ref": ref,
                "alt": alt,
                "clinical_significance": label,
            }
            for chrom, pos, ref, alt, label, _score, _baseline in rows
        ],
    )
    _write_jsonl(
        baseline_scores,
        [
            {
                "chrom": chrom,
                "pos": pos,
                "ref": ref,
                "alt": alt,
                "carbon_zero_shot_score": baseline,
                "generated_by": CARBON_ZERO_SHOT_GENERATED_BY,
            }
            for chrom, pos, ref, alt, _label, _score, baseline in rows
        ],
    )


def _write_core_artifacts(model_dir: Path, dataset_dir: Path) -> tuple[Path, Path, Path, Path]:
    checkpoint = model_dir / "predictor.safetensors"
    config = model_dir / "train_config.yaml"
    dataset_manifest = dataset_dir / "dataset_manifest.json"
    efficiency_report = model_dir / "efficiency_report.json"
    checkpoint.write_bytes(b"eval smoke fixture checkpoint bytes\n")
    config.write_text("run_id: eval-smoke\n", encoding="utf-8")
    dataset_manifest.write_text(
        json.dumps(
            {
                "schema_version": "1.0.0",
                "snapshot_id": "geno-lewm-eval-smoke-fixture-v0",
                "generated_by": GENERATED_BY,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    efficiency_report.write_text(
        json.dumps(
            {
                "schema_version": "1.0.0",
                "generated_by": GENERATED_BY,
                "limitations": ["Fixture efficiency placeholder for eval smoke artifact wiring."],
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    return checkpoint, config, dataset_manifest, efficiency_report


def _observed_metrics(report: EvalReportInput) -> dict[str, float | str]:
    by_name = {metric.name: metric for metric in report.metrics}
    auroc = by_name["auroc"]
    average_precision = by_name["average_precision"]
    balanced_accuracy = by_name["balanced_accuracy"]
    if auroc.delta_vs_baseline is None:
        raise RuntimeError("smoke AUROC metric is missing baseline delta")
    return {
        "split": auroc.split,
        "auroc": auroc.value,
        "average_precision": average_precision.value,
        "balanced_accuracy": balanced_accuracy.value,
        "auroc_delta_vs_baseline": auroc.delta_vs_baseline,
        "evaluated_variants": float(auroc.n or 0),
    }


def _threshold_regressions(
    observed: dict[str, float | str],
    thresholds: EvalSmokeThresholds,
) -> list[dict[str, float | str]]:
    checks = (
        ("auroc", thresholds.min_auroc),
        ("average_precision", thresholds.min_average_precision),
        ("balanced_accuracy", thresholds.min_balanced_accuracy),
        ("auroc_delta_vs_baseline", thresholds.min_auroc_delta_vs_baseline),
    )
    regressions: list[dict[str, float | str]] = []
    for metric, minimum in checks:
        value = observed[metric]
        if not isinstance(value, int | float):
            raise RuntimeError(f"observed metric {metric} is not numeric")
        if value < minimum:
            regressions.append({"metric": metric, "observed": float(value), "minimum": minimum})
    return regressions


def _write_jsonl(path: Path, records: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(record, sort_keys=True) + "\n" for record in records),
        encoding="utf-8",
    )


def _relative(path: Path, root: Path) -> str:
    return path.relative_to(root).as_posix()


if __name__ == "__main__":
    raise SystemExit(main())
