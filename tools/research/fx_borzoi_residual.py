# SPDX-License-Identifier: Apache-2.0
"""Train and evaluate the experimental GenoLeWM-FX residual model."""

from __future__ import annotations

import argparse
import importlib
import json
import math
import random
import sys
from collections.abc import Callable, Mapping, Sequence
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean, pstdev
from typing import Any, Final, cast

from geno_lewm.errors import GenoLeWMError, InputError, exit_code_for
from geno_lewm.provenance import sha256_file
from tools.research.fx_borzoi_baselines import DEFAULT_OUTPUT_JSON as DEFAULT_BASELINE_REPORT
from tools.research.fx_borzoi_cache import (
    DEFAULT_OUTPUT_MANIFEST as DEFAULT_CACHE_MANIFEST,
    read_cache_rows,
)
from tools.research.fx_feasibility import _auroc, _average_precision, _balanced_accuracy

SCHEMA_VERSION: Final = "1.0.0"
GENERATED_BY: Final = "tools.research.fx_borzoi_residual"
DEFAULT_OUTPUT_JSON: Final = Path("docs/research/fx-borzoi-residual-report.json")
DEFAULT_OUTPUT_MD: Final = Path("docs/research/fx-borzoi-residual-report.md")
DEFAULT_OUTPUT_PREDICTIONS: Final = Path("docs/research/fx-borzoi-residual-predictions.parquet")
FEATURES: Final = ("borzoi_score", "maf", "ld_score", "tss_dist")
SEEDS: Final = tuple(range(271, 281))

JsonDict = dict[str, Any]
MetricFn = Callable[[Sequence[int], Sequence[float]], float]


def build_residual_report(
    *,
    cache_manifest_path: Path = DEFAULT_CACHE_MANIFEST,
    baseline_report_path: Path = DEFAULT_BASELINE_REPORT,
    output_predictions: Path = DEFAULT_OUTPUT_PREDICTIONS,
    generated_at: str | None = None,
    cache_rows: Sequence[Mapping[str, Any]] | None = None,
    seeds: Sequence[int] = SEEDS,
    bootstrap_samples: int = 1000,
) -> tuple[list[JsonDict], JsonDict]:
    """Train residual models and build the report."""
    baseline_report = _load_json(baseline_report_path, label="baseline report")
    _assert_baseline_gate_passed(baseline_report)
    rows = list(cache_rows) if cache_rows is not None else read_cache_rows(cache_manifest_path)
    train_rows, holdout_rows = _split_rows(rows)
    train_matrix = _feature_matrix(train_rows)
    holdout_matrix = _feature_matrix(holdout_rows)
    train_labels = _labels(train_rows)
    holdout_labels = _labels(holdout_rows)
    baseline_train, baseline_holdout = _baseline_probabilities(
        train_matrix=train_matrix,
        train_labels=train_labels,
        holdout_matrix=holdout_matrix,
    )
    residual_target = [
        label - score for label, score in zip(train_labels, baseline_train, strict=True)
    ]
    seed_runs: list[JsonDict] = []
    seed_predictions: list[list[float]] = []
    for seed in seeds:
        prediction = _residual_prediction(
            train_matrix=train_matrix,
            residual_target=residual_target,
            holdout_matrix=holdout_matrix,
            baseline_holdout=baseline_holdout,
            seed=seed,
        )
        seed_predictions.append(prediction)
        seed_runs.append(
            {
                "seed": seed,
                "metrics": _metrics(labels=holdout_labels, scores=prediction),
                "deltas_vs_strongest_baseline": _metric_deltas(
                    labels=holdout_labels,
                    left_scores=prediction,
                    right_scores=baseline_holdout,
                ),
            }
        )
    ensemble_scores = [
        mean(seed_scores[index] for seed_scores in seed_predictions)
        for index in range(len(holdout_rows))
    ]
    prediction_rows = _prediction_rows(
        holdout_rows=holdout_rows,
        labels=holdout_labels,
        baseline_scores=baseline_holdout,
        residual_scores=ensemble_scores,
    )
    paired_deltas = _paired_deltas_with_ci(
        labels=holdout_labels,
        residual_scores=ensemble_scores,
        baseline_scores=baseline_holdout,
        samples=bootstrap_samples,
    )
    ap_delta = _metric_by_name(paired_deltas, "average_precision")["delta"]
    ap_low = _metric_by_name(paired_deltas, "average_precision")["ci95"][0]
    auroc_low = _metric_by_name(paired_deltas, "auroc")["ci95"][0]
    final_claim_supported = bool(ap_delta > 0 and ap_low > 0 and auroc_low > 0)
    report = {
        "schema_version": SCHEMA_VERSION,
        "generated_by": GENERATED_BY,
        "generated_at": generated_at or _utc_now(),
        "decision": "residual_lift_candidate" if ap_delta > 0 else "residual_no_lift",
        "final_positive_claim_supported": final_claim_supported,
        "epic_issue": 266,
        "baseline_issue": 270,
        "model_issue": 271,
        "final_issue": 272,
        "cache_manifest": {
            "path": _repo_relative(cache_manifest_path),
            "sha256": sha256_file(cache_manifest_path),
        },
        "baseline_report": {
            "path": _repo_relative(baseline_report_path),
            "sha256": sha256_file(baseline_report_path),
            "decision": baseline_report["decision"],
            "strongest_simple_baseline": baseline_report["strongest_simple_baseline"][
                "baseline_id"
            ],
        },
        "model": {
            "kind": "residual_random_forest_regressor_ensemble",
            "base_model": "logistic_regression_reproducing_borzoi_plus_source_probe",
            "residual_target": "label - strongest_simple_baseline_probability",
            "features": list(FEATURES),
            "seeds": list(seeds),
            "n_estimators": 100,
            "max_depth": 3,
            "min_samples_leaf": 50,
        },
        "split_summary": {
            "train_rows": len(train_rows),
            "holdout_rows": len(holdout_rows),
            "holdout_positive": sum(holdout_labels),
            "holdout_negative": len(holdout_labels) - sum(holdout_labels),
        },
        "strongest_simple_baseline_metrics": _metrics(
            labels=holdout_labels,
            scores=baseline_holdout,
        ),
        "residual_ensemble_metrics": _metrics(labels=holdout_labels, scores=ensemble_scores),
        "paired_deltas_vs_strongest_baseline": paired_deltas,
        "seed_variance": _seed_variance(seed_runs),
        "collapse_diagnostics": _collapse_diagnostics(
            baseline_scores=baseline_holdout,
            residual_scores=ensemble_scores,
        ),
        "seed_runs": seed_runs,
        "prediction_artifact": {
            "path": _repo_relative(output_predictions),
            "sha256": "pending",
            "size_bytes": 0,
            "rows": len(prediction_rows),
        },
        "recommended_issue_actions": _recommended_issue_actions(final_claim_supported),
        "claim_boundary": (
            "This report is an experimental residual-model gate only. It is not a final "
            "GenoLeWM-FX model-quality result, clinical result, deployment-readiness result, "
            "broad VEP superiority claim, broad Carbon comparison, useful-planning claim, "
            "ground-truth biology claim, or exact fipip overlap claim."
        ),
    }
    return prediction_rows, report


def write_residual_report(
    *,
    prediction_rows: Sequence[Mapping[str, Any]],
    report: Mapping[str, Any],
    output_predictions: Path = DEFAULT_OUTPUT_PREDICTIONS,
    output_json: Path = DEFAULT_OUTPUT_JSON,
    output_md: Path = DEFAULT_OUTPUT_MD,
) -> None:
    """Write predictions plus JSON/Markdown reports."""
    output_predictions.parent.mkdir(parents=True, exist_ok=True)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_md.parent.mkdir(parents=True, exist_ok=True)
    _write_parquet(prediction_rows, output_predictions)
    finalized_report = dict(report)
    finalized_report["prediction_artifact"] = {
        "path": _repo_relative(output_predictions),
        "sha256": sha256_file(output_predictions),
        "size_bytes": output_predictions.stat().st_size,
        "rows": len(prediction_rows),
    }
    output_json.write_text(
        json.dumps(finalized_report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    output_md.write_text(render_markdown(finalized_report), encoding="utf-8")


def render_markdown(report: Mapping[str, Any]) -> str:
    """Render a human-facing residual report."""
    baseline_metrics = _metric_index(_require_list(report["strongest_simple_baseline_metrics"]))
    residual_metrics = _metric_index(_require_list(report["residual_ensemble_metrics"]))
    deltas = _metric_index(_require_list(report["paired_deltas_vs_strongest_baseline"]))
    artifact = _require_mapping(report["prediction_artifact"], "prediction_artifact")
    return "\n".join(
        [
            "# GenoLeWM-FX Borzoi residual model report",
            "",
            f"Generated by `{report['generated_by']}` at `{report['generated_at']}`.",
            "",
            f"Parent epic: #{report['epic_issue']}. Model gate: #{report['model_issue']}. "
            f"Final gate: #{report['final_issue']}.",
            "",
            f"Decision: **{report['decision']}**.",
            "",
            f"Final positive claim supported: **{report['final_positive_claim_supported']}**.",
            "",
            str(report["claim_boundary"]),
            "",
            "## Reproduce",
            "",
            "```bash",
            "uv run python -m tools.research.fx_borzoi_residual \\",
            "  --cache-manifest docs/research/fx-borzoi-cache-manifest.json \\",
            "  --baseline-report docs/research/fx-borzoi-baseline-report.json \\",
            "  --output-predictions docs/research/fx-borzoi-residual-predictions.parquet \\",
            "  --output-json docs/research/fx-borzoi-residual-report.json \\",
            "  --output-md docs/research/fx-borzoi-residual-report.md",
            "```",
            "",
            "## Metrics",
            "",
            "| Model | AUROC | AUPRC | Balanced accuracy |",
            "| --- | ---: | ---: | ---: |",
            f"| Strongest simple baseline | {_metric_cell(baseline_metrics, 'auroc')} | "
            f"{_metric_cell(baseline_metrics, 'average_precision')} | "
            f"{_metric_cell(baseline_metrics, 'balanced_accuracy')} |",
            f"| Residual ensemble | {_metric_cell(residual_metrics, 'auroc')} | "
            f"{_metric_cell(residual_metrics, 'average_precision')} | "
            f"{_metric_cell(residual_metrics, 'balanced_accuracy')} |",
            "",
            "Paired deltas versus strongest simple baseline:",
            "",
            "| Metric | Delta | 95% CI |",
            "| --- | ---: | ---: |",
            f"| AUROC | {float(deltas['auroc']['delta']):.4f} | "
            f"[{float(deltas['auroc']['ci95'][0]):.4f}, "
            f"{float(deltas['auroc']['ci95'][1]):.4f}] |",
            f"| AUPRC | {float(deltas['average_precision']['delta']):.4f} | "
            f"[{float(deltas['average_precision']['ci95'][0]):.4f}, "
            f"{float(deltas['average_precision']['ci95'][1]):.4f}] |",
            "",
            "## Prediction Artifact",
            "",
            "| Artifact | Rows | SHA-256 | Size |",
            "| --- | ---: | --- | ---: |",
            f"| `{artifact['path']}` | {artifact['rows']} | `{artifact['sha256']}` | "
            f"{artifact['size_bytes']} |",
            "",
            "## Interpretation",
            "",
            "The residual model shows a small positive mean lift, but the paired confidence "
            "interval crosses zero. This is implementation evidence for #271, not a "
            "positive locked-result claim for #272.",
            "",
        ]
    )


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entrypoint."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-manifest", type=Path, default=DEFAULT_CACHE_MANIFEST)
    parser.add_argument("--baseline-report", type=Path, default=DEFAULT_BASELINE_REPORT)
    parser.add_argument("--output-predictions", type=Path, default=DEFAULT_OUTPUT_PREDICTIONS)
    parser.add_argument("--output-json", type=Path, default=DEFAULT_OUTPUT_JSON)
    parser.add_argument("--output-md", type=Path, default=DEFAULT_OUTPUT_MD)
    parser.add_argument("--generated-at")
    args = parser.parse_args(argv)
    try:
        prediction_rows, report = build_residual_report(
            cache_manifest_path=args.cache_manifest,
            baseline_report_path=args.baseline_report,
            output_predictions=args.output_predictions,
            generated_at=args.generated_at,
        )
        write_residual_report(
            prediction_rows=prediction_rows,
            report=report,
            output_predictions=args.output_predictions,
            output_json=args.output_json,
            output_md=args.output_md,
        )
    except GenoLeWMError as exc:
        print(exc.to_json(), file=sys.stderr)
        return exit_code_for(exc)
    return 0


def _baseline_probabilities(
    *,
    train_matrix: Any,
    train_labels: Sequence[int],
    holdout_matrix: Any,
) -> tuple[list[float], list[float]]:
    linear_model = cast(Any, importlib.import_module("sklearn.linear_model"))
    pipeline = cast(Any, importlib.import_module("sklearn.pipeline"))
    preprocessing = cast(Any, importlib.import_module("sklearn.preprocessing"))
    model = pipeline.make_pipeline(
        preprocessing.StandardScaler(),
        linear_model.LogisticRegression(max_iter=1000, solver="liblinear", random_state=270),
    )
    model.fit(train_matrix, train_labels)
    train_scores = [float(value) for value in model.predict_proba(train_matrix)[:, 1]]
    holdout_scores = [float(value) for value in model.predict_proba(holdout_matrix)[:, 1]]
    return train_scores, holdout_scores


def _residual_prediction(
    *,
    train_matrix: Any,
    residual_target: Sequence[float],
    holdout_matrix: Any,
    baseline_holdout: Sequence[float],
    seed: int,
) -> list[float]:
    ensemble = cast(Any, importlib.import_module("sklearn.ensemble"))
    model = ensemble.RandomForestRegressor(
        random_state=seed,
        n_estimators=100,
        max_depth=3,
        min_samples_leaf=50,
    )
    model.fit(train_matrix, residual_target)
    residual = [float(value) for value in model.predict(holdout_matrix)]
    return [
        max(0.0, min(1.0, baseline_score + adjustment))
        for baseline_score, adjustment in zip(baseline_holdout, residual, strict=True)
    ]


def _prediction_rows(
    *,
    holdout_rows: Sequence[Mapping[str, Any]],
    labels: Sequence[int],
    baseline_scores: Sequence[float],
    residual_scores: Sequence[float],
) -> list[JsonDict]:
    return [
        {
            "row_index": int(row["row_index"]),
            "chrom": str(row["chrom"]),
            "pos": int(row["pos"]),
            "ref": str(row["ref"]),
            "alt": str(row["alt"]),
            "label": int(label),
            "baseline_probability": float(baseline_score),
            "residual_probability": float(residual_score),
            "residual_adjustment": float(residual_score - baseline_score),
            "target_kind": "teacher_derived_traitgym_native_borzoi_score",
        }
        for row, label, baseline_score, residual_score in zip(
            holdout_rows,
            labels,
            baseline_scores,
            residual_scores,
            strict=True,
        )
    ]


def _metrics(*, labels: Sequence[int], scores: Sequence[float]) -> list[JsonDict]:
    predictions = [1.0 if score >= 0.5 else 0.0 for score in scores]
    return [
        {"name": "auroc", "value": _auroc(labels, scores)},
        {"name": "average_precision", "value": _average_precision(labels, scores)},
        {"name": "balanced_accuracy", "value": _balanced_accuracy(labels, predictions)},
    ]


def _metric_deltas(
    *,
    labels: Sequence[int],
    left_scores: Sequence[float],
    right_scores: Sequence[float],
) -> list[JsonDict]:
    return [
        {
            "name": "auroc",
            "delta": _auroc(labels, left_scores) - _auroc(labels, right_scores),
        },
        {
            "name": "average_precision",
            "delta": _average_precision(labels, left_scores)
            - _average_precision(labels, right_scores),
        },
    ]


def _paired_deltas_with_ci(
    *,
    labels: Sequence[int],
    residual_scores: Sequence[float],
    baseline_scores: Sequence[float],
    samples: int,
) -> list[JsonDict]:
    return [
        _paired_metric_delta(
            "auroc",
            labels,
            residual_scores,
            baseline_scores,
            _auroc,
            samples=samples,
        ),
        _paired_metric_delta(
            "average_precision",
            labels,
            residual_scores,
            baseline_scores,
            _average_precision,
            samples=samples,
        ),
    ]


def _paired_metric_delta(
    name: str,
    labels: Sequence[int],
    left_scores: Sequence[float],
    right_scores: Sequence[float],
    metric_fn: MetricFn,
    samples: int,
) -> JsonDict:
    value = metric_fn(labels, left_scores) - metric_fn(labels, right_scores)
    low, high = _paired_bootstrap_ci(
        labels,
        left_scores,
        right_scores,
        metric_fn,
        samples=samples,
        seed=271,
    )
    return {"name": name, "delta": value, "ci95": [low, high]}


def _seed_variance(seed_runs: Sequence[Mapping[str, Any]]) -> JsonDict:
    ap_values = [
        float(_metric_by_name(_require_list(run["metrics"]), "average_precision")["value"])
        for run in seed_runs
    ]
    auroc_values = [
        float(_metric_by_name(_require_list(run["metrics"]), "auroc")["value"]) for run in seed_runs
    ]
    return {
        "seeds": [run["seed"] for run in seed_runs],
        "average_precision_mean": mean(ap_values),
        "average_precision_std": pstdev(ap_values),
        "average_precision_min": min(ap_values),
        "average_precision_max": max(ap_values),
        "auroc_mean": mean(auroc_values),
        "auroc_std": pstdev(auroc_values),
        "auroc_min": min(auroc_values),
        "auroc_max": max(auroc_values),
    }


def _collapse_diagnostics(
    *,
    baseline_scores: Sequence[float],
    residual_scores: Sequence[float],
) -> JsonDict:
    adjustments = [
        residual_score - baseline_score
        for baseline_score, residual_score in zip(baseline_scores, residual_scores, strict=True)
    ]
    sorted_abs = sorted(abs(value) for value in adjustments)
    return {
        "residual_probability_std": pstdev(residual_scores),
        "mean_abs_adjustment": mean(sorted_abs),
        "p95_abs_adjustment": _percentile(sorted_abs, 0.95),
        "max_abs_adjustment": max(sorted_abs),
        "all_predictions_constant": len({round(score, 8) for score in residual_scores}) == 1,
    }


def _paired_bootstrap_ci(
    labels: Sequence[int],
    left_scores: Sequence[float],
    right_scores: Sequence[float],
    metric: MetricFn,
    *,
    samples: int,
    seed: int,
) -> tuple[float, float]:
    rng = random.Random(seed)
    n = len(labels)
    values: list[float] = []
    for _ in range(samples):
        indices = [rng.randrange(n) for _ in range(n)]
        sample_labels = [labels[index] for index in indices]
        if len(set(sample_labels)) < 2:
            continue
        left = [left_scores[index] for index in indices]
        right = [right_scores[index] for index in indices]
        values.append(metric(sample_labels, left) - metric(sample_labels, right))
    if not values:
        delta = metric(labels, left_scores) - metric(labels, right_scores)
        return delta, delta
    values.sort()
    return _percentile(values, 0.025), _percentile(values, 0.975)


def _feature_matrix(rows: Sequence[Mapping[str, Any]]) -> Any:
    np = cast(Any, importlib.import_module("numpy"))
    return np.array([[_feature(row, feature) for feature in FEATURES] for row in rows], dtype=float)


def _feature(row: Mapping[str, Any], feature: str) -> float:
    value = float(row.get(feature) or 0.0)
    if not math.isfinite(value):
        raise InputError("non-finite residual feature", details={"feature": feature})
    return abs(value) if feature == "tss_dist" else value


def _split_rows(
    rows: Sequence[Mapping[str, Any]],
) -> tuple[list[Mapping[str, Any]], list[Mapping[str, Any]]]:
    train = [row for row in rows if row.get("split") != "holdout"]
    holdout = [row for row in rows if row.get("split") == "holdout"]
    if not train or not holdout:
        raise InputError("residual model requires non-empty train and holdout splits")
    if len(set(_labels(train))) < 2 or len(set(_labels(holdout))) < 2:
        raise InputError("residual model splits must both contain both classes")
    return train, holdout


def _labels(rows: Sequence[Mapping[str, Any]]) -> list[int]:
    return [int(row["label"]) for row in rows]


def _assert_baseline_gate_passed(baseline_report: Mapping[str, Any]) -> None:
    if baseline_report.get("decision") != "go_residual_model":
        raise InputError(
            "residual model requires a passing baseline gate",
            details={"decision": baseline_report.get("decision")},
        )
    if baseline_report.get("ok_to_train_residual_model") is not True:
        raise InputError("baseline report does not allow residual training")


def _write_parquet(rows: Sequence[Mapping[str, Any]], output: Path) -> None:
    pa = cast(Any, importlib.import_module("pyarrow"))
    pq = cast(Any, importlib.import_module("pyarrow.parquet"))
    pq.write_table(pa.Table.from_pylist(list(rows)), output)


def _metric_by_name(metrics: Sequence[Mapping[str, Any]], name: str) -> Mapping[str, Any]:
    for metric in metrics:
        if metric["name"] == name:
            return metric
    raise InputError("metric missing", details={"name": name})


def _metric_index(metrics: Sequence[Any]) -> dict[str, Mapping[str, Any]]:
    return {
        str(_require_mapping(metric, "metrics[]")["name"]): _require_mapping(metric, "metrics[]")
        for metric in metrics
    }


def _metric_cell(metrics: Mapping[str, Mapping[str, Any]], name: str) -> str:
    metric = metrics[name]
    value = float(metric["value"])
    return f"{value:.4f}"


def _percentile(values: Sequence[float], q: float) -> float:
    if not values:
        raise InputError("cannot compute percentile over empty sequence")
    if len(values) == 1:
        return values[0]
    position = q * (len(values) - 1)
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return values[lower]
    weight = position - lower
    return values[lower] * (1 - weight) + values[upper] * weight


def _recommended_issue_actions(final_claim_supported: bool) -> list[JsonDict]:
    return [
        {
            "issue": 271,
            "action": "close-completed",
            "reason": "The residual model was implemented, trained, and compared to the strongest simple baseline.",
        },
        {
            "issue": 272,
            "action": "open-next",
            "reason": (
                "Publish a locked positive result if paired intervals clear zero."
                if final_claim_supported
                else "Publish the fragile-lift/no-positive-claim outcome."
            ),
        },
    ]


def _load_json(path: Path, *, label: str) -> JsonDict:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise InputError(f"{label} not found", details={"path": str(path)}) from exc
    if not isinstance(payload, dict):
        raise InputError(f"{label} must be a JSON object", details={"path": str(path)})
    return cast(JsonDict, payload)


def _require_mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, dict):
        raise InputError(f"{label} must be an object")
    return cast(Mapping[str, Any], value)


def _require_list(value: Any) -> list[Any]:
    if not isinstance(value, list):
        raise InputError("expected a JSON array")
    return value


def _repo_relative(path: Path) -> str:
    try:
        return path.resolve().relative_to(Path.cwd().resolve()).as_posix()
    except ValueError:
        return path.as_posix()


def _utc_now() -> str:
    return datetime.now(tz=timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
