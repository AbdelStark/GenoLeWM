# SPDX-License-Identifier: Apache-2.0
"""Run the GenoLeWM-FX Borzoi baseline and saturation gate."""

from __future__ import annotations

import argparse
import importlib
import json
import math
import random
import sys
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Final, cast

from geno_lewm.errors import GenoLeWMError, InputError, exit_code_for
from geno_lewm.provenance import sha256_file
from tools.research.fx_borzoi_cache import (
    DEFAULT_OUTPUT_MANIFEST as DEFAULT_CACHE_MANIFEST,
    read_cache_rows,
)
from tools.research.fx_feasibility import _auroc, _average_precision, _balanced_accuracy

SCHEMA_VERSION: Final = "1.0.0"
GENERATED_BY: Final = "tools.research.fx_borzoi_baselines"
DEFAULT_SOURCE_MANIFEST: Final = Path("configs/fx/borzoi_rescue_sources.json")
DEFAULT_OUTPUT_JSON: Final = Path("docs/research/fx-borzoi-baseline-report.json")
DEFAULT_OUTPUT_MD: Final = Path("docs/research/fx-borzoi-baseline-report.md")

JsonDict = dict[str, Any]
MetricFn = Callable[[Sequence[int], Sequence[float]], float]


def build_baseline_report(
    *,
    source_manifest_path: Path = DEFAULT_SOURCE_MANIFEST,
    cache_manifest_path: Path = DEFAULT_CACHE_MANIFEST,
    generated_at: str | None = None,
    cache_rows: Sequence[Mapping[str, Any]] | None = None,
) -> JsonDict:
    """Build the machine-readable baseline/saturation report."""
    source_manifest = _load_json(source_manifest_path, label="source manifest")
    cache_manifest = _load_json(cache_manifest_path, label="cache manifest")
    _assert_cache_ready(cache_manifest)
    gate = _baseline_gate(source_manifest)
    rows = list(cache_rows) if cache_rows is not None else read_cache_rows(cache_manifest_path)
    train_rows, holdout_rows = _split_rows(rows, holdout_split=str(gate["holdout_split"]))
    leakage_audit = _leakage_audit(train_rows=train_rows, holdout_rows=holdout_rows, rows=rows)
    baselines = _baseline_rows(train_rows=train_rows, holdout_rows=holdout_rows, gate=gate)
    strongest = _strongest_baseline(baselines, metric=str(gate["primary_metric"]))
    label_prior = _baseline_by_id(baselines, "label_prior_constant")
    direct_borzoi = _baseline_by_id(baselines, "direct_traitgym_native_borzoi")
    paired_comparisons = _paired_comparisons(
        holdout_rows=holdout_rows,
        strongest=strongest,
        label_prior=label_prior,
        direct_borzoi=direct_borzoi,
        gate=gate,
    )
    decision, blockers = _decision(
        strongest=strongest,
        label_prior=label_prior,
        direct_borzoi=direct_borzoi,
        leakage_audit=leakage_audit,
        gate=gate,
    )
    return {
        "schema_version": SCHEMA_VERSION,
        "generated_by": GENERATED_BY,
        "generated_at": generated_at or _utc_now(),
        "decision": decision,
        "ok_to_train_residual_model": decision == "go_residual_model",
        "epic_issue": source_manifest["epic_issue"],
        "baseline_issue": 270,
        "model_issue": 271,
        "source_manifest": {
            "path": _repo_relative(source_manifest_path),
            "sha256": sha256_file(source_manifest_path),
        },
        "cache_manifest": {
            "path": _repo_relative(cache_manifest_path),
            "sha256": sha256_file(cache_manifest_path),
            "cache_artifact": cache_manifest["cache_artifact"],
        },
        "gate": gate,
        "split_summary": _split_summary(train_rows=train_rows, holdout_rows=holdout_rows),
        "leakage_audit": leakage_audit,
        "carbon_baseline_status": "not_applicable_no_carbon_scores_in_cache",
        "fipip_exact_join_status": cache_manifest["fipip_exact_join_status"],
        "baselines": [_strip_internal_scores(baseline) for baseline in baselines],
        "strongest_simple_baseline": _strip_internal_scores(strongest),
        "paired_comparisons": paired_comparisons,
        "blockers": blockers,
        "recommended_issue_actions": _recommended_issue_actions(decision),
        "claim_boundary": (
            "This report is a baseline and saturation gate only. It is not a model-quality "
            "result, clinical result, deployment-readiness result, broad VEP superiority "
            "claim, broad Carbon comparison, useful-planning claim, ground-truth biology "
            "claim, or exact fipip overlap claim."
        ),
    }


def write_report(
    *,
    report: Mapping[str, Any],
    output_json: Path = DEFAULT_OUTPUT_JSON,
    output_md: Path = DEFAULT_OUTPUT_MD,
) -> None:
    """Write JSON and Markdown reports."""
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_md.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    output_md.write_text(render_markdown(report), encoding="utf-8")


def render_markdown(report: Mapping[str, Any]) -> str:
    """Render a human-facing baseline report."""
    strongest = _require_mapping(report["strongest_simple_baseline"], "strongest_simple_baseline")
    leakage = _require_mapping(report["leakage_audit"], "leakage_audit")
    gate = _require_mapping(report["gate"], "gate")
    lines = [
        "# GenoLeWM-FX Borzoi baseline and saturation report",
        "",
        f"Generated by `{report['generated_by']}` at `{report['generated_at']}`.",
        "",
        f"Parent epic: #{report['epic_issue']}. Baseline gate: #{report['baseline_issue']}. "
        f"Model gate: #{report['model_issue']}.",
        "",
        f"Decision: **{report['decision']}**.",
        "",
        str(report["claim_boundary"]),
        "",
        "## Reproduce",
        "",
        "```bash",
        "uv run python -m tools.research.fx_borzoi_baselines \\",
        "  --source-manifest configs/fx/borzoi_rescue_sources.json \\",
        "  --cache-manifest docs/research/fx-borzoi-cache-manifest.json \\",
        "  --output-json docs/research/fx-borzoi-baseline-report.json \\",
        "  --output-md docs/research/fx-borzoi-baseline-report.md",
        "```",
        "",
        "Machine-readable report: "
        "[fx-borzoi-baseline-report.json](fx-borzoi-baseline-report.json).",
        "",
        "## Baselines",
        "",
        "| Baseline | Family | AUROC | AUPRC | Balanced accuracy |",
        "| --- | --- | ---: | ---: | ---: |",
    ]
    for raw_baseline in _require_list(report["baselines"]):
        baseline = _require_mapping(raw_baseline, "baselines[]")
        metrics = _metric_index(_require_list(baseline["holdout_metrics"]))
        lines.append(
            "| "
            f"`{baseline['baseline_id']}` | {baseline['family']} | "
            f"{_metric_cell(metrics, 'auroc')} | "
            f"{_metric_cell(metrics, 'average_precision')} | "
            f"{_metric_cell(metrics, 'balanced_accuracy')} |"
        )
    strongest_metrics = _metric_index(_require_list(strongest["holdout_metrics"]))
    lines.extend(
        [
            "",
            "## Gate Decision",
            "",
            f"Strongest simple baseline: `{strongest['baseline_id']}` with "
            f"AUPRC `{float(strongest_metrics['average_precision']['value']):.4f}` and "
            f"AUROC `{float(strongest_metrics['auroc']['value']):.4f}`.",
            "",
            f"Saturation thresholds: AUPRC `{gate['saturation_average_precision']}`, "
            f"AUROC `{gate['saturation_auroc']}`.",
            "",
            "## Leakage Audit",
            "",
            "| Check | Value |",
            "| --- | --- |",
            f"| Duplicate variant keys | {leakage['duplicate_variant_keys']} |",
            f"| Train/holdout key overlap | {leakage['train_holdout_key_overlap']} |",
            f"| Train chromosomes | `{', '.join(leakage['train_chromosomes'])}` |",
            f"| Holdout chromosomes | `{', '.join(leakage['holdout_chromosomes'])}` |",
            f"| fipip exact join status | `{report['fipip_exact_join_status']}` |",
            f"| Carbon baseline status | `{report['carbon_baseline_status']}` |",
            "",
            "## Blockers",
            "",
        ]
    )
    blockers = _require_list(report["blockers"])
    if blockers:
        for blocker in blockers:
            item = _require_mapping(blocker, "blockers[]")
            lines.append(f"- `{item['code']}`: {item['message']}")
    else:
        lines.append("No #270 blockers remain; the residual-model issue may proceed.")
    lines.extend(
        [
            "",
            "## Recommended Issue Actions",
            "",
            "| Issue | Action | Reason |",
            "| ---: | --- | --- |",
        ]
    )
    for raw_action in _require_list(report["recommended_issue_actions"]):
        action = _require_mapping(raw_action, "recommended_issue_actions[]")
        lines.append(f"| #{action['issue']} | `{action['action']}` | {action['reason']} |")
    lines.append("")
    return "\n".join(lines)


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entrypoint."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-manifest", type=Path, default=DEFAULT_SOURCE_MANIFEST)
    parser.add_argument("--cache-manifest", type=Path, default=DEFAULT_CACHE_MANIFEST)
    parser.add_argument("--output-json", type=Path, default=DEFAULT_OUTPUT_JSON)
    parser.add_argument("--output-md", type=Path, default=DEFAULT_OUTPUT_MD)
    parser.add_argument("--generated-at")
    args = parser.parse_args(argv)
    try:
        report = build_baseline_report(
            source_manifest_path=args.source_manifest,
            cache_manifest_path=args.cache_manifest,
            generated_at=args.generated_at,
        )
        write_report(report=report, output_json=args.output_json, output_md=args.output_md)
    except GenoLeWMError as exc:
        print(exc.to_json(), file=sys.stderr)
        return exit_code_for(exc)
    return 0


def _baseline_rows(
    *,
    train_rows: Sequence[Mapping[str, Any]],
    holdout_rows: Sequence[Mapping[str, Any]],
    gate: Mapping[str, Any],
) -> list[JsonDict]:
    baselines = [
        _constant_baseline(
            baseline_id="constant_negative",
            family="constant",
            train_rows=train_rows,
            holdout_rows=holdout_rows,
            score_value=0.0,
            prediction_value=0.0,
            gate=gate,
        ),
        _constant_baseline(
            baseline_id="label_prior_constant",
            family="label_prior",
            train_rows=train_rows,
            holdout_rows=holdout_rows,
            score_value=_label_prevalence(train_rows),
            prediction_value=1.0 if _label_prevalence(train_rows) >= 0.5 else 0.0,
            gate=gate,
        ),
    ]
    baselines.extend(
        _oriented_score_baseline(feature, train_rows, holdout_rows, gate=gate)
        for feature in ("maf", "ld_score", "tss_dist")
    )
    baselines.append(
        _oriented_score_baseline(
            "borzoi_score",
            train_rows,
            holdout_rows,
            baseline_id="direct_traitgym_native_borzoi",
            family="direct_borzoi",
            gate=gate,
        )
    )
    baselines.append(
        _logistic_probe_baseline(
            baseline_id="source_logistic_probe",
            family="source_probe",
            features=("maf", "ld_score", "tss_dist"),
            train_rows=train_rows,
            holdout_rows=holdout_rows,
            gate=gate,
        )
    )
    baselines.append(
        _logistic_probe_baseline(
            baseline_id="borzoi_plus_source_logistic_probe",
            family="borzoi_probe",
            features=("borzoi_score", "maf", "ld_score", "tss_dist"),
            train_rows=train_rows,
            holdout_rows=holdout_rows,
            gate=gate,
        )
    )
    return baselines


def _constant_baseline(
    *,
    baseline_id: str,
    family: str,
    train_rows: Sequence[Mapping[str, Any]],
    holdout_rows: Sequence[Mapping[str, Any]],
    score_value: float,
    prediction_value: float,
    gate: Mapping[str, Any],
) -> JsonDict:
    train_labels = _labels(train_rows)
    holdout_labels = _labels(holdout_rows)
    train_scores = [score_value for _ in train_rows]
    holdout_scores = [score_value for _ in holdout_rows]
    train_predictions = [prediction_value for _ in train_rows]
    holdout_predictions = [prediction_value for _ in holdout_rows]
    return _baseline_result(
        baseline_id=baseline_id,
        family=family,
        features=(),
        train_labels=train_labels,
        train_scores=train_scores,
        train_predictions=train_predictions,
        holdout_labels=holdout_labels,
        holdout_scores=holdout_scores,
        holdout_predictions=holdout_predictions,
        threshold=score_value,
        selected_direction="constant",
        gate=gate,
    )


def _oriented_score_baseline(
    feature: str,
    train_rows: Sequence[Mapping[str, Any]],
    holdout_rows: Sequence[Mapping[str, Any]],
    *,
    baseline_id: str | None = None,
    family: str = "source_only",
    gate: Mapping[str, Any],
) -> JsonDict:
    train_labels = _labels(train_rows)
    raw_train = [_feature(row, feature) for row in train_rows]
    candidates = [(1.0, raw_train), (-1.0, [-value for value in raw_train])]
    direction, train_scores = max(
        candidates,
        key=lambda item: (_average_precision(train_labels, item[1]), _auroc(train_labels, item[1])),
    )
    holdout_scores = [direction * _feature(row, feature) for row in holdout_rows]
    threshold = _threshold_for_prevalence(train_scores, train_labels)
    return _baseline_result(
        baseline_id=baseline_id or f"{feature}_source_only",
        family=family,
        features=(feature,),
        train_labels=train_labels,
        train_scores=train_scores,
        train_predictions=[1.0 if score >= threshold else 0.0 for score in train_scores],
        holdout_labels=_labels(holdout_rows),
        holdout_scores=holdout_scores,
        holdout_predictions=[1.0 if score >= threshold else 0.0 for score in holdout_scores],
        threshold=threshold,
        selected_direction="positive" if direction > 0 else "negative",
        gate=gate,
    )


def _logistic_probe_baseline(
    *,
    baseline_id: str,
    family: str,
    features: Sequence[str],
    train_rows: Sequence[Mapping[str, Any]],
    holdout_rows: Sequence[Mapping[str, Any]],
    gate: Mapping[str, Any],
) -> JsonDict:
    linear_model = cast(Any, importlib.import_module("sklearn.linear_model"))
    pipeline = cast(Any, importlib.import_module("sklearn.pipeline"))
    preprocessing = cast(Any, importlib.import_module("sklearn.preprocessing"))
    model = pipeline.make_pipeline(
        preprocessing.StandardScaler(),
        linear_model.LogisticRegression(max_iter=1000, solver="liblinear", random_state=270),
    )
    train_matrix = [[_feature(row, feature) for feature in features] for row in train_rows]
    holdout_matrix = [[_feature(row, feature) for feature in features] for row in holdout_rows]
    train_labels = _labels(train_rows)
    holdout_labels = _labels(holdout_rows)
    model.fit(train_matrix, train_labels)
    train_scores = [float(value) for value in model.predict_proba(train_matrix)[:, 1]]
    holdout_scores = [float(value) for value in model.predict_proba(holdout_matrix)[:, 1]]
    return _baseline_result(
        baseline_id=baseline_id,
        family=family,
        features=features,
        train_labels=train_labels,
        train_scores=train_scores,
        train_predictions=[1.0 if score >= 0.5 else 0.0 for score in train_scores],
        holdout_labels=holdout_labels,
        holdout_scores=holdout_scores,
        holdout_predictions=[1.0 if score >= 0.5 else 0.0 for score in holdout_scores],
        threshold=0.5,
        selected_direction="fit_on_train",
        gate=gate,
    )


def _baseline_result(
    *,
    baseline_id: str,
    family: str,
    features: Sequence[str],
    train_labels: Sequence[int],
    train_scores: Sequence[float],
    train_predictions: Sequence[float],
    holdout_labels: Sequence[int],
    holdout_scores: Sequence[float],
    holdout_predictions: Sequence[float],
    threshold: float,
    selected_direction: str,
    gate: Mapping[str, Any],
) -> JsonDict:
    return {
        "baseline_id": baseline_id,
        "family": family,
        "features": list(features),
        "train_rows": len(train_labels),
        "holdout_rows": len(holdout_labels),
        "threshold": threshold,
        "selected_direction": selected_direction,
        "train_metrics": _metrics(
            labels=train_labels,
            scores=train_scores,
            predictions=train_predictions,
            gate=gate,
            include_ci=False,
        ),
        "holdout_metrics": _metrics(
            labels=holdout_labels,
            scores=holdout_scores,
            predictions=holdout_predictions,
            gate=gate,
            include_ci=True,
        ),
        "holdout_scores": list(holdout_scores),
    }


def _metrics(
    *,
    labels: Sequence[int],
    scores: Sequence[float],
    predictions: Sequence[float],
    gate: Mapping[str, Any],
    include_ci: bool,
) -> list[JsonDict]:
    samples = int(gate["bootstrap_samples"]) if include_ci else 0
    seed = int(gate["bootstrap_seed"])
    metric_inputs = (
        ("auroc", scores, _auroc, seed),
        ("average_precision", scores, _average_precision, seed + 1),
        ("balanced_accuracy", predictions, _balanced_accuracy, seed + 2),
    )
    output: list[JsonDict] = []
    for name, values, metric_fn, metric_seed in metric_inputs:
        value = metric_fn(labels, values)
        if include_ci:
            low, high = _bootstrap_ci(
                labels,
                values,
                metric_fn,
                samples=samples,
                seed=metric_seed,
            )
        else:
            low, high = value, value
        output.append({"name": name, "value": value, "ci95": [low, high]})
    return output


def _paired_comparisons(
    *,
    holdout_rows: Sequence[Mapping[str, Any]],
    strongest: Mapping[str, Any],
    label_prior: Mapping[str, Any],
    direct_borzoi: Mapping[str, Any],
    gate: Mapping[str, Any],
) -> list[JsonDict]:
    labels = _labels(holdout_rows)
    pairs = (
        ("strongest_vs_label_prior", strongest, label_prior),
        ("strongest_vs_direct_borzoi", strongest, direct_borzoi),
        ("direct_borzoi_vs_label_prior", direct_borzoi, label_prior),
    )
    comparisons: list[JsonDict] = []
    for comparison_id, left, right in pairs:
        left_scores = [float(value) for value in _require_list(left["holdout_scores"])]
        right_scores = [float(value) for value in _require_list(right["holdout_scores"])]
        comparisons.append(
            {
                "comparison_id": comparison_id,
                "left": left["baseline_id"],
                "right": right["baseline_id"],
                "metrics": [
                    _paired_metric_delta(
                        "auroc",
                        labels,
                        left_scores,
                        right_scores,
                        _auroc,
                        gate=gate,
                    ),
                    _paired_metric_delta(
                        "average_precision",
                        labels,
                        left_scores,
                        right_scores,
                        _average_precision,
                        gate=gate,
                    ),
                ],
            }
        )
    return comparisons


def _paired_metric_delta(
    name: str,
    labels: Sequence[int],
    left_scores: Sequence[float],
    right_scores: Sequence[float],
    metric_fn: MetricFn,
    *,
    gate: Mapping[str, Any],
) -> JsonDict:
    value = metric_fn(labels, left_scores) - metric_fn(labels, right_scores)
    low, high = _paired_bootstrap_ci(
        labels,
        left_scores,
        right_scores,
        metric_fn,
        samples=int(gate["bootstrap_samples"]),
        seed=int(gate["bootstrap_seed"]) + 11,
    )
    return {"name": name, "delta": value, "ci95": [low, high]}


def _decision(
    *,
    strongest: Mapping[str, Any],
    label_prior: Mapping[str, Any],
    direct_borzoi: Mapping[str, Any],
    leakage_audit: Mapping[str, Any],
    gate: Mapping[str, Any],
) -> tuple[str, list[JsonDict]]:
    blockers: list[JsonDict] = []
    strongest_metrics = _metric_index(_require_list(strongest["holdout_metrics"]))
    direct_metrics = _metric_index(_require_list(direct_borzoi["holdout_metrics"]))
    prior_metrics = _metric_index(_require_list(label_prior["holdout_metrics"]))
    if leakage_audit["duplicate_variant_keys"] or leakage_audit["train_holdout_key_overlap"]:
        blockers.append(
            {
                "code": "leakage_audit_failed",
                "message": "Duplicate keys or train/holdout key overlap were detected.",
            }
        )
    if float(strongest_metrics["auroc"]["value"]) >= float(gate["saturation_auroc"]):
        blockers.append(
            {
                "code": "simple_baseline_auroc_saturated",
                "message": "A simple baseline exceeded the AUROC saturation threshold.",
            }
        )
    if float(strongest_metrics["average_precision"]["value"]) >= float(
        gate["saturation_average_precision"]
    ):
        blockers.append(
            {
                "code": "simple_baseline_auprc_saturated",
                "message": "A simple baseline exceeded the AUPRC saturation threshold.",
            }
        )
    direct_delta = float(direct_metrics["average_precision"]["value"]) - float(
        prior_metrics["average_precision"]["value"]
    )
    if direct_delta < float(gate["minimum_direct_borzoi_ap_delta_vs_prior"]):
        blockers.append(
            {
                "code": "direct_borzoi_signal_too_weak",
                "message": "Direct Borzoi AUPRC does not clear the minimum lift over label prior.",
            }
        )
    return ("no_go_baseline_gate", blockers) if blockers else ("go_residual_model", blockers)


def _recommended_issue_actions(decision: str) -> list[JsonDict]:
    if decision == "go_residual_model":
        return [
            {
                "issue": 270,
                "action": "close-completed",
                "reason": "The baseline gate is not saturated and direct Borzoi has holdout signal.",
            },
            {
                "issue": 271,
                "action": "open-next",
                "reason": "A residual model may proceed against the strongest simple baseline.",
            },
        ]
    return [
        {
            "issue": 270,
            "action": "close-no-go",
            "reason": "The baseline/saturation gate failed; publish this report as the no-go artifact.",
        },
        {
            "issue": 271,
            "action": "close-not-planned",
            "reason": "A residual model must not proceed after a failed baseline gate.",
        },
    ]


def _strongest_baseline(baselines: Sequence[Mapping[str, Any]], *, metric: str) -> JsonDict:
    return dict(
        max(
            baselines,
            key=lambda baseline: float(
                _metric_index(_require_list(baseline["holdout_metrics"]))[metric]["value"]
            ),
        )
    )


def _baseline_by_id(baselines: Sequence[Mapping[str, Any]], baseline_id: str) -> JsonDict:
    for baseline in baselines:
        if baseline["baseline_id"] == baseline_id:
            return dict(baseline)
    raise InputError("required baseline missing", details={"baseline_id": baseline_id})


def _split_rows(
    rows: Sequence[Mapping[str, Any]],
    *,
    holdout_split: str,
) -> tuple[list[Mapping[str, Any]], list[Mapping[str, Any]]]:
    train = [row for row in rows if row.get("split") != holdout_split]
    holdout = [row for row in rows if row.get("split") == holdout_split]
    if not train or not holdout:
        raise InputError(
            "baseline split is empty",
            details={"train_rows": len(train), "holdout_rows": len(holdout)},
        )
    if len(set(_labels(train))) < 2 or len(set(_labels(holdout))) < 2:
        raise InputError("baseline train and holdout splits must both contain both classes")
    return train, holdout


def _split_summary(
    *,
    train_rows: Sequence[Mapping[str, Any]],
    holdout_rows: Sequence[Mapping[str, Any]],
) -> JsonDict:
    return {
        "train_rows": len(train_rows),
        "train_positive": sum(_labels(train_rows)),
        "train_negative": len(train_rows) - sum(_labels(train_rows)),
        "holdout_rows": len(holdout_rows),
        "holdout_positive": sum(_labels(holdout_rows)),
        "holdout_negative": len(holdout_rows) - sum(_labels(holdout_rows)),
    }


def _leakage_audit(
    *,
    train_rows: Sequence[Mapping[str, Any]],
    holdout_rows: Sequence[Mapping[str, Any]],
    rows: Sequence[Mapping[str, Any]],
) -> JsonDict:
    keys = [_variant_key(row) for row in rows]
    train_keys = {_variant_key(row) for row in train_rows}
    holdout_keys = {_variant_key(row) for row in holdout_rows}
    train_chromosomes = sorted({str(row["chrom"]) for row in train_rows}, key=_chrom_sort_key)
    holdout_chromosomes = sorted({str(row["chrom"]) for row in holdout_rows}, key=_chrom_sort_key)
    trait_counts = Counter(str(row.get("trait", "")) for row in rows)
    consequence_counts = Counter(str(row.get("consequence", "")) for row in rows)
    return {
        "duplicate_variant_keys": len(keys) - len(set(keys)),
        "train_holdout_key_overlap": len(train_keys & holdout_keys),
        "train_chromosomes": train_chromosomes,
        "holdout_chromosomes": holdout_chromosomes,
        "trait_counts_top": dict(trait_counts.most_common(12)),
        "consequence_counts_top": dict(consequence_counts.most_common(12)),
    }


def _assert_cache_ready(cache_manifest: Mapping[str, Any]) -> None:
    if cache_manifest.get("schema_version") != "1.0.0":
        raise InputError("unsupported Borzoi cache manifest schema")
    if cache_manifest.get("target_kind") != "teacher_derived_traitgym_native_borzoi_score":
        raise InputError("Borzoi cache target kind is unsupported")
    if int(cache_manifest.get("row_count", 0)) <= 0:
        raise InputError("Borzoi cache is empty")


def _baseline_gate(source_manifest: Mapping[str, Any]) -> JsonDict:
    gate = dict(_require_mapping(source_manifest["baseline_gate"], "baseline_gate"))
    required = (
        "primary_metric",
        "bootstrap_samples",
        "bootstrap_seed",
        "saturation_auroc",
        "saturation_average_precision",
        "minimum_direct_borzoi_ap_delta_vs_prior",
        "holdout_split",
    )
    missing = [key for key in required if key not in gate]
    if missing:
        raise InputError("baseline gate is missing required keys", details={"missing": missing})
    return gate


def _labels(rows: Sequence[Mapping[str, Any]]) -> list[int]:
    return [int(row["label"]) for row in rows]


def _label_prevalence(rows: Sequence[Mapping[str, Any]]) -> float:
    labels = _labels(rows)
    if not labels:
        raise InputError("cannot compute prevalence over empty rows")
    return sum(labels) / len(labels)


def _feature(row: Mapping[str, Any], feature: str) -> float:
    value = row.get(feature)
    if value is None:
        return 0.0
    score = float(value)
    if not math.isfinite(score):
        raise InputError(
            "non-finite baseline feature", details={"feature": feature, "value": value}
        )
    if feature == "tss_dist":
        return abs(score)
    return score


def _strip_internal_scores(baseline: Mapping[str, Any]) -> JsonDict:
    return {key: value for key, value in baseline.items() if key != "holdout_scores"}


def _threshold_for_prevalence(scores: Sequence[float], labels: Sequence[int]) -> float:
    prevalence = sum(labels) / len(labels)
    if prevalence <= 0:
        return math.inf
    sorted_scores = sorted(scores, reverse=True)
    index = max(0, min(len(sorted_scores) - 1, math.ceil(len(sorted_scores) * prevalence) - 1))
    return sorted_scores[index]


def _bootstrap_ci(
    labels: Sequence[int],
    values: Sequence[float],
    metric: MetricFn,
    *,
    samples: int,
    seed: int,
) -> tuple[float, float]:
    rng = random.Random(seed)
    n = len(labels)
    boot_values: list[float] = []
    for _ in range(samples):
        indices = [rng.randrange(n) for _ in range(n)]
        sample_labels = [labels[index] for index in indices]
        if len(set(sample_labels)) < 2:
            continue
        sample_values = [values[index] for index in indices]
        boot_values.append(metric(sample_labels, sample_values))
    if not boot_values:
        value = metric(labels, values)
        return value, value
    boot_values.sort()
    return _percentile(boot_values, 0.025), _percentile(boot_values, 0.975)


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
    boot_values: list[float] = []
    for _ in range(samples):
        indices = [rng.randrange(n) for _ in range(n)]
        sample_labels = [labels[index] for index in indices]
        if len(set(sample_labels)) < 2:
            continue
        left = [left_scores[index] for index in indices]
        right = [right_scores[index] for index in indices]
        boot_values.append(metric(sample_labels, left) - metric(sample_labels, right))
    if not boot_values:
        value = metric(labels, left_scores) - metric(labels, right_scores)
        return value, value
    boot_values.sort()
    return _percentile(boot_values, 0.025), _percentile(boot_values, 0.975)


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


def _metric_index(metrics: Sequence[Any]) -> dict[str, Mapping[str, Any]]:
    return {
        str(_require_mapping(metric, "metrics[]")["name"]): _require_mapping(metric, "metrics[]")
        for metric in metrics
    }


def _metric_cell(metrics: Mapping[str, Mapping[str, Any]], name: str) -> str:
    metric = metrics[name]
    value = float(metric["value"])
    low, high = [float(item) for item in _require_list(metric["ci95"])]
    return f"{value:.4f} [{low:.4f}, {high:.4f}]"


def _variant_key(row: Mapping[str, Any]) -> tuple[str, int, str, str]:
    return str(row["chrom"]), int(row["pos"]), str(row["ref"]).upper(), str(row["alt"]).upper()


def _chrom_sort_key(chrom: str) -> tuple[int, str]:
    if chrom.isdigit():
        return int(chrom), chrom
    return 10_000, chrom


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
