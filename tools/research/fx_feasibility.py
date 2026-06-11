# SPDX-License-Identifier: Apache-2.0
"""Render the GenoLeWM-FX feasibility and kill-gate report."""

from __future__ import annotations

import argparse
import importlib
import importlib.util
import json
import math
import os
import random
import shutil
import sys
from collections.abc import Callable, Mapping, Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Final, cast

from geno_lewm.errors import GenoLeWMError, InputError, exit_code_for
from geno_lewm.provenance import sha256_file

SCHEMA_VERSION: Final = "1.0.0"
GENERATED_BY: Final = "tools.research.fx_feasibility"
DEFAULT_MANIFEST: Final = Path("configs/fx/feasibility_sources.json")
DEFAULT_OUTPUT_JSON: Final = Path("docs/research/fx-feasibility-report.json")
DEFAULT_OUTPUT_MD: Final = Path("docs/research/fx-feasibility-report.md")

JsonDict = dict[str, Any]
MetricFn = Callable[[Sequence[int], Sequence[float]], float]


def build_feasibility_report(
    *,
    manifest_path: Path = DEFAULT_MANIFEST,
    generated_at: str | None = None,
    run_public_source_probe: bool = False,
) -> JsonDict:
    """Build the machine-readable GenoLeWM-FX feasibility report."""
    manifest = _load_manifest(manifest_path)
    source_probe_rows: list[JsonDict] = []
    if run_public_source_probe:
        source_probe = _require_mapping(manifest.get("source_probe"), "source_probe")
        source_probe_rows = build_source_probe_rows(
            records=_load_huggingface_records(source_probe),
            source_probe=source_probe,
        )
    runtime_checks = _runtime_checks()
    blockers = _decision_blockers(
        manifest=manifest,
        source_probe_rows=source_probe_rows,
        runtime_checks=runtime_checks,
    )
    decision = "kill" if blockers else "go"
    return {
        "schema_version": SCHEMA_VERSION,
        "generated_by": GENERATED_BY,
        "generated_at": generated_at or _utc_now(),
        "decision": decision,
        "ok_to_continue": decision == "go",
        "epic_issue": manifest["epic_issue"],
        "contract_issue": manifest["contract_issue"],
        "probe_issue": manifest["probe_issue"],
        "manifest": {
            "path": _repo_relative(manifest_path),
            "sha256": sha256_file(manifest_path),
        },
        "sample_window": manifest["sample_window"],
        "runtime_checks": runtime_checks,
        "candidate_sources": manifest["candidate_sources"],
        "candidate_teachers": manifest["candidate_teachers"],
        "required_baselines": manifest["required_baselines"],
        "source_probe_rows": source_probe_rows,
        "blockers": blockers,
        "recommended_issue_actions": manifest["recommended_issue_actions"],
        "claim_boundary": (
            "The FX pivot has no model-quality claim, no clinical utility claim, no broad "
            "variant-effect prediction claim, and no useful-planning claim. The public output "
            "is a kill report unless a future source/teacher cache passes this contract."
        ),
    }


def build_source_probe_rows(
    *,
    records: Sequence[Mapping[str, Any]],
    source_probe: Mapping[str, Any],
) -> list[JsonDict]:
    """Measure simple source-only baselines on a public TraitGym slice."""
    holdout_chromosomes = {
        str(chrom) for chrom in _require_list(source_probe["holdout_chromosomes"])
    }
    train = [record for record in records if str(record.get("chrom")) not in holdout_chromosomes]
    eval_records = [record for record in records if str(record.get("chrom")) in holdout_chromosomes]
    if not train or not eval_records:
        raise InputError(
            "source probe split is empty",
            details={
                "train_rows": len(train),
                "eval_rows": len(eval_records),
                "holdout_chromosomes": sorted(holdout_chromosomes),
            },
        )
    train_labels = [_label(record) for record in train]
    eval_labels = [_label(record) for record in eval_records]
    rows = [
        _label_prior_row(
            train_labels=train_labels,
            eval_labels=eval_labels,
            train_rows=len(train),
            eval_rows=len(eval_records),
            source_probe=source_probe,
        )
    ]
    score_columns = _require_list(source_probe["score_columns"])
    for raw_score_column in score_columns:
        score_column = _require_mapping(raw_score_column, "source_probe.score_columns[]")
        train_scores = [_score(record, score_column) for record in train]
        eval_scores = [_score(record, score_column) for record in eval_records]
        threshold = _threshold_for_prevalence(train_scores, train_labels)
        predictions = [1.0 if score >= threshold else 0.0 for score in eval_scores]
        rows.append(
            _metric_row(
                baseline_id=str(score_column["id"]),
                description=str(score_column.get("description", score_column["id"])),
                train_rows=len(train),
                eval_rows=len(eval_records),
                eval_labels=eval_labels,
                scores=eval_scores,
                predictions=predictions,
                source_probe=source_probe,
            )
        )
    return rows


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
    """Render a human-facing Markdown report for docs."""
    decision = str(report["decision"])
    lines = [
        "# GenoLeWM-FX feasibility and kill report",
        "",
        f"Generated by `{report['generated_by']}` at `{report['generated_at']}`.",
        "",
        f"Parent epic: #{report['epic_issue']}. Contract: #{report['contract_issue']}. "
        f"Probe: #{report['probe_issue']}.",
        "",
        f"Decision: **{decision}**.",
        "",
        str(report["claim_boundary"]),
        "",
        "## Reproduce",
        "",
        "```bash",
        "uv run python -m tools.research.fx_feasibility \\",
        "  --manifest configs/fx/feasibility_sources.json \\",
        "  --run-public-source-probe \\",
        "  --output-json docs/research/fx-feasibility-report.json \\",
        "  --output-md docs/research/fx-feasibility-report.md",
        "```",
        "",
        "Machine-readable report: [fx-feasibility-report.json](fx-feasibility-report.json).",
        "",
        "## Blockers",
        "",
    ]
    blockers = _require_list(report["blockers"])
    for blocker in blockers:
        item = _require_mapping(blocker, "blockers[]")
        lines.append(f"- `{item['code']}`: {item['message']}")
    lines.extend(["", "## Public Source Probe", ""])
    source_rows = _require_list(report["source_probe_rows"])
    if not source_rows:
        lines.append("No public source-only probe rows were generated.")
    else:
        lines.extend(
            [
                "| Baseline | Rows | AUROC | AUPRC | Balanced accuracy |",
                "| --- | ---: | ---: | ---: | ---: |",
            ]
        )
        for raw_row in source_rows:
            row = _require_mapping(raw_row, "source_probe_rows[]")
            metrics = _metric_index(_require_list(row["metrics"]))
            lines.append(
                "| "
                f"`{row['baseline_id']}` | "
                f"{row['eval_rows']} | "
                f"{_metric_cell(metrics, 'auroc')} | "
                f"{_metric_cell(metrics, 'average_precision')} | "
                f"{_metric_cell(metrics, 'balanced_accuracy')} |"
            )
    lines.extend(
        [
            "",
            "These rows test whether the public TraitGym label slice is trivially saturated by "
            "simple metadata. They do not provide the required teacher residual target.",
            "",
            "## Candidate Teacher Status",
            "",
            "| Teacher | Allowed role | Bulk training admissible | Local runtime | Blocking reason |",
            "| --- | --- | ---: | ---: | --- |",
        ]
    )
    for raw_teacher in _require_list(report["candidate_teachers"]):
        teacher = _require_mapping(raw_teacher, "candidate_teachers[]")
        reasons = "; ".join(str(reason) for reason in _require_list(teacher["blocking_reasons"]))
        lines.append(
            "| "
            f"`{teacher['id']}` | {teacher['role_allowed_by_contract']} | "
            f"{teacher['bulk_training_admissible']} | {teacher['local_runtime_available']} | "
            f"{reasons} |"
        )
    lines.extend(
        [
            "",
            "## Recommended Issue Resolution",
            "",
            "| Issue | Action | Reason |",
            "| ---: | --- | --- |",
        ]
    )
    for raw_action in _require_list(report["recommended_issue_actions"]):
        action = _require_mapping(raw_action, "recommended_issue_actions[]")
        lines.append(f"| #{action['issue']} | `{action['action']}` | {action['reason']} |")
    lines.extend(
        [
            "",
            "## Final Interpretation",
            "",
            "The FX pivot is stopped before teacher-cache implementation, residual model work, "
            "Hugging Face sweeps, or a public demo. The public TraitGym complex-trait split is "
            "large enough for a cheap source-only probe, but this repo does not have a locked, "
            "checksum-addressed 10k-50k ref/alt teacher-delta cache from AlphaGenome, Borzoi, "
            "Enformer, ChromBPNet, or another functional teacher. Continuing would require "
            "private credentials, heavyweight teacher setup, or newly curated artifacts before "
            "the actual GenoLeWM-FX hypothesis could be tested.",
            "",
        ]
    )
    return "\n".join(lines)


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entrypoint."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--output-json", type=Path, default=DEFAULT_OUTPUT_JSON)
    parser.add_argument("--output-md", type=Path, default=DEFAULT_OUTPUT_MD)
    parser.add_argument("--generated-at")
    parser.add_argument("--run-public-source-probe", action="store_true")
    args = parser.parse_args(argv)
    try:
        report = build_feasibility_report(
            manifest_path=args.manifest,
            generated_at=args.generated_at,
            run_public_source_probe=args.run_public_source_probe,
        )
        write_report(report=report, output_json=args.output_json, output_md=args.output_md)
    except GenoLeWMError as exc:
        print(exc.to_json(), file=sys.stderr)
        return exit_code_for(exc)
    return 0


def _load_manifest(path: Path) -> JsonDict:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise InputError("FX feasibility manifest not found", details={"path": str(path)}) from exc
    if not isinstance(payload, dict):
        raise InputError("FX feasibility manifest must be a JSON object")
    manifest = cast(JsonDict, payload)
    if manifest.get("schema_version") != SCHEMA_VERSION:
        raise InputError(
            "unsupported FX feasibility manifest schema",
            details={"expected": SCHEMA_VERSION, "actual": manifest.get("schema_version")},
        )
    required = (
        "epic_issue",
        "contract_issue",
        "probe_issue",
        "sample_window",
        "candidate_sources",
        "candidate_teachers",
        "required_baselines",
        "recommended_issue_actions",
    )
    missing = [key for key in required if key not in manifest]
    if missing:
        raise InputError(
            "FX feasibility manifest is missing required keys", details={"missing": missing}
        )
    return manifest


def _load_huggingface_records(source_probe: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    datasets = cast(Any, importlib.import_module("datasets"))
    dataset = datasets.load_dataset(
        str(source_probe["dataset"]),
        str(source_probe["config"]),
        split=str(source_probe["split"]),
    )
    return [cast(Mapping[str, Any], row) for row in dataset]


def _runtime_checks() -> JsonDict:
    return {
        "alphagenome_api_key_present": bool(os.environ.get("ALPHAGENOME_API_KEY")),
        "alphagenome_importable": _module_available("alphagenome"),
        "alphagenome_research_importable": _module_available("alphagenome_research"),
        "tensorflow_importable": _module_available("tensorflow"),
        "jax_importable": _module_available("jax"),
        "torch_importable": _module_available("torch"),
        "datasets_importable": _module_available("datasets"),
        "nvidia_smi_available": shutil.which("nvidia-smi") is not None,
    }


def _decision_blockers(
    *,
    manifest: Mapping[str, Any],
    source_probe_rows: Sequence[Mapping[str, Any]],
    runtime_checks: Mapping[str, Any],
) -> list[JsonDict]:
    blockers: list[JsonDict] = []
    candidate_sources = [
        _require_mapping(item, "candidate_sources[]")
        for item in _require_list(manifest["candidate_sources"])
    ]
    teacher_delta_sources = [
        source
        for source in candidate_sources
        if source.get("teacher_delta_cache") == "public_reproducible"
    ]
    in_scale_sources = [
        source for source in candidate_sources if source.get("within_probe_scale") is True
    ]
    if not in_scale_sources:
        blockers.append(
            {
                "code": "no_public_probe_scale_source",
                "message": "No candidate source is locked inside the 10k-50k cheap-probe scale.",
            }
        )
    if not teacher_delta_sources:
        blockers.append(
            {
                "code": "no_public_teacher_delta_cache",
                "message": (
                    "No candidate source provides a public, checksum-addressed ref/alt "
                    "teacher-delta cache for the required 10k-50k probe."
                ),
            }
        )
    teachers = [
        _require_mapping(item, "candidate_teachers[]")
        for item in _require_list(manifest["candidate_teachers"])
    ]
    admissible_teachers = [
        teacher for teacher in teachers if teacher.get("bulk_training_admissible") is True
    ]
    if not admissible_teachers:
        blockers.append(
            {
                "code": "no_bulk_admissible_teacher",
                "message": (
                    "No teacher is currently admissible for bulk training-cache use under "
                    "the contract and local artifact state."
                ),
            }
        )
    if not source_probe_rows:
        blockers.append(
            {
                "code": "source_probe_not_run",
                "message": "The public source-only probe rows were not generated for this report.",
            }
        )
    if not (
        runtime_checks.get("alphagenome_api_key_present")
        or runtime_checks.get("alphagenome_importable")
        or runtime_checks.get("alphagenome_research_importable")
        or runtime_checks.get("tensorflow_importable")
    ):
        blockers.append(
            {
                "code": "local_teacher_runtime_missing",
                "message": (
                    "The local runtime has no AlphaGenome credential/package and no TensorFlow "
                    "or AlphaGenome Research teacher stack."
                ),
            }
        )
    return blockers


def _label_prior_row(
    *,
    train_labels: Sequence[int],
    eval_labels: Sequence[int],
    train_rows: int,
    eval_rows: int,
    source_probe: Mapping[str, Any],
) -> JsonDict:
    prevalence = _mean(eval_labels)
    train_prevalence = _mean(train_labels)
    predictions = [1.0 if train_prevalence >= 0.5 else 0.0 for _ in eval_labels]
    return {
        "baseline_id": "label_prior_no_teacher",
        "description": "No-teacher label-prior baseline.",
        "target_kind": "variant_label_not_teacher_delta",
        "train_rows": train_rows,
        "eval_rows": eval_rows,
        "dataset": f"{source_probe['dataset']}:{source_probe['config']}:{source_probe['split']}",
        "holdout_chromosomes": sorted(str(chrom) for chrom in source_probe["holdout_chromosomes"]),
        "metrics": [
            _static_metric("auroc", 0.5),
            _static_metric("average_precision", prevalence),
            _static_metric("balanced_accuracy", _balanced_accuracy(eval_labels, predictions)),
        ],
    }


def _metric_row(
    *,
    baseline_id: str,
    description: str,
    train_rows: int,
    eval_rows: int,
    eval_labels: Sequence[int],
    scores: Sequence[float],
    predictions: Sequence[float],
    source_probe: Mapping[str, Any],
) -> JsonDict:
    samples = int(source_probe.get("bootstrap_samples", 200))
    seed = int(source_probe.get("bootstrap_seed", 257))
    return {
        "baseline_id": baseline_id,
        "description": description,
        "target_kind": "variant_label_not_teacher_delta",
        "train_rows": train_rows,
        "eval_rows": eval_rows,
        "dataset": f"{source_probe['dataset']}:{source_probe['config']}:{source_probe['split']}",
        "holdout_chromosomes": sorted(str(chrom) for chrom in source_probe["holdout_chromosomes"]),
        "metrics": [
            _metric_with_ci("auroc", eval_labels, scores, _auroc, samples, seed),
            _metric_with_ci(
                "average_precision",
                eval_labels,
                scores,
                _average_precision,
                samples,
                seed + 1,
            ),
            _metric_with_ci(
                "balanced_accuracy",
                eval_labels,
                predictions,
                _balanced_accuracy,
                samples,
                seed + 2,
            ),
        ],
    }


def _metric_with_ci(
    name: str,
    labels: Sequence[int],
    values: Sequence[float],
    metric: MetricFn,
    samples: int,
    seed: int,
) -> JsonDict:
    value = metric(labels, values)
    low, high = _bootstrap_ci(labels, values, metric, samples=samples, seed=seed)
    return {"name": name, "value": value, "ci95": [low, high]}


def _static_metric(name: str, value: float) -> JsonDict:
    return {"name": name, "value": value, "ci95": [value, value]}


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


def _auroc(labels: Sequence[int], scores: Sequence[float]) -> float:
    positives = sum(labels)
    negatives = len(labels) - positives
    if positives == 0 or negatives == 0:
        raise InputError("AUROC requires both positive and negative labels")
    ranks = _average_ranks(scores)
    positive_rank_sum = sum(rank for label, rank in zip(labels, ranks, strict=True) if label == 1)
    return (positive_rank_sum - positives * (positives + 1) / 2) / (positives * negatives)


def _average_precision(labels: Sequence[int], scores: Sequence[float]) -> float:
    positives = sum(labels)
    if positives == 0:
        raise InputError("average precision requires positive labels")
    order = sorted(range(len(labels)), key=lambda index: (-scores[index], index))
    true_positives = 0
    precision_sum = 0.0
    for rank, index in enumerate(order, start=1):
        if labels[index] == 1:
            true_positives += 1
            precision_sum += true_positives / rank
    return precision_sum / positives


def _balanced_accuracy(labels: Sequence[int], predictions: Sequence[float]) -> float:
    true_positive = sum(
        1 for label, pred in zip(labels, predictions, strict=True) if label == 1 and pred >= 0.5
    )
    false_negative = sum(
        1 for label, pred in zip(labels, predictions, strict=True) if label == 1 and pred < 0.5
    )
    true_negative = sum(
        1 for label, pred in zip(labels, predictions, strict=True) if label == 0 and pred < 0.5
    )
    false_positive = sum(
        1 for label, pred in zip(labels, predictions, strict=True) if label == 0 and pred >= 0.5
    )
    sensitivity = (
        true_positive / (true_positive + false_negative) if true_positive + false_negative else 0.0
    )
    specificity = (
        true_negative / (true_negative + false_positive) if true_negative + false_positive else 0.0
    )
    return (sensitivity + specificity) / 2


def _average_ranks(values: Sequence[float]) -> list[float]:
    order = sorted(range(len(values)), key=lambda index: values[index])
    ranks = [0.0] * len(values)
    start = 0
    while start < len(order):
        end = start + 1
        while end < len(order) and values[order[end]] == values[order[start]]:
            end += 1
        average_rank = (start + 1 + end) / 2
        for offset in range(start, end):
            ranks[order[offset]] = average_rank
        start = end
    return ranks


def _threshold_for_prevalence(scores: Sequence[float], labels: Sequence[int]) -> float:
    prevalence = _mean(labels)
    if prevalence <= 0:
        return math.inf
    sorted_scores = sorted(scores, reverse=True)
    index = max(0, min(len(sorted_scores) - 1, math.ceil(len(sorted_scores) * prevalence) - 1))
    return sorted_scores[index]


def _score(record: Mapping[str, Any], score_column: Mapping[str, Any]) -> float:
    value = float(record[str(score_column["column"])])
    direction = str(score_column["direction"])
    if direction == "positive":
        return value
    if direction == "negative":
        return -value
    if direction == "negative_abs":
        return -abs(value)
    raise InputError("unsupported source-probe score direction", details={"direction": direction})


def _label(record: Mapping[str, Any]) -> int:
    label = int(record["label"])
    if label not in {0, 1}:
        raise InputError("source-probe labels must be binary", details={"label": label})
    return label


def _mean(values: Sequence[int]) -> float:
    if not values:
        raise InputError("cannot compute mean over empty sequence")
    return sum(values) / len(values)


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


def _module_available(name: str) -> bool:
    return importlib.util.find_spec(name) is not None


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
