# SPDX-License-Identifier: Apache-2.0
"""``geno-lewm-rollout`` — aggregate measured rollout-fidelity state rows."""

from __future__ import annotations

import json
import math
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Annotated, Any

import typer

from geno_lewm._artifact_sources import ROLLOUT_STATES_GENERATED_BY
from geno_lewm.cli._artifact_paths import package_relative_artifact_path
from geno_lewm.cli._dispatch import SharedOptions, finalize_shared, run_app, shared_option_decls
from geno_lewm.cli._eval_config import write_effective_eval_config
from geno_lewm.errors import InputError
from geno_lewm.provenance import sha256_bytes

__all__ = ["app", "cli_main"]

EVAL_GENERATED_BY = "geno-lewm-eval"
DEFAULT_RECALL_K = 10

app = typer.Typer(
    name="geno-lewm-rollout",
    help=(
        "Aggregate measured latent rollout state rows into eval-compatible "
        "rollout-fidelity metrics."
    ),
    no_args_is_help=False,
    add_completion=True,
    pretty_exceptions_enable=False,
)

_S = shared_option_decls()


@dataclass(frozen=True, slots=True)
class RolloutStateRow:
    """One measured rollout-fidelity row."""

    row_id: str
    split: str
    horizon: int
    source_state: tuple[float, ...]
    predicted_state: tuple[float, ...]
    target_state: tuple[float, ...]
    target_rank: int
    baseline_target_rank: int


@dataclass(frozen=True, slots=True)
class RolloutSummary:
    """Aggregate metrics for one split or split/horizon group."""

    split: str
    horizon: int | None
    n: int
    row_identity: str
    cosine_similarity_mean: float
    naive_baseline_cosine_mean: float
    l2_distance_mean: float
    naive_baseline_l2_mean: float
    recall_at_k: float
    naive_baseline_recall_at_k: float


@app.callback(invoke_without_command=True)
def main(
    states_jsonl: Annotated[
        Path | None,
        typer.Option(
            "--states-jsonl",
            help=(
                "Measured rollout state JSONL with source_state, predicted_state, "
                "target_state, target_rank, and baseline_target_rank."
            ),
        ),
    ] = None,
    output_metrics: Annotated[
        Path | None,
        typer.Option(
            "--output-metrics",
            "--output",
            help="Destination metrics JSON accepted by geno-lewm-eval-all.",
        ),
    ] = None,
    artifact_root: Annotated[
        Path | None,
        typer.Option(
            "--artifact-root",
            help=(
                "Release package root used to record artifact paths in metrics JSON; "
                "defaults to --output-metrics parent."
            ),
        ),
    ] = None,
    recall_k: Annotated[
        int,
        typer.Option("--recall-k", help="K threshold for rollout Recall@k rows."),
    ] = DEFAULT_RECALL_K,
    model_id: Annotated[
        str | None,
        typer.Option("--model-id", help="sha256:<hex> model identifier for the report JSON."),
    ] = None,
    model_release: Annotated[
        str | None,
        typer.Option("--model-release", help="Model release id for the report JSON."),
    ] = None,
    dataset_snapshot: Annotated[
        str | None,
        typer.Option("--dataset-snapshot", help="Dataset snapshot id for the report JSON."),
    ] = None,
    commit: Annotated[
        str | None,
        typer.Option("--commit", help="Git commit used for the evaluated artifacts."),
    ] = None,
    hardware: Annotated[
        str | None,
        typer.Option("--hardware", help="Hardware/runtime description for the report JSON."),
    ] = None,
    checkpoint: Annotated[
        Path | None,
        typer.Option("--checkpoint", help="Checkpoint artifact path recorded in the report JSON."),
    ] = None,
    config_artifact: Annotated[
        Path | None,
        typer.Option("--config-artifact", help="Resolved config artifact path."),
    ] = None,
    dataset_manifest: Annotated[
        Path | None,
        typer.Option("--dataset-manifest", help="Dataset manifest artifact path."),
    ] = None,
    efficiency_report: Annotated[
        Path | None,
        typer.Option("--efficiency-report", help="Release efficiency report artifact path."),
    ] = None,
    config: Annotated[str | None, _S["config"]] = None,
    set_overrides: Annotated[list[str] | None, _S["set_overrides"]] = None,
    seed: Annotated[int | None, _S["seed"]] = None,
    deterministic: Annotated[bool, _S["deterministic"]] = False,
    log_level: Annotated[str, _S["log_level"]] = "info",
    log_dir: Annotated[str | None, _S["log_dir"]] = None,
    run_id: Annotated[str | None, _S["run_id"]] = None,
    wandb_project: Annotated[str | None, _S["wandb_project"]] = None,
    no_receipt: Annotated[bool, _S["no_receipt"]] = False,
    print_config: Annotated[bool, _S["print_config"]] = False,
    print_config_tree: Annotated[bool, _S["print_config_tree"]] = False,
    explain: Annotated[str | None, _S["explain"]] = None,
    quiet: Annotated[bool, _S["quiet"]] = False,
    no_banner: Annotated[bool, _S["no_banner"]] = False,
    version: Annotated[bool, _S["version"]] = False,
) -> None:
    opts: SharedOptions | None = finalize_shared(
        config=config,
        set_overrides=set_overrides,
        seed=seed,
        deterministic=deterministic,
        log_level=log_level,
        log_dir=log_dir,
        run_id=run_id,
        wandb_project=wandb_project,
        no_receipt=no_receipt,
        print_config=print_config,
        print_config_tree=print_config_tree,
        explain=explain,
        quiet=quiet,
        no_banner=no_banner,
        version=version,
        default_config_name="eval",
    )
    if opts is None:
        return

    states_path = _required_path("states-jsonl", states_jsonl)
    output_path = _required_path("output-metrics", output_metrics)
    if recall_k <= 0:
        raise InputError("--recall-k must be a positive integer")
    artifact_root_path = artifact_root if artifact_root is not None else output_path.parent
    effective_config_path = write_effective_eval_config(output_path, opts)
    artifacts = _report_artifact_paths(
        artifact_root=artifact_root_path,
        checkpoint=_required_path("checkpoint", checkpoint),
        config=_required_path("config-artifact", config_artifact),
        dataset_manifest=_required_path("dataset-manifest", dataset_manifest),
        eval_config=effective_config_path,
        efficiency_report=_required_path("efficiency-report", efficiency_report),
        rollout_states=states_path,
    )
    rows = load_rollout_state_rows(states_path)
    payload = build_rollout_metrics_payload(
        rows,
        recall_k=recall_k,
        model_id=_required_text("model-id", model_id),
        model_release=_required_text("model-release", model_release),
        dataset_snapshot=_required_text("dataset-snapshot", dataset_snapshot),
        commit=_required_text("commit", commit),
        hardware=_required_text("hardware", hardware),
        artifacts=artifacts,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    payload_metrics = payload["metrics"]
    if not isinstance(payload_metrics, list):
        raise InputError("rollout metrics payload must contain a metrics list")
    typer.echo(
        json.dumps(
            {
                "metrics_json": str(output_path),
                "rows": len(rows),
                "splits": sorted({row.split for row in rows}),
                "metrics": len(payload_metrics),
                "recall_k": recall_k,
            },
            sort_keys=True,
        )
    )


def load_rollout_state_rows(path: Path) -> tuple[RolloutStateRow, ...]:
    """Load and validate measured rollout-state JSONL."""
    rows: list[RolloutStateRow] = []
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        raise InputError("failed to read rollout state JSONL", details={"path": str(path)}) from exc
    for line_no, line in enumerate(lines, start=1):
        if not line.strip():
            continue
        try:
            payload = json.loads(line)
        except json.JSONDecodeError as exc:
            raise InputError(
                "rollout state JSONL row is invalid",
                details={"path": str(path), "line": line_no, "column": exc.colno},
            ) from exc
        rows.append(_parse_rollout_row(payload, line_no=line_no))
    if not rows:
        raise InputError("rollout state JSONL must contain at least one measured row")
    duplicates = _duplicates(row.row_id for row in rows)
    if duplicates:
        raise InputError("rollout state row ids must be unique", details={"duplicates": duplicates})
    return tuple(rows)


def build_rollout_metrics_payload(
    rows: tuple[RolloutStateRow, ...],
    *,
    recall_k: int,
    model_id: str,
    model_release: str,
    dataset_snapshot: str,
    commit: str,
    hardware: str,
    artifacts: dict[str, str],
) -> dict[str, object]:
    """Build eval-compatible rollout-fidelity metrics from measured state rows."""
    split_summaries = [
        _summarize_group(split, split_rows, recall_k=recall_k)
        for split, split_rows in _group_by_split(rows).items()
    ]
    split_summaries.sort(key=lambda summary: summary.split)
    per_k_summaries = [
        _summarize_group(split, split_rows, recall_k=recall_k, horizon=horizon)
        for (split, horizon), split_rows in _group_by_split_horizon(rows).items()
    ]
    per_k_summaries.sort(key=lambda summary: (summary.split, summary.horizon or 0))
    metrics = [
        metric
        for summary in split_summaries
        for metric in _metrics_for_summary(summary, recall_k=recall_k)
    ]
    return {
        "schema_version": "1.0.0",
        "generated_by": EVAL_GENERATED_BY,
        "generated_at": _utc_now(),
        "model_id": model_id,
        "model_release": model_release,
        "dataset_snapshot": dataset_snapshot,
        "commit": commit,
        "hardware": hardware,
        "metrics": metrics,
        "artifacts": artifacts,
        "rollout_stratification": [
            _stratification_payload(summary, recall_k=recall_k) for summary in per_k_summaries
        ],
        "limitations": [
            (
                "Rollout-fidelity metrics are computed from measured latent-state rows; "
                "this command does not generate held-out haplotypes or run Carbon encoding."
            ),
            (
                f"Recall@{recall_k} uses target-rank evidence supplied by the measured "
                "rollout-state artifact."
            ),
        ],
        "negative_findings": [
            (
                "No clinical utility, privacy, deployment, or runtime-assurance claim is "
                "established by rollout-fidelity metrics."
            )
        ],
        "conclusions": _metric_conclusions(metrics),
    }


def _parse_rollout_row(payload: Any, *, line_no: int) -> RolloutStateRow:
    if not isinstance(payload, dict):
        raise InputError("rollout state rows must be JSON objects", details={"line": line_no})
    generated_by = _required_text_from_payload(payload, "generated_by", line_no=line_no)
    if generated_by != ROLLOUT_STATES_GENERATED_BY:
        raise InputError(
            "rollout state generated_by is invalid",
            details={
                "line": line_no,
                "expected": ROLLOUT_STATES_GENERATED_BY,
                "observed": generated_by,
            },
        )
    source_state = _state_vector(payload, "source_state", line_no=line_no)
    predicted_state = _state_vector(payload, "predicted_state", line_no=line_no)
    target_state = _state_vector(payload, "target_state", line_no=line_no)
    if len(source_state) != len(predicted_state) or len(source_state) != len(target_state):
        raise InputError(
            "rollout state vectors must share the same dimension",
            details={
                "line": line_no,
                "source_dim": len(source_state),
                "predicted_dim": len(predicted_state),
                "target_dim": len(target_state),
            },
        )
    return RolloutStateRow(
        row_id=_required_text_from_payload(payload, "id", line_no=line_no),
        split=_required_text_from_payload(payload, "split", line_no=line_no),
        horizon=_required_positive_int(payload, "k", line_no=line_no),
        source_state=source_state,
        predicted_state=predicted_state,
        target_state=target_state,
        target_rank=_required_positive_int(payload, "target_rank", line_no=line_no),
        baseline_target_rank=_required_positive_int(
            payload,
            "baseline_target_rank",
            line_no=line_no,
        ),
    )


def _summarize_group(
    split: str,
    rows: tuple[RolloutStateRow, ...],
    *,
    recall_k: int,
    horizon: int | None = None,
) -> RolloutSummary:
    if not rows:
        raise InputError("rollout metric groups must be non-empty")
    return RolloutSummary(
        split=split,
        horizon=horizon,
        n=len(rows),
        row_identity=_row_identity(split=split, horizon=horizon, rows=rows),
        cosine_similarity_mean=_mean(
            _cosine(row.predicted_state, row.target_state) for row in rows
        ),
        naive_baseline_cosine_mean=_mean(
            _cosine(row.source_state, row.target_state) for row in rows
        ),
        l2_distance_mean=_mean(_l2(row.predicted_state, row.target_state) for row in rows),
        naive_baseline_l2_mean=_mean(_l2(row.source_state, row.target_state) for row in rows),
        recall_at_k=_mean(1.0 if row.target_rank <= recall_k else 0.0 for row in rows),
        naive_baseline_recall_at_k=_mean(
            1.0 if row.baseline_target_rank <= recall_k else 0.0 for row in rows
        ),
    )


def _metrics_for_summary(summary: RolloutSummary, *, recall_k: int) -> list[dict[str, object]]:
    return [
        _metric_row(
            "cosine_similarity_mean",
            summary=summary,
            value=summary.cosine_similarity_mean,
            unit="cosine",
            higher_is_better=True,
            baseline_value=summary.naive_baseline_cosine_mean,
        ),
        _metric_row(
            "l2_distance_mean",
            summary=summary,
            value=summary.l2_distance_mean,
            unit="l2",
            higher_is_better=False,
            baseline_value=summary.naive_baseline_l2_mean,
        ),
        _metric_row(
            "recall_at_k",
            summary=summary,
            value=summary.recall_at_k,
            unit=f"recall@{recall_k}",
            higher_is_better=True,
            baseline_value=summary.naive_baseline_recall_at_k,
        ),
    ]


def _metric_row(
    name: str,
    *,
    summary: RolloutSummary,
    value: float,
    unit: str,
    higher_is_better: bool,
    baseline_value: float,
) -> dict[str, object]:
    return {
        "name": name,
        "value": value,
        "split": summary.split,
        "unit": unit,
        "higher_is_better": higher_is_better,
        "baseline": "source_state",
        "baseline_value": baseline_value,
        "delta_vs_baseline": value - baseline_value,
        "n": summary.n,
        "notes": "measured final-state rollout fidelity against encoded haplotype targets",
        "evaluated_variant_keys_sha256": summary.row_identity,
        "baseline_evaluated_variant_keys_sha256": summary.row_identity,
    }


def _stratification_payload(summary: RolloutSummary, *, recall_k: int) -> dict[str, object]:
    return {
        "split": summary.split,
        "k": summary.horizon,
        "n": summary.n,
        "row_identity": summary.row_identity,
        "cosine_similarity_mean": summary.cosine_similarity_mean,
        "naive_baseline_cosine_mean": summary.naive_baseline_cosine_mean,
        "l2_distance_mean": summary.l2_distance_mean,
        "naive_baseline_l2_mean": summary.naive_baseline_l2_mean,
        "recall_at_k": summary.recall_at_k,
        "naive_baseline_recall_at_k": summary.naive_baseline_recall_at_k,
        "recall_k": recall_k,
    }


def _metric_conclusions(metrics: list[dict[str, object]]) -> list[str]:
    conclusions: list[str] = []
    for metric in metrics:
        value = _metric_number(metric, "value")
        delta = _metric_number(metric, "delta_vs_baseline")
        conclusions.append(
            f"The {metric['name']} metric on {metric['split']} measured "
            f"{value:.6g} with delta {delta:.6g} versus source_state baseline."
        )
    return conclusions


def _metric_number(metric: dict[str, object], key: str) -> float:
    value = metric.get(key)
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise InputError(f"{key} must be numeric in rollout metric rows")
    return float(value)


def _report_artifact_paths(
    *,
    artifact_root: Path,
    checkpoint: Path,
    config: Path,
    dataset_manifest: Path,
    eval_config: Path,
    efficiency_report: Path,
    rollout_states: Path,
) -> dict[str, str]:
    artifacts = {
        "checkpoint": checkpoint,
        "config": config,
        "dataset_manifest": dataset_manifest,
        "eval_config": eval_config,
        "efficiency_report": efficiency_report,
        "rollout_states": rollout_states,
        "baseline_rollout_states": rollout_states,
    }
    return {
        label: package_relative_artifact_path(
            path,
            root_dir=artifact_root,
            label=label,
            outside_message="geno-lewm-rollout artifact paths must stay inside --artifact-root",
            root_detail="artifact_root",
            remediation=(
                "stage rollout artifacts under one release package root or pass "
                "--artifact-root pointing at that root"
            ),
        )
        for label, path in artifacts.items()
    }


def _group_by_split(rows: tuple[RolloutStateRow, ...]) -> dict[str, tuple[RolloutStateRow, ...]]:
    grouped: dict[str, list[RolloutStateRow]] = defaultdict(list)
    for row in rows:
        grouped[row.split].append(row)
    return {split: tuple(split_rows) for split, split_rows in grouped.items()}


def _group_by_split_horizon(
    rows: tuple[RolloutStateRow, ...],
) -> dict[tuple[str, int], tuple[RolloutStateRow, ...]]:
    grouped: dict[tuple[str, int], list[RolloutStateRow]] = defaultdict(list)
    for row in rows:
        grouped[(row.split, row.horizon)].append(row)
    return {key: tuple(split_rows) for key, split_rows in grouped.items()}


def _row_identity(
    *,
    split: str,
    horizon: int | None,
    rows: tuple[RolloutStateRow, ...],
) -> str:
    parts = [split, "*" if horizon is None else str(horizon), *sorted(row.row_id for row in rows)]
    return sha256_bytes("\n".join(parts).encode("utf-8"))


def _cosine(left: tuple[float, ...], right: tuple[float, ...]) -> float:
    left_norm = math.sqrt(sum(value * value for value in left))
    right_norm = math.sqrt(sum(value * value for value in right))
    if left_norm <= 0.0 or right_norm <= 0.0:
        raise InputError("rollout state vectors must have non-zero norm")
    return sum(a * b for a, b in zip(left, right, strict=True)) / (left_norm * right_norm)


def _l2(left: tuple[float, ...], right: tuple[float, ...]) -> float:
    return math.sqrt(sum((a - b) * (a - b) for a, b in zip(left, right, strict=True)))


def _mean(values: Any) -> float:
    items = tuple(float(value) for value in values)
    if not items:
        raise InputError("cannot compute a mean from an empty sequence")
    return sum(items) / len(items)


def _state_vector(payload: dict[str, object], key: str, *, line_no: int) -> tuple[float, ...]:
    raw = payload.get(key)
    if not isinstance(raw, list) or not raw:
        raise InputError(f"{key} must be a non-empty numeric list", details={"line": line_no})
    values: list[float] = []
    for index, item in enumerate(raw):
        if isinstance(item, bool) or not isinstance(item, int | float) or not math.isfinite(item):
            raise InputError(
                f"{key} entries must be finite numbers",
                details={"line": line_no, "index": index},
            )
        values.append(float(item))
    return tuple(values)


def _required_positive_int(payload: dict[str, object], key: str, *, line_no: int) -> int:
    value = payload.get(key)
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise InputError(f"{key} must be a positive integer", details={"line": line_no})
    return value


def _required_text_from_payload(payload: dict[str, object], key: str, *, line_no: int) -> str:
    value = payload.get(key)
    if not isinstance(value, str) or not value.strip():
        raise InputError(f"{key} must be a non-empty string", details={"line": line_no})
    return value.strip()


def _required_path(name: str, value: Path | None) -> Path:
    if value is None:
        raise InputError(f"geno-lewm-rollout requires --{name}")
    return value


def _required_text(name: str, value: str | None) -> str:
    if value is None or not value.strip():
        raise InputError(f"geno-lewm-rollout requires --{name}")
    return value.strip()


def _duplicates(values: Any) -> list[str]:
    seen: set[str] = set()
    duplicates: set[str] = set()
    for value in values:
        if value in seen:
            duplicates.add(value)
        seen.add(value)
    return sorted(duplicates)


def _utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def cli_main() -> int:
    return run_app(app)
