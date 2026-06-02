# SPDX-License-Identifier: Apache-2.0
"""``geno-lewm-eval`` — measure one score artifact against held-out labels."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated

import typer

from geno_lewm._artifact_sources import CARBON_ZERO_SHOT_GENERATED_BY, SCORE_JSONL_GENERATED_BY
from geno_lewm.cli._artifact_paths import package_relative_artifact_path
from geno_lewm.cli._dispatch import SharedOptions, finalize_shared, run_app, shared_option_decls
from geno_lewm.cli._eval_config import write_effective_eval_config
from geno_lewm.errors import InputError
from geno_lewm.evaluation import (
    DEFAULT_BOOTSTRAP_RESAMPLES,
    DEFAULT_BOOTSTRAP_SEED,
    DEFAULT_CI_LEVEL,
    DEFAULT_EVAL_SCORE_FIELD,
    DEFAULT_EVAL_THRESHOLD,
    _require_score_jsonl_generated_by,
    build_eval_report_payload,
    evaluate_score_labels,
)

__all__ = ["app", "cli_main"]

app = typer.Typer(
    name="geno-lewm-eval",
    help=(
        "Compute measured binary metrics from score JSONL and held-out ClinVar-style label JSONL."
    ),
    no_args_is_help=False,
    add_completion=True,
    pretty_exceptions_enable=False,
)

_S = shared_option_decls()


@app.callback(invoke_without_command=True)
def main(
    scores_jsonl: Annotated[
        Path | None,
        typer.Option("--scores-jsonl", help="Score JSONL from geno-lewm-score."),
    ] = None,
    labels_jsonl: Annotated[
        Path | None,
        typer.Option("--labels-jsonl", help="Held-out label JSONL with ClinVar labels."),
    ] = None,
    baseline_scores_jsonl: Annotated[
        Path | None,
        typer.Option(
            "--baseline-scores-jsonl",
            help="Optional measured baseline score JSONL evaluated on the same labels.",
        ),
    ] = None,
    output_metrics: Annotated[
        Path | None,
        typer.Option(
            "--output-metrics",
            "--output",
            help="Destination metrics JSON accepted by tools.release.eval_report.",
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
    score_field: Annotated[
        str,
        typer.Option("--score-field", help="Numeric score field in the score JSONL."),
    ] = DEFAULT_EVAL_SCORE_FIELD,
    baseline_score_field: Annotated[
        str,
        typer.Option(
            "--baseline-score-field",
            help="Numeric score field in --baseline-scores-jsonl.",
        ),
    ] = DEFAULT_EVAL_SCORE_FIELD,
    baseline_name: Annotated[
        str | None,
        typer.Option(
            "--baseline-name",
            help="Name recorded for the optional measured baseline artifact.",
        ),
    ] = None,
    threshold: Annotated[
        float,
        typer.Option("--threshold", help="Decision threshold for accuracy metrics."),
    ] = DEFAULT_EVAL_THRESHOLD,
    split: Annotated[
        str,
        typer.Option("--split", help="Name of the evaluated split."),
    ] = "eval_clinvar",
    bootstrap_resamples: Annotated[
        int,
        typer.Option(
            "--bootstrap-resamples",
            help="Stratified bootstrap resamples for metric confidence intervals; 0 omits CIs.",
        ),
    ] = DEFAULT_BOOTSTRAP_RESAMPLES,
    bootstrap_seed: Annotated[
        int,
        typer.Option("--bootstrap-seed", help="Seed for deterministic bootstrap intervals."),
    ] = DEFAULT_BOOTSTRAP_SEED,
    ci_level: Annotated[
        float,
        typer.Option("--ci-level", help="Confidence level for bootstrap intervals."),
    ] = DEFAULT_CI_LEVEL,
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

    scores_path = _required_path("scores-jsonl", scores_jsonl)
    labels_path = _required_path("labels-jsonl", labels_jsonl)
    output_path = _required_path("output-metrics", output_metrics)
    artifact_root_path = artifact_root if artifact_root is not None else output_path.parent
    resolved_baseline_name = None
    if baseline_scores_jsonl is not None:
        resolved_baseline_name = _required_text("baseline-name", baseline_name)
    effective_config_path = write_effective_eval_config(output_path, opts)
    report_artifacts = _report_artifact_paths(
        artifact_root=artifact_root_path,
        checkpoint=_required_path("checkpoint", checkpoint),
        config=_required_path("config-artifact", config_artifact),
        dataset_manifest=_required_path("dataset-manifest", dataset_manifest),
        eval_config=effective_config_path,
        efficiency_report=_required_path("efficiency-report", efficiency_report),
        scores=scores_path,
        labels=labels_path,
        baseline_scores=baseline_scores_jsonl,
    )
    _require_score_jsonl_generated_by(scores_path, expected=SCORE_JSONL_GENERATED_BY)
    result = evaluate_score_labels(
        scores_path,
        labels_path,
        score_field=score_field,
        threshold=threshold,
        split=split,
        bootstrap_resamples=bootstrap_resamples,
        bootstrap_seed=bootstrap_seed,
        ci_level=ci_level,
    )
    baseline_result = None
    if baseline_scores_jsonl is not None:
        expected_baseline_generated_by = _baseline_expected_generated_by(
            resolved_baseline_name or ""
        )
        if expected_baseline_generated_by is not None:
            _require_score_jsonl_generated_by(
                baseline_scores_jsonl,
                expected=expected_baseline_generated_by,
            )
        baseline_result = evaluate_score_labels(
            baseline_scores_jsonl,
            labels_path,
            score_field=baseline_score_field,
            threshold=threshold,
            split=split,
            bootstrap_resamples=bootstrap_resamples,
            bootstrap_seed=bootstrap_seed,
            ci_level=ci_level,
        )
    payload = build_eval_report_payload(
        result,
        model_id=_required_text("model-id", model_id),
        model_release=_required_text("model-release", model_release),
        dataset_snapshot=_required_text("dataset-snapshot", dataset_snapshot),
        commit=_required_text("commit", commit),
        hardware=_required_text("hardware", hardware),
        checkpoint=report_artifacts["checkpoint"],
        config=report_artifacts["config"],
        dataset_manifest=report_artifacts["dataset_manifest"],
        eval_config=report_artifacts["eval_config"],
        efficiency_report=report_artifacts["efficiency_report"],
        scores=report_artifacts["scores"],
        labels=report_artifacts["labels"],
        baseline_result=baseline_result,
        baseline_name=resolved_baseline_name,
        baseline_scores=report_artifacts.get("baseline_scores"),
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    typer.echo(
        json.dumps(
            {
                "metrics_json": str(output_path),
                **result.to_summary_dict(),
                **(
                    {}
                    if baseline_result is None
                    else {
                        "baseline_name": resolved_baseline_name,
                        "baseline_auroc": baseline_result.auroc,
                        "auroc_delta_vs_baseline": result.auroc - baseline_result.auroc,
                    }
                ),
            },
            sort_keys=True,
        )
    )


def _required_path(name: str, value: Path | None) -> Path:
    if value is None:
        raise InputError(f"geno-lewm-eval requires --{name}")
    return value


def _report_artifact_paths(
    *,
    artifact_root: Path,
    checkpoint: Path,
    config: Path,
    dataset_manifest: Path,
    eval_config: Path,
    efficiency_report: Path,
    scores: Path,
    labels: Path,
    baseline_scores: Path | None,
) -> dict[str, str]:
    artifacts = {
        "checkpoint": checkpoint,
        "config": config,
        "dataset_manifest": dataset_manifest,
        "eval_config": eval_config,
        "efficiency_report": efficiency_report,
        "scores": scores,
        "labels": labels,
    }
    if baseline_scores is not None:
        artifacts["baseline_scores"] = baseline_scores
    return {
        label: package_relative_artifact_path(
            path,
            root_dir=artifact_root,
            label=label,
            outside_message="geno-lewm-eval artifact paths must stay inside --artifact-root",
            root_detail="artifact_root",
            remediation=(
                "stage eval artifacts under one release package root or pass "
                "--artifact-root pointing at that root"
            ),
        )
        for label, path in artifacts.items()
    }


def _required_text(name: str, value: str | None) -> str:
    if value is None or not value.strip():
        raise InputError(f"geno-lewm-eval requires --{name}")
    return value.strip()


def _baseline_expected_generated_by(name: str) -> str | None:
    if name == "carbon_zero_shot":
        return CARBON_ZERO_SHOT_GENERATED_BY
    return None


def cli_main() -> int:
    return run_app(app)
