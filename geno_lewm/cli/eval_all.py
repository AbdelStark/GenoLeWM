# SPDX-License-Identifier: Apache-2.0
"""``geno-lewm-eval-all`` — aggregate measured metrics into a release report."""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path
from typing import Annotated, Any

import typer

from geno_lewm.cli._artifact_paths import (
    package_relative_artifact_path,
    require_package_relative_artifacts,
)
from geno_lewm.cli._dispatch import SharedOptions, finalize_shared, run_app, shared_option_decls
from geno_lewm.cli._eval_config import write_effective_eval_config
from geno_lewm.errors import InputError
from tools.release.eval_report import (
    REQUIRED_ARTIFACTS,
    EvalReportInput,
    load_report_input,
    parse_report_input,
    render_report,
)
from tools.release.v02_benchmark_readiness import require_v02_vep_benchmark_metrics

__all__ = ["app", "cli_main"]

app = typer.Typer(
    name="geno-lewm-eval-all",
    help="Aggregate measured metrics JSON artifacts and render the release eval report.",
    no_args_is_help=False,
    add_completion=True,
    pretty_exceptions_enable=False,
)

_S = shared_option_decls()


@app.callback(invoke_without_command=True)
def main(
    metrics_json: Annotated[
        list[Path] | None,
        typer.Option(
            "--metrics-json",
            help="Measured metrics JSON from geno-lewm-eval or a prior geno-lewm-eval-all aggregate.",
        ),
    ] = None,
    output_metrics: Annotated[
        Path | None,
        typer.Option(
            "--output-metrics",
            help="Destination for the aggregated metrics JSON.",
        ),
    ] = None,
    output_report: Annotated[
        Path | None,
        typer.Option(
            "--output-report",
            "--output",
            help="Destination for the rendered eval_report.md.",
        ),
    ] = None,
    require_v02_vep_metrics: Annotated[
        bool,
        typer.Option(
            "--require-v02-vep-metrics",
            help=("Fail unless the aggregate contains the v0.2 VEP metric rows required by #197."),
        ),
    ] = False,
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

    inputs = tuple(metrics_json or ())
    if not inputs:
        raise InputError("geno-lewm-eval-all requires at least one --metrics-json")
    metrics_output = _required_path("output-metrics", output_metrics)
    report_output = _required_path("output-report", output_report)

    effective_config_path = write_effective_eval_config(metrics_output, opts)
    aggregate_payload = _aggregate_metrics(
        inputs,
        eval_config_path=effective_config_path,
        artifact_base_dir=metrics_output.parent,
    )
    aggregate_input = parse_report_input(aggregate_payload)
    if require_v02_vep_metrics:
        require_v02_vep_benchmark_metrics((aggregate_input,))
    rendered = render_report(aggregate_input)
    metrics_output.parent.mkdir(parents=True, exist_ok=True)
    report_output.parent.mkdir(parents=True, exist_ok=True)
    metrics_output.write_text(
        json.dumps(aggregate_payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    report_output.write_text(rendered, encoding="utf-8")
    typer.echo(
        json.dumps(
            {
                "eval_config": str(effective_config_path),
                "metrics_json": str(metrics_output),
                "eval_report": str(report_output),
                "inputs": len(inputs),
                "metrics": len(aggregate_input.metrics),
            },
            sort_keys=True,
        )
    )


def _aggregate_metrics(
    paths: tuple[Path, ...],
    *,
    eval_config_path: Path,
    artifact_base_dir: Path,
) -> dict[str, object]:
    reports = tuple(load_report_input(path) for path in paths)
    first = reports[0]
    _require_shared_identity(reports)
    metrics = _merged_metrics(reports)
    artifacts = _merged_artifacts(
        paths,
        reports,
        eval_config_path=eval_config_path,
        artifact_base_dir=artifact_base_dir,
    )
    limitations = _dedupe_text(item for report in reports for item in report.limitations)
    negative_findings = _dedupe_text(
        item for report in reports for item in report.negative_findings
    )
    conclusions = _dedupe_text(item for report in reports for item in report.conclusions)
    conclusions.append(
        f"Aggregated {len(metrics)} measured metric rows from {len(paths)} metrics JSON artifact(s)."
    )
    return {
        "schema_version": first.schema_version,
        "generated_by": "geno-lewm-eval-all",
        "generated_at": max(report.generated_at for report in reports),
        "model_id": first.model_id,
        "model_release": first.model_release,
        "dataset_snapshot": first.dataset_snapshot,
        "commit": first.commit,
        "hardware": first.hardware,
        "metrics": metrics,
        "artifacts": artifacts,
        "limitations": limitations,
        "negative_findings": negative_findings,
        "conclusions": conclusions,
    }


def _require_shared_identity(reports: tuple[EvalReportInput, ...]) -> None:
    first = reports[0]
    for index, report in enumerate(reports[1:], start=2):
        mismatches = {
            "model_id": (first.model_id, report.model_id),
            "model_release": (first.model_release, report.model_release),
            "dataset_snapshot": (first.dataset_snapshot, report.dataset_snapshot),
            "commit": (first.commit, report.commit),
            "hardware": (first.hardware, report.hardware),
        }
        observed = {
            key: {"first": expected, "observed": value}
            for key, (expected, value) in mismatches.items()
            if expected != value
        }
        if observed:
            raise InputError(
                "metrics JSON artifacts do not share release identity",
                details={"input_index": index, "mismatches": observed},
            )


def _merged_metrics(reports: tuple[EvalReportInput, ...]) -> list[dict[str, object]]:
    seen: set[tuple[str, str]] = set()
    metrics: list[dict[str, object]] = []
    for report in reports:
        for metric in report.metrics:
            key = (metric.split, metric.name)
            if key in seen:
                raise InputError(
                    "duplicate metric row in aggregated eval inputs",
                    details={"split": metric.split, "name": metric.name},
                )
            seen.add(key)
            metrics.append(
                {
                    key: value
                    for key, value in dataclasses.asdict(metric).items()
                    if value is not None
                }
            )
    if not metrics:
        raise InputError("aggregated eval inputs produced no metrics")
    return metrics


def _merged_artifacts(
    paths: tuple[Path, ...],
    reports: tuple[EvalReportInput, ...],
    *,
    eval_config_path: Path,
    artifact_base_dir: Path,
) -> dict[str, str]:
    first_artifacts = dict(reports[0].artifacts)
    _require_portable_artifacts(first_artifacts, input_index=1)
    required = {key: first_artifacts[key] for key in REQUIRED_ARTIFACTS}
    artifacts: dict[str, str] = dict(required)
    artifacts["eval_config"] = _portable_artifact_path(
        eval_config_path,
        base_dir=artifact_base_dir,
        label="eval_config",
    )
    for index, (path, report) in enumerate(zip(paths, reports, strict=True), start=1):
        current = dict(report.artifacts)
        _require_portable_artifacts(current, input_index=index)
        for key, expected_value in required.items():
            if current.get(key) != expected_value:
                raise InputError(
                    "metrics JSON artifacts reference different core artifacts",
                    details={
                        "input_index": index,
                        "artifact": key,
                        "expected": expected_value,
                        "observed": current.get(key),
                    },
                )
        artifacts[f"metrics_input_{index}"] = _portable_artifact_path(
            path,
            base_dir=artifact_base_dir,
            label=f"metrics_input_{index}",
        )
        for key, value in current.items():
            if key in required:
                continue
            artifacts[f"input_{index}.{key}"] = value
    return artifacts


def _portable_artifact_path(path: Path, *, base_dir: Path, label: str) -> str:
    return package_relative_artifact_path(
        path,
        root_dir=base_dir,
        label=label,
        outside_message="eval-all artifact paths must stay inside the aggregate metrics directory",
        root_detail="base_dir",
        remediation="write aggregate metrics in the model package root and keep inputs under it",
    )


def _require_portable_artifacts(artifacts: dict[str, str], *, input_index: int) -> None:
    require_package_relative_artifacts(artifacts, input_index=input_index)


def _dedupe_text(values: Any) -> list[str]:
    seen: set[str] = set()
    output: list[str] = []
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        output.append(value)
    return output


def _required_path(name: str, value: Path | None) -> Path:
    if value is None:
        raise InputError(f"geno-lewm-eval-all requires --{name}")
    return value


def cli_main() -> int:
    return run_app(app)
