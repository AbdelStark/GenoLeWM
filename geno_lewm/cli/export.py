# SPDX-License-Identifier: Apache-2.0
"""``geno-lewm-export`` — export a trained checkpoint to deployable artifacts (RFC-0018 §3.3).

Phase 1 writes the ``predictor.safetensors`` + ``action_encoder.safetensors``
deploy artifacts (plus ``export_report.json``) from a training
``predictor_checkpoint.pt``. ONNX / Core ML / GGUF targets land with #67–#70.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated

import typer

from geno_lewm.cli._dispatch import SharedOptions, finalize_shared, run_app, shared_option_decls
from geno_lewm.deploy.export import export_checkpoint
from geno_lewm.errors import InputError

__all__ = ["app", "cli_main"]

app = typer.Typer(
    name="geno-lewm-export",
    help="Export a trained checkpoint to deployable safetensors artifacts (RFC-0018 §3.3).",
    no_args_is_help=False,
    add_completion=True,
    pretty_exceptions_enable=False,
)

_S = shared_option_decls()


@app.callback(invoke_without_command=True)
def main(
    checkpoint: Annotated[
        Path | None,
        typer.Option("--checkpoint", help="Trained predictor_checkpoint.pt to export."),
    ] = None,
    output_dir: Annotated[
        Path | None,
        typer.Option(
            "--output-dir",
            "--output",
            help="Destination model directory for deploy artifacts.",
        ),
    ] = None,
    overwrite: Annotated[
        bool,
        typer.Option("--overwrite", help="Replace existing exported artifacts if present."),
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
        default_config_name="train",
    )
    if opts is None:
        return

    checkpoint_path = _required_path("checkpoint", checkpoint)
    out_dir = _required_path("output-dir", output_dir)
    report = export_checkpoint(checkpoint_path, out_dir, overwrite=overwrite)
    typer.echo(json.dumps(report, sort_keys=True))


def _required_path(name: str, value: Path | None) -> Path:
    if value is None:
        raise InputError(f"geno-lewm-export requires --{name}")
    return value


def cli_main() -> int:
    return run_app(app)
