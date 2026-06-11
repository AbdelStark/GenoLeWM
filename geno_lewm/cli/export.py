# SPDX-License-Identifier: Apache-2.0
"""``geno-lewm-export`` - export a trained checkpoint to deployable artifacts.

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

_SUPPORTED_TARGETS = frozenset({"safetensors"})
_KNOWN_TARGETS = ("safetensors", "onnx", "coreml", "gguf")
_SUPPORTED_QUANTIZATIONS = frozenset({"none"})
_KNOWN_QUANTIZATIONS = ("none", "int8", "int4")

app = typer.Typer(
    name="geno-lewm-export",
    help="Export a trained checkpoint to deployable safetensors artifacts (CLI contract).",
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
    target: Annotated[
        str,
        typer.Option(
            "--target",
            help=(
                "Export target. Only safetensors is implemented; ONNX, Core ML, "
                "and GGUF fail closed until #67-#69 land."
            ),
        ),
    ] = "safetensors",
    quantization: Annotated[
        str,
        typer.Option(
            "--quantization",
            help=(
                "Quantization mode. Only none is implemented; int8/int4 fail closed "
                "until #70 lands."
            ),
        ),
    ] = "none",
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
    _validate_export_request(target=target, quantization=quantization)
    report = export_checkpoint(checkpoint_path, out_dir, overwrite=overwrite)
    typer.echo(json.dumps(report, sort_keys=True))


def _required_path(name: str, value: Path | None) -> Path:
    if value is None:
        raise InputError(f"geno-lewm-export requires --{name}")
    return value


def _validate_export_request(*, target: str, quantization: str) -> None:
    normalized_target = target.strip().lower()
    normalized_quantization = quantization.strip().lower()
    if normalized_target not in _KNOWN_TARGETS:
        raise InputError(
            "unsupported export target",
            details={"target": target, "supported_targets": list(_KNOWN_TARGETS)},
        )
    if normalized_quantization not in _KNOWN_QUANTIZATIONS:
        raise InputError(
            "unsupported export quantization",
            details={
                "quantization": quantization,
                "supported_quantizations": list(_KNOWN_QUANTIZATIONS),
            },
        )
    if normalized_target not in _SUPPORTED_TARGETS:
        raise InputError(
            "export target is not implemented yet",
            details={
                "target": normalized_target,
                "implemented_targets": sorted(_SUPPORTED_TARGETS),
                "tracking_issues": ["#67", "#68", "#69"],
            },
            remediation="use --target safetensors or wait for the target-specific export issue",
        )
    if normalized_quantization not in _SUPPORTED_QUANTIZATIONS:
        raise InputError(
            "export quantization is not implemented yet",
            details={
                "quantization": normalized_quantization,
                "implemented_quantizations": sorted(_SUPPORTED_QUANTIZATIONS),
                "tracking_issues": ["#70"],
            },
            remediation="use --quantization none or wait for the quantization issue",
        )


def cli_main() -> int:
    return run_app(app)
