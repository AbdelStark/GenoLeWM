# SPDX-License-Identifier: Apache-2.0
"""``geno-lewm-prepare-clinvar`` — local ClinVar shard builder."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Annotated

import typer

from geno_lewm.cli._dispatch import SharedOptions, finalize_shared, run_app, shared_option_decls
from geno_lewm.cli._prepare_report import augment_prepare_report
from geno_lewm.data import prepare_clinvar_shard
from geno_lewm.errors import InputError

__all__ = ["app", "cli_main"]

app = typer.Typer(
    name="geno-lewm-prepare-clinvar",
    help=(
        "Build a schema-checked ClinVar Parquet shard from a local VCF/VCF.gz (data-pipeline contract)."
    ),
    no_args_is_help=False,
    add_completion=True,
    pretty_exceptions_enable=False,
)

_S = shared_option_decls()


@app.callback(invoke_without_command=True)
def main(
    input_vcf: Annotated[
        Path | None,
        typer.Option("--input-vcf", help="Local ClinVar VCF or VCF.gz file."),
    ] = None,
    output: Annotated[
        Path | None,
        typer.Option("--output", help="Dataset root; writes clinvar/<release>/variants.parquet."),
    ] = None,
    release: Annotated[
        str | None,
        typer.Option("--release", help="Pinned ClinVar release date, for example 2026-04-15."),
    ] = None,
    max_allele_len: Annotated[
        int,
        typer.Option("--max-allele-len", help="Maximum REF/ALT length retained for v1."),
    ] = 16,
    overwrite: Annotated[
        bool,
        typer.Option("--overwrite", help="Replace an existing shard instead of reusing it."),
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
    del opts
    if input_vcf is None:
        raise InputError("prepare-clinvar requires --input-vcf")
    if output is None:
        raise InputError("prepare-clinvar requires --output")
    if release is None:
        raise InputError("prepare-clinvar requires --release")
    started = time.perf_counter()
    report = prepare_clinvar_shard(
        input_vcf,
        output,
        release=release,
        max_allele_len=max_allele_len,
        overwrite=overwrite,
    )
    payload = augment_prepare_report(
        report.to_dict(),
        command="geno-lewm-prepare-clinvar",
        args=_command_args(
            input_vcf=input_vcf,
            output=output,
            release=release,
            max_allele_len=max_allele_len,
            overwrite=overwrite,
        ),
        input_vcf=input_vcf,
        output_path=report.output_path,
        elapsed_seconds=time.perf_counter() - started,
    )
    typer.echo(json.dumps(payload, sort_keys=True))


def _command_args(
    *,
    input_vcf: Path,
    output: Path,
    release: str,
    max_allele_len: int,
    overwrite: bool,
) -> list[str]:
    args = [
        "--input-vcf",
        str(input_vcf),
        "--output",
        str(output),
        "--release",
        release,
        "--max-allele-len",
        str(max_allele_len),
    ]
    if overwrite:
        args.append("--overwrite")
    return args


def cli_main() -> int:
    return run_app(app)
