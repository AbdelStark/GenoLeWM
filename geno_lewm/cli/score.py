# SPDX-License-Identifier: Apache-2.0
"""``geno-lewm-score`` — score a single variant or a VCF (RFC-0018 §3.3)."""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Annotated, Any

import typer

from geno_lewm.action import EditSpec
from geno_lewm.cli._dispatch import SharedOptions, finalize_shared, run_app, shared_option_decls
from geno_lewm.deploy import GenoLeWMRuntime
from geno_lewm.errors import InputError

__all__ = ["app", "cli_main"]

app = typer.Typer(
    name="geno-lewm-score",
    help="Score a single variant or a VCF using a local GenoLeWM model directory.",
    no_args_is_help=False,
    add_completion=True,
    pretty_exceptions_enable=False,
)

_S = shared_option_decls()


@app.callback(invoke_without_command=True)
def main(
    model_dir: Annotated[
        Path | None,
        typer.Option("--model-dir", help="Local model directory containing verified artifacts."),
    ] = None,
    backend: Annotated[
        str,
        typer.Option("--backend", help="Runtime backend: auto, cpu, cuda, onnx, or coreml."),
    ] = "auto",
    variant: Annotated[
        str | None,
        typer.Option("--variant", help="Single edit as CHROM:POS:REF:ALT."),
    ] = None,
    window: Annotated[
        str | None,
        typer.Option(
            "--window", help="Reference window for --variant; FASTA extraction is not hidden."
        ),
    ] = None,
    vcf: Annotated[
        Path | None,
        typer.Option("--vcf", help="Input VCF for batch scoring."),
    ] = None,
    fasta: Annotated[
        Path | None,
        typer.Option("--fasta", help="Local reference FASTA for --vcf scoring."),
    ] = None,
    output: Annotated[
        Path | None,
        typer.Option("--output", help="Output path for --vcf scores."),
    ] = None,
    batch_size: Annotated[
        int,
        typer.Option("--batch-size", help="Batch size for VCF scoring."),
    ] = 64,
    progress: Annotated[
        bool,
        typer.Option("--progress/--no-progress", help="Show progress for VCF scoring."),
    ] = True,
    receipt: Annotated[
        Path | None,
        typer.Option(
            "--receipt",
            help=(
                "Write a checksum receipt. With --variant this is one JSON file; "
                "with --vcf this is JSONL with one receipt per scored alternate."
            ),
        ),
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
        default_config_name="score",
    )
    if opts is None:
        return
    if opts.no_receipt and receipt is not None:
        raise InputError("choose either --receipt or --no-receipt, not both")

    if variant is not None and vcf is not None:
        raise InputError("choose either --variant or --vcf, not both")
    if variant is None and vcf is None:
        raise InputError("provide --variant for single-edit scoring or --vcf for batch scoring")

    runtime = _runtime(model_dir=model_dir, backend=backend)
    if variant is not None:
        edit = _parse_variant(variant)
        if window is None:
            raise InputError(
                "--variant requires --window until FASTA window extraction lands",
                remediation="pass the reference bases covering CHROM:POS:REF:ALT",
            )
        if receipt is None:
            result = runtime.score_variant(edit, window=window)
        else:
            result = runtime.score_variant(edit, window=window, receipt_path=receipt)
        typer.echo(
            json.dumps(
                {
                    "chrom": edit.chrom,
                    "pos": edit.pos,
                    "ref": edit.ref,
                    "alt": edit.alt,
                    **_result_payload(result),
                    **({} if receipt is None else {"receipt_path": str(receipt)}),
                },
                sort_keys=True,
                default=str,
            )
        )
        return

    if fasta is None:
        raise InputError("--vcf requires --fasta")
    if output is None:
        raise InputError("--vcf requires --output")
    if vcf is None:
        raise InputError("--vcf is required for batch scoring")
    runtime.score_vcf(
        vcf,
        fasta,
        output,
        batch_size=batch_size,
        progress=progress,
        receipt_path=receipt,
    )
    typer.echo(
        json.dumps(
            {
                "output_path": str(output),
                **({} if receipt is None else {"receipt_path": str(receipt)}),
            },
            sort_keys=True,
        )
    )


def _runtime(*, model_dir: Path | None, backend: str) -> GenoLeWMRuntime:
    if model_dir is None:
        raise InputError(
            "score requires --model-dir",
            remediation="provide a local verified model artifact directory",
        )
    return GenoLeWMRuntime(model_dir, backend=backend)


def _parse_variant(raw: str) -> EditSpec:
    parts = raw.split(":")
    if len(parts) != 4:
        raise InputError(
            "--variant must have the form CHROM:POS:REF:ALT",
            details={"variant": raw},
        )
    chrom, pos_text, ref, alt = parts
    try:
        pos = int(pos_text)
    except ValueError as exc:
        raise InputError(
            "--variant POS must be an integer",
            details={"variant": raw, "pos": pos_text},
        ) from exc
    return EditSpec(chrom=chrom, pos=pos, ref=ref.upper(), alt=alt.upper())


def _result_payload(result: Any) -> Mapping[str, Any]:
    to_dict = getattr(result, "to_dict", None)
    if callable(to_dict):
        payload = to_dict()
        if not isinstance(payload, Mapping):
            raise InputError(
                "score result to_dict() must return a mapping",
                details={"type": type(payload).__name__},
            )
        return payload
    if isinstance(result, Mapping):
        return result
    return {"result": result}


def cli_main() -> int:
    return run_app(app)
