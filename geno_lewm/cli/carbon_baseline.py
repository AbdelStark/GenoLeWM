# SPDX-License-Identifier: Apache-2.0
"""``geno-lewm-carbon-baseline`` — write Carbon zero-shot baseline scores."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated

import typer

from geno_lewm.carbon_zero_shot import (
    load_carbon_logp_scorer,
    write_carbon_zero_shot_scores,
)
from geno_lewm.cli._artifact_paths import package_relative_artifact_path
from geno_lewm.cli._dispatch import SharedOptions, finalize_shared, run_app, shared_option_decls
from geno_lewm.encoder.windowing import DEFAULT_WINDOW_BP
from geno_lewm.errors import InputError

__all__ = ["app", "cli_main"]

app = typer.Typer(
    name="geno-lewm-carbon-baseline",
    help="Generate Carbon zero-shot baseline score JSONL for geno-lewm-eval.",
    no_args_is_help=False,
    add_completion=True,
    pretty_exceptions_enable=False,
)

_S = shared_option_decls()


@app.callback(invoke_without_command=True)
def main(
    vcf: Annotated[
        Path | None,
        typer.Option("--vcf", help="Held-out VCF to score with the Carbon baseline."),
    ] = None,
    fasta: Annotated[
        Path | None,
        typer.Option("--fasta", help="Local reference FASTA used for VCF window extraction."),
    ] = None,
    carbon_model_dir: Annotated[
        Path | None,
        typer.Option(
            "--carbon-model-dir",
            help="Local Carbon model directory with tokenizer/model files.",
        ),
    ] = None,
    output_scores: Annotated[
        Path | None,
        typer.Option(
            "--output-scores",
            "--output",
            help="Destination carbon_zero_shot_scores.jsonl artifact.",
        ),
    ] = None,
    metadata_output: Annotated[
        Path | None,
        typer.Option(
            "--metadata-output",
            help="Optional JSON summary for the generated baseline artifact.",
        ),
    ] = None,
    artifact_root: Annotated[
        Path | None,
        typer.Option(
            "--artifact-root",
            help=(
                "Release package root used to record metadata paths; when supplied, "
                "Carbon model, VCF, FASTA, output, cache, and metadata paths must stay inside it."
            ),
        ),
    ] = None,
    logp_cache_jsonl: Annotated[
        Path | None,
        typer.Option(
            "--logp-cache-jsonl",
            help="Optional sequence log-likelihood cache JSONL reused across runs.",
        ),
    ] = None,
    carbon_revision: Annotated[
        str,
        typer.Option("--carbon-revision", help="Carbon revision recorded in metadata."),
    ] = "main",
    dtype: Annotated[
        str,
        typer.Option("--dtype", help="Carbon model dtype: bf16, fp16, or fp32."),
    ] = "bf16",
    device: Annotated[
        str | None,
        typer.Option("--device", help="Optional torch device for Carbon scoring."),
    ] = None,
    window_bp: Annotated[
        int,
        typer.Option("--window-bp", help="Reference window length for baseline scoring."),
    ] = DEFAULT_WINDOW_BP,
    allow_network_download: Annotated[
        bool,
        typer.Option(
            "--allow-network-download",
            help="Allow Transformers to fetch missing Carbon files; disabled by default.",
        ),
    ] = False,
    trust_remote_code: Annotated[
        bool,
        typer.Option(
            "--trust-remote-code",
            help="Pass trust_remote_code=True to Transformers for the Carbon model.",
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

    vcf_path = _required_path("vcf", vcf)
    fasta_path = _required_path("fasta", fasta)
    model_dir = _required_path("carbon-model-dir", carbon_model_dir)
    output_path = _required_path("output-scores", output_scores)
    _require_artifact_root_paths(
        artifact_root=artifact_root,
        vcf=vcf_path,
        fasta=fasta_path,
        carbon_model=model_dir,
        output_scores=output_path,
        logp_cache=logp_cache_jsonl,
        metadata_output=metadata_output,
    )
    scorer = load_carbon_logp_scorer(
        model_dir,
        revision=carbon_revision,
        dtype=dtype,
        device=device,
        trust_remote_code=trust_remote_code,
        local_files_only=not allow_network_download,
    )
    summary = write_carbon_zero_shot_scores(
        vcf_path=vcf_path,
        fasta_path=fasta_path,
        output_scores=output_path,
        scorer=scorer,
        carbon_model=str(model_dir),
        carbon_revision=carbon_revision,
        window_bp=window_bp,
        logp_cache_jsonl=logp_cache_jsonl,
        metadata_output=None,
        local_files_only=not allow_network_download,
    )
    payload = _summary_payload(
        summary.to_json_dict(),
        artifact_root=artifact_root,
        metadata_output=metadata_output,
    )
    if metadata_output is not None:
        metadata_output.parent.mkdir(parents=True, exist_ok=True)
        metadata_output.write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    typer.echo(json.dumps(payload, sort_keys=True))


def _required_path(name: str, value: Path | None) -> Path:
    if value is None:
        raise InputError(f"geno-lewm-carbon-baseline requires --{name}")
    return value


def _require_artifact_root_paths(
    *,
    artifact_root: Path | None,
    vcf: Path,
    fasta: Path,
    carbon_model: Path,
    output_scores: Path,
    logp_cache: Path | None,
    metadata_output: Path | None,
) -> None:
    if artifact_root is None:
        return
    paths = {
        "vcf": vcf,
        "fasta": fasta,
        "carbon_model": carbon_model,
        "output_scores": output_scores,
    }
    if logp_cache is not None:
        paths["logp_cache"] = logp_cache
    if metadata_output is not None:
        paths["metadata_output"] = metadata_output
    for label, path in paths.items():
        _release_artifact_path(path, artifact_root=artifact_root, label=label)


def _summary_payload(
    payload: dict[str, str | int | bool | None],
    *,
    artifact_root: Path | None,
    metadata_output: Path | None,
) -> dict[str, str | int | bool | None]:
    if artifact_root is None:
        return payload
    normalized = dict(payload)
    for key in ("carbon_model", "vcf", "fasta", "output_scores", "logp_cache"):
        value = normalized.get(key)
        if isinstance(value, str) and value:
            normalized[key] = _release_artifact_path(
                Path(value),
                artifact_root=artifact_root,
                label=key,
            )
    if metadata_output is not None:
        _release_artifact_path(
            metadata_output,
            artifact_root=artifact_root,
            label="metadata_output",
        )
    return normalized


def _release_artifact_path(path: Path, *, artifact_root: Path, label: str) -> str:
    return package_relative_artifact_path(
        path,
        root_dir=artifact_root,
        label=label,
        outside_message=(
            "geno-lewm-carbon-baseline metadata paths must stay inside --artifact-root"
        ),
        root_detail="artifact_root",
        remediation=(
            "stage Carbon baseline inputs and outputs under one release package root "
            "or omit --artifact-root for a non-release local run"
        ),
    )


def cli_main() -> int:
    return run_app(app)
