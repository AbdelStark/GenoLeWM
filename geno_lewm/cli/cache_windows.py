# SPDX-License-Identifier: Apache-2.0
"""``geno-lewm-cache-windows`` — window-cache management.

Issue #35 lands the repair and reindex flows for the Parquet shard
cache. Full cache construction over the Carbon-pretraining corpus
remains tracked by #36.
"""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer

from geno_lewm.cli._dispatch import (
    SharedOptions,
    finalize_shared,
    not_yet_implemented,
    run_app,
    shared_option_decls,
)
from geno_lewm.encoder.cache import default_cache_dir, reindex_cache, repair_cache
from geno_lewm.errors import InputError

__all__ = [
    "app",
    "cli_main",
]

app = typer.Typer(
    name="geno-lewm-cache-windows",
    help=(
        "Build, repair, or reindex the window embedding cache (encoder contract / CLI contract)."
    ),
    no_args_is_help=False,
    add_completion=True,
    pretty_exceptions_enable=False,
)

_S = shared_option_decls()


@app.callback(invoke_without_command=True)
def main(
    cache_dir: Annotated[
        Path | None,
        typer.Option("--cache-dir", help="Cache root; defaults to $GENO_LEWM_CACHE."),
    ] = None,
    reindex: Annotated[
        bool,
        typer.Option("--reindex", help="Rebuild embeddings/index.sqlite from Parquet shards."),
    ] = False,
    repair: Annotated[
        bool,
        typer.Option("--repair", help="Quarantine unreadable Parquet shards and reindex."),
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

    if reindex and repair:
        raise InputError("choose only one of --reindex or --repair")

    root = cache_dir if cache_dir is not None else default_cache_dir()
    if reindex:
        reindex_report = reindex_cache(root)
        typer.echo(
            "reindexed "
            f"indexed_shards={reindex_report.indexed_shards} "
            f"indexed_rows={reindex_report.indexed_rows} "
            f"index_path={reindex_report.index_path}"
        )
        return
    if repair:
        repair_report = repair_cache(root)
        typer.echo(
            "repaired "
            f"checked_shards={repair_report.checked_shards} "
            f"quarantined={len(repair_report.quarantined)} "
            f"indexed_rows={repair_report.reindex.indexed_rows}"
        )
        return

    not_yet_implemented(
        command="cache-windows",
        issue="#36",
        detail="cache build mode is not yet implemented; --reindex and --repair are available",
    )


def cli_main() -> int:
    return run_app(app)
