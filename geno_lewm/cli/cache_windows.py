# SPDX-License-Identifier: Apache-2.0
"""``geno-lewm-cache-windows`` — window-cache management.

Issue #35 lands the repair and reindex flows for the Parquet shard
cache. Full cache construction over the Carbon-pretraining corpus
remains tracked by #36.
"""

from __future__ import annotations

import json
import time
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
from geno_lewm.encoder.cache import (
    CacheReindexReport,
    CacheRepairReport,
    default_cache_dir,
    reindex_cache,
    repair_cache,
)
from geno_lewm.errors import InputError
from geno_lewm.provenance import sha256_file

__all__ = [
    "app",
    "cli_main",
]

_SCHEMA_VERSION = "1.0.0"
_GENERATED_BY = "geno-lewm-cache-windows"
_EMBEDDINGS_DIR = "embeddings"
_QUARANTINE_DIR = ".quarantine"

app = typer.Typer(
    name="geno-lewm-cache-windows",
    help=("Build, repair, or reindex the window embedding cache (RFC-0002 §3.6 / RFC-0018 §3.3)."),
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
    json_report: Annotated[
        Path | None,
        typer.Option("--json-report", help="Write a machine-readable cache operation report."),
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
        default_config_name="train",
    )
    if opts is None:
        return
    del opts

    if reindex and repair:
        raise InputError("choose only one of --reindex or --repair")

    root = cache_dir if cache_dir is not None else default_cache_dir()
    if reindex:
        started = time.perf_counter()
        reindex_report = reindex_cache(root)
        if json_report is not None:
            _write_json_report(
                json_report,
                _reindex_payload(
                    root,
                    reindex_report,
                    elapsed_seconds=time.perf_counter() - started,
                ),
            )
        typer.echo(
            "reindexed "
            f"indexed_shards={reindex_report.indexed_shards} "
            f"indexed_rows={reindex_report.indexed_rows} "
            f"index_path={reindex_report.index_path}"
        )
        return
    if repair:
        started = time.perf_counter()
        repair_report = repair_cache(root)
        if json_report is not None:
            _write_json_report(
                json_report,
                _repair_payload(
                    root,
                    repair_report,
                    elapsed_seconds=time.perf_counter() - started,
                ),
            )
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


def _write_json_report(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _reindex_payload(
    root: Path,
    report: CacheReindexReport,
    *,
    elapsed_seconds: float,
) -> dict[str, object]:
    return {
        "schema_version": _SCHEMA_VERSION,
        "generated_by": _GENERATED_BY,
        "operation": "reindex",
        "indexed_shards": report.indexed_shards,
        "indexed_rows": report.indexed_rows,
        "cache_artifacts": {
            "index": _file_identity(report.index_path, root=root),
            "shards": _active_shard_identities(root),
        },
        "throughput": _throughput(
            elapsed_seconds=elapsed_seconds,
            indexed_rows=report.indexed_rows,
        ),
    }


def _repair_payload(
    root: Path,
    report: CacheRepairReport,
    *,
    elapsed_seconds: float,
) -> dict[str, object]:
    return {
        "schema_version": _SCHEMA_VERSION,
        "generated_by": _GENERATED_BY,
        "operation": "repair",
        "checked_shards": report.checked_shards,
        "quarantined_shards": [_file_identity(path, root=root) for path in report.quarantined],
        "indexed_shards": report.reindex.indexed_shards,
        "indexed_rows": report.reindex.indexed_rows,
        "cache_artifacts": {
            "index": _file_identity(report.reindex.index_path, root=root),
            "shards": _active_shard_identities(root),
        },
        "throughput": _throughput(
            elapsed_seconds=elapsed_seconds,
            indexed_rows=report.reindex.indexed_rows,
        ),
    }


def _active_shard_identities(root: Path) -> list[dict[str, object]]:
    embeddings = root / _EMBEDDINGS_DIR
    if not embeddings.exists():
        return []
    return [
        _file_identity(path, root=root)
        for path in sorted(embeddings.rglob("*.parquet"))
        if _QUARANTINE_DIR not in path.relative_to(embeddings).parts
    ]


def _file_identity(path: Path, *, root: Path) -> dict[str, object]:
    return {
        "path": path.relative_to(root).as_posix(),
        "sha256": sha256_file(path),
        "size_bytes": path.stat().st_size,
    }


def _throughput(*, elapsed_seconds: float, indexed_rows: int) -> dict[str, object]:
    rate = None if elapsed_seconds <= 0.0 else indexed_rows / elapsed_seconds
    return {
        "elapsed_seconds": round(max(elapsed_seconds, 0.0), 6),
        "indexed_rows_per_second": rate,
    }


def cli_main() -> int:
    return run_app(app)
