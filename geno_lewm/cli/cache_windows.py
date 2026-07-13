# SPDX-License-Identifier: Apache-2.0
"""``geno-lewm-cache-windows`` — finite build, repair, and reindex flows.

Build mode is deliberately request-artifact scoped. Corpus-percentage and
24-hour throughput acceptance remain external measurements tracked by #36.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Annotated, Any, cast

import typer
import yaml

from geno_lewm.cli._dispatch import (
    SharedOptions,
    finalize_shared,
    run_app,
    shared_option_decls,
)
from geno_lewm.config import GenoLeWMConfig, config_to_dict, load_config
from geno_lewm.encoder import CarbonStateEncoder, build_window_cache
from geno_lewm.encoder.cache import (
    CacheReindexReport,
    CacheRepairReport,
    default_cache_dir,
    reindex_cache,
    repair_cache,
)
from geno_lewm.errors import InputError
from geno_lewm.observability import Severity, get_logger
from geno_lewm.provenance import Manifest, load_manifest, sha256_file

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
    json_report: Annotated[
        Path | None,
        typer.Option("--json-report", help="Write a machine-readable cache operation report."),
    ] = None,
    requests_jsonl: Annotated[
        Path | None,
        typer.Option(
            "--requests-jsonl",
            help=("Immutable JSONL rows with request_id, coordinates, window, and edit_locus."),
        ),
    ] = None,
    evidence_dir: Annotated[
        Path | None,
        typer.Option("--evidence-dir", help="Durable cache-build plan/state/report bundle."),
    ] = None,
    model_manifest: Annotated[
        Path | None,
        typer.Option("--model-manifest", help="Manifest committing the Carbon encoder identity."),
    ] = None,
    carbon_model_dir: Annotated[
        Path | None,
        typer.Option("--carbon-model-dir", help="Pinned local Carbon Transformers directory."),
    ] = None,
    batch_size: Annotated[
        int,
        typer.Option("--batch-size", help="Windows per Carbon encode_batch call."),
    ] = 8,
    rows_per_shard: Annotated[
        int,
        typer.Option("--rows-per-shard", help="Maximum unique cache keys per planned shard."),
    ] = 1_024,
    created_at_ns: Annotated[
        int | None,
        typer.Option(
            "--created-at-ns",
            help="Fixed positive UTC nanosecond timestamp persisted in every cache row.",
        ),
    ] = None,
    device: Annotated[
        str,
        typer.Option("--device", help="Carbon device (auto, cpu, cuda, or accelerator-specific)."),
    ] = "auto",
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
    build_inputs = (requests_jsonl, evidence_dir, model_manifest, carbon_model_dir, created_at_ns)
    if sum((reindex, repair)) > 1:
        raise InputError("choose only one of --reindex or --repair")
    if (reindex or repair) and any(value is not None for value in build_inputs):
        raise InputError("repair/reindex options cannot be combined with cache build inputs")

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

    missing = [
        option
        for option, value in (
            ("--requests-jsonl", requests_jsonl),
            ("--evidence-dir", evidence_dir),
            ("--model-manifest", model_manifest),
            ("--carbon-model-dir", carbon_model_dir),
            ("--created-at-ns", created_at_ns),
            ("--config", opts.config),
        )
        if value is None
    ]
    if missing:
        raise InputError(
            "cache build mode requires explicit immutable inputs",
            details={"missing": missing},
        )
    assert requests_jsonl is not None
    assert evidence_dir is not None
    assert model_manifest is not None
    assert carbon_model_dir is not None
    assert created_at_ns is not None
    assert opts.config is not None
    resolved_config = _resolve_build_config(
        config_path=Path(opts.config),
        set_overrides=opts.set_overrides,
        seed=opts.seed,
        deterministic=opts.deterministic,
        run_id=opts.run_id,
    )
    manifest = load_manifest(model_manifest)
    encoder = _build_encoder(
        config=resolved_config,
        manifest=manifest,
        carbon_model_dir=carbon_model_dir,
        device=device,
    )
    logger = get_logger(
        "cache-build",
        run_id=opts.run_id,
        log_dir=opts.log_dir,
        level=cast(Severity, opts.log_level),
    )
    report = build_window_cache(
        requests_jsonl=requests_jsonl,
        cache_dir=root,
        evidence_dir=evidence_dir,
        encoder=encoder,
        encoder_id=manifest.encoder.id,
        batch_size=batch_size,
        rows_per_shard=rows_per_shard,
        created_at_ns=created_at_ns,
        input_artifacts={
            "encoder_config.yaml": Path(opts.config),
            "model_manifest.json": model_manifest,
        },
        logger=logger,
    )
    payload = report.to_dict()
    if json_report is not None:
        _write_json_report(json_report, payload)
    build = cast(dict[str, object], payload["build"])
    typer.echo(
        "built "
        f"completed_shards={build['completed_shards']} "
        f"encoded_rows={build['encoded_rows']} "
        f"resumed_rows={build['resumed_rows']} "
        f"report={report.report_path}"
    )


def _resolve_build_config(
    *,
    config_path: Path,
    set_overrides: tuple[str, ...],
    seed: int | None,
    deterministic: bool,
    run_id: str | None,
) -> GenoLeWMConfig:
    config = load_config(config_path)
    payload = config_to_dict(config)
    for raw in set_overrides:
        _apply_set_override(payload, raw)
    if seed is not None:
        payload["seed"] = seed
    if deterministic:
        payload["deterministic"] = True
    if run_id is not None:
        payload["run_id"] = run_id
    return load_config(payload)


def _apply_set_override(payload: dict[str, Any], raw: str) -> None:
    if "=" not in raw:
        raise InputError("--set override must have the form key=value", details={"override": raw})
    key, value_text = raw.split("=", maxsplit=1)
    parts = key.split(".")
    if not parts or any(not part for part in parts):
        raise InputError("--set override key must be a non-empty dotted path", details={"key": key})
    try:
        value = yaml.safe_load(value_text)
    except yaml.YAMLError as exc:
        raise InputError(
            "--set override value is not valid YAML",
            details={"override": raw, "error": str(exc)},
        ) from exc
    target: dict[str, Any] = payload
    for part in parts[:-1]:
        child = target.get(part)
        if not isinstance(child, dict):
            raise InputError("--set override path does not resolve to a config block")
        target = child
    target[parts[-1]] = value


def _build_encoder(
    *,
    config: GenoLeWMConfig,
    manifest: Manifest,
    carbon_model_dir: Path,
    device: str,
) -> CarbonStateEncoder:
    if config.encoder.revision != manifest.encoder.revision:
        raise InputError(
            "config encoder revision does not match the model manifest",
            details={
                "config_revision": config.encoder.revision,
                "manifest_revision": manifest.encoder.revision,
            },
        )
    if not device:
        raise InputError("--device must be non-empty")
    return CarbonStateEncoder(
        str(carbon_model_dir),
        manifest.encoder.revision,
        dtype=config.encoder.dtype,
        state_layer=config.encoder.state_layer,
        pool_type=config.encoder.pool_type,
        pool_radius=config.encoder.pool_radius,
        normalize=False,
        encoder_hash=manifest.encoder.hash,
        local_files_only=True,
        trust_remote_code=config.encoder.trust_remote_code,
        device=device,
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
