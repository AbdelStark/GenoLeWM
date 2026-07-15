# SPDX-License-Identifier: Apache-2.0
"""``geno-lewm-cache-windows`` — finite build, repair, and reindex flows.

Build mode is deliberately request-artifact scoped. Corpus-percentage and
24-hour throughput acceptance remain external measurements tracked by #36.
"""

from __future__ import annotations

import json
import os
import stat
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
from geno_lewm.encoder._identity import (
    encoder_identity_hash,
    encoder_runtime_hash,
    encoder_weights_hash,
)
from geno_lewm.encoder.cache import (
    CacheReindexReport,
    CacheRepairReport,
    default_cache_dir,
    reindex_cache,
    repair_cache,
)
from geno_lewm.encoder.runtime_identity import (
    EncoderRuntimeIdentity,
    parse_encoder_runtime_identity_bytes,
)
from geno_lewm.errors import InputError
from geno_lewm.observability import Severity, get_logger
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
    encoder_runtime_identity: Annotated[
        Path | None,
        typer.Option(
            "--encoder-runtime-identity",
            help="Closed JSON identity for the exact Carbon model revision and runtime bytes.",
        ),
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
    hardware: Annotated[
        str | None,
        typer.Option(
            "--hardware",
            help="Exact accelerator/host description bound to cache-build timing evidence.",
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
        default_config_name="train",
    )
    if opts is None:
        return
    build_inputs = (
        requests_jsonl,
        evidence_dir,
        encoder_runtime_identity,
        carbon_model_dir,
        created_at_ns,
        hardware,
    )
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
            ("--encoder-runtime-identity", encoder_runtime_identity),
            ("--carbon-model-dir", carbon_model_dir),
            ("--created-at-ns", created_at_ns),
            ("--hardware", hardware),
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
    assert encoder_runtime_identity is not None
    assert carbon_model_dir is not None
    assert created_at_ns is not None
    assert hardware is not None
    assert opts.config is not None
    _reject_evidence_output_overlap(evidence_dir, json_report, option="--json-report")
    _reject_evidence_output_overlap(
        evidence_dir,
        None if opts.log_dir is None else Path(opts.log_dir),
        option="--log-dir",
    )
    request_bytes = _capture_regular_bytes(requests_jsonl, label="cache build requests")
    config_path = Path(opts.config)
    config_bytes = _capture_regular_bytes(config_path, label="encoder config")
    runtime_identity_bytes = _capture_regular_bytes(
        encoder_runtime_identity,
        label="encoder runtime identity",
    )
    resolved_config = _resolve_build_config(
        config_bytes=config_bytes,
        source_path=config_path,
        set_overrides=opts.set_overrides,
        seed=opts.seed,
        deterministic=opts.deterministic,
        run_id=opts.run_id,
    )
    runtime_contract = parse_encoder_runtime_identity_bytes(
        runtime_identity_bytes,
        source=str(encoder_runtime_identity),
    )
    runtime_identity = _capture_encoder_runtime_identity(
        config=resolved_config,
        identity=runtime_contract,
        carbon_model_dir=carbon_model_dir,
    )
    encoder = _build_encoder(
        config=resolved_config,
        identity=runtime_contract,
        carbon_model_dir=carbon_model_dir,
        device=device,
        observed_identity=runtime_contract.cache_identity_hash,
    )
    logger = get_logger(
        "cache-build",
        run_id=opts.run_id,
        log_dir=opts.log_dir,
        level=cast(Severity, opts.log_level),
    )
    build_kwargs: dict[str, Any] = {
        "requests_jsonl": request_bytes,
        "cache_dir": root,
        "evidence_dir": evidence_dir,
        "encoder": encoder,
        "encoder_id": runtime_contract.model_id,
        "batch_size": batch_size,
        "rows_per_shard": rows_per_shard,
        "created_at_ns": created_at_ns,
        "hardware": hardware,
        "resolved_config": cast(dict[str, object], config_to_dict(resolved_config)),
        "encoder_runtime_identity": runtime_identity,
        "input_artifacts": {
            "encoder_config.yaml": config_bytes,
            "encoder_runtime_identity_source.json": runtime_identity_bytes,
        },
    }
    report = build_window_cache(
        **build_kwargs,
        logger=logger,
    )
    payload = report.to_dict()
    if json_report is not None:
        _write_json_report(json_report, payload)
        # The output path was checked before the build, but immediately
        # re-verify the fixed evidence closure after the only post-checksum
        # write so alias swaps cannot return success with an unclosed bundle.
        report = build_window_cache(**build_kwargs, logger=None)
    build = cast(dict[str, object], payload["build"])
    typer.echo(
        "built "
        f"completed_shards={build['completed_shards']} "
        f"encoded_rows={build['encoded_rows']} "
        f"resumed_rows={build['resumed_rows']} "
        f"reused_rows={build['reused_rows']} "
        f"report={report.report_path}"
    )


def _resolve_build_config(
    *,
    config_bytes: bytes,
    source_path: Path,
    set_overrides: tuple[str, ...],
    seed: int | None,
    deterministic: bool,
    run_id: str | None,
) -> GenoLeWMConfig:
    try:
        text = config_bytes.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise InputError(
            "config file must be UTF-8 YAML",
            details={"path": str(source_path)},
        ) from exc
    try:
        raw_payload = yaml.safe_load(text) or {}
    except yaml.YAMLError as exc:
        raise InputError(
            "config file is not valid YAML",
            details={"path": str(source_path), "error": str(exc)},
        ) from exc
    config = load_config(raw_payload)
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
    identity: EncoderRuntimeIdentity,
    carbon_model_dir: Path,
    device: str,
    observed_identity: str | None = None,
) -> CarbonStateEncoder:
    if (
        config.encoder.model_id != identity.model_id
        or config.encoder.revision != identity.revision
        or config.encoder.state_contract_version != identity.state_contract_version
    ):
        raise InputError(
            "config encoder identity does not match the encoder runtime identity",
            details={
                "config_model_id": config.encoder.model_id,
                "identity_model_id": identity.model_id,
                "config_revision": config.encoder.revision,
                "identity_revision": identity.revision,
                "config_state_contract_version": config.encoder.state_contract_version,
                "identity_state_contract_version": identity.state_contract_version,
            },
        )
    if not device:
        raise InputError("--device must be non-empty")
    if observed_identity is None:
        observed_identity = encoder_identity_hash(
            carbon_model_dir,
            state_contract_version=config.encoder.state_contract_version,
        )
    if observed_identity != identity.cache_identity_hash:
        raise InputError(
            "local Carbon runtime identity does not match the committed identity",
            details={
                "state_contract_version": config.encoder.state_contract_version,
                "expected": identity.cache_identity_hash,
                "observed": observed_identity,
            },
            remediation="mount the exact corrected Carbon runtime committed by the manifest",
        )
    return CarbonStateEncoder(
        str(carbon_model_dir),
        identity.revision,
        dtype=config.encoder.dtype,
        state_layer=config.encoder.state_layer,
        pool_type=config.encoder.pool_type,
        pool_radius=config.encoder.pool_radius,
        normalize=False,
        encoder_hash=identity.cache_identity_hash,
        local_files_only=True,
        trust_remote_code=config.encoder.trust_remote_code,
        device=device,
    )


def _capture_encoder_runtime_identity(
    *,
    config: GenoLeWMConfig,
    identity: EncoderRuntimeIdentity,
    carbon_model_dir: Path,
) -> dict[str, object]:
    if (
        config.encoder.model_id != identity.model_id
        or config.encoder.revision != identity.revision
        or config.encoder.state_contract_version != identity.state_contract_version
    ):
        raise InputError(
            "config encoder identity does not match the encoder runtime identity",
            details={
                "config_model_id": config.encoder.model_id,
                "identity_model_id": identity.model_id,
                "config_revision": config.encoder.revision,
                "identity_revision": identity.revision,
            },
        )
    observed_runtime = encoder_runtime_hash(carbon_model_dir)
    if observed_runtime != identity.runtime_hash:
        raise InputError(
            "local Carbon runtime hash does not match the committed identity",
            details={"expected": identity.runtime_hash, "observed": observed_runtime},
            remediation="mount the exact corrected Carbon runtime committed by the identity",
        )
    if identity.weights_hash is not None:
        observed_weights = encoder_weights_hash(carbon_model_dir)
        if observed_weights != identity.weights_hash:
            raise InputError(
                "local Carbon weights hash does not match the committed identity",
                details={"expected": identity.weights_hash, "observed": observed_weights},
            )
    return identity.to_dict()


def _reject_evidence_output_overlap(
    evidence_dir: Path,
    output: Path | None,
    *,
    option: str,
) -> None:
    if output is None:
        return
    try:
        evidence = evidence_dir.resolve(strict=False)
        candidate = output.resolve(strict=False)
    except (OSError, RuntimeError) as exc:
        raise InputError(
            f"{option} path aliases could not be resolved safely",
            details={"evidence_dir": str(evidence_dir), "output": str(output)},
        ) from exc
    evidence_parts = tuple(part.casefold() for part in evidence.parts)
    candidate_parts = tuple(part.casefold() for part in candidate.parts)
    if candidate_parts[: len(evidence_parts)] != evidence_parts:
        return
    raise InputError(
        f"{option} must be outside --evidence-dir",
        details={"evidence_dir": str(evidence), "output": str(candidate)},
    )


def _capture_regular_bytes(path: Path, *, label: str) -> bytes:
    """Read one stable regular-file snapshot without following the final name."""
    try:
        mode = path.lstat().st_mode
    except OSError as exc:
        raise InputError(f"{label} could not be inspected", details={"path": str(path)}) from exc
    if stat.S_ISLNK(mode) or not stat.S_ISREG(mode):
        raise InputError(f"{label} must be a regular non-symlink file")
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError as exc:
        raise InputError(
            f"{label} could not be opened safely", details={"path": str(path)}
        ) from exc
    try:
        before = os.fstat(descriptor)
        with os.fdopen(os.dup(descriptor), "rb") as handle:
            body = handle.read()
        after = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    identity_before = (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns)
    identity_after = (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns)
    if identity_before != identity_after or len(body) != before.st_size:
        raise InputError(f"{label} changed while it was being captured")
    return body


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
