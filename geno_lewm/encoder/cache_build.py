# SPDX-License-Identifier: Apache-2.0
"""Finite, resumable construction of raw pooled window-cache shards.

The build input is an immutable JSONL request artifact. Planning resolves the
exact :class:`~geno_lewm.encoder.cache.WindowCacheKey` (including
``center_token``), deduplicates only identical keys, and commits deterministic
shard assignments before the first model forward pass. A durable state file
binds every completed shard to its bytes so a resumed invocation verifies all
prior work before calling ``encoder.encode_batch`` again.

This module deliberately proves only completion of the supplied finite request
artifact. It does not claim a percentage of Carbon's pretraining corpus or the
RFC-0006 24-hour hardware target.
"""

from __future__ import annotations

import json
import math
import os
import stat
import tempfile
import time
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import cast

from geno_lewm.encoder.cache import (
    CACHE_SCHEMA_VERSION,
    INDEX_DB_NAME,
    CacheProvenance,
    CacheShardInspection,
    WindowCacheKey,
    WindowCacheRecord,
    inspect_cache_shard,
    resolve_cache_provenances,
    shard_path_for,
    write_shard,
)
from geno_lewm.encoder.pooling import POOL_GLOBAL_MEAN
from geno_lewm.encoder.windowing import canonicalize_dna, window_sha256
from geno_lewm.errors import CacheCorruptError, InputError, RuntimeSetupError
from geno_lewm.observability import GenoLeWMLogger
from geno_lewm.provenance import canonical_json_sha256, sha256_bytes, sha256_file

__all__ = [
    "CACHE_BUILD_REPORT_NAME",
    "CACHE_BUILD_SCHEMA_VERSION",
    "CacheBuildReport",
    "build_window_cache",
]


CACHE_BUILD_SCHEMA_VERSION = "1.1.0"
CACHE_BUILD_REPORT_NAME = "cache_build_report.json"
_GENERATED_BY = "geno_lewm.encoder.cache_build"
_REQUEST_COPY_NAME = "cache_build_requests.jsonl"
_PLAN_NAME = "cache_build_plan.json"
_STATE_NAME = "cache_build_state.json"
_RESOLVED_CONFIG_NAME = "resolved_config.json"
_CHECKSUMS_NAME = "SHA256SUMS"
_REQUEST_KEYS = frozenset({"request_id", "chrom", "start_bp", "end_bp", "window", "edit_locus"})
_HASH_PREFIX = "sha256:"


@dataclass(frozen=True, slots=True)
class CacheBuildReport:
    """Completed evidence bundle and its JSON-native report payload."""

    report_path: Path
    checksums_path: Path
    payload: Mapping[str, object]

    def to_dict(self) -> dict[str, object]:
        """Return a detached JSON-native copy of the report."""
        return cast(dict[str, object], json.loads(json.dumps(self.payload)))


@dataclass(frozen=True, slots=True)
class _Request:
    request_id: str
    chrom: str
    start_bp: int
    end_bp: int
    window: str
    edit_locus: int | None


@dataclass(frozen=True, slots=True)
class _PlannedRow:
    representative: _Request
    request_ids: tuple[str, ...]
    key: WindowCacheKey


@dataclass(frozen=True, slots=True)
class _PlannedShard:
    shard_id: str
    relative_path: str
    contig: str
    stride_block: int
    rows: tuple[_PlannedRow, ...]


@dataclass(frozen=True, slots=True)
class _BuildPlan:
    payload: Mapping[str, object]
    shards: tuple[_PlannedShard, ...]
    namespace: str


@dataclass(frozen=True, slots=True)
class _EncoderContract:
    encoder_hash: bytes
    state_layer: int
    pool_type: str
    pool_radius: int
    dtype: str


@dataclass(frozen=True, slots=True)
class _InputArtifact:
    name: str
    body: bytes
    identity: Mapping[str, object]


@dataclass(frozen=True, slots=True)
class _ShardSummary:
    path: str
    sha256: str
    size_bytes: int
    row_count: int


@dataclass(frozen=True, slots=True)
class _ResolvedRow:
    provenance: CacheProvenance
    created_at_ns: int


@dataclass(frozen=True, slots=True)
class _ResolvedCache:
    rows: Mapping[WindowCacheKey, _ResolvedRow]
    shards: Mapping[str, _ShardSummary]


@dataclass(frozen=True, slots=True)
class _EncodeResult:
    records: tuple[WindowCacheRecord, ...]
    encode_batch_calls: int
    encode_batch_seconds: float


def build_window_cache(
    *,
    requests_jsonl: Path | str,
    cache_dir: Path | str,
    evidence_dir: Path | str,
    encoder: object,
    encoder_id: str,
    batch_size: int,
    rows_per_shard: int,
    created_at_ns: int,
    hardware: str,
    resolved_config: Mapping[str, object],
    input_artifacts: Mapping[str, Path | str] | None = None,
    logger: GenoLeWMLogger | None = None,
) -> CacheBuildReport:
    """Build or resume the cache for one exact request JSONL artifact.

    ``encoder`` must expose the raw-state ``CarbonStateEncoder`` contract:
    ``pooling_identity``, ``encode_batch``, an exact 32-byte ``encoder_hash``,
    and ``normalize is False``. Existing planned shards are decoded, hashed,
    compared row-for-row with the plan, and re-indexed before any missing shard
    is encoded.
    """
    started = time.perf_counter()
    batch_size = _positive_int("batch_size", batch_size)
    rows_per_shard = _positive_int("rows_per_shard", rows_per_shard)
    created_at_ns = _non_negative_int("created_at_ns", created_at_ns)
    if created_at_ns == 0:
        raise InputError("created_at_ns must be fixed to a positive UTC nanosecond value")
    if type(encoder_id) is not str or not encoder_id:
        raise InputError("encoder_id must be non-empty text")
    hardware = _text(hardware, field="hardware")
    resolved_config_payload = _json_object_copy(resolved_config, field="resolved_config")
    resolved_config_bytes = _pretty_json_bytes(resolved_config_payload)

    contract = _encoder_contract(encoder)
    encoder_device = _text(getattr(encoder, "device", None), field="encoder.device")
    request_path = Path(requests_jsonl)
    request_bytes = _read_regular_bytes(request_path, label="cache build requests")
    requests = _parse_requests(request_bytes)
    request_identity = {
        "sha256": sha256_bytes(request_bytes),
        "size_bytes": len(request_bytes),
    }

    cache_root = Path(cache_dir).absolute()
    evidence_root = Path(evidence_dir).absolute()
    _ensure_real_directory(evidence_root)
    staged_inputs = _read_input_artifacts(input_artifacts or {})
    input_identities = tuple(item.identity for item in staged_inputs)
    request_copy_path = evidence_root / _REQUEST_COPY_NAME
    plan_path = evidence_root / _PLAN_NAME
    state_path = evidence_root / _STATE_NAME
    report_path = evidence_root / CACHE_BUILD_REPORT_NAME
    checksums_path = evidence_root / _CHECKSUMS_NAME
    resolved_config_path = evidence_root / _RESOLVED_CONFIG_NAME
    resolved_config_identity = {
        "path": _RESOLVED_CONFIG_NAME,
        "sha256": sha256_bytes(resolved_config_bytes),
        "size_bytes": len(resolved_config_bytes),
    }
    expected_evidence_names = _expected_evidence_names(input_identities)
    _assert_evidence_inventory(
        evidence_root,
        expected_names=expected_evidence_names,
        require_complete=False,
    )

    namespace = f"{encoder_id}::requests::{request_identity['sha256']}"
    expected_plan = _create_plan(
        requests=requests,
        request_identity=request_identity,
        cache_root=cache_root,
        encoder=encoder,
        encoder_id=encoder_id,
        namespace=namespace,
        contract=contract,
        batch_size=batch_size,
        rows_per_shard=rows_per_shard,
        created_at_ns=created_at_ns,
        hardware=hardware,
        encoder_device=encoder_device,
        resolved_config=resolved_config_identity,
        input_artifacts=input_identities,
    )
    if os.path.lexists(plan_path):
        plan_payload = _read_json_object(plan_path, label="cache build plan")
        plan = _load_plan(
            plan_payload,
            expected=expected_plan,
        )
    else:
        plan = expected_plan
        _write_once(plan_path, _pretty_json_bytes(plan.payload), label="cache build plan")
    # The immutable plan is validated or installed before any caller-provided
    # artifact is staged. A failed invocation therefore cannot seed files that
    # a later, differently configured invocation would accidentally close.
    _write_once(request_copy_path, request_bytes, label="cache build request copy")
    _write_once(
        resolved_config_path,
        resolved_config_bytes,
        label="cache build resolved config",
    )
    _stage_input_artifacts(evidence_root, staged_inputs)
    plan_sha256 = sha256_file(plan_path)

    state = _load_or_initialize_state(state_path, plan_sha256=plan_sha256)
    completed = _completed_by_id(state, plan=plan)
    adopted_or_changed = False

    # Resume preflight: verify every evidence-owned shard and repair its index
    # rows before resolving shared-cache hits or calling encode_batch.
    for shard in plan.shards:
        absolute_path = cache_root / shard.relative_path
        prior = completed.get(shard.shard_id)
        if prior is None and not os.path.lexists(absolute_path):
            continue
        if prior is not None and not os.path.lexists(absolute_path):
            raise CacheCorruptError(
                "cache build state names a completed shard that is missing",
                details={"shard_id": shard.shard_id, "path": shard.relative_path},
            )
        inspection = inspect_cache_shard(cache_root, shard.relative_path)
        execution_shard = _execution_shard_from_records(shard, inspection.records)
        _assert_inspection_matches_plan(
            inspection,
            shard=execution_shard,
            created_at_ns=created_at_ns,
        )
        if prior is not None:
            _assert_state_identity(
                prior,
                inspection,
                shard=execution_shard,
                plan_shard=shard,
            )
        else:
            completed[shard.shard_id] = _completed_entry(
                execution_shard,
                inspection,
                plan_shard_id=shard.shard_id,
                origin="adopted_after_interruption",
                encoded_rows=0,
                encode_batch_calls=0,
                encode_batch_seconds=None,
            )
            adopted_or_changed = True
        # Re-publishing observed rows is a no-op on the immutable shard and
        # makes a missing/stale SQLite index converge without model work.
        write_shard(
            cache_root,
            encoder_id=plan.namespace,
            contig=shard.contig,
            stride_block=shard.stride_block,
            records=inspection.records,
        )

    if adopted_or_changed or not os.path.lexists(state_path):
        state = _state_payload(plan_sha256=plan_sha256, completed=completed)
        _atomic_write(state_path, _pretty_json_bytes(state))

    resolved_before = _resolve_plan_cache(cache_root, plan, require_all=False)
    resumed_rows = len(resolved_before.rows)
    if os.path.lexists(checksums_path):
        return _verify_completed_bundle(
            evidence_root=evidence_root,
            cache_root=cache_root,
            plan=plan,
            plan_sha256=plan_sha256,
            expected_evidence_names=expected_evidence_names,
        )

    if logger is not None:
        logger.info(
            "data.cache.build.start",
            request_count=len(requests),
            unique_rows=sum(len(shard.rows) for shard in plan.shards),
            planned_shards=len(plan.shards),
            requests_sha256=request_identity["sha256"],
        )

    encoded_rows = 0
    encoded_shards = 0
    for shard in plan.shards:
        missing_rows = tuple(row for row in shard.rows if row.key not in resolved_before.rows)
        if not missing_rows:
            _emit_progress(
                logger,
                shard=shard,
                completed_shards=_resolved_plan_shard_count(plan, resolved_before.rows),
                total_shards=len(plan.shards),
                encoded_rows=encoded_rows,
                resumed_rows=resumed_rows,
                throughput_per_s=None,
                status="resumed",
            )
            continue
        execution_shard = _execution_shard(shard, rows=missing_rows)
        encoded = _encode_shard(
            execution_shard,
            encoder=encoder,
            contract=contract,
            batch_size=batch_size,
            created_at_ns=created_at_ns,
        )
        try:
            path = write_shard(
                cache_root,
                encoder_id=plan.namespace,
                contig=execution_shard.contig,
                stride_block=execution_shard.stride_block,
                records=encoded.records,
            )
        except CacheCorruptError:
            # A concurrent builder may have published the same logical misses
            # after our preflight but before the serialized write reservation.
            # Accept only byte-equivalent logical winners; write_shard checks
            # key availability before publishing, so no orphan is created.
            _assert_encoded_rows_match_cache(cache_root, encoded.records)
            continue
        inspection = inspect_cache_shard(cache_root, shard.relative_path)
        _assert_inspection_matches_plan(
            inspection,
            shard=execution_shard,
            created_at_ns=created_at_ns,
        )
        if tuple(record.embedding for record in inspection.records) != tuple(
            _fp32_vector(record.embedding) for record in encoded.records
        ):
            raise CacheCorruptError(
                "published cache shard embeddings do not match encoded rows",
                details={"shard_id": shard.shard_id, "path": shard.relative_path},
            )
        completed[shard.shard_id] = _completed_entry(
            execution_shard,
            inspection,
            plan_shard_id=shard.shard_id,
            origin="encoded",
            encoded_rows=len(execution_shard.rows),
            encode_batch_calls=encoded.encode_batch_calls,
            encode_batch_seconds=encoded.encode_batch_seconds,
        )
        encoded_rows += len(execution_shard.rows)
        encoded_shards += 1
        state = _state_payload(plan_sha256=plan_sha256, completed=completed)
        _atomic_write(state_path, _pretty_json_bytes(state))
        if logger is not None:
            logger.info(
                "data.shard.write",
                shard_id=shard.shard_id,
                path=path.relative_to(cache_root).as_posix(),
                n_rows=len(execution_shard.rows),
                size_bytes=inspection.size_bytes,
            )
        rate = (
            None
            if encoded.encode_batch_seconds <= 0
            else len(execution_shard.rows) / encoded.encode_batch_seconds
        )
        _emit_progress(
            logger,
            shard=shard,
            completed_shards=_resolved_plan_shard_count(plan, resolved_before.rows)
            + encoded_shards,
            total_shards=len(plan.shards),
            encoded_rows=encoded_rows,
            resumed_rows=resumed_rows,
            throughput_per_s=rate,
            status="encoded",
        )

    resolved = _resolve_plan_cache(cache_root, plan, require_all=True)
    state = _state_payload(plan_sha256=plan_sha256, completed=completed)
    _atomic_write(state_path, _pretty_json_bytes(state))
    report_payload = _report_payload(
        plan=plan,
        plan_path=plan_path,
        state_path=state_path,
        request_copy_path=request_copy_path,
        request_identity=request_identity,
        request_count=len(requests),
        cache_root=cache_root,
        resolved=resolved,
        completed=completed,
        encoded_rows=encoded_rows,
        encoded_shards=encoded_shards,
        resumed_rows=resumed_rows,
        elapsed_seconds=time.perf_counter() - started,
        batch_size=batch_size,
        rows_per_shard=rows_per_shard,
        created_at_ns=created_at_ns,
        hardware=hardware,
        encoder_device=encoder_device,
        resolved_config=resolved_config_identity,
        logger=logger,
        input_artifacts=input_identities,
    )
    _atomic_write(report_path, _pretty_json_bytes(report_payload))
    _write_checksums(evidence_root, expected_names=expected_evidence_names)

    if logger is not None:
        throughput = cast(Mapping[str, object], report_payload["throughput"])
        logger.info(
            "data.cache.build.end",
            completed_shards=len(plan.shards),
            encoded_rows=encoded_rows,
            resumed_rows=resumed_rows,
            elapsed_s=cast(float, throughput["invocation_elapsed_seconds"]),
            throughput_per_s=throughput["measured_encoded_rows_per_second"],
            evidence_report=report_path.name,
        )
    return CacheBuildReport(
        report_path=report_path,
        checksums_path=checksums_path,
        payload=report_payload,
    )


def _create_plan(
    *,
    requests: tuple[_Request, ...],
    request_identity: Mapping[str, object],
    cache_root: Path,
    encoder: object,
    encoder_id: str,
    namespace: str,
    contract: _EncoderContract,
    batch_size: int,
    rows_per_shard: int,
    created_at_ns: int,
    hardware: str,
    encoder_device: str,
    resolved_config: Mapping[str, object],
    input_artifacts: tuple[Mapping[str, object], ...],
) -> _BuildPlan:
    resolver = getattr(encoder, "pooling_identity", None)
    if not callable(resolver):
        raise RuntimeSetupError(
            "cache build encoder must expose pooling_identity(window, edit_locus)",
            remediation="use CarbonStateEncoder so cache center_token values come from token IDs",
        )
    rows_by_key: dict[WindowCacheKey, list[_Request]] = defaultdict(list)
    for request in requests:
        pool_type, pool_radius, center_token = _resolve_pooling_identity(
            resolver(request.window, request.edit_locus)
        )
        key = WindowCacheKey(
            window_hash=window_sha256(request.window),
            encoder_hash=contract.encoder_hash,
            state_layer=contract.state_layer,
            pool_type=pool_type,
            pool_radius=pool_radius,
            center_token=center_token,
            dtype=contract.dtype,
        )
        rows_by_key[key].append(request)

    unique_rows: list[_PlannedRow] = []
    for key, aliases in rows_by_key.items():
        ordered = sorted(aliases, key=_request_sort_key)
        unique_rows.append(
            _PlannedRow(
                representative=ordered[0],
                request_ids=tuple(sorted(request.request_id for request in aliases)),
                key=key,
            )
        )
    unique_rows.sort(key=_planned_row_sort_key)
    shards = _assign_shards(
        unique_rows,
        cache_root=cache_root,
        namespace=namespace,
        rows_per_shard=rows_per_shard,
    )
    payload = _plan_payload(
        shards=shards,
        request_identity=request_identity,
        request_count=len(requests),
        encoder_id=encoder_id,
        namespace=namespace,
        contract=contract,
        batch_size=batch_size,
        rows_per_shard=rows_per_shard,
        created_at_ns=created_at_ns,
        hardware=hardware,
        encoder_device=encoder_device,
        resolved_config=resolved_config,
        input_artifacts=input_artifacts,
    )
    return _BuildPlan(payload=payload, shards=shards, namespace=namespace)


def _assign_shards(
    rows: Sequence[_PlannedRow],
    *,
    cache_root: Path,
    namespace: str,
    rows_per_shard: int,
) -> tuple[_PlannedShard, ...]:
    groups: dict[tuple[str, str, int], list[_PlannedRow]] = defaultdict(list)
    for row in rows:
        groups[(row.representative.chrom, row.key.pool_type, row.key.pool_radius)].append(row)
    shards: list[_PlannedShard] = []
    for (contig, pool_type, pool_radius), group_rows in sorted(groups.items()):
        ordered = sorted(group_rows, key=_planned_row_sort_key)
        for stride_block, offset in enumerate(range(0, len(ordered), rows_per_shard)):
            chunk = tuple(ordered[offset : offset + rows_per_shard])
            first = chunk[0]
            path = shard_path_for(
                cache_root,
                encoder_id=namespace,
                state_layer=first.key.state_layer,
                pool_type=pool_type,
                pool_radius=pool_radius,
                contig=contig,
                stride_block=stride_block,
                encoder_hash=first.key.encoder_hash,
                dtype=first.key.dtype,
            )
            relative_path = path.relative_to(cache_root).as_posix()
            shard_id = canonical_json_sha256(
                {
                    "path": relative_path,
                    "keys": [_key_payload(row.key) for row in chunk],
                }
            )
            shards.append(
                _PlannedShard(
                    shard_id=shard_id,
                    relative_path=relative_path,
                    contig=contig,
                    stride_block=stride_block,
                    rows=chunk,
                )
            )
    return tuple(sorted(shards, key=lambda shard: shard.relative_path))


def _plan_payload(
    *,
    shards: tuple[_PlannedShard, ...],
    request_identity: Mapping[str, object],
    request_count: int,
    encoder_id: str,
    namespace: str,
    contract: _EncoderContract,
    batch_size: int,
    rows_per_shard: int,
    created_at_ns: int,
    hardware: str,
    encoder_device: str,
    resolved_config: Mapping[str, object],
    input_artifacts: tuple[Mapping[str, object], ...],
) -> dict[str, object]:
    unique_rows = sum(len(shard.rows) for shard in shards)
    return {
        "schema_version": CACHE_BUILD_SCHEMA_VERSION,
        "generated_by": _GENERATED_BY,
        "requests": {
            **request_identity,
            "input_rows": request_count,
            "unique_cache_keys": unique_rows,
            "duplicate_rows": request_count - unique_rows,
        },
        "encoder": {
            "id": encoder_id,
            "cache_namespace": namespace,
            "hash": _hash_text(contract.encoder_hash),
            "state_layer": contract.state_layer,
            "pool_type": contract.pool_type,
            "pool_radius": contract.pool_radius,
            "dtype": contract.dtype,
            "normalize": False,
        },
        "created_at_ns": created_at_ns,
        "execution": {
            "batch_size": batch_size,
            "hardware": {
                "description": hardware,
                "encoder_device": encoder_device,
            },
            "resolved_config": dict(resolved_config),
            "timing_scope": (
                "wall time inside encoder.encode_batch calls only; excludes planning, "
                "record materialization, Parquet publication, indexing, and verification"
            ),
        },
        "input_artifacts": [dict(identity) for identity in input_artifacts],
        "sharding": {
            "rows_per_shard": rows_per_shard,
            "planned_shards": len(shards),
            "ordering": "chrom,pool_type,pool_radius,cache_key",
        },
        "shards": [
            {
                "shard_id": shard.shard_id,
                "path": shard.relative_path,
                "contig": shard.contig,
                "stride_block": shard.stride_block,
                "rows": [
                    {
                        "representative_request_id": row.representative.request_id,
                        "request_ids": list(row.request_ids),
                        "key": _key_payload(row.key),
                        "record": {
                            "chrom": row.representative.chrom,
                            "start_bp": row.representative.start_bp,
                            "end_bp": row.representative.end_bp,
                            "untargeted": row.representative.edit_locus is None,
                        },
                    }
                    for row in shard.rows
                ],
            }
            for shard in shards
        ],
        "claim_boundary": {
            "scope": "the exact finite cache_build_requests.jsonl artifact only",
            "ten_percent_corpus_completed": False,
            "twenty_four_hour_target_evaluated": False,
        },
    }


def _load_plan(
    payload: Mapping[str, object],
    *,
    expected: _BuildPlan,
) -> _BuildPlan:
    """Accept only the exact plan rederived from immutable build inputs.

    Parsing a persisted plan and validating identities derived from that same
    plan is circular: an altered partition can remain self-consistent. The
    caller therefore recomputes the complete canonical payload from requests,
    encoder pooling identities, runtime/config identity, and sharding inputs.
    """
    if dict(payload) != dict(expected.payload):
        raise InputError(
            "cache build plan does not match the exact plan rederived from immutable inputs",
            details={
                "expected_sha256": canonical_json_sha256(expected.payload),
                "observed_sha256": canonical_json_sha256(payload),
            },
        )
    return expected


def _encode_shard(
    shard: _PlannedShard,
    *,
    encoder: object,
    contract: _EncoderContract,
    batch_size: int,
    created_at_ns: int,
) -> _EncodeResult:
    encode_batch = getattr(encoder, "encode_batch", None)
    if not callable(encode_batch):
        raise RuntimeSetupError("cache build encoder must expose encode_batch(windows, edit_loci)")
    records: list[WindowCacheRecord] = []
    encode_batch_calls = 0
    encode_batch_seconds = 0.0
    for offset in range(0, len(shard.rows), batch_size):
        rows = shard.rows[offset : offset + batch_size]
        windows = [row.representative.window for row in rows]
        edit_loci = [row.representative.edit_locus for row in rows]
        batch_started = time.perf_counter()
        encoded = encode_batch(windows, edit_loci)
        encode_batch_seconds += time.perf_counter() - batch_started
        encode_batch_calls += 1
        if not isinstance(encoded, Sequence) or isinstance(encoded, str | bytes):
            raise InputError("encoder.encode_batch must return a sequence of state vectors")
        if len(encoded) != len(rows):
            raise InputError(
                "encoder returned a batch with the wrong length",
                details={"expected": len(rows), "observed": len(encoded)},
            )
        for row, vector in zip(rows, encoded, strict=True):
            embedding = _state_vector(vector)
            records.append(
                WindowCacheRecord(
                    chrom=row.representative.chrom,
                    start_bp=row.representative.start_bp,
                    end_bp=row.representative.end_bp,
                    window_hash=row.key.window_hash,
                    encoder_hash=contract.encoder_hash,
                    state_layer=contract.state_layer,
                    pool_type=row.key.pool_type,
                    pool_radius=row.key.pool_radius,
                    center_token=row.key.center_token,
                    dtype=contract.dtype,
                    embedding=embedding,
                    untargeted=row.representative.edit_locus is None,
                    created_at=created_at_ns,
                )
            )
    widths = {len(record.embedding) for record in records}
    if len(widths) != 1:
        raise InputError(
            "encoder returned inconsistent state widths within a shard",
            details={"widths": sorted(widths)},
        )
    return _EncodeResult(
        records=tuple(records),
        encode_batch_calls=encode_batch_calls,
        encode_batch_seconds=encode_batch_seconds,
    )


def _assert_inspection_matches_plan(
    inspection: CacheShardInspection,
    *,
    shard: _PlannedShard,
    created_at_ns: int,
) -> None:
    if len(inspection.records) != len(shard.rows):
        raise CacheCorruptError(
            "cache shard row count does not match the build plan",
            details={
                "shard_id": shard.shard_id,
                "expected": len(shard.rows),
                "observed": len(inspection.records),
            },
        )
    widths = {len(record.embedding) for record in inspection.records}
    if len(widths) != 1:
        raise CacheCorruptError(
            "cache shard contains inconsistent embedding widths",
            details={"shard_id": shard.shard_id, "widths": sorted(widths)},
        )
    for index, (record, row) in enumerate(zip(inspection.records, shard.rows, strict=True)):
        expected = row.representative
        if (
            record.key != row.key
            or record.chrom != expected.chrom
            or record.start_bp != expected.start_bp
            or record.end_bp != expected.end_bp
            or record.untargeted != (expected.edit_locus is None)
            or record.created_at != created_at_ns
        ):
            raise CacheCorruptError(
                "cache shard row does not match the deterministic build plan",
                details={"shard_id": shard.shard_id, "row_offset": index},
            )


def _execution_shard(
    plan_shard: _PlannedShard,
    *,
    rows: tuple[_PlannedRow, ...],
) -> _PlannedShard:
    if not rows:
        raise CacheCorruptError("cache build execution shard must contain at least one row")
    planned_keys = {row.key for row in plan_shard.rows}
    if any(row.key not in planned_keys for row in rows) or len({row.key for row in rows}) != len(
        rows
    ):
        raise CacheCorruptError("cache build execution shard is not a unique plan subset")
    canonical = tuple(row for row in plan_shard.rows if row.key in {item.key for item in rows})
    if rows != canonical:
        raise CacheCorruptError("cache build execution shard rows are not in plan order")
    return _PlannedShard(
        shard_id=canonical_json_sha256(
            {
                "plan_shard_id": plan_shard.shard_id,
                "path": plan_shard.relative_path,
                "keys": [_key_payload(row.key) for row in rows],
            }
        ),
        relative_path=plan_shard.relative_path,
        contig=plan_shard.contig,
        stride_block=plan_shard.stride_block,
        rows=rows,
    )


def _execution_shard_from_records(
    plan_shard: _PlannedShard,
    records: Sequence[WindowCacheRecord],
) -> _PlannedShard:
    by_key = {row.key: row for row in plan_shard.rows}
    try:
        rows = tuple(by_key[record.key] for record in records)
    except KeyError as exc:
        raise CacheCorruptError(
            "evidence-owned cache shard contains a key outside its immutable plan",
            details={"plan_shard_id": plan_shard.shard_id},
        ) from exc
    if len({record.key for record in records}) != len(records):
        raise CacheCorruptError("evidence-owned cache shard contains a duplicate key")
    return _execution_shard(plan_shard, rows=rows)


def _resolve_plan_cache(
    cache_root: Path,
    plan: _BuildPlan,
    *,
    require_all: bool,
) -> _ResolvedCache:
    planned_rows = tuple(row for shard in plan.shards for row in shard.rows)
    provenances = resolve_cache_provenances(
        cache_root,
        tuple(row.key for row in planned_rows),
        policy="require_v3",
    )
    by_path: dict[str, list[tuple[_PlannedRow, CacheProvenance]]] = defaultdict(list)
    missing: list[str] = []
    for row, provenance in zip(planned_rows, provenances, strict=True):
        if provenance is None:
            missing.extend(row.request_ids)
            continue
        if provenance.cache_schema_version != CACHE_SCHEMA_VERSION:
            raise CacheCorruptError(
                "finite cache build resolved a non-schema-3 cache row",
                details={"request_ids": list(row.request_ids)},
            )
        try:
            relative = provenance.shard_path.absolute().relative_to(cache_root).as_posix()
        except ValueError as exc:
            raise CacheCorruptError("cache index resolved a shard outside the cache root") from exc
        by_path[relative].append((row, provenance))
    if require_all and missing:
        raise CacheCorruptError(
            "cache build ended without resolving every planned logical key",
            details={"missing_request_ids": sorted(missing)},
        )

    resolved_rows: dict[WindowCacheKey, _ResolvedRow] = {}
    summaries: dict[str, _ShardSummary] = {}
    for relative in sorted(by_path):
        inspection = inspect_cache_shard(cache_root, relative)
        summaries[relative] = _summary(inspection, cache_root=cache_root)
        seen_offsets: set[int] = set()
        for row, provenance in by_path[relative]:
            offset = provenance.row_offset
            if offset in seen_offsets or offset < 0 or offset >= len(inspection.records):
                raise CacheCorruptError(
                    "cache index resolved an invalid or duplicate row offset",
                    details={"path": relative, "row_offset": offset},
                )
            seen_offsets.add(offset)
            record = inspection.records[offset]
            if record.key != row.key or record.schema_version != CACHE_SCHEMA_VERSION:
                raise CacheCorruptError(
                    "cache index logical key does not match the referenced schema-3 row",
                    details={"path": relative, "row_offset": offset},
                )
            resolved_rows[row.key] = _ResolvedRow(
                provenance=provenance,
                created_at_ns=record.created_at,
            )
        # Drop the fully decoded inspection before opening the next shard. The
        # retained evidence is O(number of keys + shards), not O(rows*d_state).
        del inspection
    return _ResolvedCache(rows=resolved_rows, shards=summaries)


def _summary(inspection: CacheShardInspection, *, cache_root: Path) -> _ShardSummary:
    return _ShardSummary(
        path=inspection.path.absolute().relative_to(cache_root).as_posix(),
        sha256=inspection.sha256,
        size_bytes=inspection.size_bytes,
        row_count=len(inspection.records),
    )


def _resolved_plan_shard_count(
    plan: _BuildPlan,
    resolved: Mapping[WindowCacheKey, _ResolvedRow],
) -> int:
    return sum(all(row.key in resolved for row in shard.rows) for shard in plan.shards)


def _assert_encoded_rows_match_cache(
    cache_root: Path,
    encoded: Sequence[WindowCacheRecord],
) -> None:
    provenances = resolve_cache_provenances(
        cache_root,
        tuple(record.key for record in encoded),
        policy="require_v3",
    )
    if any(provenance is None for provenance in provenances):
        raise CacheCorruptError(
            "cache shard publication failed without a complete logical-key winner"
        )
    by_path: dict[str, list[tuple[WindowCacheRecord, CacheProvenance]]] = defaultdict(list)
    for record, raw_provenance in zip(encoded, provenances, strict=True):
        assert raw_provenance is not None
        relative = raw_provenance.shard_path.absolute().relative_to(cache_root).as_posix()
        by_path[relative].append((record, raw_provenance))
    for relative, rows in by_path.items():
        inspection = inspect_cache_shard(cache_root, relative)
        for expected, provenance in rows:
            if provenance.row_offset >= len(inspection.records):
                raise CacheCorruptError("concurrent cache winner has an invalid row offset")
            observed = inspection.records[provenance.row_offset]
            if observed.key != expected.key or observed.embedding != _fp32_vector(
                expected.embedding
            ):
                raise CacheCorruptError(
                    "concurrent cache winner does not match the encoded logical row",
                    details={"path": relative, "row_offset": provenance.row_offset},
                )
        del inspection


def _report_payload(
    *,
    plan: _BuildPlan,
    plan_path: Path,
    state_path: Path,
    request_copy_path: Path,
    request_identity: Mapping[str, object],
    request_count: int,
    cache_root: Path,
    resolved: _ResolvedCache,
    completed: Mapping[str, Mapping[str, object]],
    encoded_rows: int,
    encoded_shards: int,
    resumed_rows: int,
    elapsed_seconds: float,
    batch_size: int,
    rows_per_shard: int,
    created_at_ns: int,
    hardware: str,
    encoder_device: str,
    resolved_config: Mapping[str, object],
    logger: GenoLeWMLogger | None,
    input_artifacts: tuple[Mapping[str, object], ...],
) -> dict[str, object]:
    measured_rows = sum(
        cast(int, entry["encoded_rows"])
        for entry in completed.values()
        if entry.get("encode_batch_seconds") is not None
    )
    measured_seconds = sum(
        cast(float, entry["encode_batch_seconds"])
        for entry in completed.values()
        if entry.get("encode_batch_seconds") is not None
    )
    rate = None if measured_seconds <= 0.0 else measured_rows / measured_seconds
    index_path = cache_root / "embeddings" / INDEX_DB_NAME
    if not index_path.is_file() or index_path.is_symlink():
        raise CacheCorruptError("completed cache build is missing its SQLite index")
    unique_rows = sum(len(shard.rows) for shard in plan.shards)
    return {
        "schema_version": CACHE_BUILD_SCHEMA_VERSION,
        "generated_by": _GENERATED_BY,
        "ok": True,
        "run_id": None if logger is None else logger.run_id,
        "requests": {
            **request_identity,
            "input_rows": request_count,
            "unique_cache_keys": unique_rows,
            "duplicate_rows": request_count - unique_rows,
        },
        "plan": _file_identity(plan_path, root=plan_path.parent),
        "configuration": {
            "batch_size": batch_size,
            "rows_per_shard": rows_per_shard,
            "created_at_ns": created_at_ns,
            "cache_namespace": plan.namespace,
            "hardware": {
                "description": hardware,
                "encoder_device": encoder_device,
            },
            "resolved_config": dict(resolved_config),
        },
        "build": {
            "planned_shards": len(plan.shards),
            "completed_shards": len(plan.shards),
            "encoded_shards": encoded_shards,
            "encoded_rows": encoded_rows,
            "resumed_rows": resumed_rows,
            "resolved_unique_rows": len(resolved.rows),
        },
        "throughput": {
            "invocation_elapsed_seconds": round(max(elapsed_seconds, 0.0), 6),
            "measured_encoded_rows": measured_rows,
            "measured_encoder_seconds": round(measured_seconds, 6),
            "measured_encoded_rows_per_second": rate,
            "measurement_scope": (
                "wall time inside encoder.encode_batch calls for evidence-owned encoded rows; "
                "excludes planning, Python record materialization, Parquet publication, indexing, "
                "verification, and reused shared-cache rows"
            ),
            "measurement_hardware": {
                "description": hardware,
                "encoder_device": encoder_device,
            },
            "measurement_batch_size": batch_size,
            "ten_percent_24h_target_evaluated": False,
        },
        "cache_contract": {
            "schema_version": "3.0.0",
            "storage_dtype": "fp32",
            "logical_dtype": cast(Mapping[str, object], plan.payload["encoder"])["dtype"],
            "normalized_states_persisted": False,
            "deduplication_key_includes_center_token": True,
        },
        "cache_artifacts": {
            "index": {
                "path": index_path.relative_to(cache_root).as_posix(),
                "schema_version": 4,
                "verified_logical_keys": len(resolved.rows),
                "identity_scope": (
                    "request-scoped logical-key mappings; mutable shared index bytes are excluded"
                ),
            },
            "shards": _resolved_shard_payloads(plan, resolved, cache_root=cache_root),
        },
        "evidence_artifacts": {
            "requests": _file_identity(request_copy_path, root=request_copy_path.parent),
            "plan": _file_identity(plan_path, root=plan_path.parent),
            "state": _file_identity(state_path, root=state_path.parent),
            "resolved_config": dict(resolved_config),
            "inputs": [dict(identity) for identity in input_artifacts],
        },
        "progress_events": [
            "data.cache.build.start",
            "data.cache.build.progress",
            "data.shard.write",
            "data.cache.build.end",
        ],
        "claim_boundary": {
            "finite_request_artifact_completed": True,
            "ten_percent_corpus_completed": False,
            "twenty_four_hour_target_evaluated": False,
            "model_quality_evaluated": False,
            "statement": (
                "This report proves construction and byte-level verification of the exact finite "
                "request artifact only; it does not establish corpus coverage, the 24-hour target, "
                "model quality, or clinical validity."
            ),
        },
    }


def _load_or_initialize_state(path: Path, *, plan_sha256: str) -> dict[str, object]:
    if not os.path.lexists(path):
        return _state_payload(plan_sha256=plan_sha256, completed={})
    payload = _read_json_object(path, label="cache build state")
    if set(payload) != {
        "schema_version",
        "generated_by",
        "plan_sha256",
        "completed_shards",
    }:
        raise CacheCorruptError("cache build state has an invalid schema")
    if (
        payload.get("schema_version") != CACHE_BUILD_SCHEMA_VERSION
        or payload.get("generated_by") != _GENERATED_BY
    ):
        raise CacheCorruptError("cache build state has an unsupported schema or producer")
    if payload.get("plan_sha256") != plan_sha256:
        raise CacheCorruptError("cache build state is bound to a different plan")
    if type(payload.get("completed_shards")) is not list:
        raise CacheCorruptError("cache build state completed_shards must be a list")
    return dict(payload)


def _completed_by_id(
    state: Mapping[str, object],
    *,
    plan: _BuildPlan,
) -> dict[str, Mapping[str, object]]:
    planned = {shard.shard_id: shard for shard in plan.shards}
    completed: dict[str, Mapping[str, object]] = {}
    raw_entries = state.get("completed_shards")
    if type(raw_entries) is not list:
        raise CacheCorruptError("cache build state completed_shards must be a list")
    for raw in raw_entries:
        entry = _corrupt_mapping(raw, field="state.completed_shard")
        if set(entry) != {
            "plan_shard_id",
            "execution_shard_id",
            "path",
            "row_keys",
            "sha256",
            "size_bytes",
            "row_count",
            "origin",
            "encoded_rows",
            "encode_batch_calls",
            "encode_batch_seconds",
        }:
            raise CacheCorruptError("cache build state shard entry has an invalid schema")
        plan_shard_id = entry.get("plan_shard_id")
        if (
            type(plan_shard_id) is not str
            or plan_shard_id not in planned
            or plan_shard_id in completed
        ):
            raise CacheCorruptError(
                "cache build state contains an unknown or duplicate shard",
                details={"plan_shard_id": repr(plan_shard_id)},
            )
        plan_shard = planned[plan_shard_id]
        raw_keys = entry.get("row_keys")
        if type(raw_keys) is not list or not raw_keys:
            raise CacheCorruptError("cache build state row_keys must be a non-empty list")
        try:
            keys = tuple(_key_from_payload(raw_key) for raw_key in raw_keys)
        except InputError as exc:
            raise CacheCorruptError("cache build state row_keys are invalid") from exc
        by_key = {row.key: row for row in plan_shard.rows}
        try:
            execution = _execution_shard(
                plan_shard,
                rows=tuple(by_key[key] for key in keys),
            )
        except (KeyError, CacheCorruptError) as exc:
            raise CacheCorruptError(
                "cache build state row_keys are not an ordered plan subset"
            ) from exc
        if (
            entry.get("execution_shard_id") != execution.shard_id
            or entry.get("path") != execution.relative_path
            or entry.get("row_count") != len(execution.rows)
        ):
            raise CacheCorruptError("cache build state shard metadata does not match the plan")
        _validate_artifact_identity(entry, label="state.completed_shard")
        encoded_rows = entry.get("encoded_rows")
        encode_batch_calls = entry.get("encode_batch_calls")
        encode_batch_seconds = entry.get("encode_batch_seconds")
        origin = entry.get("origin")
        if origin not in {"encoded", "adopted_after_interruption"}:
            raise CacheCorruptError("cache build state shard origin is invalid")
        if type(encoded_rows) is not int or encoded_rows < 0 or encoded_rows > len(execution.rows):
            raise CacheCorruptError("cache build state encoded_rows is invalid")
        if type(encode_batch_calls) is not int or encode_batch_calls < 0:
            raise CacheCorruptError("cache build state encode_batch_calls is invalid")
        if encode_batch_seconds is not None and (
            isinstance(encode_batch_seconds, bool)
            or not isinstance(encode_batch_seconds, int | float)
            or not math.isfinite(float(encode_batch_seconds))
            or encode_batch_seconds < 0
        ):
            raise CacheCorruptError("cache build state encode_batch_seconds is invalid")
        if (
            origin == "encoded"
            and (
                encoded_rows != len(execution.rows)
                or encode_batch_calls <= 0
                or encode_batch_seconds is None
            )
        ) or (
            origin == "adopted_after_interruption"
            and (encoded_rows != 0 or encode_batch_calls != 0 or encode_batch_seconds is not None)
        ):
            raise CacheCorruptError(
                "cache build state shard origin and measurements are inconsistent"
            )
        completed[plan_shard_id] = entry
    return completed


def _state_payload(
    *,
    plan_sha256: str,
    completed: Mapping[str, Mapping[str, object]],
) -> dict[str, object]:
    return {
        "schema_version": CACHE_BUILD_SCHEMA_VERSION,
        "generated_by": _GENERATED_BY,
        "plan_sha256": plan_sha256,
        "completed_shards": [dict(completed[key]) for key in sorted(completed)],
    }


def _completed_entry(
    shard: _PlannedShard,
    inspection: CacheShardInspection,
    *,
    plan_shard_id: str,
    origin: str,
    encoded_rows: int,
    encode_batch_calls: int,
    encode_batch_seconds: float | None,
) -> dict[str, object]:
    return {
        "plan_shard_id": plan_shard_id,
        "execution_shard_id": shard.shard_id,
        "path": shard.relative_path,
        "row_keys": [_key_payload(row.key) for row in shard.rows],
        "sha256": inspection.sha256,
        "size_bytes": inspection.size_bytes,
        "row_count": len(inspection.records),
        "origin": origin,
        "encoded_rows": encoded_rows,
        "encode_batch_calls": encode_batch_calls,
        "encode_batch_seconds": encode_batch_seconds,
    }


def _assert_state_identity(
    state: Mapping[str, object],
    inspection: CacheShardInspection,
    *,
    shard: _PlannedShard,
    plan_shard: _PlannedShard,
) -> None:
    if (
        state.get("plan_shard_id") != plan_shard.shard_id
        or state.get("execution_shard_id") != shard.shard_id
        or state.get("row_keys") != [_key_payload(row.key) for row in shard.rows]
        or state.get("sha256") != inspection.sha256
        or state.get("size_bytes") != inspection.size_bytes
    ):
        raise CacheCorruptError(
            "completed cache shard bytes do not match the durable build state",
            details={"shard_id": shard.shard_id, "path": shard.relative_path},
        )


def _resolved_shard_payloads(
    plan: _BuildPlan,
    resolved: _ResolvedCache,
    *,
    cache_root: Path,
) -> list[dict[str, object]]:
    rows_by_path: dict[str, list[dict[str, object]]] = defaultdict(list)
    for shard in plan.shards:
        for row in shard.rows:
            resolved_row = resolved.rows.get(row.key)
            if resolved_row is None:
                raise CacheCorruptError("cache artifact payload is missing a planned logical key")
            provenance = resolved_row.provenance
            relative = provenance.shard_path.absolute().relative_to(cache_root).as_posix()
            rows_by_path[relative].append(
                {
                    "key": _key_payload(row.key),
                    "row_offset": provenance.row_offset,
                    "created_at_ns": resolved_row.created_at_ns,
                    "request_ids": list(row.request_ids),
                }
            )
    payloads: list[dict[str, object]] = []
    for relative in sorted(rows_by_path):
        summary = resolved.shards.get(relative)
        if summary is None:
            raise CacheCorruptError("cache artifact payload is missing a shard summary")
        payloads.append(
            {
                "path": summary.path,
                "sha256": summary.sha256,
                "size_bytes": summary.size_bytes,
                "row_count": summary.row_count,
                "request_rows": sorted(
                    rows_by_path[relative],
                    key=lambda item: cast(int, item["row_offset"]),
                ),
            }
        )
    return payloads


def _expected_evidence_names(
    input_artifacts: tuple[Mapping[str, object], ...],
) -> tuple[str, ...]:
    names = {
        _REQUEST_COPY_NAME,
        _PLAN_NAME,
        _STATE_NAME,
        _RESOLVED_CONFIG_NAME,
        CACHE_BUILD_REPORT_NAME,
        _CHECKSUMS_NAME,
    }
    for identity in input_artifacts:
        path = identity.get("path")
        if type(path) is not str:
            raise InputError("cache build input artifact identity is missing its path")
        names.add(path)
    return tuple(sorted(names))


def _verify_completed_bundle(
    *,
    evidence_root: Path,
    cache_root: Path,
    plan: _BuildPlan,
    plan_sha256: str,
    expected_evidence_names: tuple[str, ...],
) -> CacheBuildReport:
    _assert_evidence_inventory(
        evidence_root,
        expected_names=expected_evidence_names,
        require_complete=True,
    )
    _verify_checksums(evidence_root, expected_names=expected_evidence_names)
    report_path = evidence_root / CACHE_BUILD_REPORT_NAME
    payload = _read_json_object(report_path, label="cache build report")
    if payload.get("schema_version") != CACHE_BUILD_SCHEMA_VERSION or payload.get("ok") is not True:
        raise CacheCorruptError("completed cache build report is invalid")
    plan_identity = _corrupt_mapping(payload.get("plan"), field="report.plan")
    if plan_identity.get("sha256") != plan_sha256:
        raise CacheCorruptError("completed cache build report is bound to another plan")
    resolved = _resolve_plan_cache(cache_root, plan, require_all=True)
    artifacts = _corrupt_mapping(payload.get("cache_artifacts"), field="report.cache_artifacts")
    raw_shards = artifacts.get("shards")
    expected_shards = _resolved_shard_payloads(plan, resolved, cache_root=cache_root)
    if raw_shards != expected_shards:
        raise CacheCorruptError("completed cache build report shard identities have drifted")
    index_identity = _corrupt_mapping(artifacts.get("index"), field="report.cache_artifacts.index")
    index_path = cache_root / "embeddings" / INDEX_DB_NAME
    expected_index = {
        "path": index_path.relative_to(cache_root).as_posix(),
        "schema_version": 4,
        "verified_logical_keys": len(resolved.rows),
        "identity_scope": (
            "request-scoped logical-key mappings; mutable shared index bytes are excluded"
        ),
    }
    if dict(index_identity) != expected_index:
        raise CacheCorruptError("completed cache build index mapping evidence has drifted")
    return CacheBuildReport(
        report_path=report_path,
        checksums_path=evidence_root / _CHECKSUMS_NAME,
        payload=payload,
    )


def _write_checksums(evidence_root: Path, *, expected_names: tuple[str, ...]) -> None:
    _assert_evidence_inventory(
        evidence_root,
        expected_names=expected_names,
        require_complete=False,
    )
    names = tuple(name for name in expected_names if name != _CHECKSUMS_NAME)
    body = "".join(
        f"{sha256_file(evidence_root / name).removeprefix(_HASH_PREFIX)}  {name}\n"
        for name in names
    ).encode("ascii")
    _write_once(evidence_root / _CHECKSUMS_NAME, body, label="cache build checksums")


def _verify_checksums(evidence_root: Path, *, expected_names: tuple[str, ...]) -> None:
    path = evidence_root / _CHECKSUMS_NAME
    body = _read_regular_bytes(path, label="cache build checksums").decode("ascii")
    checksum_names = tuple(name for name in expected_names if name != _CHECKSUMS_NAME)
    lines = body.splitlines()
    if len(lines) != len(checksum_names):
        raise CacheCorruptError("cache build SHA256SUMS has the wrong entry count")
    for line, name in zip(lines, checksum_names, strict=True):
        parts = line.split("  ")
        if len(parts) != 2 or parts[1] != name or len(parts[0]) != 64:
            raise CacheCorruptError("cache build SHA256SUMS has an invalid entry")
        observed = sha256_file(evidence_root / name).removeprefix(_HASH_PREFIX)
        if parts[0] != observed:
            raise CacheCorruptError(
                "cache build evidence checksum mismatch",
                details={"path": name, "expected": parts[0], "observed": observed},
            )


def _assert_evidence_inventory(
    evidence_root: Path,
    *,
    expected_names: tuple[str, ...],
    require_complete: bool,
) -> None:
    observed_names: list[str] = []
    allowed_directories = {
        Path(name).parent.as_posix()
        for name in expected_names
        if Path(name).parent.as_posix() != "."
    }
    for path in sorted(evidence_root.rglob("*")):
        relative = path.relative_to(evidence_root).as_posix()
        mode = path.lstat().st_mode
        if stat.S_ISLNK(mode):
            raise CacheCorruptError(
                "cache build evidence contains a symlink",
                details={"path": relative},
            )
        if stat.S_ISDIR(mode):
            if relative not in allowed_directories:
                raise CacheCorruptError(
                    "cache build evidence contains an unexpected directory",
                    details={"path": relative},
                )
            continue
        if not stat.S_ISREG(mode):
            raise CacheCorruptError(
                "cache build evidence contains a non-regular artifact",
                details={"path": relative},
            )
        observed_names.append(relative)
    unexpected = set(observed_names) - set(expected_names)
    if unexpected:
        raise CacheCorruptError(
            "cache build evidence contains an unexpected artifact",
            details={"unexpected": sorted(unexpected)},
        )
    if require_complete and set(observed_names) != set(expected_names):
        raise CacheCorruptError(
            "cache build evidence is missing a required artifact",
            details={"missing": sorted(set(expected_names) - set(observed_names))},
        )


def _parse_requests(body: bytes) -> tuple[_Request, ...]:
    try:
        text = body.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise InputError("cache build requests must be UTF-8 JSONL") from exc
    if not text or not text.endswith("\n"):
        raise InputError("cache build requests must be non-empty and newline-terminated")
    requests: list[_Request] = []
    seen_ids: set[str] = set()
    for line_no, line in enumerate(text.splitlines(), start=1):
        if not line:
            raise InputError(
                "cache build requests must not contain blank lines",
                details={"line": line_no},
            )
        try:
            raw = json.loads(line, object_pairs_hook=_reject_duplicate_keys)
        except json.JSONDecodeError as exc:
            raise InputError(
                "cache build request line is invalid JSON",
                details={"line": line_no, "error": str(exc)},
            ) from exc
        except InputError as exc:
            raise InputError(
                "cache build request contains a duplicate JSON key",
                details={"line": line_no, "error": str(exc)},
            ) from exc
        if type(raw) is not dict or set(raw) != _REQUEST_KEYS:
            raise InputError(
                "cache build request has an invalid schema",
                details={
                    "line": line_no,
                    "expected": sorted(_REQUEST_KEYS),
                    "observed": sorted(raw) if isinstance(raw, dict) else type(raw).__name__,
                },
            )
        request_id = _text(raw["request_id"], field=f"line {line_no}.request_id")
        if request_id in seen_ids:
            raise InputError(
                "cache build request_id values must be unique",
                details={"line": line_no, "request_id": request_id},
            )
        seen_ids.add(request_id)
        chrom = _text(raw["chrom"], field=f"line {line_no}.chrom")
        start_bp = _non_negative_int(f"line {line_no}.start_bp", raw["start_bp"])
        end_bp = _non_negative_int(f"line {line_no}.end_bp", raw["end_bp"])
        window = canonicalize_dna(raw["window"])
        if not window:
            raise InputError(
                "cache build request window must be non-empty", details={"line": line_no}
            )
        if end_bp - start_bp != len(window):
            raise InputError(
                "cache build request coordinates must span the exact window length",
                details={
                    "line": line_no,
                    "start_bp": start_bp,
                    "end_bp": end_bp,
                    "window_bp": len(window),
                },
            )
        edit_locus = raw["edit_locus"]
        if edit_locus is not None:
            edit_locus = _non_negative_int(f"line {line_no}.edit_locus", edit_locus)
            if edit_locus >= len(window):
                raise InputError(
                    "cache build edit_locus falls outside the window",
                    details={"line": line_no, "edit_locus": edit_locus, "window_bp": len(window)},
                )
        requests.append(
            _Request(
                request_id=request_id,
                chrom=chrom,
                start_bp=start_bp,
                end_bp=end_bp,
                window=window,
                edit_locus=edit_locus,
            )
        )
    if not requests:
        raise InputError("cache build requests must contain at least one row")
    return tuple(requests)


def _encoder_contract(encoder: object) -> _EncoderContract:
    if getattr(encoder, "normalize", None) is not False:
        raise InputError(
            "cache build requires raw pooled encoder states with normalize=false",
            remediation="construct CarbonStateEncoder(..., normalize=False) for cache production",
        )
    encoder_hash = getattr(encoder, "encoder_hash", None)
    if type(encoder_hash) is not bytes or len(encoder_hash) != 32:
        raise RuntimeSetupError("cache build encoder_hash must be exact 32-byte content identity")
    state_layer = getattr(encoder, "state_layer", None)
    if type(state_layer) is not int:
        raise RuntimeSetupError("cache build encoder state_layer must be an integer")
    pool_type = getattr(encoder, "pool_type", None)
    pool_radius = getattr(encoder, "pool_radius", None)
    dtype = getattr(encoder, "dtype", None)
    if type(pool_type) is not str or type(pool_radius) is not int or type(dtype) is not str:
        raise RuntimeSetupError("cache build encoder pooling/dtype metadata is incomplete")
    # Reuse the cache key validator for all static fields.
    center = None if pool_type == POOL_GLOBAL_MEAN else 0
    WindowCacheKey(
        window_hash=bytes(32),
        encoder_hash=encoder_hash,
        state_layer=state_layer,
        pool_type=pool_type,
        pool_radius=pool_radius,
        center_token=center,
        dtype=dtype,
    )
    return _EncoderContract(
        encoder_hash=encoder_hash,
        state_layer=state_layer,
        pool_type=pool_type,
        pool_radius=pool_radius,
        dtype=dtype,
    )


def _resolve_pooling_identity(raw: object) -> tuple[str, int, int | None]:
    if not isinstance(raw, tuple) or len(raw) != 3:
        raise InputError("encoder pooling identity must be a three-item tuple")
    pool_type, pool_radius, center_token = raw
    if type(pool_type) is not str or type(pool_radius) is not int:
        raise InputError("encoder pooling identity has invalid pool metadata")
    WindowCacheKey(
        window_hash=bytes(32),
        encoder_hash=bytes(32),
        state_layer=0,
        pool_type=pool_type,
        pool_radius=pool_radius,
        center_token=center_token,
        dtype="fp32",
    )
    return pool_type, pool_radius, cast(int | None, center_token)


def _state_vector(raw: object) -> tuple[float, ...]:
    if not isinstance(raw, Sequence) or isinstance(raw, str | bytes) or not raw:
        raise InputError("encoder state vector must be a non-empty sequence")
    vector: list[float] = []
    for index, value in enumerate(raw):
        if isinstance(value, bool) or not isinstance(value, int | float):
            raise InputError(
                "encoder state vector must contain physical numbers",
                details={"index": index, "type": type(value).__name__},
            )
        coordinate = float(value)
        if not math.isfinite(coordinate):
            raise InputError(
                "encoder state vector must contain finite values",
                details={"index": index, "value": repr(value)},
            )
        vector.append(coordinate)
    return tuple(vector)


def _fp32_vector(vector: Sequence[float]) -> tuple[float, ...]:
    import struct

    return tuple(struct.unpack("<f", struct.pack("<f", value))[0] for value in vector)


def _key_payload(key: WindowCacheKey) -> dict[str, object]:
    return {
        "window_hash": _hash_text(key.window_hash),
        "encoder_hash": _hash_text(key.encoder_hash),
        "state_layer": key.state_layer,
        "pool_type": key.pool_type,
        "pool_radius": key.pool_radius,
        "center_token": key.center_token,
        "dtype": key.dtype,
    }


def _key_from_payload(raw: object) -> WindowCacheKey:
    block = _mapping(raw, field="plan.shard.row.key")
    expected = {
        "window_hash",
        "encoder_hash",
        "state_layer",
        "pool_type",
        "pool_radius",
        "center_token",
        "dtype",
    }
    if set(block) != expected:
        raise InputError("cache build plan key has an invalid schema")
    return WindowCacheKey(
        window_hash=_hash_bytes(block["window_hash"], field="window_hash"),
        encoder_hash=_hash_bytes(block["encoder_hash"], field="encoder_hash"),
        state_layer=cast(int, block["state_layer"]),
        pool_type=cast(str, block["pool_type"]),
        pool_radius=cast(int, block["pool_radius"]),
        center_token=cast(int | None, block["center_token"]),
        dtype=cast(str, block["dtype"]),
    )


def _request_sort_key(request: _Request) -> tuple[object, ...]:
    return (
        request.chrom,
        request.start_bp,
        request.end_bp,
        request.window,
        -1 if request.edit_locus is None else request.edit_locus,
        request.request_id,
    )


def _planned_row_sort_key(row: _PlannedRow) -> tuple[object, ...]:
    key = row.key
    return (
        key.window_hash.hex(),
        key.encoder_hash.hex(),
        key.state_layer,
        key.pool_type,
        key.pool_radius,
        -1 if key.center_token is None else key.center_token,
        key.dtype,
        *_request_sort_key(row.representative),
    )


def _emit_progress(
    logger: GenoLeWMLogger | None,
    *,
    shard: _PlannedShard,
    completed_shards: int,
    total_shards: int,
    encoded_rows: int,
    resumed_rows: int,
    throughput_per_s: float | None,
    status: str,
) -> None:
    if logger is None:
        return
    logger.info(
        "data.cache.build.progress",
        shard_id=shard.shard_id,
        status=status,
        completed_shards=completed_shards,
        total_shards=total_shards,
        encoded_rows=encoded_rows,
        resumed_rows=resumed_rows,
        throughput_per_s=throughput_per_s,
    )


def _read_input_artifacts(
    artifacts: Mapping[str, Path | str],
) -> tuple[_InputArtifact, ...]:
    if any(type(name) is not str for name in artifacts):
        raise InputError("cache build input artifact names must be text")
    staged: list[_InputArtifact] = []
    for name in sorted(artifacts):
        if (
            not name
            or Path(name).name != name
            or name in {".", ".."}
            or "\x00" in name
            or name
            in {
                _REQUEST_COPY_NAME,
                _PLAN_NAME,
                _STATE_NAME,
                _RESOLVED_CONFIG_NAME,
                CACHE_BUILD_REPORT_NAME,
                _CHECKSUMS_NAME,
            }
        ):
            raise InputError(
                "cache build input artifact name must be a safe unique basename",
                details={"name": repr(name)},
            )
        source = Path(artifacts[name])
        body = _read_regular_bytes(source, label=f"cache build input artifact {name}")
        staged.append(
            _InputArtifact(
                name=name,
                body=body,
                identity={
                    "path": f"inputs/{name}",
                    "sha256": sha256_bytes(body),
                    "size_bytes": len(body),
                },
            )
        )
    return tuple(staged)


def _stage_input_artifacts(
    evidence_root: Path,
    artifacts: tuple[_InputArtifact, ...],
) -> None:
    for artifact in artifacts:
        destination = evidence_root / cast(str, artifact.identity["path"])
        _ensure_real_directory(destination.parent)
        _write_once(
            destination,
            artifact.body,
            label=f"cache build input artifact copy {artifact.name}",
        )


def _file_identity(path: Path, *, root: Path) -> dict[str, object]:
    if path.is_symlink() or not path.is_file():
        raise CacheCorruptError("evidence artifact must be a regular non-symlink file")
    return {
        "path": path.relative_to(root).as_posix(),
        "sha256": sha256_file(path),
        "size_bytes": path.stat().st_size,
    }


def _validate_artifact_identity(payload: Mapping[str, object], *, label: str) -> None:
    digest = payload.get("sha256")
    size = payload.get("size_bytes")
    if (
        type(digest) is not str
        or not digest.startswith(_HASH_PREFIX)
        or len(digest) != len(_HASH_PREFIX) + 64
        or any(character not in "0123456789abcdef" for character in digest[len(_HASH_PREFIX) :])
        or type(size) is not int
        or size < 0
    ):
        raise CacheCorruptError(f"{label} has an invalid artifact identity")


def _json_object_copy(value: Mapping[str, object], *, field: str) -> dict[str, object]:
    if not isinstance(value, Mapping) or any(type(key) is not str for key in value):
        raise InputError(f"{field} must be a JSON object with text keys")
    try:
        encoded = json.dumps(value, sort_keys=True, allow_nan=False)
        decoded = json.loads(encoded, object_pairs_hook=_reject_duplicate_keys)
    except (TypeError, ValueError, InputError) as exc:
        raise InputError(f"{field} must contain only finite JSON-native values") from exc
    if type(decoded) is not dict:
        raise InputError(f"{field} must be a JSON object")
    return cast(dict[str, object], decoded)


def _pretty_json_bytes(payload: Mapping[str, object]) -> bytes:
    return (json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n").encode("utf-8")


def _read_json_object(path: Path, *, label: str) -> dict[str, object]:
    body = _read_regular_bytes(path, label=label)
    try:
        payload = json.loads(body, object_pairs_hook=_reject_duplicate_keys)
    except (UnicodeDecodeError, json.JSONDecodeError, InputError) as exc:
        raise CacheCorruptError(f"{label} is invalid JSON", details={"error": str(exc)}) from exc
    if type(payload) is not dict:
        raise CacheCorruptError(f"{label} must contain one JSON object")
    return cast(dict[str, object], payload)


def _read_regular_bytes(path: Path, *, label: str) -> bytes:
    try:
        mode = path.lstat().st_mode
    except OSError as exc:
        raise InputError(f"{label} could not be inspected", details={"path": str(path)}) from exc
    if stat.S_ISLNK(mode) or not stat.S_ISREG(mode):
        raise InputError(f"{label} must be a regular non-symlink file")
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(path, flags)
    try:
        before = os.fstat(descriptor)
        with os.fdopen(os.dup(descriptor), "rb") as handle:
            body = handle.read()
        after = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    if (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns) != (
        after.st_dev,
        after.st_ino,
        after.st_size,
        after.st_mtime_ns,
    ):
        raise InputError(f"{label} changed while it was being read")
    return body


def _ensure_real_directory(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)
    mode = path.lstat().st_mode
    if stat.S_ISLNK(mode) or not stat.S_ISDIR(mode):
        raise InputError("cache build evidence_dir must be a real directory")


def _write_once(path: Path, body: bytes, *, label: str) -> None:
    if os.path.lexists(path):
        observed = _read_regular_bytes(path, label=label)
        if observed != body:
            raise InputError(f"existing {label} does not match this build")
        return
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(path, flags, 0o444)
    try:
        with os.fdopen(os.dup(descriptor), "wb") as handle:
            handle.write(body)
            handle.flush()
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    _fsync_directory(path.parent)


def _atomic_write(path: Path, body: bytes) -> None:
    _ensure_real_directory(path.parent)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(body)
            handle.flush()
            os.fsync(handle.fileno())
        temporary.chmod(0o444)
        temporary.replace(path)
        _fsync_directory(path.parent)
    finally:
        if os.path.lexists(temporary):
            temporary.unlink()


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _reject_duplicate_keys(pairs: list[tuple[str, object]]) -> dict[str, object]:
    payload: dict[str, object] = {}
    for key, value in pairs:
        if key in payload:
            raise InputError("duplicate JSON key is not allowed", details={"key": key})
        payload[key] = value
    return payload


def _mapping(
    value: object,
    *,
    field: str,
) -> Mapping[str, object]:
    if type(value) is not dict or any(type(key) is not str for key in value):
        raise InputError(f"{field} must be a JSON object")
    return cast(Mapping[str, object], value)


def _corrupt_mapping(value: object, *, field: str) -> Mapping[str, object]:
    if type(value) is not dict or any(type(key) is not str for key in value):
        raise CacheCorruptError(f"{field} must be a JSON object")
    return cast(Mapping[str, object], value)


def _text(value: object, *, field: str) -> str:
    if type(value) is not str or not value or "\x00" in value:
        raise InputError(f"{field} must be non-empty text without NUL")
    return value


def _positive_int(field: str, value: object) -> int:
    parsed = _non_negative_int(field, value)
    if parsed == 0:
        raise InputError(f"{field} must be a positive integer")
    return parsed


def _non_negative_int(field: str, value: object) -> int:
    if type(value) is not int or value < 0 or value > 2**63 - 1:
        raise InputError(f"{field} must be a non-negative 64-bit integer")
    return value


def _hash_text(value: bytes) -> str:
    return f"{_HASH_PREFIX}{value.hex()}"


def _hash_bytes(value: object, *, field: str) -> bytes:
    if (
        type(value) is not str
        or not value.startswith(_HASH_PREFIX)
        or len(value) != len(_HASH_PREFIX) + 64
    ):
        raise InputError(f"{field} must be sha256:<64hex>")
    try:
        return bytes.fromhex(value.removeprefix(_HASH_PREFIX))
    except ValueError as exc:
        raise InputError(f"{field} must be sha256:<64hex>") from exc


def _safe_relative_path(value: object, *, field: str) -> str:
    text = _text(value, field=field)
    path = Path(text)
    if (
        path.is_absolute()
        or path.as_posix() != text
        or any(part in {"", ".", ".."} for part in path.parts)
    ):
        raise InputError(f"{field} must be a canonical relative POSIX path")
    return text
