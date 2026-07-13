# SPDX-License-Identifier: Apache-2.0
"""Parquet raw pooled window-embedding cache and SQLite index.

New writes use the collision-safe schema-3 contract; schema-2 shards remain
read-only compatible for lookup, repair, and reindexing.
Parquet support is intentionally imported lazily so the base package
keeps its minimal dependency surface; install ``geno-lewm[train]`` or
the development extra to use this module.
"""

from __future__ import annotations

import json
import math
import os
import secrets
import sqlite3
import stat
import struct
import tempfile
import time
from collections import defaultdict
from collections.abc import Iterable, Iterator, Sequence
from contextlib import closing, contextmanager, suppress
from dataclasses import dataclass, replace
from hashlib import sha256
from pathlib import Path
from typing import Any, Literal

from geno_lewm.encoder.pooling import POOL_CENTERED_MEAN, POOL_GLOBAL_MEAN
from geno_lewm.errors import CacheCorruptError, InputError, RuntimeSetupError

__all__ = [
    "CACHE_SCHEMA_VERSION",
    "INDEX_DB_NAME",
    "CacheLookupResult",
    "CacheProvenance",
    "CacheReadPolicy",
    "CacheReindexReport",
    "CacheRepairReport",
    "WindowCacheKey",
    "WindowCacheRecord",
    "default_cache_dir",
    "read_cache_entries",
    "read_cache_entry",
    "read_embedding",
    "read_embeddings",
    "reindex_cache",
    "repair_cache",
    "shard_path_for",
    "write_shard",
]


CACHE_SCHEMA_VERSION = "3.0.0"
_LEGACY_CACHE_SCHEMA_VERSION = "2.0.0"
INDEX_DB_NAME = "index.sqlite"
_INDEX_SCHEMA_VERSION = 4
_EMBEDDINGS_DIR = "embeddings"
_QUARANTINE_DIR = ".quarantine"
_PENDING_PUBLICATION_NAME = ".pending-publication.json"
_PENDING_PUBLICATION_SCHEMA_VERSION = "1.0.0"
_STORAGE_DTYPE = "fp32"
_V3_PHYSICAL_ENCODING = "fixed_size_list<float32>"
_V2_PHYSICAL_ENCODING = "list<float16>"
_ROW_GROUP_SIZE = 1_024
_SUPPORTED_DTYPES = frozenset({"bf16", "fp16", "fp32"})
_SUPPORTED_POOL_TYPES = frozenset({POOL_CENTERED_MEAN, POOL_GLOBAL_MEAN, "attention"})
_DIR_FD_PRIMITIVES_AVAILABLE = all(
    operation in getattr(os, "supports_dir_fd", set())
    for operation in (os.open, os.mkdir, os.stat, os.unlink, os.link, os.rename)
)

_WINDOW_INDEX_SQL = """
CREATE TABLE window_index (
    window_hash ANY NOT NULL
        CHECK (
            typeof(window_hash) = 'text'
            AND length(window_hash) = 64
            AND window_hash NOT GLOB '*[^0-9a-f]*'
        ),
    encoder_hash ANY NOT NULL
        CHECK (
            typeof(encoder_hash) = 'text'
            AND length(encoder_hash) = 64
            AND encoder_hash NOT GLOB '*[^0-9a-f]*'
        ),
    state_layer ANY NOT NULL
        CHECK (typeof(state_layer) = 'integer' AND state_layer BETWEEN -128 AND 127),
    pool_type ANY NOT NULL
        CHECK (
            typeof(pool_type) = 'text'
            AND pool_type IN ('centered_mean', 'global_mean', 'attention')
        ),
    pool_radius ANY NOT NULL
        CHECK (
            typeof(pool_radius) = 'integer'
            AND pool_radius BETWEEN 0 AND 2147483647
        ),
    center_token ANY NOT NULL
        CHECK (
            typeof(center_token) = 'integer'
            AND center_token BETWEEN -1 AND 2147483647
        ),
    dtype ANY NOT NULL
        CHECK (typeof(dtype) = 'text' AND dtype IN ('bf16', 'fp16', 'fp32')),
    cache_schema_version ANY NOT NULL
        CHECK (
            typeof(cache_schema_version) = 'text'
            AND cache_schema_version IN ('2.0.0', '3.0.0')
        ),
    physical_encoding ANY NOT NULL
        CHECK (
            typeof(physical_encoding) = 'text'
            AND physical_encoding IN ('list<float16>', 'fixed_size_list<float32>')
        ),
    shard_path ANY NOT NULL
        CHECK (
            typeof(shard_path) = 'text'
            AND length(shard_path) > 0
            AND substr(shard_path, 1, 11) = 'embeddings/'
            AND substr(shard_path, -8) = '.parquet'
            AND instr(shard_path, char(0)) = 0
            AND instr(shard_path, '\\') = 0
            AND shard_path NOT LIKE '%/../%'
            AND shard_path NOT LIKE '../%'
        ),
    row_offset ANY NOT NULL
        CHECK (
            typeof(row_offset) = 'integer'
            AND row_offset BETWEEN 0 AND 9223372036854775807
        ),
    created_at ANY NOT NULL
        CHECK (
            typeof(created_at) = 'integer'
            AND created_at BETWEEN 0 AND 9223372036854775807
        ),
    CHECK (
        (pool_type = 'global_mean' AND pool_radius = 0 AND center_token = -1)
        OR
        (pool_type IN ('centered_mean', 'attention') AND center_token >= 0)
    ),
    CHECK (
        (cache_schema_version = '2.0.0' AND physical_encoding = 'list<float16>')
        OR
        (cache_schema_version = '3.0.0'
            AND physical_encoding = 'fixed_size_list<float32>')
    ),
    PRIMARY KEY (
        window_hash,
        encoder_hash,
        state_layer,
        pool_type,
        pool_radius,
        center_token,
        dtype,
        cache_schema_version,
        physical_encoding
    )
) STRICT
"""
_SHARD_PATH_INDEX_SQL = "CREATE INDEX idx_shard_path ON window_index(shard_path)"

CacheReadPolicy = Literal["require_v3", "prefer_v3", "legacy_v2_only"]


@dataclass(frozen=True, slots=True)
class WindowCacheKey:
    """Content-addressed key for a cached embedding row."""

    window_hash: bytes
    encoder_hash: bytes
    state_layer: int
    pool_type: str
    pool_radius: int
    center_token: int | None
    dtype: str

    def __post_init__(self) -> None:
        _validate_hash("window_hash", self.window_hash)
        _validate_hash("encoder_hash", self.encoder_hash)
        _validate_state_layer(self.state_layer)
        _validate_pool(self.pool_type, self.pool_radius)
        _validate_pool_locus(self.pool_type, self.center_token)
        _validate_dtype(self.dtype)


@dataclass(frozen=True, slots=True)
class WindowCacheRecord:
    """One raw pooled row in the window-embedding cache schema.

    Normalization is a consumer-side view and is intentionally absent from
    the cache key. Cache producers must never persist normalized states.
    """

    chrom: str
    start_bp: int
    end_bp: int
    window_hash: bytes
    encoder_hash: bytes
    state_layer: int
    pool_type: str
    pool_radius: int
    center_token: int | None
    dtype: str
    embedding: tuple[float, ...]
    untargeted: bool
    created_at: int = 0
    schema_version: str = CACHE_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if type(self.chrom) is not str or not self.chrom:
            raise InputError("chrom must be non-empty")
        for field, value in (
            ("start_bp", self.start_bp),
            ("end_bp", self.end_bp),
            ("created_at", self.created_at),
        ):
            if type(value) is not int or value < 0 or value > 2**63 - 1:
                raise InputError(
                    f"{field} must be a non-negative 64-bit integer",
                    details={"field": field, "value": repr(value)},
                )
        if self.end_bp <= self.start_bp:
            raise InputError(
                "end_bp must be greater than start_bp",
                details={"start_bp": self.start_bp, "end_bp": self.end_bp},
            )
        if not self.embedding:
            raise InputError("embedding must contain at least one value")
        for idx, coordinate in enumerate(self.embedding):
            if not isinstance(coordinate, float):
                raise InputError(
                    "embedding values must be physical floating-point values",
                    details={"index": idx, "value": repr(coordinate)},
                )
            if not math.isfinite(coordinate):
                raise InputError(
                    "embedding values must be finite",
                    details={"index": idx, "value": repr(coordinate)},
                )
        if type(self.untargeted) is not bool:
            raise InputError(
                "untargeted must be a boolean",
                details={"untargeted": repr(self.untargeted)},
            )
        if type(self.schema_version) is not str or self.schema_version not in {
            CACHE_SCHEMA_VERSION,
            _LEGACY_CACHE_SCHEMA_VERSION,
        }:
            raise InputError(
                "unsupported cache schema_version",
                details={
                    "schema_version": self.schema_version,
                    "supported": [CACHE_SCHEMA_VERSION, _LEGACY_CACHE_SCHEMA_VERSION],
                },
            )
        WindowCacheKey(
            window_hash=self.window_hash,
            encoder_hash=self.encoder_hash,
            state_layer=self.state_layer,
            pool_type=self.pool_type,
            pool_radius=self.pool_radius,
            center_token=self.center_token,
            dtype=self.dtype,
        )

    @property
    def key(self) -> WindowCacheKey:
        """Return the content-addressed key for this row."""
        return WindowCacheKey(
            window_hash=self.window_hash,
            encoder_hash=self.encoder_hash,
            state_layer=self.state_layer,
            pool_type=self.pool_type,
            pool_radius=self.pool_radius,
            center_token=self.center_token,
            dtype=self.dtype,
        )

    def with_created_at(self) -> WindowCacheRecord:
        """Fill ``created_at`` with current UTC nanoseconds when absent."""
        if self.created_at:
            return self
        return WindowCacheRecord(
            chrom=self.chrom,
            start_bp=self.start_bp,
            end_bp=self.end_bp,
            window_hash=self.window_hash,
            encoder_hash=self.encoder_hash,
            state_layer=self.state_layer,
            pool_type=self.pool_type,
            pool_radius=self.pool_radius,
            center_token=self.center_token,
            dtype=self.dtype,
            embedding=self.embedding,
            untargeted=self.untargeted,
            created_at=time.time_ns(),
            schema_version=self.schema_version,
        )


@dataclass(frozen=True, slots=True)
class CacheReindexReport:
    """Summary of a SQLite index rebuild."""

    indexed_shards: int
    indexed_rows: int
    index_path: Path


@dataclass(frozen=True, slots=True)
class CacheRepairReport:
    """Summary of a repair pass over Parquet shards."""

    checked_shards: int
    quarantined: tuple[Path, ...]
    reindex: CacheReindexReport


@dataclass(frozen=True, slots=True)
class CacheProvenance:
    """Physical source selected for one logical cache-key lookup."""

    cache_schema_version: str
    physical_encoding: str
    shard_path: Path
    row_offset: int


@dataclass(frozen=True, slots=True)
class CacheLookupResult:
    """One cached embedding together with its selected physical provenance."""

    embedding: tuple[float, ...]
    provenance: CacheProvenance


@dataclass(frozen=True, slots=True)
class _RecoveredPublication:
    path: Path
    records: tuple[WindowCacheRecord, ...]


def default_cache_dir() -> Path:
    """Return ``$GENO_LEWM_CACHE`` or the documented local default."""
    return Path(os.environ.get("GENO_LEWM_CACHE", ".geno-lewm-cache")).expanduser()


def shard_path_for(
    cache_dir: Path | str,
    *,
    encoder_id: str,
    state_layer: int,
    pool_type: str,
    pool_radius: int,
    contig: str,
    stride_block: int,
    encoder_hash: bytes | None = None,
    dtype: str | None = None,
) -> Path:
    """Return the canonical Parquet shard path for a cache block.

    Supplying ``encoder_hash`` and ``dtype`` selects the collision-safe v3
    namespace. Omitting both preserves the legacy v2 path contract for
    callers that need to locate existing artifacts.
    """
    _validate_state_layer(state_layer)
    _validate_pool(pool_type, pool_radius)
    if not contig:
        raise InputError("contig must be non-empty")
    if not isinstance(stride_block, int) or isinstance(stride_block, bool) or stride_block < 0:
        raise InputError(
            "stride_block must be a non-negative integer",
            details={"stride_block": stride_block},
        )
    root = Path(cache_dir)
    if (encoder_hash is None) != (dtype is None):
        raise InputError("encoder_hash and dtype must be supplied together")
    if encoder_hash is not None and dtype is not None:
        encoder_part = _digest_path_part("id", encoder_id)
        contig_part = _digest_path_part("ctg", contig)
        encoder_hash_part = _hash_path_part("encoder_hash", encoder_hash)
        _validate_dtype(dtype)
        return (
            root
            / _EMBEDDINGS_DIR
            / "v3"
            / encoder_part
            / encoder_hash_part
            / f"{dtype}_as_{_STORAGE_DTYPE}"
            / str(state_layer)
            / f"{pool_type}_{pool_radius}"
            / f"{contig_part}_{stride_block}.parquet"
        )
    encoder_part = _legacy_path_part(encoder_id)
    contig_part = _legacy_path_part(contig)
    return (
        root
        / _EMBEDDINGS_DIR
        / encoder_part
        / str(state_layer)
        / f"{pool_type}_{pool_radius}"
        / f"chr{contig_part}_{stride_block}.parquet"
    )


def write_shard(
    cache_dir: Path | str,
    *,
    encoder_id: str,
    contig: str,
    stride_block: int,
    records: Sequence[WindowCacheRecord],
) -> Path:
    """Write one immutable Parquet shard and index its rows.

    If the shard already exists with the same rows, this is a no-op.
    If it exists and new or conflicting rows are supplied, the function
    raises instead of rewriting in place (INV-DATA-3 / INV-DATA-10).
    """
    if not records:
        raise InputError("records must contain at least one cache row")
    requested = tuple(records)
    normalized = tuple(_record_for_storage(record.with_created_at()) for record in requested)
    first = normalized[0]
    if first.schema_version != CACHE_SCHEMA_VERSION:
        raise InputError(
            "write_shard only writes the current cache schema",
            details={"schema_version": first.schema_version, "supported": CACHE_SCHEMA_VERSION},
        )
    if any(record.chrom != contig for record in normalized):
        raise InputError("all records in a shard must match the contig argument")
    if any(record.state_layer != first.state_layer for record in normalized):
        raise InputError("all records in a shard must share state_layer")
    if any(record.pool_type != first.pool_type for record in normalized):
        raise InputError("all records in a shard must share pool_type")
    if any(record.pool_radius != first.pool_radius for record in normalized):
        raise InputError("all records in a shard must share pool_radius")
    if any(record.encoder_hash != first.encoder_hash for record in normalized):
        raise InputError("all records in a shard must share encoder_hash")
    if any(record.dtype != first.dtype for record in normalized):
        raise InputError("all records in a shard must share dtype")
    if any(record.schema_version != first.schema_version for record in normalized):
        raise InputError("all records in a shard must share schema_version")
    if any(len(record.embedding) != len(first.embedding) for record in normalized):
        raise InputError("all records in a shard must share embedding width")
    keys = tuple(record.key for record in normalized)
    if len(set(keys)) != len(keys):
        raise InputError("records must not contain a duplicate cache key")

    root = Path(cache_dir)
    path = shard_path_for(
        root,
        encoder_id=encoder_id,
        state_layer=first.state_layer,
        pool_type=first.pool_type,
        pool_radius=first.pool_radius,
        contig=contig,
        stride_block=stride_block,
        encoder_hash=first.encoder_hash,
        dtype=first.dtype,
    )
    _require_secure_cache_io()
    with _cache_publication_lock(root):
        recovered = _recover_pending_publication(root)
        if recovered is not None and _recovered_publication_matches(
            recovered,
            requested=requested,
            normalized=normalized,
        ):
            return recovered.path
        with _open_direct_index_database(root, create=True, write=True) as conn:
            if conn is None:  # create=True guarantees a database or raises.
                raise CacheCorruptError("cache index could not be reserved")
            written = _write_shard_locked(
                root=root,
                path=path,
                requested=requested,
                normalized=normalized,
                index=conn,
            )
        _clear_pending_publication(root)
        return written


def _write_shard_locked(
    *,
    root: Path,
    path: Path,
    requested: Sequence[WindowCacheRecord],
    normalized: Sequence[WindowCacheRecord],
    index: sqlite3.Connection,
) -> Path:
    """Publish and index one shard while holding the cross-process cache lock."""
    _assert_safe_namespace_path(root, path, final_kind="regular file")
    with _secure_parent_directory(root, path, create=True) as parent:
        if parent is None:
            raise CacheCorruptError("cache shard parent disappeared during secure creation")
        return _write_shard_at(
            root=root,
            path=path,
            parent_fd=parent,
            requested=requested,
            normalized=normalized,
            index=index,
        )


def _write_shard_at(
    *,
    root: Path,
    path: Path,
    parent_fd: int,
    requested: Sequence[WindowCacheRecord],
    normalized: Sequence[WindowCacheRecord],
    index: sqlite3.Connection,
) -> Path:
    """Publish relative to a held, no-follow directory descriptor."""
    _verify_directory_binding(path.parent, parent_fd)
    final_fd = _open_regular_at(parent_fd, path.name)
    if final_fd is not None:
        try:
            existing = _read_records_from_descriptor(final_fd, path=path)
        finally:
            os.close(final_fd)
        comparable = normalized
        if len(existing) == len(normalized):
            comparable = tuple(
                replace(incoming, created_at=prior.created_at)
                if requested[index].created_at == 0
                else incoming
                for index, (prior, incoming) in enumerate(zip(existing, normalized, strict=True))
            )
        _assert_existing_shard_equivalent(path, existing, comparable)
        _insert_index_records(index, root, path, existing)
        return path
    _assert_index_keys_available(index, normalized)
    temp_name = f".{path.name}.{secrets.token_hex(16)}.tmp"
    flags = os.O_RDWR | os.O_CREAT | os.O_EXCL
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    temp_fd = os.open(temp_name, flags, 0o600, dir_fd=parent_fd)
    try:
        with os.fdopen(os.dup(temp_fd), "wb") as handle:
            _write_records_to_parquet(handle, normalized)
        os.fsync(temp_fd)
        staged = _read_records_from_descriptor(temp_fd, path=path.with_name(temp_name))
        _assert_existing_shard_equivalent(path.with_name(temp_name), staged, normalized)
        _write_pending_publication(root, path, normalized)
        try:
            os.link(
                temp_name,
                path.name,
                src_dir_fd=parent_fd,
                dst_dir_fd=parent_fd,
                follow_symlinks=False,
            )
            try:
                _verify_directory_binding(path.parent, parent_fd)
            except CacheCorruptError:
                os.unlink(path.name, dir_fd=parent_fd)
                os.fsync(parent_fd)
                raise
        except FileExistsError:
            winner_fd = _open_regular_at(parent_fd, path.name)
            if winner_fd is None:  # The name disappeared after reporting EEXIST.
                raise CacheCorruptError(
                    "cache shard winner disappeared during no-clobber installation",
                    details={"shard_path": str(path)},
                ) from None
            try:
                winner = _read_records_from_descriptor(winner_fd, path=path)
            finally:
                os.close(winner_fd)
            comparable = normalized
            if len(winner) == len(normalized):
                comparable = tuple(
                    replace(incoming, created_at=prior.created_at)
                    if requested[index].created_at == 0
                    else incoming
                    for index, (prior, incoming) in enumerate(zip(winner, normalized, strict=True))
                )
            _assert_existing_shard_equivalent(path, winner, comparable)
            _insert_index_records(index, root, path, winner)
            return path
        os.fsync(parent_fd)
    finally:
        os.close(temp_fd)
        try:
            os.unlink(temp_name, dir_fd=parent_fd)
        except FileNotFoundError:
            pass
        else:
            os.fsync(parent_fd)
    _insert_index_records(index, root, path, normalized)
    return path


def _pending_publication_path(cache_dir: Path) -> Path:
    return cache_dir / _EMBEDDINGS_DIR / _PENDING_PUBLICATION_NAME


def _write_pending_publication(
    cache_dir: Path,
    shard: Path,
    records: Sequence[WindowCacheRecord],
) -> None:
    """Durably record the sole in-flight shard before its final name is linked."""
    pending = _pending_publication_path(cache_dir)
    relative_shard = shard.absolute().relative_to(cache_dir.absolute()).as_posix()
    payload = {
        "schema_version": _PENDING_PUBLICATION_SCHEMA_VERSION,
        "shard_path": relative_shard,
        "row_count": len(records),
        "record_sha256": [
            sha256(_canonical_record_bytes(record)).hexdigest() for record in records
        ],
    }
    body = (
        json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True) + "\n"
    ).encode("ascii")
    with _secure_parent_directory(cache_dir, pending, create=True) as parent_fd:
        if parent_fd is None:
            raise RuntimeSetupError(
                "cache publication requires secure directory-descriptor operations"
            )
        existing = _open_regular_at(
            parent_fd,
            pending.name,
            label="cache pending-publication record",
        )
        if existing is not None:
            os.close(existing)
            raise CacheCorruptError(
                "cache has an unresolved pending publication",
                details={"path": str(pending)},
            )
        temp_name = f".{pending.name}.{secrets.token_hex(16)}.tmp"
        flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW
        descriptor = os.open(temp_name, flags, 0o600, dir_fd=parent_fd)
        try:
            with os.fdopen(os.dup(descriptor), "wb") as handle:
                handle.write(body)
                handle.flush()
            os.fsync(descriptor)
            try:
                os.link(
                    temp_name,
                    pending.name,
                    src_dir_fd=parent_fd,
                    dst_dir_fd=parent_fd,
                    follow_symlinks=False,
                )
            except FileExistsError as exc:
                raise CacheCorruptError(
                    "cache pending-publication record appeared during reservation",
                    details={"path": str(pending)},
                ) from exc
            os.fsync(parent_fd)
        finally:
            os.close(descriptor)
            with suppress(FileNotFoundError):
                os.unlink(temp_name, dir_fd=parent_fd)
                os.fsync(parent_fd)


def _recover_pending_publication(cache_dir: Path) -> _RecoveredPublication | None:
    """Resolve the one durable in-flight publication without scanning cache shards."""
    resolved_root = cache_dir.resolve()
    pending = _pending_publication_path(resolved_root)
    if not os.path.lexists(pending):
        return None
    payload = _read_pending_publication(resolved_root, pending)
    shard = _indexed_shard_path(resolved_root, payload["shard_path"])
    _assert_safe_namespace_path(resolved_root, shard, final_kind="regular file")
    if not os.path.lexists(shard):
        _clear_pending_publication(resolved_root)
        return None
    records = _read_records_from_cache_shard(resolved_root, shard)
    observed_hashes = tuple(
        sha256(_canonical_record_bytes(record)).hexdigest() for record in records
    )
    expected_hashes = payload["record_sha256"]
    if len(records) != payload["row_count"] or observed_hashes != expected_hashes:
        raise CacheCorruptError(
            "pending cache publication does not match its final shard",
            details={"shard_path": str(shard)},
        )
    with _open_direct_index_database(resolved_root, create=True, write=True) as conn:
        if conn is None:
            raise CacheCorruptError("cache index could not be reserved during recovery")
        _insert_index_records(conn, resolved_root, shard, records)
    _clear_pending_publication(resolved_root)
    return _RecoveredPublication(
        path=cache_dir / Path(payload["shard_path"]),
        records=records,
    )


def _read_pending_publication(cache_dir: Path, pending: Path) -> dict[str, Any]:
    with _secure_parent_directory(cache_dir, pending, create=False) as parent_fd:
        if parent_fd is None:
            raise RuntimeSetupError(
                "cache recovery requires secure directory-descriptor operations"
            )
        descriptor = _open_regular_at(
            parent_fd,
            pending.name,
            label="cache pending-publication record",
        )
        if descriptor is None:
            raise CacheCorruptError("cache pending-publication record disappeared")
        try:
            with os.fdopen(os.dup(descriptor), "rb") as handle:
                body = handle.read(1_000_001)
        finally:
            os.close(descriptor)
    if len(body) > 1_000_000:
        raise CacheCorruptError("cache pending-publication record is unreasonably large")
    try:
        payload = json.loads(body)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise CacheCorruptError(
            "cache pending-publication record is invalid JSON",
            details={"path": str(pending)},
        ) from exc
    if type(payload) is not dict or set(payload) != {
        "record_sha256",
        "row_count",
        "schema_version",
        "shard_path",
    }:
        raise CacheCorruptError("cache pending-publication record has an invalid schema")
    if payload.get("schema_version") != _PENDING_PUBLICATION_SCHEMA_VERSION:
        raise CacheCorruptError("cache pending-publication record has an unsupported version")
    shard_path = payload.get("shard_path")
    row_count = payload.get("row_count")
    record_hashes = payload.get("record_sha256")
    if type(shard_path) is not str or not shard_path:
        raise CacheCorruptError("cache pending-publication shard_path is invalid")
    if type(row_count) is not int or row_count <= 0:
        raise CacheCorruptError("cache pending-publication row_count is invalid")
    if (
        type(record_hashes) is not list
        or len(record_hashes) != row_count
        or any(
            type(value) is not str
            or len(value) != 64
            or any(character not in "0123456789abcdef" for character in value)
            for value in record_hashes
        )
    ):
        raise CacheCorruptError("cache pending-publication record hashes are invalid")
    return {
        "schema_version": payload["schema_version"],
        "shard_path": shard_path,
        "row_count": row_count,
        "record_sha256": tuple(record_hashes),
    }


def _clear_pending_publication(cache_dir: Path) -> None:
    pending = _pending_publication_path(cache_dir)
    if not os.path.lexists(pending):
        return
    with _secure_parent_directory(cache_dir, pending, create=False) as parent_fd:
        if parent_fd is None:
            raise RuntimeSetupError(
                "cache recovery requires secure directory-descriptor operations"
            )
        descriptor = _open_regular_at(
            parent_fd,
            pending.name,
            label="cache pending-publication record",
        )
        if descriptor is None:
            return
        os.close(descriptor)
        os.unlink(pending.name, dir_fd=parent_fd)
        os.fsync(parent_fd)


def _recovered_publication_matches(
    recovered: _RecoveredPublication,
    *,
    requested: Sequence[WindowCacheRecord],
    normalized: Sequence[WindowCacheRecord],
) -> bool:
    if len(recovered.records) != len(normalized):
        return False
    for existing, requested_record, normalized_record in zip(
        recovered.records,
        requested,
        normalized,
        strict=True,
    ):
        comparable = normalized_record
        if requested_record.created_at == 0:
            comparable = replace(normalized_record, created_at=existing.created_at)
        if _canonical_record_bytes(existing) != _canonical_record_bytes(comparable):
            return False
    return True


@contextmanager
def _secure_parent_directory(
    cache_dir: Path,
    target: Path,
    *,
    create: bool,
) -> Iterator[int | None]:
    """Open a namespace parent using dirfd/no-follow traversal when supported."""
    _require_secure_cache_io()
    root_created = not os.path.lexists(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    _reject_symlink_or_non_directory(cache_dir, label="cache root")
    flags = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW
    root_fd = os.open(cache_dir, flags)
    if root_created:
        os.fsync(root_fd)
        _fsync_directory(cache_dir.parent)
    current_fd = root_fd
    try:
        relative_parent = target.parent.absolute().relative_to(cache_dir.absolute())
        for part in relative_parent.parts:
            try:
                child_fd = os.open(part, flags, dir_fd=current_fd)
            except FileNotFoundError:
                if not create:
                    yield None
                    return
                with suppress(FileExistsError):
                    os.mkdir(part, 0o700, dir_fd=current_fd)
                os.fsync(current_fd)
                try:
                    child_fd = os.open(part, flags, dir_fd=current_fd)
                except OSError as exc:
                    raise CacheCorruptError(
                        "cache namespace parent became unsafe during creation",
                        details={"component": part, "error": str(exc)},
                    ) from exc
                os.fsync(child_fd)
            except OSError as exc:
                raise CacheCorruptError(
                    "cache namespace parent is a symlink or unsafe directory",
                    details={"component": part, "error": str(exc)},
                ) from exc
            if current_fd != root_fd:
                os.close(current_fd)
            current_fd = child_fd
        yield current_fd
    finally:
        if current_fd != root_fd:
            os.close(current_fd)
        os.close(root_fd)


def _require_secure_cache_io() -> None:
    if (
        os.name == "nt"
        or not hasattr(os, "O_DIRECTORY")
        or not hasattr(os, "O_NOFOLLOW")
        or not _DIR_FD_PRIMITIVES_AVAILABLE
    ):
        raise RuntimeSetupError(
            "cache I/O requires secure POSIX directory-descriptor and no-follow primitives",
            details={"platform": os.name},
            remediation=(
                "build, read, repair, or reindex corrected caches on Linux or macOS; "
                "transfer immutable artifacts only after verification"
            ),
        )


def _open_regular_at(parent_fd: int, name: str, *, label: str = "cache shard final") -> int | None:
    flags = os.O_RDONLY
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        descriptor = os.open(name, flags, dir_fd=parent_fd)
    except FileNotFoundError:
        return None
    except OSError as exc:
        raise CacheCorruptError(
            f"{label} is a symlink or unsafe file",
            details={"name": name, "error": str(exc)},
        ) from exc
    if not stat.S_ISREG(os.fstat(descriptor).st_mode):
        os.close(descriptor)
        raise CacheCorruptError(
            f"{label} must be a regular file",
            details={"name": name},
        )
    return descriptor


def _verify_directory_binding(path: Path, descriptor: int) -> None:
    held = os.fstat(descriptor)
    try:
        observed = path.lstat()
    except OSError as exc:
        raise CacheCorruptError(
            "cache namespace directory binding changed during operation",
            details={"path": str(path), "error": str(exc)},
        ) from exc
    if stat.S_ISLNK(observed.st_mode) or not stat.S_ISDIR(observed.st_mode):
        raise CacheCorruptError("cache namespace directory became a symlink or unsafe")
    if (held.st_dev, held.st_ino) != (observed.st_dev, observed.st_ino):
        raise CacheCorruptError("cache namespace directory binding changed during operation")


@contextmanager
def _cache_publication_lock(cache_dir: Path) -> Iterator[None]:
    """Serialize shard/path/key publication across cooperating processes."""
    lock_path = cache_dir / _EMBEDDINGS_DIR / ".publish.lock"
    with _secure_parent_directory(cache_dir, lock_path, create=True) as parent_fd:
        flags = os.O_RDWR | os.O_CREAT
        flags |= os.O_NOFOLLOW
        try:
            if parent_fd is None:
                raise CacheCorruptError("cache publication-lock parent disappeared")
            existing = _open_regular_at(
                parent_fd,
                lock_path.name,
                label="cache publication lock",
            )
            if existing is not None:
                descriptor = existing
            else:
                try:
                    descriptor = os.open(
                        lock_path.name,
                        flags | os.O_EXCL,
                        0o600,
                        dir_fd=parent_fd,
                    )
                except FileExistsError:
                    reserved_descriptor = _open_regular_at(
                        parent_fd,
                        lock_path.name,
                        label="cache publication lock",
                    )
                    if reserved_descriptor is None:
                        raise CacheCorruptError(
                            "cache publication lock disappeared during reservation"
                        ) from None
                    descriptor = reserved_descriptor
                else:
                    os.fsync(descriptor)
                    os.fsync(parent_fd)
        except OSError as exc:
            raise CacheCorruptError(
                "cache publication lock is unsafe or unavailable",
                details={"path": str(lock_path), "error": str(exc)},
            ) from exc
        try:
            if not stat.S_ISREG(os.fstat(descriptor).st_mode):
                raise CacheCorruptError(
                    "cache publication lock must be a regular file",
                    details={"path": str(lock_path)},
                )
            import fcntl

            fcntl.flock(descriptor, fcntl.LOCK_EX)
            yield
        finally:
            import fcntl

            fcntl.flock(descriptor, fcntl.LOCK_UN)
            os.close(descriptor)


def _reject_symlink_or_non_directory(path: Path, *, label: str) -> None:
    try:
        mode = path.lstat().st_mode
    except OSError as exc:
        raise CacheCorruptError(
            f"{label} could not be inspected",
            details={"path": str(path), "error": str(exc)},
        ) from exc
    if stat.S_ISLNK(mode) or not stat.S_ISDIR(mode):
        raise CacheCorruptError(
            f"{label} must be a real directory",
            details={"path": str(path)},
        )


def read_embedding(
    cache_dir: Path | str,
    key: WindowCacheKey,
    *,
    policy: CacheReadPolicy = "require_v3",
) -> tuple[float, ...] | None:
    """Return a raw pooled embedding by content key, or ``None`` on cache miss."""
    result = read_cache_entry(cache_dir, key, policy=policy)
    return None if result is None else result.embedding


def read_embeddings(
    cache_dir: Path | str,
    keys: Sequence[WindowCacheKey],
    *,
    policy: CacheReadPolicy = "require_v3",
) -> tuple[tuple[float, ...] | None, ...]:
    """Return raw embeddings for ``keys`` in order, grouping reads by shard.

    Duplicate keys and misses are preserved in the returned tuple. Only the
    Parquet row groups containing requested rows are read.
    """
    return tuple(
        None if result is None else result.embedding
        for result in read_cache_entries(cache_dir, keys, policy=policy)
    )


def read_cache_entry(
    cache_dir: Path | str,
    key: WindowCacheKey,
    *,
    policy: CacheReadPolicy = "require_v3",
) -> CacheLookupResult | None:
    """Return one embedding and its selected cache provenance."""
    return read_cache_entries(cache_dir, (key,), policy=policy)[0]


def read_cache_entries(
    cache_dir: Path | str,
    keys: Sequence[WindowCacheKey],
    *,
    policy: CacheReadPolicy = "require_v3",
) -> tuple[CacheLookupResult | None, ...]:
    """Return cache entries in request order under an explicit provenance policy."""
    _validate_read_policy(policy)
    if not keys:
        return ()
    root = Path(cache_dir)
    index_path = _index_path(root)
    if not os.path.lexists(index_path):
        return tuple(None for _key in keys)
    _require_secure_cache_io()
    locations: dict[WindowCacheKey, CacheProvenance] = {}
    indexed_paths: dict[str, Path] = {}
    with _open_direct_index_database(root, create=False, write=False) as conn:
        if conn is None:
            return tuple(None for _key in keys)
        for key in dict.fromkeys(keys):
            rows = conn.execute(
                """
                SELECT cache_schema_version, physical_encoding, shard_path, row_offset
                FROM window_index
                WHERE window_hash = ?
                  AND encoder_hash = ?
                  AND state_layer = ?
                  AND pool_type = ?
                  AND pool_radius = ?
                  AND center_token = ?
                  AND dtype = ?
                ORDER BY cache_schema_version DESC, physical_encoding, shard_path, row_offset
                """,
                _index_key_params(key),
            ).fetchall()
            selected = _select_index_location(rows, policy=policy, key=key)
            if selected is not None:
                schema_version, physical_encoding, relative_path, row_offset = selected
                shard_path = indexed_paths.get(relative_path)
                if shard_path is None:
                    shard_path = _indexed_shard_path(root, relative_path)
                    indexed_paths[relative_path] = shard_path
                locations[key] = CacheProvenance(
                    cache_schema_version=schema_version,
                    physical_encoding=physical_encoding,
                    shard_path=shard_path,
                    row_offset=row_offset,
                )
    requests_by_shard: dict[Path, list[tuple[int, WindowCacheKey, int]]] = defaultdict(list)
    for result_index, key in enumerate(keys):
        provenance = locations.get(key)
        if provenance is not None:
            requests_by_shard[provenance.shard_path].append(
                (result_index, key, provenance.row_offset)
            )
    results: list[CacheLookupResult | None] = [None] * len(keys)
    for shard_path, requests in requests_by_shard.items():
        requested_offsets = {request[2] for request in requests}
        records = _read_records_at_offsets(
            shard_path,
            requested_offsets,
            cache_dir=root,
        )
        missing_offsets = requested_offsets - records.keys()
        if missing_offsets:
            raise CacheCorruptError(
                "cache index row_offset could not be resolved in shard",
                details={
                    "shard_path": str(shard_path),
                    "row_offsets": sorted(missing_offsets),
                },
            )
        for result_index, key, row_offset in requests:
            record = records[row_offset]
            if record.key != key:
                raise CacheCorruptError(
                    "cache index key does not match shard row",
                    details={"shard_path": str(shard_path), "row_offset": row_offset},
                )
            provenance = locations[key]
            if record.schema_version != provenance.cache_schema_version:
                raise CacheCorruptError(
                    "cache index provenance does not match shard row",
                    details={"shard_path": str(shard_path), "row_offset": row_offset},
                )
            results[result_index] = CacheLookupResult(
                embedding=record.embedding,
                provenance=provenance,
            )
    return tuple(results)


def reindex_cache(cache_dir: Path | str) -> CacheReindexReport:
    """Rebuild ``index.sqlite`` from every readable Parquet shard."""
    root = Path(cache_dir)
    _require_secure_cache_io()
    with _cache_publication_lock(root):
        return _reindex_cache_locked(root)


def _reindex_cache_locked(root: Path) -> CacheReindexReport:
    """Rebuild the complete index while the publication lock is held."""
    index_path = _index_path(root)
    indexed_shards = 0
    indexed_rows = 0
    with tempfile.TemporaryDirectory(prefix="geno-lewm-index-") as working_dir:
        working_path = Path(working_dir) / INDEX_DB_NAME
        with closing(sqlite3.connect(working_path)) as conn:
            _ensure_index_schema(conn, create=True)
            for shard in _iter_shards(root):
                records = _read_records_from_cache_shard(root, shard)
                _insert_index_records(conn, root, shard, records)
                indexed_shards += 1
                indexed_rows += len(records)
            conn.commit()
            integrity = conn.execute("PRAGMA integrity_check").fetchone()
            if integrity is None or type(integrity[0]) is not str or integrity[0] != "ok":
                raise CacheCorruptError(
                    "rebuilt cache index failed SQLite integrity_check",
                    details={"index_path": str(index_path), "result": integrity},
                )
        _publish_index_bytes(root, working_path.read_bytes())
        _recover_pending_publication(root)
    return CacheReindexReport(
        indexed_shards=indexed_shards,
        indexed_rows=indexed_rows,
        index_path=index_path,
    )


def repair_cache(cache_dir: Path | str) -> CacheRepairReport:
    """Quarantine unreadable Parquet shards and rebuild the SQLite index."""
    root = Path(cache_dir)
    _require_secure_cache_io()
    quarantined: list[Path] = []
    checked = 0
    with _cache_publication_lock(root):
        for shard in list(_iter_shards(root)):
            checked += 1
            try:
                _read_records_from_cache_shard(root, shard)
            except CacheCorruptError:
                quarantined.append(_quarantine_shard(root, shard))
        report = _reindex_cache_locked(root)
    return CacheRepairReport(
        checked_shards=checked,
        quarantined=tuple(quarantined),
        reindex=report,
    )


def _require_pyarrow() -> tuple[Any, Any]:
    try:
        import pyarrow as pa
        import pyarrow.parquet as pq
    except ImportError as exc:  # pragma: no cover - exercised in downstream packaging envs
        raise RuntimeSetupError(
            "window cache requires pyarrow for Parquet IO",
            remediation="install geno-lewm[train] or install pyarrow",
        ) from exc
    return pa, pq


def _write_records_to_parquet(path: Any, records: Sequence[WindowCacheRecord]) -> None:
    pa, pq = _require_pyarrow()
    table = pa.Table.from_pydict(
        {
            "chrom": [record.chrom for record in records],
            "start_bp": [record.start_bp for record in records],
            "end_bp": [record.end_bp for record in records],
            "window_hash": [record.window_hash for record in records],
            "encoder_hash": [record.encoder_hash for record in records],
            "state_layer": [record.state_layer for record in records],
            "pool_type": [record.pool_type for record in records],
            "pool_radius": [record.pool_radius for record in records],
            "center_token": [record.center_token for record in records],
            "dtype": [record.dtype for record in records],
            "storage_dtype": [_STORAGE_DTYPE for _record in records],
            "embedding": [list(record.embedding) for record in records],
            "untargeted": [record.untargeted for record in records],
            "created_at": [record.created_at for record in records],
            "schema_version": [record.schema_version for record in records],
        },
        schema=_arrow_schema(pa, embedding_size=len(records[0].embedding)),
    )
    pq.write_table(
        table,
        path,
        compression="zstd",
        compression_level=9,
        row_group_size=_ROW_GROUP_SIZE,
    )


def _read_records_from_shard(path: Path) -> tuple[WindowCacheRecord, ...]:
    _assert_path_is_regular_without_symlink(path, label="cache shard")
    with path.open("rb") as handle:
        return _read_records_from_source(handle, path=path)


def _read_records_from_descriptor(
    descriptor: int,
    *,
    path: Path,
) -> tuple[WindowCacheRecord, ...]:
    os.lseek(descriptor, 0, os.SEEK_SET)
    with os.fdopen(os.dup(descriptor), "rb") as handle:
        return _read_records_from_source(handle, path=path)


def _read_records_from_cache_shard(cache_dir: Path, path: Path) -> tuple[WindowCacheRecord, ...]:
    with _secure_parent_directory(cache_dir, path, create=False) as parent_fd:
        if parent_fd is None:
            raise CacheCorruptError(
                "cache shard parent disappeared during secure open",
                details={"shard_path": str(path)},
            )
        descriptor = _open_regular_at(parent_fd, path.name, label="cache shard")
        if descriptor is None:
            raise CacheCorruptError(
                "cache shard disappeared during secure open",
                details={"shard_path": str(path)},
            )
        try:
            return _read_records_from_descriptor(descriptor, path=path)
        finally:
            os.close(descriptor)


def _read_records_from_source(
    source: Any,
    *,
    path: Path,
) -> tuple[WindowCacheRecord, ...]:
    pa, pq = _require_pyarrow()
    try:
        table = pq.read_table(source)
    except Exception as exc:
        raise CacheCorruptError(
            "cache shard could not be read",
            details={"shard_path": str(path), "error": str(exc)},
        ) from exc
    try:
        schema_version = _schema_version_from_table(table, path=path)
        _validate_physical_schema(pa, table, path=path, schema_version=schema_version)
    except CacheCorruptError:
        raise
    except Exception as exc:
        raise CacheCorruptError(
            "cache shard contains an invalid physical schema",
            details={"shard_path": str(path), "error": str(exc)},
        ) from exc
    try:
        return tuple(_record_from_row(row) for row in table.to_pylist())
    except (InputError, TypeError, ValueError) as exc:
        raise CacheCorruptError(
            "cache shard contains an invalid row",
            details={"shard_path": str(path), "error": str(exc)},
        ) from exc


def _assert_path_is_regular_without_symlink(path: Path, *, label: str) -> None:
    try:
        mode = path.lstat().st_mode
    except OSError as exc:
        raise CacheCorruptError(
            f"{label} could not be inspected",
            details={"path": str(path), "error": str(exc)},
        ) from exc
    if stat.S_ISLNK(mode) or not stat.S_ISREG(mode):
        raise CacheCorruptError(
            f"{label} must be a regular non-symlink file",
            details={"path": str(path)},
        )


def _read_records_at_offsets(
    path: Path,
    row_offsets: set[int],
    *,
    cache_dir: Path | None = None,
) -> dict[int, WindowCacheRecord]:
    if cache_dir is not None:
        with _secure_parent_directory(cache_dir, path, create=False) as parent_fd:
            if parent_fd is None:
                raise CacheCorruptError(
                    "indexed cache shard parent disappeared during secure open",
                    details={"shard_path": str(path)},
                )
            descriptor = _open_regular_at(parent_fd, path.name, label="cache shard")
            if descriptor is None:
                raise CacheCorruptError(
                    "indexed cache shard disappeared during secure open",
                    details={"shard_path": str(path)},
                )
            try:
                with os.fdopen(os.dup(descriptor), "rb") as handle:
                    return _read_records_at_offsets_from_source(
                        handle,
                        path=path,
                        row_offsets=row_offsets,
                    )
            finally:
                os.close(descriptor)
    _assert_path_is_regular_without_symlink(path, label="cache shard")
    with path.open("rb") as handle:
        return _read_records_at_offsets_from_source(handle, path=path, row_offsets=row_offsets)


def _read_records_at_offsets_from_source(
    source: Any,
    *,
    path: Path,
    row_offsets: set[int],
) -> dict[int, WindowCacheRecord]:
    pa, pq = _require_pyarrow()
    try:
        parquet = pq.ParquetFile(source)
    except Exception as exc:
        raise CacheCorruptError(
            "cache shard could not be read",
            details={"shard_path": str(path), "error": str(exc)},
        ) from exc
    row_count = parquet.metadata.num_rows
    invalid = sorted(offset for offset in row_offsets if offset < 0 or offset >= row_count)
    if invalid:
        raise CacheCorruptError(
            "cache index row_offset points outside shard",
            details={"shard_path": str(path), "row_offsets": invalid, "rows": row_count},
        )
    records: dict[int, WindowCacheRecord] = {}
    group_start = 0
    sorted_offsets = sorted(row_offsets)
    offset_cursor = 0
    try:
        for group_index in range(parquet.num_row_groups):
            group_rows = parquet.metadata.row_group(group_index).num_rows
            group_end = group_start + group_rows
            selected: list[int] = []
            while offset_cursor < len(sorted_offsets) and sorted_offsets[offset_cursor] < group_end:
                selected.append(sorted_offsets[offset_cursor])
                offset_cursor += 1
            if selected:
                table = parquet.read_row_group(group_index)
                schema_version = _schema_version_from_table(table, path=path)
                _validate_physical_schema(
                    pa,
                    table,
                    path=path,
                    schema_version=schema_version,
                )
                for offset in selected:
                    local_offset = offset - group_start
                    row = table.slice(local_offset, 1).to_pylist()[0]
                    records[offset] = _record_from_row(row)
            group_start = group_end
    except CacheCorruptError:
        raise
    except Exception as exc:
        raise CacheCorruptError(
            "cache shard contains an invalid indexed row",
            details={"shard_path": str(path), "error": str(exc)},
        ) from exc
    return records


def _schema_version_from_table(table: Any, *, path: Path) -> str:
    required = set(_column_names(schema_version=_LEGACY_CACHE_SCHEMA_VERSION))
    observed = set(table.column_names)
    if required - observed:
        raise CacheCorruptError(
            "cache shard is missing required column(s)",
            details={"shard_path": str(path), "missing": sorted(required - observed)},
        )
    schema_version_values = table.column("schema_version").to_pylist()
    if any(type(value) is not str for value in schema_version_values):
        raise CacheCorruptError(
            "cache shard schema_version must contain only text",
            details={"shard_path": str(path)},
        )
    schema_versions = set(schema_version_values)
    if len(schema_versions) != 1:
        raise CacheCorruptError(
            "cache shard must contain one schema_version",
            details={
                "shard_path": str(path),
                "schema_versions": sorted(repr(value) for value in schema_versions),
            },
        )
    schema_version = next(iter(schema_versions))
    if type(schema_version) is not str:
        raise CacheCorruptError(
            "cache shard schema_version must be text",
            details={"shard_path": str(path), "value": repr(schema_version)},
        )
    if schema_version not in {CACHE_SCHEMA_VERSION, _LEGACY_CACHE_SCHEMA_VERSION}:
        raise CacheCorruptError(
            "cache shard uses an unsupported schema_version",
            details={"shard_path": str(path), "schema_version": schema_version},
        )
    required = set(_column_names(schema_version=schema_version))
    if required - observed:
        raise CacheCorruptError(
            "cache shard is missing required column(s)",
            details={"shard_path": str(path), "missing": sorted(required - observed)},
        )
    if schema_version == CACHE_SCHEMA_VERSION:
        storage_dtype_values = table.column("storage_dtype").to_pylist()
        if any(type(value) is not str for value in storage_dtype_values):
            raise CacheCorruptError(
                "cache shard storage_dtype must contain only text",
                details={"shard_path": str(path)},
            )
        storage_dtypes = set(storage_dtype_values)
        if storage_dtypes != {_STORAGE_DTYPE}:
            raise CacheCorruptError(
                "cache shard storage_dtype does not match its physical schema",
                details={
                    "shard_path": str(path),
                    "storage_dtypes": sorted(repr(value) for value in storage_dtypes),
                },
            )
    return schema_version


def _record_from_row(row: dict[str, Any]) -> WindowCacheRecord:
    _require_row_type(row, "chrom", str)
    for field in ("start_bp", "end_bp", "state_layer", "pool_radius", "created_at"):
        _require_row_type(row, field, int)
    for field in ("window_hash", "encoder_hash"):
        _require_row_type(row, field, bytes)
    for field in ("pool_type", "dtype", "schema_version"):
        _require_row_type(row, field, str)
    _require_row_type(row, "untargeted", bool)
    center_token = row["center_token"]
    if center_token is not None and type(center_token) is not int:
        raise InputError("center_token must be an integer or null")
    embedding = row["embedding"]
    if type(embedding) is not list or any(type(value) is not float for value in embedding):
        raise InputError("embedding must contain only physical floating-point values")
    return WindowCacheRecord(
        chrom=row["chrom"],
        start_bp=row["start_bp"],
        end_bp=row["end_bp"],
        window_hash=row["window_hash"],
        encoder_hash=row["encoder_hash"],
        state_layer=row["state_layer"],
        pool_type=row["pool_type"],
        pool_radius=row["pool_radius"],
        center_token=center_token,
        dtype=row["dtype"],
        embedding=tuple(embedding),
        untargeted=row["untargeted"],
        created_at=row["created_at"],
        schema_version=row["schema_version"],
    )


def _require_row_type(row: dict[str, Any], field: str, expected: type[Any]) -> None:
    value = row[field]
    if type(value) is not expected:
        raise InputError(f"{field} must have runtime type {expected.__name__}")


def _record_for_storage(record: WindowCacheRecord) -> WindowCacheRecord:
    try:
        embedding = tuple(
            struct.unpack("<f", struct.pack("<f", float(value)))[0] for value in record.embedding
        )
    except (OverflowError, struct.error) as exc:
        raise InputError(
            "embedding values must be representable as fp32 storage",
            details={"storage_dtype": _STORAGE_DTYPE},
        ) from exc
    return replace(record, embedding=embedding)


def _arrow_schema(pa: Any, *, embedding_size: int) -> Any:
    return pa.schema(
        [
            pa.field("chrom", pa.string(), nullable=False),
            pa.field("start_bp", pa.int64(), nullable=False),
            pa.field("end_bp", pa.int64(), nullable=False),
            pa.field("window_hash", pa.binary(32), nullable=False),
            pa.field("encoder_hash", pa.binary(32), nullable=False),
            pa.field("state_layer", pa.int8(), nullable=False),
            pa.field("pool_type", pa.string(), nullable=False),
            pa.field("pool_radius", pa.int32(), nullable=False),
            pa.field("center_token", pa.int32(), nullable=True),
            pa.field("dtype", pa.string(), nullable=False),
            pa.field("storage_dtype", pa.string(), nullable=False),
            pa.field(
                "embedding",
                pa.list_(
                    pa.field("element", pa.float32(), nullable=False),
                    embedding_size,
                ),
                nullable=False,
            ),
            pa.field("untargeted", pa.bool_(), nullable=False),
            pa.field("created_at", pa.int64(), nullable=False),
            pa.field("schema_version", pa.string(), nullable=False),
        ]
    )


def _legacy_arrow_schema(pa: Any) -> Any:
    return pa.schema(
        [
            ("chrom", pa.string()),
            ("start_bp", pa.int64()),
            ("end_bp", pa.int64()),
            ("window_hash", pa.binary(32)),
            ("encoder_hash", pa.binary(32)),
            ("state_layer", pa.int8()),
            ("pool_type", pa.string()),
            ("pool_radius", pa.int32()),
            ("center_token", pa.int32()),
            ("dtype", pa.string()),
            ("embedding", pa.list_(pa.float16())),
            ("untargeted", pa.bool_()),
            ("created_at", pa.int64()),
            ("schema_version", pa.string()),
        ]
    )


def _validate_physical_schema(
    pa: Any,
    table: Any,
    *,
    path: Path,
    schema_version: str,
) -> None:
    if schema_version == _LEGACY_CACHE_SCHEMA_VERSION:
        expected = _legacy_arrow_schema(pa)
        embedding_type = table.schema.field("embedding").type
        embedding_matches = pa.types.is_list(embedding_type) and embedding_type.value_type.equals(
            pa.float16()
        )
        schema_matches = (
            table.schema.names == expected.names
            and all(
                table.schema.field(name).type.equals(expected.field(name).type)
                and table.schema.field(name).nullable == expected.field(name).nullable
                for name in expected.names
                if name != "embedding"
            )
            and table.schema.field("embedding").nullable
            and table.schema.metadata == expected.metadata
        )
    else:
        embedding_type = table.schema.field("embedding").type
        if not pa.types.is_fixed_size_list(embedding_type):
            raise CacheCorruptError(
                "cache shard physical schema does not use fixed-width embeddings",
                details={"shard_path": str(path), "embedding_type": str(embedding_type)},
            )
        expected = _arrow_schema(pa, embedding_size=embedding_type.list_size)
        embedding_matches = embedding_type.value_type.equals(pa.float32())
        schema_matches = table.schema.equals(expected, check_metadata=True)
    if not schema_matches or not embedding_matches:
        raise CacheCorruptError(
            "cache shard physical schema does not match its schema_version",
            details={
                "shard_path": str(path),
                "expected": str(expected),
                "observed": str(table.schema),
            },
        )
    required_non_null = tuple(
        field.name for field in expected if field.name != "center_token" and not field.nullable
    )
    null_fields = [name for name in required_non_null if table.column(name).null_count]
    embedding_values = table.column("embedding").to_pylist()
    if null_fields or any(
        values is None or any(value is None for value in values) for values in embedding_values
    ):
        raise CacheCorruptError(
            "cache shard contains null values outside the center_token rule",
            details={"shard_path": str(path), "fields": null_fields},
        )


def _column_names(*, schema_version: str = CACHE_SCHEMA_VERSION) -> tuple[str, ...]:
    common = (
        "chrom",
        "start_bp",
        "end_bp",
        "window_hash",
        "encoder_hash",
        "state_layer",
        "pool_type",
        "pool_radius",
        "center_token",
        "dtype",
    )
    storage = ("storage_dtype",) if schema_version == CACHE_SCHEMA_VERSION else ()
    return (
        *common,
        *storage,
        "embedding",
        "untargeted",
        "created_at",
        "schema_version",
    )


def _assert_existing_shard_equivalent(
    path: Path,
    existing: Sequence[WindowCacheRecord],
    incoming: Sequence[WindowCacheRecord],
) -> None:
    existing_serialized = tuple(_canonical_record_bytes(record) for record in existing)
    incoming_serialized = tuple(_canonical_record_bytes(record) for record in incoming)
    if existing_serialized != incoming_serialized:
        raise CacheCorruptError(
            "existing cache shard must exactly match incoming rows and metadata",
            details={
                "shard_path": str(path),
                "existing_rows": len(existing),
                "incoming_rows": len(incoming),
            },
        )


def _canonical_record_bytes(record: WindowCacheRecord) -> bytes:
    """Serialize one canonical row without Python's lossy equality semantics."""
    payload = bytearray()

    def add_bytes(value: bytes) -> None:
        payload.extend(struct.pack(">Q", len(value)))
        payload.extend(value)

    def add_text(value: str) -> None:
        add_bytes(value.encode("utf-8"))

    add_text(record.chrom)
    payload.extend(struct.pack(">qqq", record.start_bp, record.end_bp, record.state_layer))
    add_bytes(record.window_hash)
    add_bytes(record.encoder_hash)
    add_text(record.pool_type)
    payload.extend(struct.pack(">q", record.pool_radius))
    if record.center_token is None:
        payload.extend(b"\x00")
    else:
        payload.extend(b"\x01")
        payload.extend(struct.pack(">q", record.center_token))
    add_text(record.dtype)
    payload.extend(struct.pack(">Q", len(record.embedding)))
    for value in record.embedding:
        payload.extend(struct.pack(">f", value))
    payload.extend(b"\x01" if record.untargeted else b"\x00")
    payload.extend(struct.pack(">q", record.created_at))
    add_text(record.schema_version)
    return bytes(payload)


def _assert_index_keys_available(
    conn: sqlite3.Connection,
    records: Sequence[WindowCacheRecord],
) -> None:
    for record in records:
        row = conn.execute(
            """
            SELECT shard_path, row_offset
            FROM window_index
            WHERE window_hash = ?
              AND encoder_hash = ?
              AND state_layer = ?
              AND pool_type = ?
              AND pool_radius = ?
              AND center_token = ?
              AND dtype = ?
              AND cache_schema_version = ?
              AND physical_encoding = ?
            """,
            (*_index_key_params(record.key), *_index_provenance_params(record)),
        ).fetchone()
        if row is not None:
            existing_path = _decode_index_shard_path(row[0])
            existing_offset = _decode_index_row_offset(row[1])
            raise CacheCorruptError(
                "cache key is already indexed; refusing duplicate shard write",
                details={
                    "window_hash": record.window_hash.hex(),
                    "existing_shard_path": existing_path,
                    "existing_row_offset": existing_offset,
                },
            )


@contextmanager
def _open_direct_index_database(
    cache_dir: Path,
    *,
    create: bool,
    write: bool,
) -> Iterator[sqlite3.Connection | None]:
    """Open the real index under no-follow checks; never copy it per shard."""
    index_path = _index_path(cache_dir)
    if create and not os.path.lexists(index_path):
        _bootstrap_index_database(cache_dir)
    with _secure_parent_directory(cache_dir, index_path, create=False) as parent_fd:
        if parent_fd is None:
            yield None
            return
        descriptor = _open_regular_at(parent_fd, index_path.name, label="cache index")
        if descriptor is None:
            yield None
            return
        _verify_index_binding(index_path, descriptor, parent_fd=parent_fd)
        for suffix in ("-journal", "-wal", "-shm"):
            sidecar = index_path.with_name(index_path.name + suffix)
            if os.path.lexists(sidecar):
                _assert_safe_namespace_path(cache_dir, sidecar, final_kind="regular file")
        conn = sqlite3.connect(index_path, timeout=30.0)
        try:
            _verify_index_binding(index_path, descriptor, parent_fd=parent_fd)
            conn.execute("PRAGMA cell_size_check = ON")
            if write:
                conn.execute("PRAGMA journal_mode = DELETE")
                conn.execute("PRAGMA synchronous = FULL")
                _ensure_index_schema(conn, create=False)
                conn.commit()
                conn.execute("BEGIN IMMEDIATE")
            else:
                _ensure_index_schema(conn, create=False)
                conn.execute("PRAGMA query_only = ON")
            yield conn
            if write:
                _verify_index_binding(index_path, descriptor, parent_fd=parent_fd)
                conn.commit()
                os.fsync(parent_fd)
                _verify_index_binding(index_path, descriptor, parent_fd=parent_fd)
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()
            os.close(descriptor)


def _bootstrap_index_database(cache_dir: Path) -> None:
    """Create a complete private index and expose it with one atomic replacement."""
    index_path = _index_path(cache_dir)
    if os.path.lexists(index_path):
        return
    with tempfile.TemporaryDirectory(prefix="geno-lewm-index-bootstrap-") as working_dir:
        working_path = Path(working_dir) / INDEX_DB_NAME
        with closing(sqlite3.connect(working_path)) as conn:
            _ensure_index_schema(conn, create=True)
            conn.commit()
            integrity = conn.execute("PRAGMA integrity_check").fetchone()
            if integrity != ("ok",):
                raise CacheCorruptError(
                    "new cache index failed SQLite integrity_check",
                    details={"result": integrity},
                )
        if os.path.lexists(index_path):
            return
        _publish_index_bytes(cache_dir, working_path.read_bytes())


def _verify_index_binding(index_path: Path, descriptor: int, *, parent_fd: int | None) -> None:
    expected = os.fstat(descriptor)
    try:
        if parent_fd is None:
            observed = index_path.lstat()
        else:
            observed = os.stat(index_path.name, dir_fd=parent_fd, follow_symlinks=False)
            parent = index_path.parent.lstat()
            held_parent = os.fstat(parent_fd)
            if (parent.st_dev, parent.st_ino) != (held_parent.st_dev, held_parent.st_ino):
                raise CacheCorruptError("cache index parent changed during transaction")
    except OSError as exc:
        raise CacheCorruptError(
            "cache index binding changed during transaction",
            details={"path": str(index_path), "error": str(exc)},
        ) from exc
    if stat.S_ISLNK(observed.st_mode) or not stat.S_ISREG(observed.st_mode):
        raise CacheCorruptError("cache index binding is not a regular non-symlink file")
    if (expected.st_dev, expected.st_ino) != (observed.st_dev, observed.st_ino):
        raise CacheCorruptError("cache index binding changed during transaction")


def _publish_index_bytes(cache_dir: Path, body: bytes) -> None:
    index_path = _index_path(cache_dir)
    with _secure_parent_directory(cache_dir, index_path, create=True) as parent_fd:
        if parent_fd is None:
            raise CacheCorruptError("cache index parent disappeared during secure publication")
        existing = _open_regular_at(parent_fd, index_path.name, label="cache index")
        if existing is not None:
            os.close(existing)
        temp_name = f".{index_path.name}.{secrets.token_hex(16)}.tmp"
        flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW
        descriptor = os.open(temp_name, flags, 0o600, dir_fd=parent_fd)
        try:
            with os.fdopen(os.dup(descriptor), "wb") as handle:
                handle.write(body)
                handle.flush()
            os.fsync(descriptor)
            os.rename(
                temp_name,
                index_path.name,
                src_dir_fd=parent_fd,
                dst_dir_fd=parent_fd,
            )
            os.fsync(parent_fd)
        finally:
            os.close(descriptor)
            with suppress(FileNotFoundError):
                os.unlink(temp_name, dir_fd=parent_fd)


def _insert_index_records(
    conn: sqlite3.Connection,
    cache_dir: Path,
    shard: Path,
    records: Sequence[WindowCacheRecord],
) -> None:
    rel_shard = shard.relative_to(cache_dir).as_posix()
    for row_offset, record in enumerate(records):
        existing = conn.execute(
            """
            SELECT shard_path, row_offset
            FROM window_index
            WHERE window_hash = ?
              AND encoder_hash = ?
              AND state_layer = ?
              AND pool_type = ?
              AND pool_radius = ?
              AND center_token = ?
              AND dtype = ?
              AND cache_schema_version = ?
              AND physical_encoding = ?
            """,
            (*_index_key_params(record.key), *_index_provenance_params(record)),
        ).fetchone()
        if existing is not None:
            existing_path = _decode_index_shard_path(existing[0])
            existing_offset = _decode_index_row_offset(existing[1])
            if existing_path == rel_shard and existing_offset == row_offset:
                continue
            raise CacheCorruptError(
                "cache index already contains this key",
                details={
                    "window_hash": record.window_hash.hex(),
                    "existing_shard_path": existing_path,
                    "new_shard_path": rel_shard,
                },
            )
        conn.execute(
            """
            INSERT INTO window_index (
                window_hash,
                encoder_hash,
                state_layer,
                pool_type,
                pool_radius,
                center_token,
                dtype,
                cache_schema_version,
                physical_encoding,
                shard_path,
                row_offset,
                created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                record.window_hash.hex(),
                record.encoder_hash.hex(),
                record.state_layer,
                record.pool_type,
                record.pool_radius,
                _index_center_token(record.center_token),
                record.dtype,
                record.schema_version,
                _physical_encoding(record.schema_version),
                rel_shard,
                row_offset,
                record.created_at,
            ),
        )


def _ensure_index_schema(conn: sqlite3.Connection, *, create: bool) -> None:
    _require_strict_sqlite()
    objects = _index_schema_objects(conn)
    if not objects:
        if not create:
            raise CacheCorruptError(
                "cache index is empty or incomplete; run reindex_cache before corrected lookup"
            )
        conn.execute(_WINDOW_INDEX_SQL)
        conn.execute(_SHARD_PATH_INDEX_SQL)
        conn.execute(f"PRAGMA user_version = {_INDEX_SCHEMA_VERSION}")
        objects = _index_schema_objects(conn)
    observed = conn.execute("PRAGMA table_info(window_index)").fetchall()
    user_version = conn.execute("PRAGMA user_version").fetchone()
    table_info = conn.execute(
        "SELECT strict FROM pragma_table_list WHERE name = 'window_index'"
    ).fetchone()
    expected_objects = {
        ("table", "window_index"): _normalize_sql(_WINDOW_INDEX_SQL),
        ("index", "idx_shard_path"): _normalize_sql(_SHARD_PATH_INDEX_SQL),
    }
    if (
        objects != expected_objects
        or not _index_schema_matches(observed)
        or user_version != (_INDEX_SCHEMA_VERSION,)
        or table_info != (1,)
    ):
        raise CacheCorruptError(
            "cache index schema is stale or unsafe; run reindex_cache for explicit migration"
        )


def _require_strict_sqlite() -> None:
    if sqlite3.sqlite_version_info < (3, 37, 0):
        raise RuntimeSetupError(
            "cache schema 3 requires SQLite 3.37 or newer for STRICT index tables",
            details={"sqlite_version": sqlite3.sqlite_version},
            remediation="use a Python runtime linked against SQLite 3.37 or newer",
        )


def _index_schema_objects(conn: sqlite3.Connection) -> dict[tuple[str, str], str]:
    rows = conn.execute(
        """
        SELECT type, name, sql
        FROM sqlite_schema
        WHERE name NOT LIKE 'sqlite_%'
        ORDER BY type, name
        """
    ).fetchall()
    objects: dict[tuple[str, str], str] = {}
    for object_type, name, sql in rows:
        if type(object_type) is not str or type(name) is not str or type(sql) is not str:
            raise CacheCorruptError("cache index schema contains an invalid object")
        objects[(object_type, name)] = _normalize_sql(sql)
    return objects


def _normalize_sql(sql: str) -> str:
    return " ".join(sql.split())


def _index_schema_matches(columns: Sequence[tuple[Any, ...]]) -> bool:
    expected = (
        ("window_hash", "ANY", 1, 1),
        ("encoder_hash", "ANY", 1, 2),
        ("state_layer", "ANY", 1, 3),
        ("pool_type", "ANY", 1, 4),
        ("pool_radius", "ANY", 1, 5),
        ("center_token", "ANY", 1, 6),
        ("dtype", "ANY", 1, 7),
        ("cache_schema_version", "ANY", 1, 8),
        ("physical_encoding", "ANY", 1, 9),
        ("shard_path", "ANY", 1, 0),
        ("row_offset", "ANY", 1, 0),
        ("created_at", "ANY", 1, 0),
    )
    observed = tuple(
        (str(column[1]), str(column[2]).upper(), int(column[3]), int(column[5]))
        for column in columns
    )
    return observed == expected


def _index_key_params(key: WindowCacheKey) -> tuple[Any, ...]:
    return (
        key.window_hash.hex(),
        key.encoder_hash.hex(),
        key.state_layer,
        key.pool_type,
        key.pool_radius,
        _index_center_token(key.center_token),
        key.dtype,
    )


def _index_provenance_params(record: WindowCacheRecord) -> tuple[str, str]:
    return record.schema_version, _physical_encoding(record.schema_version)


def _physical_encoding(schema_version: str) -> str:
    if schema_version == CACHE_SCHEMA_VERSION:
        return _V3_PHYSICAL_ENCODING
    if schema_version == _LEGACY_CACHE_SCHEMA_VERSION:
        return _V2_PHYSICAL_ENCODING
    raise CacheCorruptError(
        "cache row has unsupported physical provenance",
        details={"schema_version": schema_version},
    )


def _validate_read_policy(policy: str) -> None:
    if policy not in {"require_v3", "prefer_v3", "legacy_v2_only"}:
        raise InputError(
            "unsupported cache read policy",
            details={
                "policy": policy,
                "supported": ["require_v3", "prefer_v3", "legacy_v2_only"],
            },
        )


def _select_index_location(
    rows: Sequence[tuple[Any, ...]],
    *,
    policy: CacheReadPolicy,
    key: WindowCacheKey,
) -> tuple[str, str, str, int] | None:
    decoded = tuple(_decode_index_location(row) for row in rows)
    by_schema = {row[0]: row for row in decoded}
    if policy == "require_v3":
        selected = by_schema.get(CACHE_SCHEMA_VERSION)
        if selected is None and _LEGACY_CACHE_SCHEMA_VERSION in by_schema:
            raise CacheCorruptError(
                "corrected cache lookup requires cache schema 3; legacy v2 is replay-only",
                details={"window_hash": key.window_hash.hex(), "policy": policy},
            )
        return selected
    if policy == "legacy_v2_only":
        return by_schema.get(_LEGACY_CACHE_SCHEMA_VERSION)
    return by_schema.get(CACHE_SCHEMA_VERSION) or by_schema.get(_LEGACY_CACHE_SCHEMA_VERSION)


def _decode_index_location(row: Sequence[Any]) -> tuple[str, str, str, int]:
    if len(row) != 4:
        raise CacheCorruptError("cache index location has an invalid field count")
    schema_version, physical_encoding, shard_path, row_offset = row
    if type(schema_version) is not str or schema_version not in {
        CACHE_SCHEMA_VERSION,
        _LEGACY_CACHE_SCHEMA_VERSION,
    }:
        raise CacheCorruptError("cache index schema provenance is invalid")
    if type(physical_encoding) is not str or physical_encoding != _physical_encoding(
        schema_version
    ):
        raise CacheCorruptError("cache index physical encoding provenance is invalid")
    shard_path = _decode_index_shard_path(shard_path)
    row_offset = _decode_index_row_offset(row_offset)
    return schema_version, physical_encoding, shard_path, row_offset


def _decode_index_row_offset(row_offset: Any) -> int:
    if type(row_offset) is not int or row_offset < 0 or row_offset > 2**63 - 1:
        raise CacheCorruptError(
            "cache index row_offset must be a non-negative 64-bit integer",
            details={"row_offset": repr(row_offset)},
        )
    return row_offset


def _decode_index_shard_path(shard_path: Any) -> str:
    if type(shard_path) is not str or not shard_path:
        raise CacheCorruptError("cache index shard_path must be non-empty text")
    return shard_path


def _index_center_token(center_token: int | None) -> int:
    return -1 if center_token is None else center_token


def _iter_shards(cache_dir: Path) -> Iterable[Path]:
    embeddings = cache_dir / _EMBEDDINGS_DIR
    if not embeddings.exists():
        return ()
    return (
        path
        for path in sorted(embeddings.rglob("*.parquet"))
        if _QUARANTINE_DIR not in path.relative_to(embeddings).parts
    )


def _quarantine_shard(cache_dir: Path, shard: Path) -> Path:
    embeddings = cache_dir / _EMBEDDINGS_DIR
    rel = shard.relative_to(embeddings)
    destination = embeddings / _QUARANTINE_DIR / rel
    _assert_safe_namespace_path(cache_dir, shard, final_kind="regular file")
    _assert_safe_namespace_path(cache_dir, destination, final_kind="regular file")
    with (
        _secure_parent_directory(cache_dir, shard, create=False) as source_parent,
        _secure_parent_directory(cache_dir, destination, create=True) as destination_parent,
    ):
        if source_parent is None or destination_parent is None:
            raise CacheCorruptError("cache shard parent disappeared during quarantine")
        source_fd = _open_regular_at(source_parent, shard.name, label="cache shard")
        if source_fd is None:
            raise CacheCorruptError("cache shard disappeared during quarantine")
        os.close(source_fd)
        candidate = destination.name
        existing = _open_regular_at(
            destination_parent, candidate, label="cache quarantine destination"
        )
        while existing is not None:
            os.close(existing)
            candidate = f"{destination.name}.{time.time_ns()}.bad"
            existing = _open_regular_at(
                destination_parent,
                candidate,
                label="cache quarantine destination",
            )
        try:
            os.link(
                shard.name,
                candidate,
                src_dir_fd=source_parent,
                dst_dir_fd=destination_parent,
                follow_symlinks=False,
            )
        except FileExistsError:
            candidate = f"{destination.name}.{time.time_ns()}.bad"
            os.link(
                shard.name,
                candidate,
                src_dir_fd=source_parent,
                dst_dir_fd=destination_parent,
                follow_symlinks=False,
            )
        os.fsync(destination_parent)
        os.unlink(shard.name, dir_fd=source_parent)
        os.fsync(source_parent)
        return destination.with_name(candidate)


def _index_path(cache_dir: Path) -> Path:
    return cache_dir / _EMBEDDINGS_DIR / INDEX_DB_NAME


def _indexed_shard_path(cache_dir: Path, relative_path: str) -> Path:
    candidate = Path(relative_path)
    root = cache_dir.absolute()
    if candidate.is_absolute():
        raise CacheCorruptError(
            "cache index shard path points outside cache root",
            details={"shard_path": relative_path},
        )
    lexical = root / candidate
    _assert_safe_namespace_path(root, lexical, final_kind="regular file")
    resolved_root = root.resolve()
    resolved = lexical.resolve()
    if not resolved.is_relative_to(resolved_root):
        raise CacheCorruptError(
            "cache index shard path points outside cache root",
            details={"shard_path": relative_path},
        )
    return lexical


def _assert_safe_namespace_path(
    cache_dir: Path,
    target: Path,
    *,
    final_kind: str,
) -> None:
    """Reject every existing symlink or non-directory namespace ancestor."""
    root = cache_dir.absolute()
    candidate = target.absolute()
    try:
        relative = candidate.relative_to(root)
    except ValueError as exc:
        raise CacheCorruptError(
            "cache namespace path points outside cache root",
            details={"path": str(target)},
        ) from exc
    current = root
    paths = (
        root,
        *(root.joinpath(*relative.parts[:index]) for index in range(1, len(relative.parts) + 1)),
    )
    for index, current in enumerate(paths):
        if not os.path.lexists(current):
            continue
        mode = current.lstat().st_mode
        is_final = index == len(paths) - 1
        if stat.S_ISLNK(mode):
            raise CacheCorruptError(
                "cache namespace contains an unsafe symlink",
                details={"path": str(current)},
            )
        if is_final:
            if final_kind == "regular file" and not stat.S_ISREG(mode):
                raise CacheCorruptError(
                    "cache namespace final must be a regular file",
                    details={"path": str(current)},
                )
        elif not stat.S_ISDIR(mode):
            raise CacheCorruptError(
                "cache namespace parent must be a real directory",
                details={"path": str(current)},
            )


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _digest_path_part(label: str, value: str) -> str:
    """Return a fixed ASCII component without filesystem normalization aliases."""
    if type(value) is not str or not value:
        raise InputError(
            "cache path identity must be non-empty text",
            details={"label": label, "type": type(value).__name__},
        )
    return f"{label}-{sha256(value.encode('utf-8')).hexdigest()}"


def _legacy_path_part(value: str) -> str:
    """Preserve the published schema-2 path escaping contract byte-for-byte."""
    if type(value) is not str or not value or value in {".", ".."}:
        raise InputError("cache path component must be non-empty text and not a dot segment")
    return value.replace("/", "__").replace("\\", "__")


def _validate_hash(name: str, value: bytes) -> None:
    observed_len = len(value) if isinstance(value, bytes) else None
    if not isinstance(value, bytes) or observed_len != 32:
        raise InputError(
            f"{name} must be 32 bytes",
            details={"field": name, "type": type(value).__name__, "len": observed_len},
        )


def _hash_path_part(name: str, value: bytes) -> str:
    _validate_hash(name, value)
    return value.hex()


def _validate_state_layer(state_layer: int) -> None:
    if type(state_layer) is not int or state_layer < -128 or state_layer > 127:
        raise InputError(
            "state_layer must be an int8-compatible integer",
            details={"state_layer": state_layer, "type": type(state_layer).__name__},
        )


def _validate_pool(pool_type: str, pool_radius: int) -> None:
    if type(pool_type) is not str or pool_type not in _SUPPORTED_POOL_TYPES:
        raise InputError(
            "unsupported pool_type",
            details={"pool_type": pool_type, "supported": sorted(_SUPPORTED_POOL_TYPES)},
        )
    if type(pool_radius) is not int or pool_radius < 0 or pool_radius > 2**31 - 1:
        raise InputError(
            "pool_radius must be a non-negative int32-compatible integer",
            details={"pool_radius": pool_radius, "type": type(pool_radius).__name__},
        )
    if pool_type == POOL_GLOBAL_MEAN and pool_radius != 0:
        raise InputError(
            "global_mean cache keys require pool_radius=0",
            details={"pool_type": pool_type, "pool_radius": pool_radius},
        )


def _validate_pool_locus(pool_type: str, center_token: int | None) -> None:
    if pool_type == POOL_GLOBAL_MEAN:
        if center_token is not None:
            raise InputError(
                "center_token must be absent for global pooling",
                details={"pool_type": pool_type, "center_token": center_token},
            )
        return
    if type(center_token) is not int or center_token < 0 or center_token > 2**31 - 1:
        raise InputError(
            "center_token must be a non-negative int32-compatible integer for locus-aware pooling",
            details={
                "pool_type": pool_type,
                "center_token": center_token,
                "type": type(center_token).__name__,
            },
        )


def _validate_dtype(dtype: str) -> None:
    if type(dtype) is not str or dtype not in _SUPPORTED_DTYPES:
        raise InputError(
            "unsupported dtype",
            details={"dtype": dtype, "supported": sorted(_SUPPORTED_DTYPES)},
        )
