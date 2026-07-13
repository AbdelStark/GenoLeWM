# SPDX-License-Identifier: Apache-2.0
"""Parquet raw pooled window-embedding cache and SQLite index.

New writes use the collision-safe schema-3 contract; schema-2 shards remain
read-only compatible for lookup, repair, and reindexing.
Parquet support is intentionally imported lazily so the base package
keeps its minimal dependency surface; install ``geno-lewm[train]`` or
the development extra to use this module.
"""

from __future__ import annotations

import math
import os
import sqlite3
import struct
import tempfile
import time
from collections import defaultdict
from collections.abc import Iterable, Sequence
from contextlib import closing
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

from geno_lewm.encoder.pooling import POOL_CENTERED_MEAN, POOL_GLOBAL_MEAN
from geno_lewm.errors import CacheCorruptError, InputError, RuntimeSetupError

__all__ = [
    "CACHE_SCHEMA_VERSION",
    "INDEX_DB_NAME",
    "CacheReindexReport",
    "CacheRepairReport",
    "WindowCacheKey",
    "WindowCacheRecord",
    "default_cache_dir",
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
_INDEX_SCHEMA_VERSION = 2
_EMBEDDINGS_DIR = "embeddings"
_QUARANTINE_DIR = ".quarantine"
_STORAGE_DTYPE = "fp32"
_ROW_GROUP_SIZE = 1_024
_SUPPORTED_DTYPES = frozenset({"bf16", "fp16", "fp32"})
_SUPPORTED_POOL_TYPES = frozenset({POOL_CENTERED_MEAN, POOL_GLOBAL_MEAN, "attention"})


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
        if not self.chrom:
            raise InputError("chrom must be non-empty")
        if self.end_bp <= self.start_bp:
            raise InputError(
                "end_bp must be greater than start_bp",
                details={"start_bp": self.start_bp, "end_bp": self.end_bp},
            )
        if not self.embedding:
            raise InputError("embedding must contain at least one value")
        for idx, value in enumerate(self.embedding):
            if isinstance(value, bool) or not isinstance(value, int | float):
                raise InputError(
                    "embedding values must be numeric",
                    details={"index": idx, "value": repr(value)},
                )
            if not math.isfinite(value):
                raise InputError(
                    "embedding values must be finite",
                    details={"index": idx, "value": repr(value)},
                )
        if self.schema_version not in {CACHE_SCHEMA_VERSION, _LEGACY_CACHE_SCHEMA_VERSION}:
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
    encoder_part = _path_part(encoder_id)
    if (encoder_hash is None) != (dtype is None):
        raise InputError("encoder_hash and dtype must be supplied together")
    if encoder_hash is not None and dtype is not None:
        _validate_hash("encoder_hash", encoder_hash)
        _validate_dtype(dtype)
        return (
            root
            / _EMBEDDINGS_DIR
            / "v3"
            / encoder_part
            / encoder_hash.hex()
            / f"{dtype}_as_{_STORAGE_DTYPE}"
            / str(state_layer)
            / f"{pool_type}_{pool_radius}"
            / f"chr{_path_part(contig)}_{stride_block}.parquet"
        )
    return (
        root
        / _EMBEDDINGS_DIR
        / encoder_part
        / str(state_layer)
        / f"{pool_type}_{pool_radius}"
        / f"chr{_path_part(contig)}_{stride_block}.parquet"
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
    if path.exists():
        existing = _read_records_from_shard(path)
        comparable = normalized
        if len(existing) == len(normalized):
            comparable = tuple(
                replace(incoming, created_at=prior.created_at)
                if requested[index].created_at == 0
                else incoming
                for index, (prior, incoming) in enumerate(zip(existing, normalized, strict=True))
            )
        _assert_existing_shard_equivalent(path, existing, comparable)
        _index_records(root, path, existing)
        return path

    _assert_index_keys_available(root, normalized)
    path.parent.mkdir(parents=True, exist_ok=True)
    file_descriptor, temp_name = tempfile.mkstemp(
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
    )
    os.close(file_descriptor)
    temp_path = Path(temp_name)
    try:
        _write_records_to_parquet(temp_path, normalized)
        _fsync_file(temp_path)
        staged = _read_records_from_shard(temp_path)
        _assert_existing_shard_equivalent(temp_path, staged, normalized)
        temp_path.replace(path)
        _fsync_directory(path.parent)
    finally:
        temp_path.unlink(missing_ok=True)
    _index_records(root, path, normalized)
    return path


def read_embedding(cache_dir: Path | str, key: WindowCacheKey) -> tuple[float, ...] | None:
    """Return a raw pooled embedding by content key, or ``None`` on cache miss."""
    return read_embeddings(cache_dir, (key,))[0]


def read_embeddings(
    cache_dir: Path | str,
    keys: Sequence[WindowCacheKey],
) -> tuple[tuple[float, ...] | None, ...]:
    """Return raw embeddings for ``keys`` in order, grouping reads by shard.

    Duplicate keys and misses are preserved in the returned tuple. Only the
    Parquet row groups containing requested rows are read.
    """
    if not keys:
        return ()
    root = Path(cache_dir)
    index_path = _index_path(root)
    if not index_path.exists():
        return tuple(None for _key in keys)
    locations: dict[WindowCacheKey, tuple[Path, int]] = {}
    indexed_paths: dict[str, Path] = {}
    with closing(sqlite3.connect(index_path)) as conn:
        _ensure_index_schema(conn)
        for key in dict.fromkeys(keys):
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
                """,
                _index_key_params(key),
            ).fetchone()
            if row is not None:
                relative_path = str(row[0])
                shard_path = indexed_paths.get(relative_path)
                if shard_path is None:
                    shard_path = _indexed_shard_path(root, relative_path)
                    indexed_paths[relative_path] = shard_path
                locations[key] = (shard_path, int(row[1]))
        conn.commit()
    requests_by_shard: dict[Path, list[tuple[int, WindowCacheKey, int]]] = defaultdict(list)
    for result_index, key in enumerate(keys):
        location = locations.get(key)
        if location is not None:
            shard_path, row_offset = location
            requests_by_shard[shard_path].append((result_index, key, row_offset))
    results: list[tuple[float, ...] | None] = [None] * len(keys)
    for shard_path, requests in requests_by_shard.items():
        requested_offsets = {request[2] for request in requests}
        records = _read_records_at_offsets(shard_path, requested_offsets)
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
            results[result_index] = record.embedding
    return tuple(results)


def reindex_cache(cache_dir: Path | str) -> CacheReindexReport:
    """Rebuild ``index.sqlite`` from every readable Parquet shard."""
    root = Path(cache_dir)
    index_path = _index_path(root)
    index_path.parent.mkdir(parents=True, exist_ok=True)
    indexed_shards = 0
    indexed_rows = 0
    file_descriptor, temp_name = tempfile.mkstemp(
        dir=index_path.parent,
        prefix=f".{index_path.name}.",
        suffix=".tmp",
    )
    os.close(file_descriptor)
    temp_path = Path(temp_name)
    try:
        with closing(sqlite3.connect(temp_path)) as conn:
            _ensure_index_schema(conn)
            for shard in _iter_shards(root):
                records = _read_records_from_shard(shard)
                _insert_index_records(conn, root, shard, records)
                indexed_shards += 1
                indexed_rows += len(records)
            conn.commit()
            integrity = conn.execute("PRAGMA integrity_check").fetchone()
            if integrity is None or str(integrity[0]) != "ok":
                raise CacheCorruptError(
                    "rebuilt cache index failed SQLite integrity_check",
                    details={"index_path": str(index_path), "result": integrity},
                )
        _fsync_file(temp_path)
        temp_path.replace(index_path)
        _fsync_directory(index_path.parent)
    finally:
        temp_path.unlink(missing_ok=True)
    return CacheReindexReport(
        indexed_shards=indexed_shards,
        indexed_rows=indexed_rows,
        index_path=index_path,
    )


def repair_cache(cache_dir: Path | str) -> CacheRepairReport:
    """Quarantine unreadable Parquet shards and rebuild the SQLite index."""
    root = Path(cache_dir)
    quarantined: list[Path] = []
    checked = 0
    for shard in list(_iter_shards(root)):
        checked += 1
        try:
            _read_records_from_shard(shard)
        except CacheCorruptError:
            quarantined.append(_quarantine_shard(root, shard))
    report = reindex_cache(root)
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


def _write_records_to_parquet(path: Path, records: Sequence[WindowCacheRecord]) -> None:
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
    pa, pq = _require_pyarrow()
    try:
        table = pq.read_table(path)
    except Exception as exc:
        raise CacheCorruptError(
            "cache shard could not be read",
            details={"shard_path": str(path), "error": str(exc)},
        ) from exc
    schema_version = _schema_version_from_table(table, path=path)
    _validate_physical_schema(pa, table, path=path, schema_version=schema_version)
    try:
        return tuple(_record_from_row(row) for row in table.to_pylist())
    except (InputError, TypeError, ValueError) as exc:
        raise CacheCorruptError(
            "cache shard contains an invalid row",
            details={"shard_path": str(path), "error": str(exc)},
        ) from exc


def _read_records_at_offsets(
    path: Path,
    row_offsets: set[int],
) -> dict[int, WindowCacheRecord]:
    pa, pq = _require_pyarrow()
    try:
        parquet = pq.ParquetFile(path)
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
    schema_versions = set(table.column("schema_version").to_pylist())
    if len(schema_versions) != 1:
        raise CacheCorruptError(
            "cache shard must contain one schema_version",
            details={
                "shard_path": str(path),
                "schema_versions": sorted(repr(value) for value in schema_versions),
            },
        )
    schema_version = str(next(iter(schema_versions)))
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
        storage_dtypes = set(table.column("storage_dtype").to_pylist())
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
    return WindowCacheRecord(
        chrom=str(row["chrom"]),
        start_bp=int(row["start_bp"]),
        end_bp=int(row["end_bp"]),
        window_hash=bytes(row["window_hash"]),
        encoder_hash=bytes(row["encoder_hash"]),
        state_layer=int(row["state_layer"]),
        pool_type=str(row["pool_type"]),
        pool_radius=int(row["pool_radius"]),
        center_token=(None if row["center_token"] is None else int(row["center_token"])),
        dtype=str(row["dtype"]),
        embedding=tuple(float(value) for value in row["embedding"]),
        untargeted=bool(row["untargeted"]),
        created_at=int(row["created_at"]),
        schema_version=str(row["schema_version"]),
    )


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
            ("storage_dtype", pa.string()),
            ("embedding", pa.list_(pa.float32(), embedding_size)),
            ("untargeted", pa.bool_()),
            ("created_at", pa.int64()),
            ("schema_version", pa.string()),
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
    else:
        embedding_type = table.schema.field("embedding").type
        if not pa.types.is_fixed_size_list(embedding_type):
            raise CacheCorruptError(
                "cache shard physical schema does not use fixed-width embeddings",
                details={"shard_path": str(path), "embedding_type": str(embedding_type)},
            )
        expected = _arrow_schema(pa, embedding_size=embedding_type.list_size)
        embedding_matches = embedding_type.value_type.equals(pa.float32())
    non_embedding_matches = all(
        table.schema.field(name).type.equals(expected.field(name).type)
        for name in expected.names
        if name != "embedding" and name in table.schema.names
    )
    if (
        table.schema.names != expected.names
        or not embedding_matches
        or not non_embedding_matches
        or table.schema.metadata != expected.metadata
    ):
        raise CacheCorruptError(
            "cache shard physical schema does not match its schema_version",
            details={
                "shard_path": str(path),
                "expected": str(expected),
                "observed": str(table.schema),
            },
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
    if tuple(existing) != tuple(incoming):
        raise CacheCorruptError(
            "existing cache shard must exactly match incoming rows and metadata",
            details={
                "shard_path": str(path),
                "existing_rows": len(existing),
                "incoming_rows": len(incoming),
            },
        )


def _assert_index_keys_available(cache_dir: Path, records: Sequence[WindowCacheRecord]) -> None:
    index_path = _index_path(cache_dir)
    if not index_path.exists():
        return
    with closing(sqlite3.connect(index_path)) as conn:
        _ensure_index_schema(conn)
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
                """,
                _index_key_params(record.key),
            ).fetchone()
            if row is not None:
                raise CacheCorruptError(
                    "cache key is already indexed; refusing duplicate shard write",
                    details={
                        "window_hash": record.window_hash.hex(),
                        "existing_shard_path": str(row[0]),
                        "existing_row_offset": int(row[1]),
                    },
                )
        conn.commit()


def _index_records(
    cache_dir: Path,
    shard: Path,
    records: Sequence[WindowCacheRecord],
) -> None:
    index_path = _index_path(cache_dir)
    index_path.parent.mkdir(parents=True, exist_ok=True)
    with closing(sqlite3.connect(index_path)) as conn:
        _ensure_index_schema(conn)
        _insert_index_records(conn, cache_dir, shard, records)
        conn.commit()


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
            """,
            _index_key_params(record.key),
        ).fetchone()
        if existing is not None:
            if str(existing[0]) == rel_shard and int(existing[1]) == row_offset:
                continue
            raise CacheCorruptError(
                "cache index already contains this key",
                details={
                    "window_hash": record.window_hash.hex(),
                    "existing_shard_path": str(existing[0]),
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
                shard_path,
                row_offset,
                created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                record.window_hash.hex(),
                record.encoder_hash.hex(),
                record.state_layer,
                record.pool_type,
                record.pool_radius,
                _index_center_token(record.center_token),
                record.dtype,
                rel_shard,
                row_offset,
                record.created_at,
            ),
        )


def _ensure_index_schema(conn: sqlite3.Connection) -> None:
    observed = conn.execute("PRAGMA table_info(window_index)").fetchall()
    if observed and not _index_schema_matches(observed):
        conn.execute("DROP INDEX IF EXISTS idx_shard_path")
        conn.execute("DROP TABLE window_index")
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS window_index (
            window_hash TEXT NOT NULL,
            encoder_hash TEXT NOT NULL,
            state_layer INTEGER NOT NULL,
            pool_type TEXT NOT NULL,
            pool_radius INTEGER NOT NULL,
            center_token INTEGER NOT NULL,
            dtype TEXT NOT NULL,
            shard_path TEXT NOT NULL,
            row_offset INTEGER NOT NULL,
            created_at INTEGER NOT NULL,
            PRIMARY KEY (
                window_hash,
                encoder_hash,
                state_layer,
                pool_type,
                pool_radius,
                center_token,
                dtype
            )
        )
        """
    )
    conn.execute("CREATE INDEX IF NOT EXISTS idx_shard_path ON window_index(shard_path)")
    conn.execute(f"PRAGMA user_version = {_INDEX_SCHEMA_VERSION}")


def _index_schema_matches(columns: Sequence[tuple[Any, ...]]) -> bool:
    expected = (
        ("window_hash", 1, 1),
        ("encoder_hash", 1, 2),
        ("state_layer", 1, 3),
        ("pool_type", 1, 4),
        ("pool_radius", 1, 5),
        ("center_token", 1, 6),
        ("dtype", 1, 7),
        ("shard_path", 1, 0),
        ("row_offset", 1, 0),
        ("created_at", 1, 0),
    )
    observed = tuple((str(column[1]), int(column[3]), int(column[5])) for column in columns)
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
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        destination = destination.with_name(f"{destination.name}.{time.time_ns()}.bad")
    shard.replace(destination)
    return destination


def _index_path(cache_dir: Path) -> Path:
    return cache_dir / _EMBEDDINGS_DIR / INDEX_DB_NAME


def _indexed_shard_path(cache_dir: Path, relative_path: str) -> Path:
    candidate = Path(relative_path)
    root = cache_dir.resolve()
    if candidate.is_absolute():
        raise CacheCorruptError(
            "cache index shard path points outside cache root",
            details={"shard_path": relative_path},
        )
    resolved = (root / candidate).resolve()
    if not resolved.is_relative_to(root):
        raise CacheCorruptError(
            "cache index shard path points outside cache root",
            details={"shard_path": relative_path},
        )
    return resolved


def _fsync_file(path: Path) -> None:
    with path.open("rb") as handle:
        os.fsync(handle.fileno())


def _fsync_directory(path: Path) -> None:
    if os.name == "nt":  # Directory handles cannot be fsynced this way on Windows.
        return
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _path_part(value: str) -> str:
    if not value or value in {".", ".."}:
        raise InputError("cache path component must be non-empty and not a dot segment")
    return value.replace("/", "__").replace("\\", "__")


def _validate_hash(name: str, value: bytes) -> None:
    observed_len = len(value) if isinstance(value, bytes) else None
    if not isinstance(value, bytes) or observed_len != 32:
        raise InputError(
            f"{name} must be 32 bytes",
            details={"field": name, "type": type(value).__name__, "len": observed_len},
        )


def _validate_state_layer(state_layer: int) -> None:
    if not isinstance(state_layer, int) or isinstance(state_layer, bool):
        raise InputError(
            "state_layer must be an integer",
            details={"state_layer": state_layer, "type": type(state_layer).__name__},
        )


def _validate_pool(pool_type: str, pool_radius: int) -> None:
    if pool_type not in _SUPPORTED_POOL_TYPES:
        raise InputError(
            "unsupported pool_type",
            details={"pool_type": pool_type, "supported": sorted(_SUPPORTED_POOL_TYPES)},
        )
    if not isinstance(pool_radius, int) or isinstance(pool_radius, bool) or pool_radius < 0:
        raise InputError(
            "pool_radius must be a non-negative integer",
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
    if not isinstance(center_token, int) or isinstance(center_token, bool) or center_token < 0:
        raise InputError(
            "center_token must be a non-negative integer for locus-aware pooling",
            details={
                "pool_type": pool_type,
                "center_token": center_token,
                "type": type(center_token).__name__,
            },
        )


def _validate_dtype(dtype: str) -> None:
    if dtype not in _SUPPORTED_DTYPES:
        raise InputError(
            "unsupported dtype",
            details={"dtype": dtype, "supported": sorted(_SUPPORTED_DTYPES)},
        )
