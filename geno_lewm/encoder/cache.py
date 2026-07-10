# SPDX-License-Identifier: Apache-2.0
"""Parquet raw pooled window-embedding cache and SQLite index.

Implements the on-disk cache contract from encoder contract and
``public API contract``.
Parquet support is intentionally imported lazily so the base package
keeps its minimal dependency surface; install ``geno-lewm[train]`` or
the development extra to use this module.
"""

from __future__ import annotations

import os
import sqlite3
import time
from collections.abc import Iterable, Sequence
from contextlib import closing
from dataclasses import dataclass
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
    "reindex_cache",
    "repair_cache",
    "shard_path_for",
    "write_shard",
]


CACHE_SCHEMA_VERSION = "2.0.0"
INDEX_DB_NAME = "index.sqlite"
_INDEX_SCHEMA_VERSION = 2
_EMBEDDINGS_DIR = "embeddings"
_QUARANTINE_DIR = ".quarantine"
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
            if not isinstance(value, int | float):
                raise InputError(
                    "embedding values must be numeric",
                    details={"index": idx, "value": repr(value)},
                )
        if self.schema_version != CACHE_SCHEMA_VERSION:
            raise InputError(
                "unsupported cache schema_version",
                details={"schema_version": self.schema_version, "supported": CACHE_SCHEMA_VERSION},
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
) -> Path:
    """Return the canonical Parquet shard path for a cache block."""
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
    normalized = tuple(record.with_created_at() for record in records)
    first = normalized[0]
    if any(record.chrom != contig for record in normalized):
        raise InputError("all records in a shard must match the contig argument")
    if any(record.state_layer != first.state_layer for record in normalized):
        raise InputError("all records in a shard must share state_layer")
    if any(record.pool_type != first.pool_type for record in normalized):
        raise InputError("all records in a shard must share pool_type")
    if any(record.pool_radius != first.pool_radius for record in normalized):
        raise InputError("all records in a shard must share pool_radius")

    root = Path(cache_dir)
    path = shard_path_for(
        root,
        encoder_id=encoder_id,
        state_layer=first.state_layer,
        pool_type=first.pool_type,
        pool_radius=first.pool_radius,
        contig=contig,
        stride_block=stride_block,
    )
    if path.exists():
        existing = _read_records_from_shard(path)
        _assert_existing_shard_equivalent(path, existing, normalized)
        _index_records(root, path, existing)
        return path

    _assert_index_keys_available(root, normalized)
    path.parent.mkdir(parents=True, exist_ok=True)
    _write_records_to_parquet(path, normalized)
    _index_records(root, path, normalized)
    return path


def read_embedding(cache_dir: Path | str, key: WindowCacheKey) -> tuple[float, ...] | None:
    """Return a raw pooled embedding by content key, or ``None`` on cache miss."""
    root = Path(cache_dir)
    index_path = _index_path(root)
    if not index_path.exists():
        return None
    with closing(sqlite3.connect(index_path)) as conn:
        _ensure_index_schema(conn)
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
        conn.commit()
    if row is None:
        return None
    shard_path = root / str(row[0])
    row_offset = int(row[1])
    try:
        records = _read_records_from_shard(shard_path)
    except CacheCorruptError:
        raise
    if row_offset < 0 or row_offset >= len(records):
        raise CacheCorruptError(
            "cache index row_offset points outside shard",
            details={"shard_path": str(shard_path), "row_offset": row_offset},
        )
    record = records[row_offset]
    if record.key != key:
        raise CacheCorruptError(
            "cache index key does not match shard row",
            details={"shard_path": str(shard_path), "row_offset": row_offset},
        )
    return record.embedding


def reindex_cache(cache_dir: Path | str) -> CacheReindexReport:
    """Rebuild ``index.sqlite`` from every readable Parquet shard."""
    root = Path(cache_dir)
    index_path = _index_path(root)
    index_path.parent.mkdir(parents=True, exist_ok=True)
    if index_path.exists():
        index_path.unlink()
    indexed_shards = 0
    indexed_rows = 0
    with closing(sqlite3.connect(index_path)) as conn:
        _ensure_index_schema(conn)
        for shard in _iter_shards(root):
            records = _read_records_from_shard(shard)
            _insert_index_records(conn, root, shard, records)
            indexed_shards += 1
            indexed_rows += len(records)
        conn.commit()
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
            "embedding": [list(record.embedding) for record in records],
            "untargeted": [record.untargeted for record in records],
            "created_at": [record.created_at for record in records],
            "schema_version": [record.schema_version for record in records],
        },
        schema=_arrow_schema(pa),
    )
    pq.write_table(table, path, compression="zstd", compression_level=9)


def _read_records_from_shard(path: Path) -> tuple[WindowCacheRecord, ...]:
    _pa, pq = _require_pyarrow()
    try:
        table = pq.read_table(path)
    except Exception as exc:
        raise CacheCorruptError(
            "cache shard could not be read",
            details={"shard_path": str(path), "error": str(exc)},
        ) from exc
    required = set(_column_names())
    observed = set(table.column_names)
    if required - observed:
        raise CacheCorruptError(
            "cache shard is missing required column(s)",
            details={"shard_path": str(path), "missing": sorted(required - observed)},
        )
    try:
        return tuple(
            WindowCacheRecord(
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
            for row in table.to_pylist()
        )
    except (InputError, TypeError, ValueError) as exc:
        raise CacheCorruptError(
            "cache shard contains an invalid row",
            details={"shard_path": str(path), "error": str(exc)},
        ) from exc


def _arrow_schema(pa: Any) -> Any:
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


def _column_names() -> tuple[str, ...]:
    return (
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
    existing_by_key = {record.key: record for record in existing}
    for record in incoming:
        prior = existing_by_key.get(record.key)
        if prior is None:
            raise CacheCorruptError(
                "cache shard already exists; refusing in-place append",
                details={"shard_path": str(path), "window_hash": record.window_hash.hex()},
            )
        if prior.embedding != record.embedding:
            raise CacheCorruptError(
                "cache shard contains conflicting row for key",
                details={"shard_path": str(path), "window_hash": record.window_hash.hex()},
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


def _path_part(value: str) -> str:
    if not value:
        raise InputError("cache path component must be non-empty")
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
