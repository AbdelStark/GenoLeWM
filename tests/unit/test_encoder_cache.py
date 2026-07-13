"""Unit tests for ``geno_lewm.encoder.cache``."""

from __future__ import annotations

import multiprocessing
import os
import sqlite3
import string
import struct
import threading
from contextlib import closing
from dataclasses import replace
from pathlib import Path

import pytest
from hypothesis import HealthCheck, given, settings, strategies as st

import geno_lewm.encoder.cache as cache_module
from geno_lewm.encoder import (
    CACHE_SCHEMA_VERSION,
    POOL_CENTERED_MEAN,
    POOL_GLOBAL_MEAN,
    CacheLookupResult,
    WindowCacheKey,
    WindowCacheRecord,
    pool_hidden_states,
    read_cache_entry,
    read_embedding,
    read_embeddings,
    reindex_cache,
    repair_cache,
    shard_path_for,
    write_shard,
)
from geno_lewm.encoder._normalization import l2_normalize_state
from geno_lewm.errors import (
    CacheCorruptError,
    CacheKeyAlreadyIndexedError,
    InputError,
    RuntimeSetupError,
)

pytestmark = pytest.mark.skipif(
    os.name == "nt",
    reason="corrected cache I/O is intentionally fail-closed without POSIX dirfd primitives",
)


def _hash(seed: int) -> bytes:
    return bytes([seed % 256]) * 32


def _record(
    seed: int,
    *,
    pool_radius: int = 256,
    center_token: int = 128,
    dtype: str = "fp16",
) -> WindowCacheRecord:
    return WindowCacheRecord(
        chrom="1",
        start_bp=seed * 10,
        end_bp=(seed * 10) + 12_288,
        window_hash=_hash(seed),
        encoder_hash=_hash(100),
        state_layer=-1,
        pool_type=POOL_CENTERED_MEAN,
        pool_radius=pool_radius,
        center_token=center_token,
        dtype=dtype,
        embedding=(float(seed), float(seed + 1), float(seed + 2)),
        untargeted=False,
        created_at=seed + 1,
    )


def _write_v2_shard(tmp_path: Path, record: WindowCacheRecord, *, stride_block: int = 0) -> Path:
    pa = pytest.importorskip("pyarrow")
    pq = pytest.importorskip("pyarrow.parquet")
    shard = shard_path_for(
        tmp_path,
        encoder_id="HuggingFaceBio/Carbon-500M",
        state_layer=record.state_layer,
        pool_type=record.pool_type,
        pool_radius=record.pool_radius,
        contig=record.chrom,
        stride_block=stride_block,
    )
    shard.parent.mkdir(parents=True, exist_ok=True)
    schema = pa.schema(
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
    pq.write_table(
        pa.Table.from_pylist(
            [
                {
                    "chrom": record.chrom,
                    "start_bp": record.start_bp,
                    "end_bp": record.end_bp,
                    "window_hash": record.window_hash,
                    "encoder_hash": record.encoder_hash,
                    "state_layer": record.state_layer,
                    "pool_type": record.pool_type,
                    "pool_radius": record.pool_radius,
                    "center_token": record.center_token,
                    "dtype": record.dtype,
                    "embedding": list(record.embedding),
                    "untargeted": record.untargeted,
                    "created_at": record.created_at,
                    "schema_version": "2.0.0",
                }
            ],
            schema=schema,
        ),
        shard,
    )
    return shard


def _concurrent_write_worker(
    cache_dir: str,
    stride_block: int,
    record: WindowCacheRecord,
    barrier: object,
    queue: object,
) -> None:
    try:
        barrier.wait(timeout=20)  # type: ignore[attr-defined]
        path = write_shard(
            cache_dir,
            encoder_id="carbon",
            contig=record.chrom,
            stride_block=stride_block,
            records=[record],
        )
        queue.put(("ok", str(path)))  # type: ignore[attr-defined]
    except Exception as exc:  # pragma: no cover - asserted in the parent process
        queue.put((type(exc).__name__, str(exc)))  # type: ignore[attr-defined]


def _run_concurrent_writes(
    tmp_path: Path,
    requests: tuple[tuple[int, WindowCacheRecord], tuple[int, WindowCacheRecord]],
) -> list[tuple[str, str]]:
    context = multiprocessing.get_context("spawn")
    barrier = context.Barrier(len(requests))
    queue = context.Queue()
    processes = [
        context.Process(
            target=_concurrent_write_worker,
            args=(str(tmp_path), stride_block, record, barrier, queue),
        )
        for stride_block, record in requests
    ]
    for process in processes:
        process.start()
    results = [queue.get(timeout=30) for _process in processes]
    for process in processes:
        process.join(timeout=30)
        assert process.exitcode == 0
    return results


def test_record_with_created_at_fills_timestamp() -> None:
    record = WindowCacheRecord(
        chrom="1",
        start_bp=0,
        end_bp=12_288,
        window_hash=_hash(1),
        encoder_hash=_hash(2),
        state_layer=-1,
        pool_type=POOL_CENTERED_MEAN,
        pool_radius=256,
        center_token=128,
        dtype="fp16",
        embedding=(1.0,),
        untargeted=False,
    )

    assert record.created_at == 0
    assert record.with_created_at().created_at > 0


@given(st.integers(min_value=1, max_value=20))
@settings(deadline=None, suppress_health_check=[HealthCheck.function_scoped_fixture])
def test_round_trip_write_read_is_bit_equal(tmp_path: Path, seed: int) -> None:
    record = _record(seed)

    write_shard(
        tmp_path,
        encoder_id="HuggingFaceBio/Carbon-500M",
        contig="1",
        stride_block=seed,
        records=[record],
    )

    assert read_embedding(tmp_path, record.key) == record.embedding


def test_v3_shard_separates_logical_dtype_from_fixed_width_storage(tmp_path: Path) -> None:
    pa = pytest.importorskip("pyarrow")
    pq = pytest.importorskip("pyarrow.parquet")
    record = _record(24, dtype="bf16")

    shard = write_shard(
        tmp_path,
        encoder_id="HuggingFaceBio/Carbon-500M",
        contig="1",
        stride_block=0,
        records=[record],
    )
    table = pq.read_table(shard)

    assert CACHE_SCHEMA_VERSION == "3.0.0"
    embedding_type = table.schema.field("embedding").type
    assert pa.types.is_fixed_size_list(embedding_type)
    assert embedding_type.list_size == 3
    assert embedding_type.value_type == pa.float32()
    assert table.column("dtype").to_pylist() == ["bf16"]
    assert table.column("storage_dtype").to_pylist() == ["fp32"]
    assert read_embedding(tmp_path, record.key) == record.embedding


@pytest.mark.parametrize("dtype", ["bf16", "fp16", "fp32"])
def test_live_and_cached_states_are_bit_identical_for_every_logical_dtype(
    tmp_path: Path,
    dtype: str,
) -> None:
    raw_live = pool_hidden_states(
        ((0.1, 1.0), (0.2, 2.0)),
        edit_locus=0,
        center_token=0,
        content_token_bounds=(0, 2),
        pool_radius=1,
    ).vector
    record = replace(_record(58, dtype=dtype), embedding=raw_live)
    write_shard(
        tmp_path,
        encoder_id="carbon",
        contig=record.chrom,
        stride_block=0,
        records=[record],
    )

    cached_raw = read_embedding(tmp_path, record.key)
    missing = read_embedding(tmp_path, replace(_record(59), dtype=dtype).key)

    assert cached_raw is not None
    assert tuple(struct.pack("<f", value) for value in cached_raw) == tuple(
        struct.pack("<f", value) for value in raw_live
    )
    cached_normalized = l2_normalize_state(cached_raw)
    live_normalized = l2_normalize_state(raw_live)
    assert cached_normalized == live_normalized
    assert cached_normalized == tuple(
        struct.unpack("<f", struct.pack("<f", value))[0] for value in cached_normalized
    )
    assert missing is None


def test_v3_arrow_contract_marks_only_center_token_nullable(tmp_path: Path) -> None:
    pq = pytest.importorskip("pyarrow.parquet")
    record = _record(52)
    shard = write_shard(
        tmp_path,
        encoder_id="carbon",
        contig=record.chrom,
        stride_block=0,
        records=[record],
    )

    schema = pq.read_schema(shard)

    assert {field.name for field in schema if field.nullable} == {"center_token"}


def test_v3_loader_rejects_inner_embedding_null_without_coercion(tmp_path: Path) -> None:
    pa = pytest.importorskip("pyarrow")
    pq = pytest.importorskip("pyarrow.parquet")
    record = _record(53)
    shard = write_shard(
        tmp_path,
        encoder_id="carbon",
        contig=record.chrom,
        stride_block=0,
        records=[record],
    )
    table = pq.read_table(shard)
    embedding_field = pa.field(
        "embedding",
        pa.list_(pa.field("element", pa.float32(), nullable=True), 3),
        nullable=False,
    )
    malformed = table.set_column(
        table.schema.get_field_index("embedding"),
        embedding_field,
        pa.array([[record.embedding[0], None, record.embedding[2]]], type=embedding_field.type),
    )
    pq.write_table(malformed, shard)

    with pytest.raises(CacheCorruptError, match=r"physical schema|null"):
        reindex_cache(tmp_path)


def test_v3_loader_rejects_nullable_required_field_even_without_nulls(tmp_path: Path) -> None:
    pa = pytest.importorskip("pyarrow")
    pq = pytest.importorskip("pyarrow.parquet")
    record = _record(54)
    shard = write_shard(
        tmp_path,
        encoder_id="carbon",
        contig=record.chrom,
        stride_block=0,
        records=[record],
    )
    table = pq.read_table(shard)
    schema = table.schema.set(
        table.schema.get_field_index("chrom"),
        pa.field("chrom", pa.string(), nullable=True),
    )
    pq.write_table(pa.Table.from_arrays(table.columns, schema=schema), shard)

    with pytest.raises(CacheCorruptError, match="physical schema"):
        reindex_cache(tmp_path)


def test_reindex_wraps_unhashable_arrow_schema_values_as_cache_corruption(tmp_path: Path) -> None:
    pa = pytest.importorskip("pyarrow")
    pq = pytest.importorskip("pyarrow.parquet")
    record = _record(182)
    shard = write_shard(
        tmp_path,
        encoder_id="carbon",
        contig=record.chrom,
        stride_block=0,
        records=[record],
    )
    table = pq.read_table(shard)
    schema_index = table.schema.get_field_index("schema_version")
    malformed = table.set_column(
        schema_index,
        "schema_version",
        pa.array([[CACHE_SCHEMA_VERSION]], type=pa.list_(pa.string())),
    )
    pq.write_table(malformed, shard)

    with pytest.raises(CacheCorruptError, match="schema"):
        reindex_cache(tmp_path)


def test_repair_quarantines_unhashable_arrow_schema_values(tmp_path: Path) -> None:
    pa = pytest.importorskip("pyarrow")
    pq = pytest.importorskip("pyarrow.parquet")
    record = _record(183)
    shard = write_shard(
        tmp_path,
        encoder_id="carbon",
        contig=record.chrom,
        stride_block=0,
        records=[record],
    )
    table = pq.read_table(shard)
    storage_index = table.schema.get_field_index("storage_dtype")
    malformed = table.set_column(
        storage_index,
        "storage_dtype",
        pa.array([["fp32"]], type=pa.list_(pa.string())),
    )
    pq.write_table(malformed, shard)

    report = repair_cache(tmp_path)

    assert report.checked_shards == 1
    assert len(report.quarantined) == 1
    assert report.quarantined[0].is_file()
    assert report.reindex.indexed_rows == 0


def test_v2_shard_remains_readable_after_v3_reindex(tmp_path: Path) -> None:
    record = _record(25, dtype="bf16")
    _write_v2_shard(tmp_path, record)

    report = reindex_cache(tmp_path)

    assert report.indexed_rows == 1
    with pytest.raises(CacheCorruptError, match="requires cache schema 3"):
        read_embedding(tmp_path, record.key)
    assert read_embedding(tmp_path, record.key, policy="legacy_v2_only") == record.embedding


def test_v2_and_v3_coexist_with_explicit_deterministic_provenance_policy(
    tmp_path: Path,
) -> None:
    record = _record(51, dtype="fp32")
    legacy = replace(record, embedding=(1.0, 2.0, 3.0), schema_version="2.0.0")
    _write_v2_shard(tmp_path, legacy)
    write_shard(
        tmp_path,
        encoder_id="HuggingFaceBio/Carbon-500M",
        contig=record.chrom,
        stride_block=1,
        records=[record],
    )
    report = reindex_cache(tmp_path)

    corrected = read_cache_entry(tmp_path, record.key, policy="require_v3")
    preferred = read_cache_entry(tmp_path, record.key, policy="prefer_v3")
    replay = read_cache_entry(tmp_path, record.key, policy="legacy_v2_only")

    assert report.indexed_rows == 2
    assert isinstance(corrected, CacheLookupResult)
    assert corrected.embedding == record.embedding
    assert corrected.provenance.cache_schema_version == "3.0.0"
    assert corrected.provenance.physical_encoding == "fixed_size_list<float32>"
    assert preferred == corrected
    assert replay is not None
    assert replay.embedding == legacy.embedding
    assert replay.provenance.cache_schema_version == "2.0.0"
    assert replay.provenance.physical_encoding == "list<float16>"
    with closing(sqlite3.connect(tmp_path / "embeddings" / "index.sqlite")) as conn:
        rows = conn.execute(
            "SELECT cache_schema_version, physical_encoding FROM window_index ORDER BY 1"
        ).fetchall()
    assert rows == [("2.0.0", "list<float16>"), ("3.0.0", "fixed_size_list<float32>")]


def test_cache_key_invariance_distinct_configs_do_not_collide(tmp_path: Path) -> None:
    base = _record(1, pool_radius=128)
    distinct = WindowCacheRecord(
        chrom=base.chrom,
        start_bp=base.start_bp,
        end_bp=base.end_bp,
        window_hash=base.window_hash,
        encoder_hash=base.encoder_hash,
        state_layer=base.state_layer,
        pool_type=base.pool_type,
        pool_radius=256,
        center_token=base.center_token,
        dtype=base.dtype,
        embedding=(7.0, 8.0, 9.0),
        untargeted=base.untargeted,
        created_at=base.created_at + 1,
    )

    write_shard(
        tmp_path,
        encoder_id="carbon",
        contig="1",
        stride_block=0,
        records=[base],
    )
    write_shard(
        tmp_path,
        encoder_id="carbon",
        contig="1",
        stride_block=1,
        records=[distinct],
    )

    assert base.key != distinct.key
    assert read_embedding(tmp_path, base.key) == base.embedding
    assert read_embedding(tmp_path, distinct.key) == distinct.embedding


def test_v3_shard_paths_isolate_encoder_hash_and_logical_dtype(tmp_path: Path) -> None:
    base = _record(26, dtype="bf16")
    other_encoder = WindowCacheRecord(
        chrom=base.chrom,
        start_bp=base.start_bp + 1,
        end_bp=base.end_bp + 1,
        window_hash=_hash(27),
        encoder_hash=_hash(200),
        state_layer=base.state_layer,
        pool_type=base.pool_type,
        pool_radius=base.pool_radius,
        center_token=base.center_token,
        dtype=base.dtype,
        embedding=(31.0, 32.0, 33.0),
        untargeted=base.untargeted,
        created_at=base.created_at,
    )
    other_dtype = WindowCacheRecord(
        chrom=base.chrom,
        start_bp=base.start_bp + 2,
        end_bp=base.end_bp + 2,
        window_hash=_hash(28),
        encoder_hash=base.encoder_hash,
        state_layer=base.state_layer,
        pool_type=base.pool_type,
        pool_radius=base.pool_radius,
        center_token=base.center_token,
        dtype="fp32",
        embedding=(41.0, 42.0, 43.0),
        untargeted=base.untargeted,
        created_at=base.created_at,
    )

    paths = {
        write_shard(tmp_path, encoder_id="carbon", contig="1", stride_block=0, records=[record])
        for record in (base, other_encoder, other_dtype)
    }

    assert len(paths) == 3
    assert read_embedding(tmp_path, base.key) == base.embedding
    assert read_embedding(tmp_path, other_encoder.key) == other_encoder.embedding
    assert read_embedding(tmp_path, other_dtype.key) == other_dtype.embedding


def test_cache_key_includes_center_token(tmp_path: Path) -> None:
    first = _record(21, center_token=127)
    second = WindowCacheRecord(
        chrom=first.chrom,
        start_bp=first.start_bp,
        end_bp=first.end_bp,
        window_hash=first.window_hash,
        encoder_hash=first.encoder_hash,
        state_layer=first.state_layer,
        pool_type=first.pool_type,
        pool_radius=first.pool_radius,
        center_token=128,
        dtype=first.dtype,
        embedding=(91.0, 92.0, 93.0),
        untargeted=first.untargeted,
        created_at=first.created_at + 1,
    )

    write_shard(tmp_path, encoder_id="carbon", contig="1", stride_block=0, records=[first])
    write_shard(tmp_path, encoder_id="carbon", contig="1", stride_block=1, records=[second])

    assert first.key != second.key
    assert read_embedding(tmp_path, first.key) == first.embedding
    assert read_embedding(tmp_path, second.key) == second.embedding


def test_read_embedding_misses_when_index_absent(tmp_path: Path) -> None:
    assert read_embedding(tmp_path, _record(1).key) is None


def test_read_embeddings_preserves_order_duplicates_and_misses(tmp_path: Path) -> None:
    first = _record(35)
    second = _record(36)
    missing = _record(37)
    write_shard(
        tmp_path,
        encoder_id="carbon",
        contig="1",
        stride_block=0,
        records=[first, second],
    )

    observed = read_embeddings(
        tmp_path,
        [second.key, missing.key, first.key, second.key],
    )

    assert observed == (second.embedding, None, first.embedding, second.embedding)


def test_read_embeddings_reads_shared_row_group_once(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pq = pytest.importorskip("pyarrow.parquet")
    records = [_record(41), _record(42), _record(43)]
    write_shard(
        tmp_path,
        encoder_id="carbon",
        contig="1",
        stride_block=0,
        records=records,
    )
    observed_row_groups: list[int] = []
    original_read_row_group = pq.ParquetFile.read_row_group

    def tracked_read_row_group(parquet: object, row_group: int, *args: object, **kwargs: object):
        observed_row_groups.append(row_group)
        return original_read_row_group(parquet, row_group, *args, **kwargs)

    monkeypatch.setattr(pq.ParquetFile, "read_row_group", tracked_read_row_group)

    observed = read_embeddings(
        tmp_path,
        [records[2].key, records[0].key, records[2].key],
    )

    assert observed == (records[2].embedding, records[0].embedding, records[2].embedding)
    assert observed_row_groups == [0]


def test_reindex_rebuilds_sqlite_without_data_loss(tmp_path: Path) -> None:
    records = [_record(2), _record(3)]
    write_shard(tmp_path, encoder_id="carbon", contig="1", stride_block=0, records=records)
    index = tmp_path / "embeddings" / "index.sqlite"
    index.unlink()

    report = reindex_cache(tmp_path)

    assert report.indexed_shards == 1
    assert report.indexed_rows == 2
    assert read_embedding(tmp_path, records[0].key) == records[0].embedding
    with closing(sqlite3.connect(index)) as conn:
        assert conn.execute("SELECT COUNT(*) FROM window_index").fetchone()[0] == 2


def test_reindex_empty_cache_creates_empty_index(tmp_path: Path) -> None:
    report = reindex_cache(tmp_path)

    assert report.indexed_shards == 0
    assert report.indexed_rows == 0
    assert report.index_path.is_file()


def test_failed_reindex_preserves_previous_complete_index(tmp_path: Path) -> None:
    record = _record(34)
    write_shard(tmp_path, encoder_id="carbon", contig="1", stride_block=0, records=[record])
    index = tmp_path / "embeddings" / "index.sqlite"
    original_index = index.read_bytes()
    corrupt = tmp_path / "embeddings" / "zz-corrupt" / "bad.parquet"
    corrupt.parent.mkdir(parents=True)
    corrupt.write_bytes(b"not parquet")

    with pytest.raises(CacheCorruptError, match="could not be read"):
        reindex_cache(tmp_path)

    assert index.read_bytes() == original_index
    assert read_embedding(tmp_path, record.key) == record.embedding
    assert list((tmp_path / "embeddings").glob("*.tmp")) == []


def test_index_stores_hashes_as_hex_text(tmp_path: Path) -> None:
    record = _record(4)
    write_shard(tmp_path, encoder_id="carbon", contig="1", stride_block=0, records=[record])

    index = tmp_path / "embeddings" / "index.sqlite"
    with closing(sqlite3.connect(index)) as conn:
        window_hash, encoder_hash = conn.execute(
            "SELECT window_hash, encoder_hash FROM window_index"
        ).fetchone()

    assert isinstance(window_hash, str)
    assert isinstance(encoder_hash, str)
    assert window_hash == record.window_hash.hex()
    assert encoder_hash == record.encoder_hash.hex()


def test_legacy_sqlite_index_requires_explicit_reindex_migration(tmp_path: Path) -> None:
    record = _record(22)
    write_shard(tmp_path, encoder_id="carbon", contig="1", stride_block=0, records=[record])
    index = tmp_path / "embeddings" / "index.sqlite"
    index.unlink()
    with closing(sqlite3.connect(index)) as conn:
        conn.execute(
            """
            CREATE TABLE window_index (
                window_hash TEXT NOT NULL,
                encoder_hash TEXT NOT NULL,
                state_layer INTEGER NOT NULL,
                pool_type TEXT NOT NULL,
                pool_radius INTEGER NOT NULL,
                dtype TEXT NOT NULL,
                shard_path TEXT NOT NULL,
                row_offset INTEGER NOT NULL,
                created_at INTEGER NOT NULL,
                PRIMARY KEY (
                    window_hash, encoder_hash, state_layer, pool_type, pool_radius, dtype
                )
            )
            """
        )
        conn.execute(
            """
            INSERT INTO window_index VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                record.window_hash.hex(),
                record.encoder_hash.hex(),
                record.state_layer,
                record.pool_type,
                record.pool_radius,
                record.dtype,
                "legacy.parquet",
                0,
                record.created_at,
            ),
        )
        conn.commit()

    with pytest.raises(CacheCorruptError, match="reindex"):
        read_embedding(tmp_path, record.key)
    with closing(sqlite3.connect(index)) as conn:
        columns = [row[1] for row in conn.execute("PRAGMA table_info(window_index)")]
        assert "center_token" not in columns
        assert conn.execute("SELECT COUNT(*) FROM window_index").fetchone()[0] == 1

    reindex_cache(tmp_path)
    assert read_embedding(tmp_path, record.key) == record.embedding


def test_index_is_strict_and_rejects_invalid_row_offsets(tmp_path: Path) -> None:
    record = _record(67)
    write_shard(tmp_path, encoder_id="carbon", contig="1", stride_block=0, records=[record])
    index = tmp_path / "embeddings" / "index.sqlite"

    with closing(sqlite3.connect(index)) as conn:
        table = conn.execute(
            "SELECT strict FROM pragma_table_list WHERE name = 'window_index'"
        ).fetchone()
        if sqlite3.sqlite_version_info >= (3, 37, 0):
            assert table == (1,)
        for value in (-1, 1.5, "1", None):
            with pytest.raises(sqlite3.IntegrityError):
                conn.execute("UPDATE window_index SET row_offset = ?", (value,))
            conn.rollback()
        with pytest.raises(OverflowError):
            conn.execute("UPDATE window_index SET row_offset = ?", (2**63,))


def test_index_rejects_shape_compatible_table_without_contract_checks(tmp_path: Path) -> None:
    index = tmp_path / "embeddings" / "index.sqlite"
    index.parent.mkdir(parents=True)
    with closing(sqlite3.connect(index)) as conn:
        conn.execute(
            """
            CREATE TABLE window_index (
                window_hash ANY NOT NULL,
                encoder_hash ANY NOT NULL,
                state_layer ANY NOT NULL,
                pool_type ANY NOT NULL,
                pool_radius ANY NOT NULL,
                center_token ANY NOT NULL,
                dtype ANY NOT NULL,
                cache_schema_version ANY NOT NULL,
                physical_encoding ANY NOT NULL,
                shard_path ANY NOT NULL,
                row_offset ANY NOT NULL,
                created_at ANY NOT NULL,
                PRIMARY KEY (
                    window_hash, encoder_hash, state_layer, pool_type, pool_radius,
                    center_token, dtype, cache_schema_version, physical_encoding
                )
            ) STRICT
            """
        )
        conn.execute("CREATE INDEX idx_shard_path ON window_index(shard_path)")
        conn.execute("PRAGMA user_version = 4")
        conn.commit()

    with pytest.raises(CacheCorruptError, match=r"schema.*unsafe|reindex"):
        read_embedding(tmp_path, _record(178).key)


def test_index_requires_the_canonical_secondary_index(tmp_path: Path) -> None:
    record = _record(179)
    write_shard(
        tmp_path, encoder_id="carbon", contig=record.chrom, stride_block=0, records=[record]
    )
    with closing(sqlite3.connect(tmp_path / "embeddings" / "index.sqlite")) as conn:
        conn.execute("DROP INDEX idx_shard_path")
        conn.commit()

    with pytest.raises(CacheCorruptError, match=r"schema.*unsafe|reindex"):
        read_embedding(tmp_path, record.key)


def test_index_rejects_losslessly_coercible_non_integer_contract_values(tmp_path: Path) -> None:
    record = _record(180)
    write_shard(
        tmp_path, encoder_id="carbon", contig=record.chrom, stride_block=0, records=[record]
    )
    index = tmp_path / "embeddings" / "index.sqlite"

    with closing(sqlite3.connect(index)) as conn:
        for field in ("state_layer", "pool_radius", "center_token", "created_at"):
            with pytest.raises(sqlite3.IntegrityError):
                conn.execute(f"UPDATE window_index SET {field} = ?", (1.0,))
            conn.rollback()


def test_index_fails_closed_when_sqlite_cannot_enforce_strict_tables(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(cache_module.sqlite3, "sqlite_version_info", (3, 36, 0))

    with pytest.raises(RuntimeSetupError, match=r"SQLite.*3\.37|STRICT"):
        write_shard(
            tmp_path,
            encoder_id="carbon",
            contig="1",
            stride_block=0,
            records=[_record(181)],
        )


def test_append_uses_one_direct_index_transaction_without_full_copy_or_replace(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    first = _record(72)
    write_shard(tmp_path, encoder_id="carbon", contig="1", stride_block=0, records=[first])
    index = tmp_path / "embeddings" / "index.sqlite"
    before_inode = index.stat().st_ino

    original_read_bytes = Path.read_bytes

    def reject_index_copy(path: Path) -> bytes:
        if path.name == "index.sqlite":
            raise AssertionError("per-shard append copied the complete SQLite index")
        return original_read_bytes(path)

    monkeypatch.setattr(Path, "read_bytes", reject_index_copy)
    monkeypatch.setattr(
        cache_module,
        "_publish_index_bytes",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("per-shard append atomically replaced the complete SQLite index")
        ),
    )
    batch = [_record(seed) for seed in range(73, 173)]

    write_shard(tmp_path, encoder_id="carbon", contig="1", stride_block=1, records=batch)

    assert index.stat().st_ino == before_inode
    with closing(sqlite3.connect(index)) as conn:
        assert conn.execute("SELECT COUNT(*) FROM window_index").fetchone() == (101,)


def test_repair_quarantines_truncated_shards(tmp_path: Path) -> None:
    record = _record(5)
    shard = write_shard(tmp_path, encoder_id="carbon", contig="1", stride_block=0, records=[record])
    body = shard.read_bytes()
    shard.write_bytes(body[: max(1, len(body) // 3)])

    report = repair_cache(tmp_path)

    assert report.checked_shards == 1
    assert len(report.quarantined) == 1
    assert not shard.exists()
    assert report.quarantined[0].is_file()
    assert read_embedding(tmp_path, record.key) is None


def test_repair_quarantines_legacy_shard_without_center_token(tmp_path: Path) -> None:
    pq = pytest.importorskip("pyarrow.parquet")
    record = _record(23)
    shard = write_shard(tmp_path, encoder_id="carbon", contig="1", stride_block=0, records=[record])
    table = pq.read_table(shard)
    pq.write_table(table.drop(["center_token"]), shard)

    report = repair_cache(tmp_path)

    assert report.checked_shards == 1
    assert len(report.quarantined) == 1
    assert report.reindex.indexed_rows == 0
    assert not shard.exists()


def test_repair_disambiguates_existing_quarantine_file(tmp_path: Path) -> None:
    record = _record(6)
    shard = write_shard(tmp_path, encoder_id="carbon", contig="1", stride_block=0, records=[record])
    quarantine_file = (
        tmp_path / "embeddings" / ".quarantine" / shard.relative_to(tmp_path / "embeddings")
    )
    quarantine_file.parent.mkdir(parents=True)
    quarantine_file.write_bytes(b"already quarantined")
    shard.write_bytes(b"truncated")

    report = repair_cache(tmp_path)

    assert len(report.quarantined) == 1
    assert report.quarantined[0].name.startswith("ctg-")
    assert ".parquet." in report.quarantined[0].name


def test_repair_rejects_symlinked_quarantine_without_moving_shard_outside_root(
    tmp_path: Path,
) -> None:
    record = _record(70)
    shard = write_shard(tmp_path, encoder_id="carbon", contig="1", stride_block=0, records=[record])
    shard.write_bytes(b"corrupt")
    outside = tmp_path.parent / f"{tmp_path.name}-quarantine"
    outside.mkdir()
    (tmp_path / "embeddings" / ".quarantine").symlink_to(
        outside,
        target_is_directory=True,
    )

    with pytest.raises(CacheCorruptError, match=r"symlink|unsafe|quarantine"):
        repair_cache(tmp_path)

    assert shard.exists()
    assert list(outside.iterdir()) == []


def test_existing_shard_is_noop_for_identical_rows(tmp_path: Path) -> None:
    record = _record(7)

    first = write_shard(tmp_path, encoder_id="carbon", contig="1", stride_block=0, records=[record])
    second = write_shard(
        tmp_path, encoder_id="carbon", contig="1", stride_block=0, records=[record]
    )

    assert first == second
    assert read_embedding(tmp_path, record.key) == record.embedding


def test_concurrent_same_path_same_key_has_one_verified_winner(tmp_path: Path) -> None:
    record = _record(60)

    results = _run_concurrent_writes(tmp_path, ((0, record), (0, record)))

    assert [status for status, _detail in results].count("ok") == 2
    assert len({detail for status, detail in results if status == "ok"}) == 1
    assert len(list(tmp_path.rglob("*.parquet"))) == 1
    assert list(tmp_path.rglob("*.tmp")) == []
    assert read_embedding(tmp_path, record.key) == record.embedding


def test_concurrent_same_path_distinct_keys_never_overwrites_winner(tmp_path: Path) -> None:
    first = _record(61)
    second = _record(62)

    results = _run_concurrent_writes(tmp_path, ((0, first), (0, second)))

    assert sorted(status for status, _detail in results) == ["CacheCorruptError", "ok"]
    assert len(list(tmp_path.rglob("*.parquet"))) == 1
    hits = [read_embedding(tmp_path, key) for key in (first.key, second.key)]
    assert sum(hit is not None for hit in hits) == 1
    assert list(tmp_path.rglob("*.tmp")) == []


def test_concurrent_different_paths_same_key_leaves_no_orphan(tmp_path: Path) -> None:
    record = _record(63)

    results = _run_concurrent_writes(tmp_path, ((0, record), (1, record)))

    assert sorted(status for status, _detail in results) == [
        CacheKeyAlreadyIndexedError.__name__,
        "ok",
    ]
    assert len(list(tmp_path.rglob("*.parquet"))) == 1
    assert read_embedding(tmp_path, record.key) == record.embedding
    assert list(tmp_path.rglob("*.tmp")) == []


def test_first_index_bootstrap_is_never_visible_in_an_incomplete_state(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    record = _record(177)
    entered_schema_creation = threading.Event()
    release_schema_creation = threading.Event()
    original = cache_module._ensure_index_schema
    failures: list[BaseException] = []

    def paused_schema_creation(conn: sqlite3.Connection, *, create: bool) -> None:
        if create and not entered_schema_creation.is_set():
            entered_schema_creation.set()
            if not release_schema_creation.wait(timeout=10):
                raise AssertionError("timed out waiting to release index bootstrap")
        original(conn, create=create)

    def write_first_shard() -> None:
        try:
            write_shard(
                tmp_path,
                encoder_id="carbon",
                contig=record.chrom,
                stride_block=0,
                records=[record],
            )
        except BaseException as exc:  # pragma: no cover - surfaced below in the main thread
            failures.append(exc)

    monkeypatch.setattr(cache_module, "_ensure_index_schema", paused_schema_creation)
    writer = threading.Thread(target=write_first_shard)
    writer.start()
    assert entered_schema_creation.wait(timeout=10)
    try:
        assert not (tmp_path / "embeddings" / "index.sqlite").exists()
        assert read_embedding(tmp_path, record.key) is None
    finally:
        release_schema_creation.set()
        writer.join(timeout=10)

    assert not writer.is_alive()
    assert failures == []
    assert read_embedding(tmp_path, record.key) == record.embedding


def test_publication_rejects_symlinked_namespace_parent_without_outside_write(
    tmp_path: Path,
) -> None:
    outside = tmp_path.parent / f"{tmp_path.name}-outside-parent"
    outside.mkdir()
    embeddings = tmp_path / "embeddings"
    embeddings.mkdir()
    (embeddings / "v3").symlink_to(outside, target_is_directory=True)
    record = _record(64)

    with pytest.raises(CacheCorruptError, match=r"symlink|unsafe|directory"):
        write_shard(
            tmp_path,
            encoder_id="carbon",
            contig=record.chrom,
            stride_block=0,
            records=[record],
        )

    assert list(outside.iterdir()) == []


def test_publication_rejects_symlinked_final_without_touching_target(tmp_path: Path) -> None:
    record = _record(65)
    final = shard_path_for(
        tmp_path,
        encoder_id="carbon",
        encoder_hash=record.encoder_hash,
        dtype=record.dtype,
        state_layer=record.state_layer,
        pool_type=record.pool_type,
        pool_radius=record.pool_radius,
        contig=record.chrom,
        stride_block=0,
    )
    final.parent.mkdir(parents=True)
    outside = tmp_path.parent / f"{tmp_path.name}-winner.parquet"
    outside.write_bytes(b"do not touch")
    final.symlink_to(outside)

    with pytest.raises(CacheCorruptError, match=r"symlink|unsafe|regular"):
        write_shard(
            tmp_path,
            encoder_id="carbon",
            contig=record.chrom,
            stride_block=0,
            records=[record],
        )

    assert outside.read_bytes() == b"do not touch"


def test_publication_rejects_symlinked_index_without_touching_target(tmp_path: Path) -> None:
    embeddings = tmp_path / "embeddings"
    embeddings.mkdir(parents=True)
    outside = tmp_path.parent / f"{tmp_path.name}-index.sqlite"
    outside.write_bytes(b"not an index")
    (embeddings / "index.sqlite").symlink_to(outside)
    record = _record(66)

    with pytest.raises(CacheCorruptError, match=r"index.*symlink|index.*regular|unsafe"):
        write_shard(
            tmp_path,
            encoder_id="carbon",
            contig=record.chrom,
            stride_block=0,
            records=[record],
        )

    assert outside.read_bytes() == b"not an index"
    assert list(tmp_path.rglob("*.parquet")) == []


def test_dirfd_publication_fails_closed_if_namespace_parent_is_swapped(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    if cache_module.os.name == "nt":
        pytest.skip("dirfd/no-follow swap protection is POSIX-specific")
    record = _record(173)
    expected = shard_path_for(
        tmp_path,
        encoder_id="carbon",
        encoder_hash=record.encoder_hash,
        dtype=record.dtype,
        state_layer=record.state_layer,
        pool_type=record.pool_type,
        pool_radius=record.pool_radius,
        contig=record.chrom,
        stride_block=0,
    )
    outside = tmp_path.parent / f"{tmp_path.name}-swap-target"
    outside.mkdir()
    original_link = cache_module.os.link
    swapped = False

    def swap_before_install(source: object, destination: object, **kwargs: object) -> None:
        nonlocal swapped
        if not swapped and str(destination).endswith(".parquet"):
            swapped = True
            moved = expected.parent.with_name(expected.parent.name + "-held")
            expected.parent.rename(moved)
            expected.parent.symlink_to(outside, target_is_directory=True)
        original_link(source, destination, **kwargs)

    monkeypatch.setattr(cache_module.os, "link", swap_before_install)

    with pytest.raises(CacheCorruptError, match=r"binding|symlink|unsafe"):
        write_shard(
            tmp_path,
            encoder_id="carbon",
            contig=record.chrom,
            stride_block=0,
            records=[record],
        )

    assert swapped
    assert list(outside.iterdir()) == []
    assert list(tmp_path.rglob("*.parquet")) == []


def test_inspection_rejects_parent_swap_after_held_descriptor_read(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    record = _record(174)
    shard = write_shard(
        tmp_path,
        encoder_id="carbon",
        contig=record.chrom,
        stride_block=0,
        records=[record],
    )
    outside = tmp_path.parent / f"{tmp_path.name}-inspect-outside"
    outside.mkdir()
    victim = outside / "victim.txt"
    victim.write_text("untouched", encoding="utf-8")
    original_read = cache_module._read_records_from_descriptor

    def read_then_swap(descriptor: int, *, path: Path) -> object:
        records = original_read(descriptor, path=path)
        held = shard.parent.with_name(shard.parent.name + "-held")
        shard.parent.rename(held)
        shard.parent.symlink_to(outside, target_is_directory=True)
        return records

    monkeypatch.setattr(cache_module, "_read_records_from_descriptor", read_then_swap)

    with pytest.raises(CacheCorruptError, match=r"binding|symlink|unsafe"):
        cache_module.inspect_cache_shard(tmp_path, shard)

    assert victim.read_text(encoding="utf-8") == "untouched"


def test_inspection_rejects_same_parent_final_name_replacement(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    record = _record(175)
    shard = write_shard(
        tmp_path,
        encoder_id="carbon",
        contig=record.chrom,
        stride_block=0,
        records=[record],
    )
    replacement = shard.with_name("replacement.parquet")
    replacement.write_bytes(shard.read_bytes())
    original_read = cache_module._read_records_from_descriptor

    def read_then_replace(descriptor: int, *, path: Path) -> object:
        records = original_read(descriptor, path=path)
        replacement.replace(shard)
        return records

    monkeypatch.setattr(cache_module, "_read_records_from_descriptor", read_then_replace)

    with pytest.raises(CacheCorruptError, match=r"binding"):
        cache_module.inspect_cache_shard(tmp_path, shard)


def test_publication_fsyncs_created_namespace_ancestors_before_completion(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    observed: list[str] = []
    original = cache_module.os.fsync

    def tracked(descriptor: int) -> None:
        mode = cache_module.os.fstat(descriptor).st_mode
        observed.append("directory" if cache_module.stat.S_ISDIR(mode) else "file")
        original(descriptor)

    monkeypatch.setattr(cache_module.os, "fsync", tracked)
    record = _record(68)

    write_shard(tmp_path, encoder_id="carbon", contig="1", stride_block=0, records=[record])

    assert observed.count("directory") >= 16
    assert "file" in observed
    assert observed[-1] == "directory"


def test_publication_fsyncs_parent_after_staged_name_is_unlinked(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events: list[str] = []
    original_fsync = cache_module.os.fsync
    original_unlink = cache_module.os.unlink

    def tracked_fsync(descriptor: int) -> None:
        mode = cache_module.os.fstat(descriptor).st_mode
        events.append("directory_fsync" if cache_module.stat.S_ISDIR(mode) else "file_fsync")
        original_fsync(descriptor)

    def tracked_unlink(path: object, *args: object, **kwargs: object) -> None:
        original_unlink(path, *args, **kwargs)
        if str(path).endswith(".tmp"):
            events.append("temp_unlink")

    monkeypatch.setattr(cache_module.os, "fsync", tracked_fsync)
    monkeypatch.setattr(cache_module.os, "unlink", tracked_unlink)
    record = _record(71)

    write_shard(tmp_path, encoder_id="carbon", contig="1", stride_block=0, records=[record])

    unlink_index = events.index("temp_unlink")
    assert events[unlink_index + 1] == "directory_fsync"


def test_directory_fsync_failure_never_publishes_shard_or_index_row(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original = cache_module.os.fsync
    directory_calls = 0

    def fail_during_ancestor_durability(descriptor: int) -> None:
        nonlocal directory_calls
        mode = cache_module.os.fstat(descriptor).st_mode
        if cache_module.stat.S_ISDIR(mode):
            directory_calls += 1
            if directory_calls == 4:
                raise OSError("injected ancestor fsync failure")
        original(descriptor)

    monkeypatch.setattr(cache_module.os, "fsync", fail_during_ancestor_durability)
    record = _record(69)

    with pytest.raises(OSError, match="injected ancestor fsync failure"):
        write_shard(
            tmp_path,
            encoder_id="carbon",
            contig="1",
            stride_block=0,
            records=[record],
        )

    assert list(tmp_path.rglob("*.parquet")) == []
    index = tmp_path / "embeddings" / "index.sqlite"
    assert index.is_file()
    with closing(sqlite3.connect(index)) as conn:
        assert conn.execute("SELECT COUNT(*) FROM window_index").fetchone() == (0,)


def test_existing_shard_resume_reuses_generated_created_at(tmp_path: Path) -> None:
    base = _record(44)
    record = WindowCacheRecord(
        chrom=base.chrom,
        start_bp=base.start_bp,
        end_bp=base.end_bp,
        window_hash=base.window_hash,
        encoder_hash=base.encoder_hash,
        state_layer=base.state_layer,
        pool_type=base.pool_type,
        pool_radius=base.pool_radius,
        center_token=base.center_token,
        dtype=base.dtype,
        embedding=base.embedding,
        untargeted=base.untargeted,
    )

    first = write_shard(tmp_path, encoder_id="carbon", contig="1", stride_block=0, records=[record])
    second = write_shard(
        tmp_path, encoder_id="carbon", contig="1", stride_block=0, records=[record]
    )

    assert second == first
    assert read_embedding(tmp_path, record.key) == record.embedding


def test_existing_shard_refuses_in_place_append(tmp_path: Path) -> None:
    first = _record(8)
    second = _record(9)

    write_shard(tmp_path, encoder_id="carbon", contig="1", stride_block=0, records=[first])

    with pytest.raises(CacheCorruptError):
        write_shard(tmp_path, encoder_id="carbon", contig="1", stride_block=0, records=[second])


def test_existing_shard_refuses_subset_as_completed_resume(tmp_path: Path) -> None:
    first = _record(29)
    second = _record(30)
    write_shard(
        tmp_path,
        encoder_id="carbon",
        contig="1",
        stride_block=0,
        records=[first, second],
    )

    with pytest.raises(CacheCorruptError, match="exactly match"):
        write_shard(
            tmp_path,
            encoder_id="carbon",
            contig="1",
            stride_block=0,
            records=[first],
        )


def test_existing_shard_refuses_row_reordering(tmp_path: Path) -> None:
    first = _record(55)
    second = _record(56)
    write_shard(
        tmp_path,
        encoder_id="carbon",
        contig="1",
        stride_block=0,
        records=[first, second],
    )

    with pytest.raises(CacheCorruptError, match="exactly match"):
        write_shard(
            tmp_path,
            encoder_id="carbon",
            contig="1",
            stride_block=0,
            records=[second, first],
        )


def test_existing_shard_distinguishes_positive_and_negative_zero_bits(tmp_path: Path) -> None:
    record = replace(_record(57), embedding=(0.0, 1.0, 2.0))
    signed = replace(record, embedding=(-0.0, 1.0, 2.0))
    assert struct.pack("<f", record.embedding[0]) != struct.pack("<f", signed.embedding[0])
    write_shard(
        tmp_path,
        encoder_id="carbon",
        contig="1",
        stride_block=0,
        records=[record],
    )

    with pytest.raises(CacheCorruptError, match="exactly match"):
        write_shard(
            tmp_path,
            encoder_id="carbon",
            contig="1",
            stride_block=0,
            records=[signed],
        )


def test_existing_shard_refuses_metadata_drift_for_same_key(tmp_path: Path) -> None:
    record = _record(38)
    drifted = WindowCacheRecord(
        chrom=record.chrom,
        start_bp=record.start_bp + 1,
        end_bp=record.end_bp + 1,
        window_hash=record.window_hash,
        encoder_hash=record.encoder_hash,
        state_layer=record.state_layer,
        pool_type=record.pool_type,
        pool_radius=record.pool_radius,
        center_token=record.center_token,
        dtype=record.dtype,
        embedding=record.embedding,
        untargeted=True,
        created_at=record.created_at,
    )
    write_shard(tmp_path, encoder_id="carbon", contig="1", stride_block=0, records=[record])

    with pytest.raises(CacheCorruptError, match="rows and metadata"):
        write_shard(
            tmp_path,
            encoder_id="carbon",
            contig="1",
            stride_block=0,
            records=[drifted],
        )

    assert read_embedding(tmp_path, record.key) == record.embedding


def test_existing_shard_refuses_created_at_drift(tmp_path: Path) -> None:
    record = _record(174)
    write_shard(tmp_path, encoder_id="carbon", contig="1", stride_block=0, records=[record])

    with pytest.raises(CacheCorruptError, match="exactly match"):
        write_shard(
            tmp_path,
            encoder_id="carbon",
            contig="1",
            stride_block=0,
            records=[replace(record, created_at=record.created_at + 1)],
        )


def test_failed_parquet_write_leaves_no_final_or_partial_shard(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pq = pytest.importorskip("pyarrow.parquet")
    record = _record(31)
    expected = shard_path_for(
        tmp_path,
        encoder_id="carbon",
        state_layer=record.state_layer,
        pool_type=record.pool_type,
        pool_radius=record.pool_radius,
        contig=record.chrom,
        stride_block=0,
        encoder_hash=record.encoder_hash,
        dtype=record.dtype,
    )

    def fail_after_partial_write(_table: object, where: object, **_kwargs: object) -> None:
        if hasattr(where, "write"):
            where.write(b"partial parquet")  # type: ignore[attr-defined]
            where.flush()  # type: ignore[attr-defined]
        else:
            Path(where).write_bytes(b"partial parquet")  # type: ignore[arg-type]
        raise OSError("injected parquet failure")

    monkeypatch.setattr(pq, "write_table", fail_after_partial_write)

    with pytest.raises(OSError, match="injected parquet failure"):
        write_shard(
            tmp_path,
            encoder_id="carbon",
            contig="1",
            stride_block=0,
            records=[record],
        )

    assert not expected.exists()
    assert list(tmp_path.rglob("*.tmp")) == []
    assert read_embedding(tmp_path, record.key) is None
    assert not any(path.name.startswith("<_io.") for path in Path.cwd().iterdir())


def test_index_transaction_failure_is_recoverable_from_valid_final_shard(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    baseline = _record(39)
    pending = _record(40)
    write_shard(
        tmp_path,
        encoder_id="carbon",
        contig="1",
        stride_block=0,
        records=[baseline],
    )
    original_insert = cache_module._insert_index_records
    failed = False

    def fail_after_insert(
        conn: sqlite3.Connection,
        cache_dir: Path,
        shard: Path,
        records: tuple[WindowCacheRecord, ...],
    ) -> None:
        nonlocal failed
        original_insert(conn, cache_dir, shard, records)
        if not failed:
            failed = True
            raise sqlite3.IntegrityError("injected index failure")

    monkeypatch.setattr(cache_module, "_insert_index_records", fail_after_insert)
    expected = shard_path_for(
        tmp_path,
        encoder_id="carbon",
        state_layer=pending.state_layer,
        pool_type=pending.pool_type,
        pool_radius=pending.pool_radius,
        contig=pending.chrom,
        stride_block=1,
        encoder_hash=pending.encoder_hash,
        dtype=pending.dtype,
    )

    with pytest.raises(sqlite3.IntegrityError, match="injected index failure"):
        write_shard(
            tmp_path,
            encoder_id="carbon",
            contig="1",
            stride_block=1,
            records=[pending],
        )

    assert expected.is_file()
    assert read_embedding(tmp_path, pending.key) is None
    monkeypatch.setattr(cache_module, "_insert_index_records", original_insert)

    retried = write_shard(
        tmp_path,
        encoder_id="carbon",
        contig="1",
        stride_block=1,
        records=[pending],
    )

    assert retried == expected
    assert read_embedding(tmp_path, baseline.key) == baseline.embedding
    assert read_embedding(tmp_path, pending.key) == pending.embedding


@pytest.mark.parametrize("cache_mode", ["absolute", "relative", "symlinked_ancestor"])
def test_index_failure_recovers_orphan_before_a_different_path_retry(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    cache_mode: str,
) -> None:
    if cache_mode == "relative":
        monkeypatch.chdir(tmp_path)
        cache_dir = Path("cache")
    elif cache_mode == "symlinked_ancestor":
        real_parent = tmp_path / "real-parent"
        real_parent.mkdir()
        alias_parent = tmp_path / "alias-parent"
        alias_parent.symlink_to(real_parent, target_is_directory=True)
        cache_dir = alias_parent / "cache"
    else:
        cache_dir = tmp_path
    record = _record(176)
    original_insert = cache_module._insert_index_records
    failed = False

    def fail_after_insert(
        conn: sqlite3.Connection,
        cache_dir: Path,
        shard: Path,
        records: tuple[WindowCacheRecord, ...],
    ) -> None:
        nonlocal failed
        original_insert(conn, cache_dir, shard, records)
        if not failed:
            failed = True
            raise sqlite3.IntegrityError("injected post-publication index failure")

    monkeypatch.setattr(cache_module, "_insert_index_records", fail_after_insert)
    with pytest.raises(sqlite3.IntegrityError, match="post-publication index failure"):
        write_shard(
            cache_dir,
            encoder_id="carbon",
            contig=record.chrom,
            stride_block=0,
            records=[record],
        )

    first_path = shard_path_for(
        cache_dir,
        encoder_id="carbon",
        encoder_hash=record.encoder_hash,
        dtype=record.dtype,
        state_layer=record.state_layer,
        pool_type=record.pool_type,
        pool_radius=record.pool_radius,
        contig=record.chrom,
        stride_block=0,
    )
    second_path = shard_path_for(
        cache_dir,
        encoder_id="carbon",
        encoder_hash=record.encoder_hash,
        dtype=record.dtype,
        state_layer=record.state_layer,
        pool_type=record.pool_type,
        pool_radius=record.pool_radius,
        contig=record.chrom,
        stride_block=1,
    )
    assert first_path.is_file()
    assert read_embedding(cache_dir, record.key) is None

    monkeypatch.setattr(cache_module, "_insert_index_records", original_insert)
    recovered = write_shard(
        cache_dir,
        encoder_id="carbon",
        contig=record.chrom,
        stride_block=1,
        records=[record],
    )

    assert recovered == first_path
    assert not second_path.exists()
    assert read_embedding(cache_dir, record.key) == record.embedding
    report = reindex_cache(cache_dir)
    assert report.indexed_shards == 1
    assert report.indexed_rows == 1


def test_reindex_resolves_a_durable_pending_publication(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    record = _record(185)
    original_insert = cache_module._insert_index_records
    failed = False

    def fail_after_insert(
        conn: sqlite3.Connection,
        cache_dir: Path,
        shard: Path,
        records: tuple[WindowCacheRecord, ...],
    ) -> None:
        nonlocal failed
        original_insert(conn, cache_dir, shard, records)
        if not failed:
            failed = True
            raise sqlite3.IntegrityError("injected post-publication index failure")

    monkeypatch.setattr(cache_module, "_insert_index_records", fail_after_insert)
    with pytest.raises(sqlite3.IntegrityError, match="post-publication index failure"):
        write_shard(
            tmp_path,
            encoder_id="carbon",
            contig=record.chrom,
            stride_block=0,
            records=[record],
        )
    pending = tmp_path / "embeddings" / ".pending-publication.json"
    assert pending.is_file()

    monkeypatch.setattr(cache_module, "_insert_index_records", original_insert)
    report = reindex_cache(tmp_path)

    assert report.indexed_rows == 1
    assert not pending.exists()
    assert read_embedding(tmp_path, record.key) == record.embedding


@pytest.mark.parametrize("retry_mode", ["subset", "reordered"])
def test_recovery_only_acknowledges_the_exact_original_batch(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    retry_mode: str,
) -> None:
    records = (_record(186), _record(187))
    original_insert = cache_module._insert_index_records
    failed = False

    def fail_after_insert(
        conn: sqlite3.Connection,
        cache_dir: Path,
        shard: Path,
        rows: tuple[WindowCacheRecord, ...],
    ) -> None:
        nonlocal failed
        original_insert(conn, cache_dir, shard, rows)
        if not failed:
            failed = True
            raise sqlite3.IntegrityError("injected post-publication index failure")

    monkeypatch.setattr(cache_module, "_insert_index_records", fail_after_insert)
    with pytest.raises(sqlite3.IntegrityError, match="post-publication index failure"):
        write_shard(
            tmp_path,
            encoder_id="carbon",
            contig="1",
            stride_block=0,
            records=records,
        )

    monkeypatch.setattr(cache_module, "_insert_index_records", original_insert)
    retried_records = [records[0]] if retry_mode == "subset" else list(reversed(records))
    with pytest.raises(CacheCorruptError, match="already indexed"):
        write_shard(
            tmp_path,
            encoder_id="carbon",
            contig="1",
            stride_block=1,
            records=retried_records,
        )

    assert read_embedding(tmp_path, records[0].key) == records[0].embedding
    assert read_embedding(tmp_path, records[1].key) == records[1].embedding


def test_staged_shard_rejects_physical_dtype_that_contradicts_metadata(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pa = pytest.importorskip("pyarrow")
    pq = pytest.importorskip("pyarrow.parquet")
    record = _record(32, dtype="bf16")
    original_write_table = pq.write_table

    def write_mislabeled_storage(table: object, where: object, **kwargs: object) -> None:
        embedding_index = table.schema.get_field_index("embedding")
        wrong_embedding = pa.array(
            table.column("embedding").to_pylist(),
            type=pa.list_(pa.float16()),
        )
        wrong_table = table.set_column(embedding_index, "embedding", wrong_embedding)
        original_write_table(wrong_table, where, **kwargs)

    monkeypatch.setattr(pq, "write_table", write_mislabeled_storage)

    with pytest.raises(CacheCorruptError, match="physical schema"):
        write_shard(
            tmp_path,
            encoder_id="carbon",
            contig="1",
            stride_block=0,
            records=[record],
        )

    assert list(tmp_path.rglob("*.parquet")) == []
    assert list(tmp_path.rglob("*.tmp")) == []


def test_existing_shard_refuses_conflicting_duplicate_key(tmp_path: Path) -> None:
    record = _record(10)
    conflicting = WindowCacheRecord(
        chrom=record.chrom,
        start_bp=record.start_bp,
        end_bp=record.end_bp,
        window_hash=record.window_hash,
        encoder_hash=record.encoder_hash,
        state_layer=record.state_layer,
        pool_type=record.pool_type,
        pool_radius=record.pool_radius,
        center_token=record.center_token,
        dtype=record.dtype,
        embedding=(99.0,),
        untargeted=record.untargeted,
        created_at=record.created_at,
    )

    write_shard(tmp_path, encoder_id="carbon", contig="1", stride_block=0, records=[record])

    with pytest.raises(CacheCorruptError):
        write_shard(
            tmp_path, encoder_id="carbon", contig="1", stride_block=0, records=[conflicting]
        )


def test_duplicate_key_in_new_shard_is_rejected_before_write(tmp_path: Path) -> None:
    record = _record(11)
    write_shard(tmp_path, encoder_id="carbon", contig="1", stride_block=0, records=[record])
    duplicate_path = shard_path_for(
        tmp_path,
        encoder_id="carbon",
        state_layer=record.state_layer,
        pool_type=record.pool_type,
        pool_radius=record.pool_radius,
        contig="1",
        stride_block=1,
        encoder_hash=record.encoder_hash,
        dtype=record.dtype,
    )

    with pytest.raises(CacheCorruptError):
        write_shard(tmp_path, encoder_id="carbon", contig="1", stride_block=1, records=[record])

    assert not duplicate_path.exists()


def test_duplicate_key_within_batch_is_rejected_before_write(tmp_path: Path) -> None:
    record = _record(33)

    with pytest.raises(InputError, match="duplicate cache key"):
        write_shard(
            tmp_path,
            encoder_id="carbon",
            contig="1",
            stride_block=0,
            records=[record, record],
        )

    assert list(tmp_path.rglob("*.parquet")) == []
    assert not (tmp_path / "embeddings" / "index.sqlite").exists()


def test_stale_index_row_is_rejected_on_reindexing_existing_shard(tmp_path: Path) -> None:
    record = _record(12)
    write_shard(tmp_path, encoder_id="carbon", contig="1", stride_block=0, records=[record])
    with closing(sqlite3.connect(tmp_path / "embeddings" / "index.sqlite")) as conn:
        conn.execute("UPDATE window_index SET row_offset = 99")
        conn.commit()

    with pytest.raises(CacheCorruptError):
        write_shard(tmp_path, encoder_id="carbon", contig="1", stride_block=0, records=[record])


def test_read_embedding_detects_stale_row_offset(tmp_path: Path) -> None:
    record = _record(13)
    write_shard(tmp_path, encoder_id="carbon", contig="1", stride_block=0, records=[record])
    with closing(sqlite3.connect(tmp_path / "embeddings" / "index.sqlite")) as conn:
        conn.execute("UPDATE window_index SET row_offset = 99")
        conn.commit()

    with pytest.raises(CacheCorruptError):
        read_embedding(tmp_path, record.key)


def test_read_embedding_detects_stale_key_mapping(tmp_path: Path) -> None:
    first = _record(14)
    second = _record(15)
    write_shard(tmp_path, encoder_id="carbon", contig="1", stride_block=0, records=[first, second])
    with closing(sqlite3.connect(tmp_path / "embeddings" / "index.sqlite")) as conn:
        conn.execute(
            "UPDATE window_index SET row_offset = 1 WHERE window_hash = ?",
            (first.window_hash.hex(),),
        )
        conn.commit()

    with pytest.raises(CacheCorruptError):
        read_embedding(tmp_path, first.key)


def test_read_embedding_rejects_index_path_outside_cache_root(tmp_path: Path) -> None:
    record = _record(45)
    write_shard(tmp_path, encoder_id="carbon", contig="1", stride_block=0, records=[record])
    with closing(sqlite3.connect(tmp_path / "embeddings" / "index.sqlite")) as conn:
        conn.execute("PRAGMA ignore_check_constraints = ON")
        conn.execute("UPDATE window_index SET shard_path = '../../outside.parquet'")
        conn.commit()

    with pytest.raises(CacheCorruptError, match="outside cache root"):
        read_embedding(tmp_path, record.key)


def test_read_embedding_rejects_symlinked_index_path_inside_cache_root(tmp_path: Path) -> None:
    if cache_module.os.name == "nt":
        pytest.skip("directory-symlink setup is POSIX-specific")
    record = _record(179)
    shard = write_shard(
        tmp_path,
        encoder_id="carbon",
        contig="1",
        stride_block=0,
        records=[record],
    )
    alias = tmp_path / "embeddings" / "alias"
    alias.symlink_to(shard.parent, target_is_directory=True)
    relative_alias = (alias / shard.name).relative_to(tmp_path).as_posix()
    with closing(sqlite3.connect(tmp_path / "embeddings" / "index.sqlite")) as conn:
        conn.execute("UPDATE window_index SET shard_path = ?", (relative_alias,))
        conn.commit()

    with pytest.raises(CacheCorruptError, match="symlink"):
        read_embedding(tmp_path, record.key)


def test_shard_path_without_v3_identity_preserves_the_literal_legacy_contract(
    tmp_path: Path,
) -> None:
    path = shard_path_for(
        tmp_path,
        encoder_id="HuggingFaceBio/Carbon-500M",
        state_layer=-1,
        pool_type=POOL_CENTERED_MEAN,
        pool_radius=256,
        contig="1",
        stride_block=0,
    )

    assert path == (
        tmp_path
        / "embeddings"
        / "HuggingFaceBio__Carbon-500M"
        / "-1"
        / "centered_mean_256"
        / "chr1_0.parquet"
    )


@pytest.mark.parametrize(
    ("left", "right"),
    [
        ("a/b", "a__b"),
        (r"a\b", "a__b"),
        ("Carbon", "carbon"),
        ("\N{LATIN SMALL LETTER E WITH ACUTE}", "e\N{COMBINING ACUTE ACCENT}"),
        ("CON", "con"),
        ("punctuation!?", "punctuation"),
        ("trailing.", "trailing "),
    ],
)
def test_v3_namespace_digest_is_injective_for_hostile_encoder_ids(
    tmp_path: Path,
    left: str,
    right: str,
) -> None:
    record = _record(49)

    def component(encoder_id: str) -> str:
        path = shard_path_for(
            tmp_path,
            encoder_id=encoder_id,
            encoder_hash=record.encoder_hash,
            dtype=record.dtype,
            state_layer=record.state_layer,
            pool_type=record.pool_type,
            pool_radius=record.pool_radius,
            contig="1",
            stride_block=0,
        )
        return path.parts[path.parts.index("v3") + 1]

    assert component(left) != component(right)


@pytest.mark.parametrize(
    ("left", "right"),
    [
        ("1/alt", "1__alt"),
        (r"1\alt", "1__alt"),
        ("X", "x"),
        ("\N{LATIN SMALL LETTER E WITH ACUTE}", "e\N{COMBINING ACUTE ACCENT}"),
        ("NUL", "nul"),
        ("chr!?", "chr"),
        ("1.", "1 "),
    ],
)
def test_v3_namespace_digest_is_injective_for_hostile_contigs(
    tmp_path: Path,
    left: str,
    right: str,
) -> None:
    record = _record(50)

    def filename(contig: str) -> str:
        return shard_path_for(
            tmp_path,
            encoder_id="carbon",
            encoder_hash=record.encoder_hash,
            dtype=record.dtype,
            state_layer=record.state_layer,
            pool_type=record.pool_type,
            pool_radius=record.pool_radius,
            contig=contig,
            stride_block=0,
        ).name

    assert filename(left) != filename(right)
    for observed in (filename(left), filename(right)):
        digest = observed.removeprefix("ctg-").removesuffix("_0.parquet")
        assert len(digest) == 64
        assert set(digest) <= set(string.hexdigits.lower())


@pytest.mark.parametrize(
    "kwargs",
    [
        {"chrom": ""},
        {"end_bp": 0},
        {"start_bp": True},
        {"created_at": True},
        {"untargeted": 0},
        {"embedding": ()},
        {"embedding": ("bad",)},
        {"embedding": (True,)},
        {"embedding": (float("nan"),)},
        {"embedding": (float("inf"),)},
        {"embedding": (float("-inf"),)},
        {"schema_version": "9.9.9"},
        {"window_hash": b"short"},
        {"encoder_hash": "not-bytes"},
        {"state_layer": True},
        {"pool_type": "unsupported"},
        {"pool_type": []},
        {"pool_radius": -1},
        {"pool_radius": 2**31},
        {"center_token": None},
        {"center_token": -1},
        {"center_token": True},
        {"center_token": 2**31},
        {"dtype": "int8"},
        {"dtype": []},
    ],
)
def test_record_validation_rejects_invalid_fields(kwargs: dict[str, object]) -> None:
    fields = {
        "chrom": "1",
        "start_bp": 0,
        "end_bp": 12_288,
        "window_hash": _hash(1),
        "encoder_hash": _hash(2),
        "state_layer": -1,
        "pool_type": POOL_CENTERED_MEAN,
        "pool_radius": 256,
        "center_token": 128,
        "dtype": "fp16",
        "embedding": (1.0,),
        "untargeted": False,
        **kwargs,
    }

    with pytest.raises(InputError):
        WindowCacheRecord(**fields)  # type: ignore[arg-type]


def test_window_cache_key_validation_rejects_invalid_fields() -> None:
    with pytest.raises(InputError):
        WindowCacheKey(
            window_hash=_hash(1),
            encoder_hash=_hash(2),
            state_layer=-1,
            pool_type=POOL_CENTERED_MEAN,
            pool_radius=256,
            center_token=128,
            dtype="int4",
        )


def test_global_pooling_key_requires_absent_center_token() -> None:
    key = WindowCacheKey(
        window_hash=_hash(1),
        encoder_hash=_hash(2),
        state_layer=-1,
        pool_type=POOL_GLOBAL_MEAN,
        pool_radius=0,
        center_token=None,
        dtype="fp16",
    )

    assert key.center_token is None
    with pytest.raises(InputError, match="center_token must be absent"):
        WindowCacheKey(
            window_hash=_hash(1),
            encoder_hash=_hash(2),
            state_layer=-1,
            pool_type=POOL_GLOBAL_MEAN,
            pool_radius=0,
            center_token=128,
            dtype="fp16",
        )


def test_v3_center_token_null_is_allowed_only_for_global_pooling(tmp_path: Path) -> None:
    base = _record(175)
    record = replace(
        base,
        pool_type=POOL_GLOBAL_MEAN,
        pool_radius=0,
        center_token=None,
        untargeted=True,
    )

    write_shard(tmp_path, encoder_id="carbon", contig="1", stride_block=0, records=[record])

    assert read_embedding(tmp_path, record.key) == record.embedding


@pytest.mark.parametrize(
    "kwargs",
    [
        {"contig": ""},
        {"contig": None},
        {"contig": 1},
        {"stride_block": -1},
        {"encoder_id": ""},
        {"encoder_id": None},
        {"encoder_id": 1},
        {"state_layer": True},
        {"pool_type": "unsupported"},
        {"pool_type": []},
        {"pool_radius": -1},
        {"pool_radius": 2**31},
    ],
)
def test_shard_path_validation_rejects_invalid_fields(
    tmp_path: Path, kwargs: dict[str, object]
) -> None:
    fields = {
        "encoder_id": "carbon",
        "state_layer": -1,
        "pool_type": POOL_CENTERED_MEAN,
        "pool_radius": 256,
        "contig": "1",
        "stride_block": 0,
        **kwargs,
    }

    with pytest.raises(InputError):
        shard_path_for(tmp_path, **fields)  # type: ignore[arg-type]


def test_write_shard_validation_rejects_invalid_batches(tmp_path: Path) -> None:
    base = _record(16)
    with pytest.raises(InputError):
        write_shard(tmp_path, encoder_id="carbon", contig="1", stride_block=0, records=[])
    with pytest.raises(InputError):
        write_shard(
            tmp_path,
            encoder_id="carbon",
            contig="2",
            stride_block=0,
            records=[base],
        )

    with pytest.raises(InputError):
        write_shard(
            tmp_path,
            encoder_id="carbon",
            contig="1",
            stride_block=0,
            records=[
                base,
                WindowCacheRecord(
                    chrom=base.chrom,
                    start_bp=base.start_bp,
                    end_bp=base.end_bp,
                    window_hash=_hash(17),
                    encoder_hash=base.encoder_hash,
                    state_layer=base.state_layer + 1,
                    pool_type=base.pool_type,
                    pool_radius=base.pool_radius,
                    center_token=base.center_token,
                    dtype=base.dtype,
                    embedding=base.embedding,
                    untargeted=base.untargeted,
                    created_at=base.created_at,
                ),
            ],
        )
    with pytest.raises(InputError):
        write_shard(
            tmp_path,
            encoder_id="carbon",
            contig="1",
            stride_block=0,
            records=[
                base,
                WindowCacheRecord(
                    chrom=base.chrom,
                    start_bp=base.start_bp,
                    end_bp=base.end_bp,
                    window_hash=_hash(19),
                    encoder_hash=base.encoder_hash,
                    state_layer=base.state_layer,
                    pool_type=base.pool_type,
                    pool_radius=base.pool_radius + 1,
                    center_token=base.center_token,
                    dtype=base.dtype,
                    embedding=base.embedding,
                    untargeted=base.untargeted,
                    created_at=base.created_at,
                ),
            ],
        )
    with pytest.raises(InputError):
        write_shard(
            tmp_path,
            encoder_id="carbon",
            contig="1",
            stride_block=0,
            records=[
                base,
                WindowCacheRecord(
                    chrom=base.chrom,
                    start_bp=base.start_bp,
                    end_bp=base.end_bp,
                    window_hash=_hash(18),
                    encoder_hash=base.encoder_hash,
                    state_layer=base.state_layer,
                    pool_type=POOL_GLOBAL_MEAN,
                    pool_radius=base.pool_radius,
                    center_token=None,
                    dtype=base.dtype,
                    embedding=base.embedding,
                    untargeted=base.untargeted,
                    created_at=base.created_at,
                ),
            ],
        )


@pytest.mark.parametrize(
    ("changes", "message"),
    [
        ({"window_hash": _hash(46), "encoder_hash": _hash(200)}, "encoder_hash"),
        ({"window_hash": _hash(47), "dtype": "bf16"}, "dtype"),
        ({"window_hash": _hash(48), "embedding": (1.0,)}, "embedding width"),
    ],
)
def test_write_shard_rejects_mixed_storage_identity_before_write(
    tmp_path: Path,
    changes: dict[str, object],
    message: str,
) -> None:
    base = _record(46)
    distinct = replace(base, **changes)

    with pytest.raises(InputError, match=message):
        write_shard(
            tmp_path,
            encoder_id="carbon",
            contig="1",
            stride_block=0,
            records=[base, distinct],
        )

    assert list(tmp_path.rglob("*.parquet")) == []
