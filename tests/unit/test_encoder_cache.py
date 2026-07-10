"""Unit tests for ``geno_lewm.encoder.cache``."""

from __future__ import annotations

import sqlite3
from contextlib import closing
from pathlib import Path

import pytest
from hypothesis import HealthCheck, given, settings, strategies as st

from geno_lewm.encoder import (
    POOL_CENTERED_MEAN,
    POOL_GLOBAL_MEAN,
    WindowCacheKey,
    WindowCacheRecord,
    read_embedding,
    reindex_cache,
    repair_cache,
    shard_path_for,
    write_shard,
)
from geno_lewm.errors import CacheCorruptError, InputError


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
        encoder_hash=_hash(seed + 100),
        state_layer=-1,
        pool_type=POOL_CENTERED_MEAN,
        pool_radius=pool_radius,
        center_token=center_token,
        dtype=dtype,
        embedding=(float(seed), float(seed + 1), float(seed + 2)),
        untargeted=False,
        created_at=seed + 1,
    )


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


def test_legacy_sqlite_index_is_dropped_before_lookup(tmp_path: Path) -> None:
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

    assert read_embedding(tmp_path, record.key) is None
    with closing(sqlite3.connect(index)) as conn:
        columns = [row[1] for row in conn.execute("PRAGMA table_info(window_index)")]
        assert "center_token" in columns
        assert conn.execute("SELECT COUNT(*) FROM window_index").fetchone()[0] == 0
        assert conn.execute("PRAGMA user_version").fetchone()[0] == 2

    reindex_cache(tmp_path)
    assert read_embedding(tmp_path, record.key) == record.embedding


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
    destination = tmp_path / "embeddings" / ".quarantine" / "carbon" / "-1" / "centered_mean_256"
    destination.mkdir(parents=True)
    (destination / "chr1_0.parquet").write_bytes(b"already quarantined")
    shard.write_bytes(b"truncated")

    report = repair_cache(tmp_path)

    assert len(report.quarantined) == 1
    assert report.quarantined[0].name.startswith("chr1_0.parquet.")


def test_existing_shard_is_noop_for_identical_rows(tmp_path: Path) -> None:
    record = _record(7)

    first = write_shard(tmp_path, encoder_id="carbon", contig="1", stride_block=0, records=[record])
    second = write_shard(
        tmp_path, encoder_id="carbon", contig="1", stride_block=0, records=[record]
    )

    assert first == second
    assert read_embedding(tmp_path, record.key) == record.embedding


def test_existing_shard_refuses_in_place_append(tmp_path: Path) -> None:
    first = _record(8)
    second = _record(9)

    write_shard(tmp_path, encoder_id="carbon", contig="1", stride_block=0, records=[first])

    with pytest.raises(CacheCorruptError):
        write_shard(tmp_path, encoder_id="carbon", contig="1", stride_block=0, records=[second])


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
    )

    with pytest.raises(CacheCorruptError):
        write_shard(tmp_path, encoder_id="carbon", contig="1", stride_block=1, records=[record])

    assert not duplicate_path.exists()


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


def test_shard_path_sanitizes_encoder_id(tmp_path: Path) -> None:
    path = shard_path_for(
        tmp_path,
        encoder_id="HuggingFaceBio/Carbon-500M",
        state_layer=-1,
        pool_type=POOL_CENTERED_MEAN,
        pool_radius=256,
        contig="1",
        stride_block=0,
    )

    assert "HuggingFaceBio__Carbon-500M" in path.parts


@pytest.mark.parametrize(
    "kwargs",
    [
        {"chrom": ""},
        {"end_bp": 0},
        {"embedding": ()},
        {"embedding": ("bad",)},
        {"schema_version": "9.9.9"},
        {"window_hash": b"short"},
        {"encoder_hash": "not-bytes"},
        {"state_layer": True},
        {"pool_type": "unsupported"},
        {"pool_radius": -1},
        {"center_token": None},
        {"center_token": -1},
        {"center_token": True},
        {"dtype": "int8"},
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


@pytest.mark.parametrize(
    "kwargs",
    [
        {"contig": ""},
        {"stride_block": -1},
        {"encoder_id": ""},
        {"state_layer": True},
        {"pool_type": "unsupported"},
        {"pool_radius": -1},
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
