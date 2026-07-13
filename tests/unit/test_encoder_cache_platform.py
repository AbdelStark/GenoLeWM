"""Platform boundary tests for corrected cache publication."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

import geno_lewm.encoder.cache as cache_module
from geno_lewm.encoder import (
    POOL_CENTERED_MEAN,
    WindowCacheRecord,
    read_embedding,
    reindex_cache,
    write_shard,
)
from geno_lewm.errors import RuntimeSetupError


def test_cache_publication_fails_closed_without_secure_dirfd_primitives(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    if os.name != "nt":
        monkeypatch.delattr(cache_module.os, "O_NOFOLLOW")
    record = WindowCacheRecord(
        chrom="1",
        start_bp=0,
        end_bp=12_288,
        window_hash=b"w" * 32,
        encoder_hash=b"e" * 32,
        state_layer=-1,
        pool_type=POOL_CENTERED_MEAN,
        pool_radius=256,
        center_token=128,
        dtype="fp16",
        embedding=(1.0, 2.0),
        untargeted=False,
        created_at=1,
    )

    with pytest.raises(RuntimeSetupError, match=r"secure.*directory|POSIX"):
        write_shard(
            tmp_path,
            encoder_id="carbon",
            contig="1",
            stride_block=0,
            records=[record],
        )

    assert not os.path.lexists(tmp_path / "embeddings")


def test_existing_cache_reads_and_reindex_fail_closed_without_secure_dirfd_primitives(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    if os.name == "nt":
        pytest.skip("creating the fixture itself is intentionally unsupported on Windows")
    record = WindowCacheRecord(
        chrom="1",
        start_bp=0,
        end_bp=12_288,
        window_hash=b"w" * 32,
        encoder_hash=b"e" * 32,
        state_layer=-1,
        pool_type=POOL_CENTERED_MEAN,
        pool_radius=256,
        center_token=128,
        dtype="fp16",
        embedding=(1.0, 2.0),
        untargeted=False,
        created_at=1,
    )
    write_shard(
        tmp_path,
        encoder_id="carbon",
        contig="1",
        stride_block=0,
        records=[record],
    )
    monkeypatch.delattr(cache_module.os, "O_NOFOLLOW")

    with pytest.raises(RuntimeSetupError, match=r"secure.*directory|POSIX"):
        read_embedding(tmp_path, record.key)
    with pytest.raises(RuntimeSetupError, match=r"secure.*directory|POSIX"):
        reindex_cache(tmp_path)
