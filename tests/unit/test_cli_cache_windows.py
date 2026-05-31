"""CLI tests for ``geno-lewm-cache-windows`` repair/reindex flows."""

from __future__ import annotations

from pathlib import Path

import pytest

from geno_lewm.cli._dispatch import run_app
from geno_lewm.cli.cache_windows import app
from geno_lewm.encoder import POOL_CENTERED_MEAN, WindowCacheRecord, write_shard


def _hash(seed: int) -> bytes:
    return bytes([seed % 256]) * 32


def _record() -> WindowCacheRecord:
    return WindowCacheRecord(
        chrom="1",
        start_bp=0,
        end_bp=12_288,
        window_hash=_hash(1),
        encoder_hash=_hash(2),
        state_layer=-1,
        pool_type=POOL_CENTERED_MEAN,
        pool_radius=256,
        dtype="fp16",
        embedding=(1.0, 2.0, 3.0),
        untargeted=False,
        created_at=1,
    )


def test_cache_windows_reindex_cli(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    write_shard(tmp_path, encoder_id="carbon", contig="1", stride_block=0, records=[_record()])
    (tmp_path / "embeddings" / "index.sqlite").unlink()

    rc = run_app(
        app,
        argv=["--quiet", "--no-banner", "--cache-dir", str(tmp_path), "--reindex"],
    )
    captured = capsys.readouterr()

    assert rc == 0
    assert "indexed_rows=1" in captured.out


def test_cache_windows_repair_cli(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    shard = write_shard(
        tmp_path, encoder_id="carbon", contig="1", stride_block=0, records=[_record()]
    )
    shard.write_bytes(shard.read_bytes()[:8])

    rc = run_app(
        app,
        argv=["--quiet", "--no-banner", "--cache-dir", str(tmp_path), "--repair"],
    )
    captured = capsys.readouterr()

    assert rc == 0
    assert "quarantined=1" in captured.out
