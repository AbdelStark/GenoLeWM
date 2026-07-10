"""CLI tests for ``geno-lewm-cache-windows`` repair/reindex flows."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from geno_lewm.cli._dispatch import run_app
from geno_lewm.cli.cache_windows import app
from geno_lewm.encoder import POOL_CENTERED_MEAN, WindowCacheRecord, write_shard
from geno_lewm.provenance import sha256_file


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
        center_token=128,
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


def test_cache_windows_reindex_writes_json_report(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    shard = write_shard(
        tmp_path, encoder_id="carbon", contig="1", stride_block=0, records=[_record()]
    )
    (tmp_path / "embeddings" / "index.sqlite").unlink()
    report_path = tmp_path / "reports" / "reindex.json"

    rc = run_app(
        app,
        argv=[
            "--quiet",
            "--no-banner",
            "--cache-dir",
            str(tmp_path),
            "--reindex",
            "--json-report",
            str(report_path),
        ],
    )
    captured = capsys.readouterr()
    payload = json.loads(report_path.read_text(encoding="utf-8"))

    assert rc == 0
    assert "indexed_rows=1" in captured.out
    assert payload["schema_version"] == "1.0.0"
    assert payload["generated_by"] == "geno-lewm-cache-windows"
    assert payload["operation"] == "reindex"
    assert payload["indexed_shards"] == 1
    assert payload["indexed_rows"] == 1
    assert payload["cache_artifacts"]["index"] == {
        "path": "embeddings/index.sqlite",
        "sha256": sha256_file(tmp_path / "embeddings" / "index.sqlite"),
        "size_bytes": (tmp_path / "embeddings" / "index.sqlite").stat().st_size,
    }
    assert payload["cache_artifacts"]["shards"] == [
        {
            "path": shard.relative_to(tmp_path).as_posix(),
            "sha256": sha256_file(shard),
            "size_bytes": shard.stat().st_size,
        }
    ]
    assert payload["throughput"]["elapsed_seconds"] >= 0.0
    assert payload["throughput"]["indexed_rows_per_second"] is not None


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


def test_cache_windows_repair_writes_json_report(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    shard = write_shard(
        tmp_path, encoder_id="carbon", contig="1", stride_block=0, records=[_record()]
    )
    shard.write_bytes(shard.read_bytes()[:8])
    report_path = tmp_path / "repair.json"

    rc = run_app(
        app,
        argv=[
            "--quiet",
            "--no-banner",
            "--cache-dir",
            str(tmp_path),
            "--repair",
            "--json-report",
            str(report_path),
        ],
    )
    captured = capsys.readouterr()
    payload = json.loads(report_path.read_text(encoding="utf-8"))

    assert rc == 0
    assert "quarantined=1" in captured.out
    assert payload["operation"] == "repair"
    assert payload["checked_shards"] == 1
    assert payload["indexed_shards"] == 0
    assert payload["indexed_rows"] == 0
    assert payload["cache_artifacts"]["shards"] == []
    quarantined = payload["quarantined_shards"]
    assert len(quarantined) == 1
    assert quarantined[0]["path"].startswith("embeddings/.quarantine/")
    assert quarantined[0]["sha256"] == sha256_file(tmp_path / quarantined[0]["path"])
    assert payload["throughput"]["elapsed_seconds"] >= 0.0
