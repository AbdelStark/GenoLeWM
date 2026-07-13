"""CLI tests for ``geno-lewm-cache-windows`` repair/reindex flows."""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

import geno_lewm.cli.cache_windows as cache_cli
from geno_lewm.cli._dispatch import run_app
from geno_lewm.cli.cache_windows import app
from geno_lewm.encoder import (
    POOL_CENTERED_MEAN,
    POOL_GLOBAL_MEAN,
    WindowCacheRecord,
    write_shard,
)
from geno_lewm.observability import shutdown_run
from geno_lewm.provenance import (
    SCHEMA_VERSION,
    Manifest,
    ManifestArtifact,
    ManifestEncoder,
    ManifestTraining,
    sha256_file,
    write_manifest,
)

pytestmark = pytest.mark.skipif(
    os.name == "nt",
    reason="corrected cache I/O is intentionally fail-closed without POSIX dirfd primitives",
)


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


class _FakeRawEncoder:
    encoder_hash = _hash(7)
    state_layer = 20
    pool_type = POOL_CENTERED_MEAN
    pool_radius = 8
    dtype = "bf16"
    normalize = False

    def pooling_identity(self, window: str, edit_locus: int | None) -> tuple[str, int, int | None]:
        del window
        if edit_locus is None:
            return POOL_GLOBAL_MEAN, 0, None
        return POOL_CENTERED_MEAN, 8, 1 + edit_locus // 6

    def encode_batch(
        self,
        windows: list[str],
        edit_loci: list[int | None],
    ) -> tuple[tuple[float, ...], ...]:
        return tuple(
            (float(len(window)), float(-1 if locus is None else locus))
            for window, locus in zip(windows, edit_loci, strict=True)
        )


def _write_build_inputs(tmp_path: Path) -> tuple[Path, Path, Path]:
    requests = tmp_path / "requests.jsonl"
    requests.write_text(
        json.dumps(
            {
                "request_id": "chr22-center-a",
                "chrom": "22",
                "start_bp": 100,
                "end_bp": 112,
                "window": "ACGTACGTACGT",
                "edit_locus": 0,
            },
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    config = tmp_path / "cache.yaml"
    config.write_text(
        """\
schema_version: 1.1.0
encoder:
  revision: pinned-revision
  dtype: bf16
  state_layer: 20
  pool_type: centered_mean
  pool_radius: 8
  normalize: true
  state_contract_version: l2_normalized_v2
""",
        encoding="utf-8",
    )
    digest = "sha256:" + "07" * 32
    artifact = ManifestArtifact(file="artifact.bin", hash=digest)
    manifest = Manifest(
        schema_version=SCHEMA_VERSION,
        model_name="cache-proof",
        model_version="0.0.0",
        release_id="cache-proof",
        encoder=ManifestEncoder(
            id="HuggingFaceBio/Carbon-500M",
            revision="pinned-revision",
            hash=digest,
        ),
        predictor=artifact,
        action_encoder=artifact,
        calibration=artifact,
        training=ManifestTraining(config_file="cache.yaml", hash=digest),
        eval=artifact,
    )
    manifest_path = write_manifest(manifest, tmp_path / "manifest.json")
    return requests, config, manifest_path


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


def test_cache_windows_build_cli_writes_finite_evidence_bundle(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    requests, config, manifest = _write_build_inputs(tmp_path)
    monkeypatch.setattr(cache_cli, "_build_encoder", lambda **_kwargs: _FakeRawEncoder())
    report_copy = tmp_path / "report-copy.json"

    rc = run_app(
        app,
        argv=[
            "--quiet",
            "--no-banner",
            "--cache-dir",
            str(tmp_path / "cache"),
            "--requests-jsonl",
            str(requests),
            "--evidence-dir",
            str(tmp_path / "evidence"),
            "--model-manifest",
            str(manifest),
            "--carbon-model-dir",
            str(tmp_path / "carbon"),
            "--config",
            str(config),
            "--created-at-ns",
            "1750000000000000000",
            "--rows-per-shard",
            "1",
            "--json-report",
            str(report_copy),
            "--run-id",
            "cache-cli-test",
            "--log-dir",
            str(tmp_path / "logs"),
        ],
    )
    shutdown_run("cache-cli-test", tmp_path / "logs")
    captured = capsys.readouterr()
    payload = json.loads(report_copy.read_text(encoding="utf-8"))

    assert rc == 0
    assert "completed_shards=1" in captured.out
    assert payload["ok"] is True
    assert payload["build"]["encoded_rows"] == 1
    assert payload["claim_boundary"]["ten_percent_corpus_completed"] is False
    assert [item["path"] for item in payload["evidence_artifacts"]["inputs"]] == [
        "inputs/encoder_config.yaml",
        "inputs/model_manifest.json",
    ]
    assert (tmp_path / "evidence" / "SHA256SUMS").is_file()


def test_cache_windows_build_cli_requires_all_immutable_inputs(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    rc = run_app(
        app,
        argv=["--quiet", "--no-banner", "--cache-dir", str(tmp_path / "cache")],
    )
    captured = capsys.readouterr()

    assert rc == 2
    assert "cache build mode requires explicit immutable inputs" in captured.err
