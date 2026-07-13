"""CLI tests for ``geno-lewm-cache-windows`` repair/reindex flows."""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

import geno_lewm.cli.cache_windows as cache_cli
from geno_lewm.cli._dispatch import run_app
from geno_lewm.cli.cache_windows import app
from geno_lewm.config import load_config
from geno_lewm.encoder import (
    POOL_CENTERED_MEAN,
    POOL_GLOBAL_MEAN,
    WindowCacheRecord,
    write_shard,
)
from geno_lewm.errors import InputError
from geno_lewm.observability import shutdown_run
from geno_lewm.provenance import (
    SCHEMA_VERSION,
    Manifest,
    ManifestArtifact,
    ManifestEncoder,
    ManifestTraining,
    load_manifest,
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
    device = "cpu"

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
    monkeypatch.setattr(
        cache_cli,
        "encoder_identity_hash",
        lambda *_args, **_kwargs: "sha256:" + "07" * 32,
    )
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
            "--hardware",
            "fixture CPU",
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
    assert payload["configuration"]["encoder_runtime_identity"]["path"] == (
        "encoder_runtime_identity.json"
    )
    assert (tmp_path / "evidence" / "encoder_runtime_identity.json").is_file()
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


def test_build_encoder_rejects_corrected_runtime_identity_at_cli_boundary(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _requests, config_path, manifest_path = _write_build_inputs(tmp_path)
    config = load_config(config_path)
    manifest = load_manifest(manifest_path)
    carbon = tmp_path / "carbon"
    carbon.mkdir()
    monkeypatch.setattr(
        cache_cli, "encoder_identity_hash", lambda *_args, **_kwargs: "sha256:" + "00" * 32
    )

    def must_not_construct(*_args: object, **_kwargs: object) -> object:
        raise AssertionError("runtime mismatch must fail before Carbon construction")

    monkeypatch.setattr(cache_cli, "CarbonStateEncoder", must_not_construct)

    with pytest.raises(InputError, match="runtime identity does not match"):
        cache_cli._build_encoder(
            config=config,
            manifest=manifest,
            carbon_model_dir=carbon,
            device="cpu",
        )


@pytest.mark.parametrize(
    ("option", "relative_output"),
    [("--json-report", "copy.json"), ("--log-dir", "logs")],
)
def test_cache_windows_build_cli_rejects_mutable_outputs_inside_evidence(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    option: str,
    relative_output: str,
) -> None:
    requests, config, manifest = _write_build_inputs(tmp_path)
    monkeypatch.setattr(cache_cli, "_build_encoder", lambda **_kwargs: _FakeRawEncoder())
    evidence = tmp_path / "evidence"
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
            str(evidence),
            "--model-manifest",
            str(manifest),
            "--carbon-model-dir",
            str(tmp_path / "carbon"),
            "--config",
            str(config),
            "--created-at-ns",
            "1750000000000000000",
            "--hardware",
            "fixture CPU",
            option,
            str(evidence / relative_output),
        ],
    )
    captured = capsys.readouterr()

    assert rc == 2
    assert f"{option} must be outside --evidence-dir" in captured.err
    assert not evidence.exists()


@pytest.mark.parametrize("alias_kind", ["symlink", "case", "parent"])
def test_json_report_rejects_portable_evidence_path_aliases(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    alias_kind: str,
) -> None:
    requests, config, manifest = _write_build_inputs(tmp_path)
    evidence = tmp_path / ("Evidence" if alias_kind == "case" else "evidence")
    if alias_kind == "symlink":
        evidence.mkdir()
        alias = tmp_path / "evidence-alias"
        alias.symlink_to(evidence, target_is_directory=True)
        report = alias / "report.json"
    elif alias_kind == "case":
        report = tmp_path / "evidence" / "report.json"
    else:
        report = evidence / "nested" / ".." / "report.json"

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
            str(evidence),
            "--model-manifest",
            str(manifest),
            "--carbon-model-dir",
            str(tmp_path / "carbon"),
            "--config",
            str(config),
            "--created-at-ns",
            "1750000000000000000",
            "--hardware",
            "fixture CPU",
            "--json-report",
            str(report),
        ],
    )
    captured = capsys.readouterr()

    assert rc == 2
    assert "--json-report must be outside --evidence-dir" in captured.err
    assert not report.exists()


def test_cli_stages_and_uses_one_immutable_snapshot_of_every_file_input(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    requests, config, manifest = _write_build_inputs(tmp_path)
    request_snapshot = requests.read_bytes()
    config_snapshot = config.read_bytes()
    manifest_snapshot = manifest.read_bytes()
    monkeypatch.setattr(
        cache_cli,
        "encoder_identity_hash",
        lambda *_args, **_kwargs: "sha256:" + "07" * 32,
    )

    def mutate_sources_after_validation(**_kwargs: object) -> _FakeRawEncoder:
        requests.write_text("mutated after capture\n", encoding="utf-8")
        config.write_text("mutated: true\n", encoding="utf-8")
        manifest.write_text("{}\n", encoding="utf-8")
        return _FakeRawEncoder()

    monkeypatch.setattr(cache_cli, "_build_encoder", mutate_sources_after_validation)
    evidence = tmp_path / "evidence"
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
            str(evidence),
            "--model-manifest",
            str(manifest),
            "--carbon-model-dir",
            str(tmp_path / "carbon"),
            "--config",
            str(config),
            "--created-at-ns",
            "1750000000000000000",
            "--hardware",
            "fixture CPU",
        ],
    )
    captured = capsys.readouterr()

    assert rc == 0, captured.err
    assert (evidence / "cache_build_requests.jsonl").read_bytes() == request_snapshot
    assert (evidence / "inputs/encoder_config.yaml").read_bytes() == config_snapshot
    assert (evidence / "inputs/model_manifest.json").read_bytes() == manifest_snapshot
    runtime_identity = json.loads(
        (evidence / "encoder_runtime_identity.json").read_text(encoding="utf-8")
    )
    assert runtime_identity["observed"] == "sha256:" + "07" * 32
    report = json.loads((evidence / "cache_build_report.json").read_text(encoding="utf-8"))
    assert report["configuration"]["resolved_config"]["sha256"] == sha256_file(
        evidence / "resolved_config.json"
    )


def test_post_checksum_report_write_cannot_return_success_with_open_evidence(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    requests, config, manifest = _write_build_inputs(tmp_path)
    evidence = tmp_path / "evidence"
    external_report = tmp_path / "external-report.json"
    monkeypatch.setattr(cache_cli, "_build_encoder", lambda **_kwargs: _FakeRawEncoder())
    monkeypatch.setattr(
        cache_cli,
        "encoder_identity_hash",
        lambda *_args, **_kwargs: "sha256:" + "07" * 32,
    )
    real_write_report = cache_cli._write_json_report

    def write_report_then_open_bundle(path: Path, payload: dict[str, object]) -> None:
        real_write_report(path, payload)
        (evidence / "late-unclosed-artifact.json").write_text("{}\n", encoding="utf-8")

    monkeypatch.setattr(cache_cli, "_write_json_report", write_report_then_open_bundle)
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
            str(evidence),
            "--model-manifest",
            str(manifest),
            "--carbon-model-dir",
            str(tmp_path / "carbon"),
            "--config",
            str(config),
            "--created-at-ns",
            "1750000000000000000",
            "--hardware",
            "fixture CPU",
            "--json-report",
            str(external_report),
        ],
    )
    captured = capsys.readouterr()

    assert rc == 4
    assert "unexpected artifact" in captured.err
    assert external_report.is_file()
    assert (evidence / "SHA256SUMS").is_file()
