"""Tests for the terminal-demo transcript generator."""

from __future__ import annotations

import json
import subprocess
from collections.abc import Sequence
from datetime import datetime, timezone
from pathlib import Path

import pytest

from geno_lewm._artifact_sources import SCORE_JSONL_GENERATED_BY, SCORE_JSONL_SCHEMA_VERSION
from geno_lewm.errors import InputError
from geno_lewm.provenance import (
    SCHEMA_VERSION,
    Manifest,
    ManifestArtifact,
    ManifestEncoder,
    ManifestTraining,
    Receipt,
    ReceiptOutput,
    ReceiptProvenance,
    ReceiptRuntime,
    compute_output_commitment,
    load_manifest,
    sha256_bytes,
    sha256_file,
    write_manifest,
)
from tools.demo.terminal_inference import (
    DEMO_MANIFEST_NAME,
    DemoRequest,
    build_score_command,
    run_demo_transcript,
)


def test_build_score_command_records_explicit_score_and_receipt_paths(tmp_path: Path) -> None:
    request = DemoRequest(
        model_dir=tmp_path / "model",
        vcf=tmp_path / "input.vcf",
        fasta=tmp_path / "ref.fa",
        output_dir=tmp_path / "demo",
        backend="cpu",
        batch_size=8,
    )

    command = build_score_command(request)

    assert command[:3] == ("geno-lewm-score", "--quiet", "--no-banner")
    assert "--output" in command
    assert str(request.scores_path) in command
    assert "--receipt" in command
    assert str(request.receipts_path) in command
    assert command[-1] == "--no-progress"


def test_demo_transcript_rejects_fixture_manifest_by_default(tmp_path: Path) -> None:
    model_dir = _write_model_dir(tmp_path / "model", release_id="geno-lewm-fixture-r1")
    vcf, fasta = _write_demo_inputs(tmp_path)
    request = DemoRequest(
        model_dir=model_dir,
        vcf=vcf,
        fasta=fasta,
        output_dir=tmp_path / "demo",
        require_native_runtime=False,
    )

    with pytest.raises(InputError, match="fixture/test manifests"):
        run_demo_transcript(request, runner=_successful_runner)


def test_demo_transcript_writes_command_output_and_manifest_identity(tmp_path: Path) -> None:
    model_dir = _write_model_dir(tmp_path / "model", release_id="geno-lewm-v0.1.0-r1")
    vcf, fasta = _write_demo_inputs(tmp_path)
    request = DemoRequest(
        model_dir=model_dir,
        vcf=vcf,
        fasta=fasta,
        output_dir=tmp_path / "demo",
        backend="cpu",
        require_native_runtime=False,
    )

    transcript = run_demo_transcript(
        request,
        runner=_successful_runner,
        now=datetime(2026, 6, 1, 12, 0, tzinfo=timezone.utc),
    )

    text = transcript.read_text(encoding="utf-8")
    manifest = _load_written_manifest(model_dir)
    assert "Status: passed" in text
    assert "geno-lewm-score --quiet --no-banner" in text
    assert '{"output_path":"scores.jsonl"}' in text
    assert f"Model release: {manifest.release_id}" in text
    assert f"Model id: {manifest.model_id()}" in text
    assert "Input VCF records: 1" in text
    assert "Input alternate alleles: 1" in text
    assert "Input contigs: 1" in text
    assert "First input variants: 1:10:A>T" in text
    assert str(tmp_path) not in text
    assert "- Scores: demo/scores.jsonl" in text
    assert "- Receipts: demo/receipts.jsonl" in text
    assert "- Runtime preflight report: demo/runtime_preflight_report.json" in text
    assert "- Batch receipt report: demo/batch_receipt_report.json" in text
    assert "- Demo manifest: demo/terminal_demo_manifest.json" in text
    assert "Scores SHA-256: sha256:" in text
    assert "Scores JSONL rows: 1" in text
    assert "Scores JSONL fields:" in text
    assert "chrom" in text
    assert "sigma_raw" in text
    assert "Receipts SHA-256: sha256:" in text
    assert "Receipts JSONL rows: 1" in text
    assert "Receipt stream: jsonl_per_scored_alternate_v1" in text
    assert f"Receipt model id: {manifest.model_id()}" in text
    assert "Checked score fields: sigma_raw, sigma_calibrated" in text
    assert "Runtime Preflight Report SHA-256: sha256:" in text
    assert "Batch Receipt Report SHA-256: sha256:" in text
    assert request.runtime_preflight_report_path.is_file()
    assert request.batch_receipt_report_path.is_file()
    manifest_payload = json.loads((request.output_dir / DEMO_MANIFEST_NAME).read_text())
    assert str(tmp_path) not in json.dumps(manifest_payload)
    assert manifest_payload["status"] == "passed"
    assert manifest_payload["model"]["model_id"] == manifest.model_id()
    assert manifest_payload["inputs"]["vcf_summary"] == {
        "format": "vcf",
        "variant_records": 1,
        "alternate_alleles": 1,
        "contigs": ["1"],
        "first_variants": [{"chrom": "1", "pos": 10, "ref": "A", "alts": ["T"]}],
    }
    assert {artifact["label"] for artifact in manifest_payload["artifacts"]} == {
        "scores",
        "receipts",
        "runtime preflight report",
        "batch receipt report",
        "terminal transcript",
    }
    assert manifest_payload["runtime_preflight"]["generated_by"] == (
        "tools.release.runtime_preflight"
    )
    assert manifest_payload["runtime_preflight"]["ok"] is True
    assert manifest_payload["runtime_preflight"]["model_id"] == manifest.model_id()
    assert (
        manifest_payload["runtime_preflight"]["command"]["argv"]
        == manifest_payload["command"]["argv"]
    )
    score_artifact = next(
        artifact for artifact in manifest_payload["artifacts"] if artifact["label"] == "scores"
    )
    assert set(score_artifact["jsonl_fields"]) >= {
        "schema_version",
        "generated_by",
        "chrom",
        "pos",
        "ref",
        "alt",
        "sigma_raw",
    }
    assert manifest_payload["score_receipt_batch"]["model_id"] == manifest.model_id()
    assert manifest_payload["score_receipt_batch"]["records"] == 1


def test_demo_transcript_rejects_mutated_runtime_preflight_report(tmp_path: Path) -> None:
    model_dir = _write_model_dir(tmp_path / "model", release_id="geno-lewm-v0.1.0-r1")
    vcf, fasta = _write_demo_inputs(tmp_path)
    request = DemoRequest(
        model_dir=model_dir,
        vcf=vcf,
        fasta=fasta,
        output_dir=tmp_path / "demo",
        backend="cpu",
        require_native_runtime=False,
    )

    with pytest.raises(InputError, match="artifact verification failed"):
        run_demo_transcript(request, runner=_mutating_runtime_preflight_runner)

    text = request.transcript_path.read_text(encoding="utf-8")
    assert "Status: failed" in text
    assert "terminal demo runtime preflight report is inconsistent" in text


def test_demo_transcript_rejects_missing_score_outputs(tmp_path: Path) -> None:
    model_dir = _write_model_dir(tmp_path / "model", release_id="geno-lewm-v0.1.0-r1")
    vcf, fasta = _write_demo_inputs(tmp_path)
    request = DemoRequest(
        model_dir=model_dir,
        vcf=vcf,
        fasta=fasta,
        output_dir=tmp_path / "demo",
        require_native_runtime=False,
    )

    with pytest.raises(InputError, match="artifact verification failed"):
        run_demo_transcript(request, runner=_successful_runner_without_outputs)

    text = request.transcript_path.read_text(encoding="utf-8")
    assert "Status: failed" in text
    assert "scores JSONL artifact is missing" in text


def test_demo_transcript_does_not_reuse_stale_score_outputs(tmp_path: Path) -> None:
    model_dir = _write_model_dir(tmp_path / "model", release_id="geno-lewm-v0.1.0-r1")
    vcf, fasta = _write_demo_inputs(tmp_path)
    request = DemoRequest(
        model_dir=model_dir,
        vcf=vcf,
        fasta=fasta,
        output_dir=tmp_path / "demo",
        require_native_runtime=False,
    )
    request.output_dir.mkdir()
    request.scores_path.write_text('{"stale":true}\n', encoding="utf-8")
    request.receipts_path.write_text('{"stale":true}\n', encoding="utf-8")

    with pytest.raises(InputError, match="artifact verification failed"):
        run_demo_transcript(request, runner=_successful_runner_without_outputs)

    assert not request.scores_path.exists()
    assert not request.receipts_path.exists()
    text = request.transcript_path.read_text(encoding="utf-8")
    assert "scores JSONL artifact is missing" in text


def test_demo_transcript_records_failure_before_raising(tmp_path: Path) -> None:
    model_dir = _write_model_dir(tmp_path / "model", release_id="geno-lewm-v0.1.0-r1")
    vcf, fasta = _write_demo_inputs(tmp_path)
    request = DemoRequest(
        model_dir=model_dir,
        vcf=vcf,
        fasta=fasta,
        output_dir=tmp_path / "demo",
        require_native_runtime=False,
    )

    with pytest.raises(InputError, match="command failed"):
        run_demo_transcript(request, runner=_failing_runner)

    text = request.transcript_path.read_text(encoding="utf-8")
    assert "Status: failed" in text
    assert "runtime missing" in text


def _successful_runner(command: Sequence[str]) -> subprocess.CompletedProcess[str]:
    assert "--receipt" in command
    model_dir = Path(command[command.index("--model-dir") + 1])
    output = Path(command[command.index("--output") + 1])
    receipt = Path(command[command.index("--receipt") + 1])
    manifest = load_manifest(model_dir / "manifest.json")
    output.parent.mkdir(parents=True, exist_ok=True)
    receipt.parent.mkdir(parents=True, exist_ok=True)
    receipt_output = ReceiptOutput(
        sigma_raw=0.1,
        sigma_calibrated=0.2,
        bucket_id="coding_missense|mid|none",
        confidence=0.9,
        low_confidence=False,
    )
    output.write_text(
        json.dumps(
            {
                "schema_version": SCORE_JSONL_SCHEMA_VERSION,
                "generated_by": SCORE_JSONL_GENERATED_BY,
                "chrom": "1",
                "pos": 10,
                "ref": "A",
                "alt": "T",
                "sigma_raw": receipt_output.sigma_raw,
                "sigma_calibrated": receipt_output.sigma_calibrated,
                "bucket_id": receipt_output.bucket_id,
                "confidence": receipt_output.confidence,
                "low_confidence": receipt_output.low_confidence,
            },
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    receipt.write_text(
        _receipt_json(
            model_id=manifest.model_id(),
            output=receipt_output,
            row_index=1,
        )
        + "\n",
        encoding="utf-8",
    )
    return subprocess.CompletedProcess(
        args=list(command),
        returncode=0,
        stdout='{"output_path":"scores.jsonl"}\n',
        stderr="",
    )


def _successful_runner_without_outputs(command: Sequence[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.CompletedProcess(
        args=list(command),
        returncode=0,
        stdout='{"output_path":"scores.jsonl"}\n',
        stderr="",
    )


def _mutating_runtime_preflight_runner(command: Sequence[str]) -> subprocess.CompletedProcess[str]:
    completed = _successful_runner(command)
    output = Path(command[command.index("--output") + 1])
    preflight = output.parent / "runtime_preflight_report.json"
    payload = json.loads(preflight.read_text(encoding="utf-8"))
    payload["ok"] = False
    payload["command"]["argv"][-1] = "999"
    preflight.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return completed


def _failing_runner(command: Sequence[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.CompletedProcess(
        args=list(command),
        returncode=2,
        stdout="",
        stderr="runtime missing\n",
    )


def _write_model_dir(root: Path, *, release_id: str) -> Path:
    root.mkdir(parents=True)
    (root / "predictor.safetensors").write_bytes(b"predictor")
    (root / "action_encoder.safetensors").write_bytes(b"action")
    (root / "calibration.parquet").write_bytes(b"calibration")
    (root / "train_config.yaml").write_text("seed: 0\n", encoding="utf-8")
    (root / "eval_report.md").write_text("# eval\n", encoding="utf-8")
    (root / "model_card.md").write_text("# Model Card\n", encoding="utf-8")
    manifest = Manifest(
        schema_version=SCHEMA_VERSION,
        model_name="geno-lewm",
        model_version="0.1.0",
        release_id=release_id,
        encoder=ManifestEncoder(
            id="HuggingFaceBio/Carbon-500M",
            revision="main",
            hash=sha256_bytes(b"encoder"),
        ),
        predictor=ManifestArtifact(
            file="predictor.safetensors",
            hash=sha256_file(root / "predictor.safetensors"),
            dtype="bf16",
        ),
        action_encoder=ManifestArtifact(
            file="action_encoder.safetensors",
            hash=sha256_file(root / "action_encoder.safetensors"),
            dtype="bf16",
        ),
        calibration=ManifestArtifact(
            file="calibration.parquet",
            hash=sha256_file(root / "calibration.parquet"),
            version="1.0.0",
        ),
        training=ManifestTraining(
            config_file="train_config.yaml",
            hash=sha256_file(root / "train_config.yaml"),
            data_snapshot={"snapshot": "carbon-slice-v1"},
        ),
        eval=ManifestArtifact(file="eval_report.md", hash=sha256_file(root / "eval_report.md")),
    )
    write_manifest(manifest, root / "manifest.json")
    return root


def _write_demo_inputs(root: Path) -> tuple[Path, Path]:
    vcf = root / "input.vcf"
    fasta = root / "ref.fa"
    vcf.write_text(
        "##fileformat=VCFv4.2\n"
        "#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\n"
        "1\t10\t.\tA\tT\t.\tPASS\t.\n",
        encoding="utf-8",
    )
    fasta.write_text(">1\nAAAAAAAAAAAAAAAAAAAA\n", encoding="utf-8")
    return vcf, fasta


def _load_written_manifest(model_dir: Path) -> Manifest:
    return load_manifest(model_dir / "manifest.json")


def _receipt_json(*, model_id: str, output: ReceiptOutput, row_index: int) -> str:
    receipt = Receipt(
        schema_version="1.0.0",
        model_id=model_id,
        input_commitment="sha256:" + f"{row_index}".zfill(64),
        output=output,
        output_commitment=compute_output_commitment(output),
        calibration_hash="sha256:" + "c" * 64,
        runtime=ReceiptRuntime(
            backend="cpu",
            device="CPU",
            geno_lewm_version="0.1.0",
            carbon_revision="main",
        ),
        timestamp=f"2026-06-01T12:00:0{row_index}Z",
        provenance=ReceiptProvenance(
            kind="checksum_only",
            details={
                "scope": "vcf_row",
                "receipt_stream": "jsonl_per_scored_alternate_v1",
                "row_index": row_index,
            },
        ),
    )
    return receipt.to_canonical_json().decode("utf-8")
