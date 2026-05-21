"""Tests for the ``geno-lewm-verify`` CLI (checksum mode)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from geno_lewm.action import EditSpec
from geno_lewm.attestation import (
    RECEIPT_SCHEMA_VERSION,
    SCHEMA_VERSION,
    DtypeConfig,
    Manifest,
    ManifestArtifact,
    ManifestEncoder,
    ManifestTraining,
    PoolingConfig,
    Receipt,
    ReceiptAttestation,
    ReceiptOutput,
    ReceiptRuntime,
    compute_input_commitment,
    compute_output_commitment,
    write_manifest,
    write_receipt,
)
from geno_lewm.cli import verify as verify_cli

_HEX = "sha256:" + "0" * 64


def _make_pair(tmp_path: Path) -> tuple[Path, Path, Manifest, Receipt]:
    manifest = Manifest(
        schema_version=SCHEMA_VERSION,
        model_name="geno-lewm",
        model_version="0.1.0",
        release_id="geno-lewm-v0.1.0-carbon-500m-r1",
        encoder=ManifestEncoder(
            id="HuggingFaceBio/Carbon-500M",
            revision="main@deadbeef",
            hash=_HEX,
        ),
        predictor=ManifestArtifact(file="predictor.safetensors", hash=_HEX, dtype="bf16"),
        action_encoder=ManifestArtifact(file="action_encoder.safetensors", hash=_HEX, dtype="bf16"),
        calibration=ManifestArtifact(file="calibration.parquet", hash=_HEX, version="1.0.0"),
        training=ManifestTraining(config_file="train_config.yaml", hash=_HEX),
        eval=ManifestArtifact(file="eval_report.md", hash=_HEX),
    )
    mpath = write_manifest(manifest, tmp_path / "manifest.json")

    # Build a receipt whose model_id matches the manifest.
    output = ReceiptOutput(
        sigma_raw=0.123,
        sigma_calibrated=0.456,
        bucket_id="coding_missense|low|none",
        confidence=0.9,
        low_confidence=False,
    )

    edit = EditSpec(chrom="chr1", pos=100, ref="A", alt="T")
    pool = PoolingConfig(state_layer=12, pool_type="centered_mean", pool_radius=24, normalize=True)
    dtype = DtypeConfig(encoder_dtype="bf16", predictor_dtype="bf16")
    input_commitment = compute_input_commitment("ACGT" * 3072, edit, pool, dtype)

    receipt = Receipt(
        schema_version=RECEIPT_SCHEMA_VERSION,
        model_id=manifest.model_id(),
        input_commitment=input_commitment,
        output=output,
        output_commitment=compute_output_commitment(output),
        calibration_hash=_HEX,
        runtime=ReceiptRuntime(
            backend="coreml",
            device="Apple M3 Max",
            geno_lewm_version="0.1.0",
            carbon_revision="main@deadbeef",
        ),
        timestamp="2026-05-20T12:34:56Z",
        attestation=ReceiptAttestation(kind="checksum_only"),
    )
    rpath = write_receipt(receipt, tmp_path / "receipt.json")
    return mpath, rpath, manifest, receipt


# ---------------------------------------------------------------------------
# Happy path.


def test_valid_receipt_exit_zero(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    mpath, rpath, _m, _r = _make_pair(tmp_path)
    rc = verify_cli.main([str(rpath), "--manifest", str(mpath)])
    captured = capsys.readouterr()
    assert rc == 0
    assert "ok" in captured.out
    assert "model_id ok" in captured.out


def test_valid_receipt_with_rerun_flag_still_passes_checksum(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    mpath, rpath, _m, _r = _make_pair(tmp_path)
    rc = verify_cli.main([str(rpath), "--manifest", str(mpath), "--rerun"])
    captured = capsys.readouterr()
    assert rc == 0
    assert "--rerun: not yet implemented" in captured.out


# ---------------------------------------------------------------------------
# Tampering scenarios.


def test_tampered_manifest_exit_8(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    mpath, rpath, _m, _r = _make_pair(tmp_path)
    raw = json.loads(mpath.read_text())
    raw["model_version"] = "0.2.0-tampered"  # changes the hash
    mpath.write_text(json.dumps(raw, separators=(",", ":"), sort_keys=True), encoding="utf-8")
    rc = verify_cli.main([str(rpath), "--manifest", str(mpath)])
    err = capsys.readouterr().err
    assert rc == 8
    assert "ATTESTATION.MANIFEST_HASH_MISMATCH" in err


def test_tampered_output_exit_8(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    mpath, rpath, _m, _r = _make_pair(tmp_path)
    raw = json.loads(rpath.read_text())
    raw["output"]["sigma_calibrated"] = (
        0.999  # changes output_commitment, but receipt's stored one is unchanged
    )
    # Re-serialise canonical-style.
    rpath.write_text(
        json.dumps(raw, separators=(",", ":"), sort_keys=True),
        encoding="utf-8",
    )
    rc = verify_cli.main([str(rpath), "--manifest", str(mpath)])
    err = capsys.readouterr().err
    assert rc == 8
    assert "ATTESTATION.OUTPUT_COMMITMENT_MISMATCH" in err


def test_tampered_input_commitment_exit_8(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    mpath, rpath, _m, _r = _make_pair(tmp_path)
    rc = verify_cli.main(
        [
            str(rpath),
            "--manifest",
            str(mpath),
            "--input-window",
            "ACGT" * 3072,
            "--edit-chrom",
            "chr1",
            "--edit-pos",
            "100",
            "--edit-ref",
            "A",
            "--edit-alt",
            "G",  # NB: alt is wrong (G instead of T)
            "--state-layer",
            "12",
            "--pool-type",
            "centered_mean",
            "--pool-radius",
            "24",
            "--normalize",
            "--encoder-dtype",
            "bf16",
            "--predictor-dtype",
            "bf16",
        ]
    )
    err = capsys.readouterr().err
    assert rc == 8
    assert "ATTESTATION.INPUT_COMMITMENT_MISMATCH" in err


def test_input_recomputation_happy_path(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    mpath, rpath, _m, _r = _make_pair(tmp_path)
    rc = verify_cli.main(
        [
            str(rpath),
            "--manifest",
            str(mpath),
            "--input-window",
            "ACGT" * 3072,
            "--edit-chrom",
            "chr1",
            "--edit-pos",
            "100",
            "--edit-ref",
            "A",
            "--edit-alt",
            "T",
            "--state-layer",
            "12",
            "--pool-type",
            "centered_mean",
            "--pool-radius",
            "24",
            "--normalize",
            "--encoder-dtype",
            "bf16",
            "--predictor-dtype",
            "bf16",
        ]
    )
    assert rc == 0
    out = capsys.readouterr().out
    assert "input_commitment ok" in out


def test_partial_input_flags_rejected(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    mpath, rpath, _m, _r = _make_pair(tmp_path)
    rc = verify_cli.main(
        [
            str(rpath),
            "--manifest",
            str(mpath),
            "--input-window",
            "ACGT",  # but no edit flags
        ]
    )
    err = capsys.readouterr().err
    assert rc == 2
    assert "INPUT" in err


def test_unsupported_attestation_kind_exit_8(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    mpath, rpath, _m, _r = _make_pair(tmp_path)
    raw = json.loads(rpath.read_text())
    raw["attestation"] = {"kind": "tee", "details": {"vendor": "apple-secure-enclave"}}
    rpath.write_text(json.dumps(raw, separators=(",", ":"), sort_keys=True), encoding="utf-8")
    rc = verify_cli.main([str(rpath), "--manifest", str(mpath)])
    err = capsys.readouterr().err
    assert rc == 8
    assert "ATTESTATION.KIND_UNSUPPORTED" in err


def test_missing_receipt_file_exit_nonzero(tmp_path: Path) -> None:
    rc = verify_cli.main([str(tmp_path / "no-such.json"), "--manifest", str(tmp_path / "no.json")])
    assert rc != 0
