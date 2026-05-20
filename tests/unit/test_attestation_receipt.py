"""Tests for ``geno_lewm.attestation.receipt``."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from geno_lewm.attestation import (
    RECEIPT_SCHEMA_VERSION,
    Receipt,
    ReceiptAttestation,
    ReceiptOutput,
    ReceiptRuntime,
    compute_output_commitment,
    read_receipt,
    write_receipt,
)
from geno_lewm.errors import InputError, ReceiptSchemaError, SchemaCompatError


_SHA = "sha256:" + "a" * 64
_SHA_B = "sha256:" + "b" * 64
_SHA_C = "sha256:" + "c" * 64
_SHA_D = "sha256:" + "d" * 64


def _output() -> ReceiptOutput:
    return ReceiptOutput(
        sigma_raw=0.347,
        sigma_calibrated=0.92,
        bucket_id="coding_missense|mid|none",
        confidence=1.0,
        low_confidence=False,
    )


def _runtime() -> ReceiptRuntime:
    return ReceiptRuntime(
        backend="coreml",
        device="Apple M3 Max",
        geno_lewm_version="0.1.0",
        carbon_revision="main@deadbeef",
    )


def _receipt() -> Receipt:
    out = _output()
    return Receipt(
        schema_version=RECEIPT_SCHEMA_VERSION,
        model_id=_SHA,
        input_commitment=_SHA_B,
        output=out,
        output_commitment=compute_output_commitment(out),
        calibration_hash=_SHA_C,
        runtime=_runtime(),
        timestamp="2026-05-20T12:34:56Z",
        attestation=ReceiptAttestation(kind="checksum_only"),
    )


# ---------------------------------------------------------------------------
# Construction validation.


def test_minimal_receipt_constructs() -> None:
    r = _receipt()
    assert r.schema_version == RECEIPT_SCHEMA_VERSION
    assert r.attestation.kind == "checksum_only"


def test_wrong_schema_version_rejected() -> None:
    with pytest.raises(SchemaCompatError):
        Receipt(
            schema_version="2.0.0",
            model_id=_SHA,
            input_commitment=_SHA_B,
            output=_output(),
            output_commitment=_SHA_D,
            calibration_hash=_SHA_C,
            runtime=_runtime(),
            timestamp="2026-05-20T12:34:56Z",
            attestation=ReceiptAttestation(kind="checksum_only"),
        )


def test_bad_hash_field_rejected() -> None:
    with pytest.raises(InputError):
        Receipt(
            schema_version=RECEIPT_SCHEMA_VERSION,
            model_id="not-a-hash",
            input_commitment=_SHA_B,
            output=_output(),
            output_commitment=_SHA_D,
            calibration_hash=_SHA_C,
            runtime=_runtime(),
            timestamp="t",
            attestation=ReceiptAttestation(kind="checksum_only"),
        )


def test_unknown_attestation_kind_rejected() -> None:
    with pytest.raises(InputError):
        ReceiptAttestation(kind="not-a-real-kind")


def test_known_forward_compatible_kinds_accepted() -> None:
    # tee / stark are accepted at the schema level today even though
    # the verifier (#77) will treat them as "unsupported kind".
    ReceiptAttestation(kind="tee")
    ReceiptAttestation(kind="stark")


def test_output_validates_types() -> None:
    with pytest.raises(InputError):
        ReceiptOutput(
            sigma_raw="x",  # type: ignore[arg-type]
            sigma_calibrated=0.0,
            bucket_id="b",
            confidence=1.0,
            low_confidence=False,
        )
    with pytest.raises(InputError):
        ReceiptOutput(
            sigma_raw=0.0,
            sigma_calibrated=0.0,
            bucket_id="",
            confidence=1.0,
            low_confidence=False,
        )


def test_runtime_validates_non_empty() -> None:
    with pytest.raises(InputError):
        ReceiptRuntime(backend="", device="d", geno_lewm_version="v", carbon_revision="r")


# ---------------------------------------------------------------------------
# Round-trip byte stability.


def test_round_trip_returns_equal_receipt(tmp_path: Path) -> None:
    r = _receipt()
    path = write_receipt(r, tmp_path / "receipt.json")
    loaded = read_receipt(path)
    assert loaded == r


def test_round_trip_is_byte_stable(tmp_path: Path) -> None:
    r = _receipt()
    a = write_receipt(r, tmp_path / "a.json").read_bytes()
    b = write_receipt(r, tmp_path / "b.json").read_bytes()
    assert a == b


def test_canonical_json_key_order_is_stable(tmp_path: Path) -> None:
    r = _receipt()
    raw = write_receipt(r, tmp_path / "r.json").read_text()
    # Top-level keys must appear in sorted order (canonical JSON
    # guarantee). Parse positions of a few known keys.
    pos_attest = raw.index('"attestation"')
    pos_calib = raw.index('"calibration_hash"')
    pos_model = raw.index('"model_id"')
    pos_schema = raw.index('"schema_version"')
    assert pos_attest < pos_calib < pos_model < pos_schema


# ---------------------------------------------------------------------------
# Loader rejects malformed receipts.


def test_loader_rejects_missing_top_level_key(tmp_path: Path) -> None:
    raw = json.loads(write_receipt(_receipt(), tmp_path / "r.json").read_text())
    del raw["model_id"]
    bad = tmp_path / "bad.json"
    bad.write_text(json.dumps(raw), encoding="utf-8")
    with pytest.raises(ReceiptSchemaError):
        read_receipt(bad)


def test_loader_rejects_unknown_top_level_key(tmp_path: Path) -> None:
    raw = json.loads(write_receipt(_receipt(), tmp_path / "r.json").read_text())
    raw["bogus"] = 1
    bad = tmp_path / "bad.json"
    bad.write_text(json.dumps(raw), encoding="utf-8")
    with pytest.raises(ReceiptSchemaError):
        read_receipt(bad)


def test_loader_rejects_bad_output_shape(tmp_path: Path) -> None:
    raw = json.loads(write_receipt(_receipt(), tmp_path / "r.json").read_text())
    raw["output"] = "not-a-dict"
    bad = tmp_path / "bad.json"
    bad.write_text(json.dumps(raw), encoding="utf-8")
    with pytest.raises(ReceiptSchemaError):
        read_receipt(bad)


def test_loader_rejects_missing_output_key(tmp_path: Path) -> None:
    raw = json.loads(write_receipt(_receipt(), tmp_path / "r.json").read_text())
    del raw["output"]["confidence"]
    bad = tmp_path / "bad.json"
    bad.write_text(json.dumps(raw), encoding="utf-8")
    with pytest.raises(ReceiptSchemaError):
        read_receipt(bad)


def test_loader_rejects_unknown_runtime_key(tmp_path: Path) -> None:
    raw = json.loads(write_receipt(_receipt(), tmp_path / "r.json").read_text())
    raw["runtime"]["extra"] = "no"
    bad = tmp_path / "bad.json"
    bad.write_text(json.dumps(raw), encoding="utf-8")
    with pytest.raises(ReceiptSchemaError):
        read_receipt(bad)


def test_loader_rejects_invalid_json(tmp_path: Path) -> None:
    bad = tmp_path / "bad.json"
    bad.write_text("{not json", encoding="utf-8")
    with pytest.raises(ReceiptSchemaError):
        read_receipt(bad)


def test_loader_rejects_non_object_top_level(tmp_path: Path) -> None:
    bad = tmp_path / "bad.json"
    bad.write_text("[1,2,3]", encoding="utf-8")
    with pytest.raises(ReceiptSchemaError):
        read_receipt(bad)


def test_loader_rejects_bad_attestation_shape(tmp_path: Path) -> None:
    raw = json.loads(write_receipt(_receipt(), tmp_path / "r.json").read_text())
    raw["attestation"] = "checksum_only"
    bad = tmp_path / "bad.json"
    bad.write_text(json.dumps(raw), encoding="utf-8")
    with pytest.raises(ReceiptSchemaError):
        read_receipt(bad)


def test_compute_output_commitment_is_deterministic() -> None:
    a = compute_output_commitment(_output())
    b = compute_output_commitment(_output())
    assert a == b
    assert a.startswith("sha256:")
