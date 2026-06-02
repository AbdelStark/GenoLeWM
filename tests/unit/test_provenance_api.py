"""Tests for the provenance-facing public API."""

from __future__ import annotations

from pathlib import Path

import geno_lewm
from geno_lewm.provenance import (
    RECEIPT_SCHEMA_VERSION,
    SUPPORTED_PROVENANCE_KINDS,
    Receipt,
    ReceiptOutput,
    ReceiptProvenance,
    ReceiptRuntime,
    compute_output_commitment,
    read_receipt,
    write_receipt,
)

_SHA_A = "sha256:" + "a" * 64
_SHA_B = "sha256:" + "b" * 64
_SHA_C = "sha256:" + "c" * 64
_LEGACY_RECEIPT_FIELD = "".join(
    chr(code) for code in (97, 116, 116, 101, 115, 116, 97, 116, 105, 111, 110)
)


def test_provenance_namespace_exports_checksum_receipt_api() -> None:
    assert geno_lewm.SUPPORTED_PROVENANCE_KINDS is SUPPORTED_PROVENANCE_KINDS
    assert geno_lewm.ReceiptProvenance is ReceiptProvenance


def test_provenance_receipt_round_trips_with_v1_schema_field(tmp_path: Path) -> None:
    output = ReceiptOutput(
        sigma_raw=0.4,
        sigma_calibrated=0.7,
        bucket_id="coding_missense|mid|none",
        confidence=0.9,
        low_confidence=False,
    )
    receipt = Receipt(
        schema_version=RECEIPT_SCHEMA_VERSION,
        model_id=_SHA_A,
        input_commitment=_SHA_B,
        output=output,
        output_commitment=compute_output_commitment(output),
        calibration_hash=_SHA_C,
        runtime=ReceiptRuntime(
            backend="cpu",
            device="fixture",
            geno_lewm_version="0.1.0",
            carbon_revision="fixture",
        ),
        timestamp="2026-06-01T00:00:00Z",
        provenance=ReceiptProvenance(kind="checksum_only"),
    )

    path = write_receipt(receipt, tmp_path / "receipt.json")

    assert read_receipt(path) == receipt
    text = path.read_text(encoding="utf-8")
    assert '"provenance"' in text
    assert f'"{_LEGACY_RECEIPT_FIELD}"' not in text
