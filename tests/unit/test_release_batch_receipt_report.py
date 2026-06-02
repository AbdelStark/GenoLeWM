"""Tests for the demo batch receipt report generator."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from geno_lewm.errors import InputError
from geno_lewm.provenance import (
    RECEIPT_SCHEMA_VERSION,
    Receipt,
    ReceiptOutput,
    ReceiptProvenance,
    ReceiptRuntime,
    compute_output_commitment,
)
from tools.release.batch_receipt_report import (
    RECEIPT_STREAM,
    build_batch_receipt_report,
    main,
)

_MODEL_ID = "sha256:" + "a" * 64
_CALIBRATION_HASH = "sha256:" + "b" * 64


def test_build_batch_receipt_report_verifies_score_and_receipt_streams(tmp_path: Path) -> None:
    scores, receipts = _write_score_and_receipt_jsonl(tmp_path)

    report = build_batch_receipt_report(
        scores,
        receipts,
        generated_at="2026-06-01T12:00:00Z",
    )
    payload = report.to_dict()

    assert payload["schema_version"] == "1.0.0"
    assert payload["generated_by"] == "tools.release.batch_receipt_report"
    assert payload["model_id"] == _MODEL_ID
    assert payload["calibration_hash"] == _CALIBRATION_HASH
    assert payload["receipt_schema_version"] == RECEIPT_SCHEMA_VERSION
    assert payload["receipt_stream"] == RECEIPT_STREAM
    assert payload["records"] == 2
    assert payload["checked_score_fields"] == [
        "sigma_raw",
        "sigma_calibrated",
        "bucket_id",
        "confidence",
        "low_confidence",
    ]
    assert payload["scores"]["jsonl_rows"] == 2
    assert payload["receipts"]["jsonl_rows"] == 2


def test_batch_receipt_report_main_writes_json(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    scores, receipts = _write_score_and_receipt_jsonl(tmp_path)
    output = tmp_path / "batch_receipt_report.json"

    rc = main(
        [
            "--scores-jsonl",
            str(scores),
            "--receipts-jsonl",
            str(receipts),
            "--output",
            str(output),
        ]
    )
    captured = capsys.readouterr()

    assert rc == 0
    assert captured.out == f"wrote {output}\n"
    assert json.loads(output.read_text(encoding="utf-8"))["records"] == 2


def test_batch_receipt_report_rejects_row_count_mismatch(tmp_path: Path) -> None:
    scores, receipts = _write_score_and_receipt_jsonl(tmp_path)
    receipts.write_text(
        receipts.read_text(encoding="utf-8").splitlines()[0] + "\n", encoding="utf-8"
    )

    with pytest.raises(InputError, match="row counts differ"):
        build_batch_receipt_report(scores, receipts)


def test_batch_receipt_report_rejects_score_output_mismatch(tmp_path: Path) -> None:
    scores, receipts = _write_score_and_receipt_jsonl(tmp_path)
    rows = [json.loads(line) for line in scores.read_text(encoding="utf-8").splitlines()]
    rows[0]["sigma_calibrated"] = 0.123
    scores.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows), encoding="utf-8"
    )

    with pytest.raises(InputError, match="does not match receipt output"):
        build_batch_receipt_report(scores, receipts)


def _write_score_and_receipt_jsonl(root: Path) -> tuple[Path, Path]:
    scores_path = root / "scores.jsonl"
    receipts_path = root / "receipts.jsonl"
    score_rows: list[dict[str, object]] = []
    receipt_rows: list[str] = []
    for row_index, sigma in enumerate((0.25, 0.75), start=1):
        output = ReceiptOutput(
            sigma_raw=sigma,
            sigma_calibrated=sigma + 0.1,
            bucket_id="coding_missense|mid|none",
            confidence=0.9,
            low_confidence=False,
        )
        score_rows.append(
            {
                "chrom": "1",
                "pos": row_index,
                "ref": "A",
                "alt": "T",
                "sigma_raw": output.sigma_raw,
                "sigma_calibrated": output.sigma_calibrated,
                "bucket_id": output.bucket_id,
                "confidence": output.confidence,
                "low_confidence": output.low_confidence,
            }
        )
        receipt = Receipt(
            schema_version=RECEIPT_SCHEMA_VERSION,
            model_id=_MODEL_ID,
            input_commitment="sha256:" + f"{row_index}".zfill(64),
            output=output,
            output_commitment=compute_output_commitment(output),
            calibration_hash=_CALIBRATION_HASH,
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
                    "receipt_stream": RECEIPT_STREAM,
                    "row_index": row_index,
                },
            ),
        )
        receipt_rows.append(receipt.to_canonical_json().decode("utf-8"))
    scores_path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in score_rows),
        encoding="utf-8",
    )
    receipts_path.write_text("\n".join(receipt_rows) + "\n", encoding="utf-8")
    return scores_path, receipts_path
