# SPDX-License-Identifier: Apache-2.0
"""Build a release report for VCF score and receipt JSONL artifacts."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Final

from geno_lewm.errors import GenoLeWMError, InputError, exit_code_for
from geno_lewm.provenance import (
    RECEIPT_SCHEMA_VERSION,
    Receipt,
    compute_output_commitment,
    parse_receipt_payload,
    sha256_file,
)

REPORT_NAME: Final = "batch_receipt_report.json"
SCHEMA_VERSION: Final = "1.0.0"
GENERATED_BY: Final = "tools.release.batch_receipt_report"
RECEIPT_STREAM: Final = "jsonl_per_scored_alternate_v1"
CHECKED_SCORE_FIELDS: Final = (
    "sigma_raw",
    "sigma_calibrated",
    "bucket_id",
    "confidence",
    "low_confidence",
)


@dataclass(frozen=True, slots=True)
class BatchArtifact:
    """File identity for one demo batch artifact."""

    path: str
    sha256: str
    size_bytes: int
    jsonl_rows: int

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class BatchReceiptReport:
    """Aggregate verification evidence for a VCF score batch."""

    schema_version: str
    generated_by: str
    generated_at: str
    model_id: str
    calibration_hash: str
    receipt_schema_version: str
    receipt_stream: str
    records: int
    runtime: dict[str, object]
    scores: BatchArtifact
    receipts: BatchArtifact
    checked_score_fields: tuple[str, ...]

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "generated_by": self.generated_by,
            "generated_at": self.generated_at,
            "model_id": self.model_id,
            "calibration_hash": self.calibration_hash,
            "receipt_schema_version": self.receipt_schema_version,
            "receipt_stream": self.receipt_stream,
            "records": self.records,
            "runtime": self.runtime,
            "scores": self.scores.to_dict(),
            "receipts": self.receipts.to_dict(),
            "checked_score_fields": list(self.checked_score_fields),
        }


def build_batch_receipt_report(
    scores_jsonl: str | Path,
    receipts_jsonl: str | Path,
    *,
    generated_at: str | None = None,
) -> BatchReceiptReport:
    """Verify score/receipt JSONL streams and return an aggregate report."""
    scores_path = Path(scores_jsonl)
    receipts_path = Path(receipts_jsonl)
    scores = _load_jsonl_objects(scores_path, label="scores")
    receipts = _load_receipt_jsonl(receipts_path)
    if len(scores) != len(receipts):
        raise InputError(
            "score and receipt JSONL row counts differ",
            details={"scores": len(scores), "receipts": len(receipts)},
        )
    if not scores:
        raise InputError("score and receipt JSONL artifacts must contain at least one row")
    _verify_receipts(scores, receipts)
    first = receipts[0]
    return BatchReceiptReport(
        schema_version=SCHEMA_VERSION,
        generated_by=GENERATED_BY,
        generated_at=generated_at or _utc_now(),
        model_id=first.model_id,
        calibration_hash=first.calibration_hash,
        receipt_schema_version=RECEIPT_SCHEMA_VERSION,
        receipt_stream=RECEIPT_STREAM,
        records=len(scores),
        runtime=asdict(first.runtime),
        scores=_artifact(scores_path, rows=len(scores)),
        receipts=_artifact(receipts_path, rows=len(receipts)),
        checked_score_fields=CHECKED_SCORE_FIELDS,
    )


def write_batch_receipt_report(
    scores_jsonl: str | Path,
    receipts_jsonl: str | Path,
    output: str | Path,
    *,
    generated_at: str | None = None,
) -> Path:
    """Build and write ``batch_receipt_report.json``."""
    report = build_batch_receipt_report(
        scores_jsonl,
        receipts_jsonl,
        generated_at=generated_at,
    )
    path = Path(output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report.to_dict(), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def main(argv: list[str] | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    try:
        output = write_batch_receipt_report(
            args.scores_jsonl,
            args.receipts_jsonl,
            args.output,
        )
    except GenoLeWMError as exc:
        sys.stderr.write(f"error: {exc}\n")
        return exit_code_for(exc)
    sys.stdout.write(f"wrote {output}\n")
    return 0


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build batch_receipt_report.json from score and receipt JSONL artifacts.",
    )
    parser.add_argument("--scores-jsonl", type=Path, required=True)
    parser.add_argument("--receipts-jsonl", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def _load_jsonl_objects(path: Path, *, label: str) -> tuple[dict[str, Any], ...]:
    if not path.is_file():
        raise InputError(f"{label} JSONL artifact is missing", details={"path": str(path)})
    rows: list[dict[str, Any]] = []
    try:
        with path.open("r", encoding="utf-8") as handle:
            for line_no, raw in enumerate(handle, start=1):
                line = raw.strip()
                if not line:
                    continue
                try:
                    payload = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise InputError(
                        f"{label} JSONL artifact must contain JSON objects",
                        details={"path": str(path), "line": line_no},
                    ) from exc
                if not isinstance(payload, dict):
                    raise InputError(
                        f"{label} JSONL rows must be objects",
                        details={"path": str(path), "line": line_no},
                    )
                rows.append(payload)
    except OSError as exc:
        raise InputError(
            f"failed to read {label} JSONL artifact", details={"path": str(path)}
        ) from exc
    if not rows:
        raise InputError(f"{label} JSONL artifact must contain at least one row")
    return tuple(rows)


def _load_receipt_jsonl(path: Path) -> tuple[Receipt, ...]:
    payloads = _load_jsonl_objects(path, label="receipts")
    receipts: list[Receipt] = []
    for index, payload in enumerate(payloads, start=1):
        try:
            receipts.append(parse_receipt_payload(payload))
        except GenoLeWMError as exc:
            raise InputError(
                "receipt JSONL row is not a valid v1 receipt",
                details={"path": str(path), "row": index, "error": exc.message or str(exc)},
            ) from exc
    return tuple(receipts)


def _verify_receipts(
    scores: tuple[dict[str, Any], ...],
    receipts: tuple[Receipt, ...],
) -> None:
    first = receipts[0]
    expected_runtime = first.runtime
    for index, (score, receipt) in enumerate(zip(scores, receipts, strict=True), start=1):
        if receipt.model_id != first.model_id:
            raise InputError(
                "receipt stream contains multiple model ids",
                details={"row": index, "expected": first.model_id, "observed": receipt.model_id},
            )
        if receipt.calibration_hash != first.calibration_hash:
            raise InputError(
                "receipt stream contains multiple calibration hashes",
                details={
                    "row": index,
                    "expected": first.calibration_hash,
                    "observed": receipt.calibration_hash,
                },
            )
        if receipt.runtime != expected_runtime:
            raise InputError(
                "receipt stream contains multiple runtime identities", details={"row": index}
            )
        if receipt.output_commitment != compute_output_commitment(receipt.output):
            raise InputError(
                "receipt output commitment does not match receipt output",
                details={"row": index},
            )
        details = receipt.provenance.details or {}
        if details.get("scope") != "vcf_row":
            raise InputError("receipt row scope must be vcf_row", details={"row": index})
        if details.get("receipt_stream") != RECEIPT_STREAM:
            raise InputError(
                "receipt row stream marker is missing or unsupported",
                details={"row": index, "expected": RECEIPT_STREAM},
            )
        if details.get("row_index") != index:
            raise InputError(
                "receipt row_index does not match JSONL order",
                details={"row": index, "observed": details.get("row_index")},
            )
        _verify_score_matches_receipt(score, receipt, row=index)


def _verify_score_matches_receipt(score: dict[str, Any], receipt: Receipt, *, row: int) -> None:
    output = receipt.output
    expected = {
        "sigma_raw": output.sigma_raw,
        "sigma_calibrated": output.sigma_calibrated,
        "bucket_id": output.bucket_id,
        "confidence": output.confidence,
        "low_confidence": output.low_confidence,
    }
    for key, expected_value in expected.items():
        observed = score.get(key)
        if isinstance(expected_value, float):
            if isinstance(observed, bool) or not isinstance(observed, int | float):
                raise InputError(
                    "score row is missing a numeric receipt output field",
                    details={"row": row, "field": key},
                )
            if float(observed) != expected_value:
                raise InputError(
                    "score row value does not match receipt output",
                    details={
                        "row": row,
                        "field": key,
                        "expected": expected_value,
                        "observed": observed,
                    },
                )
        elif observed != expected_value:
            raise InputError(
                "score row value does not match receipt output",
                details={
                    "row": row,
                    "field": key,
                    "expected": expected_value,
                    "observed": observed,
                },
            )


def _artifact(path: Path, *, rows: int) -> BatchArtifact:
    return BatchArtifact(
        path=path.name,
        sha256=sha256_file(path),
        size_bytes=path.stat().st_size,
        jsonl_rows=rows,
    )


def _utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
