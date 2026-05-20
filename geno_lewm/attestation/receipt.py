"""Receipt writer / reader (RFC-0011 §3.3).

Receipt schema v1.0.0. The on-disk format is canonical JSON so two
identical Python ``Receipt`` objects produce byte-stable disk content;
the loader rejects malformed receipts and unknown top-level keys.

The ``attestation.kind`` field is forward-compatible: ``checksum_only``
is the v1 default. ``tee`` and ``stark`` kinds will land in later
issues; the schema accepts them but the verifier (#77) emits a
clear "unverified attestation kind" message for anything it doesn't
support yet.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from geno_lewm.attestation.hashing import (
    canonical_json_bytes,
    canonical_json_sha256,
    looks_like_sha256,
)
from geno_lewm.errors import InputError, ReceiptSchemaError, SchemaCompatError

__all__ = [
    "RECEIPT_SCHEMA_VERSION",
    "SUPPORTED_ATTESTATION_KINDS",
    "ReceiptOutput",
    "ReceiptRuntime",
    "ReceiptAttestation",
    "Receipt",
    "write_receipt",
    "read_receipt",
    "compute_output_commitment",
]


RECEIPT_SCHEMA_VERSION: str = "1.0.0"

#: Attestation kinds the verifier knows about today. ``tee`` and
#: ``stark`` are accepted in the schema for forward compatibility, but
#: a checksum-only verifier (#77) emits an explicit unsupported-kind
#: message rather than failing.
SUPPORTED_ATTESTATION_KINDS: frozenset[str] = frozenset(
    {"checksum_only", "tee", "stark"}
)


def _require_hash(name: str, value: str) -> None:
    if not looks_like_sha256(value):
        raise InputError(
            f"{name} must be 'sha256:<64hex>'",
            details={"field": name, "value": value},
        )


@dataclass(frozen=True, slots=True)
class ReceiptOutput:
    """Score-call output committed by the receipt."""

    sigma_raw: float
    sigma_calibrated: float
    bucket_id: str
    confidence: float
    low_confidence: bool

    def __post_init__(self) -> None:
        for name in ("sigma_raw", "sigma_calibrated", "confidence"):
            v = getattr(self, name)
            if not isinstance(v, int | float) or isinstance(v, bool):
                raise InputError(
                    f"{name} must be float", details={"name": name, "value": v}
                )
        if not self.bucket_id:
            raise InputError("bucket_id must be non-empty", details={"bucket_id": self.bucket_id})


@dataclass(frozen=True, slots=True)
class ReceiptRuntime:
    """Runtime / environment block."""

    backend: str
    device: str
    geno_lewm_version: str
    carbon_revision: str

    def __post_init__(self) -> None:
        for name in ("backend", "device", "geno_lewm_version", "carbon_revision"):
            v = getattr(self, name)
            if not v:
                raise InputError(
                    f"runtime.{name} must be non-empty", details={name: v}
                )


@dataclass(frozen=True, slots=True)
class ReceiptAttestation:
    """``attestation`` block; ``details`` is forward-compatible."""

    kind: str
    details: dict[str, Any] | None = None

    def __post_init__(self) -> None:
        if not self.kind:
            raise InputError("attestation.kind must be non-empty", details={"kind": self.kind})
        # The schema knows about the v1 + planned kinds; anything else
        # is rejected at write time so we don't ship receipts the
        # verifier cannot reason about. The verifier's own
        # "unsupported kind" path covers schema-valid kinds that lack
        # a verification implementation yet.
        if self.kind not in SUPPORTED_ATTESTATION_KINDS:
            raise InputError(
                "attestation.kind is not a recognized value",
                details={
                    "kind": self.kind,
                    "supported": sorted(SUPPORTED_ATTESTATION_KINDS),
                },
            )


@dataclass(frozen=True, slots=True)
class Receipt:
    """Top-level receipt (RFC-0011 §3.3)."""

    schema_version: str
    model_id: str
    input_commitment: str
    output: ReceiptOutput
    output_commitment: str
    calibration_hash: str
    runtime: ReceiptRuntime
    timestamp: str
    attestation: ReceiptAttestation

    def __post_init__(self) -> None:
        if self.schema_version != RECEIPT_SCHEMA_VERSION:
            raise SchemaCompatError(
                "receipt schema_version mismatch",
                details={"got": self.schema_version, "expected": RECEIPT_SCHEMA_VERSION},
            )
        _require_hash("model_id", self.model_id)
        _require_hash("input_commitment", self.input_commitment)
        _require_hash("output_commitment", self.output_commitment)
        _require_hash("calibration_hash", self.calibration_hash)
        if not self.timestamp:
            raise InputError(
                "timestamp must be non-empty (ISO-8601 UTC)",
                details={"timestamp": self.timestamp},
            )

    def to_canonical_dict(self) -> dict[str, Any]:
        return asdict(self)

    def to_canonical_json(self) -> bytes:
        return canonical_json_bytes(self.to_canonical_dict())


def compute_output_commitment(output: ReceiptOutput) -> str:
    """Compute the output-commitment hash for an output block.

    Separated from ``Receipt`` so callers can pre-compute the
    commitment before assembling the receipt.
    """
    return canonical_json_sha256(asdict(output))


# ---------------------------------------------------------------------------
# Disk I/O


def write_receipt(receipt: Receipt, path: str | Path) -> Path:
    """Write a receipt as canonical JSON; round-trip byte-stable."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_bytes(receipt.to_canonical_json())
    return p


_REQUIRED_TOP = {
    "schema_version",
    "model_id",
    "input_commitment",
    "output",
    "output_commitment",
    "calibration_hash",
    "runtime",
    "timestamp",
    "attestation",
}
_REQUIRED_OUTPUT = {"sigma_raw", "sigma_calibrated", "bucket_id", "confidence", "low_confidence"}
_REQUIRED_RUNTIME = {"backend", "device", "geno_lewm_version", "carbon_revision"}


def _require_keys(name: str, d: dict[str, Any], required: set[str]) -> None:
    missing = required - set(d)
    extra = set(d) - required
    if missing or extra:
        raise ReceiptSchemaError(
            f"{name} has invalid key set",
            details={"missing": sorted(missing), "extra": sorted(extra)},
        )


def read_receipt(path: str | Path) -> Receipt:
    """Load and validate a receipt from disk."""
    p = Path(path)
    raw = p.read_bytes()
    try:
        d = json.loads(raw.decode("utf-8"))
    except json.JSONDecodeError as exc:
        raise ReceiptSchemaError(
            "receipt is not valid JSON",
            details={"path": str(p), "error": str(exc)},
        ) from exc

    if not isinstance(d, dict):
        raise ReceiptSchemaError(
            "receipt top-level must be an object",
            details={"path": str(p), "type": type(d).__name__},
        )

    _require_keys("receipt", d, _REQUIRED_TOP)

    out = d["output"]
    if not isinstance(out, dict):
        raise ReceiptSchemaError("output must be an object", details={"got": type(out).__name__})
    _require_keys("output", out, _REQUIRED_OUTPUT)

    rt = d["runtime"]
    if not isinstance(rt, dict):
        raise ReceiptSchemaError("runtime must be an object", details={"got": type(rt).__name__})
    _require_keys("runtime", rt, _REQUIRED_RUNTIME)

    att = d["attestation"]
    if not isinstance(att, dict) or set(att) - {"kind", "details"} or "kind" not in att:
        raise ReceiptSchemaError(
            "attestation must be an object with 'kind' (and optional 'details')",
            details={"attestation": att},
        )

    return Receipt(
        schema_version=d["schema_version"],
        model_id=d["model_id"],
        input_commitment=d["input_commitment"],
        output=ReceiptOutput(
            sigma_raw=out["sigma_raw"],
            sigma_calibrated=out["sigma_calibrated"],
            bucket_id=out["bucket_id"],
            confidence=out["confidence"],
            low_confidence=out["low_confidence"],
        ),
        output_commitment=d["output_commitment"],
        calibration_hash=d["calibration_hash"],
        runtime=ReceiptRuntime(
            backend=rt["backend"],
            device=rt["device"],
            geno_lewm_version=rt["geno_lewm_version"],
            carbon_revision=rt["carbon_revision"],
        ),
        timestamp=d["timestamp"],
        attestation=ReceiptAttestation(
            kind=att["kind"],
            details=att.get("details"),
        ),
    )
