# SPDX-License-Identifier: Apache-2.0
"""Closed identity contract for a cache-building Carbon encoder runtime."""

from __future__ import annotations

import json
from dataclasses import dataclass

from geno_lewm.config._state_contract import (
    L2_NORMALIZED_STATE_CONTRACT,
    LEGACY_RAW_STATE_CONTRACT,
)
from geno_lewm.errors import InputError

__all__ = [
    "ENCODER_RUNTIME_IDENTITY_SCHEMA_VERSION",
    "SUPPORTED_ENCODER_STATE_CONTRACTS",
    "EncoderRuntimeIdentity",
    "parse_encoder_runtime_identity_bytes",
]

ENCODER_RUNTIME_IDENTITY_SCHEMA_VERSION = "1.0.0"
SUPPORTED_ENCODER_STATE_CONTRACTS = frozenset(
    {LEGACY_RAW_STATE_CONTRACT, L2_NORMALIZED_STATE_CONTRACT}
)
_REQUIRED_KEYS = frozenset(
    {
        "schema_version",
        "model_id",
        "revision",
        "state_contract_version",
        "runtime_hash",
    }
)
_OPTIONAL_KEYS = frozenset({"weights_hash"})


@dataclass(frozen=True, slots=True)
class EncoderRuntimeIdentity:
    """Pinned model and byte identities required for cache construction."""

    model_id: str
    revision: str
    state_contract_version: str
    runtime_hash: str
    weights_hash: str | None = None
    schema_version: str = ENCODER_RUNTIME_IDENTITY_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != ENCODER_RUNTIME_IDENTITY_SCHEMA_VERSION:
            raise InputError("encoder runtime identity has an unsupported schema_version")
        if not isinstance(self.model_id, str) or not self.model_id.strip():
            raise InputError("encoder runtime identity model_id must be non-empty text")
        if not isinstance(self.revision, str) or not self.revision.strip():
            raise InputError("encoder runtime identity revision must be exact non-empty text")
        if self.revision.casefold() in {"main", "master", "latest", "head"}:
            raise InputError(
                "encoder runtime identity revision must be immutable, not a floating ref"
            )
        if self.state_contract_version not in SUPPORTED_ENCODER_STATE_CONTRACTS:
            raise InputError(
                "encoder runtime identity has an unsupported state_contract_version",
                details={"state_contract_version": self.state_contract_version},
            )
        _validate_sha256(self.runtime_hash, field="runtime_hash")
        if self.weights_hash is not None:
            _validate_sha256(self.weights_hash, field="weights_hash")
        if self.state_contract_version == LEGACY_RAW_STATE_CONTRACT and self.weights_hash is None:
            raise InputError("legacy_raw_v1 encoder runtime identity requires weights_hash")

    @property
    def cache_identity_hash(self) -> str:
        """Return the identity committed into cache keys for this state contract."""
        if self.state_contract_version == LEGACY_RAW_STATE_CONTRACT:
            assert self.weights_hash is not None
            return self.weights_hash
        return self.runtime_hash

    def to_dict(self) -> dict[str, object]:
        """Return the canonical JSON-native contract payload."""
        payload: dict[str, object] = {
            "schema_version": self.schema_version,
            "model_id": self.model_id,
            "revision": self.revision,
            "state_contract_version": self.state_contract_version,
            "runtime_hash": self.runtime_hash,
        }
        if self.weights_hash is not None:
            payload["weights_hash"] = self.weights_hash
        return payload


def parse_encoder_runtime_identity_bytes(
    body: bytes,
    *,
    source: str,
) -> EncoderRuntimeIdentity:
    """Parse one duplicate-key-free, closed runtime identity JSON object."""
    try:
        payload = json.loads(body, object_pairs_hook=_reject_duplicate_keys)
    except (UnicodeDecodeError, json.JSONDecodeError, InputError) as exc:
        raise InputError(
            "encoder runtime identity is invalid JSON",
            details={"source": source, "error": str(exc)},
        ) from exc
    if type(payload) is not dict:
        raise InputError("encoder runtime identity must contain one JSON object")
    observed = frozenset(payload)
    if not _REQUIRED_KEYS.issubset(observed) or not observed.issubset(
        _REQUIRED_KEYS | _OPTIONAL_KEYS
    ):
        raise InputError(
            "encoder runtime identity has an invalid closed schema",
            details={
                "required": sorted(_REQUIRED_KEYS),
                "optional": sorted(_OPTIONAL_KEYS),
                "observed": sorted(observed),
            },
        )
    return EncoderRuntimeIdentity(
        schema_version=payload["schema_version"],
        model_id=payload["model_id"],
        revision=payload["revision"],
        state_contract_version=payload["state_contract_version"],
        runtime_hash=payload["runtime_hash"],
        weights_hash=payload.get("weights_hash"),
    )


def encoder_runtime_identity_from_mapping(raw: object) -> EncoderRuntimeIdentity:
    """Validate a JSON-native mapping through the same closed parser."""
    try:
        body = json.dumps(raw, sort_keys=True, allow_nan=False).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise InputError("encoder_runtime_identity must be finite JSON-native data") from exc
    return parse_encoder_runtime_identity_bytes(body, source="encoder_runtime_identity")


def _validate_sha256(value: object, *, field: str) -> None:
    if (
        type(value) is not str
        or not value.startswith("sha256:")
        or len(value) != 71
        or any(character not in "0123456789abcdef" for character in value[7:])
    ):
        raise InputError(f"encoder runtime identity {field} must be a sha256 digest")


def _reject_duplicate_keys(pairs: list[tuple[str, object]]) -> dict[str, object]:
    payload: dict[str, object] = {}
    for key, value in pairs:
        if key in payload:
            raise InputError("duplicate JSON key is not allowed", details={"key": key})
        payload[key] = value
    return payload
