# SPDX-License-Identifier: Apache-2.0
"""Manifest schema for GenoLeWM artifacts (RFC-0011 §3.7).

A manifest is the trust anchor: every byte downstream of the model
file is identified by content hash, and ``model_id = SHA-256(
canonical_json(manifest))``.

This module ships the strict, JSON-native dataclass mirror of the
schema, ``write_manifest`` / ``load_manifest`` (round-trip stable),
and the ``model_id`` derivation.

The schema is intentionally narrow — every field is required and the
loader rejects unknown top-level keys. Adding a field is a MINOR
schema bump (handled by raising :data:`SCHEMA_VERSION` to ``1.1.0``
and accepting both); removing or renaming is MAJOR.
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
from geno_lewm.errors import InputError, SchemaCompatError

__all__ = [
    "SCHEMA_VERSION",
    "Manifest",
    "ManifestArtifact",
    "ManifestEncoder",
    "ManifestTraining",
    "load_manifest",
    "write_manifest",
]


#: The manifest schema version. Bumped on MINOR field-add; MAJOR on
#: removal / rename. RFC-0011 §3.7.
SCHEMA_VERSION: str = "1.0.0"


def _require_hash(name: str, value: str) -> None:
    if not looks_like_sha256(value):
        raise InputError(
            f"{name} must be 'sha256:<64hex>'",
            details={"field": name, "value": value},
        )


@dataclass(frozen=True, slots=True)
class ManifestArtifact:
    """A single artifact file referenced by the manifest."""

    file: str
    hash: str
    dtype: str | None = None
    version: str | None = None

    def __post_init__(self) -> None:
        if not self.file:
            raise InputError("artifact.file must be non-empty", details={"file": self.file})
        _require_hash("artifact.hash", self.hash)


@dataclass(frozen=True, slots=True)
class ManifestEncoder:
    """Carbon encoder identity."""

    id: str
    revision: str
    hash: str

    def __post_init__(self) -> None:
        for name in ("id", "revision"):
            value = getattr(self, name)
            if not value:
                raise InputError(f"encoder.{name} must be non-empty", details={name: value})
        _require_hash("encoder.hash", self.hash)


@dataclass(frozen=True, slots=True)
class ManifestTraining:
    """Training config + data-snapshot identifiers."""

    config_file: str
    hash: str
    data_snapshot: dict[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.config_file:
            raise InputError(
                "training.config_file must be non-empty",
                details={"config_file": self.config_file},
            )
        _require_hash("training.hash", self.hash)


@dataclass(frozen=True, slots=True)
class Manifest:
    """Top-level manifest (RFC-0011 §3.7)."""

    schema_version: str
    model_name: str
    model_version: str
    release_id: str
    encoder: ManifestEncoder
    predictor: ManifestArtifact
    action_encoder: ManifestArtifact
    calibration: ManifestArtifact
    training: ManifestTraining
    eval: ManifestArtifact

    def __post_init__(self) -> None:
        if self.schema_version != SCHEMA_VERSION:
            raise SchemaCompatError(
                "manifest schema_version mismatch",
                details={"got": self.schema_version, "expected": SCHEMA_VERSION},
                remediation="upgrade geno_lewm or regenerate the manifest",
            )
        for name in ("model_name", "model_version", "release_id"):
            value = getattr(self, name)
            if not value:
                raise InputError(f"{name} must be non-empty", details={name: value})

    # ------------------------------------------------------------------
    # Serialization

    def to_canonical_dict(self) -> dict[str, Any]:
        """Return the dict mirror used for canonical JSON.

        Dataclass ``asdict`` walks nested frozen dataclasses and is
        deterministic, so the result is byte-stable when fed to the
        canonical JSON encoder (which also sorts keys).
        """
        return asdict(self)

    def model_id(self) -> str:
        """Return ``model_id = SHA-256(canonical_json(manifest))``."""
        return canonical_json_sha256(self.to_canonical_dict())

    def to_canonical_json(self) -> bytes:
        return canonical_json_bytes(self.to_canonical_dict())


# ---------------------------------------------------------------------------
# Disk I/O


_ARTIFACT_KEYS: dict[str, type[ManifestArtifact]] = {
    "predictor": ManifestArtifact,
    "action_encoder": ManifestArtifact,
    "calibration": ManifestArtifact,
    "eval": ManifestArtifact,
}


def _build_artifact(name: str, blob: dict[str, Any]) -> ManifestArtifact:
    extra = set(blob) - {"file", "hash", "dtype", "version"}
    if extra:
        raise InputError(
            f"unknown keys in {name}: {sorted(extra)}",
            details={"section": name, "extra": sorted(extra)},
        )
    return ManifestArtifact(
        file=blob.get("file", ""),
        hash=blob.get("hash", ""),
        dtype=blob.get("dtype"),
        version=blob.get("version"),
    )


def _from_dict(d: dict[str, Any]) -> Manifest:
    required_top = {
        "schema_version",
        "model_name",
        "model_version",
        "release_id",
        "encoder",
        "predictor",
        "action_encoder",
        "calibration",
        "training",
        "eval",
    }
    missing = required_top - set(d)
    if missing:
        raise InputError(
            "manifest is missing required keys",
            details={"missing": sorted(missing)},
        )
    extra = set(d) - required_top
    if extra:
        raise InputError(
            "manifest has unknown top-level keys",
            details={"extra": sorted(extra)},
        )

    enc = d["encoder"]
    encoder = ManifestEncoder(
        id=enc.get("id", ""),
        revision=enc.get("revision", ""),
        hash=enc.get("hash", ""),
    )
    tr = d["training"]
    training = ManifestTraining(
        config_file=tr.get("config_file", ""),
        hash=tr.get("hash", ""),
        data_snapshot=dict(tr.get("data_snapshot", {})),
    )
    artifacts = {k: _build_artifact(k, d[k]) for k in _ARTIFACT_KEYS}

    return Manifest(
        schema_version=d["schema_version"],
        model_name=d["model_name"],
        model_version=d["model_version"],
        release_id=d["release_id"],
        encoder=encoder,
        training=training,
        **artifacts,
    )


def write_manifest(manifest: Manifest, path: str | Path) -> Path:
    """Write a manifest to disk as canonical JSON.

    The on-disk bytes are byte-stable across platforms.
    """
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_bytes(manifest.to_canonical_json())
    return p


def load_manifest(path: str | Path) -> Manifest:
    """Load and validate a manifest from disk."""
    p = Path(path)
    raw = p.read_bytes()
    try:
        d = json.loads(raw.decode("utf-8"))
    except json.JSONDecodeError as exc:
        raise InputError(
            "manifest is not valid JSON",
            details={"path": str(p), "error": str(exc)},
        ) from exc
    return _from_dict(d)
