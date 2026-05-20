"""Verifiable-inference primitives for GenoLeWM (RFC-0011).

Today this package exposes the content-addressed model identifiers and
manifest schema (#74). Input commitments (#75), receipts (#76), and
the verify CLI (#77) land in follow-up issues; STARK / TEE attestation
is Phase 4 work.
"""

from geno_lewm.attestation.hashing import canonical_json_sha256, sha256_file, sha256_bytes
from geno_lewm.attestation.manifest import (
    Manifest,
    ManifestArtifact,
    ManifestEncoder,
    ManifestTraining,
    SCHEMA_VERSION,
    load_manifest,
    write_manifest,
)

__all__ = [
    "Manifest",
    "ManifestArtifact",
    "ManifestEncoder",
    "ManifestTraining",
    "SCHEMA_VERSION",
    "canonical_json_sha256",
    "load_manifest",
    "sha256_bytes",
    "sha256_file",
    "write_manifest",
]
