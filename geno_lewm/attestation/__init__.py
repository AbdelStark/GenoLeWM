"""Verifiable-inference primitives for GenoLeWM (RFC-0011).

Today this package exposes the content-addressed model identifiers and
manifest schema (#74). Input commitments (#75), receipts (#76), and
the verify CLI (#77) land in follow-up issues; STARK / TEE attestation
is Phase 4 work.
"""

from geno_lewm.attestation.commitment import (
    DtypeConfig,
    PoolingConfig,
    compute_input_commitment,
)
from geno_lewm.attestation.hashing import canonical_json_sha256, sha256_bytes, sha256_file
from geno_lewm.attestation.manifest import (
    SCHEMA_VERSION,
    Manifest,
    ManifestArtifact,
    ManifestEncoder,
    ManifestTraining,
    load_manifest,
    write_manifest,
)
from geno_lewm.attestation.receipt import (
    RECEIPT_SCHEMA_VERSION,
    SUPPORTED_ATTESTATION_KINDS,
    Receipt,
    ReceiptAttestation,
    ReceiptOutput,
    ReceiptRuntime,
    compute_output_commitment,
    read_receipt,
    write_receipt,
)

__all__ = [
    "DtypeConfig",
    "Manifest",
    "ManifestArtifact",
    "ManifestEncoder",
    "ManifestTraining",
    "PoolingConfig",
    "RECEIPT_SCHEMA_VERSION",
    "Receipt",
    "ReceiptAttestation",
    "ReceiptOutput",
    "ReceiptRuntime",
    "SCHEMA_VERSION",
    "SUPPORTED_ATTESTATION_KINDS",
    "canonical_json_sha256",
    "compute_input_commitment",
    "compute_output_commitment",
    "load_manifest",
    "read_receipt",
    "sha256_bytes",
    "sha256_file",
    "write_manifest",
    "write_receipt",
]
