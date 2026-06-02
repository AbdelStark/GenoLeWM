# SPDX-License-Identifier: Apache-2.0
"""Artifact provenance primitives for GenoLeWM.

This is the preferred public import path for manifests, hashes,
input/output commitments, and checksum receipts. The package does not
implement or claim runtime assurance beyond checksum provenance.
"""

from geno_lewm.provenance.commitment import (
    DtypeConfig,
    PoolingConfig,
    compute_input_commitment,
)
from geno_lewm.provenance.hashing import canonical_json_sha256, sha256_bytes, sha256_file
from geno_lewm.provenance.manifest import (
    SCHEMA_VERSION,
    Manifest,
    ManifestArtifact,
    ManifestEncoder,
    ManifestTraining,
    load_manifest,
    write_manifest,
)
from geno_lewm.provenance.receipt import (
    RECEIPT_SCHEMA_VERSION,
    SUPPORTED_PROVENANCE_KINDS,
    Receipt,
    ReceiptOutput,
    ReceiptProvenance,
    ReceiptRuntime,
    compute_output_commitment,
    parse_receipt_payload,
    read_receipt,
    write_receipt,
)

__all__ = [
    "RECEIPT_SCHEMA_VERSION",
    "SCHEMA_VERSION",
    "SUPPORTED_PROVENANCE_KINDS",
    "DtypeConfig",
    "Manifest",
    "ManifestArtifact",
    "ManifestEncoder",
    "ManifestTraining",
    "PoolingConfig",
    "Receipt",
    "ReceiptOutput",
    "ReceiptProvenance",
    "ReceiptRuntime",
    "canonical_json_sha256",
    "compute_input_commitment",
    "compute_output_commitment",
    "load_manifest",
    "parse_receipt_payload",
    "read_receipt",
    "sha256_bytes",
    "sha256_file",
    "write_manifest",
    "write_receipt",
]
