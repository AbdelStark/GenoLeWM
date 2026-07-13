# SPDX-License-Identifier: Apache-2.0
"""Scalable, lineage-bound membership artifacts for v0.3 datasets.

This stable public facade keeps the production-scale membership contract
separate from its writer, verifier, storage, lineage, and runtime adapters.
PyArrow is imported only by the Parquet build and verification paths.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

from geno_lewm.data._membership_store_contract import (
    MEMBERSHIP_STORE_SCHEMA_VERSION,
    MembershipSourceBinding,
    MembershipSourceInput,
    MembershipStoreFile,
    MembershipStoreManifest,
    MembershipStoreVerification,
    SnapshotLineageBinding,
)
from geno_lewm.data._membership_store_runtime import (
    MembershipStore,
    MembershipStoreHoldoutPolicy,
)
from geno_lewm.data._membership_store_verifier import (
    verify_membership_store as _verify_membership_store,
)
from geno_lewm.data._membership_store_writer import (
    build_membership_store as _build_membership_store,
)

__all__ = [
    "MEMBERSHIP_STORE_SCHEMA_VERSION",
    "MembershipSourceBinding",
    "MembershipSourceInput",
    "MembershipStore",
    "MembershipStoreFile",
    "MembershipStoreHoldoutPolicy",
    "MembershipStoreManifest",
    "MembershipStoreVerification",
    "SnapshotLineageBinding",
    "build_membership_store",
    "verify_membership_store",
]


def build_membership_store(
    *,
    artifact_id: str,
    snapshot_lineage_path: Path,
    sources: Sequence[MembershipSourceInput],
    output_dir: Path,
) -> MembershipStoreManifest:
    """Build one immutable store from exact local staged-source bytes."""
    return _build_membership_store(
        artifact_id=artifact_id,
        snapshot_lineage_path=snapshot_lineage_path,
        sources=sources,
        output_dir=output_dir,
    )


def verify_membership_store(store_dir: Path) -> MembershipStoreVerification:
    """Independently verify manifest, files, Parquet rows, and SQLite rows."""
    return _verify_membership_store(store_dir)


for _public_type in (
    MembershipSourceBinding,
    MembershipSourceInput,
    MembershipStore,
    MembershipStoreFile,
    MembershipStoreHoldoutPolicy,
    MembershipStoreManifest,
    MembershipStoreVerification,
    SnapshotLineageBinding,
):
    _public_type.__module__ = __name__

del _public_type
