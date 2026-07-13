# SPDX-License-Identifier: Apache-2.0
"""Closed build provenance for physical membership-store artifacts."""

from __future__ import annotations

import platform
import re
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

from geno_lewm import __version__
from geno_lewm.data._membership_store_contract import (
    MembershipStoreManifest,
    SnapshotLineageBinding,
    _read_json_mapping,
    _require_commit,
    _require_exact_keys,
    _require_mapping,
    _require_text,
)
from geno_lewm.errors import InputError

_BUILD_RECEIPT_SCHEMA_VERSION = "geno-lewm.membership-build-receipt.v1"
_DIGEST_PINNED_IMAGE = re.compile(r"[^@\s]+@sha256:[0-9a-f]{64}")


def _create_build_receipt(
    *,
    artifact_id: str,
    content_identity: str,
    snapshot_lineage: SnapshotLineageBinding,
    builder_git_commit: str,
    container_image: str,
    pyarrow_version: str,
) -> dict[str, object]:
    _require_commit(builder_git_commit, "membership builder git_commit")
    _require_container_image(container_image)
    _require_text(pyarrow_version, "membership builder pyarrow_version")
    return {
        "schema_version": _BUILD_RECEIPT_SCHEMA_VERSION,
        "artifact_id": artifact_id,
        "content_identity": content_identity,
        "snapshot_lineage": snapshot_lineage.to_dict(),
        "builder": {
            "git_commit": builder_git_commit,
            "container_image": container_image,
            "geno_lewm_version": __version__,
        },
        "runtime": {
            "python_implementation": platform.python_implementation(),
            "python_version": platform.python_version(),
            "platform": platform.platform(),
            "sqlite_version": sqlite3.sqlite_version,
            "pyarrow_version": pyarrow_version,
        },
        "created_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
    }


def _verify_build_receipt(path: Path, manifest: MembershipStoreManifest) -> None:
    _raw, payload = _read_json_mapping(path, "membership build receipt")
    _require_exact_keys(
        payload,
        {
            "schema_version",
            "artifact_id",
            "content_identity",
            "snapshot_lineage",
            "builder",
            "runtime",
            "created_at",
        },
        "membership build receipt",
    )
    if payload.get("schema_version") != _BUILD_RECEIPT_SCHEMA_VERSION:
        raise InputError("membership build receipt schema version mismatch")
    if payload.get("artifact_id") != manifest.artifact_id:
        raise InputError("membership build receipt artifact_id mismatch")
    if payload.get("content_identity") != manifest.content_identity:
        raise InputError("membership build receipt content identity mismatch")
    lineage = SnapshotLineageBinding.from_dict(
        _require_mapping(payload.get("snapshot_lineage"), "receipt snapshot_lineage")
    )
    if lineage != manifest.snapshot_lineage:
        raise InputError("membership build receipt snapshot lineage mismatch")

    builder = _require_mapping(payload.get("builder"), "membership receipt builder")
    _require_exact_keys(
        builder,
        {"git_commit", "container_image", "geno_lewm_version"},
        "membership receipt builder",
    )
    _require_commit(builder.get("git_commit"), "membership receipt builder git_commit")
    _require_container_image(builder.get("container_image"))
    _require_text(builder.get("geno_lewm_version"), "membership receipt geno_lewm_version")

    runtime = _require_mapping(payload.get("runtime"), "membership receipt runtime")
    runtime_fields = {
        "python_implementation",
        "python_version",
        "platform",
        "sqlite_version",
        "pyarrow_version",
    }
    _require_exact_keys(runtime, runtime_fields, "membership receipt runtime")
    for field in runtime_fields:
        _require_text(runtime.get(field), f"membership receipt runtime {field}")
    _require_timestamp(payload.get("created_at"))


def _require_container_image(value: object) -> str:
    image = _require_text(value, "membership builder container_image")
    if _DIGEST_PINNED_IMAGE.fullmatch(image) is None:
        raise InputError("membership builder container_image must be digest-pinned")
    return image


def _require_timestamp(value: object) -> str:
    timestamp = _require_text(value, "membership receipt created_at")
    try:
        parsed = datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
    except ValueError as exc:
        raise InputError("membership receipt created_at must be ISO-8601") from exc
    if parsed.tzinfo is None:
        raise InputError("membership receipt created_at must include a timezone")
    return timestamp
