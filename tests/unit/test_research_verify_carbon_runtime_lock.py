# SPDX-License-Identifier: Apache-2.0
"""Offline lock tests for the corrected v0.3 Carbon runtime identity."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import jsonschema
import pytest

from tools.research.verify_carbon_runtime_lock import (
    IMPLEMENTATION_FILE_PATHS,
    CarbonRuntimeLockError,
    verify_carbon_runtime_lock,
)

LOCK_PATH = Path("configs/data_v03/carbon-500m-runtime-content-lock.json")
LOCK_SCHEMA_PATH = Path("configs/data_v03/carbon-500m-runtime-content-lock.schema.json")
CORRECTED_RUNTIME_HASH = "sha256:a1fd1dd20756c7248b7f9ca95c59c821f0329530fd49c6fea253a8df9a6a6311"


def test_committed_carbon_runtime_lock_recomputes_offline() -> None:
    report = verify_carbon_runtime_lock(LOCK_PATH)

    assert report == {
        "implementation_file_count": 8,
        "model_revision": "5d31d59b3c845b288a13aedb1358934196852eec",
        "runtime_file_count": 10,
        "runtime_hash": CORRECTED_RUNTIME_HASH,
        "schema_version": "geno-lewm.carbon-runtime-content-lock.v1",
        "verified": True,
        "weights_hash": ("sha256:e257506988203fdb8bb46976ee81c97e24f29073754bbff70137c7704dbadaa8"),
    }


def test_committed_carbon_runtime_lock_satisfies_its_closed_schema() -> None:
    schema = json.loads(LOCK_SCHEMA_PATH.read_bytes())
    lock = json.loads(LOCK_PATH.read_bytes())

    validator = jsonschema.validators.validator_for(schema)
    validator.check_schema(schema)
    validator(schema).validate(lock)


def test_correction_receipt_binds_the_replayable_source_transition() -> None:
    lock = json.loads(LOCK_PATH.read_bytes())
    receipt = lock["correction_receipt"]

    assert receipt["runtime_hash_provenance"] == {
        "first_affected_merge_commit": "8a278c5e8ec57d1e4839336f9cd65823d3a216e2",
        "post_rebase_runtime_hash": CORRECTED_RUNTIME_HASH,
        "pre_rebase_runtime_hash": (
            "sha256:add3c1a663a35fb92fbd3fd935b067da1aed8aeb143ea01f7d92c2cd3ed2aa5e"
        ),
        "pre_rebase_source_commit": "e9845cffb4ff1dcdb00cce0215564d83d6ce8317",
        "typing_change_commit": "e12273f538370c23235743cf559f19fcb344cfc0",
    }
    reason = receipt["training_trace"]["reason"]
    assert "ordered request construction" in reason
    assert "rows contain neither an encoder runtime hash nor a cache key" in reason


def test_offline_lock_rejects_implementation_file_drift(tmp_path: Path) -> None:
    for relative in IMPLEMENTATION_FILE_PATHS:
        destination = tmp_path / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(relative, destination)
    with (tmp_path / "geno_lewm/encoder/pooling.py").open("ab") as stream:
        stream.write(b"\n# unintended source drift\n")

    with pytest.raises(CarbonRuntimeLockError, match="implementation file hash drifted"):
        verify_carbon_runtime_lock(LOCK_PATH, repository_root=tmp_path)


def test_offline_lock_rejects_malformed_json_at_the_boundary(tmp_path: Path) -> None:
    malformed = tmp_path / "runtime-lock.json"
    malformed.write_bytes(b'{"schema_version":')

    with pytest.raises(CarbonRuntimeLockError, match="invalid JSON"):
        verify_carbon_runtime_lock(malformed)


def test_offline_lock_rejects_duplicate_json_keys(tmp_path: Path) -> None:
    duplicate = tmp_path / "runtime-lock.json"
    duplicate.write_bytes(b'{"schema_version":"first","schema_version":"second"}')

    with pytest.raises(CarbonRuntimeLockError, match="duplicate key 'schema_version'"):
        verify_carbon_runtime_lock(duplicate)


def test_offline_lock_rejects_unknown_top_level_keys(tmp_path: Path) -> None:
    payload = json.loads(LOCK_PATH.read_bytes())
    payload["unreviewed_extension"] = True
    extended = tmp_path / "runtime-lock.json"
    extended.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(CarbonRuntimeLockError, match="closed key set"):
        verify_carbon_runtime_lock(extended)


def test_offline_lock_rejects_canonical_runtime_hash_mismatch(tmp_path: Path) -> None:
    payload = json.loads(LOCK_PATH.read_bytes())
    payload["runtime_hash"] = "sha256:" + "0" * 64
    mismatched = tmp_path / "runtime-lock.json"
    mismatched.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(CarbonRuntimeLockError, match="canonical runtime hash drifted"):
        verify_carbon_runtime_lock(mismatched)
