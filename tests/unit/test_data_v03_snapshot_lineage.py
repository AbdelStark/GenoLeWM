# SPDX-License-Identifier: Apache-2.0
"""Contracts for the v0.3 immutable snapshot-lineage assembler."""

from __future__ import annotations

import copy
import hashlib
import json
import subprocess
import sys
import threading
import time
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import pytest
from jsonschema import Draft202012Validator

from geno_lewm.data import _snapshot_lineage as snapshot_lineage_core
from geno_lewm.data._snapshot_lineage import (
    CLINVAR_REQUIRED_CLAIM_BOUNDARY,
    GENERATED_BY,
    LINEAGE_CLAIM_BOUNDARY,
    LINEAGE_SCHEMA_VERSION,
    MEMBERSHIP_STATUS,
    SnapshotLineageError,
    capture_verified_snapshot_lineage,
    verify_snapshot_lineage,
)
from geno_lewm.provenance import canonical_json_sha256, sha256_file
from tools.data import v03_snapshot_lineage as snapshot_lineage_tool
from tools.data._immutable_json import supports_secure_immutable_json_publication
from tools.data.v03_clinvar_postflight import (
    REMOTE_POSTFLIGHT_SCHEMA_VERSION as CLINVAR_REMOTE_POSTFLIGHT_SCHEMA_VERSION,
)
from tools.data.v03_gnomad_lock import REMOTE_POSTFLIGHT_SCHEMA_VERSION, select_source
from tools.data.v03_snapshot_lineage import (
    _fsync_directory,
    assemble_snapshot_lineage,
    main,
)

SOURCE_LOCK = Path("configs/data_v03/gnomad-v4.1-exomes-autosomes.source-lock.json")
SPEC_SCHEMA = Path("configs/data_v03/snapshot-lineage-spec.schema.json")
LINEAGE_SCHEMA = Path("configs/data_v03/snapshot-lineage.schema.json")
CONTAINER_IMAGE = (
    "ghcr.io/astral-sh/uv@sha256:35b0aa516fbcf6f18624919cfc38fa02ab3458e0ffcd3c03e932051b37f315db"
)
CODE_COMMIT = "c" * 40
CLINVAR_ARCHIVE_URL = (
    "https://ftp.ncbi.nlm.nih.gov/pub/clinvar/vcf_GRCh38/archive_2.0/2026/clinvar_20260415.vcf.gz"
)
REMOTE_NAMESPACE_FILES = (
    "data/gnomad/v4.1/variants.parquet",
    "evidence/gcs-metadata-verification.json",
    "evidence/gcs-object-metadata.json",
    "evidence/prepare-report.json",
    "evidence/receipt.json",
    "evidence/selection.json",
    "evidence/source-lock.json",
    "evidence/source-lock.schema.json",
    "evidence/source-stream-identity.json",
)
REMOTE_POSTFLIGHT_CHECKS = (
    "exact_hub_revision_resolved",
    "complete_namespace_file_set",
    "source_lock_and_schema_bound",
    "source_lock_and_schema_match_source_commit",
    "selection_rederived_from_source_lock",
    "metadata_verification_recomputed",
    "receipt_evidence_identities_recomputed",
    "parquet_sha256_and_size_recomputed",
    "parquet_full_scan_recomputed",
)
CLINVAR_REMOTE_FILES = (
    "clinvar/2026-04-15/variants.parquet",
    "evidence/audit.json",
    "evidence/prepare_report.json",
    "evidence/runtime_report.json",
)
CLINVAR_REMOTE_CHECKS = (
    "exact_hub_revision_resolved",
    "complete_namespace_file_set",
    "source_contract_loaded_from_exact_git_commit",
    "source_contract_derived_from_ast",
    "audit_prepare_runtime_reconciled",
    "source_release_sha256_and_size_reconciled",
    "parquet_sha256_and_size_recomputed",
    "parquet_schema_derived_from_source_commit",
    "parquet_full_scan_recomputed",
)
CLINVAR_PARQUET_SCHEMA = (
    {"name": "chrom", "type": "string"},
    {"name": "pos", "type": "int64"},
    {"name": "ref", "type": "string"},
    {"name": "alt", "type": "string"},
    {"name": "clinical_significance", "type": "string"},
    {"name": "review_status", "type": "string"},
    {"name": "gene_symbol", "type": "string"},
    {"name": "clinvar_id", "type": "int64"},
    {"name": "schema_version", "type": "string"},
)

requires_secure_immutable_json_publication = pytest.mark.skipif(
    not supports_secure_immutable_json_publication(),
    reason="secure immutable publication requires anchored dir_fd operations",
)


def test_tool_reexports_the_installable_snapshot_lineage_verifier() -> None:
    assert snapshot_lineage_tool.SnapshotLineageError is snapshot_lineage_core.SnapshotLineageError
    assert (
        snapshot_lineage_tool.VerifiedSnapshotLineage
        is snapshot_lineage_core.VerifiedSnapshotLineage
    )
    assert (
        snapshot_lineage_tool.capture_verified_snapshot_lineage
        is snapshot_lineage_core.capture_verified_snapshot_lineage
    )
    assert (
        snapshot_lineage_tool.verify_snapshot_lineage
        is snapshot_lineage_core.verify_snapshot_lineage
    )


@requires_secure_immutable_json_publication
def test_assembler_builds_deterministic_lineage_without_memberships(tmp_path: Path) -> None:
    spec_path, _spec = _write_evidence_bundle(tmp_path)
    output_path = tmp_path / "snapshot-lineage.json"

    lineage = assemble_snapshot_lineage(
        spec_path=spec_path,
        gnomad_source_lock_path=SOURCE_LOCK,
        output_path=output_path,
    )

    assert lineage["schema_version"] == LINEAGE_SCHEMA_VERSION
    assert lineage["generated_by"] == GENERATED_BY
    assert lineage["candidate_snapshot_id"] == "geno-lewm-data-v0.3.0-r1"
    assert lineage["reference_genome"] == "GRCh38"
    assert lineage["membership_status"] == MEMBERSHIP_STATUS == "not_created"
    assert "memberships" not in lineage
    assert [shard["chromosome"] for shard in lineage["gnomad"]["shards"]] == [
        str(chromosome) for chromosome in range(1, 23)
    ]
    assert lineage["gnomad"]["split_policy"] == {
        "train": [*(str(chromosome) for chromosome in range(1, 20)), "22"],
        "validation": ["20"],
        "evaluation": ["21"],
    }
    assert lineage["gnomad"]["total_records"] == sum(
        1000 + chromosome for chromosome in range(1, 23)
    )
    assert lineage["gnomad"]["common_execution"]["commit_sha"] == CODE_COMMIT
    assert lineage["gnomad"]["data_use"]["license"] == {
        "scope": "gnomAD primary exome data; third-party annotations may carry separate terms.",
        "spdx": "CC0-1.0",
    }
    assert lineage["gnomad"]["data_use"]["terms_checked_on"] == "2026-07-13"
    assert (
        "Do not attempt to reidentify participants."
        in lineage["gnomad"]["data_use"]["restrictions"]
    )
    assert {
        shard["remote_postflight"]["schema_version"] for shard in lineage["gnomad"]["shards"]
    } == {REMOTE_POSTFLIGHT_SCHEMA_VERSION}
    assert all(
        shard["remote_postflight"]["sha256"].startswith("sha256:")
        for shard in lineage["gnomad"]["shards"]
    )
    assert all(
        shard["remote_postflight"]["parquet_audit"]["canonical_chromosome"] == shard["chromosome"]
        for shard in lineage["gnomad"]["shards"]
    )
    assert lineage["clinvar"]["revision"] == "d" * 40
    assert lineage["clinvar"]["data_use"]["license"]["spdx"] == "NOASSERTION"
    assert lineage["clinvar"]["data_use"]["terms_urls"] == [
        "https://www.ncbi.nlm.nih.gov/clinvar/docs/maintenance_use/",
        "https://www.ncbi.nlm.nih.gov/home/about/policies/",
    ]
    checked_schema = json.loads(LINEAGE_SCHEMA.read_text(encoding="utf-8"))
    assert (
        lineage["gnomad"]["data_use"]
        == checked_schema["properties"]["gnomad"]["properties"]["data_use"]["const"]
    )
    assert (
        lineage["clinvar"]["data_use"]
        == checked_schema["properties"]["clinvar"]["properties"]["data_use"]["const"]
    )
    assert lineage["clinvar"]["output"]["class_balance"] == {
        "B": 2,
        "LB": 3,
        "LP": 5,
        "OTHER": 7,
        "P": 11,
        "VUS": 13,
    }
    assert lineage["clinvar"]["output"]["records"] == 41
    raw_clinvar_postflight = json.loads(
        (tmp_path / "clinvar-postflight.json").read_text(encoding="utf-8")
    )
    assert lineage["clinvar"]["remote_postflight"] == {
        "schema_version": CLINVAR_REMOTE_POSTFLIGHT_SCHEMA_VERSION,
        "sha256": _spec["clinvar"]["postflight_sha256"],
        "size_bytes": (tmp_path / "clinvar-postflight.json").stat().st_size,
        "verified_files": list(CLINVAR_REMOTE_FILES),
        "file_identities": {
            path: {
                "sha256": "sha256:" + raw_clinvar_postflight["file_identities"][path]["sha256"],
                "size_bytes": raw_clinvar_postflight["file_identities"][path]["size_bytes"],
            }
            for path in CLINVAR_REMOTE_FILES
        },
        "checks": list(CLINVAR_REMOTE_CHECKS),
        "parquet_audit": _clinvar_parquet_audit(),
    }
    commitment_payload = dict(lineage)
    del commitment_payload["lineage_id"]
    assert lineage["lineage_id"] == canonical_json_sha256(commitment_payload)
    assert json.loads(output_path.read_text(encoding="utf-8")) == lineage
    assert "does not create snapshot memberships" in lineage["claim_boundary"]


def test_assembler_rejects_duplicate_json_keys(tmp_path: Path) -> None:
    spec_path, _spec = _write_evidence_bundle(tmp_path)
    raw = spec_path.read_text(encoding="utf-8").replace(
        '  "reference_genome": "GRCh38",',
        '  "reference_genome": "GRCh37",\n  "reference_genome": "GRCh38",',
    )
    spec_path.write_text(raw, encoding="utf-8")

    with pytest.raises(SnapshotLineageError, match=r"duplicate JSON key.*reference_genome"):
        assemble_snapshot_lineage(
            spec_path=spec_path,
            gnomad_source_lock_path=SOURCE_LOCK,
        )


def test_assembler_rejects_unknown_gnomad_receipt_fields(tmp_path: Path) -> None:
    spec_path, spec = _write_evidence_bundle(tmp_path)
    receipt_path = tmp_path / "receipts" / "chr1.json"
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    receipt["unreviewed_claim"] = "looks good"
    _write_json(receipt_path, receipt)
    spec["gnomad"]["shards"][0]["receipt_sha256"] = sha256_file(receipt_path)
    _write_json(spec_path, spec)

    with pytest.raises(SnapshotLineageError, match="chr1 staging receipt keys drifted"):
        assemble_snapshot_lineage(
            spec_path=spec_path,
            gnomad_source_lock_path=SOURCE_LOCK,
        )


def test_assembler_rejects_empty_v41_population_columns(tmp_path: Path) -> None:
    spec_path, spec = _write_evidence_bundle(tmp_path)
    receipt_path = tmp_path / "receipts" / "chr1.json"
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    receipt["output"]["parquet_audit"]["population_af_non_null_counts"]["af_mid"] = 0
    _write_json(receipt_path, receipt)
    spec["gnomad"]["shards"][0]["receipt_sha256"] = sha256_file(receipt_path)
    _write_json(spec_path, spec)

    with pytest.raises(SnapshotLineageError, match=r"required v4.1 population.*af_mid"):
        assemble_snapshot_lineage(
            spec_path=spec_path,
            gnomad_source_lock_path=SOURCE_LOCK,
        )


def test_checked_schemas_are_closed_and_membership_free() -> None:
    spec_schema = json.loads(SPEC_SCHEMA.read_text(encoding="utf-8"))
    lineage_schema = json.loads(LINEAGE_SCHEMA.read_text(encoding="utf-8"))

    assert spec_schema["$schema"] == "https://json-schema.org/draft/2020-12/schema"
    assert spec_schema["additionalProperties"] is False
    assert spec_schema["properties"]["schema_version"]["const"] == (
        "geno-lewm.v03-snapshot-lineage-spec.v1"
    )
    shards = spec_schema["properties"]["gnomad"]["properties"]["shards"]
    assert shards["minItems"] == shards["maxItems"] == 22
    assert shards["items"]["additionalProperties"] is False
    assert shards["items"]["properties"]["revision"]["pattern"] == "^[0-9a-f]{40}$"
    assert {"postflight_file", "postflight_sha256"} <= set(shards["items"]["required"])
    assert shards["items"]["properties"]["postflight_sha256"]["pattern"] == (
        "^sha256:[0-9a-f]{64}$"
    )
    spec_clinvar = spec_schema["properties"]["clinvar"]
    assert {"postflight_file", "postflight_sha256"} <= set(spec_clinvar["required"])
    assert spec_clinvar["properties"]["postflight_sha256"]["pattern"] == ("^sha256:[0-9a-f]{64}$")
    assert lineage_schema["additionalProperties"] is False
    assert "does not recompute lineage_id" in lineage_schema["$comment"]
    assert lineage_schema["properties"]["schema_version"]["const"] == LINEAGE_SCHEMA_VERSION
    assert lineage_schema["properties"]["membership_status"]["const"] == "not_created"
    assert "memberships" not in lineage_schema["properties"]
    lineage_shard = lineage_schema["properties"]["gnomad"]["properties"]["shards"]["items"]
    assert "remote_postflight" in lineage_shard["required"]
    remote_postflight = lineage_shard["properties"]["remote_postflight"]
    assert remote_postflight["additionalProperties"] is False
    assert remote_postflight["properties"]["schema_version"]["const"] == (
        REMOTE_POSTFLIGHT_SCHEMA_VERSION
    )
    assert remote_postflight["properties"]["verified_files"]["const"] == list(
        REMOTE_NAMESPACE_FILES
    )
    assert remote_postflight["properties"]["file_identities"]["$ref"] == (
        "#/$defs/gnomadFileIdentities"
    )
    assert "file_identities" in remote_postflight["required"]
    assert remote_postflight["properties"]["checks"]["const"] == list(REMOTE_POSTFLIGHT_CHECKS)
    assert remote_postflight["properties"]["parquet_audit"]["$ref"] == (
        "#/$defs/gnomadParquetAudit"
    )
    assert "parquet_audit" in remote_postflight["required"]
    gnomad_schema = lineage_schema["properties"]["gnomad"]
    assert "data_use" in gnomad_schema["required"]
    assert gnomad_schema["properties"]["data_use"]["const"]["license"]["spdx"] == "CC0-1.0"
    clinvar_schema = lineage_schema["properties"]["clinvar"]
    assert "data_use" in clinvar_schema["required"]
    assert clinvar_schema["properties"]["data_use"]["const"]["license"]["spdx"] == ("NOASSERTION")
    assert "remote_postflight" in clinvar_schema["required"]
    clinvar_postflight = clinvar_schema["properties"]["remote_postflight"]
    assert clinvar_postflight["additionalProperties"] is False
    assert clinvar_postflight["properties"]["schema_version"]["const"] == (
        CLINVAR_REMOTE_POSTFLIGHT_SCHEMA_VERSION
    )
    assert clinvar_postflight["properties"]["verified_files"]["prefixItems"] == [
        {"const": path} for path in CLINVAR_REMOTE_FILES
    ]
    assert clinvar_postflight["properties"]["file_identities"]["$ref"] == (
        "#/$defs/clinvarFileIdentities"
    )
    assert "file_identities" in clinvar_postflight["required"]
    assert clinvar_postflight["properties"]["checks"]["prefixItems"] == [
        {"const": check} for check in CLINVAR_REMOTE_CHECKS
    ]
    assert clinvar_postflight["properties"]["parquet_audit"]["$ref"] == (
        "#/$defs/clinvarParquetAudit"
    )


def test_draft_2020_12_schemas_validate_real_artifacts_and_exact_arrays(
    tmp_path: Path,
) -> None:
    spec_path, spec = _write_evidence_bundle(tmp_path)
    lineage = assemble_snapshot_lineage(
        spec_path=spec_path,
        gnomad_source_lock_path=SOURCE_LOCK,
    )
    spec_schema = json.loads(SPEC_SCHEMA.read_text(encoding="utf-8"))
    lineage_schema = json.loads(LINEAGE_SCHEMA.read_text(encoding="utf-8"))
    Draft202012Validator.check_schema(spec_schema)
    Draft202012Validator.check_schema(lineage_schema)
    assert not list(Draft202012Validator(spec_schema).iter_errors(spec))
    validator = Draft202012Validator(lineage_schema)
    assert not list(validator.iter_errors(lineage))

    for document, document_validator in (
        (spec, Draft202012Validator(spec_schema)),
        (lineage, validator),
    ):
        duplicate_autosome = copy.deepcopy(document)
        duplicate_autosome["gnomad"]["shards"][1]["chromosome"] = "1"
        assert list(document_validator.iter_errors(duplicate_autosome))

        wrong_validation_role = copy.deepcopy(document)
        wrong_validation_role["gnomad"]["shards"][19]["split_role"] = "train"
        assert list(document_validator.iter_errors(wrong_validation_role))

        wrong_evaluation_role = copy.deepcopy(document)
        wrong_evaluation_role["gnomad"]["shards"][20]["split_role"] = "train"
        assert list(document_validator.iter_errors(wrong_evaluation_role))

    for path in (
        ("clinvar", "remote_postflight", "checks"),
        ("clinvar", "remote_postflight", "verified_files"),
        ("clinvar", "remote_postflight", "parquet_audit", "schema"),
    ):
        invalid = copy.deepcopy(lineage)
        target: Any = invalid
        for component in path[:-1]:
            target = target[component]
        target[path[-1]] = []
        assert list(validator.iter_errors(invalid)), path

    for field, value in _walk_json(lineage_schema):
        if isinstance(value, dict) and "prefixItems" in value:
            expected = len(value["prefixItems"])
            assert value.get("minItems") == expected, field
            assert value.get("maxItems") == expected, field


@pytest.mark.parametrize(
    "unsafe_path",
    ["../outside.json", "/tmp/outside.json", r"C:\outside.json", r"..\outside.json", "x.bin"],
    ids=["parent", "absolute", "windows-drive", "backslash-parent", "wrong-extension"],
)
def test_spec_schema_and_runtime_reject_unsafe_json_paths(
    unsafe_path: str,
    tmp_path: Path,
) -> None:
    spec_path, spec = _write_evidence_bundle(tmp_path)
    spec["gnomad"]["shards"][0]["receipt_file"] = unsafe_path
    _write_json(spec_path, spec)
    schema = json.loads(SPEC_SCHEMA.read_text(encoding="utf-8"))

    assert list(Draft202012Validator(schema).iter_errors(spec))
    with pytest.raises(SnapshotLineageError, match=r"relative in-bundle JSON path"):
        assemble_snapshot_lineage(
            spec_path=spec_path,
            gnomad_source_lock_path=SOURCE_LOCK,
        )


def test_assembler_rejects_unknown_nested_receipt_fields(tmp_path: Path) -> None:
    spec_path, spec = _write_evidence_bundle(tmp_path)
    receipt_path = tmp_path / "receipts" / "chr1.json"
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    receipt["source"]["inferred_ancestry"] = "not-a-source-identity"
    _write_json(receipt_path, receipt)
    spec["gnomad"]["shards"][0]["receipt_sha256"] = sha256_file(receipt_path)
    _write_json(spec_path, spec)

    with pytest.raises(SnapshotLineageError, match="chr1 source keys drifted"):
        assemble_snapshot_lineage(
            spec_path=spec_path,
            gnomad_source_lock_path=SOURCE_LOCK,
        )


def test_assembler_requires_independent_full_parquet_audit(tmp_path: Path) -> None:
    spec_path, spec = _write_evidence_bundle(tmp_path)
    receipt_path = tmp_path / "receipts" / "chr1.json"
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    receipt["output"]["parquet_audit"]["audit_method"] = "metadata_only"
    _write_json(receipt_path, receipt)
    spec["gnomad"]["shards"][0]["receipt_sha256"] = sha256_file(receipt_path)
    _write_json(spec_path, spec)

    with pytest.raises(SnapshotLineageError, match="Parquet audit method drifted"):
        assemble_snapshot_lineage(
            spec_path=spec_path,
            gnomad_source_lock_path=SOURCE_LOCK,
        )


def test_assembler_rejects_zero_placeholder_revisions(tmp_path: Path) -> None:
    spec_path, spec = _write_evidence_bundle(tmp_path)
    spec["gnomad"]["shards"][0]["revision"] = "0" * 40
    _write_json(spec_path, spec)

    with pytest.raises(SnapshotLineageError, match="revision must be a non-zero"):
        assemble_snapshot_lineage(
            spec_path=spec_path,
            gnomad_source_lock_path=SOURCE_LOCK,
        )


def test_assembler_requires_exactly_one_receipt_per_autosome(tmp_path: Path) -> None:
    spec_path, spec = _write_evidence_bundle(tmp_path)
    spec["gnomad"]["shards"].pop()
    _write_json(spec_path, spec)

    with pytest.raises(SnapshotLineageError, match="exactly 22"):
        assemble_snapshot_lineage(
            spec_path=spec_path,
            gnomad_source_lock_path=SOURCE_LOCK,
        )


@requires_secure_immutable_json_publication
def test_lineage_output_is_idempotent_but_immutable(tmp_path: Path) -> None:
    spec_path, spec = _write_evidence_bundle(tmp_path)
    output_path = tmp_path / "snapshot-lineage.json"
    first = assemble_snapshot_lineage(
        spec_path=spec_path,
        gnomad_source_lock_path=SOURCE_LOCK,
        output_path=output_path,
    )

    second = assemble_snapshot_lineage(
        spec_path=spec_path,
        gnomad_source_lock_path=SOURCE_LOCK,
        output_path=output_path,
    )
    assert second == first

    spec["candidate_snapshot_id"] = "geno-lewm-data-v0.3.0-r2"
    _write_json(spec_path, spec)
    with pytest.raises(SnapshotLineageError, match="refusing to replace different lineage bytes"):
        assemble_snapshot_lineage(
            spec_path=spec_path,
            gnomad_source_lock_path=SOURCE_LOCK,
            output_path=output_path,
        )


@requires_secure_immutable_json_publication
def test_lineage_output_rejects_a_symlink_even_when_target_bytes_match(tmp_path: Path) -> None:
    spec_path, _spec = _write_evidence_bundle(tmp_path)
    target = tmp_path / "target.json"
    lineage = assemble_snapshot_lineage(
        spec_path=spec_path,
        gnomad_source_lock_path=SOURCE_LOCK,
        output_path=target,
    )
    alias = tmp_path / "snapshot-lineage.json"
    alias.symlink_to(target.name)

    with pytest.raises(SnapshotLineageError, match="symlink or non-regular"):
        assemble_snapshot_lineage(
            spec_path=spec_path,
            gnomad_source_lock_path=SOURCE_LOCK,
            output_path=alias,
        )

    assert json.loads(target.read_text(encoding="utf-8")) == lineage


@requires_secure_immutable_json_publication
def test_lineage_output_rejects_an_existing_non_regular_path(tmp_path: Path) -> None:
    spec_path, _spec = _write_evidence_bundle(tmp_path)
    output_path = tmp_path / "snapshot-lineage.json"
    output_path.mkdir()

    with pytest.raises(SnapshotLineageError, match="symlink or non-regular"):
        assemble_snapshot_lineage(
            spec_path=spec_path,
            gnomad_source_lock_path=SOURCE_LOCK,
            output_path=output_path,
        )

    assert not list(tmp_path.glob(".snapshot-lineage.json.*.tmp"))


def test_directory_fsync_is_skipped_when_the_platform_has_no_directory_flag(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delattr("tools.data._immutable_json.os.O_DIRECTORY", raising=False)

    def unexpected_open(*_args: object, **_kwargs: object) -> int:
        raise AssertionError("directory open must be skipped without O_DIRECTORY")

    monkeypatch.setattr("tools.data._immutable_json.os.open", unexpected_open)

    _fsync_directory(tmp_path)


@requires_secure_immutable_json_publication
def test_concurrent_different_lineage_writers_never_displace_a_successful_caller(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    first_root = tmp_path / "first"
    second_root = tmp_path / "second"
    first_root.mkdir()
    second_root.mkdir()
    first_spec, _first = _write_evidence_bundle(first_root)
    second_spec, second = _write_evidence_bundle(second_root)
    second["candidate_snapshot_id"] = "geno-lewm-data-v0.3.0-r2"
    _write_json(second_spec, second)
    output = tmp_path / "snapshot-lineage.json"
    legacy_temporary = output.with_name(output.name + ".tmp")
    start = threading.Barrier(2)
    first_written = threading.Event()
    second_written = threading.Event()
    first_published = threading.Event()
    path_write_bytes = Path.write_bytes
    path_replace = Path.replace

    def coordinated_write(path: Path, data: bytes) -> int:
        if path != legacy_temporary:
            return path_write_bytes(path, data)
        if threading.current_thread().name == "lineage-first":
            result = path_write_bytes(path, data)
            first_written.set()
            assert second_written.wait(timeout=10)
            return result
        assert first_written.wait(timeout=10)
        result = path_write_bytes(path, data)
        second_written.set()
        return result

    def coordinated_replace(path: Path, target: Path) -> Path:
        if path != legacy_temporary:
            return path_replace(path, target)
        if threading.current_thread().name == "lineage-first":
            result = path_replace(path, target)
            first_published.set()
            return result
        assert first_published.wait(timeout=10)
        return path_replace(path, target)

    monkeypatch.setattr(Path, "write_bytes", coordinated_write)
    monkeypatch.setattr(Path, "replace", coordinated_replace)
    outcomes: dict[str, dict[str, Any] | BaseException] = {}

    def write(name: str, spec_path: Path) -> None:
        start.wait()
        try:
            outcomes[name] = assemble_snapshot_lineage(
                spec_path=spec_path,
                gnomad_source_lock_path=SOURCE_LOCK,
                output_path=output,
            )
        except BaseException as exc:  # capture the complete competing-writer outcome
            outcomes[name] = exc

    threads = [
        threading.Thread(target=write, name="lineage-first", args=("first", first_spec)),
        threading.Thread(target=write, name="lineage-second", args=("second", second_spec)),
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=20)
        assert not thread.is_alive()

    successful = {name: value for name, value in outcomes.items() if isinstance(value, dict)}
    rejected = {name: value for name, value in outcomes.items() if isinstance(value, BaseException)}
    assert len(successful) == 1
    assert len(rejected) == 1
    assert all(isinstance(error, SnapshotLineageError) for error in rejected.values())
    winner = next(iter(successful.values()))
    assert json.loads(output.read_text(encoding="utf-8")) == winner
    assert not legacy_temporary.exists()
    assert not list(tmp_path.glob(f".{output.name}.*.tmp"))


@requires_secure_immutable_json_publication
def test_competing_processes_publish_one_immutable_lineage_without_orphans(
    tmp_path: Path,
) -> None:
    first_root = tmp_path / "first"
    second_root = tmp_path / "second"
    first_root.mkdir()
    second_root.mkdir()
    first_spec, _first = _write_evidence_bundle(first_root)
    second_spec, second = _write_evidence_bundle(second_root)
    second["candidate_snapshot_id"] = "geno-lewm-data-v0.3.0-r2"
    _write_json(second_spec, second)
    output = tmp_path / "snapshot-lineage.json"
    go = tmp_path / "go"
    runner = tmp_path / "runner.py"
    runner.write_text(
        """from pathlib import Path
import sys
import time
from tools.data.v03_snapshot_lineage import main

ready, go, spec, lock, output = map(Path, sys.argv[1:])
ready.write_text("ready", encoding="utf-8")
while not go.exists():
    time.sleep(0.001)
raise SystemExit(main([
    "assemble",
    "--spec-json", str(spec),
    "--gnomad-source-lock-json", str(lock),
    "--output-json", str(output),
]))
""",
        encoding="utf-8",
    )
    ready_paths = [tmp_path / "first.ready", tmp_path / "second.ready"]
    processes = [
        subprocess.Popen(
            [
                sys.executable,
                str(runner),
                str(ready),
                str(go),
                str(spec),
                str(SOURCE_LOCK.resolve()),
                str(output),
            ],
            cwd=Path.cwd(),
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        for ready, spec in zip(ready_paths, (first_spec, second_spec), strict=True)
    ]
    deadline = time.monotonic() + 10
    while not all(path.exists() for path in ready_paths):
        assert time.monotonic() < deadline
        time.sleep(0.001)
    go.write_text("go", encoding="utf-8")
    results = [process.communicate(timeout=30) for process in processes]
    returncodes = [process.returncode for process in processes]

    assert sorted(returncodes) == [0, 2]
    winner_index = returncodes.index(0)
    summary = json.loads(results[winner_index][0])
    published = json.loads(output.read_text(encoding="utf-8"))
    assert published["candidate_snapshot_id"] == summary["candidate_snapshot_id"]
    assert published["lineage_id"] == summary["lineage_id"]
    assert "refusing to replace different lineage bytes" in results[1 - winner_index][1]
    assert not list(tmp_path.glob(f".{output.name}.*.tmp"))


def test_assembler_rejects_receipt_byte_tampering(tmp_path: Path) -> None:
    spec_path, _spec = _write_evidence_bundle(tmp_path)
    receipt_path = tmp_path / "receipts" / "chr1.json"
    receipt_path.write_text(
        receipt_path.read_text(encoding="utf-8") + "\n",
        encoding="utf-8",
    )

    with pytest.raises(SnapshotLineageError, match="chr1 receipt bytes drifted"):
        assemble_snapshot_lineage(
            spec_path=spec_path,
            gnomad_source_lock_path=SOURCE_LOCK,
        )


def test_assembler_keeps_source_lock_bound_to_its_first_captured_bytes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    spec_path, _spec = _write_evidence_bundle(tmp_path)
    copied_lock = tmp_path / SOURCE_LOCK.name
    copied_schema = tmp_path / "gnomad-v4.1-exomes-autosomes.source-lock.schema.json"
    original_lock_bytes = SOURCE_LOCK.read_bytes()
    copied_lock.write_bytes(original_lock_bytes)
    copied_schema.write_bytes(
        Path("configs/data_v03/gnomad-v4.1-exomes-autosomes.source-lock.schema.json").read_bytes()
    )
    drifted_lock = json.loads(original_lock_bytes)
    drifted_lock["job"]["namespace_root"] = "staging/v0.3/replaced-after-capture"
    drifted_lock_bytes = (json.dumps(drifted_lock, indent=2, sort_keys=True) + "\n").encode()
    original_read_bytes = Path.read_bytes
    replaced = False

    def _read_then_replace(path: Path) -> bytes:
        nonlocal replaced
        payload = original_read_bytes(path)
        if path.resolve() == copied_lock.resolve() and not replaced:
            replaced = True
            copied_lock.write_bytes(drifted_lock_bytes)
        return payload

    monkeypatch.setattr(Path, "read_bytes", _read_then_replace)

    lineage = assemble_snapshot_lineage(
        spec_path=spec_path,
        gnomad_source_lock_path=copied_lock,
    )

    assert replaced is True
    assert lineage["gnomad"]["source_lock"]["sha256"] == (
        "sha256:" + hashlib.sha256(original_lock_bytes).hexdigest()
    )


def test_assembler_keeps_receipt_semantics_bound_to_hashed_bytes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    spec_path, _spec = _write_evidence_bundle(tmp_path)
    receipt_path = tmp_path / "receipts/chr1.json"
    original = json.loads(receipt_path.read_text(encoding="utf-8"))
    original_streamed_sha256 = original["source"]["streamed_sha256"]
    replacement = json.loads(receipt_path.read_text(encoding="utf-8"))
    replacement["source"]["streamed_sha256"] = "9" * 64
    state = _replace_after_first_binary_read(
        monkeypatch,
        target=receipt_path,
        replacement=replacement,
    )

    lineage = assemble_snapshot_lineage(
        spec_path=spec_path,
        gnomad_source_lock_path=SOURCE_LOCK,
    )

    assert state["replaced"] is True
    assert lineage["gnomad"]["shards"][0]["source"]["streamed_sha256"] == (
        "sha256:" + original_streamed_sha256
    )


def test_assembler_keeps_gnomad_postflight_semantics_bound_to_hashed_bytes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    spec_path, _spec = _write_evidence_bundle(tmp_path)
    postflight_path = tmp_path / "postflights/chr1.json"
    replacement = json.loads(postflight_path.read_text(encoding="utf-8"))
    replacement["checks"] = replacement["checks"][:-1]
    state = _replace_after_first_binary_read(
        monkeypatch,
        target=postflight_path,
        replacement=replacement,
    )

    lineage = assemble_snapshot_lineage(
        spec_path=spec_path,
        gnomad_source_lock_path=SOURCE_LOCK,
    )

    assert state["replaced"] is True
    assert lineage["gnomad"]["shards"][0]["remote_postflight"]["checks"] == list(
        REMOTE_POSTFLIGHT_CHECKS
    )


def test_assembler_keeps_clinvar_audit_semantics_bound_to_hashed_bytes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    spec_path, _spec = _write_evidence_bundle(tmp_path)
    audit_path = tmp_path / "clinvar-audit.json"
    original = json.loads(audit_path.read_text(encoding="utf-8"))
    replacement = json.loads(audit_path.read_text(encoding="utf-8"))
    replacement["source"]["sha256"] = "sha256:" + "9" * 64
    state = _replace_after_first_binary_read(
        monkeypatch,
        target=audit_path,
        replacement=replacement,
    )

    lineage = assemble_snapshot_lineage(
        spec_path=spec_path,
        gnomad_source_lock_path=SOURCE_LOCK,
    )

    assert state["replaced"] is True
    assert lineage["clinvar"]["source"]["sha256"] == original["source"]["sha256"]


def test_assembler_keeps_clinvar_postflight_semantics_bound_to_hashed_bytes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    spec_path, _spec = _write_evidence_bundle(tmp_path)
    postflight_path = tmp_path / "clinvar-postflight.json"
    replacement = json.loads(postflight_path.read_text(encoding="utf-8"))
    replacement["checks"] = replacement["checks"][:-1]
    state = _replace_after_first_binary_read(
        monkeypatch,
        target=postflight_path,
        replacement=replacement,
    )

    lineage = assemble_snapshot_lineage(
        spec_path=spec_path,
        gnomad_source_lock_path=SOURCE_LOCK,
    )

    assert state["replaced"] is True
    assert lineage["clinvar"]["remote_postflight"]["checks"] == list(CLINVAR_REMOTE_CHECKS)


def test_assembler_requires_per_shard_postflight_binding(tmp_path: Path) -> None:
    spec_path, spec = _write_evidence_bundle(tmp_path)
    del spec["gnomad"]["shards"][0]["postflight_file"]
    _write_json(spec_path, spec)

    with pytest.raises(
        SnapshotLineageError,
        match=r"lineage spec\.gnomad\.shards\[0\] keys drifted.*postflight_file",
    ):
        assemble_snapshot_lineage(
            spec_path=spec_path,
            gnomad_source_lock_path=SOURCE_LOCK,
        )


def test_assembler_rejects_postflight_byte_tampering(tmp_path: Path) -> None:
    spec_path, _spec = _write_evidence_bundle(tmp_path)
    postflight_path = tmp_path / "postflights" / "chr1.json"
    postflight_path.write_text(
        postflight_path.read_text(encoding="utf-8") + "\n",
        encoding="utf-8",
    )

    with pytest.raises(SnapshotLineageError, match="chr1 postflight bytes drifted"):
        assemble_snapshot_lineage(
            spec_path=spec_path,
            gnomad_source_lock_path=SOURCE_LOCK,
        )


@pytest.mark.parametrize(
    ("field", "drifted_value"),
    [
        ("repo_id", "other/geno-lewm-data"),
        ("revision", "a" * 40),
        ("namespace", "staging/v0.3/not-chr1"),
        ("source_commit", "f" * 40),
        ("chromosome", "2"),
    ],
)
def test_assembler_cross_checks_postflight_declarations(
    tmp_path: Path,
    field: str,
    drifted_value: str,
) -> None:
    spec_path, spec = _write_evidence_bundle(tmp_path)
    postflight_path = tmp_path / "postflights" / "chr1.json"
    postflight = json.loads(postflight_path.read_text(encoding="utf-8"))
    postflight[field] = drifted_value
    _write_bound_postflight(spec_path, spec, postflight)

    with pytest.raises(SnapshotLineageError, match=rf"postflight\.{field} drifted"):
        assemble_snapshot_lineage(
            spec_path=spec_path,
            gnomad_source_lock_path=SOURCE_LOCK,
        )


def test_assembler_requires_boolean_postflight_ok(tmp_path: Path) -> None:
    spec_path, spec = _write_evidence_bundle(tmp_path)
    postflight_path = tmp_path / "postflights" / "chr1.json"
    postflight = json.loads(postflight_path.read_text(encoding="utf-8"))
    postflight["ok"] = 1
    _write_bound_postflight(spec_path, spec, postflight)

    with pytest.raises(SnapshotLineageError, match=r"postflight\.ok drifted"):
        assemble_snapshot_lineage(
            spec_path=spec_path,
            gnomad_source_lock_path=SOURCE_LOCK,
        )


def test_assembler_requires_boolean_gnomad_receipt_ok(tmp_path: Path) -> None:
    spec_path, spec = _write_evidence_bundle(tmp_path)
    receipt_path = tmp_path / "receipts" / "chr1.json"
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    receipt["ok"] = 1
    _write_json(receipt_path, receipt)
    shard = spec["gnomad"]["shards"][0]
    shard["receipt_sha256"] = sha256_file(receipt_path)
    postflight = _gnomad_postflight(
        receipt=receipt,
        receipt_path=receipt_path,
        chromosome="1",
        revision=shard["revision"],
        namespace=shard["namespace"],
    )
    _write_bound_postflight(spec_path, spec, postflight)

    with pytest.raises(SnapshotLineageError, match=r"receipt\.ok drifted"):
        assemble_snapshot_lineage(
            spec_path=spec_path,
            gnomad_source_lock_path=SOURCE_LOCK,
        )


def test_assembler_requires_boolean_clinvar_audit_ok(tmp_path: Path) -> None:
    spec_path, spec = _write_evidence_bundle(tmp_path)
    audit_path = tmp_path / "clinvar-audit.json"
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    audit["ok"] = 1
    _write_json(audit_path, audit)
    spec["clinvar"]["audit_sha256"] = sha256_file(audit_path)
    _write_json(spec_path, spec)

    with pytest.raises(SnapshotLineageError, match=r"ClinVar audit\.ok drifted"):
        assemble_snapshot_lineage(
            spec_path=spec_path,
            gnomad_source_lock_path=SOURCE_LOCK,
        )


@pytest.mark.parametrize("field", ["postflight_file", "postflight_sha256"])
def test_assembler_requires_clinvar_postflight_binding(tmp_path: Path, field: str) -> None:
    spec_path, spec = _write_evidence_bundle(tmp_path)
    del spec["clinvar"][field]
    _write_json(spec_path, spec)

    with pytest.raises(
        SnapshotLineageError,
        match=rf"lineage spec\.clinvar keys drifted.*{field}",
    ):
        assemble_snapshot_lineage(
            spec_path=spec_path,
            gnomad_source_lock_path=SOURCE_LOCK,
        )


def test_assembler_rejects_clinvar_postflight_byte_tampering(tmp_path: Path) -> None:
    spec_path, _spec = _write_evidence_bundle(tmp_path)
    postflight_path = tmp_path / "clinvar-postflight.json"
    postflight_path.write_text(
        postflight_path.read_text(encoding="utf-8") + "\n",
        encoding="utf-8",
    )

    with pytest.raises(SnapshotLineageError, match="ClinVar postflight bytes drifted"):
        assemble_snapshot_lineage(
            spec_path=spec_path,
            gnomad_source_lock_path=SOURCE_LOCK,
        )


def test_assembler_requires_prefixed_clinvar_postflight_hash(tmp_path: Path) -> None:
    spec_path, spec = _write_evidence_bundle(tmp_path)
    spec["clinvar"]["postflight_sha256"] = spec["clinvar"]["postflight_sha256"].removeprefix(
        "sha256:"
    )
    _write_json(spec_path, spec)

    with pytest.raises(SnapshotLineageError, match="must be a sha256-prefixed lowercase digest"):
        assemble_snapshot_lineage(
            spec_path=spec_path,
            gnomad_source_lock_path=SOURCE_LOCK,
        )


def test_assembler_keeps_clinvar_postflight_inside_bundle(tmp_path: Path) -> None:
    spec_path, spec = _write_evidence_bundle(tmp_path)
    spec["clinvar"]["postflight_file"] = "../outside-postflight.json"
    _write_json(spec_path, spec)

    with pytest.raises(SnapshotLineageError, match=r"relative in-bundle JSON path"):
        assemble_snapshot_lineage(
            spec_path=spec_path,
            gnomad_source_lock_path=SOURCE_LOCK,
        )


def test_assembler_rejects_duplicate_clinvar_postflight_keys(tmp_path: Path) -> None:
    spec_path, spec = _write_evidence_bundle(tmp_path)
    postflight_path = tmp_path / "clinvar-postflight.json"
    raw = postflight_path.read_text(encoding="utf-8").replace(
        '  "ok": true,',
        '  "ok": false,\n  "ok": true,',
    )
    postflight_path.write_text(raw, encoding="utf-8")
    spec["clinvar"]["postflight_sha256"] = sha256_file(postflight_path)
    _write_json(spec_path, spec)

    with pytest.raises(SnapshotLineageError, match=r"duplicate JSON key.*ok"):
        assemble_snapshot_lineage(
            spec_path=spec_path,
            gnomad_source_lock_path=SOURCE_LOCK,
        )


@pytest.mark.parametrize(
    ("field", "drifted_value"),
    [
        ("schema_version", "geno-lewm.clinvar-remote-postflight.v0"),
        ("ok", 1),
        ("repo_id", "other/geno-lewm-data"),
        ("repo_type", "model"),
        ("revision", "a" * 40),
        ("namespace", "staging/clinvar-2026-04-15-archive-other-r1"),
        ("source_commit", "f" * 40),
        ("release", "2026-04-14"),
        ("claim_boundary", "unbounded claim"),
    ],
)
def test_assembler_reconciles_every_clinvar_postflight_declaration(
    tmp_path: Path,
    field: str,
    drifted_value: object,
) -> None:
    spec_path, spec = _write_evidence_bundle(tmp_path)
    postflight = _read_clinvar_postflight(spec_path)
    postflight[field] = drifted_value
    _write_bound_clinvar_postflight(spec_path, spec, postflight)

    with pytest.raises(SnapshotLineageError, match=rf"ClinVar postflight\.{field} drifted"):
        assemble_snapshot_lineage(
            spec_path=spec_path,
            gnomad_source_lock_path=SOURCE_LOCK,
        )


def test_assembler_requires_exact_clinvar_postflight_surface(tmp_path: Path) -> None:
    spec_path, spec = _write_evidence_bundle(tmp_path)
    postflight = _read_clinvar_postflight(spec_path)
    postflight["unreviewed_claim"] = "looks good"
    _write_bound_clinvar_postflight(spec_path, spec, postflight)

    with pytest.raises(
        SnapshotLineageError,
        match=r"ClinVar remote postflight keys drifted.*unreviewed_claim",
    ):
        assemble_snapshot_lineage(
            spec_path=spec_path,
            gnomad_source_lock_path=SOURCE_LOCK,
        )


@pytest.mark.parametrize(
    "verified_files",
    [
        list(CLINVAR_REMOTE_FILES[:-1]),
        [*CLINVAR_REMOTE_FILES, "evidence/unreviewed.json"],
        list(reversed(CLINVAR_REMOTE_FILES)),
    ],
    ids=["missing", "extra", "reordered"],
)
def test_assembler_requires_exact_clinvar_verified_files(
    tmp_path: Path,
    verified_files: list[str],
) -> None:
    spec_path, spec = _write_evidence_bundle(tmp_path)
    postflight = _read_clinvar_postflight(spec_path)
    postflight["verified_files"] = verified_files
    _write_bound_clinvar_postflight(spec_path, spec, postflight)

    with pytest.raises(SnapshotLineageError, match="ClinVar postflight verified file set"):
        assemble_snapshot_lineage(
            spec_path=spec_path,
            gnomad_source_lock_path=SOURCE_LOCK,
        )


def test_assembler_requires_exact_clinvar_file_identity_set(tmp_path: Path) -> None:
    spec_path, spec = _write_evidence_bundle(tmp_path)
    postflight = _read_clinvar_postflight(spec_path)
    del postflight["file_identities"]["evidence/runtime_report.json"]
    _write_bound_clinvar_postflight(spec_path, spec, postflight)

    with pytest.raises(
        SnapshotLineageError,
        match=r"ClinVar postflight\.file_identities keys drifted.*runtime_report",
    ):
        assemble_snapshot_lineage(
            spec_path=spec_path,
            gnomad_source_lock_path=SOURCE_LOCK,
        )


@pytest.mark.parametrize(
    ("field", "drifted_value", "match"),
    [
        ("sha256", "sha256:" + "a" * 64, "must be a lowercase SHA-256 digest"),
        ("size_bytes", 0, "must be a positive integer"),
        ("size_bytes", True, "must be a positive integer"),
    ],
    ids=["prefixed-hash", "zero-size", "boolean-size"],
)
def test_assembler_validates_every_clinvar_file_identity_field(
    tmp_path: Path,
    field: str,
    drifted_value: object,
    match: str,
) -> None:
    spec_path, spec = _write_evidence_bundle(tmp_path)
    postflight = _read_clinvar_postflight(spec_path)
    postflight["file_identities"]["evidence/runtime_report.json"][field] = drifted_value
    _write_bound_clinvar_postflight(spec_path, spec, postflight)

    with pytest.raises(SnapshotLineageError, match=match):
        assemble_snapshot_lineage(
            spec_path=spec_path,
            gnomad_source_lock_path=SOURCE_LOCK,
        )


def test_assembler_requires_closed_clinvar_file_identity_shape(tmp_path: Path) -> None:
    spec_path, spec = _write_evidence_bundle(tmp_path)
    postflight = _read_clinvar_postflight(spec_path)
    postflight["file_identities"]["evidence/runtime_report.json"]["etag"] = "untrusted"
    _write_bound_clinvar_postflight(spec_path, spec, postflight)

    with pytest.raises(SnapshotLineageError, match=r"runtime_report.*keys drifted.*etag"):
        assemble_snapshot_lineage(
            spec_path=spec_path,
            gnomad_source_lock_path=SOURCE_LOCK,
        )


@pytest.mark.parametrize(
    ("relative_path", "field", "drifted_value", "match"),
    [
        (
            "evidence/audit.json",
            "sha256",
            "0" * 64,
            "ClinVar postflight audit identity drifted",
        ),
        (
            "evidence/audit.json",
            "size_bytes",
            1,
            "ClinVar postflight audit identity drifted",
        ),
        (
            "clinvar/2026-04-15/variants.parquet",
            "sha256",
            "0" * 64,
            "ClinVar postflight Parquet identity drifted",
        ),
        (
            "clinvar/2026-04-15/variants.parquet",
            "size_bytes",
            1,
            "ClinVar postflight Parquet identity drifted",
        ),
    ],
    ids=["audit-hash", "audit-size", "parquet-hash", "parquet-size"],
)
def test_assembler_reconciles_clinvar_bound_file_identities(
    tmp_path: Path,
    relative_path: str,
    field: str,
    drifted_value: object,
    match: str,
) -> None:
    spec_path, spec = _write_evidence_bundle(tmp_path)
    postflight = _read_clinvar_postflight(spec_path)
    postflight["file_identities"][relative_path][field] = drifted_value
    _write_bound_clinvar_postflight(spec_path, spec, postflight)

    with pytest.raises(SnapshotLineageError, match=match):
        assemble_snapshot_lineage(
            spec_path=spec_path,
            gnomad_source_lock_path=SOURCE_LOCK,
        )


@pytest.mark.parametrize(
    "checks",
    [
        list(CLINVAR_REMOTE_CHECKS[:-1]),
        [*CLINVAR_REMOTE_CHECKS, "unreviewed_check"],
        list(reversed(CLINVAR_REMOTE_CHECKS)),
    ],
    ids=["missing", "extra", "reordered"],
)
def test_assembler_requires_exact_clinvar_postflight_checks(
    tmp_path: Path,
    checks: list[str],
) -> None:
    spec_path, spec = _write_evidence_bundle(tmp_path)
    postflight = _read_clinvar_postflight(spec_path)
    postflight["checks"] = checks
    _write_bound_clinvar_postflight(spec_path, spec, postflight)

    with pytest.raises(SnapshotLineageError, match="ClinVar postflight checks drifted"):
        assemble_snapshot_lineage(
            spec_path=spec_path,
            gnomad_source_lock_path=SOURCE_LOCK,
        )


@pytest.mark.parametrize(
    ("field", "drifted_value"),
    [
        ("schema_version", "1.0.1"),
        ("parquet_schema", list(reversed(CLINVAR_PARQUET_SCHEMA))),
        ("nullable_fields", []),
        ("normalized_classes", ["VUS", "P", "OTHER", "LP", "LB", "B"]),
        ("labelled_classes", ["P", "LP", "LB", "B"]),
        ("allele_alphabet", ["T", "G", "C", "A"]),
        ("cli_command", "python -m unreviewed"),
        ("max_allele_len", 16.0),
        ("output_path_template", "clinvar/{release}/unreviewed.parquet"),
        ("prepare_report_enrichments", ["runtime", "output_parquet", "input_vcf", "command"]),
        ("file_identity_fields", ["size_bytes", "sha256", "path"]),
    ],
)
def test_assembler_locks_every_clinvar_trusted_contract_field(
    tmp_path: Path,
    field: str,
    drifted_value: object,
) -> None:
    spec_path, spec = _write_evidence_bundle(tmp_path)
    postflight = _read_clinvar_postflight(spec_path)
    postflight["trusted_source_contract"][field] = drifted_value
    _write_bound_clinvar_postflight(spec_path, spec, postflight)

    with pytest.raises(SnapshotLineageError, match=rf"ClinVar trusted source {field} drifted"):
        assemble_snapshot_lineage(
            spec_path=spec_path,
            gnomad_source_lock_path=SOURCE_LOCK,
        )


def test_assembler_requires_exact_clinvar_trusted_source_file_set(tmp_path: Path) -> None:
    spec_path, spec = _write_evidence_bundle(tmp_path)
    postflight = _read_clinvar_postflight(spec_path)
    del postflight["trusted_source_contract"]["files"]["geno_lewm/data/clinvar.py"]
    _write_bound_clinvar_postflight(spec_path, spec, postflight)

    with pytest.raises(
        SnapshotLineageError,
        match=r"ClinVar trusted source files keys drifted.*geno_lewm/data/clinvar.py",
    ):
        assemble_snapshot_lineage(
            spec_path=spec_path,
            gnomad_source_lock_path=SOURCE_LOCK,
        )


@pytest.mark.parametrize(
    ("field", "drifted_value", "match"),
    [
        ("sha256", "sha256:" + "a" * 64, "must be a lowercase SHA-256 digest"),
        ("size_bytes", False, "must be a positive integer"),
    ],
    ids=["prefixed-hash", "boolean-size"],
)
def test_assembler_validates_clinvar_trusted_source_file_identities(
    tmp_path: Path,
    field: str,
    drifted_value: object,
    match: str,
) -> None:
    spec_path, spec = _write_evidence_bundle(tmp_path)
    postflight = _read_clinvar_postflight(spec_path)
    identity = postflight["trusted_source_contract"]["files"]["geno_lewm/data/clinvar.py"]
    identity[field] = drifted_value
    _write_bound_clinvar_postflight(spec_path, spec, postflight)

    with pytest.raises(SnapshotLineageError, match=match):
        assemble_snapshot_lineage(
            spec_path=spec_path,
            gnomad_source_lock_path=SOURCE_LOCK,
        )


def test_assembler_requires_closed_clinvar_trusted_source_identity_shape(
    tmp_path: Path,
) -> None:
    spec_path, spec = _write_evidence_bundle(tmp_path)
    postflight = _read_clinvar_postflight(spec_path)
    identity = postflight["trusted_source_contract"]["files"]["geno_lewm/data/clinvar.py"]
    identity["path"] = "untrusted"
    _write_bound_clinvar_postflight(spec_path, spec, postflight)

    with pytest.raises(SnapshotLineageError, match=r"geno_lewm/data/clinvar.py.*keys drifted"):
        assemble_snapshot_lineage(
            spec_path=spec_path,
            gnomad_source_lock_path=SOURCE_LOCK,
        )


@pytest.mark.parametrize(
    ("field", "drifted_value"),
    [
        ("url", "https://example.invalid/clinvar.vcf.gz"),
        ("release", "2026-04-14"),
        ("md5", "0" * 32),
        ("sha256", "0" * 64),
        ("size_bytes", 1),
        ("verification_scope", ["size_bytes_reconciled", "sha256_reconciled"]),
        (
            "verification_limitation",
            "The postflight recomputed every upstream source identity from archived bytes.",
        ),
    ],
)
def test_assembler_reconciles_every_clinvar_source_identity_field(
    tmp_path: Path,
    field: str,
    drifted_value: object,
) -> None:
    spec_path, spec = _write_evidence_bundle(tmp_path)
    postflight = _read_clinvar_postflight(spec_path)
    postflight["source_identity"][field] = drifted_value
    _write_bound_clinvar_postflight(spec_path, spec, postflight)

    with pytest.raises(SnapshotLineageError, match="ClinVar postflight source identity drifted"):
        assemble_snapshot_lineage(
            spec_path=spec_path,
            gnomad_source_lock_path=SOURCE_LOCK,
        )


def test_assembler_requires_closed_clinvar_source_identity_shape(tmp_path: Path) -> None:
    spec_path, spec = _write_evidence_bundle(tmp_path)
    postflight = _read_clinvar_postflight(spec_path)
    postflight["source_identity"]["archive_bytes_recomputed"] = True
    _write_bound_clinvar_postflight(spec_path, spec, postflight)

    with pytest.raises(
        SnapshotLineageError,
        match=r"ClinVar postflight\.source_identity keys drifted.*archive_bytes_recomputed",
    ):
        assemble_snapshot_lineage(
            spec_path=spec_path,
            gnomad_source_lock_path=SOURCE_LOCK,
        )


@pytest.mark.parametrize(
    ("field", "drifted_value"),
    [
        ("path", "clinvar/2026-04-15/unreviewed.parquet"),
        ("sha256", "sha256:" + "b" * 64),
        ("size_bytes", 1),
        ("records", 40),
        (
            "class_balance",
            {"B": 1, "LB": 3, "LP": 5, "OTHER": 7, "P": 12, "VUS": 13},
        ),
    ],
)
def test_assembler_reconciles_every_clinvar_output_identity_field(
    tmp_path: Path,
    field: str,
    drifted_value: object,
) -> None:
    spec_path, spec = _write_evidence_bundle(tmp_path)
    postflight = _read_clinvar_postflight(spec_path)
    postflight["output_identity"][field] = drifted_value
    _write_bound_clinvar_postflight(spec_path, spec, postflight)

    with pytest.raises(SnapshotLineageError, match="ClinVar postflight output identity drifted"):
        assemble_snapshot_lineage(
            spec_path=spec_path,
            gnomad_source_lock_path=SOURCE_LOCK,
        )


def test_assembler_requires_closed_clinvar_output_identity_shape(tmp_path: Path) -> None:
    spec_path, spec = _write_evidence_bundle(tmp_path)
    postflight = _read_clinvar_postflight(spec_path)
    postflight["output_identity"]["schema_version"] = "1.0.0"
    _write_bound_clinvar_postflight(spec_path, spec, postflight)

    with pytest.raises(
        SnapshotLineageError,
        match=r"ClinVar postflight\.output_identity keys drifted.*schema_version",
    ):
        assemble_snapshot_lineage(
            spec_path=spec_path,
            gnomad_source_lock_path=SOURCE_LOCK,
        )


@pytest.mark.parametrize(
    ("field", "drifted_value", "match"),
    [
        ("metadata_row_count", 40, "metadata row count drifted"),
        ("scanned_row_count", 40, "scanned row count drifted"),
        (
            "class_balance",
            {"B": 1, "LB": 3, "LP": 5, "OTHER": 7, "P": 12, "VUS": 13},
            "class balance drifted",
        ),
        ("chromosome_balance", {"1": 40}, "chromosome total drifted"),
        ("schema_version_balance", {"1.0.0": 40}, "schema-version balance drifted"),
        (
            "null_counts",
            {"chrom": 0},
            "null counts keys drifted",
        ),
        ("position_range", {"min": 0, "max": 1}, "position_range.min"),
        ("clinvar_id_range", {"min": 2, "max": 1}, "clinvar_id_range.min exceeds max"),
        ("schema", list(reversed(CLINVAR_PARQUET_SCHEMA)), "Parquet schema drifted"),
    ],
)
def test_assembler_validates_every_clinvar_recomputed_audit_field(
    tmp_path: Path,
    field: str,
    drifted_value: object,
    match: str,
) -> None:
    spec_path, spec = _write_evidence_bundle(tmp_path)
    postflight = _read_clinvar_postflight(spec_path)
    postflight["parquet_audit"][field] = drifted_value
    _write_bound_clinvar_postflight(spec_path, spec, postflight)

    with pytest.raises(SnapshotLineageError, match=match):
        assemble_snapshot_lineage(
            spec_path=spec_path,
            gnomad_source_lock_path=SOURCE_LOCK,
        )


def test_assembler_requires_closed_clinvar_recomputed_audit_shape(tmp_path: Path) -> None:
    spec_path, spec = _write_evidence_bundle(tmp_path)
    postflight = _read_clinvar_postflight(spec_path)
    postflight["parquet_audit"]["audit_method"] = "metadata-only"
    _write_bound_clinvar_postflight(spec_path, spec, postflight)

    with pytest.raises(
        SnapshotLineageError,
        match=r"ClinVar postflight\.parquet_audit keys drifted.*audit_method",
    ):
        assemble_snapshot_lineage(
            spec_path=spec_path,
            gnomad_source_lock_path=SOURCE_LOCK,
        )


@pytest.mark.parametrize("field", [field["name"] for field in CLINVAR_PARQUET_SCHEMA])
def test_assembler_rejects_boolean_clinvar_null_counts(tmp_path: Path, field: str) -> None:
    spec_path, spec = _write_evidence_bundle(tmp_path)
    postflight = _read_clinvar_postflight(spec_path)
    postflight["parquet_audit"]["null_counts"][field] = False
    _write_bound_clinvar_postflight(spec_path, spec, postflight)

    with pytest.raises(SnapshotLineageError, match=rf"null count {field} must be"):
        assemble_snapshot_lineage(
            spec_path=spec_path,
            gnomad_source_lock_path=SOURCE_LOCK,
        )


@pytest.mark.parametrize(
    "field",
    [field["name"] for field in CLINVAR_PARQUET_SCHEMA if field["name"] != "gene_symbol"],
)
def test_assembler_rejects_nulls_in_required_clinvar_fields(tmp_path: Path, field: str) -> None:
    spec_path, spec = _write_evidence_bundle(tmp_path)
    postflight = _read_clinvar_postflight(spec_path)
    postflight["parquet_audit"]["null_counts"][field] = 1
    _write_bound_clinvar_postflight(spec_path, spec, postflight)

    with pytest.raises(SnapshotLineageError, match=rf"required field '{field}' contains nulls"):
        assemble_snapshot_lineage(
            spec_path=spec_path,
            gnomad_source_lock_path=SOURCE_LOCK,
        )


def test_assembler_rejects_clinvar_nullable_count_above_records(tmp_path: Path) -> None:
    spec_path, spec = _write_evidence_bundle(tmp_path)
    postflight = _read_clinvar_postflight(spec_path)
    postflight["parquet_audit"]["null_counts"]["gene_symbol"] = 42
    _write_bound_clinvar_postflight(spec_path, spec, postflight)

    with pytest.raises(SnapshotLineageError, match=r"null count 'gene_symbol' exceeds records"):
        assemble_snapshot_lineage(
            spec_path=spec_path,
            gnomad_source_lock_path=SOURCE_LOCK,
        )


@pytest.mark.parametrize("range_field", ["position_range", "clinvar_id_range"])
@pytest.mark.parametrize(
    ("minimum", "maximum", "match"),
    [
        (0, 1, r"\.min must be a positive integer"),
        (True, 1, r"\.min must be a positive integer"),
        (1, 1.5, r"\.max must be a positive integer"),
        (2, 1, r"\.min exceeds max"),
    ],
    ids=["zero", "boolean", "float", "inverted"],
)
def test_assembler_validates_clinvar_positive_ranges(
    tmp_path: Path,
    range_field: str,
    minimum: object,
    maximum: object,
    match: str,
) -> None:
    spec_path, spec = _write_evidence_bundle(tmp_path)
    postflight = _read_clinvar_postflight(spec_path)
    postflight["parquet_audit"][range_field] = {"min": minimum, "max": maximum}
    _write_bound_clinvar_postflight(spec_path, spec, postflight)

    with pytest.raises(SnapshotLineageError, match=rf"{range_field}{match}"):
        assemble_snapshot_lineage(
            spec_path=spec_path,
            gnomad_source_lock_path=SOURCE_LOCK,
        )


@pytest.mark.parametrize(
    ("chromosome_balance", "match"),
    [
        ({}, "must not be empty"),
        ({"": 41}, "names must be non-empty"),
        ({"1": 0, "2": 41}, "must be a positive integer"),
        ({"1": True, "2": 40}, "must be a positive integer"),
        ({"1": 40}, "chromosome total drifted"),
    ],
    ids=["empty", "empty-name", "zero-count", "boolean-count", "wrong-total"],
)
def test_assembler_validates_clinvar_chromosome_balance(
    tmp_path: Path,
    chromosome_balance: dict[str, object],
    match: str,
) -> None:
    spec_path, spec = _write_evidence_bundle(tmp_path)
    postflight = _read_clinvar_postflight(spec_path)
    postflight["parquet_audit"]["chromosome_balance"] = chromosome_balance
    _write_bound_clinvar_postflight(spec_path, spec, postflight)

    with pytest.raises(SnapshotLineageError, match=match):
        assemble_snapshot_lineage(
            spec_path=spec_path,
            gnomad_source_lock_path=SOURCE_LOCK,
        )


def test_assembler_preserves_noncanonical_clinvar_chromosome_scope(tmp_path: Path) -> None:
    spec_path, spec = _write_evidence_bundle(tmp_path)
    postflight = _read_clinvar_postflight(spec_path)
    postflight["parquet_audit"]["chromosome_balance"] = {"chrUn_KI270442v1": 41}
    _write_bound_clinvar_postflight(spec_path, spec, postflight)

    lineage = assemble_snapshot_lineage(
        spec_path=spec_path,
        gnomad_source_lock_path=SOURCE_LOCK,
    )

    assert lineage["clinvar"]["remote_postflight"]["parquet_audit"]["chromosome_balance"] == {
        "chrUn_KI270442v1": 41
    }


def test_assembler_compares_postflight_audit_json_types_exactly(tmp_path: Path) -> None:
    spec_path, spec = _write_evidence_bundle(tmp_path)
    postflight_path = tmp_path / "postflights" / "chr1.json"
    postflight = json.loads(postflight_path.read_text(encoding="utf-8"))
    postflight["parquet_audit"]["position_min"] = True
    _write_bound_postflight(spec_path, spec, postflight)

    with pytest.raises(SnapshotLineageError, match="postflight fresh Parquet audit drifted"):
        assemble_snapshot_lineage(
            spec_path=spec_path,
            gnomad_source_lock_path=SOURCE_LOCK,
        )


def test_assembler_requires_exact_postflight_verified_file_set(tmp_path: Path) -> None:
    spec_path, spec = _write_evidence_bundle(tmp_path)
    postflight_path = tmp_path / "postflights" / "chr1.json"
    postflight = json.loads(postflight_path.read_text(encoding="utf-8"))
    postflight["verified_files"].pop()
    _write_bound_postflight(spec_path, spec, postflight)

    with pytest.raises(SnapshotLineageError, match="postflight verified file set drifted"):
        assemble_snapshot_lineage(
            spec_path=spec_path,
            gnomad_source_lock_path=SOURCE_LOCK,
        )


def test_assembler_binds_postflight_to_local_receipt_identity(tmp_path: Path) -> None:
    spec_path, spec = _write_evidence_bundle(tmp_path)
    postflight_path = tmp_path / "postflights" / "chr1.json"
    postflight = json.loads(postflight_path.read_text(encoding="utf-8"))
    postflight["file_identities"]["evidence/receipt.json"]["sha256"] = "0" * 64
    _write_bound_postflight(spec_path, spec, postflight)

    with pytest.raises(SnapshotLineageError, match="postflight receipt identity drifted"):
        assemble_snapshot_lineage(
            spec_path=spec_path,
            gnomad_source_lock_path=SOURCE_LOCK,
        )


def test_assembler_binds_postflight_to_receipt_parquet_identity(tmp_path: Path) -> None:
    spec_path, spec = _write_evidence_bundle(tmp_path)
    postflight_path = tmp_path / "postflights" / "chr1.json"
    postflight = json.loads(postflight_path.read_text(encoding="utf-8"))
    parquet = postflight["file_identities"]["data/gnomad/v4.1/variants.parquet"]
    parquet["sha256"] = "0" * 64
    _write_bound_postflight(spec_path, spec, postflight)

    with pytest.raises(SnapshotLineageError, match="postflight Parquet identity drifted"):
        assemble_snapshot_lineage(
            spec_path=spec_path,
            gnomad_source_lock_path=SOURCE_LOCK,
        )


def test_assembler_binds_fresh_postflight_audit_to_receipt(tmp_path: Path) -> None:
    spec_path, spec = _write_evidence_bundle(tmp_path)
    postflight_path = tmp_path / "postflights" / "chr1.json"
    postflight = json.loads(postflight_path.read_text(encoding="utf-8"))
    postflight["parquet_audit"]["scanned_row_count"] += 1
    _write_bound_postflight(spec_path, spec, postflight)

    with pytest.raises(SnapshotLineageError, match="postflight fresh Parquet audit drifted"):
        assemble_snapshot_lineage(
            spec_path=spec_path,
            gnomad_source_lock_path=SOURCE_LOCK,
        )


def test_assembler_requires_exact_postflight_checks(tmp_path: Path) -> None:
    spec_path, spec = _write_evidence_bundle(tmp_path)
    postflight_path = tmp_path / "postflights" / "chr1.json"
    postflight = json.loads(postflight_path.read_text(encoding="utf-8"))
    postflight["checks"].pop()
    _write_bound_postflight(spec_path, spec, postflight)

    with pytest.raises(SnapshotLineageError, match="postflight checks drifted"):
        assemble_snapshot_lineage(
            spec_path=spec_path,
            gnomad_source_lock_path=SOURCE_LOCK,
        )


def test_assembler_rejects_cross_receipt_execution_drift(tmp_path: Path) -> None:
    spec_path, spec = _write_evidence_bundle(tmp_path)
    selection = select_source(
        SOURCE_LOCK,
        chromosome="22",
        commit_sha="f" * 40,
        container_image=CONTAINER_IMAGE,
    )
    receipt_path = tmp_path / "receipts" / "chr22.json"
    receipt = _gnomad_receipt(selection, 22)
    _write_json(receipt_path, receipt)
    shard = spec["gnomad"]["shards"][21]
    shard["namespace"] = selection["publication"]["namespace"]
    shard["receipt_sha256"] = sha256_file(receipt_path)
    postflight_path = tmp_path / "postflights" / "chr22.json"
    _write_json(
        postflight_path,
        _gnomad_postflight(
            receipt=receipt,
            receipt_path=receipt_path,
            chromosome="22",
            revision=shard["revision"],
            namespace=shard["namespace"],
        ),
    )
    shard["postflight_sha256"] = sha256_file(postflight_path)
    _write_json(spec_path, spec)

    with pytest.raises(SnapshotLineageError, match="common execution identity drifted"):
        assemble_snapshot_lineage(
            spec_path=spec_path,
            gnomad_source_lock_path=SOURCE_LOCK,
        )


def test_assembler_reconciles_clinvar_class_balance(tmp_path: Path) -> None:
    spec_path, spec = _write_evidence_bundle(tmp_path)
    audit_path = tmp_path / "clinvar-audit.json"
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    audit["output"]["class_balance"]["VUS"] += 1
    _write_json(audit_path, audit)
    spec["clinvar"]["audit_sha256"] = sha256_file(audit_path)
    _write_json(spec_path, spec)

    with pytest.raises(SnapshotLineageError, match="ClinVar class-balance total drifted"):
        assemble_snapshot_lineage(
            spec_path=spec_path,
            gnomad_source_lock_path=SOURCE_LOCK,
        )


def test_assembler_reconciles_all_clinvar_source_checksums(tmp_path: Path) -> None:
    spec_path, spec = _write_evidence_bundle(tmp_path)
    audit_path = tmp_path / "clinvar-audit.json"
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    audit["prepare_report"]["input_vcf"]["sha256"] = "sha256:" + "0" * 64
    _write_json(audit_path, audit)
    spec["clinvar"]["audit_sha256"] = sha256_file(audit_path)
    _write_json(spec_path, spec)

    with pytest.raises(SnapshotLineageError, match=r"ClinVar input_vcf\.sha256 drifted"):
        assemble_snapshot_lineage(
            spec_path=spec_path,
            gnomad_source_lock_path=SOURCE_LOCK,
        )


@requires_secure_immutable_json_publication
def test_lineage_verifier_recomputes_the_content_id_and_split_semantics(tmp_path: Path) -> None:
    spec_path, _spec = _write_evidence_bundle(tmp_path)
    output_path = tmp_path / "lineage.json"
    expected = assemble_snapshot_lineage(
        spec_path=spec_path,
        gnomad_source_lock_path=SOURCE_LOCK,
        output_path=output_path,
    )

    assert verify_snapshot_lineage(output_path) == expected

    stale = copy.deepcopy(expected)
    stale["candidate_snapshot_id"] = "geno-lewm-data-v0.3.0-r2"
    _write_json(output_path, stale)
    with pytest.raises(SnapshotLineageError, match="lineage_id drifted"):
        verify_snapshot_lineage(output_path)

    wrong_role = copy.deepcopy(expected)
    wrong_role["gnomad"]["shards"][19]["split_role"] = "train"
    commitment = dict(wrong_role)
    del commitment["lineage_id"]
    wrong_role["lineage_id"] = canonical_json_sha256(commitment)
    _write_json(output_path, wrong_role)
    with pytest.raises(SnapshotLineageError, match=r"chr20 split_role drifted"):
        verify_snapshot_lineage(output_path)

    wrong_total = copy.deepcopy(expected)
    wrong_total["gnomad"]["total_records"] += 1
    commitment = dict(wrong_total)
    del commitment["lineage_id"]
    wrong_total["lineage_id"] = canonical_json_sha256(commitment)
    _write_json(output_path, wrong_total)
    with pytest.raises(SnapshotLineageError, match="gnomAD total_records drifted"):
        verify_snapshot_lineage(output_path)


def test_verified_lineage_capture_preserves_the_exact_single_read_bytes(tmp_path: Path) -> None:
    spec_path, _spec = _write_evidence_bundle(tmp_path)
    output_path = tmp_path / "lineage.json"
    expected = assemble_snapshot_lineage(
        spec_path=spec_path,
        gnomad_source_lock_path=SOURCE_LOCK,
    )
    payload = ("\n" + json.dumps(expected, separators=(",", ":")) + "\n").encode()
    output_path.write_bytes(payload)

    captured = capture_verified_snapshot_lineage(output_path)

    assert captured.payload == payload
    assert captured.payload_sha256 == "sha256:" + hashlib.sha256(payload).hexdigest()
    assert captured.size_bytes == len(payload)
    assert verify_snapshot_lineage(output_path) == expected


@requires_secure_immutable_json_publication
def test_verified_lineage_capture_is_deeply_immutable(tmp_path: Path) -> None:
    spec_path, _spec = _write_evidence_bundle(tmp_path)
    output_path = tmp_path / "lineage.json"
    assemble_snapshot_lineage(
        spec_path=spec_path,
        gnomad_source_lock_path=SOURCE_LOCK,
        output_path=output_path,
    )

    captured = capture_verified_snapshot_lineage(output_path)
    gnomad = captured.lineage["gnomad"]
    assert isinstance(gnomad, Mapping)
    shards = gnomad["shards"]
    assert isinstance(shards, tuple)
    first = shards[0]
    assert isinstance(first, Mapping)

    with pytest.raises(TypeError):
        captured.lineage["membership_status"] = "created"  # type: ignore[index]
    with pytest.raises(TypeError):
        first["chromosome"] = "22"  # type: ignore[index]
    with pytest.raises(AttributeError):
        shards.append(first)  # type: ignore[attr-defined]


@pytest.mark.parametrize("constant", ["NaN", "Infinity", "-Infinity", "1e999"])
def test_verify_cli_rejects_nonfinite_json_constants_cleanly(
    constant: str,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    output_path = tmp_path / "lineage.json"
    output_path.write_text('{"value": ' + constant + "}\n", encoding="utf-8")

    assert main(["verify", "--lineage-json", str(output_path)]) == 2
    captured = capsys.readouterr()
    assert captured.out == ""
    assert f"non-finite JSON number is not allowed: {constant}" in captured.err


def test_lineage_verifier_cross_checks_gnomad_output_against_postflight_identity(
    tmp_path: Path,
) -> None:
    spec_path, _spec = _write_evidence_bundle(tmp_path)
    output_path = tmp_path / "lineage.json"
    lineage = assemble_snapshot_lineage(
        spec_path=spec_path,
        gnomad_source_lock_path=SOURCE_LOCK,
    )
    shard = lineage["gnomad"]["shards"][0]
    assert shard["remote_postflight"]["file_identities"]["data/gnomad/v4.1/variants.parquet"] == {
        "sha256": shard["output"]["sha256"],
        "size_bytes": shard["output"]["size_bytes"],
    }
    shard["output"]["sha256"] = "sha256:" + "0" * 64
    _recommit_lineage(lineage)
    _write_json(output_path, lineage)

    with pytest.raises(SnapshotLineageError, match="postflight Parquet identity drifted"):
        verify_snapshot_lineage(output_path)


def test_lineage_verifier_cross_checks_clinvar_output_and_audit_postflight_identities(
    tmp_path: Path,
) -> None:
    spec_path, _spec = _write_evidence_bundle(tmp_path)
    output_path = tmp_path / "lineage.json"
    lineage = assemble_snapshot_lineage(
        spec_path=spec_path,
        gnomad_source_lock_path=SOURCE_LOCK,
    )
    identities = lineage["clinvar"]["remote_postflight"]["file_identities"]
    assert identities["evidence/audit.json"] == {
        "sha256": lineage["clinvar"]["audit"]["sha256"],
        "size_bytes": lineage["clinvar"]["audit"]["size_bytes"],
    }
    assert identities["clinvar/2026-04-15/variants.parquet"] == {
        "sha256": lineage["clinvar"]["output"]["sha256"],
        "size_bytes": lineage["clinvar"]["output"]["size_bytes"],
    }
    lineage["clinvar"]["audit"]["sha256"] = "sha256:" + "0" * 64
    _recommit_lineage(lineage)
    _write_json(output_path, lineage)

    with pytest.raises(SnapshotLineageError, match="postflight audit identity drifted"):
        verify_snapshot_lineage(output_path)


def test_lineage_verifier_requires_the_exact_clinvar_archive_url(tmp_path: Path) -> None:
    spec_path, _spec = _write_evidence_bundle(tmp_path)
    output_path = tmp_path / "lineage.json"
    lineage = assemble_snapshot_lineage(
        spec_path=spec_path,
        gnomad_source_lock_path=SOURCE_LOCK,
    )
    assert lineage["clinvar"]["source"]["url"] == CLINVAR_ARCHIVE_URL
    lineage["clinvar"]["source"]["url"] = (
        "https://attacker.invalid/vcf_GRCh38/clinvar_20260415.vcf.gz"
    )
    _recommit_lineage(lineage)
    _write_json(output_path, lineage)

    with pytest.raises(SnapshotLineageError, match=r"ClinVar source\.url drifted"):
        verify_snapshot_lineage(output_path)


def test_lineage_verifier_requires_canonical_numeric_gnomad_shard_order(tmp_path: Path) -> None:
    spec_path, _spec = _write_evidence_bundle(tmp_path)
    output_path = tmp_path / "lineage.json"
    lineage = assemble_snapshot_lineage(
        spec_path=spec_path,
        gnomad_source_lock_path=SOURCE_LOCK,
    )
    shards = lineage["gnomad"]["shards"]
    shards[0], shards[1] = shards[1], shards[0]
    _recommit_lineage(lineage)
    _write_json(output_path, lineage)

    with pytest.raises(SnapshotLineageError, match="canonical chromosome order"):
        verify_snapshot_lineage(output_path)


@pytest.mark.parametrize(
    ("field", "value", "error"),
    [
        ("membership_status", "created", "membership_status drifted"),
        ("claim_boundary", "Lineage verified.", "claim_boundary drifted"),
    ],
)
def test_lineage_verifier_rejects_recommitted_claim_expansion(
    tmp_path: Path,
    field: str,
    value: str,
    error: str,
) -> None:
    spec_path, _spec = _write_evidence_bundle(tmp_path)
    output_path = tmp_path / "lineage.json"
    lineage = assemble_snapshot_lineage(
        spec_path=spec_path,
        gnomad_source_lock_path=SOURCE_LOCK,
    )
    lineage[field] = value
    commitment = dict(lineage)
    del commitment["lineage_id"]
    lineage["lineage_id"] = canonical_json_sha256(commitment)
    _write_json(output_path, lineage)

    with pytest.raises(SnapshotLineageError, match=error):
        verify_snapshot_lineage(output_path)


@requires_secure_immutable_json_publication
def test_verify_cli_emits_a_machine_readable_summary_and_rejects_stale_ids(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    spec_path, _spec = _write_evidence_bundle(tmp_path)
    output_path = tmp_path / "lineage.json"
    lineage = assemble_snapshot_lineage(
        spec_path=spec_path,
        gnomad_source_lock_path=SOURCE_LOCK,
        output_path=output_path,
    )

    assert main(["verify", "--lineage-json", str(output_path)]) == 0
    captured = capsys.readouterr()
    summary = json.loads(captured.out)
    assert summary == {
        "candidate_snapshot_id": lineage["candidate_snapshot_id"],
        "input_json": str(output_path),
        "lineage_id": lineage["lineage_id"],
        "membership_status": "not_created",
        "ok": True,
    }
    assert captured.err == ""

    stale = copy.deepcopy(lineage)
    stale["claim_boundary"] = LINEAGE_CLAIM_BOUNDARY + " Expanded."
    _write_json(output_path, stale)
    assert main(["verify", "--lineage-json", str(output_path)]) == 2
    captured = capsys.readouterr()
    assert captured.out == ""
    assert "lineage_id drifted" in captured.err


@requires_secure_immutable_json_publication
def test_cli_writes_machine_readable_lineage_summary(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    spec_path, _spec = _write_evidence_bundle(tmp_path)
    output_path = tmp_path / "lineage.json"

    rc = main(
        [
            "assemble",
            "--spec-json",
            str(spec_path),
            "--gnomad-source-lock-json",
            str(SOURCE_LOCK),
            "--output-json",
            str(output_path),
        ]
    )
    captured = capsys.readouterr()

    assert rc == 0
    summary = json.loads(captured.out)
    assert summary["ok"] is True
    assert summary["membership_status"] == "not_created"
    assert summary["lineage_id"].startswith("sha256:")
    assert summary["output_json"] == str(output_path)
    assert captured.err == ""
    assert output_path.is_file()


def test_cli_reports_unsupported_secure_publication_without_creating_output_parent(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    spec_path, _spec = _write_evidence_bundle(tmp_path)
    output_path = tmp_path / "publication" / "lineage.json"
    monkeypatch.setattr(
        "tools.data._immutable_json.supports_secure_immutable_json_publication",
        lambda: False,
    )

    rc = main(
        [
            "assemble",
            "--spec-json",
            str(spec_path),
            "--gnomad-source-lock-json",
            str(SOURCE_LOCK),
            "--output-json",
            str(output_path),
        ]
    )
    captured = capsys.readouterr()

    assert rc == 2
    assert captured.out == ""
    assert "requires anchored dir_fd operations; this platform is unsupported" in captured.err
    assert not output_path.parent.exists()


def test_cli_fails_closed_without_partial_output(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    spec_path, spec = _write_evidence_bundle(tmp_path)
    spec["gnomad"]["shards"].pop()
    _write_json(spec_path, spec)
    output_path = tmp_path / "lineage.json"

    rc = main(
        [
            "assemble",
            "--spec-json",
            str(spec_path),
            "--gnomad-source-lock-json",
            str(SOURCE_LOCK),
            "--output-json",
            str(output_path),
        ]
    )
    captured = capsys.readouterr()

    assert rc == 2
    assert captured.out == ""
    assert "must contain exactly 22 entries" in captured.err
    assert not output_path.exists()


def _write_evidence_bundle(tmp_path: Path) -> tuple[Path, dict[str, Any]]:
    receipts_dir = tmp_path / "receipts"
    postflights_dir = tmp_path / "postflights"
    receipts_dir.mkdir()
    postflights_dir.mkdir()
    shards: list[dict[str, str]] = []
    for chromosome in range(1, 23):
        chromosome_text = str(chromosome)
        selection = select_source(
            SOURCE_LOCK,
            chromosome=chromosome_text,
            commit_sha=CODE_COMMIT,
            container_image=CONTAINER_IMAGE,
        )
        revision = f"{chromosome:040x}"
        namespace = str(selection["publication"]["namespace"])
        receipt = _gnomad_receipt(selection, chromosome)
        receipt_path = receipts_dir / f"chr{chromosome_text}.json"
        _write_json(receipt_path, receipt)
        postflight_path = postflights_dir / f"chr{chromosome_text}.json"
        _write_json(
            postflight_path,
            _gnomad_postflight(
                receipt=receipt,
                receipt_path=receipt_path,
                chromosome=chromosome_text,
                revision=revision,
                namespace=namespace,
            ),
        )
        shards.append(
            {
                "chromosome": chromosome_text,
                "split_role": str(selection["source"]["split_role"]),
                "revision": revision,
                "namespace": namespace,
                "receipt_file": f"receipts/chr{chromosome_text}.json",
                "receipt_sha256": sha256_file(receipt_path),
                "postflight_file": f"postflights/chr{chromosome_text}.json",
                "postflight_sha256": sha256_file(postflight_path),
            }
        )

    clinvar_audit_path = tmp_path / "clinvar-audit.json"
    clinvar_audit = _clinvar_audit()
    _write_json(clinvar_audit_path, clinvar_audit)
    clinvar_postflight_path = tmp_path / "clinvar-postflight.json"
    _write_json(
        clinvar_postflight_path,
        _clinvar_postflight(audit=clinvar_audit, audit_path=clinvar_audit_path),
    )
    spec: dict[str, Any] = {
        "$schema": "./snapshot-lineage-spec.schema.json",
        "schema_version": "geno-lewm.v03-snapshot-lineage-spec.v1",
        "candidate_snapshot_id": "geno-lewm-data-v0.3.0-r1",
        "reference_genome": "GRCh38",
        "gnomad": {
            "repo": "abdelstark/geno-lewm-data",
            "repo_type": "dataset",
            "shards": shards,
        },
        "clinvar": {
            "repo": "abdelstark/geno-lewm-data",
            "repo_type": "dataset",
            "revision": "d" * 40,
            "namespace": "staging/clinvar-2026-04-15-archive-eeeeeeeeeeee-r1",
            "audit_file": "clinvar-audit.json",
            "audit_sha256": sha256_file(clinvar_audit_path),
            "postflight_file": "clinvar-postflight.json",
            "postflight_sha256": sha256_file(clinvar_postflight_path),
        },
    }
    spec_path = tmp_path / "lineage-spec.json"
    _write_json(spec_path, spec)
    return spec_path, spec


def _gnomad_receipt(selection: dict[str, object], chromosome: int) -> dict[str, object]:
    source = selection["source"]
    transform = selection["transform"]
    records = 1000 + chromosome
    assert isinstance(source, dict)
    assert isinstance(transform, dict)
    return {
        "schema_version": "geno-lewm.gnomad-staging-receipt.v1",
        "created_at": "2026-07-13T12:00:00Z",
        "ok": True,
        "dataset_id": selection["dataset_id"],
        "release": selection["release"],
        "reference_genome": selection["reference_genome"],
        "source_lock": selection["source_lock"],
        "source": {
            "chromosome": source["chromosome"],
            "split_role": source["split_role"],
            "bucket": source["bucket"],
            "object": source["object"],
            "generation": source["generation"],
            "size_bytes": source["size_bytes"],
            "upstream_md5_base64": source["md5_base64"],
            "upstream_md5_hex": source["md5_hex"],
            "streamed_sha256": f"{chromosome:064x}",
        },
        "transform": {
            "command": transform["command"],
            "argv": ["uv", "run", str(transform["command"])],
            "filters": {
                "filter": transform["filter"],
                "min_af": transform["min_af"],
                "max_allele_len": transform["max_allele_len"],
            },
            "runtime": {
                "elapsed_seconds": 1.0,
                "process_peak_rss_bytes": 1024,
                "peak_memory_note": "fixture process peak",
            },
            "counts": {
                "records_read": records + 9,
                "allele_records_seen": records + 9,
                "records_written": records,
                "skipped_filter": 3,
                "skipped_af": 4,
                "skipped_allele": 2,
            },
        },
        "output": {
            "path": f"/tmp/chr{chromosome}/variants.parquet",
            "sha256": f"{chromosome + 100:064x}",
            "size_bytes": 10_000 + chromosome,
            "parquet_audit": {
                "audit_method": "pyarrow_metadata_and_full_iter_batches_scan_v1",
                "batch_size_rows": 131072,
                "metadata_row_count": records,
                "scanned_row_count": records,
                "canonical_chromosome": str(chromosome),
                "position_min": 1,
                "position_max": 10_000,
                "schema_version": "2.0.0",
                "population_af_non_null_counts": {
                    "af_afr": records,
                    "af_ami": 0,
                    "af_amr": records,
                    "af_asj": records,
                    "af_eas": records,
                    "af_fin": records,
                    "af_mid": records,
                    "af_nfe": records,
                    "af_oth": 0,
                    "af_remaining": records,
                    "af_sas": records,
                },
                "locked_min_af": 0.01,
                "stored_min_af_float32": 0.009999999776482582,
                "schema": [
                    {
                        "name": name,
                        "type": field_type,
                        "nullable": True,
                    }
                    for name, field_type in (
                        ("chrom", "string"),
                        ("pos", "int64"),
                        ("ref", "string"),
                        ("alt", "string"),
                        ("af_global", "float"),
                        ("af_afr", "float"),
                        ("af_ami", "float"),
                        ("af_amr", "float"),
                        ("af_asj", "float"),
                        ("af_eas", "float"),
                        ("af_fin", "float"),
                        ("af_mid", "float"),
                        ("af_nfe", "float"),
                        ("af_oth", "float"),
                        ("af_remaining", "float"),
                        ("af_sas", "float"),
                        ("filter", "string"),
                        ("schema_version", "string"),
                    )
                ],
                "checks": [
                    "exact_arrow_schema",
                    "canonical_chromosome",
                    "positive_position",
                    "explicit_acgt_alleles",
                    "distinct_ref_alt",
                    "finite_global_af_in_locked_range",
                    "finite_population_af_in_unit_interval_or_null",
                    "pass_filter",
                    "exact_schema_version",
                    "v41_population_columns_nonempty",
                    "metadata_scan_and_preparer_row_counts_equal",
                ],
            },
        },
        "execution": selection["execution"],
        "publication": selection["publication"],
        "evidence": {
            "selection": {"sha256": "1" * 64},
            "metadata_verification": {"sha256": "2" * 64},
            "source_identity": {"sha256": "3" * 64},
            "prepare_report": {"sha256": "4" * 64},
        },
        "claim_boundary": selection["claim_boundary"],
    }


def _gnomad_postflight(
    *,
    receipt: dict[str, object],
    receipt_path: Path,
    chromosome: str,
    revision: str,
    namespace: str,
) -> dict[str, object]:
    execution = receipt["execution"]
    output = receipt["output"]
    assert isinstance(execution, dict)
    assert isinstance(output, dict)
    parquet_audit = output["parquet_audit"]
    assert isinstance(parquet_audit, dict)
    file_identities = {
        relative_path: {
            "sha256": hashlib.sha256(relative_path.encode("utf-8")).hexdigest(),
            "size_bytes": 100 + index,
        }
        for index, relative_path in enumerate(REMOTE_NAMESPACE_FILES)
    }
    file_identities["evidence/receipt.json"] = {
        "sha256": hashlib.sha256(receipt_path.read_bytes()).hexdigest(),
        "size_bytes": receipt_path.stat().st_size,
    }
    file_identities["data/gnomad/v4.1/variants.parquet"] = {
        "sha256": output["sha256"],
        "size_bytes": output["size_bytes"],
    }
    return {
        "schema_version": REMOTE_POSTFLIGHT_SCHEMA_VERSION,
        "ok": True,
        "repo_id": "abdelstark/geno-lewm-data",
        "repo_type": "dataset",
        "revision": revision,
        "namespace": namespace,
        "source_commit": execution["commit_sha"],
        "chromosome": chromosome,
        "verified_files": list(REMOTE_NAMESPACE_FILES),
        "file_identities": file_identities,
        "parquet_audit": parquet_audit,
        "checks": list(REMOTE_POSTFLIGHT_CHECKS),
    }


def _write_bound_postflight(
    spec_path: Path,
    spec: dict[str, Any],
    postflight: dict[str, Any],
    *,
    shard_index: int = 0,
) -> None:
    shard = spec["gnomad"]["shards"][shard_index]
    postflight_path = spec_path.parent / shard["postflight_file"]
    _write_json(postflight_path, postflight)
    shard["postflight_sha256"] = sha256_file(postflight_path)
    _write_json(spec_path, spec)


def _read_clinvar_postflight(spec_path: Path) -> dict[str, Any]:
    payload = json.loads((spec_path.parent / "clinvar-postflight.json").read_text(encoding="utf-8"))
    assert isinstance(payload, dict)
    return payload


def _write_bound_clinvar_postflight(
    spec_path: Path,
    spec: dict[str, Any],
    postflight: dict[str, Any],
) -> None:
    postflight_path = spec_path.parent / spec["clinvar"]["postflight_file"]
    _write_json(postflight_path, postflight)
    spec["clinvar"]["postflight_sha256"] = sha256_file(postflight_path)
    _write_json(spec_path, spec)


def _clinvar_audit() -> dict[str, object]:
    source_sha256 = "sha256:" + "a" * 64
    output_sha256 = "sha256:" + "b" * 64
    return {
        "claim_boundary": CLINVAR_REQUIRED_CLAIM_BOUNDARY,
        "commit_sha": "e" * 40,
        "container_image": CONTAINER_IMAGE,
        "generated_at": "2026-07-13T08:36:11Z",
        "generated_by": "hf-job:clinvar-corrected-shard-audit",
        "ok": True,
        "output": {
            "class_balance": {
                "B": 2,
                "LB": 3,
                "LP": 5,
                "OTHER": 7,
                "P": 11,
                "VUS": 13,
            },
            "path": "clinvar/2026-04-15/variants.parquet",
            "records": 41,
            "sha256": output_sha256,
            "size_bytes": 4096,
        },
        "prepare_report": {
            "allele_records_seen": 43,
            "already_exists": False,
            "command": "geno-lewm-prepare-clinvar --release 2026-04-15",
            "input_sha256": source_sha256,
            "input_size_bytes": 8192,
            "input_vcf": {
                "path": "/tmp/clinvar.vcf.gz",
                "sha256": source_sha256,
                "size_bytes": 8192,
            },
            "output_parquet": {
                "path": "/tmp/clinvar/2026-04-15/variants.parquet",
                "sha256": output_sha256,
                "size_bytes": 4096,
            },
            "records_read": 43,
            "records_written": 41,
            "release": "2026-04-15",
            "runtime": {
                "elapsed_seconds": 1.0,
                "peak_memory_note": "fixture process peak",
                "process_peak_rss_bytes": 1024,
            },
            "skipped_allele": 2,
        },
        "runtime": {
            "command": ["geno-lewm-prepare-clinvar"],
            "cpu_count": 8,
            "flavor": "cpu-upgrade",
            "peak_rss_bytes": 2048,
            "peak_rss_source": "fixture",
            "platform": "Linux-fixture",
            "python": "3.13.11",
            "ram_gb": 32,
            "returncode": 0,
            "wall_time_seconds": 2.0,
        },
        "schema_version": "1.0.0",
        "source": {
            "md5": "f" * 32,
            "release": "2026-04-15",
            "sha256": source_sha256,
            "size_bytes": 8192,
            "url": (
                "https://ftp.ncbi.nlm.nih.gov/pub/clinvar/vcf_GRCh38/archive_2.0/"
                "2026/clinvar_20260415.vcf.gz"
            ),
        },
    }


def _clinvar_postflight(*, audit: dict[str, object], audit_path: Path) -> dict[str, object]:
    source = audit["source"]
    output = audit["output"]
    assert isinstance(source, dict)
    assert isinstance(output, dict)
    parquet_schema = _clinvar_parquet_schema()
    file_identities = {
        relative_path: {
            "sha256": hashlib.sha256(relative_path.encode("utf-8")).hexdigest(),
            "size_bytes": 200 + index,
        }
        for index, relative_path in enumerate(CLINVAR_REMOTE_FILES)
    }
    file_identities["evidence/audit.json"] = {
        "sha256": hashlib.sha256(audit_path.read_bytes()).hexdigest(),
        "size_bytes": audit_path.stat().st_size,
    }
    file_identities["clinvar/2026-04-15/variants.parquet"] = {
        "sha256": str(output["sha256"]).removeprefix("sha256:"),
        "size_bytes": output["size_bytes"],
    }
    trusted_files = {
        path: {
            "sha256": hashlib.sha256(path.encode("utf-8")).hexdigest(),
            "size_bytes": 100 + index,
        }
        for index, path in enumerate(
            (
                "geno_lewm/cli/_prepare_report.py",
                "geno_lewm/cli/prepare_clinvar.py",
                "geno_lewm/data/_vcf.py",
                "geno_lewm/data/clinvar.py",
            )
        )
    }
    return {
        "schema_version": CLINVAR_REMOTE_POSTFLIGHT_SCHEMA_VERSION,
        "ok": True,
        "repo_id": "abdelstark/geno-lewm-data",
        "repo_type": "dataset",
        "revision": "d" * 40,
        "namespace": "staging/clinvar-2026-04-15-archive-eeeeeeeeeeee-r1",
        "source_commit": audit["commit_sha"],
        "release": source["release"],
        "verified_files": list(CLINVAR_REMOTE_FILES),
        "file_identities": file_identities,
        "trusted_source_contract": {
            "files": trusted_files,
            "schema_version": "1.0.0",
            "parquet_schema": parquet_schema,
            "nullable_fields": ["gene_symbol"],
            "normalized_classes": ["B", "LB", "LP", "OTHER", "P", "VUS"],
            "labelled_classes": ["B", "LB", "LP", "P"],
            "allele_alphabet": ["A", "C", "G", "T"],
            "cli_command": "geno-lewm-prepare-clinvar",
            "max_allele_len": 16,
            "output_path_template": "clinvar/{release}/variants.parquet",
            "prepare_report_enrichments": [
                "command",
                "input_vcf",
                "output_parquet",
                "runtime",
            ],
            "file_identity_fields": ["path", "sha256", "size_bytes"],
        },
        "source_identity": {
            "url": source["url"],
            "release": source["release"],
            "md5": source["md5"],
            "sha256": str(source["sha256"]).removeprefix("sha256:"),
            "size_bytes": source["size_bytes"],
            "verification_scope": [
                "release_reconciled",
                "sha256_reconciled",
                "size_bytes_reconciled",
            ],
            "verification_limitation": (
                "The source archive is not included in the Hub namespace; its MD5 and URL "
                "are receipt fields, not bytes recomputed by this postflight."
            ),
        },
        "output_identity": {
            "path": output["path"],
            "sha256": str(output["sha256"]).removeprefix("sha256:"),
            "size_bytes": output["size_bytes"],
            "records": output["records"],
            "class_balance": output["class_balance"],
        },
        "parquet_audit": _clinvar_parquet_audit(),
        "claim_boundary": audit["claim_boundary"],
        "checks": list(CLINVAR_REMOTE_CHECKS),
    }


def _clinvar_parquet_schema() -> list[dict[str, str]]:
    return [dict(field) for field in CLINVAR_PARQUET_SCHEMA]


def _clinvar_parquet_audit() -> dict[str, object]:
    return {
        "metadata_row_count": 41,
        "scanned_row_count": 41,
        "class_balance": {
            "B": 2,
            "LB": 3,
            "LP": 5,
            "OTHER": 7,
            "P": 11,
            "VUS": 13,
        },
        "chromosome_balance": {"1": 17, "2": 24},
        "schema_version_balance": {"1.0.0": 41},
        "null_counts": {
            "alt": 0,
            "chrom": 0,
            "clinical_significance": 0,
            "clinvar_id": 0,
            "gene_symbol": 1,
            "pos": 0,
            "ref": 0,
            "review_status": 0,
            "schema_version": 0,
        },
        "position_range": {"min": 1, "max": 249_250_621},
        "clinvar_id_range": {"min": 1, "max": 9_999_999},
        "schema": _clinvar_parquet_schema(),
    }


def _replace_after_first_binary_read(
    monkeypatch: pytest.MonkeyPatch,
    *,
    target: Path,
    replacement: object,
) -> dict[str, bool]:
    replacement_path = target.with_name(target.name + ".replacement")
    _write_json(replacement_path, replacement)
    original_open = Path.open
    state = {"wrapped": False, "replaced": False}

    class _ReplaceOnExit:
        def __init__(self, stream: Any) -> None:
            self._stream = stream

        def __enter__(self) -> Any:
            return self._stream.__enter__()

        def __exit__(self, *args: object) -> object:
            result = self._stream.__exit__(*args)
            replacement_path.replace(target)
            state["replaced"] = True
            return result

    def _open(path: Path, *args: object, **kwargs: object) -> Any:
        stream = original_open(path, *args, **kwargs)
        mode = str(args[0]) if args else str(kwargs.get("mode", "r"))
        if path.resolve() == target.resolve() and mode == "rb" and not state["wrapped"]:
            state["wrapped"] = True
            return _ReplaceOnExit(stream)
        return stream

    monkeypatch.setattr(Path, "open", _open)
    return state


def _walk_json(value: object, field: str = "$") -> list[tuple[str, object]]:
    rows = [(field, value)]
    if isinstance(value, dict):
        for key, child in value.items():
            rows.extend(_walk_json(child, f"{field}.{key}"))
    elif isinstance(value, list):
        for index, child in enumerate(value):
            rows.extend(_walk_json(child, f"{field}[{index}]"))
    return rows


def _write_json(path: Path, payload: object) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _recommit_lineage(lineage: dict[str, Any]) -> None:
    commitment = dict(lineage)
    del commitment["lineage_id"]
    lineage["lineage_id"] = canonical_json_sha256(commitment)
