# SPDX-License-Identifier: Apache-2.0
"""Assemble immutable v0.3 staging evidence into a lineage candidate.

This tool records source-to-staged-artifact lineage only. It deliberately does
not create dataset memberships or claim that a publishable snapshot exists.
"""

from __future__ import annotations

import argparse
import base64
import binascii
import hashlib
import json
import math
import re
import struct
import sys
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path, PureWindowsPath
from types import MappingProxyType
from typing import Any, Final, cast

from geno_lewm.provenance import canonical_json_sha256
from tools.data._immutable_json import (
    ImmutableJsonError,
    _fsync_directory as _immutable_fsync_directory,
    write_immutable_json,
)
from tools.data.v03_clinvar_postflight import (
    REMOTE_POSTFLIGHT_SCHEMA_VERSION as CLINVAR_REMOTE_POSTFLIGHT_SCHEMA_VERSION,
)
from tools.data.v03_gnomad_lock import (
    LOCK_SCHEMA_VERSION,
    REMOTE_POSTFLIGHT_SCHEMA_VERSION,
    STAGING_RECEIPT_SCHEMA_VERSION,
    SourceLockError,
    SourceLockSnapshot,
    capture_source_lock,
    select_source_from_snapshot,
)

SPEC_SCHEMA_VERSION: Final = "geno-lewm.v03-snapshot-lineage-spec.v1"
LINEAGE_SCHEMA_VERSION: Final = "geno-lewm.v03-snapshot-lineage.v1"
GENERATED_BY: Final = "tools.data.v03_snapshot_lineage"
MEMBERSHIP_STATUS: Final = "not_created"
CLINVAR_REQUIRED_CLAIM_BOUNDARY: Final = (
    "This receipt covers normalization of the pinned ClinVar GRCh38 2026-04-15 archive. "
    "It does not define a leakage-safe eval split or establish label correctness, "
    "representativeness, clinical utility, or model quality."
)
LINEAGE_CLAIM_BOUNDARY: Final = (
    "This artifact records immutable source-to-stage lineage only. It does not create "
    "snapshot memberships, materialize train/validation/evaluation datasets, prove split "
    "leakage controls, establish representativeness, support model-quality or benchmark "
    "claims, or confer clinical validity."
)

_AUTOSOMES: Final = frozenset(str(chromosome) for chromosome in range(1, 23))
_SHA256: Final = re.compile(r"sha256:[0-9a-f]{64}")
_BARE_SHA256: Final = re.compile(r"[0-9a-f]{64}")
_COMMIT: Final = re.compile(r"[0-9a-f]{40}")
_CONTAINER: Final = re.compile(r"[^@]+@sha256:[0-9a-f]{64}")
_CANDIDATE_ID: Final = re.compile(r"geno-lewm-data-v0\.3\.[0-9]+-r[1-9][0-9]*")
_SAFE_BUNDLE_JSON_PATH: Final = re.compile(
    r"(?!/)(?![A-Za-z]:)(?!.*\\)(?!.*(?:^|/)\.\.(?:/|$)).+\.json"
)
_CLINVAR_CLASSES: Final = frozenset({"B", "LB", "LP", "OTHER", "P", "VUS"})
_CLINVAR_REMOTE_FILES: Final = (
    "clinvar/2026-04-15/variants.parquet",
    "evidence/audit.json",
    "evidence/prepare_report.json",
    "evidence/runtime_report.json",
)
_CLINVAR_REMOTE_CHECKS: Final = (
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
_CLINVAR_SOURCE_CONTRACT_FILES: Final = (
    "geno_lewm/cli/_prepare_report.py",
    "geno_lewm/cli/prepare_clinvar.py",
    "geno_lewm/data/_vcf.py",
    "geno_lewm/data/clinvar.py",
)
_CLINVAR_PARQUET_SCHEMA: Final = (
    ("chrom", "string"),
    ("pos", "int64"),
    ("ref", "string"),
    ("alt", "string"),
    ("clinical_significance", "string"),
    ("review_status", "string"),
    ("gene_symbol", "string"),
    ("clinvar_id", "int64"),
    ("schema_version", "string"),
)
_CLINVAR_SOURCE_SCOPE: Final = (
    "release_reconciled",
    "sha256_reconciled",
    "size_bytes_reconciled",
)
_CLINVAR_SOURCE_LIMITATION: Final = (
    "The source archive is not included in the Hub namespace; its MD5 and URL are "
    "receipt fields, not bytes recomputed by this postflight."
)
_CLINVAR_ARCHIVE_URL: Final = (
    "https://ftp.ncbi.nlm.nih.gov/pub/clinvar/vcf_GRCh38/archive_2.0/2026/clinvar_20260415.vcf.gz"
)
_GNOMAD_POPULATION_COLUMNS: Final = frozenset(
    {
        "af_afr",
        "af_ami",
        "af_amr",
        "af_asj",
        "af_eas",
        "af_fin",
        "af_mid",
        "af_nfe",
        "af_oth",
        "af_remaining",
        "af_sas",
    }
)
_GNOMAD_V41_REQUIRED_POPULATIONS: Final = frozenset(
    {
        "af_afr",
        "af_amr",
        "af_asj",
        "af_eas",
        "af_fin",
        "af_mid",
        "af_nfe",
        "af_remaining",
        "af_sas",
    }
)
_REMOTE_NAMESPACE_FILES: Final = (
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
_REMOTE_POSTFLIGHT_CHECKS: Final = (
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
_GNOMAD_PARQUET_SCHEMA: Final = (
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
_GNOMAD_PARQUET_CHECKS: Final = (
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
)


class SnapshotLineageError(ValueError):
    """Raised when snapshot-lineage evidence violates the closed contract."""


@dataclass(frozen=True, slots=True)
class VerifiedSnapshotLineage:
    """Exact bytes and parsed value from one verified lineage-file capture."""

    payload: bytes
    lineage: Mapping[str, object]
    payload_sha256: str
    size_bytes: int


@dataclass(frozen=True, slots=True)
class _CapturedJson:
    """One JSON file captured once for hashing, sizing, parsing, and semantics."""

    payload: bytes
    value: Mapping[str, object]
    sha256: str
    size_bytes: int


def assemble_snapshot_lineage(
    *,
    spec_path: Path,
    gnomad_source_lock_path: Path,
    output_path: Path | None = None,
) -> dict[str, Any]:
    """Validate exact staging evidence and return a deterministic lineage candidate."""
    spec_capture = _capture_json(spec_path, "lineage spec")
    spec = spec_capture.value
    _require_exact_keys(
        spec,
        {
            "$schema",
            "schema_version",
            "candidate_snapshot_id",
            "reference_genome",
            "gnomad",
            "clinvar",
        },
        "lineage spec",
    )
    _require_equal(
        spec.get("$schema"), "./snapshot-lineage-spec.schema.json", "lineage spec.$schema"
    )
    _require_equal(spec.get("schema_version"), SPEC_SCHEMA_VERSION, "lineage spec.schema_version")
    candidate_snapshot_id = _require_str(
        spec.get("candidate_snapshot_id"), "lineage spec.candidate_snapshot_id"
    )
    if _CANDIDATE_ID.fullmatch(candidate_snapshot_id) is None:
        raise SnapshotLineageError(
            "lineage spec.candidate_snapshot_id must be a versioned v0.3 candidate id"
        )
    _require_equal(spec.get("reference_genome"), "GRCh38", "lineage spec.reference_genome")

    source_lock_snapshot = capture_source_lock(gnomad_source_lock_path)
    gnomad = _assemble_gnomad(
        spec=_require_mapping(spec.get("gnomad"), "lineage spec.gnomad"),
        spec_dir=spec_path.parent,
        source_lock_snapshot=source_lock_snapshot,
    )
    clinvar = _assemble_clinvar(
        spec=_require_mapping(spec.get("clinvar"), "lineage spec.clinvar"),
        spec_dir=spec_path.parent,
    )
    _require_equal(clinvar["repo"], gnomad["repo"], "ClinVar and gnomAD publication repository")

    lineage: dict[str, Any] = {
        "schema_version": LINEAGE_SCHEMA_VERSION,
        "generated_by": GENERATED_BY,
        "candidate_snapshot_id": candidate_snapshot_id,
        "reference_genome": "GRCh38",
        "membership_status": MEMBERSHIP_STATUS,
        "assembly_inputs": {
            "spec": {
                "sha256": spec_capture.sha256,
                "size_bytes": spec_capture.size_bytes,
            },
            "gnomad_source_lock": {
                "sha256": "sha256:" + hashlib.sha256(source_lock_snapshot.lock_bytes).hexdigest(),
                "size_bytes": len(source_lock_snapshot.lock_bytes),
            },
        },
        "gnomad": gnomad,
        "clinvar": clinvar,
        "claim_boundary": LINEAGE_CLAIM_BOUNDARY,
    }
    lineage["lineage_id"] = canonical_json_sha256(lineage)
    if output_path is not None:
        _write_immutable_json(output_path, lineage)
    return lineage


def capture_verified_snapshot_lineage(path: Path) -> VerifiedSnapshotLineage:
    """Capture once and return the exact bytes of a verified lineage candidate."""
    lineage_capture = _capture_json(path, "snapshot lineage")
    _verify_snapshot_lineage_value(lineage_capture.value)
    return VerifiedSnapshotLineage(
        payload=lineage_capture.payload,
        lineage=cast(Mapping[str, object], _deep_freeze_json(lineage_capture.value)),
        payload_sha256=lineage_capture.sha256,
        size_bytes=lineage_capture.size_bytes,
    )


def verify_snapshot_lineage(path: Path) -> dict[str, Any]:
    """Fail-closed verify one existing lineage candidate.

    JSON Schema validation is structural only. This verifier additionally
    recomputes the content-addressed ``lineage_id`` and enforces the semantic
    no-membership and autosome split contracts.
    """
    captured = capture_verified_snapshot_lineage(path)
    return cast(dict[str, Any], _deep_thaw_json(captured.lineage))


def _verify_snapshot_lineage_value(lineage: Mapping[str, object]) -> None:
    """Verify the closed semantic contract of an already captured JSON value."""
    _require_exact_keys(
        lineage,
        {
            "schema_version",
            "generated_by",
            "candidate_snapshot_id",
            "reference_genome",
            "membership_status",
            "assembly_inputs",
            "gnomad",
            "clinvar",
            "claim_boundary",
            "lineage_id",
        },
        "snapshot lineage",
    )
    observed_lineage_id = _require_sha256(lineage.get("lineage_id"), "lineage.lineage_id")
    commitment = dict(lineage)
    del commitment["lineage_id"]
    _require_equal(
        observed_lineage_id,
        canonical_json_sha256(commitment),
        "lineage_id",
    )

    _require_equal(lineage.get("schema_version"), LINEAGE_SCHEMA_VERSION, "lineage.schema_version")
    _require_equal(lineage.get("generated_by"), GENERATED_BY, "lineage.generated_by")
    candidate_snapshot_id = _require_str(
        lineage.get("candidate_snapshot_id"), "lineage.candidate_snapshot_id"
    )
    if _CANDIDATE_ID.fullmatch(candidate_snapshot_id) is None:
        raise SnapshotLineageError(
            "lineage.candidate_snapshot_id must be a versioned v0.3 candidate id"
        )
    _require_equal(lineage.get("reference_genome"), "GRCh38", "lineage.reference_genome")
    _require_equal(lineage.get("membership_status"), MEMBERSHIP_STATUS, "lineage.membership_status")
    _require_equal(lineage.get("claim_boundary"), LINEAGE_CLAIM_BOUNDARY, "lineage.claim_boundary")

    assembly_source_lock_sha = _verify_lineage_assembly_inputs(
        _require_mapping(lineage.get("assembly_inputs"), "lineage.assembly_inputs")
    )
    gnomad_repo = _verify_gnomad_lineage(
        _require_mapping(lineage.get("gnomad"), "lineage.gnomad"),
        assembly_source_lock_sha=assembly_source_lock_sha,
    )
    _verify_clinvar_lineage(
        _require_mapping(lineage.get("clinvar"), "lineage.clinvar"),
        expected_repo=gnomad_repo,
    )


def _verify_lineage_assembly_inputs(assembly_inputs: Mapping[str, object]) -> str:
    _require_exact_keys(
        assembly_inputs,
        {"spec", "gnomad_source_lock"},
        "lineage.assembly_inputs",
    )
    _verify_file_identity(assembly_inputs.get("spec"), "lineage assembly spec")
    source_lock = _verify_file_identity(
        assembly_inputs.get("gnomad_source_lock"),
        "lineage assembly gnomAD source lock",
    )
    return _require_sha256(source_lock.get("sha256"), "assembly source-lock sha256")


def _verify_file_identity(value: object, field: str) -> Mapping[str, object]:
    identity = _require_mapping(value, field)
    _require_exact_keys(identity, {"sha256", "size_bytes"}, field)
    _require_sha256(identity.get("sha256"), f"{field}.sha256")
    _require_positive_int(identity.get("size_bytes"), f"{field}.size_bytes")
    return identity


def _verify_remote_file_identities(
    value: object,
    field: str,
    *,
    expected_paths: tuple[str, ...],
) -> dict[str, Mapping[str, object]]:
    identities = _require_mapping(value, field)
    _require_exact_keys(identities, set(expected_paths), field)
    return {
        path: _verify_file_identity(identities.get(path), f"{field}[{path!r}]")
        for path in expected_paths
    }


def _verify_artifact_identity(
    value: object,
    field: str,
    *,
    expected_path: str,
) -> Mapping[str, object]:
    identity = _require_mapping(value, field)
    _require_exact_keys(identity, {"artifact_path", "sha256", "size_bytes"}, field)
    _require_equal(identity.get("artifact_path"), expected_path, f"{field}.artifact_path")
    _require_sha256(identity.get("sha256"), f"{field}.sha256")
    _require_positive_int(identity.get("size_bytes"), f"{field}.size_bytes")
    return identity


def _verify_gnomad_lineage(
    gnomad: Mapping[str, object],
    *,
    assembly_source_lock_sha: str,
) -> str:
    _require_exact_keys(
        gnomad,
        {
            "dataset_id",
            "release",
            "repo",
            "repo_type",
            "data_use",
            "source_lock",
            "transform",
            "common_execution",
            "split_policy",
            "total_records",
            "total_size_bytes",
            "shards",
        },
        "lineage.gnomad",
    )
    _require_equal(gnomad.get("dataset_id"), "gnomad-v4.1-exomes-autosomes", "gnomAD dataset")
    _require_equal(gnomad.get("release"), "v4.1", "gnomAD release")
    _require_equal(gnomad.get("repo_type"), "dataset", "gnomAD repo_type")
    _require_equal(gnomad.get("data_use"), _gnomad_data_use(), "gnomAD data_use")
    repo = _require_repo(gnomad.get("repo"), "gnomAD repo")

    source_lock = _require_mapping(gnomad.get("source_lock"), "gnomAD source_lock")
    _require_exact_keys(
        source_lock,
        {"schema_version", "sha256", "schema_sha256"},
        "gnomAD source_lock",
    )
    _require_equal(
        source_lock.get("schema_version"), LOCK_SCHEMA_VERSION, "gnomAD source-lock schema version"
    )
    source_lock_sha = _require_sha256(source_lock.get("sha256"), "gnomAD source-lock sha256")
    _require_sha256(source_lock.get("schema_sha256"), "gnomAD source-lock schema sha256")
    _require_equal(source_lock_sha, assembly_source_lock_sha, "gnomAD assembly source-lock sha256")

    transform = _require_mapping(gnomad.get("transform"), "gnomAD transform")
    _require_exact_keys(
        transform, {"command", "filter", "min_af", "max_allele_len"}, "gnomAD transform"
    )
    _require_equal(
        transform,
        {
            "command": "geno-lewm-prepare-gnomad",
            "filter": "PASS",
            "min_af": 0.01,
            "max_allele_len": 16,
        },
        "gnomAD transform",
    )

    execution = _require_mapping(gnomad.get("common_execution"), "gnomAD common_execution")
    _require_exact_keys(
        execution,
        {"commit_sha", "container_image", "repository"},
        "gnomAD common_execution",
    )
    commit_sha = _require_commit(execution.get("commit_sha"), "gnomAD execution.commit_sha")
    _require_container(execution.get("container_image"), "gnomAD execution.container_image")
    _require_equal(
        execution.get("repository"),
        "https://github.com/AbdelStark/GenoLeWM.git",
        "gnomAD execution.repository",
    )

    split_policy = _require_mapping(gnomad.get("split_policy"), "lineage.gnomad.split_policy")
    _require_exact_keys(split_policy, {"train", "validation", "evaluation"}, "split_policy")
    _require_equal(
        split_policy,
        {
            "train": [*(str(chromosome) for chromosome in range(1, 20)), "22"],
            "validation": ["20"],
            "evaluation": ["21"],
        },
        "lineage.gnomad.split_policy",
    )

    shards = _require_list(gnomad.get("shards"), "lineage.gnomad.shards")
    if len(shards) != 22:
        raise SnapshotLineageError("lineage.gnomad.shards must contain exactly 22 entries")
    seen_chromosomes: set[str] = set()
    seen_namespaces: set[str] = set()
    total_records = 0
    total_size_bytes = 0
    for index, raw_shard in enumerate(shards):
        records, size_bytes, chromosome, namespace = _verify_gnomad_lineage_shard(
            raw_shard,
            index=index,
            transform=transform,
            commit_sha=commit_sha,
            source_lock_sha=source_lock_sha,
        )
        if chromosome in seen_chromosomes:
            raise SnapshotLineageError(f"duplicate lineage gnomAD chromosome: {chromosome}")
        if namespace in seen_namespaces:
            raise SnapshotLineageError(f"duplicate lineage gnomAD namespace: {namespace}")
        seen_chromosomes.add(chromosome)
        seen_namespaces.add(namespace)
        total_records += records
        total_size_bytes += size_bytes
    if seen_chromosomes != _AUTOSOMES:
        missing = sorted(_AUTOSOMES - seen_chromosomes, key=int)
        raise SnapshotLineageError(
            f"lineage gnomAD autosome coverage incomplete: missing={missing}"
        )
    _require_equal(gnomad.get("total_records"), total_records, "gnomAD total_records")
    _require_equal(gnomad.get("total_size_bytes"), total_size_bytes, "gnomAD total_size_bytes")
    return repo


def _verify_gnomad_lineage_shard(
    value: object,
    *,
    index: int,
    transform: Mapping[str, object],
    commit_sha: str,
    source_lock_sha: str,
) -> tuple[int, int, str, str]:
    field = f"lineage.gnomad.shards[{index}]"
    shard = _require_mapping(value, field)
    _require_exact_keys(
        shard,
        {
            "chromosome",
            "split_role",
            "revision",
            "namespace",
            "receipt",
            "remote_postflight",
            "source",
            "transform",
            "output",
        },
        field,
    )
    chromosome = _require_str(shard.get("chromosome"), f"{field}.chromosome")
    if chromosome not in _AUTOSOMES:
        raise SnapshotLineageError(
            f"lineage gnomAD chromosome must be one of 1..22: {chromosome!r}"
        )
    _require_equal(
        chromosome,
        str(index + 1),
        f"gnomAD canonical chromosome order at index {index}",
    )
    _require_equal(shard.get("split_role"), _split_role(chromosome), f"chr{chromosome} split_role")
    _require_commit(shard.get("revision"), f"chr{chromosome} revision")
    namespace = _require_namespace(shard.get("namespace"), f"chr{chromosome} namespace")
    _require_equal(shard.get("transform"), transform, f"chr{chromosome} transform")

    source = _require_mapping(shard.get("source"), f"chr{chromosome} source")
    _require_exact_keys(
        source,
        {
            "bucket",
            "object",
            "generation",
            "size_bytes",
            "upstream_md5_base64",
            "upstream_md5_hex",
            "streamed_sha256",
        },
        f"chr{chromosome} source",
    )
    _require_equal(source.get("bucket"), "gcp-public-data--gnomad", f"chr{chromosome} bucket")
    _require_equal(
        source.get("object"),
        f"release/4.1/vcf/exomes/gnomad.exomes.v4.1.sites.chr{chromosome}.vcf.bgz",
        f"chr{chromosome} source object",
    )
    generation = _require_str(source.get("generation"), f"chr{chromosome} generation")
    if not generation.isdigit() or int(generation) <= 0:
        raise SnapshotLineageError(f"chr{chromosome} generation must be a positive decimal")
    _require_positive_int(source.get("size_bytes"), f"chr{chromosome} source size_bytes")
    _require_sha256(source.get("streamed_sha256"), f"chr{chromosome} streamed_sha256")
    md5_hex = _require_str(source.get("upstream_md5_hex"), f"chr{chromosome} upstream MD5")
    if re.fullmatch(r"[0-9a-f]{32}", md5_hex) is None:
        raise SnapshotLineageError(f"chr{chromosome} upstream MD5 must be lowercase hexadecimal")
    md5_base64 = _require_str(
        source.get("upstream_md5_base64"), f"chr{chromosome} upstream MD5 base64"
    )
    try:
        decoded_md5 = base64.b64decode(md5_base64, validate=True)
    except (ValueError, binascii.Error) as exc:
        raise SnapshotLineageError(f"chr{chromosome} upstream MD5 base64 is invalid") from exc
    _require_equal(decoded_md5.hex(), md5_hex, f"chr{chromosome} upstream MD5 encodings")

    expected_namespace = (
        "staging/v0.3/gnomad-v4.1-exomes-autosomes/"
        f"lock-{source_lock_sha.removeprefix('sha256:')[:12]}/"
        f"chr{chromosome}-g{generation}-{commit_sha[:12]}"
    )
    _require_equal(namespace, expected_namespace, f"chr{chromosome} namespace identity")
    receipt = _verify_artifact_identity(
        shard.get("receipt"),
        f"chr{chromosome} receipt",
        expected_path=f"{namespace}/evidence/receipt.json",
    )

    output = _require_mapping(shard.get("output"), f"chr{chromosome} output")
    _require_exact_keys(
        output,
        {"artifact_path", "sha256", "size_bytes", "records", "schema_version"},
        f"chr{chromosome} output",
    )
    _require_equal(
        output.get("artifact_path"),
        f"{namespace}/data/gnomad/v4.1/variants.parquet",
        f"chr{chromosome} output.artifact_path",
    )
    _require_sha256(output.get("sha256"), f"chr{chromosome} output.sha256")
    size_bytes = _require_positive_int(
        output.get("size_bytes"), f"chr{chromosome} output.size_bytes"
    )
    records = _require_positive_int(output.get("records"), f"chr{chromosome} output.records")
    _require_equal(output.get("schema_version"), "2.0.0", f"chr{chromosome} schema_version")

    postflight = _require_mapping(
        shard.get("remote_postflight"), f"chr{chromosome} remote_postflight"
    )
    _require_exact_keys(
        postflight,
        {
            "schema_version",
            "sha256",
            "size_bytes",
            "verified_files",
            "file_identities",
            "checks",
            "parquet_audit",
        },
        f"chr{chromosome} remote_postflight",
    )
    _require_equal(
        postflight.get("schema_version"),
        REMOTE_POSTFLIGHT_SCHEMA_VERSION,
        f"chr{chromosome} postflight schema_version",
    )
    _require_sha256(postflight.get("sha256"), f"chr{chromosome} postflight sha256")
    _require_positive_int(postflight.get("size_bytes"), f"chr{chromosome} postflight size_bytes")
    _require_equal(
        _require_exact_string_list(
            postflight.get("verified_files"), f"chr{chromosome} postflight verified_files"
        ),
        list(_REMOTE_NAMESPACE_FILES),
        f"chr{chromosome} postflight verified files",
    )
    file_identities = _verify_remote_file_identities(
        postflight.get("file_identities"),
        f"chr{chromosome} postflight file_identities",
        expected_paths=_REMOTE_NAMESPACE_FILES,
    )
    _require_equal(
        file_identities["evidence/receipt.json"],
        {"sha256": receipt.get("sha256"), "size_bytes": receipt.get("size_bytes")},
        f"chr{chromosome} postflight receipt identity",
    )
    _require_equal(
        file_identities["data/gnomad/v4.1/variants.parquet"],
        {"sha256": output.get("sha256"), "size_bytes": output.get("size_bytes")},
        f"chr{chromosome} postflight Parquet identity",
    )
    _require_equal(
        _require_exact_string_list(postflight.get("checks"), f"chr{chromosome} postflight checks"),
        list(_REMOTE_POSTFLIGHT_CHECKS),
        f"chr{chromosome} postflight checks",
    )
    _verify_gnomad_parquet_audit(
        _require_mapping(
            postflight.get("parquet_audit"), f"chr{chromosome} postflight parquet_audit"
        ),
        chromosome=chromosome,
        records=records,
        min_af=transform.get("min_af"),
    )
    return records, size_bytes, chromosome, namespace


def _verify_gnomad_parquet_audit(
    audit: Mapping[str, object],
    *,
    chromosome: str,
    records: int,
    min_af: object,
) -> None:
    field = f"chr{chromosome} postflight parquet_audit"
    _require_exact_keys(
        audit,
        {
            "audit_method",
            "batch_size_rows",
            "metadata_row_count",
            "scanned_row_count",
            "canonical_chromosome",
            "position_min",
            "position_max",
            "schema_version",
            "population_af_non_null_counts",
            "locked_min_af",
            "stored_min_af_float32",
            "schema",
            "checks",
        },
        field,
    )
    _require_equal(
        audit.get("audit_method"),
        "pyarrow_metadata_and_full_iter_batches_scan_v1",
        f"{field}.audit_method",
    )
    _require_equal(audit.get("batch_size_rows"), 131_072, f"{field}.batch_size_rows")
    _require_equal(audit.get("metadata_row_count"), records, f"{field}.metadata_row_count")
    _require_equal(audit.get("scanned_row_count"), records, f"{field}.scanned_row_count")
    _require_equal(audit.get("canonical_chromosome"), chromosome, f"{field}.chromosome")
    position_min = _require_positive_int(audit.get("position_min"), f"{field}.position_min")
    position_max = _require_positive_int(audit.get("position_max"), f"{field}.position_max")
    if position_min > position_max:
        raise SnapshotLineageError(f"{field} position_min exceeds position_max")
    _require_equal(audit.get("schema_version"), "2.0.0", f"{field}.schema_version")
    _require_equal(audit.get("locked_min_af"), min_af, f"{field}.locked_min_af")
    if isinstance(min_af, bool) or not isinstance(min_af, int | float):
        raise SnapshotLineageError(f"{field}.locked_min_af must be numeric")
    expected_stored_min = struct.unpack("<f", struct.pack("<f", float(min_af)))[0]
    _require_equal(
        audit.get("stored_min_af_float32"),
        expected_stored_min,
        f"{field}.stored_min_af_float32",
    )

    population_counts = _require_mapping(
        audit.get("population_af_non_null_counts"), f"{field}.population counts"
    )
    _require_exact_keys(
        population_counts, set(_GNOMAD_POPULATION_COLUMNS), f"{field}.population counts"
    )
    for population in sorted(_GNOMAD_POPULATION_COLUMNS):
        count = _require_nonnegative_int(
            population_counts.get(population), f"{field}.{population} non-null count"
        )
        if count > records:
            raise SnapshotLineageError(f"{field}.{population} non-null count exceeds records")
        if population in _GNOMAD_V41_REQUIRED_POPULATIONS and count == 0:
            raise SnapshotLineageError(f"{field}.{population} required population is empty")
    expected_schema = [
        {"name": name, "type": field_type, "nullable": True}
        for name, field_type in _GNOMAD_PARQUET_SCHEMA
    ]
    _require_equal(audit.get("schema"), expected_schema, f"{field}.schema")
    _require_equal(audit.get("checks"), list(_GNOMAD_PARQUET_CHECKS), f"{field}.checks")


def _verify_clinvar_lineage(
    clinvar: Mapping[str, object],
    *,
    expected_repo: str,
) -> None:
    _require_exact_keys(
        clinvar,
        {
            "release",
            "reference_genome",
            "repo",
            "repo_type",
            "data_use",
            "revision",
            "namespace",
            "audit",
            "source",
            "output",
            "remote_postflight",
            "execution",
            "evidence_claim_boundary",
        },
        "lineage.clinvar",
    )
    _require_equal(clinvar.get("release"), "2026-04-15", "ClinVar release")
    _require_equal(clinvar.get("reference_genome"), "GRCh38", "ClinVar reference_genome")
    _require_equal(clinvar.get("repo_type"), "dataset", "ClinVar repo_type")
    _require_equal(clinvar.get("data_use"), _clinvar_data_use(), "ClinVar data_use")
    _require_equal(clinvar.get("repo"), expected_repo, "ClinVar and gnomAD repository")
    _require_repo(clinvar.get("repo"), "ClinVar repo")
    _require_commit(clinvar.get("revision"), "ClinVar revision")

    execution = _require_mapping(clinvar.get("execution"), "ClinVar execution")
    _require_exact_keys(execution, {"commit_sha", "container_image"}, "ClinVar execution")
    commit_sha = _require_commit(execution.get("commit_sha"), "ClinVar execution.commit_sha")
    _require_container(execution.get("container_image"), "ClinVar execution.container_image")
    namespace = _require_namespace(clinvar.get("namespace"), "ClinVar namespace")
    _require_equal(
        namespace,
        f"staging/clinvar-2026-04-15-archive-{commit_sha[:12]}-r1",
        "ClinVar exact-revision namespace",
    )
    _require_equal(
        clinvar.get("evidence_claim_boundary"),
        CLINVAR_REQUIRED_CLAIM_BOUNDARY,
        "ClinVar evidence_claim_boundary",
    )
    audit_identity = _verify_artifact_identity(
        clinvar.get("audit"),
        "ClinVar audit",
        expected_path=f"{namespace}/evidence/audit.json",
    )

    source = _require_mapping(clinvar.get("source"), "ClinVar source")
    _require_exact_keys(source, {"url", "md5", "sha256", "size_bytes"}, "ClinVar source")
    source_url = _require_str(source.get("url"), "ClinVar source.url")
    _require_equal(source_url, _CLINVAR_ARCHIVE_URL, "ClinVar source.url")
    source_md5 = _require_str(source.get("md5"), "ClinVar source.md5")
    if re.fullmatch(r"[0-9a-f]{32}", source_md5) is None:
        raise SnapshotLineageError("ClinVar source.md5 must be a lowercase MD5 digest")
    _require_sha256(source.get("sha256"), "ClinVar source.sha256")
    _require_positive_int(source.get("size_bytes"), "ClinVar source.size_bytes")

    output = _require_mapping(clinvar.get("output"), "ClinVar output")
    _require_exact_keys(
        output,
        {"artifact_path", "sha256", "size_bytes", "records", "class_balance"},
        "ClinVar output",
    )
    _require_equal(
        output.get("artifact_path"),
        f"{namespace}/clinvar/2026-04-15/variants.parquet",
        "ClinVar output.artifact_path",
    )
    _require_sha256(output.get("sha256"), "ClinVar output.sha256")
    _require_positive_int(output.get("size_bytes"), "ClinVar output.size_bytes")
    records = _require_positive_int(output.get("records"), "ClinVar output.records")
    class_balance_raw = _require_mapping(output.get("class_balance"), "ClinVar class_balance")
    _require_exact_keys(class_balance_raw, set(_CLINVAR_CLASSES), "ClinVar class_balance")
    class_balance = {
        label: _require_nonnegative_int(
            class_balance_raw.get(label), f"ClinVar class_balance.{label}"
        )
        for label in sorted(_CLINVAR_CLASSES)
    }
    _require_equal(sum(class_balance.values()), records, "ClinVar class-balance total")
    for label in ("B", "LB", "LP", "P"):
        if class_balance[label] <= 0:
            raise SnapshotLineageError(f"ClinVar labelled class {label} must be non-empty")

    postflight = _require_mapping(clinvar.get("remote_postflight"), "ClinVar remote_postflight")
    _require_exact_keys(
        postflight,
        {
            "schema_version",
            "sha256",
            "size_bytes",
            "verified_files",
            "file_identities",
            "checks",
            "parquet_audit",
        },
        "ClinVar remote_postflight",
    )
    _require_equal(
        postflight.get("schema_version"),
        CLINVAR_REMOTE_POSTFLIGHT_SCHEMA_VERSION,
        "ClinVar postflight.schema_version",
    )
    _require_sha256(postflight.get("sha256"), "ClinVar postflight.sha256")
    _require_positive_int(postflight.get("size_bytes"), "ClinVar postflight.size_bytes")
    _require_equal(
        _require_exact_string_list(
            postflight.get("verified_files"), "ClinVar postflight.verified_files"
        ),
        list(_CLINVAR_REMOTE_FILES),
        "ClinVar postflight verified files",
    )
    file_identities = _verify_remote_file_identities(
        postflight.get("file_identities"),
        "ClinVar postflight file_identities",
        expected_paths=_CLINVAR_REMOTE_FILES,
    )
    _require_equal(
        file_identities["evidence/audit.json"],
        {
            "sha256": audit_identity.get("sha256"),
            "size_bytes": audit_identity.get("size_bytes"),
        },
        "ClinVar postflight audit identity",
    )
    _require_equal(
        file_identities["clinvar/2026-04-15/variants.parquet"],
        {"sha256": output.get("sha256"), "size_bytes": output.get("size_bytes")},
        "ClinVar postflight Parquet identity",
    )
    _require_equal(
        _require_exact_string_list(postflight.get("checks"), "ClinVar postflight.checks"),
        list(_CLINVAR_REMOTE_CHECKS),
        "ClinVar postflight checks",
    )
    parquet_audit = _require_mapping(
        postflight.get("parquet_audit"), "ClinVar postflight.parquet_audit"
    )
    _validate_clinvar_parquet_audit(
        parquet_audit,
        records=records,
        class_balance=class_balance,
        trusted_schema=_clinvar_parquet_schema(),
    )


def _assemble_gnomad(
    *,
    spec: Mapping[str, object],
    spec_dir: Path,
    source_lock_snapshot: SourceLockSnapshot,
) -> dict[str, Any]:
    _require_exact_keys(spec, {"repo", "repo_type", "shards"}, "lineage spec.gnomad")
    repo = _require_repo(spec.get("repo"), "lineage spec.gnomad.repo")
    _require_equal(spec.get("repo_type"), "dataset", "lineage spec.gnomad.repo_type")
    shard_specs = _require_list(spec.get("shards"), "lineage spec.gnomad.shards")
    if len(shard_specs) != 22:
        raise SnapshotLineageError("lineage spec.gnomad.shards must contain exactly 22 entries")

    seen_chromosomes: set[str] = set()
    seen_namespaces: set[str] = set()
    shards: list[dict[str, Any]] = []
    common_executions: list[Mapping[str, object]] = []
    common_lock_schema_hashes: list[str] = []
    for index, raw_shard_spec in enumerate(shard_specs):
        shard_spec = _require_mapping(raw_shard_spec, f"lineage spec.gnomad.shards[{index}]")
        _require_exact_keys(
            shard_spec,
            {
                "chromosome",
                "split_role",
                "revision",
                "namespace",
                "receipt_file",
                "receipt_sha256",
                "postflight_file",
                "postflight_sha256",
            },
            f"lineage spec.gnomad.shards[{index}]",
        )
        chromosome = _require_str(shard_spec.get("chromosome"), "gnomAD chromosome")
        if chromosome not in _AUTOSOMES:
            raise SnapshotLineageError(f"gnomAD chromosome must be one of 1..22: {chromosome!r}")
        if chromosome in seen_chromosomes:
            raise SnapshotLineageError(f"duplicate gnomAD chromosome: {chromosome}")
        seen_chromosomes.add(chromosome)
        expected_role = _split_role(chromosome)
        _require_equal(shard_spec.get("split_role"), expected_role, f"chr{chromosome} split_role")
        revision = _require_commit(shard_spec.get("revision"), f"chr{chromosome} revision")
        namespace = _require_namespace(shard_spec.get("namespace"), f"chr{chromosome} namespace")
        if namespace in seen_namespaces:
            raise SnapshotLineageError(f"duplicate gnomAD namespace: {namespace}")
        seen_namespaces.add(namespace)
        receipt_path = _resolve_bundle_file(
            spec_dir,
            shard_spec.get("receipt_file"),
            f"chr{chromosome} receipt_file",
        )
        expected_receipt_sha256 = _require_sha256(
            shard_spec.get("receipt_sha256"), f"chr{chromosome} receipt_sha256"
        )
        receipt_capture = _capture_json(receipt_path, f"chr{chromosome} staging receipt")
        _require_equal(
            receipt_capture.sha256, expected_receipt_sha256, f"chr{chromosome} receipt bytes"
        )
        receipt = receipt_capture.value
        shard, execution, schema_hash = _validate_gnomad_receipt(
            receipt=receipt,
            chromosome=chromosome,
            expected_role=expected_role,
            revision=revision,
            namespace=namespace,
            repo=repo,
            receipt_sha256=expected_receipt_sha256,
            receipt_size_bytes=receipt_capture.size_bytes,
            source_lock_snapshot=source_lock_snapshot,
        )
        postflight_path = _resolve_bundle_file(
            spec_dir,
            shard_spec.get("postflight_file"),
            f"chr{chromosome} postflight_file",
        )
        expected_postflight_sha256 = _require_sha256(
            shard_spec.get("postflight_sha256"), f"chr{chromosome} postflight_sha256"
        )
        postflight_capture = _capture_json(postflight_path, f"chr{chromosome} remote postflight")
        _require_equal(
            postflight_capture.sha256,
            expected_postflight_sha256,
            f"chr{chromosome} postflight bytes",
        )
        postflight = postflight_capture.value
        shard["remote_postflight"] = _validate_remote_postflight(
            postflight=postflight,
            postflight_sha256=expected_postflight_sha256,
            postflight_size_bytes=postflight_capture.size_bytes,
            repo=repo,
            revision=revision,
            namespace=namespace,
            chromosome=chromosome,
            execution=execution,
            receipt=receipt,
            receipt_sha256=expected_receipt_sha256,
            receipt_size_bytes=receipt_capture.size_bytes,
            shard=shard,
        )
        shards.append(shard)
        common_executions.append(execution)
        common_lock_schema_hashes.append(schema_hash)

    if seen_chromosomes != _AUTOSOMES:
        missing = sorted(_AUTOSOMES - seen_chromosomes, key=int)
        raise SnapshotLineageError(f"gnomAD autosome coverage incomplete: missing={missing}")
    first_execution = dict(common_executions[0])
    for execution in common_executions[1:]:
        _require_equal(execution, first_execution, "gnomAD common execution identity")
    if len(set(common_lock_schema_hashes)) != 1:
        raise SnapshotLineageError("gnomAD receipt source-lock schema identities differ")

    shards.sort(key=lambda shard: int(shard["chromosome"]))
    first = shards[0]
    return {
        "dataset_id": "gnomad-v4.1-exomes-autosomes",
        "release": "v4.1",
        "repo": repo,
        "repo_type": "dataset",
        "data_use": _gnomad_data_use(),
        "source_lock": {
            "schema_version": LOCK_SCHEMA_VERSION,
            "sha256": "sha256:" + hashlib.sha256(source_lock_snapshot.lock_bytes).hexdigest(),
            "schema_sha256": "sha256:" + common_lock_schema_hashes[0],
        },
        "transform": first["transform"],
        "common_execution": first_execution,
        "split_policy": {
            "train": [*(str(chromosome) for chromosome in range(1, 20)), "22"],
            "validation": ["20"],
            "evaluation": ["21"],
        },
        "total_records": sum(shard["output"]["records"] for shard in shards),
        "total_size_bytes": sum(shard["output"]["size_bytes"] for shard in shards),
        "shards": shards,
    }


def _validate_gnomad_receipt(
    *,
    receipt: Mapping[str, object],
    chromosome: str,
    expected_role: str,
    revision: str,
    namespace: str,
    repo: str,
    receipt_sha256: str,
    receipt_size_bytes: int,
    source_lock_snapshot: SourceLockSnapshot,
) -> tuple[dict[str, Any], Mapping[str, object], str]:
    _require_exact_keys(
        receipt,
        {
            "schema_version",
            "created_at",
            "ok",
            "dataset_id",
            "release",
            "reference_genome",
            "source_lock",
            "source",
            "transform",
            "output",
            "execution",
            "publication",
            "evidence",
            "claim_boundary",
        },
        f"chr{chromosome} staging receipt",
    )

    _require_equal(
        receipt.get("schema_version"),
        STAGING_RECEIPT_SCHEMA_VERSION,
        f"chr{chromosome} receipt.schema_version",
    )
    _require_equal(receipt.get("ok"), True, f"chr{chromosome} receipt.ok")
    execution = _require_mapping(receipt.get("execution"), f"chr{chromosome} execution")
    commit_sha = _require_commit(execution.get("commit_sha"), f"chr{chromosome} commit_sha")
    container_image = _require_container(
        execution.get("container_image"), f"chr{chromosome} container_image"
    )
    selection = select_source_from_snapshot(
        source_lock_snapshot,
        chromosome=chromosome,
        commit_sha=commit_sha,
        container_image=container_image,
    )
    _require_equal(receipt.get("dataset_id"), selection["dataset_id"], "gnomAD dataset_id")
    _require_equal(receipt.get("release"), selection["release"], "gnomAD release")
    _require_equal(
        receipt.get("reference_genome"), selection["reference_genome"], "gnomAD reference_genome"
    )
    _require_equal(execution, selection["execution"], f"chr{chromosome} execution")
    publication = _require_mapping(receipt.get("publication"), f"chr{chromosome} publication")
    _require_equal(publication, selection["publication"], f"chr{chromosome} publication")
    _require_equal(publication.get("repo"), repo, f"chr{chromosome} publication.repo")
    _require_equal(publication.get("repo_type"), "dataset", "gnomAD publication.repo_type")
    _require_equal(publication.get("namespace"), namespace, f"chr{chromosome} namespace")
    _require_equal(receipt.get("claim_boundary"), selection["claim_boundary"], "claim_boundary")

    selected_source = _require_mapping(selection["source"], "selected gnomAD source")
    source = _require_mapping(receipt.get("source"), f"chr{chromosome} source")
    _require_exact_keys(
        source,
        {
            "chromosome",
            "split_role",
            "bucket",
            "object",
            "generation",
            "size_bytes",
            "upstream_md5_base64",
            "upstream_md5_hex",
            "streamed_sha256",
        },
        f"chr{chromosome} source",
    )
    source_pairs = {
        "chromosome": "chromosome",
        "split_role": "split_role",
        "bucket": "bucket",
        "object": "object",
        "generation": "generation",
        "size_bytes": "size_bytes",
        "upstream_md5_base64": "md5_base64",
        "upstream_md5_hex": "md5_hex",
    }
    for receipt_field, selected_field in source_pairs.items():
        _require_equal(
            source.get(receipt_field),
            selected_source.get(selected_field),
            f"chr{chromosome} source.{receipt_field}",
        )
    _require_equal(source.get("split_role"), expected_role, f"chr{chromosome} source.split_role")
    source_sha256 = _require_bare_sha256(
        source.get("streamed_sha256"), f"chr{chromosome} source.streamed_sha256"
    )

    receipt_lock = _require_mapping(receipt.get("source_lock"), f"chr{chromosome} source_lock")
    selected_lock = _require_mapping(selection["source_lock"], "selected source lock")
    for field in ("sha256", "schema_version"):
        _require_equal(
            receipt_lock.get(field),
            selected_lock.get(field),
            f"chr{chromosome} source_lock.{field}",
        )
    receipt_lock_schema = _require_mapping(
        receipt_lock.get("schema"), f"chr{chromosome} source_lock.schema"
    )
    selected_lock_schema = _require_mapping(selected_lock.get("schema"), "selected lock schema")
    for field in ("sha256", "draft"):
        _require_equal(
            receipt_lock_schema.get(field),
            selected_lock_schema.get(field),
            f"chr{chromosome} source_lock.schema.{field}",
        )
    schema_hash = _require_bare_sha256(
        receipt_lock_schema.get("sha256"), f"chr{chromosome} source_lock.schema.sha256"
    )

    selected_transform = _require_mapping(selection["transform"], "selected transform")
    transform = _require_mapping(receipt.get("transform"), f"chr{chromosome} transform")
    _require_equal(
        transform.get("command"), selected_transform.get("command"), "gnomAD transform.command"
    )
    filters = _require_mapping(transform.get("filters"), f"chr{chromosome} transform.filters")
    for field in ("filter", "min_af", "max_allele_len"):
        _require_equal(
            filters.get(field), selected_transform.get(field), f"chr{chromosome} filters.{field}"
        )
    counts = _require_mapping(transform.get("counts"), f"chr{chromosome} transform.counts")
    records = _require_positive_int(counts.get("records_written"), "gnomAD records_written")
    classified = records + sum(
        _require_nonnegative_int(counts.get(field), f"gnomAD {field}")
        for field in ("skipped_filter", "skipped_af", "skipped_allele")
    )
    _require_equal(
        counts.get("allele_records_seen"),
        classified,
        f"chr{chromosome} allele count reconciliation",
    )
    runtime = _require_mapping(transform.get("runtime"), f"chr{chromosome} transform.runtime")
    _require_positive_number(runtime.get("elapsed_seconds"), "gnomAD elapsed_seconds")
    _require_positive_int(runtime.get("process_peak_rss_bytes"), "gnomAD peak RSS")

    output = _require_mapping(receipt.get("output"), f"chr{chromosome} output")
    output_sha256 = _require_bare_sha256(output.get("sha256"), f"chr{chromosome} output.sha256")
    output_size = _require_positive_int(output.get("size_bytes"), f"chr{chromosome} output.size")
    audit = _require_mapping(output.get("parquet_audit"), f"chr{chromosome} parquet_audit")
    _require_equal(
        audit.get("audit_method"),
        "pyarrow_metadata_and_full_iter_batches_scan_v1",
        "Parquet audit method",
    )
    _require_equal(audit.get("canonical_chromosome"), chromosome, "Parquet chromosome")
    _require_equal(audit.get("schema_version"), "2.0.0", "Parquet schema_version")
    _require_equal(audit.get("metadata_row_count"), records, "Parquet metadata row count")
    _require_equal(audit.get("scanned_row_count"), records, "Parquet scanned row count")
    population_counts = _require_mapping(
        audit.get("population_af_non_null_counts"), "Parquet population AF counts"
    )
    _require_exact_keys(
        population_counts, set(_GNOMAD_POPULATION_COLUMNS), "Parquet population AF counts"
    )
    missing_populations = sorted(
        population
        for population in _GNOMAD_V41_REQUIRED_POPULATIONS
        if _require_nonnegative_int(
            population_counts.get(population), f"Parquet {population} non-null count"
        )
        == 0
    )
    if missing_populations:
        raise SnapshotLineageError(
            f"Parquet required v4.1 population AF columns are empty: {missing_populations}"
        )

    return (
        {
            "chromosome": chromosome,
            "split_role": expected_role,
            "revision": revision,
            "namespace": namespace,
            "receipt": {
                "artifact_path": f"{namespace}/evidence/receipt.json",
                "sha256": receipt_sha256,
                "size_bytes": receipt_size_bytes,
            },
            "source": {
                "bucket": source["bucket"],
                "object": source["object"],
                "generation": source["generation"],
                "size_bytes": source["size_bytes"],
                "upstream_md5_base64": source["upstream_md5_base64"],
                "upstream_md5_hex": source["upstream_md5_hex"],
                "streamed_sha256": "sha256:" + source_sha256,
            },
            "transform": {
                "command": transform["command"],
                "filter": filters["filter"],
                "min_af": filters["min_af"],
                "max_allele_len": filters["max_allele_len"],
            },
            "output": {
                "artifact_path": f"{namespace}/data/gnomad/v4.1/variants.parquet",
                "sha256": "sha256:" + output_sha256,
                "size_bytes": output_size,
                "records": records,
                "schema_version": "2.0.0",
            },
        },
        execution,
        schema_hash,
    )


def _validate_remote_postflight(
    *,
    postflight: Mapping[str, object],
    postflight_sha256: str,
    postflight_size_bytes: int,
    repo: str,
    revision: str,
    namespace: str,
    chromosome: str,
    execution: Mapping[str, object],
    receipt: Mapping[str, object],
    receipt_sha256: str,
    receipt_size_bytes: int,
    shard: Mapping[str, Any],
) -> dict[str, object]:
    _require_exact_keys(
        postflight,
        {
            "schema_version",
            "ok",
            "repo_id",
            "repo_type",
            "revision",
            "namespace",
            "source_commit",
            "chromosome",
            "verified_files",
            "file_identities",
            "parquet_audit",
            "checks",
        },
        f"chr{chromosome} remote postflight",
    )
    _require_equal(
        postflight.get("schema_version"),
        REMOTE_POSTFLIGHT_SCHEMA_VERSION,
        f"chr{chromosome} postflight.schema_version",
    )
    _require_equal(postflight.get("ok"), True, f"chr{chromosome} postflight.ok")
    _require_equal(postflight.get("repo_id"), repo, f"chr{chromosome} postflight.repo_id")
    _require_equal(postflight.get("repo_type"), "dataset", f"chr{chromosome} postflight.repo_type")
    _require_equal(postflight.get("revision"), revision, f"chr{chromosome} postflight.revision")
    _require_equal(postflight.get("namespace"), namespace, f"chr{chromosome} postflight.namespace")
    _require_equal(
        postflight.get("source_commit"),
        execution.get("commit_sha"),
        f"chr{chromosome} postflight.source_commit",
    )
    _require_equal(
        postflight.get("chromosome"), chromosome, f"chr{chromosome} postflight.chromosome"
    )

    verified_files = _require_exact_string_list(
        postflight.get("verified_files"), f"chr{chromosome} postflight.verified_files"
    )
    _require_equal(
        verified_files,
        list(_REMOTE_NAMESPACE_FILES),
        f"chr{chromosome} postflight verified file set",
    )
    file_identities = _require_mapping(
        postflight.get("file_identities"), f"chr{chromosome} postflight.file_identities"
    )
    _require_exact_keys(
        file_identities,
        set(_REMOTE_NAMESPACE_FILES),
        f"chr{chromosome} postflight.file_identities",
    )
    normalized_identities: dict[str, dict[str, object]] = {}
    for relative_path in _REMOTE_NAMESPACE_FILES:
        identity = _require_mapping(
            file_identities.get(relative_path),
            f"chr{chromosome} postflight.file_identities[{relative_path!r}]",
        )
        _require_exact_keys(
            identity,
            {"sha256", "size_bytes"},
            f"chr{chromosome} postflight.file_identities[{relative_path!r}]",
        )
        normalized_identities[relative_path] = {
            "sha256": "sha256:"
            + _require_bare_sha256(
                identity.get("sha256"),
                f"chr{chromosome} postflight.file_identities[{relative_path!r}].sha256",
            ),
            "size_bytes": _require_positive_int(
                identity.get("size_bytes"),
                f"chr{chromosome} postflight.file_identities[{relative_path!r}].size_bytes",
            ),
        }

    _require_equal(
        normalized_identities["evidence/receipt.json"],
        {"sha256": receipt_sha256, "size_bytes": receipt_size_bytes},
        f"chr{chromosome} postflight receipt identity",
    )
    output = _require_mapping(shard.get("output"), f"chr{chromosome} lineage output")
    _require_equal(
        normalized_identities["data/gnomad/v4.1/variants.parquet"],
        {
            "sha256": _require_sha256(
                output.get("sha256"), f"chr{chromosome} lineage output.sha256"
            ),
            "size_bytes": _require_positive_int(
                output.get("size_bytes"), f"chr{chromosome} lineage output.size_bytes"
            ),
        },
        f"chr{chromosome} postflight Parquet identity",
    )
    receipt_output = _require_mapping(receipt.get("output"), f"chr{chromosome} receipt.output")
    parquet_audit = _require_mapping(
        postflight.get("parquet_audit"), f"chr{chromosome} postflight.parquet_audit"
    )
    _require_equal(
        parquet_audit,
        receipt_output.get("parquet_audit"),
        f"chr{chromosome} postflight fresh Parquet audit",
    )
    checks = _require_exact_string_list(
        postflight.get("checks"), f"chr{chromosome} postflight.checks"
    )
    _require_equal(
        checks,
        list(_REMOTE_POSTFLIGHT_CHECKS),
        f"chr{chromosome} postflight checks",
    )
    return {
        "schema_version": REMOTE_POSTFLIGHT_SCHEMA_VERSION,
        "sha256": postflight_sha256,
        "size_bytes": postflight_size_bytes,
        "verified_files": verified_files,
        "file_identities": normalized_identities,
        "checks": checks,
        "parquet_audit": dict(parquet_audit),
    }


def _assemble_clinvar(*, spec: Mapping[str, object], spec_dir: Path) -> dict[str, Any]:
    _require_exact_keys(
        spec,
        {
            "repo",
            "repo_type",
            "revision",
            "namespace",
            "audit_file",
            "audit_sha256",
            "postflight_file",
            "postflight_sha256",
        },
        "lineage spec.clinvar",
    )
    repo = _require_repo(spec.get("repo"), "lineage spec.clinvar.repo")
    _require_equal(spec.get("repo_type"), "dataset", "lineage spec.clinvar.repo_type")
    revision = _require_commit(spec.get("revision"), "lineage spec.clinvar.revision")
    namespace = _require_namespace(spec.get("namespace"), "lineage spec.clinvar.namespace")
    audit_path = _resolve_bundle_file(
        spec_dir, spec.get("audit_file"), "lineage spec.clinvar.audit_file"
    )
    audit_sha256 = _require_sha256(spec.get("audit_sha256"), "lineage spec.clinvar.audit_sha256")
    audit_capture = _capture_json(audit_path, "ClinVar audit")
    _require_equal(audit_capture.sha256, audit_sha256, "ClinVar audit bytes")
    postflight_path = _resolve_bundle_file(
        spec_dir, spec.get("postflight_file"), "lineage spec.clinvar.postflight_file"
    )
    postflight_sha256 = _require_sha256(
        spec.get("postflight_sha256"), "lineage spec.clinvar.postflight_sha256"
    )
    postflight_capture = _capture_json(postflight_path, "ClinVar remote postflight")
    _require_equal(postflight_capture.sha256, postflight_sha256, "ClinVar postflight bytes")
    audit = audit_capture.value
    _require_exact_keys(
        audit,
        {
            "claim_boundary",
            "commit_sha",
            "container_image",
            "generated_at",
            "generated_by",
            "ok",
            "output",
            "prepare_report",
            "runtime",
            "schema_version",
            "source",
        },
        "ClinVar audit",
    )
    _require_equal(audit.get("ok"), True, "ClinVar audit.ok")
    _require_equal(audit.get("schema_version"), "1.0.0", "ClinVar audit.schema_version")
    _require_equal(
        audit.get("generated_by"),
        "hf-job:clinvar-corrected-shard-audit",
        "ClinVar audit.generated_by",
    )
    _require_equal(
        audit.get("claim_boundary"), CLINVAR_REQUIRED_CLAIM_BOUNDARY, "ClinVar claim_boundary"
    )
    commit_sha = _require_commit(audit.get("commit_sha"), "ClinVar commit_sha")
    container_image = _require_container(audit.get("container_image"), "ClinVar container_image")

    source = _require_mapping(audit.get("source"), "ClinVar source")
    _require_exact_keys(source, {"md5", "release", "sha256", "size_bytes", "url"}, "ClinVar source")
    _require_equal(source.get("release"), "2026-04-15", "ClinVar source.release")
    source_sha256 = _require_sha256(source.get("sha256"), "ClinVar source.sha256")
    source_size = _require_positive_int(source.get("size_bytes"), "ClinVar source.size_bytes")
    source_md5 = _require_str(source.get("md5"), "ClinVar source.md5")
    if re.fullmatch(r"[0-9a-f]{32}", source_md5) is None:
        raise SnapshotLineageError("ClinVar source.md5 must be a lowercase MD5 digest")
    source_url = _require_str(source.get("url"), "ClinVar source.url")
    _require_equal(source_url, _CLINVAR_ARCHIVE_URL, "ClinVar source.url")

    output = _require_mapping(audit.get("output"), "ClinVar output")
    _require_exact_keys(
        output, {"class_balance", "path", "records", "sha256", "size_bytes"}, "ClinVar output"
    )
    _require_equal(output.get("path"), "clinvar/2026-04-15/variants.parquet", "ClinVar output.path")
    output_sha256 = _require_sha256(output.get("sha256"), "ClinVar output.sha256")
    output_size = _require_positive_int(output.get("size_bytes"), "ClinVar output.size_bytes")
    records = _require_positive_int(output.get("records"), "ClinVar output.records")
    class_balance_raw = _require_mapping(output.get("class_balance"), "ClinVar class_balance")
    _require_exact_keys(class_balance_raw, set(_CLINVAR_CLASSES), "ClinVar class_balance")
    class_balance = {
        label: _require_nonnegative_int(class_balance_raw.get(label), f"ClinVar class {label}")
        for label in sorted(_CLINVAR_CLASSES)
    }
    _require_equal(sum(class_balance.values()), records, "ClinVar class-balance total")
    for label in ("B", "LB", "LP", "P"):
        if class_balance[label] <= 0:
            raise SnapshotLineageError(f"ClinVar labelled class {label} must be non-empty")

    prepare = _require_mapping(audit.get("prepare_report"), "ClinVar prepare_report")
    _require_equal(prepare.get("already_exists"), False, "ClinVar prepare_report.already_exists")
    _require_equal(prepare.get("release"), source["release"], "ClinVar prepare release")
    _require_equal(prepare.get("input_sha256"), source_sha256, "ClinVar prepare input_sha256")
    _require_equal(prepare.get("input_size_bytes"), source_size, "ClinVar prepare input_size")
    input_vcf = _require_mapping(prepare.get("input_vcf"), "ClinVar input_vcf")
    _require_exact_keys(input_vcf, {"path", "sha256", "size_bytes"}, "ClinVar input_vcf")
    _require_equal(input_vcf.get("sha256"), source_sha256, "ClinVar input_vcf.sha256")
    _require_equal(input_vcf.get("size_bytes"), source_size, "ClinVar input_vcf.size_bytes")
    _require_equal(prepare.get("records_written"), records, "ClinVar prepare records_written")
    _require_equal(
        prepare.get("output_sha256", output_sha256), output_sha256, "ClinVar output hash"
    )
    output_parquet = _require_mapping(prepare.get("output_parquet"), "ClinVar output_parquet")
    _require_equal(output_parquet.get("sha256"), output_sha256, "ClinVar output_parquet.sha256")
    _require_equal(output_parquet.get("size_bytes"), output_size, "ClinVar output_parquet.size")
    allele_records = _require_positive_int(
        prepare.get("allele_records_seen"), "ClinVar allele_records_seen"
    )
    skipped_allele = _require_nonnegative_int(
        prepare.get("skipped_allele"), "ClinVar skipped_allele"
    )
    _require_equal(allele_records, records + skipped_allele, "ClinVar allele reconciliation")

    runtime = _require_mapping(audit.get("runtime"), "ClinVar runtime")
    _require_equal(runtime.get("returncode"), 0, "ClinVar runtime.returncode")
    _require_positive_number(runtime.get("wall_time_seconds"), "ClinVar wall_time_seconds")
    _require_positive_int(runtime.get("peak_rss_bytes"), "ClinVar peak_rss_bytes")

    postflight = postflight_capture.value
    remote_postflight = _validate_clinvar_remote_postflight(
        postflight=postflight,
        postflight_sha256=postflight_sha256,
        postflight_size_bytes=postflight_capture.size_bytes,
        repo=repo,
        revision=revision,
        namespace=namespace,
        audit=audit,
        audit_sha256=audit_sha256,
        audit_size_bytes=audit_capture.size_bytes,
        commit_sha=commit_sha,
        source=source,
        output=output,
        source_sha256=source_sha256,
        source_size=source_size,
        source_md5=source_md5,
        source_url=source_url,
        output_sha256=output_sha256,
        output_size=output_size,
        records=records,
        class_balance=class_balance,
    )

    return {
        "release": "2026-04-15",
        "reference_genome": "GRCh38",
        "repo": repo,
        "repo_type": "dataset",
        "data_use": _clinvar_data_use(),
        "revision": revision,
        "namespace": namespace,
        "audit": {
            "artifact_path": f"{namespace}/evidence/audit.json",
            "sha256": audit_sha256,
            "size_bytes": audit_capture.size_bytes,
        },
        "source": {
            "url": source_url,
            "md5": source_md5,
            "sha256": source_sha256,
            "size_bytes": source_size,
        },
        "output": {
            "artifact_path": f"{namespace}/{output['path']}",
            "sha256": output_sha256,
            "size_bytes": output_size,
            "records": records,
            "class_balance": class_balance,
        },
        "remote_postflight": remote_postflight,
        "execution": {"commit_sha": commit_sha, "container_image": container_image},
        "evidence_claim_boundary": CLINVAR_REQUIRED_CLAIM_BOUNDARY,
    }


def _validate_clinvar_remote_postflight(
    *,
    postflight: Mapping[str, object],
    postflight_sha256: str,
    postflight_size_bytes: int,
    repo: str,
    revision: str,
    namespace: str,
    audit: Mapping[str, object],
    audit_sha256: str,
    audit_size_bytes: int,
    commit_sha: str,
    source: Mapping[str, object],
    output: Mapping[str, object],
    source_sha256: str,
    source_size: int,
    source_md5: str,
    source_url: str,
    output_sha256: str,
    output_size: int,
    records: int,
    class_balance: Mapping[str, int],
) -> dict[str, object]:
    """Reconcile the exact-revision ClinVar postflight with local audit bytes."""
    _require_exact_keys(
        postflight,
        {
            "schema_version",
            "ok",
            "repo_id",
            "repo_type",
            "revision",
            "namespace",
            "source_commit",
            "release",
            "verified_files",
            "file_identities",
            "trusted_source_contract",
            "source_identity",
            "output_identity",
            "parquet_audit",
            "claim_boundary",
            "checks",
        },
        "ClinVar remote postflight",
    )
    _require_equal(
        postflight.get("schema_version"),
        CLINVAR_REMOTE_POSTFLIGHT_SCHEMA_VERSION,
        "ClinVar postflight.schema_version",
    )
    _require_equal(postflight.get("ok"), True, "ClinVar postflight.ok")
    _require_equal(postflight.get("repo_id"), repo, "ClinVar postflight.repo_id")
    _require_equal(postflight.get("repo_type"), "dataset", "ClinVar postflight.repo_type")
    _require_equal(postflight.get("revision"), revision, "ClinVar postflight.revision")
    _require_equal(postflight.get("namespace"), namespace, "ClinVar postflight.namespace")
    _require_equal(postflight.get("source_commit"), commit_sha, "ClinVar postflight.source_commit")
    _require_equal(postflight.get("release"), "2026-04-15", "ClinVar postflight.release")
    expected_namespace = f"staging/clinvar-2026-04-15-archive-{commit_sha[:12]}-r1"
    _require_equal(namespace, expected_namespace, "ClinVar exact-revision namespace")
    _require_equal(
        postflight.get("claim_boundary"),
        audit.get("claim_boundary"),
        "ClinVar postflight.claim_boundary",
    )

    verified_files = _require_exact_string_list(
        postflight.get("verified_files"), "ClinVar postflight.verified_files"
    )
    _require_equal(
        verified_files,
        list(_CLINVAR_REMOTE_FILES),
        "ClinVar postflight verified file set",
    )
    file_identities = _require_mapping(
        postflight.get("file_identities"), "ClinVar postflight.file_identities"
    )
    _require_exact_keys(
        file_identities,
        set(_CLINVAR_REMOTE_FILES),
        "ClinVar postflight.file_identities",
    )
    normalized_file_identities: dict[str, dict[str, object]] = {}
    for relative_path in _CLINVAR_REMOTE_FILES:
        identity = _require_mapping(
            file_identities.get(relative_path),
            f"ClinVar postflight.file_identities[{relative_path!r}]",
        )
        _require_exact_keys(
            identity,
            {"sha256", "size_bytes"},
            f"ClinVar postflight.file_identities[{relative_path!r}]",
        )
        normalized_file_identities[relative_path] = {
            "sha256": "sha256:"
            + _require_bare_sha256(
                identity.get("sha256"),
                f"ClinVar postflight.file_identities[{relative_path!r}].sha256",
            ),
            "size_bytes": _require_positive_int(
                identity.get("size_bytes"),
                f"ClinVar postflight.file_identities[{relative_path!r}].size_bytes",
            ),
        }
    _require_equal(
        normalized_file_identities["evidence/audit.json"],
        {"sha256": audit_sha256, "size_bytes": audit_size_bytes},
        "ClinVar postflight audit identity",
    )
    _require_equal(
        normalized_file_identities["clinvar/2026-04-15/variants.parquet"],
        {"sha256": output_sha256, "size_bytes": output_size},
        "ClinVar postflight Parquet identity",
    )

    trusted = _require_mapping(
        postflight.get("trusted_source_contract"),
        "ClinVar postflight.trusted_source_contract",
    )
    _validate_clinvar_trusted_source_contract(trusted)
    expected_schema = _clinvar_parquet_schema()

    source_identity = _require_mapping(
        postflight.get("source_identity"), "ClinVar postflight.source_identity"
    )
    _require_exact_keys(
        source_identity,
        {
            "url",
            "release",
            "md5",
            "sha256",
            "size_bytes",
            "verification_scope",
            "verification_limitation",
        },
        "ClinVar postflight.source_identity",
    )
    expected_source_identity = {
        "url": source_url,
        "release": source.get("release"),
        "md5": source_md5,
        "sha256": source_sha256.removeprefix("sha256:"),
        "size_bytes": source_size,
        "verification_scope": list(_CLINVAR_SOURCE_SCOPE),
        "verification_limitation": _CLINVAR_SOURCE_LIMITATION,
    }
    _require_equal(
        dict(source_identity), expected_source_identity, "ClinVar postflight source identity"
    )

    output_identity = _require_mapping(
        postflight.get("output_identity"), "ClinVar postflight.output_identity"
    )
    _require_exact_keys(
        output_identity,
        {"path", "sha256", "size_bytes", "records", "class_balance"},
        "ClinVar postflight.output_identity",
    )
    expected_output_identity = {
        "path": output.get("path"),
        "sha256": output_sha256.removeprefix("sha256:"),
        "size_bytes": output_size,
        "records": records,
        "class_balance": dict(class_balance),
    }
    _require_equal(
        dict(output_identity), expected_output_identity, "ClinVar postflight output identity"
    )

    parquet_audit = _require_mapping(
        postflight.get("parquet_audit"), "ClinVar postflight.parquet_audit"
    )
    _validate_clinvar_parquet_audit(
        parquet_audit,
        records=records,
        class_balance=class_balance,
        trusted_schema=expected_schema,
    )
    checks = _require_exact_string_list(postflight.get("checks"), "ClinVar postflight.checks")
    _require_equal(checks, list(_CLINVAR_REMOTE_CHECKS), "ClinVar postflight checks")
    return {
        "schema_version": CLINVAR_REMOTE_POSTFLIGHT_SCHEMA_VERSION,
        "sha256": postflight_sha256,
        "size_bytes": postflight_size_bytes,
        "verified_files": verified_files,
        "file_identities": normalized_file_identities,
        "checks": checks,
        "parquet_audit": dict(parquet_audit),
    }


def _validate_clinvar_trusted_source_contract(trusted: Mapping[str, object]) -> None:
    _require_exact_keys(
        trusted,
        {
            "files",
            "schema_version",
            "parquet_schema",
            "nullable_fields",
            "normalized_classes",
            "labelled_classes",
            "allele_alphabet",
            "cli_command",
            "max_allele_len",
            "output_path_template",
            "prepare_report_enrichments",
            "file_identity_fields",
        },
        "ClinVar postflight.trusted_source_contract",
    )
    files = _require_mapping(trusted.get("files"), "ClinVar trusted source files")
    _require_exact_keys(files, set(_CLINVAR_SOURCE_CONTRACT_FILES), "ClinVar trusted source files")
    for path in _CLINVAR_SOURCE_CONTRACT_FILES:
        identity = _require_mapping(files.get(path), f"ClinVar trusted source files[{path!r}]")
        _require_exact_keys(
            identity, {"sha256", "size_bytes"}, f"ClinVar trusted source files[{path!r}]"
        )
        _require_bare_sha256(identity.get("sha256"), f"ClinVar trusted source {path} SHA-256")
        _require_positive_int(identity.get("size_bytes"), f"ClinVar trusted source {path} size")
    expected_values: dict[str, object] = {
        "schema_version": "1.0.0",
        "parquet_schema": _clinvar_parquet_schema(),
        "nullable_fields": ["gene_symbol"],
        "normalized_classes": ["B", "LB", "LP", "OTHER", "P", "VUS"],
        "labelled_classes": ["B", "LB", "LP", "P"],
        "allele_alphabet": ["A", "C", "G", "T"],
        "cli_command": "geno-lewm-prepare-clinvar",
        "max_allele_len": 16,
        "output_path_template": "clinvar/{release}/variants.parquet",
        "prepare_report_enrichments": ["command", "input_vcf", "output_parquet", "runtime"],
        "file_identity_fields": ["path", "sha256", "size_bytes"],
    }
    for field, expected in expected_values.items():
        _require_equal(trusted.get(field), expected, f"ClinVar trusted source {field}")


def _validate_clinvar_parquet_audit(
    parquet_audit: Mapping[str, object],
    *,
    records: int,
    class_balance: Mapping[str, int],
    trusted_schema: list[dict[str, str]],
) -> None:
    _require_exact_keys(
        parquet_audit,
        {
            "metadata_row_count",
            "scanned_row_count",
            "class_balance",
            "chromosome_balance",
            "schema_version_balance",
            "null_counts",
            "position_range",
            "clinvar_id_range",
            "schema",
        },
        "ClinVar postflight.parquet_audit",
    )
    metadata_rows = _require_positive_int(
        parquet_audit.get("metadata_row_count"), "ClinVar postflight metadata row count"
    )
    scanned_rows = _require_positive_int(
        parquet_audit.get("scanned_row_count"), "ClinVar postflight scanned row count"
    )
    _require_equal(metadata_rows, records, "ClinVar postflight metadata row count")
    _require_equal(scanned_rows, records, "ClinVar postflight scanned row count")
    _require_equal(
        parquet_audit.get("class_balance"),
        dict(class_balance),
        "ClinVar postflight class balance",
    )
    if any(count <= 0 for count in class_balance.values()):
        raise SnapshotLineageError("ClinVar postflight class counts must be positive")

    chromosome_balance = _require_mapping(
        parquet_audit.get("chromosome_balance"), "ClinVar postflight chromosome balance"
    )
    if not chromosome_balance:
        raise SnapshotLineageError("ClinVar postflight chromosome balance must not be empty")
    chromosome_total = 0
    for chromosome, value in chromosome_balance.items():
        if not chromosome:
            raise SnapshotLineageError("ClinVar postflight chromosome names must be non-empty")
        chromosome_total += _require_positive_int(
            value, f"ClinVar postflight chromosome {chromosome!r} count"
        )
    _require_equal(chromosome_total, records, "ClinVar postflight chromosome total")
    _require_equal(
        parquet_audit.get("schema_version_balance"),
        {"1.0.0": records},
        "ClinVar postflight schema-version balance",
    )

    null_counts = _require_mapping(
        parquet_audit.get("null_counts"), "ClinVar postflight null counts"
    )
    field_names = {name for name, _kind in _CLINVAR_PARQUET_SCHEMA}
    _require_exact_keys(null_counts, field_names, "ClinVar postflight null counts")
    for field in sorted(field_names):
        count = _require_nonnegative_int(
            null_counts.get(field), f"ClinVar postflight null count {field}"
        )
        if field != "gene_symbol" and count != 0:
            raise SnapshotLineageError(
                f"ClinVar postflight required field {field!r} contains nulls"
            )
        if count > records:
            raise SnapshotLineageError(f"ClinVar postflight null count {field!r} exceeds records")
    for field in ("position_range", "clinvar_id_range"):
        bounds = _require_mapping(parquet_audit.get(field), f"ClinVar postflight {field}")
        _require_exact_keys(bounds, {"min", "max"}, f"ClinVar postflight {field}")
        minimum = _require_positive_int(bounds.get("min"), f"ClinVar postflight {field}.min")
        maximum = _require_positive_int(bounds.get("max"), f"ClinVar postflight {field}.max")
        if minimum > maximum:
            raise SnapshotLineageError(f"ClinVar postflight {field}.min exceeds max")
    _require_equal(parquet_audit.get("schema"), trusted_schema, "ClinVar postflight Parquet schema")


def _clinvar_parquet_schema() -> list[dict[str, str]]:
    return [{"name": name, "type": kind} for name, kind in _CLINVAR_PARQUET_SCHEMA]


def main(argv: list[str] | None = None) -> int:
    """CLI entry point for exact-evidence lineage assembly and verification."""
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    assemble_parser = subparsers.add_parser("assemble", help="assemble one lineage candidate")
    assemble_parser.add_argument("--spec-json", type=Path, required=True)
    assemble_parser.add_argument("--gnomad-source-lock-json", type=Path, required=True)
    assemble_parser.add_argument("--output-json", type=Path, required=True)
    verify_parser = subparsers.add_parser(
        "verify", help="recompute and verify one lineage candidate"
    )
    verify_parser.add_argument("--lineage-json", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        if args.command == "assemble":
            lineage = assemble_snapshot_lineage(
                spec_path=args.spec_json,
                gnomad_source_lock_path=args.gnomad_source_lock_json,
                output_path=args.output_json,
            )
            summary_path = ("output_json", str(args.output_json))
        else:
            lineage = verify_snapshot_lineage(args.lineage_json)
            summary_path = ("input_json", str(args.lineage_json))
    except (
        OSError,
        json.JSONDecodeError,
        UnicodeError,
        SourceLockError,
        SnapshotLineageError,
    ) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    print(
        json.dumps(
            {
                "ok": True,
                "candidate_snapshot_id": lineage["candidate_snapshot_id"],
                "lineage_id": lineage["lineage_id"],
                "membership_status": lineage["membership_status"],
                summary_path[0]: summary_path[1],
            },
            sort_keys=True,
        )
    )
    return 0


def _capture_json(path: Path, field: str) -> _CapturedJson:
    payload = path.read_bytes()
    try:
        raw: object = json.loads(
            payload,
            object_pairs_hook=_reject_duplicate_pairs,
            parse_constant=_reject_nonfinite_json_constant,
        )
    except SnapshotLineageError:
        raise
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        raise SnapshotLineageError(f"{field} is not valid JSON: {exc}") from exc
    return _CapturedJson(
        payload=payload,
        value=_require_mapping(raw, field),
        sha256="sha256:" + hashlib.sha256(payload).hexdigest(),
        size_bytes=len(payload),
    )


def _reject_duplicate_pairs(pairs: list[tuple[str, object]]) -> dict[str, object]:
    payload: dict[str, object] = {}
    for key, value in pairs:
        if key in payload:
            raise SnapshotLineageError(f"duplicate JSON key is not allowed: {key}")
        payload[key] = value
    return payload


def _reject_nonfinite_json_constant(constant: str) -> object:
    raise SnapshotLineageError(f"non-finite JSON number is not allowed: {constant}")


def _deep_freeze_json(value: object) -> object:
    if isinstance(value, Mapping):
        return MappingProxyType({key: _deep_freeze_json(child) for key, child in value.items()})
    if isinstance(value, list | tuple):
        return tuple(_deep_freeze_json(child) for child in value)
    return value


def _deep_thaw_json(value: object) -> object:
    if isinstance(value, Mapping):
        return {key: _deep_thaw_json(child) for key, child in value.items()}
    if isinstance(value, tuple | list):
        return [_deep_thaw_json(child) for child in value]
    return value


def _resolve_bundle_file(root: Path, value: object, field: str) -> Path:
    text = _require_str(value, field)
    candidate = Path(text)
    windows = PureWindowsPath(text)
    if (
        _SAFE_BUNDLE_JSON_PATH.fullmatch(text) is None
        or candidate.is_absolute()
        or windows.is_absolute()
        or windows.drive
        or ".." in candidate.parts
    ):
        raise SnapshotLineageError(
            f"{field} must be a relative in-bundle JSON path without traversal or backslashes"
        )
    root_resolved = root.resolve()
    resolved = (root / candidate).resolve()
    try:
        resolved.relative_to(root_resolved)
    except ValueError as exc:
        raise SnapshotLineageError(f"{field} resolves outside the evidence bundle") from exc
    if not resolved.is_file():
        raise SnapshotLineageError(f"{field} is not a regular file: {text}")
    return resolved


def _write_immutable_json(path: Path, payload: Mapping[str, object]) -> None:
    try:
        write_immutable_json(path, payload)
    except ImmutableJsonError as exc:
        message = str(exc).replace(
            "different bytes at immutable output",
            "different lineage bytes at immutable output",
        )
        message = message.replace("immutable output is", "lineage output is")
        raise SnapshotLineageError(message) from exc


def _fsync_directory(path: Path) -> None:
    _immutable_fsync_directory(path)


def _split_role(chromosome: str) -> str:
    if chromosome == "20":
        return "validation"
    if chromosome == "21":
        return "evaluation"
    return "train"


def _gnomad_data_use() -> dict[str, object]:
    return {
        "source": "gnomAD v4.1 exomes primary data",
        "terms_checked_on": "2026-07-13",
        "terms_urls": ["https://gnomad.broadinstitute.org/policies"],
        "license": {
            "spdx": "CC0-1.0",
            "scope": (
                "gnomAD primary exome data; third-party annotations may carry separate terms."
            ),
        },
        "attribution": (
            "Requested: cite https://doi.org/10.1038/s41586-023-06045-0 and link to "
            "https://gnomad.broadinstitute.org/."
        ),
        "restrictions": [
            "Do not attempt to reidentify participants.",
            "Review separate licenses before adding or redistributing third-party annotations.",
            "Recheck current upstream terms before redistribution or a new use.",
        ],
        "materialized_fields": [
            "chrom",
            "pos",
            "ref",
            "alt",
            "af_global",
            "af_afr",
            "af_ami",
            "af_amr",
            "af_asj",
            "af_eas",
            "af_fin",
            "af_mid",
            "af_nfe",
            "af_oth",
            "af_remaining",
            "af_sas",
            "filter",
            "schema_version",
        ],
    }


def _clinvar_data_use() -> dict[str, object]:
    return {
        "source": "ClinVar GRCh38 archived VCF release 2026-04-15",
        "terms_checked_on": "2026-07-13",
        "terms_urls": [
            "https://www.ncbi.nlm.nih.gov/clinvar/docs/maintenance_use/",
            "https://www.ncbi.nlm.nih.gov/home/about/policies/",
        ],
        "license": {
            "spdx": "NOASSERTION",
            "scope": (
                "NCBI places no restrictions on use or distribution of molecular data, but "
                "does not receive or transfer submitter rights."
            ),
        },
        "attribution": (
            "Requested: identify ClinVar as the data source and cite a current ClinVar publication."
        ),
        "restrictions": [
            (
                "Do not use ClinVar information directly for diagnosis or medical "
                "decision-making without review by a genetics professional."
            ),
            "NCBI does not independently verify submitted information.",
            (
                "Submitter or source-country intellectual-property claims may apply; NCBI "
                "cannot grant rights it does not hold."
            ),
            "Recheck current upstream terms before redistribution or a new use.",
        ],
        "materialized_fields": [
            "chrom",
            "pos",
            "ref",
            "alt",
            "clinical_significance",
            "review_status",
            "gene_symbol",
            "clinvar_id",
            "schema_version",
        ],
    }


def _require_mapping(value: object, field: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise SnapshotLineageError(f"{field} must be an object")
    return value


def _require_list(value: object, field: str) -> list[object]:
    if not isinstance(value, list):
        raise SnapshotLineageError(f"{field} must be an array")
    return value


def _require_exact_string_list(value: object, field: str) -> list[str]:
    raw_values = _require_list(value, field)
    values: list[str] = []
    for index, item in enumerate(raw_values):
        values.append(_require_str(item, f"{field}[{index}]"))
    return values


def _require_str(value: object, field: str) -> str:
    if not isinstance(value, str) or not value:
        raise SnapshotLineageError(f"{field} must be a non-empty string")
    return value


def _require_repo(value: object, field: str) -> str:
    repo = _require_str(value, field)
    if re.fullmatch(r"[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+", repo) is None:
        raise SnapshotLineageError(f"{field} must be a Hugging Face namespace/name pair")
    return repo


def _require_namespace(value: object, field: str) -> str:
    namespace = _require_str(value, field)
    if namespace.strip("/") != namespace or not namespace.startswith("staging/"):
        raise SnapshotLineageError(f"{field} must be a normalized staging namespace")
    if any(part in {"", ".", ".."} for part in namespace.split("/")):
        raise SnapshotLineageError(f"{field} contains an unsafe path component")
    return namespace


def _require_commit(value: object, field: str) -> str:
    commit = _require_str(value, field)
    if _COMMIT.fullmatch(commit) is None or commit == "0" * 40:
        raise SnapshotLineageError(f"{field} must be a non-zero full lowercase 40-character commit")
    return commit


def _require_container(value: object, field: str) -> str:
    container = _require_str(value, field)
    if _CONTAINER.fullmatch(container) is None:
        raise SnapshotLineageError(f"{field} must be pinned by a sha256 digest")
    return container


def _require_sha256(value: object, field: str) -> str:
    digest = _require_str(value, field)
    if _SHA256.fullmatch(digest) is None:
        raise SnapshotLineageError(f"{field} must be a sha256-prefixed lowercase digest")
    return digest


def _require_bare_sha256(value: object, field: str) -> str:
    digest = _require_str(value, field)
    if _BARE_SHA256.fullmatch(digest) is None:
        raise SnapshotLineageError(f"{field} must be a lowercase SHA-256 digest")
    return digest


def _require_positive_int(value: object, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise SnapshotLineageError(f"{field} must be a positive integer")
    return value


def _require_nonnegative_int(value: object, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise SnapshotLineageError(f"{field} must be a non-negative integer")
    return value


def _require_positive_number(value: object, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise SnapshotLineageError(f"{field} must be numeric")
    normalized = float(value)
    if not math.isfinite(normalized) or normalized <= 0:
        raise SnapshotLineageError(f"{field} must be finite and positive")
    return normalized


def _require_exact_keys(value: Mapping[str, object], expected: set[str], field: str) -> None:
    observed = set(value)
    if observed != expected:
        raise SnapshotLineageError(
            f"{field} keys drifted: missing={sorted(expected - observed)}, "
            f"unexpected={sorted(observed - expected)}"
        )


def _require_equal(observed: object, expected: object, field: str) -> None:
    if not _json_equal(observed, expected):
        raise SnapshotLineageError(f"{field} drifted: expected {expected!r}, observed {observed!r}")


def _json_equal(observed: object, expected: object) -> bool:
    if type(observed) is not type(expected):
        return False
    if isinstance(observed, Mapping):
        if not isinstance(expected, Mapping) or set(observed) != set(expected):
            return False
        return all(_json_equal(observed[key], expected[key]) for key in observed)
    if isinstance(observed, list):
        if not isinstance(expected, list) or len(observed) != len(expected):
            return False
        return all(
            _json_equal(observed_item, expected_item)
            for observed_item, expected_item in zip(observed, expected, strict=True)
        )
    return observed == expected


if __name__ == "__main__":  # pragma: no cover - exercised through ``main`` in tests
    raise SystemExit(main())
