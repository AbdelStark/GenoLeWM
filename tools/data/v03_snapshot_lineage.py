# SPDX-License-Identifier: Apache-2.0
"""Assemble immutable v0.3 staging evidence into a lineage candidate.

This tool records source-to-staged-artifact lineage only. It deliberately does
not create dataset memberships or claim that a publishable snapshot exists.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from collections.abc import Mapping
from pathlib import Path, PureWindowsPath
from typing import Any

from geno_lewm.data._snapshot_lineage import (
    _AUTOSOMES,
    _CANDIDATE_ID,
    _CLINVAR_ARCHIVE_URL,
    _CLINVAR_CLASSES,
    _CLINVAR_REMOTE_CHECKS,
    _CLINVAR_REMOTE_FILES,
    _CLINVAR_SOURCE_CONTRACT_FILES,
    _CLINVAR_SOURCE_LIMITATION,
    _CLINVAR_SOURCE_SCOPE,
    _GNOMAD_POPULATION_COLUMNS,
    _GNOMAD_V41_REQUIRED_POPULATIONS,
    _REMOTE_NAMESPACE_FILES,
    _REMOTE_POSTFLIGHT_CHECKS,
    _SAFE_BUNDLE_JSON_PATH,
    CLINVAR_REQUIRED_CLAIM_BOUNDARY,
    GENERATED_BY,
    LINEAGE_CLAIM_BOUNDARY,
    LINEAGE_SCHEMA_VERSION,
    MEMBERSHIP_STATUS,
    SPEC_SCHEMA_VERSION,
    SnapshotLineageError,
    VerifiedSnapshotLineage,
    _capture_json,
    _clinvar_data_use,
    _clinvar_parquet_schema,
    _gnomad_data_use,
    _require_bare_sha256,
    _require_commit,
    _require_container,
    _require_equal,
    _require_exact_keys,
    _require_exact_string_list,
    _require_list,
    _require_mapping,
    _require_namespace,
    _require_nonnegative_int,
    _require_positive_int,
    _require_positive_number,
    _require_repo,
    _require_sha256,
    _require_str,
    _split_role,
    _validate_clinvar_parquet_audit,
    capture_verified_snapshot_lineage,
    verify_snapshot_lineage,
)
from geno_lewm.data._v03_evidence_contract import (
    CLINVAR_REMOTE_POSTFLIGHT_SCHEMA_VERSION,
    GNOMAD_REMOTE_POSTFLIGHT_SCHEMA_VERSION as REMOTE_POSTFLIGHT_SCHEMA_VERSION,
    GNOMAD_SOURCE_LOCK_SCHEMA_VERSION as LOCK_SCHEMA_VERSION,
    GNOMAD_STAGING_RECEIPT_SCHEMA_VERSION as STAGING_RECEIPT_SCHEMA_VERSION,
)
from geno_lewm.provenance import canonical_json_sha256
from tools.data._immutable_json import (
    ImmutableJsonError,
    _fsync_directory as _immutable_fsync_directory,
    write_immutable_json,
)
from tools.data.v03_gnomad_lock import (
    SourceLockError,
    SourceLockSnapshot,
    capture_source_lock,
    select_source_from_snapshot,
)

__all__ = [
    "SnapshotLineageError",
    "VerifiedSnapshotLineage",
    "capture_verified_snapshot_lineage",
    "verify_snapshot_lineage",
]


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


if __name__ == "__main__":  # pragma: no cover - exercised through ``main`` in tests
    raise SystemExit(main())
