# SPDX-License-Identifier: Apache-2.0
"""Snapshot-lineage parsing and exact source binding."""

from __future__ import annotations

import hashlib
import os
import stat
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

from geno_lewm.data._membership_store_contract import (
    _AUTOSOMES,
    _CLINVAR_PARQUET_FIELDS,
    _CLINVAR_REMOTE_POSTFLIGHT_CHECKS,
    _CLINVAR_REMOTE_POSTFLIGHT_FILES,
    _GNOMAD_REMOTE_POSTFLIGHT_CHECKS,
    _GNOMAD_REMOTE_POSTFLIGHT_FILES,
    _SNAPSHOT_LINEAGE_SCHEMA_VERSION,
    MembershipSourceBinding,
    MembershipSourceInput,
    SnapshotLineageBinding,
    _parse_count_mapping,
    _read_json_mapping,
    _require_exact_keys,
    _require_mapping,
    _require_positive_int,
    _require_sha256,
    _require_text,
)
from geno_lewm.data._snapshot_lineage import (
    SnapshotLineageError,
    capture_verified_snapshot_lineage,
)
from geno_lewm.data.clinvar import CLINVAR_SCHEMA_VERSION
from geno_lewm.data.gnomad import GNOMAD_SCHEMA_VERSION
from geno_lewm.data.membership import V03_CHROMOSOME_ROLES
from geno_lewm.data.variant_identity import canonicalize_chromosome
from geno_lewm.errors import InputError, SchemaCompatError
from geno_lewm.provenance.hashing import canonical_json_sha256


@dataclass(frozen=True, slots=True)
class _ExpectedSource:
    binding: MembershipSourceBinding
    chromosome: str | None


@dataclass(frozen=True, slots=True)
class _CapturedSnapshotLineage:
    payload: bytes
    lineage: Mapping[str, object]
    payload_sha256: str
    size_bytes: int


def _load_snapshot_lineage(
    path: Path,
) -> tuple[SnapshotLineageBinding, dict[str, _ExpectedSource], bytes]:
    capture, is_fixture = _capture_snapshot_lineage(path)
    raw_bytes = capture.payload
    payload = capture.lineage
    _require_exact_keys(
        payload,
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
    if payload.get("schema_version") != _SNAPSHOT_LINEAGE_SCHEMA_VERSION:
        raise SchemaCompatError("snapshot lineage schema version mismatch")
    if payload.get("generated_by") != "tools.data.v03_snapshot_lineage":
        raise InputError("snapshot lineage generator is not recognized")
    if payload.get("reference_genome") != "GRCh38":
        raise InputError("snapshot lineage reference genome must be GRCh38")
    if payload.get("membership_status") != "not_created":
        raise InputError("snapshot lineage must precede membership creation")
    assembly_inputs = _require_mapping(
        payload.get("assembly_inputs"), "snapshot lineage assembly_inputs"
    )
    if is_fixture != (assembly_inputs == {"fixture": True}):
        raise InputError("snapshot lineage fixture status changed after capture")
    declared_lineage_id = _require_sha256(payload.get("lineage_id"), "snapshot lineage_id")
    if is_fixture:
        identity_payload = {key: value for key, value in payload.items() if key != "lineage_id"}
        computed_lineage_id = canonical_json_sha256(identity_payload)
        if declared_lineage_id != computed_lineage_id:
            raise InputError(
                "snapshot lineage identity mismatch",
                details={"declared": declared_lineage_id, "computed": computed_lineage_id},
            )
    candidate_id = _require_text(
        payload.get("candidate_snapshot_id"), "snapshot candidate_snapshot_id"
    )
    binding = SnapshotLineageBinding(
        lineage_id=declared_lineage_id,
        sha256=capture.payload_sha256,
        size_bytes=capture.size_bytes,
        candidate_snapshot_id=candidate_id,
    )
    gnomad = _require_mapping(payload.get("gnomad"), "snapshot lineage gnomad")
    clinvar = _require_mapping(payload.get("clinvar"), "snapshot lineage clinvar")
    expected: dict[str, _ExpectedSource] = {}
    repo = _require_text(gnomad.get("repo"), "gnomAD repository")
    shards = _require_sequence(gnomad.get("shards"), "gnomAD shards")
    if len(shards) != 22:
        raise InputError("snapshot lineage must contain exactly 22 gnomAD shards")
    for index, item in enumerate(shards):
        shard = _require_mapping(item, f"gnomAD shards[{index}]")
        chromosome = canonicalize_chromosome(
            _require_text(shard.get("chromosome"), "gnomAD chromosome")
        )
        if chromosome not in _AUTOSOMES:
            raise InputError("snapshot lineage gnomAD chromosome must be one of 1..22")
        source_id = f"gnomad-v4.1-chr{chromosome}"
        if source_id in expected:
            raise InputError("snapshot lineage contains duplicate gnomAD chromosome")
        role = _require_text(shard.get("split_role"), "gnomAD split_role")
        if role != V03_CHROMOSOME_ROLES.role_for(chromosome):
            raise InputError("snapshot lineage gnomAD role drifts from v0.3 split")
        output = _require_mapping(shard.get("output"), "gnomAD output")
        postflight = _require_mapping(shard.get("remote_postflight"), "gnomAD postflight")
        if is_fixture:
            _validate_reduced_remote_postflight(
                postflight,
                field=f"gnomAD chr{chromosome} remote_postflight",
                schema_version="geno-lewm.gnomad-remote-postflight.v1",
                verified_files=_GNOMAD_REMOTE_POSTFLIGHT_FILES,
                checks=_GNOMAD_REMOTE_POSTFLIGHT_CHECKS,
            )
        artifact_schema_version = _require_text(
            output.get("schema_version"), "gnomAD artifact schema"
        )
        if artifact_schema_version != GNOMAD_SCHEMA_VERSION:
            raise InputError("snapshot lineage gnomAD artifact schema version mismatch")
        expected[source_id] = _ExpectedSource(
            binding=MembershipSourceBinding(
                source_id=source_id,
                kind="gnomad",
                repository=repo,
                revision=_require_text(shard.get("revision"), "gnomAD revision"),
                namespace=_require_text(shard.get("namespace"), "gnomAD namespace"),
                artifact_path=_require_text(output.get("artifact_path"), "gnomAD artifact_path"),
                artifact_sha256=_require_text(output.get("sha256"), "gnomAD artifact sha256"),
                artifact_size_bytes=_require_positive_int(
                    output.get("size_bytes"), "gnomAD artifact size"
                ),
                artifact_row_count=_require_positive_int(
                    output.get("records"), "gnomAD artifact rows"
                ),
                artifact_schema_version=artifact_schema_version,
                verification_kind="remote_postflight",
                verification_sha256=_require_text(
                    postflight.get("sha256"), "gnomAD postflight sha256"
                ),
                membership_row_count=0,
                filtered_row_count=_require_positive_int(output.get("records"), "gnomAD rows"),
            ),
            chromosome=chromosome,
        )
    if {expected_source.chromosome for expected_source in expected.values()} != _AUTOSOMES:
        raise InputError("snapshot lineage gnomAD autosome coverage is incomplete")

    clinvar_repo = _require_text(clinvar.get("repo"), "ClinVar repository")
    if clinvar_repo != repo:
        raise InputError("snapshot lineage source repositories differ")
    clinvar_output = _require_mapping(clinvar.get("output"), "ClinVar output")
    _require_mapping(clinvar.get("audit"), "ClinVar audit")
    clinvar_postflight = _require_mapping(
        clinvar.get("remote_postflight"), "ClinVar remote_postflight"
    )
    source_id = "clinvar-2026-04-15"
    artifact_rows = _require_positive_int(clinvar_output.get("records"), "ClinVar rows")
    if is_fixture:
        _validate_reduced_remote_postflight(
            clinvar_postflight,
            field="ClinVar remote_postflight",
            schema_version="geno-lewm.clinvar-remote-postflight.v1",
            verified_files=_CLINVAR_REMOTE_POSTFLIGHT_FILES,
            checks=_CLINVAR_REMOTE_POSTFLIGHT_CHECKS,
            clinvar_rows=artifact_rows,
        )
    expected[source_id] = _ExpectedSource(
        binding=MembershipSourceBinding(
            source_id=source_id,
            kind="clinvar",
            repository=clinvar_repo,
            revision=_require_text(clinvar.get("revision"), "ClinVar revision"),
            namespace=_require_text(clinvar.get("namespace"), "ClinVar namespace"),
            artifact_path=_require_text(
                clinvar_output.get("artifact_path"), "ClinVar artifact_path"
            ),
            artifact_sha256=_require_text(clinvar_output.get("sha256"), "ClinVar artifact sha256"),
            artifact_size_bytes=_require_positive_int(
                clinvar_output.get("size_bytes"), "ClinVar artifact size"
            ),
            artifact_row_count=artifact_rows,
            artifact_schema_version=CLINVAR_SCHEMA_VERSION,
            verification_kind="remote_postflight",
            verification_sha256=_require_text(
                clinvar_postflight.get("sha256"), "ClinVar remote-postflight sha256"
            ),
            membership_row_count=0,
            filtered_row_count=artifact_rows,
        ),
        chromosome=None,
    )
    return binding, expected, raw_bytes


def _capture_snapshot_lineage(path: Path) -> tuple[_CapturedSnapshotLineage, bool]:
    """Use the official one-read verifier, with a closed synthetic-fixture fallback."""
    try:
        return _capture_official_snapshot_lineage(path), False
    except SnapshotLineageError as exc:
        official_error = exc
    except OSError as exc:
        raise InputError(
            "failed to read snapshot lineage",
            details={"path": str(path)},
        ) from exc

    raw_bytes, payload = _read_json_mapping(path, "snapshot lineage fixture")
    assembly_inputs = _require_mapping(
        payload.get("assembly_inputs"), "snapshot lineage fixture assembly_inputs"
    )
    if assembly_inputs != {"fixture": True}:
        raise InputError(
            "non-fixture snapshot lineage failed official verification",
            details={"path": str(path), "reason": str(official_error)},
            remediation="regenerate and verify the v0.3 snapshot-lineage artifact",
        ) from official_error
    return (
        _CapturedSnapshotLineage(
            payload=raw_bytes,
            lineage=payload,
            payload_sha256="sha256:" + hashlib.sha256(raw_bytes).hexdigest(),
            size_bytes=len(raw_bytes),
        ),
        True,
    )


def _capture_official_snapshot_lineage(path: Path) -> _CapturedSnapshotLineage:
    verified = capture_verified_snapshot_lineage(path)
    payload = verified.payload
    if not isinstance(payload, bytes):
        raise InputError("official snapshot-lineage capture payload must be bytes")
    lineage = _require_mapping(verified.lineage, "official verified snapshot lineage")
    sha256 = _require_sha256(
        verified.payload_sha256,
        "official snapshot-lineage capture payload_sha256",
    )
    size_bytes = _require_positive_int(
        verified.size_bytes,
        "official snapshot-lineage capture size_bytes",
    )
    observed_sha256 = "sha256:" + hashlib.sha256(payload).hexdigest()
    if sha256 != observed_sha256 or size_bytes != len(payload):
        raise InputError(
            "official snapshot-lineage capture identity mismatch",
            details={
                "declared": {"sha256": sha256, "size_bytes": size_bytes},
                "observed": {"sha256": observed_sha256, "size_bytes": len(payload)},
            },
        )
    return _CapturedSnapshotLineage(
        payload=payload,
        lineage=lineage,
        payload_sha256=sha256,
        size_bytes=size_bytes,
    )


def _require_sequence(value: object, field: str) -> Sequence[object]:
    if isinstance(value, str | bytes | bytearray) or not isinstance(value, Sequence):
        raise InputError(f"{field} must be an array")
    return value


def _capture_source_artifact(
    source_input: MembershipSourceInput,
    binding: MembershipSourceBinding,
    capture_root: Path,
) -> MembershipSourceInput:
    """Copy one source through a single checked descriptor into private storage."""
    path = source_input.path
    source_fd: int | None = None
    destination_fd: int | None = None
    destination = capture_root / f"{binding.source_id}-{binding.artifact_sha256[7:]}.parquet"
    try:
        source_flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
        source_fd = os.open(path, source_flags)
        source_stat = os.fstat(source_fd)
        if not stat.S_ISREG(source_stat.st_mode):
            raise InputError("membership source artifact must be a regular file")
        destination_fd = os.open(
            destination,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL,
            0o400,
        )
        digest = hashlib.sha256()
        size_bytes = 0
        while chunk := os.read(source_fd, 1024 * 1024):
            digest.update(chunk)
            size_bytes += len(chunk)
            view = memoryview(chunk)
            while view:
                written = os.write(destination_fd, view)
                if written < 1:
                    raise InputError("membership source capture produced a short write")
                view = view[written:]
        os.fsync(destination_fd)
    except InputError:
        _delete_captured_artifact(destination)
        raise
    except OSError as exc:
        _delete_captured_artifact(destination)
        raise InputError(
            "membership source artifact cannot be read",
            details={"source_id": binding.source_id, "path": str(path)},
        ) from exc
    finally:
        if destination_fd is not None:
            os.close(destination_fd)
        if source_fd is not None:
            os.close(source_fd)

    observed = {"sha256": "sha256:" + digest.hexdigest(), "size_bytes": size_bytes}
    expected = {"sha256": binding.artifact_sha256, "size_bytes": binding.artifact_size_bytes}
    if observed != expected:
        _delete_captured_artifact(destination)
        raise InputError(
            "membership source artifact identity mismatch",
            details={"source_id": binding.source_id, "declared": expected, "observed": observed},
        )
    return MembershipSourceInput(
        kind=source_input.kind,
        path=destination,
        chromosome=source_input.chromosome,
    )


def _delete_captured_artifact(path: Path) -> None:
    try:
        path.chmod(stat.S_IRUSR | stat.S_IWUSR)
    except FileNotFoundError:
        return
    except OSError as exc:
        raise InputError("private membership source capture cannot be removed") from exc
    try:
        path.unlink()
    except OSError as exc:
        raise InputError("private membership source capture cannot be removed") from exc


def _validate_reduced_remote_postflight(
    value: Mapping[str, object],
    *,
    field: str,
    schema_version: str,
    verified_files: tuple[str, ...],
    checks: tuple[str, ...],
    clinvar_rows: int | None = None,
) -> None:
    expected_keys = {"schema_version", "sha256", "size_bytes", "verified_files", "checks"}
    if clinvar_rows is not None:
        expected_keys.add("parquet_audit")
    _require_exact_keys(value, expected_keys, field)
    if value.get("schema_version") != schema_version:
        raise InputError(f"{field} schema version mismatch")
    _require_sha256(value.get("sha256"), f"{field} sha256")
    _require_positive_int(value.get("size_bytes"), f"{field} size_bytes")
    _require_exact_string_list(value.get("verified_files"), verified_files, f"{field} files")
    _require_exact_string_list(value.get("checks"), checks, f"{field} checks")
    if clinvar_rows is not None:
        audit = _require_mapping(value.get("parquet_audit"), f"{field} parquet_audit")
        _validate_clinvar_parquet_audit(audit, rows=clinvar_rows)


def _validate_clinvar_parquet_audit(value: Mapping[str, object], *, rows: int) -> None:
    _require_exact_keys(
        value,
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
        "ClinVar remote_postflight parquet_audit",
    )
    if value.get("metadata_row_count") != rows or value.get("scanned_row_count") != rows:
        raise InputError("ClinVar remote-postflight row counts drift from lineage output")
    for field in ("class_balance", "chromosome_balance"):
        counts = _parse_count_mapping(
            value.get(field), f"ClinVar remote_postflight parquet_audit.{field}"
        )
        if not counts or sum(counts.values()) != rows:
            raise InputError(f"ClinVar remote-postflight {field} does not sum to output rows")
    if value.get("schema_version_balance") != {CLINVAR_SCHEMA_VERSION: rows}:
        raise InputError("ClinVar remote-postflight schema-version balance mismatch")
    null_counts = _parse_count_mapping(
        value.get("null_counts"), "ClinVar remote_postflight parquet_audit.null_counts"
    )
    expected_fields = {name for name, _kind in _CLINVAR_PARQUET_FIELDS}
    if set(null_counts) != expected_fields:
        raise InputError("ClinVar remote-postflight null-count fields mismatch")
    if any(count > rows for count in null_counts.values()):
        raise InputError("ClinVar remote-postflight null count exceeds output rows")
    if any(null_counts[field] for field in expected_fields - {"gene_symbol"}):
        raise InputError("ClinVar remote-postflight required field contains nulls")
    for field in ("position_range", "clinvar_id_range"):
        bounds = _require_mapping(value.get(field), f"ClinVar remote_postflight {field}")
        _require_exact_keys(bounds, {"min", "max"}, f"ClinVar remote_postflight {field}")
        minimum = _require_positive_int(bounds.get("min"), f"ClinVar {field}.min")
        maximum = _require_positive_int(bounds.get("max"), f"ClinVar {field}.max")
        if minimum > maximum:
            raise InputError(f"ClinVar remote-postflight {field} minimum exceeds maximum")
    expected_schema = [{"name": name, "type": kind} for name, kind in _CLINVAR_PARQUET_FIELDS]
    if value.get("schema") != expected_schema:
        raise InputError("ClinVar remote-postflight Parquet schema mismatch")


def _require_exact_string_list(value: object, expected: tuple[str, ...], field: str) -> None:
    if not isinstance(value, list) or any(not isinstance(item, str) for item in value):
        raise InputError(f"{field} must be an ordered string array")
    if tuple(value) != expected:
        raise InputError(f"{field} does not match the closed contract")
