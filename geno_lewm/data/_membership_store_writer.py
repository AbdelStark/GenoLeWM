# SPDX-License-Identifier: Apache-2.0
"""Streaming, disk-backed membership-store writer."""

from __future__ import annotations

import os
import shutil
import sqlite3
import tempfile
from collections.abc import Callable, Iterator, Mapping, Sequence
from contextlib import suppress
from dataclasses import replace
from pathlib import Path

from geno_lewm.data._membership_store_contract import (
    _ARTIFACT_ID,
    _AUTOSOMES,
    _CHROMOSOME_RANK,
    _CLINVAR_CLASSES,
    _CLINVAR_HOLDOUT_CLASSES,
    _CLINVAR_REASON_MASK,
    _GNOMAD_REASON_MASK,
    _INDEX_NAME,
    _LINEAGE_NAME,
    _MANIFEST_NAME,
    _PARQUET_BATCH_ROWS,
    _PARQUET_NAME,
    _RECEIPT_NAME,
    MEMBERSHIP_STORE_SCHEMA_VERSION,
    MembershipSourceBinding,
    MembershipSourceInput,
    MembershipStoreFile,
    MembershipStoreManifest,
    _require_commit,
    _require_sha256,
    _write_json,
)
from geno_lewm.data._membership_store_lineage import (
    _capture_source_artifact,
    _delete_captured_artifact,
    _ExpectedSource,
    _load_snapshot_lineage,
)
from geno_lewm.data._membership_store_receipt import (
    _create_build_receipt,
    _require_container_image,
)
from geno_lewm.data._membership_store_storage import (
    _clinvar_schema,
    _create_index,
    _create_lookup_indexes,
    _gnomad_schema,
    _require_no_cross_role_leakage,
    _require_pyarrow,
    _summarize_index,
    _write_index_metadata,
    _write_membership_parquet,
)
from geno_lewm.data._membership_store_verifier import verify_membership_store
from geno_lewm.data.clinvar import CLINVAR_SCHEMA_VERSION
from geno_lewm.data.gnomad import GNOMAD_SCHEMA_VERSION
from geno_lewm.data.membership import (
    REQUIRED_MEMBERSHIP_ROLES,
    V03_CHROMOSOME_ROLES,
    MembershipRow,
)
from geno_lewm.data.variant_identity import CanonicalVariant, canonicalize_chromosome
from geno_lewm.errors import InputError
from geno_lewm.provenance.hashing import canonical_json_sha256, sha256_file


def build_membership_store(
    *,
    artifact_id: str,
    snapshot_lineage_path: Path,
    expected_snapshot_lineage_sha256: str,
    builder_git_commit: str,
    container_image: str,
    sources: Sequence[MembershipSourceInput],
    output_dir: Path,
) -> MembershipStoreManifest:
    """Build one immutable store from exact local staged-source bytes.

    The source list must exactly cover all 22 gnomAD shards and the ClinVar
    shard named by the lineage.  There is no API for injecting preconstructed
    membership rows: every row is derived by the checked source adapters.
    """
    if _ARTIFACT_ID.fullmatch(artifact_id) is None:
        raise InputError("membership store artifact_id is not canonical")
    _require_commit(builder_git_commit, "membership builder git_commit")
    _require_container_image(container_image)
    pa, _pq = _require_pyarrow()
    expected_lineage_sha256 = _require_sha256(
        expected_snapshot_lineage_sha256,
        "expected snapshot lineage sha256",
    )
    lineage_binding, expected_sources, lineage_bytes = _load_snapshot_lineage(
        Path(snapshot_lineage_path)
    )
    if lineage_binding.sha256 != expected_lineage_sha256:
        raise InputError(
            "snapshot lineage does not match the expected byte identity",
            details={
                "expected": expected_lineage_sha256,
                "observed": lineage_binding.sha256,
            },
        )
    normalized_sources = tuple(sources)
    if not normalized_sources or not all(
        isinstance(source, MembershipSourceInput) for source in normalized_sources
    ):
        raise InputError("membership sources must contain MembershipSourceInput values")
    by_id = {source.source_id: source for source in normalized_sources}
    if len(by_id) != len(normalized_sources):
        raise InputError("membership source inputs must have unique source identifiers")
    missing = set(expected_sources) - set(by_id)
    extra = set(by_id) - set(expected_sources)
    if missing or extra:
        raise InputError(
            "membership source inputs do not exactly cover snapshot lineage",
            details={"missing": sorted(missing), "extra": sorted(extra)},
        )

    output = Path(output_dir)
    if output.exists():
        raise InputError(
            "membership store output already exists",
            details={"path": str(output)},
            remediation="choose a new immutable artifact directory",
        )
    try:
        output.parent.mkdir(parents=True, exist_ok=True)
        temporary = Path(tempfile.mkdtemp(prefix=f".{output.name}.tmp-", dir=output.parent))
        capture_root = Path(tempfile.mkdtemp(prefix=f".{output.name}.sources-", dir=output.parent))
    except OSError as exc:
        if "temporary" in locals():
            shutil.rmtree(temporary, ignore_errors=True)
        raise InputError("membership store temporary directories cannot be created") from exc
    index_path = temporary / _INDEX_NAME
    parquet_path = temporary / _PARQUET_NAME
    connection: sqlite3.Connection | None = None
    try:
        connection = _create_index(index_path)
        derived_bindings: list[MembershipSourceBinding] = []
        for source_id in sorted(by_id):
            source_input = by_id[source_id]
            expected = expected_sources[source_id]
            captured_input = _capture_source_artifact(
                source_input,
                expected.binding,
                capture_root,
            )
            membership_rows, filtered_rows = _ingest_source(
                connection,
                source_input=captured_input,
                expected=expected,
            )
            _delete_captured_artifact(captured_input.path)
            derived_bindings.append(
                replace(
                    expected.binding,
                    membership_row_count=membership_rows,
                    filtered_row_count=filtered_rows,
                )
            )
        connection.commit()
        _create_lookup_indexes(connection)
        summary = _summarize_index(connection)
        _require_nonvacuous_roles(summary.role_counts)
        _require_no_cross_role_leakage(connection)
        rowset_sha256 = _write_membership_parquet(connection, parquet_path)
        if rowset_sha256 != summary.rowset_sha256:
            raise InputError("membership Parquet row digest drifted from SQLite staging")

        semantic_payload = {
            "schema_version": MEMBERSHIP_STORE_SCHEMA_VERSION,
            "artifact_id": artifact_id,
            "assembly": "GRCh38",
            "chromosome_roles": V03_CHROMOSOME_ROLES.to_dict(),
            "snapshot_lineage": lineage_binding.to_dict(),
            "sources": [
                binding.to_dict()
                for binding in sorted(derived_bindings, key=lambda item: item.source_id)
            ],
            "row_count": summary.row_count,
            "variant_count": summary.variant_count,
            "role_counts": summary.role_counts,
            "source_counts": dict(sorted(summary.source_counts.items())),
            "rowset_sha256": rowset_sha256,
        }
        content_identity = canonical_json_sha256(semantic_payload)
        _write_index_metadata(connection, semantic_payload, content_identity)
        connection.commit()
        connection.execute("VACUUM")
        connection.close()
        connection = None

        (temporary / _LINEAGE_NAME).write_bytes(lineage_bytes)
        receipt = _create_build_receipt(
            artifact_id=artifact_id,
            content_identity=content_identity,
            snapshot_lineage=lineage_binding,
            builder_git_commit=builder_git_commit,
            container_image=container_image,
            pyarrow_version=str(pa.__version__),
        )
        _write_json(temporary / _RECEIPT_NAME, receipt)

        files = tuple(
            MembershipStoreFile(
                path=path.name, sha256=sha256_file(path), size_bytes=path.stat().st_size
            )
            for path in sorted(
                (
                    index_path,
                    parquet_path,
                    temporary / _LINEAGE_NAME,
                    temporary / _RECEIPT_NAME,
                ),
                key=lambda item: item.name,
            )
        )
        manifest = MembershipStoreManifest(
            artifact_id=artifact_id,
            assembly="GRCh38",
            chromosome_roles=V03_CHROMOSOME_ROLES,
            snapshot_lineage=lineage_binding,
            sources=tuple(derived_bindings),
            row_count=summary.row_count,
            variant_count=summary.variant_count,
            role_counts=summary.role_counts,
            source_counts=summary.source_counts,
            rowset_sha256=rowset_sha256,
            files=files,
            content_identity=content_identity,
        )
        _write_json(temporary / _MANIFEST_NAME, manifest.to_dict())
        verified = verify_membership_store(temporary)
        _fsync_artifact(temporary)
        if output.exists():
            raise InputError(
                "membership store output appeared before publication",
                details={"path": str(output)},
            )
        temporary.rename(output)
        _fsync_directory(output.parent)
        return verified.manifest
    except InputError:
        if connection is not None:
            with suppress(Exception):
                connection.close()
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    except (OSError, sqlite3.DatabaseError, pa.ArrowException) as exc:
        if connection is not None:
            with suppress(Exception):
                connection.close()
        shutil.rmtree(temporary, ignore_errors=True)
        raise InputError("membership store build failed before publication") from exc
    except Exception:
        if connection is not None:
            with suppress(Exception):
                connection.close()
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    finally:
        with suppress(OSError):
            for captured in capture_root.iterdir():
                captured.chmod(0o600)
        shutil.rmtree(capture_root, ignore_errors=True)


def _fsync_artifact(root: Path) -> None:
    for path in sorted(root.iterdir(), key=lambda item: item.name):
        descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
    _fsync_directory(root)


def _fsync_directory(path: Path) -> None:
    directory_flag = getattr(os, "O_DIRECTORY", None)
    if directory_flag is None:
        # Windows has no directory descriptor that ``os.fsync`` accepts. The
        # artifact files are still flushed before the same-volume atomic move.
        return
    descriptor = os.open(path, os.O_RDONLY | directory_flag)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _ingest_source(
    connection: sqlite3.Connection,
    *,
    source_input: MembershipSourceInput,
    expected: _ExpectedSource,
) -> tuple[int, int]:
    rows = _iter_source_memberships(source_input, expected)
    membership_count = 0
    for row in rows:
        try:
            connection.execute(
                "INSERT INTO memberships ("
                "schema_version, variant_key, variant_digest, chrom, chrom_rank, pos, "
                "start_bp, end_bp, ref, alt, role, role_rank, reason_mask, source, source_row_id"
                ") VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                _sqlite_row(row),
            )
        except sqlite3.IntegrityError as exc:
            raise InputError(
                "duplicate membership identity in source artifact",
                details={
                    "variant_key": row.variant.key,
                    "source": row.source,
                    "source_row_id": row.source_row_id,
                },
            ) from exc
        membership_count += 1
    filtered = rows.filtered_count
    if membership_count + filtered != expected.binding.artifact_row_count:
        raise InputError(
            "membership source scan row count mismatch",
            details={
                "source_id": source_input.source_id,
                "declared": expected.binding.artifact_row_count,
                "membership_rows": membership_count,
                "filtered_rows": filtered,
            },
        )
    return membership_count, filtered


def _iter_source_memberships(
    source_input: MembershipSourceInput,
    expected: _ExpectedSource,
) -> _CountedIterator:
    pa, pq = _require_pyarrow()
    parquet = pq.ParquetFile(source_input.path)
    expected_schema = _gnomad_schema(pa) if source_input.kind == "gnomad" else _clinvar_schema(pa)
    if not parquet.schema_arrow.equals(expected_schema):
        raise InputError(
            "membership source Parquet schema mismatch",
            details={
                "source_id": source_input.source_id,
                "expected": str(expected_schema),
                "observed": str(parquet.schema_arrow),
            },
        )
    filtered_count = 0

    def _rows() -> Iterator[MembershipRow]:
        nonlocal filtered_count
        for batch in parquet.iter_batches(batch_size=_PARQUET_BATCH_ROWS):
            for raw in batch.to_pylist():
                if source_input.kind == "gnomad":
                    yield _gnomad_membership(raw, source_input, expected)
                    continue
                row = _clinvar_membership(raw, source_input)
                if row is None:
                    filtered_count += 1
                else:
                    yield row

    # ``filtered_count`` is finalized only after the iterator is exhausted. A
    # small wrapper exposes it to the caller without buffering rows.
    return _CountedIterator(_rows(), lambda: filtered_count)


class _CountedIterator(Iterator[MembershipRow]):
    def __init__(self, rows: Iterator[MembershipRow], filtered: Callable[[], int]) -> None:
        self._rows = rows
        self._filtered = filtered

    def __iter__(self) -> _CountedIterator:
        return self

    def __next__(self) -> MembershipRow:
        return next(self._rows)

    @property
    def filtered_count(self) -> int:
        return int(self._filtered())


def _gnomad_membership(
    raw: Mapping[str, object],
    source_input: MembershipSourceInput,
    expected: _ExpectedSource,
) -> MembershipRow:
    chromosome = _required_row_text(raw, "chrom", source_input.source_id)
    if chromosome != source_input.chromosome or chromosome != expected.chromosome:
        raise InputError(
            "gnomAD membership shard chromosome mismatch",
            details={"source_id": source_input.source_id, "observed": chromosome},
        )
    if raw.get("schema_version") != GNOMAD_SCHEMA_VERSION:
        raise InputError("gnomAD membership row schema version mismatch")
    if raw.get("filter") != "PASS":
        raise InputError("gnomAD membership row must be PASS")
    variant = _variant_from_source_row(raw, source_input.source_id)
    role = V03_CHROMOSOME_ROLES.role_for(variant.chrom)
    raw_identity = f"{chromosome}:{raw['pos']}:{raw['ref']}:{raw['alt']}"
    return MembershipRow(
        variant=variant,
        role=role,
        reason_mask=_GNOMAD_REASON_MASK,
        source=source_input.source_id,
        source_row_id=raw_identity,
    )


def _clinvar_membership(
    raw: Mapping[str, object], source_input: MembershipSourceInput
) -> MembershipRow | None:
    if raw.get("schema_version") != CLINVAR_SCHEMA_VERSION:
        raise InputError("ClinVar membership row schema version mismatch")
    significance = _required_row_text(raw, "clinical_significance", source_input.source_id)
    if significance not in _CLINVAR_CLASSES:
        raise InputError("ClinVar membership row has an unknown normalized class")
    if significance not in _CLINVAR_HOLDOUT_CLASSES:
        return None
    raw_chromosome = _required_row_text(raw, "chrom", source_input.source_id)
    try:
        chromosome = canonicalize_chromosome(raw_chromosome)
    except InputError:
        # The pinned ClinVar archive may contain alternate/non-primary
        # contigs. v0.3 membership is explicitly primary-autosome-only, so
        # these rows are counted as filtered rather than accepted implicitly.
        return None
    if chromosome not in _AUTOSOMES:
        return None
    variant = _variant_from_source_row(raw, source_input.source_id)
    clinvar_id = raw.get("clinvar_id")
    if isinstance(clinvar_id, bool) or not isinstance(clinvar_id, int) or clinvar_id < 1:
        raise InputError("ClinVar membership row clinvar_id must be positive")
    raw_identity = f"{clinvar_id}:{raw['chrom']}:{raw['pos']}:{raw['ref']}:{raw['alt']}"
    return MembershipRow(
        variant=variant,
        role=V03_CHROMOSOME_ROLES.role_for(variant.chrom),
        reason_mask=_CLINVAR_REASON_MASK,
        source=source_input.source_id,
        source_row_id=raw_identity,
    )


def _variant_from_source_row(raw: Mapping[str, object], source_id: str) -> CanonicalVariant:
    pos = raw.get("pos")
    if isinstance(pos, bool) or not isinstance(pos, int):
        raise InputError(
            "membership source position must be an integer", details={"source_id": source_id}
        )
    return CanonicalVariant(
        assembly="GRCh38",
        chrom=_required_row_text(raw, "chrom", source_id),
        pos=pos,
        ref=_required_row_text(raw, "ref", source_id),
        alt=_required_row_text(raw, "alt", source_id),
    )


def _sqlite_row(row: MembershipRow) -> tuple[object, ...]:
    variant = row.variant
    return (
        MEMBERSHIP_STORE_SCHEMA_VERSION,
        variant.key,
        variant.digest,
        variant.chrom,
        _CHROMOSOME_RANK[variant.chrom],
        variant.pos,
        variant.pos - 1,
        variant.pos - 1 + len(variant.ref),
        variant.ref,
        variant.alt,
        row.role,
        REQUIRED_MEMBERSHIP_ROLES.index(row.role),
        row.reason_mask,
        row.source,
        row.source_row_id,
    )


def _require_nonvacuous_roles(role_counts: Mapping[str, int]) -> None:
    empty = [role for role in REQUIRED_MEMBERSHIP_ROLES if role_counts.get(role, 0) < 1]
    if empty:
        raise InputError(
            "membership store must contain every required role", details={"empty": empty}
        )


def _required_row_text(raw: Mapping[str, object], name: str, source_id: str) -> str:
    value = raw.get(name)
    if not isinstance(value, str) or not value:
        raise InputError(
            "membership source row field must be non-empty text",
            details={"source_id": source_id, "field": name},
        )
    return value
