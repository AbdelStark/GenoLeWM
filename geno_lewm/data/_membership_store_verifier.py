# SPDX-License-Identifier: Apache-2.0
"""Independent bounded-memory membership-store verifier."""

from __future__ import annotations

import hashlib
import sqlite3
import stat
import tempfile
from dataclasses import replace
from pathlib import Path

from geno_lewm.data._membership_store_contract import (
    _ARTIFACT_FILE_NAMES,
    _CHROMOSOME_RANK,
    _INDEX_NAME,
    _LINEAGE_NAME,
    _MANIFEST_NAME,
    _PARQUET_BATCH_ROWS,
    _PARQUET_NAME,
    _RECEIPT_NAME,
    MembershipStoreManifest,
    MembershipStoreVerification,
    _read_manifest,
    _require_positive_int,
)
from geno_lewm.data._membership_store_lineage import _load_snapshot_lineage
from geno_lewm.data._membership_store_receipt import _verify_build_receipt
from geno_lewm.data._membership_store_snapshot import (
    _capture_membership_store,
    _CapturedMembershipStore,
)
from geno_lewm.data._membership_store_storage import (
    _ORDER_BY,
    _SELECT_INDEX_ROWS,
    _clinvar_class_role_counts,
    _empty_role_crosstab,
    _membership_schema,
    _require_no_clinvar_label_conflicts,
    _require_no_cross_role_leakage,
    _require_pyarrow,
    _ScanSummary,
    _semantic_order_key,
    _summary_dict,
    _update_rowset_digest,
    _validate_semantic_row,
    _verify_index_metadata,
    _verify_index_schema,
    _verify_interval_index,
)
from geno_lewm.data.membership import REQUIRED_MEMBERSHIP_ROLES
from geno_lewm.errors import InputError


def verify_membership_store(store_dir: Path) -> MembershipStoreVerification:
    """Independently verify manifest, files, Parquet rows, and SQLite rows."""
    with _capture_membership_store(Path(store_dir)) as captured:
        return _verify_captured_membership_store(captured)


def _verify_captured_membership_store(
    captured: _CapturedMembershipStore,
) -> MembershipStoreVerification:
    root = captured.root
    manifest = _read_manifest(root / _MANIFEST_NAME)
    for binding in manifest.files:
        identity = captured.files[binding.path]
        observed = {"sha256": identity.sha256, "size_bytes": identity.size_bytes}
        expected = {"sha256": binding.sha256, "size_bytes": binding.size_bytes}
        if observed != expected:
            raise InputError(
                "membership store file identity mismatch",
                details={"path": binding.path, "declared": expected, "observed": observed},
            )
    bundled_lineage, expected_sources, _raw = _load_snapshot_lineage(root / _LINEAGE_NAME)
    if bundled_lineage != manifest.snapshot_lineage:
        raise InputError("bundled snapshot lineage does not match manifest")
    if set(expected_sources) != {source.source_id for source in manifest.sources}:
        raise InputError("bundled snapshot lineage source set does not match manifest")
    for source in manifest.sources:
        expected_binding = replace(
            expected_sources[source.source_id].binding,
            membership_row_count=source.membership_row_count,
            filtered_row_count=source.filtered_row_count,
        )
        if source != expected_binding:
            raise InputError("bundled snapshot lineage source binding does not match manifest")
    _verify_build_receipt(root / _RECEIPT_NAME, manifest)
    parquet_summary = _scan_parquet(root / _PARQUET_NAME, manifest)
    sqlite_summary = _scan_sqlite(root / _INDEX_NAME, manifest)
    if parquet_summary != sqlite_summary:
        raise InputError(
            "membership Parquet and SQLite semantic scans differ",
            details={
                "parquet": _summary_dict(parquet_summary),
                "sqlite": _summary_dict(sqlite_summary),
            },
        )
    _require_summary_matches_manifest(parquet_summary, manifest)
    return MembershipStoreVerification(manifest=manifest)


def _verify_exact_layout(root: Path) -> None:
    try:
        invalid_root = root.is_symlink() or not root.is_dir()
        paths = tuple(root.iterdir()) if not invalid_root else ()
    except OSError as exc:
        raise InputError("membership store layout cannot be read") from exc
    if invalid_root:
        raise InputError("membership store root must be a real directory")
    observed: set[str] = set()
    for path in paths:
        try:
            mode = path.lstat().st_mode
        except OSError as exc:
            raise InputError("membership store layout cannot be read") from exc
        if stat.S_ISLNK(mode):
            raise InputError(
                "membership store must not contain symlinks",
                details={"path": path.name},
            )
        if not stat.S_ISREG(mode):
            raise InputError(
                "membership store must contain only exact top-level files",
                details={"path": path.name},
            )
        observed.add(path.name)
    if observed != _ARTIFACT_FILE_NAMES:
        raise InputError(
            "membership store files do not match the exact layout",
            details={
                "missing": sorted(_ARTIFACT_FILE_NAMES - observed),
                "unexpected": sorted(observed - _ARTIFACT_FILE_NAMES),
            },
        )


def _scan_parquet(path: Path, manifest: MembershipStoreManifest) -> _ScanSummary:
    pa, _pq = _require_pyarrow()
    try:
        return _scan_parquet_unchecked(path, manifest)
    except InputError:
        raise
    except (OSError, sqlite3.DatabaseError, pa.ArrowException) as exc:
        raise InputError("membership Parquet scan failed") from exc


def _scan_parquet_unchecked(path: Path, manifest: MembershipStoreManifest) -> _ScanSummary:
    pa, pq = _require_pyarrow()
    parquet = pq.ParquetFile(path)
    expected_schema = _membership_schema(pa)
    if not parquet.schema_arrow.equals(expected_schema):
        raise InputError("membership Parquet schema does not match closed contract")
    if parquet.metadata.num_rows != manifest.row_count:
        raise InputError("membership Parquet metadata row count mismatch")
    role_counts = dict.fromkeys(REQUIRED_MEMBERSHIP_ROLES, 0)
    source_counts = dict.fromkeys(manifest.source_counts, 0)
    source_role_counts = _empty_role_crosstab(tuple(manifest.source_counts))
    source_kind_role_counts = _empty_role_crosstab(("gnomad", "clinvar"))
    source_bindings = {binding.source_id: binding for binding in manifest.sources}
    digest = hashlib.sha256()
    previous: tuple[object, ...] | None = None
    row_count = 0
    with tempfile.TemporaryDirectory(prefix="geno-lewm-membership-verify-") as temporary:
        variants = sqlite3.connect(Path(temporary) / "variants.sqlite")
        try:
            variants.execute("CREATE TABLE variants (key TEXT PRIMARY KEY) WITHOUT ROWID")
            variants.execute(
                "CREATE TABLE identities (variant_key TEXT, source TEXT, source_row_id TEXT, "
                "PRIMARY KEY (variant_key, source, source_row_id)) WITHOUT ROWID"
            )
            variants.execute(
                "CREATE TABLE labels (variant_key TEXT, role TEXT, clinical_significance TEXT, "
                "source_row_id TEXT)"
            )
            for batch in parquet.iter_batches(batch_size=_PARQUET_BATCH_ROWS):
                for raw in batch.to_pylist():
                    payload = _validate_semantic_row(raw, manifest, source_bindings)
                    order_key = _semantic_order_key(payload)
                    if previous is not None and order_key <= previous:
                        raise InputError(
                            "membership Parquet rows are not in strict canonical order"
                        )
                    previous = order_key
                    try:
                        variants.execute(
                            "INSERT OR IGNORE INTO variants VALUES (?)", (payload["variant_key"],)
                        )
                        variants.execute(
                            "INSERT INTO identities VALUES (?, ?, ?)",
                            (payload["variant_key"], payload["source"], payload["source_row_id"]),
                        )
                    except sqlite3.IntegrityError as exc:
                        raise InputError(
                            "membership Parquet contains duplicate membership identity"
                        ) from exc
                    role_counts[str(payload["role"])] += 1
                    source_counts[str(payload["source"])] += 1
                    source_role_counts[str(payload["source"])][str(payload["role"])] += 1
                    source_kind = source_bindings[str(payload["source"])].kind
                    source_kind_role_counts[source_kind][str(payload["role"])] += 1
                    if payload["clinical_significance"] is not None:
                        variants.execute(
                            "INSERT INTO labels VALUES (?, ?, ?, ?)",
                            (
                                payload["variant_key"],
                                payload["role"],
                                payload["clinical_significance"],
                                payload["source_row_id"],
                            ),
                        )
                    _update_rowset_digest(digest, payload)
                    row_count += 1
            variant_count = int(variants.execute("SELECT COUNT(*) FROM variants").fetchone()[0])
            conflict = variants.execute(
                "SELECT variant_key FROM labels GROUP BY variant_key "
                "HAVING MIN(clinical_significance IN ('LP', 'P')) "
                "!= MAX(clinical_significance IN ('LP', 'P')) LIMIT 1"
            ).fetchone()
            if conflict is not None:
                raise InputError("ClinVar memberships contain conflicting binary targets")
            clinvar_class_role_counts = _clinvar_class_role_counts(variants, table="labels")
        finally:
            variants.close()
    return _ScanSummary(
        row_count=row_count,
        variant_count=variant_count,
        role_counts=role_counts,
        source_counts=source_counts,
        source_role_counts=source_role_counts,
        source_kind_role_counts=source_kind_role_counts,
        clinvar_class_role_counts=clinvar_class_role_counts,
        rowset_sha256="sha256:" + digest.hexdigest(),
    )


def _scan_sqlite(path: Path, manifest: MembershipStoreManifest) -> _ScanSummary:
    connection: sqlite3.Connection | None = None
    try:
        connection = sqlite3.connect(f"{path.resolve().as_uri()}?mode=ro&immutable=1", uri=True)
        integrity = connection.execute("PRAGMA integrity_check").fetchone()
        if integrity != ("ok",):
            raise InputError("membership SQLite integrity check failed")
        _verify_index_schema(connection)
        _verify_index_metadata(connection, manifest)
        summary = _scan_index_rows(connection, manifest)
        _require_no_cross_role_leakage(connection)
        _require_no_clinvar_label_conflicts(connection)
        _verify_interval_index(connection)
        return summary
    except sqlite3.DatabaseError as exc:
        raise InputError("membership SQLite scan failed") from exc
    finally:
        if connection is not None:
            connection.close()


def _scan_index_rows(
    connection: sqlite3.Connection, manifest: MembershipStoreManifest
) -> _ScanSummary:
    source_bindings = {binding.source_id: binding for binding in manifest.sources}
    role_counts = dict.fromkeys(REQUIRED_MEMBERSHIP_ROLES, 0)
    source_counts = dict.fromkeys(manifest.source_counts, 0)
    source_role_counts = _empty_role_crosstab(tuple(manifest.source_counts))
    source_kind_role_counts = _empty_role_crosstab(("gnomad", "clinvar"))
    digest = hashlib.sha256()
    previous_order: tuple[object, ...] | None = None
    previous_identity: tuple[str, str, str] | None = None
    previous_variant: str | None = None
    row_count = 0
    variant_count = 0
    cursor = connection.execute(_SELECT_INDEX_ROWS + " " + _ORDER_BY)
    while rows := cursor.fetchmany(_PARQUET_BATCH_ROWS):
        for raw in rows:
            payload = _validate_semantic_row(
                {
                    "schema_version": raw[0],
                    "variant_key": raw[1],
                    "variant_digest": raw[2],
                    "chrom": raw[3],
                    "pos": raw[5],
                    "ref": raw[8],
                    "alt": raw[9],
                    "role": raw[10],
                    "reason_mask": raw[12],
                    "source": raw[13],
                    "source_row_id": raw[14],
                    "clinical_significance": raw[15],
                },
                manifest,
                source_bindings,
            )
            position = _require_positive_int(payload["pos"], "membership lookup position")
            expected_derived = (
                _CHROMOSOME_RANK[str(payload["chrom"])],
                position - 1,
                position - 1 + len(str(payload["ref"])),
                REQUIRED_MEMBERSHIP_ROLES.index(str(payload["role"])),
            )
            observed_derived = (raw[4], raw[6], raw[7], raw[11])
            if observed_derived != expected_derived:
                raise InputError("membership lookup derived columns are inconsistent")
            order_key = _semantic_order_key(payload)
            if previous_order is not None and order_key <= previous_order:
                raise InputError("membership lookup rows are not in strict canonical order")
            previous_order = order_key
            identity = (
                str(payload["variant_key"]),
                str(payload["source"]),
                str(payload["source_row_id"]),
            )
            if identity == previous_identity:
                raise InputError("membership lookup contains duplicate membership identity")
            previous_identity = identity
            variant_key = str(payload["variant_key"])
            if variant_key != previous_variant:
                variant_count += 1
                previous_variant = variant_key
            role_counts[str(payload["role"])] += 1
            source_counts[str(payload["source"])] += 1
            source_role_counts[str(payload["source"])][str(payload["role"])] += 1
            source_kind = source_bindings[str(payload["source"])].kind
            source_kind_role_counts[source_kind][str(payload["role"])] += 1
            _update_rowset_digest(digest, payload)
            row_count += 1
    return _ScanSummary(
        row_count=row_count,
        variant_count=variant_count,
        role_counts=role_counts,
        source_counts=source_counts,
        source_role_counts=source_role_counts,
        source_kind_role_counts=source_kind_role_counts,
        clinvar_class_role_counts=_clinvar_class_role_counts(connection),
        rowset_sha256="sha256:" + digest.hexdigest(),
    )


def _require_summary_matches_manifest(
    summary: _ScanSummary, manifest: MembershipStoreManifest
) -> None:
    declared = {
        "row_count": manifest.row_count,
        "variant_count": manifest.variant_count,
        "role_counts": manifest.role_counts,
        "source_counts": manifest.source_counts,
        "source_role_counts": manifest.source_role_counts,
        "source_kind_role_counts": manifest.source_kind_role_counts,
        "clinvar_class_role_counts": manifest.clinvar_class_role_counts,
        "rowset_sha256": manifest.rowset_sha256,
    }
    observed = _summary_dict(summary)
    if observed != declared:
        raise InputError(
            "membership semantic scan does not match manifest",
            details={"declared": declared, "observed": observed},
        )
