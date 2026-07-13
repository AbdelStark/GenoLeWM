# SPDX-License-Identifier: Apache-2.0
"""Canonical SQLite and Parquet representation internals."""

from __future__ import annotations

import hashlib
import importlib
import sqlite3
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final

from geno_lewm.data._membership_store_contract import (
    _CHROMOSOME_RANK,
    _CLINVAR_REASON_MASK,
    _GNOMAD_REASON_MASK,
    _PARQUET_BATCH_ROWS,
    MEMBERSHIP_STORE_SCHEMA_VERSION,
    MembershipSourceBinding,
    MembershipStoreManifest,
    _require_exact_keys,
    _require_positive_int,
    _require_text,
)
from geno_lewm.data.membership import REQUIRED_MEMBERSHIP_ROLES, MembershipRow
from geno_lewm.data.variant_identity import CanonicalVariant
from geno_lewm.errors import InputError, RuntimeSetupError
from geno_lewm.provenance.hashing import canonical_json_bytes

_SELECT_ROWS: Final = (
    "SELECT schema_version, variant_key, variant_digest, chrom, pos, ref, alt, role, "
    "reason_mask, source, source_row_id FROM memberships"
)
_SELECT_INDEX_ROWS: Final = (
    "SELECT schema_version, variant_key, variant_digest, chrom, chrom_rank, pos, start_bp, "
    "end_bp, ref, alt, role, role_rank, reason_mask, source, source_row_id FROM memberships"
)
_ORDER_BY: Final = (
    "ORDER BY chrom_rank, pos, ref, alt, role_rank, source, source_row_id, reason_mask"
)


@dataclass(frozen=True, slots=True)
class _ScanSummary:
    row_count: int
    variant_count: int
    role_counts: dict[str, int]
    source_counts: dict[str, int]
    rowset_sha256: str


def _create_index(path: Path) -> sqlite3.Connection:
    connection = sqlite3.connect(path)
    connection.execute("PRAGMA page_size=4096")
    connection.execute("PRAGMA journal_mode=DELETE")
    connection.execute("PRAGMA synchronous=FULL")
    connection.execute("PRAGMA temp_store=FILE")
    connection.executescript(
        """
        CREATE TABLE memberships (
            schema_version TEXT NOT NULL,
            variant_key TEXT NOT NULL,
            variant_digest TEXT NOT NULL,
            chrom TEXT NOT NULL,
            chrom_rank INTEGER NOT NULL,
            pos INTEGER NOT NULL,
            start_bp INTEGER NOT NULL,
            end_bp INTEGER NOT NULL,
            ref TEXT NOT NULL,
            alt TEXT NOT NULL,
            role TEXT NOT NULL,
            role_rank INTEGER NOT NULL,
            reason_mask INTEGER NOT NULL,
            source TEXT NOT NULL,
            source_row_id TEXT NOT NULL,
            PRIMARY KEY (variant_key, source, source_row_id)
        ) WITHOUT ROWID;
        """
    )
    return connection


def _create_lookup_indexes(connection: sqlite3.Connection) -> None:
    connection.executescript(
        """
        CREATE INDEX memberships_interval
            ON memberships (chrom_rank, start_bp, end_bp, role);
        CREATE INDEX memberships_role_order
            ON memberships (
                role, chrom_rank, pos, ref, alt, role_rank, source, source_row_id, reason_mask
            );
        """
    )
    connection.commit()


def _summarize_index(connection: sqlite3.Connection) -> _ScanSummary:
    row_count = int(connection.execute("SELECT COUNT(*) FROM memberships").fetchone()[0])
    variant_count = int(
        connection.execute("SELECT COUNT(DISTINCT variant_key) FROM memberships").fetchone()[0]
    )
    role_counts = dict.fromkeys(REQUIRED_MEMBERSHIP_ROLES, 0)
    for role, count in connection.execute("SELECT role, COUNT(*) FROM memberships GROUP BY role"):
        if role not in role_counts:
            raise InputError("membership index contains an unknown role")
        role_counts[str(role)] = int(count)
    source_counts = {
        str(source): int(count)
        for source, count in connection.execute(
            "SELECT source, COUNT(*) FROM memberships GROUP BY source ORDER BY source"
        )
    }
    digest = _digest_sql_rows(connection)
    return _ScanSummary(row_count, variant_count, role_counts, source_counts, digest)


def _require_no_cross_role_leakage(connection: sqlite3.Connection) -> None:
    row = connection.execute(
        "SELECT variant_key FROM memberships GROUP BY variant_key "
        "HAVING COUNT(DISTINCT role) > 1 LIMIT 1"
    ).fetchone()
    if row is not None:
        raise InputError(
            "membership variant leaks across split roles",
            details={"variant_key": row[0]},
        )


def _write_membership_parquet(connection: sqlite3.Connection, path: Path) -> str:
    pa, pq = _require_pyarrow()
    schema = _membership_schema(pa)
    writer = pq.ParquetWriter(path, schema, compression="zstd", use_dictionary=False)
    digest = hashlib.sha256()
    cursor = connection.execute(_SELECT_ROWS + " " + _ORDER_BY)
    try:
        while rows := cursor.fetchmany(_PARQUET_BATCH_ROWS):
            payloads = [_semantic_row_from_sql(row) for row in rows]
            for payload in payloads:
                _update_rowset_digest(digest, payload)
            writer.write_table(pa.Table.from_pylist(payloads, schema=schema))
    finally:
        writer.close()
    return "sha256:" + digest.hexdigest()


def _digest_sql_rows(connection: sqlite3.Connection) -> str:
    digest = hashlib.sha256()
    cursor = connection.execute(_SELECT_ROWS + " " + _ORDER_BY)
    while rows := cursor.fetchmany(_PARQUET_BATCH_ROWS):
        for row in rows:
            _update_rowset_digest(digest, _semantic_row_from_sql(row))
    return "sha256:" + digest.hexdigest()


def _update_rowset_digest(digest: Any, payload: Mapping[str, object]) -> None:
    encoded = canonical_json_bytes(payload)
    digest.update(len(encoded).to_bytes(8, "big"))
    digest.update(encoded)


def _semantic_row_from_sql(row: Sequence[object]) -> dict[str, object]:
    return {
        "schema_version": row[0],
        "variant_key": row[1],
        "variant_digest": row[2],
        "chrom": row[3],
        "pos": row[4],
        "ref": row[5],
        "alt": row[6],
        "role": row[7],
        "reason_mask": row[8],
        "source": row[9],
        "source_row_id": row[10],
    }


def _write_index_metadata(
    connection: sqlite3.Connection,
    semantic_payload: Mapping[str, object],
    content_identity: str,
) -> None:
    connection.execute(
        "CREATE TABLE metadata (key TEXT PRIMARY KEY, value TEXT NOT NULL) WITHOUT ROWID"
    )
    values = {
        "schema_version": MEMBERSHIP_STORE_SCHEMA_VERSION,
        "content_identity": content_identity,
        "rowset_sha256": str(semantic_payload["rowset_sha256"]),
        "row_count": str(semantic_payload["row_count"]),
        "variant_count": str(semantic_payload["variant_count"]),
    }
    connection.executemany("INSERT INTO metadata VALUES (?, ?)", sorted(values.items()))


def _verify_index_metadata(
    connection: sqlite3.Connection, manifest: MembershipStoreManifest
) -> None:
    try:
        values = dict(connection.execute("SELECT key, value FROM metadata"))
    except sqlite3.DatabaseError as exc:
        raise InputError("membership lookup metadata cannot be read") from exc
    expected = {
        "schema_version": manifest.schema_version,
        "content_identity": manifest.content_identity,
        "rowset_sha256": manifest.rowset_sha256,
        "row_count": str(manifest.row_count),
        "variant_count": str(manifest.variant_count),
    }
    if values != expected:
        raise InputError(
            "membership lookup metadata does not match manifest",
            details={"declared": expected, "observed": values},
        )


def _verify_index_schema(connection: sqlite3.Connection) -> None:
    expected_tables = {
        "memberships": (
            ("schema_version", "TEXT", 1, 0),
            ("variant_key", "TEXT", 1, 1),
            ("variant_digest", "TEXT", 1, 0),
            ("chrom", "TEXT", 1, 0),
            ("chrom_rank", "INTEGER", 1, 0),
            ("pos", "INTEGER", 1, 0),
            ("start_bp", "INTEGER", 1, 0),
            ("end_bp", "INTEGER", 1, 0),
            ("ref", "TEXT", 1, 0),
            ("alt", "TEXT", 1, 0),
            ("role", "TEXT", 1, 0),
            ("role_rank", "INTEGER", 1, 0),
            ("reason_mask", "INTEGER", 1, 0),
            ("source", "TEXT", 1, 2),
            ("source_row_id", "TEXT", 1, 3),
        ),
        "metadata": (("key", "TEXT", 1, 1), ("value", "TEXT", 1, 0)),
    }
    observed_tables = {
        str(row[0])
        for row in connection.execute(
            "SELECT name FROM sqlite_master WHERE type = 'table' AND name NOT LIKE 'sqlite_%'"
        )
    }
    if observed_tables != set(expected_tables):
        raise InputError("membership lookup table set does not match the closed contract")
    for table, expected_table in expected_tables.items():
        observed_table = tuple(
            (str(row[1]), str(row[2]), int(row[3]), int(row[5]))
            for row in connection.execute(f"PRAGMA table_info({table})")
        )
        if observed_table != expected_table:
            raise InputError(
                "membership lookup table schema does not match the closed contract",
                details={"table": table},
            )
    expected_indexes = {
        "memberships_interval": ("chrom_rank", "start_bp", "end_bp", "role"),
        "memberships_role_order": (
            "role",
            "chrom_rank",
            "pos",
            "ref",
            "alt",
            "role_rank",
            "source",
            "source_row_id",
            "reason_mask",
        ),
        "sqlite_autoindex_memberships_1": ("variant_key", "source", "source_row_id"),
    }
    index_rows = tuple(connection.execute("PRAGMA index_list(memberships)"))
    if {str(row[1]) for row in index_rows} != set(expected_indexes):
        raise InputError("membership lookup index set does not match the closed contract")
    for name, expected_columns in expected_indexes.items():
        observed_columns = tuple(
            str(row[2]) for row in connection.execute(f"PRAGMA index_info({name})")
        )
        if observed_columns != expected_columns:
            raise InputError(
                "membership lookup index columns do not match the closed contract",
                details={"index": name},
            )


def _validate_semantic_row(
    raw: Mapping[str, object],
    manifest: MembershipStoreManifest,
    source_bindings: Mapping[str, MembershipSourceBinding],
) -> dict[str, object]:
    expected = {
        "schema_version",
        "variant_key",
        "variant_digest",
        "chrom",
        "pos",
        "ref",
        "alt",
        "role",
        "reason_mask",
        "source",
        "source_row_id",
    }
    _require_exact_keys(raw, expected, "membership Parquet row")
    if raw.get("schema_version") != MEMBERSHIP_STORE_SCHEMA_VERSION:
        raise InputError("membership Parquet row schema version mismatch")
    variant_key = _require_text(raw.get("variant_key"), "membership row variant_key")
    variant = CanonicalVariant.from_key(variant_key)
    if raw.get("variant_digest") != variant.digest:
        raise InputError("membership row variant digest mismatch")
    observed_variant = {
        "chrom": raw.get("chrom"),
        "pos": raw.get("pos"),
        "ref": raw.get("ref"),
        "alt": raw.get("alt"),
    }
    expected_variant = {
        "chrom": variant.chrom,
        "pos": variant.pos,
        "ref": variant.ref,
        "alt": variant.alt,
    }
    if observed_variant != expected_variant:
        raise InputError("membership row variant columns drift from canonical key")
    role = _require_text(raw.get("role"), "membership row role")
    if role != manifest.chromosome_roles.role_for(variant.chrom):
        raise InputError("membership row role drifts from chromosome split")
    reason_mask = _require_positive_int(raw.get("reason_mask"), "membership row reason_mask")
    source = _require_text(raw.get("source"), "membership row source")
    if source not in manifest.source_counts:
        raise InputError("membership row references an unbound source")
    source_binding = source_bindings[source]
    expected_reason = (
        _GNOMAD_REASON_MASK if source_binding.kind == "gnomad" else _CLINVAR_REASON_MASK
    )
    if reason_mask != expected_reason:
        raise InputError("membership row reason mask drifts from its bound source kind")
    source_row_id = _require_text(raw.get("source_row_id"), "membership row source_row_id")
    return {
        "schema_version": MEMBERSHIP_STORE_SCHEMA_VERSION,
        "variant_key": variant.key,
        "variant_digest": variant.digest,
        "chrom": variant.chrom,
        "pos": variant.pos,
        "ref": variant.ref,
        "alt": variant.alt,
        "role": role,
        "reason_mask": reason_mask,
        "source": source,
        "source_row_id": source_row_id,
    }


def _semantic_order_key(payload: Mapping[str, object]) -> tuple[object, ...]:
    return (
        _CHROMOSOME_RANK[str(payload["chrom"])],
        payload["pos"],
        payload["ref"],
        payload["alt"],
        REQUIRED_MEMBERSHIP_ROLES.index(str(payload["role"])),
        payload["source"],
        payload["source_row_id"],
        payload["reason_mask"],
    )


def _membership_row_from_sql(raw: Sequence[object]) -> MembershipRow:
    payload = _semantic_row_from_sql(raw)
    return MembershipRow(
        variant=CanonicalVariant.from_key(str(payload["variant_key"])),
        role=str(payload["role"]),
        reason_mask=_require_positive_int(payload["reason_mask"], "membership reason_mask"),
        source=str(payload["source"]),
        source_row_id=str(payload["source_row_id"]),
    )


def _membership_schema(pa: Any) -> Any:
    return pa.schema(
        [
            ("schema_version", pa.string(), False),
            ("variant_key", pa.string(), False),
            ("variant_digest", pa.string(), False),
            ("chrom", pa.string(), False),
            ("pos", pa.int64(), False),
            ("ref", pa.string(), False),
            ("alt", pa.string(), False),
            ("role", pa.string(), False),
            ("reason_mask", pa.int64(), False),
            ("source", pa.string(), False),
            ("source_row_id", pa.string(), False),
        ]
    )


def _gnomad_schema(pa: Any) -> Any:
    return pa.schema(
        [
            ("chrom", pa.string()),
            ("pos", pa.int64()),
            ("ref", pa.string()),
            ("alt", pa.string()),
            ("af_global", pa.float32()),
            ("af_afr", pa.float32()),
            ("af_ami", pa.float32()),
            ("af_amr", pa.float32()),
            ("af_asj", pa.float32()),
            ("af_eas", pa.float32()),
            ("af_fin", pa.float32()),
            ("af_mid", pa.float32()),
            ("af_nfe", pa.float32()),
            ("af_oth", pa.float32()),
            ("af_remaining", pa.float32()),
            ("af_sas", pa.float32()),
            ("filter", pa.string()),
            ("schema_version", pa.string()),
        ]
    )


def _clinvar_schema(pa: Any) -> Any:
    return pa.schema(
        [
            ("chrom", pa.string()),
            ("pos", pa.int64()),
            ("ref", pa.string()),
            ("alt", pa.string()),
            ("clinical_significance", pa.string()),
            ("review_status", pa.string()),
            ("gene_symbol", pa.string()),
            ("clinvar_id", pa.int64()),
            ("schema_version", pa.string()),
        ]
    )


def _require_pyarrow() -> tuple[Any, Any]:
    try:
        pa = importlib.import_module("pyarrow")
        pq = importlib.import_module("pyarrow.parquet")
    except ImportError as exc:
        raise RuntimeSetupError(
            "membership Parquet build and verification requires pyarrow",
            remediation="install geno-lewm[dev], geno-lewm[train], or pyarrow",
        ) from exc
    return pa, pq


def _summary_dict(summary: _ScanSummary) -> dict[str, object]:
    return {
        "row_count": summary.row_count,
        "variant_count": summary.variant_count,
        "role_counts": summary.role_counts,
        "source_counts": summary.source_counts,
        "rowset_sha256": summary.rowset_sha256,
    }
