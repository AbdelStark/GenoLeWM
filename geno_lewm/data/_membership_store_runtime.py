# SPDX-License-Identifier: Apache-2.0
"""Read-only indexed lookup and tuple-builder holdout adapter."""

from __future__ import annotations

import sqlite3
from collections.abc import Iterator, Mapping, Sequence
from pathlib import Path
from types import TracebackType

from geno_lewm.action import RelEdit
from geno_lewm.data._membership_store_contract import (
    _CHROMOSOME_RANK,
    _INDEX_NAME,
    _MANIFEST_NAME,
    MEMBERSHIP_STORE_SCHEMA_VERSION,
    _read_manifest,
    _require_nonnegative_int,
    _require_positive_int,
)
from geno_lewm.data._membership_store_storage import (
    _ORDER_BY,
    _SELECT_ROWS,
    _membership_row_from_sql,
    _verify_index_metadata,
    _verify_index_schema,
)
from geno_lewm.data._membership_store_verifier import verify_membership_store
from geno_lewm.data.builder import HoldoutPolicy, WindowContext
from geno_lewm.data.membership import REQUIRED_MEMBERSHIP_ROLES, MembershipRow
from geno_lewm.data.variant_identity import CanonicalVariant, canonicalize_chromosome
from geno_lewm.errors import InputError
from geno_lewm.provenance.hashing import canonical_json_sha256


class MembershipStore:
    """Read-only, indexed membership lookup without a PyArrow dependency."""

    def __init__(self, store_dir: Path, *, verify: bool = True) -> None:
        root = Path(store_dir)
        manifest = (
            verify_membership_store(root).manifest
            if verify
            else _read_manifest(root / _MANIFEST_NAME)
        )
        path = (root / _INDEX_NAME).resolve()
        if not path.is_file():
            raise InputError("membership lookup index is missing", details={"path": str(path)})
        connection = sqlite3.connect(f"{path.as_uri()}?mode=ro&immutable=1", uri=True)
        connection.row_factory = sqlite3.Row
        try:
            _verify_index_schema(connection)
            _verify_index_metadata(connection, manifest)
        except Exception:
            connection.close()
            raise
        self.root = root
        self.manifest = manifest
        self._connection = connection
        self._closed = False

    @classmethod
    def open(cls, store_dir: Path, *, verify: bool = True) -> MembershipStore:
        """Open a store read-only, optionally running its full streaming verifier."""
        return cls(store_dir, verify=verify)

    def close(self) -> None:
        """Close the read-only SQLite connection."""
        if not self._closed:
            self._connection.close()
            self._closed = True

    def __enter__(self) -> MembershipStore:
        self._require_open()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        del exc_type, exc_value, traceback
        self.close()

    def contains_variant(
        self,
        variant: CanonicalVariant | str,
        *,
        roles: Sequence[str] = ("validation", "evaluation"),
        sources: Sequence[str] | None = None,
    ) -> bool:
        """Return whether ``variant`` has any indexed membership in ``roles``."""
        self._require_open()
        normalized = _canonical_variant(variant)
        selected_roles = _normalize_roles(roles)
        selected_sources = _normalize_sources(sources, self.manifest.source_counts)
        role_placeholders = ",".join("?" for _ in selected_roles)
        source_clause = ""
        parameters: tuple[object, ...] = (normalized.key, *selected_roles)
        if selected_sources is not None:
            source_placeholders = ",".join("?" for _ in selected_sources)
            source_clause = f" AND source IN ({source_placeholders})"
            parameters = (*parameters, *selected_sources)
        row = self._connection.execute(
            f"SELECT 1 FROM memberships WHERE variant_key = ? "
            f"AND role IN ({role_placeholders}){source_clause} LIMIT 1",
            parameters,
        ).fetchone()
        return row is not None

    def overlaps_interval(
        self,
        chrom: str,
        *,
        start_bp: int,
        end_bp: int,
        roles: Sequence[str] = ("validation", "evaluation"),
        sources: Sequence[str] | None = None,
    ) -> bool:
        """Return whether an indexed variant intersects a 0-based half-open interval."""
        self._require_open()
        chromosome = canonicalize_chromosome(chrom)
        _require_nonnegative_int(start_bp, "membership interval start_bp")
        _require_positive_int(end_bp, "membership interval end_bp")
        if end_bp <= start_bp:
            raise InputError("membership interval end_bp must be greater than start_bp")
        selected_roles = _normalize_roles(roles)
        selected_sources = _normalize_sources(sources, self.manifest.source_counts)
        role_placeholders = ",".join("?" for _ in selected_roles)
        source_clause = ""
        parameters: tuple[object, ...] = (
            _CHROMOSOME_RANK[chromosome],
            end_bp,
            start_bp,
            *selected_roles,
        )
        if selected_sources is not None:
            source_placeholders = ",".join("?" for _ in selected_sources)
            source_clause = f" AND source IN ({source_placeholders})"
            parameters = (*parameters, *selected_sources)
        row = self._connection.execute(
            f"SELECT 1 FROM memberships "
            f"WHERE chrom_rank = ? AND start_bp < ? AND end_bp > ? "
            f"AND role IN ({role_placeholders}){source_clause} LIMIT 1",
            parameters,
        ).fetchone()
        return row is not None

    def iter_role(self, role: str, *, batch_size: int = 65_536) -> Iterator[MembershipRow]:
        """Stream one split role in canonical order with bounded Python memory."""
        self._require_open()
        selected_role = _normalize_roles((role,))[0]
        _require_positive_int(batch_size, "membership iteration batch_size")
        cursor = self._connection.execute(
            _SELECT_ROWS + " WHERE role = ? " + _ORDER_BY,
            (selected_role,),
        )
        while batch := cursor.fetchmany(batch_size):
            for raw in batch:
                yield _membership_row_from_sql(raw)

    def _require_open(self) -> None:
        if self._closed:
            raise InputError("membership store is closed")


class MembershipStoreHoldoutPolicy(HoldoutPolicy):
    """Existing tuple-builder holdout adapter backed by :class:`MembershipStore`."""

    __slots__ = ("_clinvar_sources", "store")
    store: MembershipStore
    _clinvar_sources: tuple[str, ...]

    def __init__(self, store: MembershipStore) -> None:
        if not isinstance(store, MembershipStore):
            raise InputError("membership holdout adapter requires MembershipStore")
        super().__init__(
            holdout_chroms=(
                *store.manifest.chromosome_roles.validation,
                *store.manifest.chromosome_roles.evaluation,
            )
        )
        object.__setattr__(self, "store", store)
        object.__setattr__(
            self,
            "_clinvar_sources",
            tuple(
                source.source_id for source in store.manifest.sources if source.kind == "clinvar"
            ),
        )

    def excludes_window(self, window: WindowContext) -> bool:
        """Exclude every validation/evaluation chromosome before tuple emission."""
        if not isinstance(window, WindowContext):
            raise InputError("window must be a WindowContext")
        if window.chrom is None:
            return False
        chromosome = canonicalize_chromosome(window.chrom)
        if chromosome in self.holdout_chroms:
            return True
        return self.store.overlaps_interval(
            chromosome,
            start_bp=window.start_bp,
            end_bp=window.end_bp,
            roles=REQUIRED_MEMBERSHIP_ROLES,
            sources=self._clinvar_sources,
        )

    def excludes_edit(self, window: WindowContext, edit: RelEdit) -> bool:
        """Exclude a held chromosome or a validation/evaluation variant lookup."""
        if not isinstance(window, WindowContext):
            raise InputError("window must be a WindowContext")
        if not isinstance(edit, RelEdit):
            raise InputError("edit must be a RelEdit")
        if window.chrom is None:
            return False
        chromosome = canonicalize_chromosome(window.chrom)
        if chromosome in self.holdout_chroms:
            return True
        variant = CanonicalVariant(
            assembly=self.store.manifest.assembly,
            chrom=chromosome,
            pos=window.start_bp + edit.rel_pos + 1,
            ref=edit.ref_bases,
            alt=edit.alt_bases,
        )
        return self.store.contains_variant(
            variant,
            roles=REQUIRED_MEMBERSHIP_ROLES,
            sources=self._clinvar_sources,
        )

    def to_dict(self) -> dict[str, object]:
        """Return a compact policy binding, never the complete key set."""
        return {
            "schema_version": MEMBERSHIP_STORE_SCHEMA_VERSION,
            "membership_content_identity": self.store.manifest.content_identity,
            "excluded_chromosomes": list(self.holdout_chroms),
            "excluded_source_kinds": ["clinvar"],
            "lookup": _INDEX_NAME,
        }

    def identity(self) -> str:
        """Return the semantic identity of this indexed holdout policy."""
        return canonical_json_sha256(self.to_dict())


def _canonical_variant(value: CanonicalVariant | str) -> CanonicalVariant:
    if isinstance(value, CanonicalVariant):
        return value
    if isinstance(value, str):
        return CanonicalVariant.from_key(value)
    raise InputError("membership lookup variant must be CanonicalVariant or canonical key")


def _normalize_roles(values: Sequence[str]) -> tuple[str, ...]:
    if isinstance(values, str | bytes) or not isinstance(values, Sequence) or not values:
        raise InputError("membership lookup roles must be a non-empty sequence")
    normalized = tuple(values)
    unknown = sorted(set(normalized) - set(REQUIRED_MEMBERSHIP_ROLES))
    if unknown:
        raise InputError("membership lookup role is not recognized", details={"roles": unknown})
    if len(set(normalized)) != len(normalized):
        raise InputError("membership lookup roles must be unique")
    return normalized


def _normalize_sources(
    values: Sequence[str] | None, available: Mapping[str, int]
) -> tuple[str, ...] | None:
    if values is None:
        return None
    if isinstance(values, str | bytes) or not isinstance(values, Sequence) or not values:
        raise InputError("membership lookup sources must be a non-empty sequence")
    normalized = tuple(values)
    if not all(isinstance(value, str) and value for value in normalized):
        raise InputError("membership lookup sources must contain non-empty strings")
    unknown = sorted(set(normalized) - set(available))
    if unknown:
        raise InputError(
            "membership lookup source is not bound by the manifest",
            details={"sources": unknown},
        )
    if len(set(normalized)) != len(normalized):
        raise InputError("membership lookup sources must be unique")
    return normalized
