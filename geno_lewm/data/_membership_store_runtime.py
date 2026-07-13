# SPDX-License-Identifier: Apache-2.0
"""Read-only indexed lookup and tuple-builder holdout adapter."""

from __future__ import annotations

import os
import sqlite3
import threading
import weakref
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from types import TracebackType

from geno_lewm.action import RelEdit
from geno_lewm.data._membership_store_contract import (
    _CHROMOSOME_RANK,
    _INDEX_NAME,
    _MANIFEST_NAME,
    MEMBERSHIP_STORE_SCHEMA_VERSION,
    LabeledClinVarMembership,
    MembershipStoreManifest,
    _read_manifest,
    _require_nonnegative_int,
    _require_positive_int,
)
from geno_lewm.data._membership_store_snapshot import (
    _capture_membership_store,
    _CapturedMembershipStore,
)
from geno_lewm.data._membership_store_storage import (
    _ORDER_BY,
    _SELECT_ROWS,
    _SELECT_UNIQUE_CLINVAR_ROWS,
    _labeled_clinvar_membership_from_sql,
    _membership_row_from_sql,
    _verify_index_metadata,
    _verify_index_schema,
)
from geno_lewm.data._membership_store_verifier import (
    _verify_captured_membership_store,
    _verify_exact_layout,
)
from geno_lewm.data.builder import HoldoutPolicy, WindowContext
from geno_lewm.data.membership import REQUIRED_MEMBERSHIP_ROLES, MembershipRow
from geno_lewm.data.variant_identity import CanonicalVariant, canonicalize_chromosome
from geno_lewm.errors import InputError
from geno_lewm.provenance.hashing import canonical_json_sha256


@dataclass(slots=True)
class _ThreadConnection:
    connection: sqlite3.Connection
    finalizer: weakref.finalize[[], threading.Thread]

    def close(self) -> None:
        if self.finalizer.detach() is not None:
            self.connection.close()


class MembershipStore:
    """Read-only, indexed membership lookup without a PyArrow dependency."""

    __slots__ = (
        "_capture",
        "_capture_owner_pid",
        "_closed",
        "_connections",
        "_lock",
        "_lookup_root",
        "_owner_pid",
        "manifest",
        "root",
    )

    def __init__(self, store_dir: Path, *, verify: bool = True) -> None:
        root = Path(store_dir).absolute()
        capture: _CapturedMembershipStore | None = None
        if verify:
            capture = _capture_membership_store(root)
            try:
                manifest = _verify_captured_membership_store(capture).manifest
            except Exception:
                capture.close()
                raise
            capture.retain_only({_INDEX_NAME})
            lookup_root = capture.root
        else:
            _verify_exact_layout(root)
            manifest = _read_manifest(root / _MANIFEST_NAME)
            lookup_root = root
        path = lookup_root / _INDEX_NAME
        if not path.is_file():
            raise InputError("membership lookup index is missing", details={"path": str(path)})
        self.root = root
        self.manifest = manifest
        self._lookup_root = lookup_root
        self._capture = capture
        self._capture_owner_pid = os.getpid() if capture is not None else None
        self._closed = False
        self._connections: weakref.WeakKeyDictionary[threading.Thread, _ThreadConnection] = (
            weakref.WeakKeyDictionary()
        )
        self._lock = threading.Lock()
        self._owner_pid = os.getpid()

    @classmethod
    def open(cls, store_dir: Path, *, verify: bool = True) -> MembershipStore:
        """Open a store read-only, optionally running its full streaming verifier."""
        return cls(store_dir, verify=verify)

    def close(self) -> None:
        """Close this handle and every per-thread read-only connection it owns."""
        if not self._closed:
            with self._lock:
                for connection in self._connections.values():
                    connection.close()
                self._connections.clear()
            self._closed = True
            if self._capture is not None and self._capture_owner_pid == os.getpid():
                self._capture.close()
            self._capture = None

    def __getstate__(self) -> dict[str, object]:
        """Serialize stable bindings only; live SQLite handles never cross processes."""
        return {
            "root": str(self.root),
            "manifest": self.manifest,
            "closed": self._closed,
        }

    def __setstate__(self, state: Mapping[str, object]) -> None:
        root = state["root"]
        if not isinstance(root, str):
            raise InputError("pickled membership store root is invalid")
        self.root = Path(root)
        manifest = state["manifest"]
        if not isinstance(manifest, MembershipStoreManifest):
            raise InputError("pickled membership store manifest is invalid")
        self.manifest = manifest
        self._closed = bool(state["closed"])
        self._capture = None
        self._capture_owner_pid = None
        self._lookup_root = self.root
        if not self._closed:
            capture = _capture_membership_store(self.root)
            try:
                verified = _verify_captured_membership_store(capture).manifest
            except Exception:
                capture.close()
                raise
            if verified.physical_identity != self.manifest.physical_identity:
                capture.close()
                raise InputError(
                    "pickled membership store physical identity is no longer available"
                )
            self.manifest = verified
            capture.retain_only({_INDEX_NAME})
            self._capture = capture
            self._capture_owner_pid = os.getpid()
            self._lookup_root = capture.root
        self._connections = weakref.WeakKeyDictionary()
        self._lock = threading.Lock()
        self._owner_pid = os.getpid()

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, MembershipStore):
            return NotImplemented
        return self.manifest.content_identity == other.manifest.content_identity

    def __hash__(self) -> int:
        return hash((MembershipStore, self.manifest.content_identity))

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
        row = (
            self._connection()
            .execute(
                f"SELECT 1 FROM memberships WHERE variant_key = ? "
                f"AND role IN ({role_placeholders}){source_clause} LIMIT 1",
                parameters,
            )
            .fetchone()
        )
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
            _CHROMOSOME_RANK[chromosome],
            end_bp,
            start_bp,
            *selected_roles,
        )
        if selected_sources is not None:
            source_placeholders = ",".join("?" for _ in selected_sources)
            source_clause = f" AND memberships.source IN ({source_placeholders})"
            parameters = (*parameters, *selected_sources)
        row = (
            self._connection()
            .execute(
                "SELECT 1 FROM membership_intervals AS intervals "
                "CROSS JOIN memberships AS memberships "
                "ON memberships.membership_id = intervals.membership_id "
                "WHERE intervals.chrom_min <= ? AND intervals.chrom_max >= ? "
                "AND intervals.start_min < ? AND intervals.end_max > ? "
                f"AND memberships.role IN ({role_placeholders}){source_clause} LIMIT 1",
                parameters,
            )
            .fetchone()
        )
        return row is not None

    def iter_role(self, role: str, *, batch_size: int = 65_536) -> Iterator[MembershipRow]:
        """Stream one split role in canonical order with bounded Python memory."""
        self._require_open()
        selected_role = _normalize_roles((role,))[0]
        _require_positive_int(batch_size, "membership iteration batch_size")
        cursor = self._connection().execute(
            _SELECT_ROWS + " WHERE role = ? " + _ORDER_BY,
            (selected_role,),
        )
        while batch := cursor.fetchmany(batch_size):
            for raw in batch:
                yield _membership_row_from_sql(raw)

    def iter_labeled_clinvar(
        self, role: str, *, batch_size: int = 65_536
    ) -> Iterator[LabeledClinVarMembership]:
        """Stream binary-labeled ClinVar rows for one chromosome-assigned role."""
        self._require_open()
        selected_role = _normalize_roles((role,))[0]
        _require_positive_int(batch_size, "membership iteration batch_size")
        cursor = self._connection().execute(
            _SELECT_UNIQUE_CLINVAR_ROWS,
            (selected_role,),
        )
        while batch := cursor.fetchmany(batch_size):
            for raw in batch:
                yield _labeled_clinvar_membership_from_sql(raw)

    def _require_open(self) -> None:
        if self._closed:
            raise InputError("membership store is closed")

    def _connection(self) -> sqlite3.Connection:
        self._require_open()
        current_pid = os.getpid()
        if self._owner_pid != current_pid:
            for inherited in self._connections.values():
                inherited.close()
            self._connections = weakref.WeakKeyDictionary()
            self._lock = threading.Lock()
            self._owner_pid = current_pid
        key = threading.current_thread()
        with self._lock:
            thread_connection = self._connections.get(key)
        if thread_connection is not None:
            return thread_connection.connection
        with self._lock:
            thread_connection = self._connections.get(key)
            if thread_connection is not None:
                return thread_connection.connection
            connection: sqlite3.Connection | None = None
            path = self._lookup_root / _INDEX_NAME
            try:
                connection = sqlite3.connect(
                    f"{path.as_uri()}?mode=ro&immutable=1",
                    uri=True,
                    check_same_thread=False,
                )
                connection.row_factory = sqlite3.Row
                _verify_index_schema(connection)
                _verify_index_metadata(connection, self.manifest)
            except (OSError, sqlite3.DatabaseError) as exc:
                if connection is not None:
                    connection.close()
                raise InputError("membership lookup index cannot be opened") from exc
            except Exception:
                if connection is not None:
                    connection.close()
                raise
            self._connections[key] = _ThreadConnection(
                connection=connection,
                finalizer=weakref.finalize(key, connection.close),
            )
            return connection


class MembershipStoreHoldoutPolicy(HoldoutPolicy):
    """Existing tuple-builder holdout adapter backed by :class:`MembershipStore`."""

    __slots__ = ("store",)
    store: MembershipStore

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

    def __reduce__(self) -> tuple[object, tuple[MembershipStore]]:
        return type(self), (self.store,)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, MembershipStoreHoldoutPolicy):
            return NotImplemented
        return self.identity() == other.identity()

    def __hash__(self) -> int:
        return hash((MembershipStoreHoldoutPolicy, self.identity()))

    def excludes_window(self, window: WindowContext) -> bool:
        """Exclude every validation/evaluation chromosome before tuple emission."""
        if not isinstance(window, WindowContext):
            raise InputError("window must be a WindowContext")
        _chromosome, role = self._placed_role(window)
        return role != "train"

    def excludes_edit(self, window: WindowContext, edit: RelEdit) -> bool:
        """Exclude a held chromosome or a validation/evaluation variant lookup."""
        if not isinstance(window, WindowContext):
            raise InputError("window must be a WindowContext")
        if not isinstance(edit, RelEdit):
            raise InputError("edit must be a RelEdit")
        _chromosome, role = self._placed_role(window)
        return role != "train"

    def _placed_role(self, window: WindowContext) -> tuple[str, str]:
        if window.chrom is None:
            raise InputError(
                "membership holdout windows must be placed and assigned to the v0.3 split"
            )
        chromosome = canonicalize_chromosome(window.chrom)
        try:
            role = self.store.manifest.chromosome_roles.role_for(chromosome)
        except InputError as exc:
            raise InputError(
                "membership holdout windows must be placed and assigned to the v0.3 split",
                details={"chrom": chromosome},
            ) from exc
        return chromosome, role

    def to_dict(self) -> dict[str, object]:
        """Return a compact policy binding, never the complete key set."""
        return {
            "schema_version": MEMBERSHIP_STORE_SCHEMA_VERSION,
            "membership_content_identity": self.store.manifest.content_identity,
            "excluded_chromosomes": list(self.holdout_chroms),
            "selection": "chromosome_roles",
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
