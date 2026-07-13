# SPDX-License-Identifier: Apache-2.0
"""Closed types and validation primitives for scalable membership stores."""

from __future__ import annotations

import json
import re
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path, PureWindowsPath
from typing import Final

from geno_lewm.data.membership import (
    REQUIRED_MEMBERSHIP_ROLES,
    V03_CHROMOSOME_ROLES,
    ChromosomeRoles,
    MembershipRow,
)
from geno_lewm.data.variant_identity import canonicalize_chromosome
from geno_lewm.errors import InputError, SchemaCompatError
from geno_lewm.provenance.hashing import (
    canonical_json_sha256,
    looks_like_sha256,
)

MEMBERSHIP_STORE_SCHEMA_VERSION: Final = "geno-lewm.membership-store.v1"
"""Closed manifest and row schema for the scalable membership store."""

_SNAPSHOT_LINEAGE_SCHEMA_VERSION: Final = "geno-lewm.v03-snapshot-lineage.v1"
_MANIFEST_NAME: Final = "manifest.json"
_PARQUET_NAME: Final = "memberships.parquet"
_INDEX_NAME: Final = "lookup.sqlite"
_LINEAGE_NAME: Final = "snapshot-lineage.json"
_RECEIPT_NAME: Final = "build-receipt.json"
_BOUND_FILE_NAMES: Final = frozenset({_INDEX_NAME, _PARQUET_NAME, _LINEAGE_NAME, _RECEIPT_NAME})
_ARTIFACT_FILE_NAMES: Final = frozenset({_MANIFEST_NAME, *_BOUND_FILE_NAMES})
_PARQUET_BATCH_ROWS: Final = 65_536
_GNOMAD_REASON_MASK: Final = 1
_CLINVAR_REASON_MASK: Final = 2
_AUTOSOMES: Final = frozenset(str(chromosome) for chromosome in range(1, 23))
_CLINVAR_CLASSES: Final = frozenset({"B", "LB", "LP", "OTHER", "P", "VUS"})
_CLINVAR_LABELED_CLASSES: Final = frozenset({"B", "LB", "LP", "P"})
_LINEAGE_EVIDENCE_PROFILES: Final = frozenset({"official", "synthetic_fixture"})
_GNOMAD_REMOTE_POSTFLIGHT_FILES: Final = (
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
_GNOMAD_REMOTE_POSTFLIGHT_CHECKS: Final = (
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
_CLINVAR_REMOTE_POSTFLIGHT_FILES: Final = (
    "clinvar/2026-04-15/variants.parquet",
    "evidence/audit.json",
    "evidence/prepare_report.json",
    "evidence/runtime_report.json",
)
_CLINVAR_REMOTE_POSTFLIGHT_CHECKS: Final = (
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
_CLINVAR_PARQUET_FIELDS: Final = (
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
_CHROMOSOME_RANK: Final = {
    chromosome: rank for rank, chromosome in enumerate((*map(str, range(1, 23)), "X", "Y", "MT"))
}
_COMMIT: Final = re.compile(r"[0-9a-f]{40}")
_SOURCE_ID: Final = re.compile(r"[a-z0-9][a-z0-9._:-]*")
_ARTIFACT_ID: Final = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*")


@dataclass(frozen=True, slots=True)
class LabeledClinVarMembership:
    """One chromosome-assigned binary ClinVar label and its membership identity."""

    membership: MembershipRow
    clinical_significance: str

    def __post_init__(self) -> None:
        if not isinstance(self.membership, MembershipRow):
            raise InputError("labeled ClinVar membership requires a MembershipRow")
        if self.clinical_significance not in _CLINVAR_LABELED_CLASSES:
            raise InputError("labeled ClinVar membership class must be B, LB, LP, or P")

    @property
    def is_pathogenic(self) -> bool:
        """Return the closed binary target for this normalized ClinVar class."""
        return self.clinical_significance in {"LP", "P"}


@dataclass(frozen=True, slots=True)
class MembershipSourceInput:
    """One local lineage-bound source shard consumed by the builder."""

    kind: str
    path: Path
    chromosome: str | None = None

    def __post_init__(self) -> None:
        if self.kind not in {"gnomad", "clinvar"}:
            raise InputError(
                "membership source kind must be 'gnomad' or 'clinvar'",
                details={"kind": self.kind},
            )
        object.__setattr__(self, "path", Path(self.path))
        if self.kind == "gnomad":
            if self.chromosome is None:
                raise InputError("gnomAD membership source requires chromosome")
            chromosome = canonicalize_chromosome(self.chromosome)
            if chromosome not in _AUTOSOMES:
                raise InputError("gnomAD membership chromosome must be one of 1..22")
            object.__setattr__(self, "chromosome", chromosome)
        elif self.chromosome is not None:
            raise InputError("ClinVar membership source must not declare chromosome")

    @property
    def source_id(self) -> str:
        """Return the stable source identifier recorded on every derived row."""
        if self.kind == "gnomad":
            return f"gnomad-v4.1-chr{self.chromosome}"
        return "clinvar-2026-04-15"


@dataclass(frozen=True, slots=True)
class SnapshotLineageBinding:
    """Exact byte and semantic identity of the source snapshot lineage."""

    lineage_id: str
    sha256: str
    size_bytes: int
    candidate_snapshot_id: str
    _evidence_profile: str = field(init=False, repr=False, default="official")

    def __post_init__(self) -> None:
        _require_sha256(self.lineage_id, "snapshot lineage_id")
        _require_sha256(self.sha256, "snapshot lineage sha256")
        _require_positive_int(self.size_bytes, "snapshot lineage size_bytes")
        _require_text(self.candidate_snapshot_id, "snapshot candidate_snapshot_id")
        if self._evidence_profile not in _LINEAGE_EVIDENCE_PROFILES:
            raise InputError("snapshot lineage evidence_profile is not recognized")

    @property
    def evidence_profile(self) -> str:
        """Return whether the lineage passed official or fixture-only verification."""
        return self._evidence_profile

    def to_dict(self) -> dict[str, object]:
        return {
            "lineage_id": self.lineage_id,
            "sha256": self.sha256,
            "size_bytes": self.size_bytes,
            "candidate_snapshot_id": self.candidate_snapshot_id,
            "evidence_profile": self.evidence_profile,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, object]) -> SnapshotLineageBinding:
        _require_exact_keys(
            payload,
            {
                "lineage_id",
                "sha256",
                "size_bytes",
                "candidate_snapshot_id",
                "evidence_profile",
            },
            "snapshot lineage binding",
        )
        binding = cls(
            lineage_id=_require_text(payload.get("lineage_id"), "snapshot lineage_id"),
            sha256=_require_text(payload.get("sha256"), "snapshot lineage sha256"),
            size_bytes=_require_positive_int(
                payload.get("size_bytes"), "snapshot lineage size_bytes"
            ),
            candidate_snapshot_id=_require_text(
                payload.get("candidate_snapshot_id"), "snapshot candidate_snapshot_id"
            ),
        )
        evidence_profile = _require_text(
            payload.get("evidence_profile"), "snapshot lineage evidence_profile"
        )
        if evidence_profile not in _LINEAGE_EVIDENCE_PROFILES:
            raise InputError("snapshot lineage evidence_profile is not recognized")
        object.__setattr__(binding, "_evidence_profile", evidence_profile)
        return binding


@dataclass(frozen=True, slots=True)
class MembershipSourceBinding:
    """Lineage, staged-byte, and derived-row commitments for one source."""

    source_id: str
    kind: str
    repository: str
    revision: str
    namespace: str
    artifact_path: str
    artifact_sha256: str
    artifact_size_bytes: int
    artifact_row_count: int
    artifact_schema_version: str
    verification_kind: str
    verification_sha256: str
    membership_row_count: int
    filtered_row_count: int

    def __post_init__(self) -> None:
        if _SOURCE_ID.fullmatch(self.source_id) is None:
            raise InputError(
                "membership source_id is not canonical", details={"source_id": self.source_id}
            )
        if self.kind not in {"gnomad", "clinvar"}:
            raise InputError("membership source binding kind is invalid")
        _require_repository(self.repository, "membership source repository")
        _require_commit(self.revision, "membership source revision")
        _require_namespace(self.namespace, "membership source namespace")
        _require_artifact_path(self.artifact_path)
        _require_sha256(self.artifact_sha256, "membership source artifact_sha256")
        _require_positive_int(self.artifact_size_bytes, "membership source artifact_size_bytes")
        _require_positive_int(self.artifact_row_count, "membership source artifact_row_count")
        _require_text(self.artifact_schema_version, "membership source artifact_schema_version")
        if self.verification_kind != "remote_postflight":
            raise InputError("membership source verification_kind is invalid")
        _require_sha256(self.verification_sha256, "membership source verification_sha256")
        _require_nonnegative_int(self.membership_row_count, "membership source row count")
        _require_nonnegative_int(self.filtered_row_count, "membership source filtered row count")
        if self.membership_row_count + self.filtered_row_count != self.artifact_row_count:
            raise InputError(
                "membership source row reconciliation failed",
                details={
                    "source_id": self.source_id,
                    "artifact_rows": self.artifact_row_count,
                    "membership_rows": self.membership_row_count,
                    "filtered_rows": self.filtered_row_count,
                },
            )

    def to_dict(self) -> dict[str, object]:
        return {
            "source_id": self.source_id,
            "kind": self.kind,
            "repository": self.repository,
            "revision": self.revision,
            "namespace": self.namespace,
            "artifact_path": self.artifact_path,
            "artifact_sha256": self.artifact_sha256,
            "artifact_size_bytes": self.artifact_size_bytes,
            "artifact_row_count": self.artifact_row_count,
            "artifact_schema_version": self.artifact_schema_version,
            "verification_kind": self.verification_kind,
            "verification_sha256": self.verification_sha256,
            "membership_row_count": self.membership_row_count,
            "filtered_row_count": self.filtered_row_count,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, object]) -> MembershipSourceBinding:
        required = {
            "source_id",
            "kind",
            "repository",
            "revision",
            "namespace",
            "artifact_path",
            "artifact_sha256",
            "artifact_size_bytes",
            "artifact_row_count",
            "artifact_schema_version",
            "verification_kind",
            "verification_sha256",
            "membership_row_count",
            "filtered_row_count",
        }
        _require_exact_keys(payload, required, "membership source binding")
        return cls(
            source_id=_require_text(payload.get("source_id"), "membership source_id"),
            kind=_require_text(payload.get("kind"), "membership source kind"),
            repository=_require_text(payload.get("repository"), "membership source repository"),
            revision=_require_text(payload.get("revision"), "membership source revision"),
            namespace=_require_text(payload.get("namespace"), "membership source namespace"),
            artifact_path=_require_text(
                payload.get("artifact_path"), "membership source artifact_path"
            ),
            artifact_sha256=_require_text(
                payload.get("artifact_sha256"), "membership source artifact_sha256"
            ),
            artifact_size_bytes=_require_positive_int(
                payload.get("artifact_size_bytes"), "membership source artifact_size_bytes"
            ),
            artifact_row_count=_require_positive_int(
                payload.get("artifact_row_count"), "membership source artifact_row_count"
            ),
            artifact_schema_version=_require_text(
                payload.get("artifact_schema_version"),
                "membership source artifact_schema_version",
            ),
            verification_kind=_require_text(
                payload.get("verification_kind"), "membership source verification_kind"
            ),
            verification_sha256=_require_text(
                payload.get("verification_sha256"), "membership source verification_sha256"
            ),
            membership_row_count=_require_nonnegative_int(
                payload.get("membership_row_count"), "membership source row count"
            ),
            filtered_row_count=_require_nonnegative_int(
                payload.get("filtered_row_count"), "membership source filtered row count"
            ),
        )


@dataclass(frozen=True, slots=True)
class MembershipStoreFile:
    """Exact identity of one physical store file."""

    path: str
    sha256: str
    size_bytes: int

    def __post_init__(self) -> None:
        if self.path not in _BOUND_FILE_NAMES:
            raise InputError("membership store file path is not recognized")
        _require_sha256(self.sha256, "membership store file sha256")
        _require_positive_int(self.size_bytes, "membership store file size_bytes")

    def to_dict(self) -> dict[str, object]:
        return {"path": self.path, "sha256": self.sha256, "size_bytes": self.size_bytes}

    @classmethod
    def from_dict(cls, payload: Mapping[str, object]) -> MembershipStoreFile:
        _require_exact_keys(payload, {"path", "sha256", "size_bytes"}, "membership store file")
        return cls(
            path=_require_text(payload.get("path"), "membership store file path"),
            sha256=_require_text(payload.get("sha256"), "membership store file sha256"),
            size_bytes=_require_positive_int(
                payload.get("size_bytes"), "membership store file size_bytes"
            ),
        )


@dataclass(frozen=True, slots=True)
class MembershipStoreManifest:
    """Closed semantic and physical contract for one membership store."""

    artifact_id: str
    assembly: str
    chromosome_roles: ChromosomeRoles
    snapshot_lineage: SnapshotLineageBinding
    sources: tuple[MembershipSourceBinding, ...]
    row_count: int
    variant_count: int
    role_counts: dict[str, int]
    source_counts: dict[str, int]
    source_role_counts: dict[str, dict[str, int]]
    source_kind_role_counts: dict[str, dict[str, int]]
    clinvar_class_role_counts: dict[str, dict[str, int]]
    rowset_sha256: str
    files: tuple[MembershipStoreFile, ...]
    content_identity: str
    schema_version: str = MEMBERSHIP_STORE_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != MEMBERSHIP_STORE_SCHEMA_VERSION:
            raise SchemaCompatError(
                "membership store manifest schema mismatch",
                details={
                    "observed": self.schema_version,
                    "expected": MEMBERSHIP_STORE_SCHEMA_VERSION,
                },
            )
        if _ARTIFACT_ID.fullmatch(self.artifact_id) is None:
            raise InputError("membership store artifact_id is not canonical")
        if self.assembly != "GRCh38":
            raise InputError("membership store assembly must be GRCh38")
        if self.chromosome_roles != V03_CHROMOSOME_ROLES:
            raise InputError("membership store chromosome roles must equal the v0.3 split")
        if not isinstance(self.snapshot_lineage, SnapshotLineageBinding):
            raise InputError("membership store snapshot_lineage is invalid")
        sources = tuple(self.sources)
        if not sources or not all(
            isinstance(source, MembershipSourceBinding) for source in sources
        ):
            raise InputError("membership store sources must be non-empty bindings")
        if len({source.source_id for source in sources}) != len(sources):
            raise InputError("membership store source identifiers must be unique")
        object.__setattr__(self, "sources", tuple(sorted(sources, key=lambda item: item.source_id)))
        _require_positive_int(self.row_count, "membership store row_count")
        _require_positive_int(self.variant_count, "membership store variant_count")
        if self.variant_count > self.row_count:
            raise InputError("membership store variant_count cannot exceed row_count")
        _require_count_mapping(self.role_counts, set(REQUIRED_MEMBERSHIP_ROLES), "role_counts")
        if sum(self.role_counts.values()) != self.row_count:
            raise InputError("membership store role counts do not sum to row_count")
        source_ids = {source.source_id for source in sources}
        _require_count_mapping(self.source_counts, source_ids, "source_counts")
        if sum(self.source_counts.values()) != self.row_count:
            raise InputError("membership store source counts do not sum to row_count")
        for source in sources:
            if self.source_counts[source.source_id] != source.membership_row_count:
                raise InputError("membership store source count drifted from source binding")
        source_role_counts = _require_role_crosstab(
            self.source_role_counts,
            source_ids,
            "source_role_counts",
        )
        for source_id, expected_total in self.source_counts.items():
            if sum(source_role_counts[source_id].values()) != expected_total:
                raise InputError("membership store source-role counts drift from source rows")
        for role in REQUIRED_MEMBERSHIP_ROLES:
            if (
                sum(counts[role] for counts in source_role_counts.values())
                != self.role_counts[role]
            ):
                raise InputError("membership store source-role counts drift from role counts")
        source_kind_role_counts = _require_role_crosstab(
            self.source_kind_role_counts,
            {"gnomad", "clinvar"},
            "source_kind_role_counts",
        )
        sources_by_id = {source.source_id: source for source in sources}
        for kind in source_kind_role_counts:
            for role in REQUIRED_MEMBERSHIP_ROLES:
                expected = sum(
                    source_role_counts[source_id][role]
                    for source_id, source in sources_by_id.items()
                    if source.kind == kind
                )
                if source_kind_role_counts[kind][role] != expected:
                    raise InputError(
                        "membership store source-kind-role counts drift from source roles"
                    )
        class_role_counts = _require_role_crosstab(
            self.clinvar_class_role_counts,
            set(_CLINVAR_LABELED_CLASSES),
            "clinvar_class_role_counts",
        )
        for role in REQUIRED_MEMBERSHIP_ROLES:
            unique_labels = sum(class_role_counts[label][role] for label in class_role_counts)
            if unique_labels > source_kind_role_counts["clinvar"][role]:
                raise InputError("unique ClinVar class counts exceed raw ClinVar memberships")
        object.__setattr__(self, "source_role_counts", source_role_counts)
        object.__setattr__(self, "source_kind_role_counts", source_kind_role_counts)
        object.__setattr__(self, "clinvar_class_role_counts", class_role_counts)
        _require_sha256(self.rowset_sha256, "membership store rowset_sha256")
        files = tuple(sorted(self.files, key=lambda item: item.path))
        if (
            len(files) != len(_BOUND_FILE_NAMES)
            or {binding.path for binding in files} != _BOUND_FILE_NAMES
        ):
            raise InputError("membership store file bindings do not match the exact layout")
        object.__setattr__(self, "files", files)
        _require_sha256(self.content_identity, "membership store content_identity")
        expected_identity = canonical_json_sha256(self._identity_payload())
        if self.content_identity != expected_identity:
            raise InputError(
                "membership store content identity mismatch",
                details={"declared": self.content_identity, "computed": expected_identity},
            )

    def _identity_payload(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "artifact_id": self.artifact_id,
            "assembly": self.assembly,
            "chromosome_roles": self.chromosome_roles.to_dict(),
            "snapshot_lineage": self.snapshot_lineage.to_dict(),
            "sources": [source.to_dict() for source in self.sources],
            "row_count": self.row_count,
            "variant_count": self.variant_count,
            "role_counts": {role: self.role_counts[role] for role in REQUIRED_MEMBERSHIP_ROLES},
            "source_counts": dict(sorted(self.source_counts.items())),
            "source_role_counts": self.source_role_counts,
            "source_kind_role_counts": self.source_kind_role_counts,
            "clinvar_class_role_counts": self.clinvar_class_role_counts,
            "rowset_sha256": self.rowset_sha256,
        }

    def to_dict(self) -> dict[str, object]:
        return {
            **self._identity_payload(),
            "files": [binding.to_dict() for binding in self.files],
            "content_identity": self.content_identity,
            "physical_identity": self.physical_identity,
        }

    def _physical_identity_payload(self) -> dict[str, object]:
        return {
            "content_identity": self.content_identity,
            "files": [binding.to_dict() for binding in self.files],
        }

    @property
    def physical_identity(self) -> str:
        """Return the exact identity of semantic and bound physical bytes."""
        return canonical_json_sha256(self._physical_identity_payload())

    @classmethod
    def from_dict(cls, payload: Mapping[str, object]) -> MembershipStoreManifest:
        required = {
            "schema_version",
            "artifact_id",
            "assembly",
            "chromosome_roles",
            "snapshot_lineage",
            "sources",
            "row_count",
            "variant_count",
            "role_counts",
            "source_counts",
            "source_role_counts",
            "source_kind_role_counts",
            "clinvar_class_role_counts",
            "rowset_sha256",
            "files",
            "content_identity",
            "physical_identity",
        }
        _require_exact_keys(payload, required, "membership store manifest")
        roles = _require_mapping(payload.get("chromosome_roles"), "membership chromosome_roles")
        lineage = _require_mapping(payload.get("snapshot_lineage"), "membership snapshot_lineage")
        raw_sources = _require_list(payload.get("sources"), "membership sources")
        raw_files = _require_list(payload.get("files"), "membership files")
        role_counts = _parse_count_mapping(payload.get("role_counts"), "membership role_counts")
        source_counts = _parse_count_mapping(
            payload.get("source_counts"), "membership source_counts"
        )
        declared_physical_identity = _require_sha256(
            payload.get("physical_identity"), "membership physical_identity"
        )
        manifest = cls(
            schema_version=_require_text(
                payload.get("schema_version"), "membership schema_version"
            ),
            artifact_id=_require_text(payload.get("artifact_id"), "membership artifact_id"),
            assembly=_require_text(payload.get("assembly"), "membership assembly"),
            chromosome_roles=ChromosomeRoles.from_dict(roles),
            snapshot_lineage=SnapshotLineageBinding.from_dict(lineage),
            sources=tuple(
                MembershipSourceBinding.from_dict(
                    _require_mapping(item, f"membership sources[{index}]")
                )
                for index, item in enumerate(raw_sources)
            ),
            row_count=_require_positive_int(payload.get("row_count"), "membership row_count"),
            variant_count=_require_positive_int(
                payload.get("variant_count"), "membership variant_count"
            ),
            role_counts=role_counts,
            source_counts=source_counts,
            source_role_counts=_parse_role_crosstab(
                payload.get("source_role_counts"), "membership source_role_counts"
            ),
            source_kind_role_counts=_parse_role_crosstab(
                payload.get("source_kind_role_counts"),
                "membership source_kind_role_counts",
            ),
            clinvar_class_role_counts=_parse_role_crosstab(
                payload.get("clinvar_class_role_counts"),
                "membership clinvar_class_role_counts",
            ),
            rowset_sha256=_require_text(payload.get("rowset_sha256"), "membership rowset_sha256"),
            files=tuple(
                MembershipStoreFile.from_dict(_require_mapping(item, f"membership files[{index}]"))
                for index, item in enumerate(raw_files)
            ),
            content_identity=_require_text(
                payload.get("content_identity"), "membership content_identity"
            ),
        )
        if manifest.physical_identity != declared_physical_identity:
            raise InputError(
                "membership store physical identity mismatch",
                details={
                    "declared": declared_physical_identity,
                    "computed": manifest.physical_identity,
                },
            )
        return manifest


@dataclass(frozen=True, slots=True)
class MembershipStoreVerification:
    """Result of independently scanning a membership store."""

    manifest: MembershipStoreManifest
    ok: bool = True

    def to_dict(self) -> dict[str, object]:
        return {
            "ok": self.ok,
            "schema_version": self.manifest.schema_version,
            "artifact_id": self.manifest.artifact_id,
            "content_identity": self.manifest.content_identity,
            "physical_identity": self.manifest.physical_identity,
            "lineage_evidence_profile": self.manifest.snapshot_lineage.evidence_profile,
            "rowset_sha256": self.manifest.rowset_sha256,
            "row_count": self.manifest.row_count,
            "variant_count": self.manifest.variant_count,
            "role_counts": dict(self.manifest.role_counts),
            "source_counts": dict(sorted(self.manifest.source_counts.items())),
            "source_filtered_counts": {
                source.source_id: source.filtered_row_count for source in self.manifest.sources
            },
            "source_kind_filtered_counts": {
                kind: sum(
                    source.filtered_row_count
                    for source in self.manifest.sources
                    if source.kind == kind
                )
                for kind in ("gnomad", "clinvar")
            },
            "source_role_counts": self.manifest.source_role_counts,
            "source_kind_role_counts": self.manifest.source_kind_role_counts,
            "clinvar_class_role_counts": self.manifest.clinvar_class_role_counts,
        }


def _read_manifest(path: Path) -> MembershipStoreManifest:
    _raw, payload = _read_json_mapping(path, "membership store manifest")
    return MembershipStoreManifest.from_dict(payload)


def _read_json_mapping(path: Path, field: str) -> tuple[bytes, Mapping[str, object]]:
    try:
        raw = path.read_bytes()
        payload: object = json.loads(raw, object_pairs_hook=_reject_duplicate_pairs)
    except OSError as exc:
        raise InputError(f"failed to read {field}", details={"path": str(path)}) from exc
    except json.JSONDecodeError as exc:
        raise InputError(
            f"{field} JSON is invalid",
            details={"path": str(path), "line": exc.lineno, "column": exc.colno},
        ) from exc
    return raw, _require_mapping(payload, field)


def _reject_duplicate_pairs(pairs: list[tuple[str, object]]) -> dict[str, object]:
    payload: dict[str, object] = {}
    for key, value in pairs:
        if key in payload:
            raise InputError("duplicate JSON key is not allowed", details={"key": key})
        payload[key] = value
    return payload


def _require_mapping(value: object, field: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise InputError(f"{field} must be an object")
    return value


def _require_list(value: object, field: str) -> list[object]:
    if not isinstance(value, list):
        raise InputError(f"{field} must be an array")
    return value


def _require_text(value: object, field: str) -> str:
    if not isinstance(value, str) or not value:
        raise InputError(f"{field} must be a non-empty string")
    return value


def _require_positive_int(value: object, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise InputError(f"{field} must be a positive integer")
    return value


def _require_nonnegative_int(value: object, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise InputError(f"{field} must be a non-negative integer")
    return value


def _require_sha256(value: object, field: str) -> str:
    text = _require_text(value, field)
    if not looks_like_sha256(text):
        raise InputError(f"{field} must be a sha256-prefixed lowercase digest")
    return text


def _require_commit(value: object, field: str) -> str:
    text = _require_text(value, field)
    if _COMMIT.fullmatch(text) is None or text == "0" * 40:
        raise InputError(f"{field} must be a non-zero lowercase 40-character commit")
    return text


def _require_repository(value: object, field: str) -> str:
    text = _require_text(value, field)
    if re.fullmatch(r"[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+", text) is None:
        raise InputError(f"{field} must be a namespace/name pair")
    return text


def _require_namespace(value: object, field: str) -> str:
    text = _require_text(value, field)
    if text.strip("/") != text or not text.startswith("staging/"):
        raise InputError(f"{field} must be a normalized staging namespace")
    if any(part in {"", ".", ".."} for part in text.split("/")):
        raise InputError(f"{field} contains an unsafe path component")
    return text


def _require_artifact_path(value: object) -> str:
    text = _require_text(value, "membership source artifact_path")
    candidate = Path(text)
    windows = PureWindowsPath(text)
    if (
        candidate.is_absolute()
        or windows.is_absolute()
        or bool(windows.drive)
        or ".." in candidate.parts
        or any(part in {"", "."} for part in candidate.parts)
    ):
        raise InputError("membership source artifact_path is unsafe")
    return text


def _require_exact_keys(value: Mapping[str, object], expected: set[str], field: str) -> None:
    observed = set(value)
    if observed != expected:
        raise InputError(
            f"{field} keys do not match the closed schema",
            details={
                "missing": sorted(expected - observed),
                "unexpected": sorted(observed - expected),
            },
        )


def _parse_count_mapping(value: object, field: str) -> dict[str, int]:
    mapping = _require_mapping(value, field)
    parsed: dict[str, int] = {}
    for key, count in mapping.items():
        if not isinstance(key, str) or not key:
            raise InputError(f"{field} keys must be non-empty strings")
        parsed[key] = _require_nonnegative_int(count, f"{field}.{key}")
    return parsed


def _require_count_mapping(value: Mapping[str, int], expected_keys: set[str], field: str) -> None:
    if set(value) != expected_keys:
        raise InputError(
            f"membership store {field} keys do not match",
            details={
                "missing": sorted(expected_keys - set(value)),
                "unexpected": sorted(set(value) - expected_keys),
            },
        )
    for key, count in value.items():
        _require_positive_int(count, f"membership store {field}.{key}")


def _parse_role_crosstab(value: object, field: str) -> dict[str, dict[str, int]]:
    mapping = _require_mapping(value, field)
    parsed: dict[str, dict[str, int]] = {}
    for key, counts in mapping.items():
        if not isinstance(key, str) or not key:
            raise InputError(f"{field} keys must be non-empty strings")
        parsed[key] = _parse_count_mapping(counts, f"{field}.{key}")
    return parsed


def _require_role_crosstab(
    value: Mapping[str, Mapping[str, int]], expected_keys: set[str], field: str
) -> dict[str, dict[str, int]]:
    if set(value) != expected_keys:
        raise InputError(f"membership store {field} outer keys do not match")
    normalized: dict[str, dict[str, int]] = {}
    for key in sorted(expected_keys):
        counts = value[key]
        if set(counts) != set(REQUIRED_MEMBERSHIP_ROLES):
            raise InputError(f"membership store {field}.{key} role keys do not match")
        normalized[key] = {
            role: _require_nonnegative_int(counts[role], f"membership store {field}.{key}.{role}")
            for role in REQUIRED_MEMBERSHIP_ROLES
        }
    return normalized


def _write_json(path: Path, payload: Mapping[str, object]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
