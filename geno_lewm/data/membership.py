# SPDX-License-Identifier: Apache-2.0
"""Explicit, checksum-bound variant membership for v0.3 datasets."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import cast

from geno_lewm.data.variant_identity import (
    CANONICAL_CHROMOSOMES,
    CanonicalVariant,
    canonicalize_assembly,
    canonicalize_chromosome,
)
from geno_lewm.errors import InputError, SchemaCompatError
from geno_lewm.provenance import canonical_json_sha256
from geno_lewm.provenance.hashing import looks_like_sha256

__all__ = [
    "MEMBERSHIP_SCHEMA_VERSION",
    "REQUIRED_MEMBERSHIP_ROLES",
    "V03_CHROMOSOME_ROLES",
    "ChromosomeRoles",
    "MembershipArtifact",
    "MembershipArtifactBinding",
    "MembershipHoldoutPolicy",
    "MembershipRow",
    "derive_holdout_policy",
]


MEMBERSHIP_SCHEMA_VERSION = "1.0.0"
"""Schema version for canonical membership artifacts and policies."""

REQUIRED_MEMBERSHIP_ROLES: tuple[str, ...] = ("train", "validation", "evaluation")
"""Roles that every membership contract must represent non-vacuously."""

_CHROMOSOME_ORDER = {chrom: index for index, chrom in enumerate(CANONICAL_CHROMOSOMES)}


def _normalize_role_chromosomes(name: str, values: Sequence[str]) -> tuple[str, ...]:
    if isinstance(values, str | bytes) or not isinstance(values, Sequence):
        raise InputError(
            f"{name} chromosomes must be a sequence",
            details={"role": name, "type": type(values).__name__},
        )
    normalized = tuple(canonicalize_chromosome(value) for value in values)
    if not normalized:
        raise InputError(
            f"{name} chromosome role must be non-empty",
            details={"role": name},
        )
    if len(set(normalized)) != len(normalized):
        raise InputError(
            f"{name} chromosome role contains duplicates after canonicalization",
            details={"role": name, "chromosomes": list(normalized)},
        )
    return tuple(sorted(normalized, key=_CHROMOSOME_ORDER.__getitem__))


@dataclass(frozen=True, slots=True)
class ChromosomeRoles:
    """Disjoint chromosome assignments for train, validation, and evaluation."""

    train: tuple[str, ...]
    validation: tuple[str, ...]
    evaluation: tuple[str, ...]

    def __post_init__(self) -> None:
        for role in REQUIRED_MEMBERSHIP_ROLES:
            object.__setattr__(self, role, _normalize_role_chromosomes(role, getattr(self, role)))
        owners: dict[str, str] = {}
        overlaps: dict[str, list[str]] = {}
        for role in REQUIRED_MEMBERSHIP_ROLES:
            for chrom in getattr(self, role):
                owner = owners.setdefault(chrom, role)
                if owner != role:
                    overlaps.setdefault(chrom, [owner]).append(role)
        if overlaps:
            raise InputError(
                "chromosome roles must be disjoint",
                details={"overlaps": overlaps},
            )

    def role_for(self, chrom: str) -> str:
        """Return the explicit role for ``chrom`` or fail if it is unassigned."""
        canonical = canonicalize_chromosome(chrom)
        for role in REQUIRED_MEMBERSHIP_ROLES:
            if canonical in getattr(self, role):
                return role
        raise InputError(
            "chromosome is not assigned to a membership role",
            details={"chrom": canonical},
        )

    def to_dict(self) -> dict[str, list[str]]:
        """Return the canonical JSON-native chromosome-role payload."""
        return {role: list(getattr(self, role)) for role in REQUIRED_MEMBERSHIP_ROLES}

    @classmethod
    def from_dict(cls, payload: Mapping[str, object]) -> ChromosomeRoles:
        """Parse an explicit chromosome-role payload."""
        required = set(REQUIRED_MEMBERSHIP_ROLES)
        if not isinstance(payload, Mapping):
            raise InputError(
                "chromosome roles must be a mapping",
                details={"type": type(payload).__name__},
            )
        missing = required - set(payload)
        extra = set(payload) - required
        if missing or extra:
            raise InputError(
                "chromosome role keys do not match the schema",
                details={"missing": sorted(missing), "extra": sorted(extra)},
            )
        values: dict[str, tuple[str, ...]] = {}
        for role in REQUIRED_MEMBERSHIP_ROLES:
            raw = payload[role]
            if isinstance(raw, str | bytes) or not isinstance(raw, Sequence):
                raise InputError(
                    f"{role} chromosomes must be a sequence",
                    details={"role": role, "type": type(raw).__name__},
                )
            if not all(isinstance(value, str) for value in raw):
                raise InputError(
                    f"{role} chromosomes must contain strings",
                    details={"role": role},
                )
            values[role] = tuple(cast(Sequence[str], raw))
        return cls(
            train=values["train"],
            validation=values["validation"],
            evaluation=values["evaluation"],
        )


@dataclass(frozen=True, slots=True)
class MembershipRow:
    """One canonical variant assignment with source-level provenance."""

    variant: CanonicalVariant
    role: str
    reason_mask: int
    source: str
    source_row_id: str

    def __post_init__(self) -> None:
        if not isinstance(self.variant, CanonicalVariant):
            raise InputError(
                "membership variant must be a CanonicalVariant",
                details={"type": type(self.variant).__name__},
            )
        if self.role not in REQUIRED_MEMBERSHIP_ROLES:
            raise InputError(
                "membership role is not recognized",
                details={"role": self.role, "required": list(REQUIRED_MEMBERSHIP_ROLES)},
            )
        if (
            not isinstance(self.reason_mask, int)
            or isinstance(self.reason_mask, bool)
            or self.reason_mask < 1
        ):
            raise InputError(
                "membership reason_mask must be a positive integer",
                details={"reason_mask": self.reason_mask},
            )
        for field_name in ("source", "source_row_id"):
            value = getattr(self, field_name)
            if not isinstance(value, str) or not value:
                raise InputError(
                    f"membership {field_name} must be a non-empty string",
                    details={"field": field_name, "value": value},
                )

    def to_dict(self) -> dict[str, object]:
        """Return the strict artifact-row payload."""
        return {
            "variant_key": self.variant.key,
            "variant_digest": self.variant.digest,
            "role": self.role,
            "reason_mask": self.reason_mask,
            "source": self.source,
            "source_row_id": self.source_row_id,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, object]) -> MembershipRow:
        """Parse one strict row and verify its key/digest binding."""
        required = {
            "variant_key",
            "variant_digest",
            "role",
            "reason_mask",
            "source",
            "source_row_id",
        }
        if not isinstance(payload, Mapping):
            raise InputError(
                "membership row must be a mapping",
                details={"type": type(payload).__name__},
            )
        missing = required - set(payload)
        extra = set(payload) - required
        if missing or extra:
            raise InputError(
                "membership row keys do not match the schema",
                details={"missing": sorted(missing), "extra": sorted(extra)},
            )
        variant_key = payload["variant_key"]
        if not isinstance(variant_key, str):
            raise InputError(
                "membership variant_key must be a canonical variant key",
                details={"value": variant_key},
            )
        variant = CanonicalVariant.from_key(variant_key)
        if payload["variant_digest"] != variant.digest:
            raise InputError(
                "membership variant digest does not match its canonical key",
                details={
                    "variant_key": variant.key,
                    "declared": payload["variant_digest"],
                    "computed": variant.digest,
                },
            )
        return cls(
            variant=variant,
            role=payload["role"],  # type: ignore[arg-type]
            reason_mask=payload["reason_mask"],  # type: ignore[arg-type]
            source=payload["source"],  # type: ignore[arg-type]
            source_row_id=payload["source_row_id"],  # type: ignore[arg-type]
        )


@dataclass(frozen=True, slots=True)
class MembershipArtifact:
    """Canonical membership rows plus the chromosome policy that assigned them."""

    artifact_id: str
    assembly: str
    chromosome_roles: ChromosomeRoles
    rows: tuple[MembershipRow, ...]
    schema_version: str = MEMBERSHIP_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != MEMBERSHIP_SCHEMA_VERSION:
            raise SchemaCompatError(
                "membership artifact schema_version mismatch",
                details={"got": self.schema_version, "expected": MEMBERSHIP_SCHEMA_VERSION},
            )
        if not isinstance(self.artifact_id, str) or not self.artifact_id:
            raise InputError(
                "membership artifact_id must be a non-empty string",
                details={"artifact_id": self.artifact_id},
            )
        object.__setattr__(self, "assembly", canonicalize_assembly(self.assembly))
        if not isinstance(self.chromosome_roles, ChromosomeRoles):
            raise InputError(
                "membership chromosome_roles must be ChromosomeRoles",
                details={"type": type(self.chromosome_roles).__name__},
            )
        rows_value: object = self.rows
        if not isinstance(rows_value, Sequence):
            raise InputError(
                "membership rows must be a sequence",
                details={"type": type(self.rows).__name__},
            )
        rows = tuple(self.rows)
        if not rows:
            raise InputError("membership artifact rows must be non-empty")
        if not all(isinstance(row, MembershipRow) for row in rows):
            raise InputError("membership artifact rows must contain MembershipRow values")

        seen_memberships: set[tuple[str, str, str]] = set()
        role_counts = dict.fromkeys(REQUIRED_MEMBERSHIP_ROLES, 0)
        for row in rows:
            if row.variant.assembly != self.assembly:
                raise InputError(
                    "membership row assembly does not match its artifact",
                    details={
                        "artifact": self.assembly,
                        "variant": row.variant.assembly,
                        "variant_key": row.variant.key,
                    },
                )
            expected_role = self.chromosome_roles.role_for(row.variant.chrom)
            if row.role != expected_role:
                raise InputError(
                    "membership row role is inconsistent with chromosome roles",
                    details={
                        "variant_key": row.variant.key,
                        "declared": row.role,
                        "expected": expected_role,
                    },
                )
            membership_identity = (row.variant.key, row.source, row.source_row_id)
            if membership_identity in seen_memberships:
                raise InputError(
                    "membership variant/source/source_row_id must be unique within an artifact",
                    details={
                        "variant_key": row.variant.key,
                        "source": row.source,
                        "source_row_id": row.source_row_id,
                    },
                )
            seen_memberships.add(membership_identity)
            role_counts[row.role] += 1
        empty_roles = [role for role, count in role_counts.items() if count == 0]
        if empty_roles:
            raise InputError(
                "membership artifact must contain rows for every required role",
                details={"empty_roles": empty_roles},
            )
        object.__setattr__(
            self,
            "rows",
            tuple(
                sorted(
                    rows,
                    key=lambda row: (
                        row.variant.key,
                        row.source,
                        row.source_row_id,
                        row.reason_mask,
                    ),
                )
            ),
        )

    @property
    def role_counts(self) -> dict[str, int]:
        """Return row counts for each required role."""
        return {
            role: sum(row.role == role for row in self.rows) for role in REQUIRED_MEMBERSHIP_ROLES
        }

    @property
    def row_count(self) -> int:
        """Return the number of source membership rows."""
        return len(self.rows)

    @property
    def variant_count(self) -> int:
        """Return the number of distinct canonical variants."""
        return len({row.variant.key for row in self.rows})

    def to_dict(self) -> dict[str, object]:
        """Return the deterministic, checksum-addressed artifact payload."""
        return {
            "schema_version": self.schema_version,
            "artifact_id": self.artifact_id,
            "assembly": self.assembly,
            "chromosome_roles": self.chromosome_roles.to_dict(),
            "rows": [row.to_dict() for row in self.rows],
        }

    @property
    def content_sha256(self) -> str:
        """Return the SHA-256 digest of the canonical artifact payload."""
        return canonical_json_sha256(self.to_dict())

    @classmethod
    def from_dict(cls, payload: Mapping[str, object]) -> MembershipArtifact:
        """Parse and validate a strict membership artifact payload."""
        required = {"schema_version", "artifact_id", "assembly", "chromosome_roles", "rows"}
        if not isinstance(payload, Mapping):
            raise InputError(
                "membership artifact must be a mapping",
                details={"type": type(payload).__name__},
            )
        missing = required - set(payload)
        extra = set(payload) - required
        if missing or extra:
            raise InputError(
                "membership artifact keys do not match the schema",
                details={"missing": sorted(missing), "extra": sorted(extra)},
            )
        roles_payload = payload["chromosome_roles"]
        rows_payload = payload["rows"]
        if not isinstance(roles_payload, Mapping):
            raise InputError("membership chromosome_roles must be a mapping")
        if isinstance(rows_payload, str | bytes) or not isinstance(rows_payload, Sequence):
            raise InputError("membership rows must be a sequence")
        rows: list[MembershipRow] = []
        for row_payload in rows_payload:
            if not isinstance(row_payload, Mapping):
                raise InputError("membership rows must contain mappings")
            rows.append(MembershipRow.from_dict(row_payload))
        return cls(
            artifact_id=payload["artifact_id"],  # type: ignore[arg-type]
            assembly=payload["assembly"],  # type: ignore[arg-type]
            chromosome_roles=ChromosomeRoles.from_dict(roles_payload),
            rows=tuple(rows),
            schema_version=payload["schema_version"],  # type: ignore[arg-type]
        )


@dataclass(frozen=True, slots=True)
class MembershipArtifactBinding:
    """Hash and count commitments for one membership artifact."""

    artifact_id: str
    sha256: str
    row_count: int
    variant_count: int
    train_rows: int
    validation_rows: int
    evaluation_rows: int

    def __post_init__(self) -> None:
        if not isinstance(self.artifact_id, str) or not self.artifact_id:
            raise InputError("membership binding artifact_id must be non-empty")
        if not looks_like_sha256(self.sha256):
            raise InputError(
                "membership binding checksum must be 'sha256:<64hex>'",
                details={"artifact_id": self.artifact_id, "sha256": self.sha256},
            )
        counts = {
            "row_count": self.row_count,
            "variant_count": self.variant_count,
            "train_rows": self.train_rows,
            "validation_rows": self.validation_rows,
            "evaluation_rows": self.evaluation_rows,
        }
        for name, value in counts.items():
            if not isinstance(value, int) or isinstance(value, bool) or value < 1:
                raise InputError(
                    f"membership binding {name} must be a positive integer",
                    details={"artifact_id": self.artifact_id, "field": name, "value": value},
                )
        if self.row_count != self.train_rows + self.validation_rows + self.evaluation_rows:
            raise InputError(
                "membership binding row_count does not match its role counts",
                details={"artifact_id": self.artifact_id, **counts},
            )
        if self.variant_count > self.row_count:
            raise InputError(
                "membership binding variant_count cannot exceed row_count",
                details={"artifact_id": self.artifact_id, **counts},
            )

    @classmethod
    def from_artifact(cls, artifact: MembershipArtifact) -> MembershipArtifactBinding:
        """Build an immutable commitment from validated artifact content."""
        counts = artifact.role_counts
        return cls(
            artifact_id=artifact.artifact_id,
            sha256=artifact.content_sha256,
            row_count=artifact.row_count,
            variant_count=artifact.variant_count,
            train_rows=counts["train"],
            validation_rows=counts["validation"],
            evaluation_rows=counts["evaluation"],
        )

    def to_dict(self) -> dict[str, object]:
        """Return the canonical JSON-native binding payload."""
        return {
            "artifact_id": self.artifact_id,
            "sha256": self.sha256,
            "row_count": self.row_count,
            "variant_count": self.variant_count,
            "role_counts": {
                "train": self.train_rows,
                "validation": self.validation_rows,
                "evaluation": self.evaluation_rows,
            },
        }

    @classmethod
    def from_dict(
        cls,
        payload: Mapping[str, object],
        *,
        artifact: MembershipArtifact,
    ) -> MembershipArtifactBinding:
        """Parse a strict binding and verify it against its source artifact."""
        required = {"artifact_id", "sha256", "row_count", "variant_count", "role_counts"}
        if not isinstance(payload, Mapping):
            raise InputError(
                "membership artifact binding must be a mapping",
                details={"type": type(payload).__name__},
            )
        missing = required - set(payload)
        extra = set(payload) - required
        if missing or extra:
            raise InputError(
                "membership artifact binding keys do not match the schema",
                details={"missing": sorted(missing), "extra": sorted(extra)},
            )
        role_counts = payload["role_counts"]
        if not isinstance(role_counts, Mapping):
            raise InputError("membership artifact binding role_counts must be a mapping")
        role_missing = set(REQUIRED_MEMBERSHIP_ROLES) - set(role_counts)
        role_extra = set(role_counts) - set(REQUIRED_MEMBERSHIP_ROLES)
        if role_missing or role_extra:
            raise InputError(
                "membership artifact binding role_counts keys do not match the schema",
                details={"missing": sorted(role_missing), "extra": sorted(role_extra)},
            )
        binding = cls(
            artifact_id=payload["artifact_id"],  # type: ignore[arg-type]
            sha256=payload["sha256"],  # type: ignore[arg-type]
            row_count=payload["row_count"],  # type: ignore[arg-type]
            variant_count=payload["variant_count"],  # type: ignore[arg-type]
            train_rows=role_counts["train"],
            validation_rows=role_counts["validation"],
            evaluation_rows=role_counts["evaluation"],
        )
        if not isinstance(artifact, MembershipArtifact):
            raise InputError("binding artifact must be a MembershipArtifact")
        expected = cls.from_artifact(artifact)
        if binding != expected:
            raise InputError(
                "membership artifact binding does not match its source artifact",
                details={
                    "artifact_id": artifact.artifact_id,
                    "declared": binding.to_dict(),
                    "expected": expected.to_dict(),
                },
            )
        return binding


@dataclass(frozen=True, slots=True)
class MembershipHoldoutPolicy:
    """Validation/evaluation exclusions bound to membership hashes and counts."""

    assembly: str
    chromosome_roles: ChromosomeRoles
    artifact_bindings: tuple[MembershipArtifactBinding, ...]
    excluded_chromosomes: tuple[str, ...]
    excluded_variant_keys: tuple[str, ...]
    schema_version: str = MEMBERSHIP_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != MEMBERSHIP_SCHEMA_VERSION:
            raise SchemaCompatError(
                "membership holdout schema_version mismatch",
                details={"got": self.schema_version, "expected": MEMBERSHIP_SCHEMA_VERSION},
            )
        object.__setattr__(self, "assembly", canonicalize_assembly(self.assembly))
        if not isinstance(self.chromosome_roles, ChromosomeRoles):
            raise InputError("holdout chromosome_roles must be ChromosomeRoles")
        bindings = tuple(self.artifact_bindings)
        if not bindings or not all(
            isinstance(binding, MembershipArtifactBinding) for binding in bindings
        ):
            raise InputError(
                "holdout artifact_bindings must contain at least one MembershipArtifactBinding"
            )
        if len({binding.artifact_id for binding in bindings}) != len(bindings):
            raise InputError("holdout artifact bindings must have unique artifact_id values")
        object.__setattr__(
            self,
            "artifact_bindings",
            tuple(sorted(bindings, key=lambda binding: binding.artifact_id)),
        )

        expected_chromosomes = tuple(
            sorted(
                set(self.chromosome_roles.validation) | set(self.chromosome_roles.evaluation),
                key=_CHROMOSOME_ORDER.__getitem__,
            )
        )
        declared_chromosomes = tuple(
            sorted(
                (canonicalize_chromosome(chrom) for chrom in self.excluded_chromosomes),
                key=_CHROMOSOME_ORDER.__getitem__,
            )
        )
        if len(set(declared_chromosomes)) != len(declared_chromosomes):
            raise InputError("holdout excluded chromosomes must be unique")
        if declared_chromosomes != expected_chromosomes:
            raise InputError(
                "holdout chromosomes must equal validation plus evaluation roles",
                details={
                    "declared": list(declared_chromosomes),
                    "expected": list(expected_chromosomes),
                },
            )
        object.__setattr__(self, "excluded_chromosomes", expected_chromosomes)

        variants = tuple(CanonicalVariant.from_key(key) for key in self.excluded_variant_keys)
        if any(variant.assembly != self.assembly for variant in variants):
            raise InputError("holdout variant assembly does not match the policy assembly")
        wrong_role_keys = [
            variant.key for variant in variants if variant.chrom not in expected_chromosomes
        ]
        if wrong_role_keys:
            raise InputError(
                "holdout variant keys must belong to validation or evaluation chromosomes",
                details={"variant_keys": wrong_role_keys},
            )
        canonical_keys = tuple(sorted({variant.key for variant in variants}))
        if len(canonical_keys) != len(variants):
            raise InputError("holdout variant keys must be unique")
        if not canonical_keys:
            raise InputError("holdout variant keys must be non-empty")
        object.__setattr__(self, "excluded_variant_keys", canonical_keys)

    def excludes_variant(self, variant: CanonicalVariant) -> bool:
        """Return whether a canonical variant is withheld from training."""
        if not isinstance(variant, CanonicalVariant):
            raise InputError("variant must be a CanonicalVariant")
        if variant.assembly != self.assembly:
            raise InputError(
                "variant assembly does not match the holdout policy",
                details={"variant": variant.assembly, "policy": self.assembly},
            )
        return (
            variant.chrom in self.excluded_chromosomes or variant.key in self.excluded_variant_keys
        )

    def to_dict(self) -> dict[str, object]:
        """Return the checksum- and count-bound policy plus its identity."""
        payload = self._identity_payload()
        payload["policy_identity"] = self.identity
        return payload

    def _identity_payload(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "assembly": self.assembly,
            "chromosome_roles": self.chromosome_roles.to_dict(),
            "artifact_bindings": [binding.to_dict() for binding in self.artifact_bindings],
            "excluded_chromosomes": list(self.excluded_chromosomes),
            "excluded_variant_keys": list(self.excluded_variant_keys),
        }

    @property
    def identity(self) -> str:
        """Return the canonical SHA-256 identity of this holdout policy."""
        return canonical_json_sha256(self._identity_payload())

    @classmethod
    def from_dict(
        cls,
        payload: Mapping[str, object],
        *,
        artifacts: Sequence[MembershipArtifact],
    ) -> MembershipHoldoutPolicy:
        """Parse a strict policy, verify its identity, and re-derive its bindings."""
        required = {
            "schema_version",
            "assembly",
            "chromosome_roles",
            "artifact_bindings",
            "excluded_chromosomes",
            "excluded_variant_keys",
            "policy_identity",
        }
        if not isinstance(payload, Mapping):
            raise InputError(
                "membership holdout policy must be a mapping",
                details={"type": type(payload).__name__},
            )
        missing = required - set(payload)
        extra = set(payload) - required
        if missing or extra:
            raise InputError(
                "membership holdout policy keys do not match the schema",
                details={"missing": sorted(missing), "extra": sorted(extra)},
            )
        roles_payload = payload["chromosome_roles"]
        bindings_payload = payload["artifact_bindings"]
        excluded_chromosomes = payload["excluded_chromosomes"]
        excluded_variant_keys = payload["excluded_variant_keys"]
        declared_identity = payload["policy_identity"]
        if not isinstance(roles_payload, Mapping):
            raise InputError("membership holdout chromosome_roles must be a mapping")
        if isinstance(bindings_payload, str | bytes) or not isinstance(bindings_payload, Sequence):
            raise InputError("membership holdout artifact_bindings must be a sequence")
        artifacts_value: object = artifacts
        if isinstance(artifacts_value, MembershipArtifact) or not isinstance(
            artifacts_value, Sequence
        ):
            raise InputError("membership holdout source artifacts must be a sequence")
        source_artifacts = tuple(artifacts)
        if not source_artifacts or not all(
            isinstance(artifact, MembershipArtifact) for artifact in source_artifacts
        ):
            raise InputError(
                "membership holdout source artifacts must contain MembershipArtifact values"
            )
        artifacts_by_id = {artifact.artifact_id: artifact for artifact in source_artifacts}
        if len(artifacts_by_id) != len(source_artifacts):
            raise InputError("membership holdout source artifacts must have unique identifiers")
        bindings: list[MembershipArtifactBinding] = []
        for binding_payload in bindings_payload:
            if not isinstance(binding_payload, Mapping):
                raise InputError("membership holdout artifact_bindings must contain mappings")
            artifact_id = binding_payload.get("artifact_id")
            if not isinstance(artifact_id, str) or artifact_id not in artifacts_by_id:
                raise InputError(
                    "membership holdout binding artifact_id is absent from source artifacts",
                    details={"artifact_id": artifact_id},
                )
            bindings.append(
                MembershipArtifactBinding.from_dict(
                    binding_payload,
                    artifact=artifacts_by_id[artifact_id],
                )
            )
        for name, values in (
            ("excluded_chromosomes", excluded_chromosomes),
            ("excluded_variant_keys", excluded_variant_keys),
        ):
            if isinstance(values, str | bytes) or not isinstance(values, Sequence):
                raise InputError(f"membership holdout {name} must be a sequence")
            if not all(isinstance(value, str) for value in values):
                raise InputError(f"membership holdout {name} must contain strings")
        if not isinstance(declared_identity, str) or not looks_like_sha256(declared_identity):
            raise InputError(
                "membership holdout policy_identity must be 'sha256:<64hex>'",
                details={"policy_identity": declared_identity},
            )
        policy = cls(
            assembly=payload["assembly"],  # type: ignore[arg-type]
            chromosome_roles=ChromosomeRoles.from_dict(roles_payload),
            artifact_bindings=tuple(bindings),
            excluded_chromosomes=tuple(cast(Sequence[str], excluded_chromosomes)),
            excluded_variant_keys=tuple(cast(Sequence[str], excluded_variant_keys)),
            schema_version=payload["schema_version"],  # type: ignore[arg-type]
        )
        if declared_identity != policy.identity:
            raise InputError(
                "membership holdout policy identity drift",
                details={"declared": declared_identity, "computed": policy.identity},
            )
        declared_hashes = {
            binding.artifact_id: binding.sha256 for binding in policy.artifact_bindings
        }
        expected = derive_holdout_policy(source_artifacts, expected_sha256=declared_hashes)
        if policy != expected:
            raise InputError(
                "membership holdout policy does not match its source artifacts",
                details={"declared": policy.to_dict(), "expected": expected.to_dict()},
            )
        return policy


def derive_holdout_policy(
    artifacts: Sequence[MembershipArtifact],
    *,
    expected_sha256: Mapping[str, str] | None = None,
) -> MembershipHoldoutPolicy:
    """Derive validation/evaluation exclusions from validated membership artifacts.

    When ``expected_sha256`` is provided, its keys must exactly match the
    artifact identifiers and every declared checksum must match the artifact's
    canonical payload.  Either way, the returned policy records the computed
    checksums and role/row counts.
    """
    if isinstance(artifacts, MembershipArtifact) or not isinstance(artifacts, Sequence):
        raise InputError("membership artifacts must be a sequence")
    normalized = tuple(artifacts)
    if not normalized or not all(isinstance(item, MembershipArtifact) for item in normalized):
        raise InputError("membership artifacts must contain at least one MembershipArtifact")
    artifact_ids = [artifact.artifact_id for artifact in normalized]
    if len(set(artifact_ids)) != len(artifact_ids):
        raise InputError("membership artifacts must have unique artifact_id values")

    if expected_sha256 is not None:
        if not isinstance(expected_sha256, Mapping):
            raise InputError(
                "expected membership checksums must be a mapping",
                details={"type": type(expected_sha256).__name__},
            )
        missing = set(artifact_ids) - set(expected_sha256)
        extra = set(expected_sha256) - set(artifact_ids)
        if missing or extra:
            raise InputError(
                "membership checksum keys do not match artifact identifiers",
                details={"missing": sorted(missing), "extra": sorted(extra)},
            )
        for artifact in normalized:
            declared = expected_sha256[artifact.artifact_id]
            if not looks_like_sha256(declared) or declared != artifact.content_sha256:
                raise InputError(
                    "membership artifact checksum mismatch",
                    details={
                        "artifact_id": artifact.artifact_id,
                        "declared": declared,
                        "computed": artifact.content_sha256,
                    },
                )

    reference = normalized[0]
    for artifact in normalized[1:]:
        if artifact.assembly != reference.assembly:
            raise InputError(
                "membership artifacts must use one assembly",
                details={
                    "reference": reference.assembly,
                    "artifact": artifact.assembly,
                    "artifact_id": artifact.artifact_id,
                },
            )
        if artifact.chromosome_roles != reference.chromosome_roles:
            raise InputError(
                "membership artifacts must use identical chromosome roles",
                details={"artifact_id": artifact.artifact_id},
            )

    excluded_keys = {
        row.variant.key
        for artifact in normalized
        for row in artifact.rows
        if row.role in {"validation", "evaluation"}
    }
    return MembershipHoldoutPolicy(
        assembly=reference.assembly,
        chromosome_roles=reference.chromosome_roles,
        artifact_bindings=tuple(
            MembershipArtifactBinding.from_artifact(artifact) for artifact in normalized
        ),
        excluded_chromosomes=reference.chromosome_roles.validation
        + reference.chromosome_roles.evaluation,
        excluded_variant_keys=tuple(sorted(excluded_keys)),
    )


V03_CHROMOSOME_ROLES = ChromosomeRoles(
    train=(*map(str, range(1, 20)), "22"),
    validation=("20",),
    evaluation=("21",),
)
"""Canonical chromosome split for the corrected v0.3 dataset."""
