"""Behavior tests for checksum-bound dataset membership."""

from __future__ import annotations

import pytest

from geno_lewm.data.membership import (
    MEMBERSHIP_SCHEMA_VERSION,
    V03_CHROMOSOME_ROLES,
    ChromosomeRoles,
    MembershipArtifact,
    MembershipArtifactBinding,
    MembershipHoldoutPolicy,
    MembershipRow,
    derive_holdout_policy,
)
from geno_lewm.data.variant_identity import CanonicalVariant
from geno_lewm.errors import InputError
from geno_lewm.provenance import canonical_json_sha256


def test_v03_chromosome_roles_are_explicit_disjoint_and_prefix_insensitive() -> None:
    roles = V03_CHROMOSOME_ROLES

    assert roles.train == (*map(str, range(1, 20)), "22")
    assert roles.validation == ("20",)
    assert roles.evaluation == ("21",)
    assert roles.role_for("chr22") == "train"
    assert roles.role_for("chr20") == "validation"
    assert roles.role_for("21") == "evaluation"


def test_chromosome_roles_reject_empty_overlap_duplicate_and_unassigned_roles() -> None:
    with pytest.raises(InputError, match="validation chromosome role must be non-empty"):
        ChromosomeRoles(train=("1",), validation=(), evaluation=("2",))
    with pytest.raises(InputError, match="chromosome roles must be disjoint"):
        ChromosomeRoles(train=("chr1",), validation=("1",), evaluation=("2",))
    with pytest.raises(InputError, match="duplicates after canonicalization"):
        ChromosomeRoles(train=("1", "chr1"), validation=("20",), evaluation=("21",))
    with pytest.raises(InputError, match="not assigned"):
        V03_CHROMOSOME_ROLES.role_for("X")


def test_chromosome_roles_round_trip_and_reject_malformed_payloads() -> None:
    assert ChromosomeRoles.from_dict(V03_CHROMOSOME_ROLES.to_dict()) == V03_CHROMOSOME_ROLES

    with pytest.raises(InputError, match="keys do not match"):
        ChromosomeRoles.from_dict({"train": ["1"], "validation": ["20"]})
    with pytest.raises(InputError, match="must be a sequence"):
        ChromosomeRoles.from_dict({"train": "1", "validation": ["20"], "evaluation": ["21"]})
    with pytest.raises(InputError, match="must contain strings"):
        ChromosomeRoles.from_dict({"train": [1], "validation": ["20"], "evaluation": ["21"]})


def test_membership_row_round_trip_preserves_identity_reason_and_source_provenance() -> None:
    row = MembershipRow(
        variant=CanonicalVariant("GRCh38", "chr20", 101, "a", "t"),
        role="validation",
        reason_mask=3,
        source="gnomad-v4.1-exomes",
        source_row_id="chr20:101:A:T",
    )

    payload = row.to_dict()

    assert payload == {
        "variant_key": "GRCh38:20:101:A:T",
        "variant_digest": row.variant.digest,
        "role": "validation",
        "reason_mask": 3,
        "source": "gnomad-v4.1-exomes",
        "source_row_id": "chr20:101:A:T",
    }
    assert MembershipRow.from_dict(payload) == row


def test_membership_row_rejects_key_digest_schema_and_reason_drift() -> None:
    row = _row("20", "validation", "validation-1")
    payload = row.to_dict()

    with pytest.raises(InputError, match="canonical variant key"):
        MembershipRow.from_dict({**payload, "variant_key": "GRCh38:chr20:100:A:G"})
    with pytest.raises(InputError, match="digest does not match"):
        MembershipRow.from_dict({**payload, "variant_digest": "sha256:" + "0" * 64})
    with pytest.raises(InputError, match="keys do not match"):
        MembershipRow.from_dict({**payload, "unexpected": True})
    with pytest.raises(InputError, match="positive integer"):
        MembershipRow(
            variant=row.variant,
            role=row.role,
            reason_mask=0,
            source=row.source,
            source_row_id=row.source_row_id,
        )
    with pytest.raises(InputError, match="role is not recognized"):
        MembershipRow(row.variant, "test", 1, row.source, row.source_row_id)
    with pytest.raises(InputError, match="source must be a non-empty string"):
        MembershipRow(row.variant, row.role, 1, "", row.source_row_id)
    with pytest.raises(InputError, match="variant_key must be"):
        MembershipRow.from_dict({**payload, "variant_key": 20})


def test_membership_artifact_is_order_independent_and_content_addressed() -> None:
    train = _row("22", "train", "train-1")
    validation = _row("20", "validation", "validation-1")
    evaluation = _row("21", "evaluation", "evaluation-1")

    artifact = MembershipArtifact(
        artifact_id="gnomad-membership-v0.3",
        assembly="GRCh38",
        chromosome_roles=V03_CHROMOSOME_ROLES,
        rows=(evaluation, train, validation),
    )
    reordered = MembershipArtifact(
        artifact_id="gnomad-membership-v0.3",
        assembly="grch38",
        chromosome_roles=V03_CHROMOSOME_ROLES,
        rows=(train, validation, evaluation),
    )

    assert artifact.rows == (validation, evaluation, train)
    assert artifact.role_counts == {"train": 1, "validation": 1, "evaluation": 1}
    assert artifact.row_count == 3
    assert artifact.variant_count == 3
    assert artifact.content_sha256 == reordered.content_sha256
    assert artifact.to_dict()["schema_version"] == MEMBERSHIP_SCHEMA_VERSION
    assert MembershipArtifact.from_dict(artifact.to_dict()) == artifact


def test_membership_artifact_rejects_vacuous_inconsistent_and_duplicate_membership() -> None:
    train = _row("22", "train", "train-1")
    validation = _row("20", "validation", "validation-1")
    evaluation = _row("21", "evaluation", "evaluation-1")

    with pytest.raises(InputError, match="rows for every required role"):
        MembershipArtifact(
            artifact_id="vacuous",
            assembly="GRCh38",
            chromosome_roles=V03_CHROMOSOME_ROLES,
            rows=(train, validation),
        )
    with pytest.raises(InputError, match="inconsistent with chromosome roles"):
        MembershipArtifact(
            artifact_id="inconsistent",
            assembly="GRCh38",
            chromosome_roles=V03_CHROMOSOME_ROLES,
            rows=(train, validation, _row("20", "evaluation", "wrong-role")),
        )
    with pytest.raises(InputError, match="variant/source/source_row_id must be unique"):
        MembershipArtifact(
            artifact_id="duplicate-source-row",
            assembly="GRCh38",
            chromosome_roles=V03_CHROMOSOME_ROLES,
            rows=(
                _row("22", "train", "duplicate"),
                _row("22", "train", "duplicate", reason_mask=2),
                validation,
                evaluation,
            ),
        )


def test_membership_artifact_loader_rejects_schema_and_shape_drift() -> None:
    artifact = MembershipArtifact(
        artifact_id="fixture",
        assembly="GRCh38",
        chromosome_roles=V03_CHROMOSOME_ROLES,
        rows=(
            _row("22", "train", "train"),
            _row("20", "validation", "validation"),
            _row("21", "evaluation", "evaluation"),
        ),
    )
    payload = artifact.to_dict()

    with pytest.raises(InputError, match="keys do not match"):
        MembershipArtifact.from_dict({**payload, "unexpected": True})
    with pytest.raises(InputError, match="chromosome_roles must be a mapping"):
        MembershipArtifact.from_dict({**payload, "chromosome_roles": []})
    with pytest.raises(InputError, match="rows must be a sequence"):
        MembershipArtifact.from_dict({**payload, "rows": "not-rows"})
    with pytest.raises(InputError, match="rows must contain mappings"):
        MembershipArtifact.from_dict({**payload, "rows": [42]})


def test_membership_artifact_binding_round_trip_verifies_source_artifact() -> None:
    artifact = MembershipArtifact(
        artifact_id="fixture",
        assembly="GRCh38",
        chromosome_roles=V03_CHROMOSOME_ROLES,
        rows=(
            _row("22", "train", "train"),
            _row("20", "validation", "validation"),
            _row("21", "evaluation", "evaluation"),
        ),
    )
    binding = MembershipArtifactBinding.from_artifact(artifact)

    assert MembershipArtifactBinding.from_dict(binding.to_dict(), artifact=artifact) == binding


def test_membership_artifact_binding_rejects_schema_count_and_hash_drift() -> None:
    artifact = MembershipArtifact(
        artifact_id="fixture",
        assembly="GRCh38",
        chromosome_roles=V03_CHROMOSOME_ROLES,
        rows=(
            _row("22", "train", "train"),
            _row("20", "validation", "validation"),
            _row("21", "evaluation", "evaluation"),
        ),
    )
    payload = MembershipArtifactBinding.from_artifact(artifact).to_dict()

    with pytest.raises(InputError, match="keys do not match"):
        MembershipArtifactBinding.from_dict(
            {key: value for key, value in payload.items() if key != "sha256"},
            artifact=artifact,
        )
    with pytest.raises(InputError, match="keys do not match"):
        MembershipArtifactBinding.from_dict({**payload, "unexpected": True}, artifact=artifact)
    with pytest.raises(InputError, match="role_counts must be a mapping"):
        MembershipArtifactBinding.from_dict({**payload, "role_counts": []}, artifact=artifact)
    with pytest.raises(InputError, match="role_counts keys do not match"):
        MembershipArtifactBinding.from_dict(
            {**payload, "role_counts": {"train": 1, "validation": 1}},
            artifact=artifact,
        )
    with pytest.raises(InputError, match="does not match its source artifact"):
        MembershipArtifactBinding.from_dict(
            {
                **payload,
                "row_count": 4,
                "role_counts": {"train": 2, "validation": 1, "evaluation": 1},
            },
            artifact=artifact,
        )
    with pytest.raises(InputError, match="does not match its source artifact"):
        MembershipArtifactBinding.from_dict(
            {**payload, "sha256": "sha256:" + "0" * 64},
            artifact=artifact,
        )


def test_holdout_policy_binds_artifact_hashes_counts_and_union_of_non_train_membership() -> None:
    gnomad = MembershipArtifact(
        artifact_id="gnomad-membership-v0.3",
        assembly="GRCh38",
        chromosome_roles=V03_CHROMOSOME_ROLES,
        rows=(
            _row("22", "train", "g-train"),
            _row("20", "validation", "g-validation"),
            _row("21", "evaluation", "g-evaluation"),
        ),
    )
    clinvar = MembershipArtifact(
        artifact_id="clinvar-membership-v0.3",
        assembly="GRCh38",
        chromosome_roles=V03_CHROMOSOME_ROLES,
        rows=(
            _row("19", "train", "c-train", reason_mask=2),
            _row("20", "validation", "c-validation", reason_mask=2),
            _row("21", "evaluation", "c-evaluation", reason_mask=2),
        ),
    )
    expected = {
        gnomad.artifact_id: gnomad.content_sha256,
        clinvar.artifact_id: clinvar.content_sha256,
    }

    policy = derive_holdout_policy((gnomad, clinvar), expected_sha256=expected)

    assert isinstance(policy, MembershipHoldoutPolicy)
    assert policy.excluded_chromosomes == ("20", "21")
    assert policy.excluded_variant_keys == (
        "GRCh38:20:100:A:G",
        "GRCh38:21:100:A:G",
    )
    assert [binding.artifact_id for binding in policy.artifact_bindings] == [
        "clinvar-membership-v0.3",
        "gnomad-membership-v0.3",
    ]
    assert policy.artifact_bindings[0].to_dict()["role_counts"] == {
        "train": 1,
        "validation": 1,
        "evaluation": 1,
    }
    assert policy.excludes_variant(CanonicalVariant("GRCh38", "chr20", 999, "A", "C"))
    assert not policy.excludes_variant(CanonicalVariant("GRCh38", "22", 100, "A", "G"))
    assert policy.identity.startswith("sha256:")

    with pytest.raises(InputError, match="checksum"):
        derive_holdout_policy(
            (gnomad, clinvar),
            expected_sha256={**expected, gnomad.artifact_id: "sha256:" + "0" * 64},
        )
    with pytest.raises(InputError, match="checksum keys"):
        derive_holdout_policy((gnomad, clinvar), expected_sha256={})
    with pytest.raises(InputError, match="validation or evaluation chromosomes"):
        MembershipHoldoutPolicy(
            assembly=policy.assembly,
            chromosome_roles=policy.chromosome_roles,
            artifact_bindings=policy.artifact_bindings,
            excluded_chromosomes=policy.excluded_chromosomes,
            excluded_variant_keys=("GRCh38:22:100:A:G",),
        )


def test_membership_holdout_policy_round_trip_rederives_artifact_bindings_and_identity() -> None:
    artifact = MembershipArtifact(
        artifact_id="fixture",
        assembly="GRCh38",
        chromosome_roles=V03_CHROMOSOME_ROLES,
        rows=(
            _row("22", "train", "train"),
            _row("20", "validation", "validation"),
            _row("21", "evaluation", "evaluation"),
        ),
    )
    policy = derive_holdout_policy((artifact,))
    payload = policy.to_dict()

    assert payload["policy_identity"] == policy.identity
    assert MembershipHoldoutPolicy.from_dict(payload, artifacts=(artifact,)) == policy


def test_membership_holdout_policy_rejects_schema_duplicates_and_identity_drift() -> None:
    artifact = MembershipArtifact(
        artifact_id="fixture",
        assembly="GRCh38",
        chromosome_roles=V03_CHROMOSOME_ROLES,
        rows=(
            _row("22", "train", "train"),
            _row("20", "validation", "validation"),
            _row("21", "evaluation", "evaluation"),
        ),
    )
    payload = derive_holdout_policy((artifact,)).to_dict()

    with pytest.raises(InputError, match="keys do not match"):
        MembershipHoldoutPolicy.from_dict(
            {key: value for key, value in payload.items() if key != "policy_identity"},
            artifacts=(artifact,),
        )
    with pytest.raises(InputError, match="keys do not match"):
        MembershipHoldoutPolicy.from_dict({**payload, "unexpected": True}, artifacts=(artifact,))
    with pytest.raises(InputError, match="role_counts must be a mapping"):
        malformed_binding = dict(payload["artifact_bindings"][0])  # type: ignore[index]
        malformed_binding["role_counts"] = []
        MembershipHoldoutPolicy.from_dict(
            {**payload, "artifact_bindings": [malformed_binding]}, artifacts=(artifact,)
        )
    with pytest.raises(InputError, match="artifact bindings must have unique"):
        MembershipHoldoutPolicy.from_dict(
            {
                **payload,
                "artifact_bindings": [
                    *payload["artifact_bindings"],  # type: ignore[misc]
                    payload["artifact_bindings"][0],  # type: ignore[index]
                ],
            },
            artifacts=(artifact,),
        )
    with pytest.raises(InputError, match="variant keys must be unique"):
        MembershipHoldoutPolicy.from_dict(
            {
                **payload,
                "excluded_variant_keys": [
                    *payload["excluded_variant_keys"],  # type: ignore[misc]
                    payload["excluded_variant_keys"][0],  # type: ignore[index]
                ],
            },
            artifacts=(artifact,),
        )
    with pytest.raises(InputError, match="excluded chromosomes must be unique"):
        MembershipHoldoutPolicy.from_dict(
            {
                **payload,
                "excluded_chromosomes": [
                    *payload["excluded_chromosomes"],  # type: ignore[misc]
                    payload["excluded_chromosomes"][0],  # type: ignore[index]
                ],
            },
            artifacts=(artifact,),
        )
    with pytest.raises(InputError, match="policy identity drift"):
        MembershipHoldoutPolicy.from_dict(
            {**payload, "policy_identity": "sha256:" + "0" * 64}, artifacts=(artifact,)
        )


def test_membership_holdout_policy_rejects_rehashed_count_and_artifact_hash_drift() -> None:
    artifact = MembershipArtifact(
        artifact_id="fixture",
        assembly="GRCh38",
        chromosome_roles=V03_CHROMOSOME_ROLES,
        rows=(
            _row("22", "train", "train"),
            _row("20", "validation", "validation"),
            _row("21", "evaluation", "evaluation"),
        ),
    )
    payload = derive_holdout_policy((artifact,)).to_dict()
    binding = dict(payload["artifact_bindings"][0])  # type: ignore[index]

    count_drift = {**binding, "variant_count": 2}
    count_payload = _rebind_policy_identity({**payload, "artifact_bindings": [count_drift]})
    with pytest.raises(InputError, match="does not match its source artifact"):
        MembershipHoldoutPolicy.from_dict(count_payload, artifacts=(artifact,))

    hash_drift = {**binding, "sha256": "sha256:" + "0" * 64}
    hash_payload = _rebind_policy_identity({**payload, "artifact_bindings": [hash_drift]})
    with pytest.raises(InputError, match="does not match its source artifact"):
        MembershipHoldoutPolicy.from_dict(hash_payload, artifacts=(artifact,))


def test_holdout_policy_rejects_membership_artifacts_with_inconsistent_roles() -> None:
    canonical = MembershipArtifact(
        artifact_id="canonical",
        assembly="GRCh38",
        chromosome_roles=V03_CHROMOSOME_ROLES,
        rows=(
            _row("22", "train", "canonical-train"),
            _row("20", "validation", "canonical-validation"),
            _row("21", "evaluation", "canonical-evaluation"),
        ),
    )
    alternate_roles = ChromosomeRoles(train=("22",), validation=("20",), evaluation=("19",))
    alternate = MembershipArtifact(
        artifact_id="alternate",
        assembly="GRCh38",
        chromosome_roles=alternate_roles,
        rows=(
            _row("22", "train", "alternate-train"),
            _row("20", "validation", "alternate-validation"),
            _row("19", "evaluation", "alternate-evaluation"),
        ),
    )

    with pytest.raises(InputError, match="identical chromosome roles"):
        derive_holdout_policy((canonical, alternate))


def _row(chrom: str, role: str, source_row_id: str, *, reason_mask: int = 1) -> MembershipRow:
    return MembershipRow(
        variant=CanonicalVariant("GRCh38", chrom, 100, "A", "G"),
        role=role,
        reason_mask=reason_mask,
        source="fixture",
        source_row_id=source_row_id,
    )


def _rebind_policy_identity(payload: dict[str, object]) -> dict[str, object]:
    identity_payload = {key: value for key, value in payload.items() if key != "policy_identity"}
    return {**identity_payload, "policy_identity": canonical_json_sha256(identity_payload)}
