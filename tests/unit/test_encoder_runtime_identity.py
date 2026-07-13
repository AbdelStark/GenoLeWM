"""Tests for the closed cache-build encoder runtime identity contract."""

from __future__ import annotations

import json

import pytest

from geno_lewm.encoder.runtime_identity import parse_encoder_runtime_identity_bytes
from geno_lewm.errors import InputError


def test_corrected_carbon_runtime_identity_accepts_published_exact_constants() -> None:
    payload = {
        "schema_version": "1.0.0",
        "model_id": "/carbon",
        "revision": "5d31d59b3c845b288a13aedb1358934196852eec",
        "state_contract_version": "l2_normalized_v2",
        "runtime_hash": ("sha256:add3c1a663a35fb92fbd3fd935b067da1aed8aeb143ea01f7d92c2cd3ed2aa5e"),
    }

    identity = parse_encoder_runtime_identity_bytes(
        json.dumps(payload).encode("utf-8"),
        source="fixture",
    )

    assert identity.to_dict() == payload
    assert identity.cache_identity_hash == payload["runtime_hash"]


@pytest.mark.parametrize(
    "revision",
    [
        "5d31d59b",
        "v1.2.3",
        "5D31D59B3C845B288A13AEDB1358934196852EEC",
        "main",
    ],
)
def test_runtime_identity_requires_exact_lowercase_commit_sha(revision: str) -> None:
    payload = {
        "schema_version": "1.0.0",
        "model_id": "/carbon",
        "revision": revision,
        "state_contract_version": "l2_normalized_v2",
        "runtime_hash": "sha256:" + "0" * 64,
    }

    with pytest.raises(
        InputError,
        match="exact lowercase 40-character hexadecimal commit SHA",
    ):
        parse_encoder_runtime_identity_bytes(
            json.dumps(payload).encode("utf-8"),
            source="fixture",
        )


@pytest.mark.parametrize(
    "payload",
    [
        {
            "schema_version": "1.0.0",
            "model_id": "/carbon",
            "revision": "5d31d59b3c845b288a13aedb1358934196852eec",
            "state_contract_version": "future_v9",
            "runtime_hash": "sha256:" + "0" * 64,
        },
        {
            "schema_version": "1.0.0",
            "model_id": "/carbon",
            "revision": "main",
            "state_contract_version": "l2_normalized_v2",
            "runtime_hash": "sha256:" + "0" * 64,
        },
        {
            "schema_version": "1.0.0",
            "model_id": "/carbon",
            "revision": "5d31d59b3c845b288a13aedb1358934196852eec",
            "state_contract_version": "l2_normalized_v2",
            "runtime_hash": "not-a-hash",
        },
        {
            "schema_version": "1.0.0",
            "model_id": "/carbon",
            "revision": "5d31d59b3c845b288a13aedb1358934196852eec",
            "state_contract_version": "l2_normalized_v2",
            "runtime_hash": "sha256:" + "0" * 64,
            "predictor": {"hash": "sha256:" + "1" * 64},
        },
    ],
)
def test_runtime_identity_rejects_malformed_or_unrelated_release_fields(
    payload: dict[str, object],
) -> None:
    with pytest.raises(InputError):
        parse_encoder_runtime_identity_bytes(
            json.dumps(payload).encode("utf-8"),
            source="fixture",
        )


def test_legacy_runtime_identity_requires_explicit_weights_hash() -> None:
    payload = {
        "schema_version": "1.0.0",
        "model_id": "HuggingFaceBio/Carbon-500M",
        "revision": "5d31d59b3c845b288a13aedb1358934196852eec",
        "state_contract_version": "legacy_raw_v1",
        "runtime_hash": "sha256:" + "0" * 64,
    }

    with pytest.raises(InputError, match="requires weights_hash"):
        parse_encoder_runtime_identity_bytes(
            json.dumps(payload).encode("utf-8"),
            source="fixture",
        )
