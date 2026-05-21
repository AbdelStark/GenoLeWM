"""Tests for ``geno_lewm.attestation`` manifest + hashing primitives."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from geno_lewm.attestation import (
    SCHEMA_VERSION,
    Manifest,
    ManifestArtifact,
    ManifestEncoder,
    ManifestTraining,
    canonical_json_sha256,
    load_manifest,
    sha256_bytes,
    sha256_file,
    write_manifest,
)
from geno_lewm.attestation.hashing import canonical_json_bytes, looks_like_sha256
from geno_lewm.errors import InputError, SchemaCompatError

_SHA = "sha256:" + "0" * 64


def _build_minimal_manifest() -> Manifest:
    return Manifest(
        schema_version=SCHEMA_VERSION,
        model_name="geno-lewm",
        model_version="0.1.0",
        release_id="geno-lewm-v0.1.0-carbon-500m-r1",
        encoder=ManifestEncoder(
            id="HuggingFaceBio/Carbon-500M",
            revision="main@deadbeef",
            hash=_SHA,
        ),
        predictor=ManifestArtifact(file="predictor.safetensors", hash=_SHA, dtype="bf16"),
        action_encoder=ManifestArtifact(file="action_encoder.safetensors", hash=_SHA, dtype="bf16"),
        calibration=ManifestArtifact(file="calibration.parquet", hash=_SHA, version="1.0.0"),
        training=ManifestTraining(
            config_file="train_config.yaml",
            hash=_SHA,
            data_snapshot={
                "corpus_id": "HuggingFaceBio/carbon-pretraining-corpus",
                "corpus_revision": "main@cafef00d",
                "gnomad_release": "v4.1",
                "clinvar_release": "2026-04-15",
            },
        ),
        eval=ManifestArtifact(file="eval_report.md", hash=_SHA),
    )


# ---------------------------------------------------------------------------
# Canonical JSON encoding properties.


def test_canonical_json_is_key_sorted_compact() -> None:
    b = canonical_json_bytes({"b": 1, "a": 2, "nested": {"y": 3, "x": 4}})
    assert b == b'{"a":2,"b":1,"nested":{"x":4,"y":3}}'


def test_canonical_json_rejects_nan_and_inf() -> None:
    import math

    with pytest.raises(InputError):
        canonical_json_bytes({"x": math.nan})
    with pytest.raises(InputError):
        canonical_json_bytes({"x": math.inf})


def test_canonical_json_rejects_bytes() -> None:
    with pytest.raises(InputError):
        canonical_json_bytes({"x": b"hi"})


def test_canonical_json_sha256_is_byte_stable() -> None:
    obj1 = {"b": 1, "a": 2}
    obj2 = {"a": 2, "b": 1}
    assert canonical_json_sha256(obj1) == canonical_json_sha256(obj2)


def test_sha256_bytes_matches_known_value() -> None:
    # Empty input — well-known SHA-256.
    assert (
        sha256_bytes(b"")
        == "sha256:e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
    )


def test_sha256_file_streams_correctly(tmp_path: Path) -> None:
    big = tmp_path / "big.bin"
    big.write_bytes(b"\x00" * (1 << 21))  # 2 MiB → two chunks.
    expected = sha256_bytes(b"\x00" * (1 << 21))
    assert sha256_file(big) == expected


def test_looks_like_sha256() -> None:
    assert looks_like_sha256("sha256:" + "f" * 64)
    assert not looks_like_sha256("sha256:" + "f" * 63)
    assert not looks_like_sha256("sha1:" + "f" * 64)
    assert not looks_like_sha256("")


# ---------------------------------------------------------------------------
# Manifest dataclass validation.


def test_minimal_manifest_constructs() -> None:
    m = _build_minimal_manifest()
    assert m.schema_version == SCHEMA_VERSION
    assert m.model_id().startswith("sha256:")


def test_manifest_rejects_wrong_schema_version() -> None:
    with pytest.raises(SchemaCompatError):
        Manifest(
            schema_version="2.0.0",
            model_name="x",
            model_version="0.1.0",
            release_id="r",
            encoder=ManifestEncoder("a", "b", _SHA),
            predictor=ManifestArtifact("p", _SHA),
            action_encoder=ManifestArtifact("a", _SHA),
            calibration=ManifestArtifact("c", _SHA),
            training=ManifestTraining("t", _SHA),
            eval=ManifestArtifact("e", _SHA),
        )


def test_artifact_rejects_bad_hash() -> None:
    with pytest.raises(InputError):
        ManifestArtifact(file="p", hash="not-a-hash")
    with pytest.raises(InputError):
        ManifestArtifact(file="", hash=_SHA)


def test_encoder_rejects_empty_fields() -> None:
    with pytest.raises(InputError):
        ManifestEncoder(id="", revision="r", hash=_SHA)
    with pytest.raises(InputError):
        ManifestEncoder(id="x", revision="", hash=_SHA)
    with pytest.raises(InputError):
        ManifestEncoder(id="x", revision="y", hash="bogus")


def test_training_rejects_empty_config_file() -> None:
    with pytest.raises(InputError):
        ManifestTraining(config_file="", hash=_SHA)


# ---------------------------------------------------------------------------
# Round-trip.


def test_manifest_disk_round_trip_is_byte_stable(tmp_path: Path) -> None:
    m = _build_minimal_manifest()
    p1 = write_manifest(m, tmp_path / "manifest.json")
    p2 = write_manifest(m, tmp_path / "manifest2.json")
    assert p1.read_bytes() == p2.read_bytes()
    loaded = load_manifest(p1)
    assert loaded == m
    assert loaded.model_id() == m.model_id()


def test_load_manifest_rejects_unknown_top_level_keys(tmp_path: Path) -> None:
    m = _build_minimal_manifest()
    raw = json.loads(write_manifest(m, tmp_path / "m.json").read_text())
    raw["bogus"] = 1
    bad = tmp_path / "bad.json"
    bad.write_text(json.dumps(raw), encoding="utf-8")
    with pytest.raises(InputError):
        load_manifest(bad)


def test_load_manifest_rejects_missing_keys(tmp_path: Path) -> None:
    bad = tmp_path / "bad.json"
    bad.write_text(json.dumps({"schema_version": SCHEMA_VERSION}), encoding="utf-8")
    with pytest.raises(InputError):
        load_manifest(bad)


def test_load_manifest_rejects_invalid_json(tmp_path: Path) -> None:
    bad = tmp_path / "bad.json"
    bad.write_text("{not json", encoding="utf-8")
    with pytest.raises(InputError):
        load_manifest(bad)


def test_model_id_is_deterministic() -> None:
    a = _build_minimal_manifest()
    b = _build_minimal_manifest()
    assert a.model_id() == b.model_id()


def test_model_id_changes_on_field_tweak() -> None:
    a = _build_minimal_manifest()
    different_sha = "sha256:" + "1" * 64
    b = Manifest(
        schema_version=SCHEMA_VERSION,
        model_name=a.model_name,
        model_version=a.model_version,
        release_id=a.release_id,
        encoder=a.encoder,
        predictor=ManifestArtifact(file=a.predictor.file, hash=different_sha, dtype="bf16"),
        action_encoder=a.action_encoder,
        calibration=a.calibration,
        training=a.training,
        eval=a.eval,
    )
    assert a.model_id() != b.model_id()


# ---------------------------------------------------------------------------
# Tampering detection: any single-byte change to a weights file flips
# its sha256_file hash, which is what the loader (downstream) compares
# against the manifest. We assert the property here.


def test_single_byte_tamper_changes_file_hash(tmp_path: Path) -> None:
    src = tmp_path / "weights.bin"
    src.write_bytes(b"\x00" * 1024)
    h0 = sha256_file(src)
    # Flip a byte.
    raw = bytearray(src.read_bytes())
    raw[512] = 0xFF
    src.write_bytes(bytes(raw))
    h1 = sha256_file(src)
    assert h0 != h1
