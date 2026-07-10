"""Tests for the Carbon state-contract audit report."""

from __future__ import annotations

from pathlib import Path

import pytest

from geno_lewm.encoder._identity import encoder_runtime_hash
from geno_lewm.errors import InputError
from geno_lewm.provenance import sha256_file
from tools.research.state_contract_audit import (
    build_state_contract_report,
    verify_encoder_runtime,
    verify_encoder_weights,
)


def _report(
    raw: tuple[float, ...],
    normalized: tuple[float, ...],
    *,
    expected_d_state: int | None = None,
    carbon_weights_hash: str | None = None,
    carbon_runtime_hash: str | None = None,
    encoder_parameter_count: int = 500_000_000,
    encoder_trainable_parameter_count: int = 0,
) -> dict[str, object]:
    expected_weights_hash = "sha256:" + ("d" * 64)
    expected_runtime_hash = "sha256:" + ("e" * 64)
    return build_state_contract_report(
        raw_states=(raw,),
        normalized_states=(normalized,),
        sequence_hashes=("a" * 64,),
        commit_sha="b" * 40,
        carbon_model_dir=Path("/carbon"),
        carbon_revision="c" * 40,
        carbon_weights_hash=carbon_weights_hash or expected_weights_hash,
        expected_carbon_weights_hash=expected_weights_hash,
        carbon_runtime_hash=carbon_runtime_hash or expected_runtime_hash,
        expected_carbon_runtime_hash=expected_runtime_hash,
        encoder_parameter_count=encoder_parameter_count,
        encoder_trainable_parameter_count=encoder_trainable_parameter_count,
        expected_d_state=expected_d_state or len(raw),
        resolved_pool_type="centered_mean",
        resolved_pool_radius=8,
        resolved_center_token=1025,
        expected_center_token=1025,
        execution_device="cpu",
        state_layer=20,
        pool_radius=8,
        dtype="bf16",
        generated_at="2026-07-10T00:00:00Z",
    )


def test_state_contract_audit_accepts_unit_norm_raw_view_parity() -> None:
    report = _report((3.0, 4.0), (0.6, 0.8))

    assert report["ok"] is True
    assert report["blockers"] == []
    row = report["rows"][0]
    assert row["raw_norm"] == 5.0
    assert row["normalized_norm"] == 1.0
    assert row["max_abs_diff_vs_normalized_raw"] == 0.0
    assert report["encoder"]["weights_identity_verified"] is True
    assert report["encoder"]["runtime_identity_verified"] is True
    assert report["encoder"]["parameters_frozen"] is True


def test_state_contract_audit_rejects_uninformative_or_mismatched_view() -> None:
    report = _report((1.0, 0.0), (0.0, 1.0))

    assert report["ok"] is False
    assert {blocker["code"] for blocker in report["blockers"]} == {
        "raw_state_already_unit_norm",
        "normalized_state_contract_mismatch",
    }


def test_state_contract_audit_rejects_wrong_width_identity_or_freeze_state() -> None:
    report = _report(
        (3.0, 4.0),
        (0.6, 0.8),
        expected_d_state=1024,
        carbon_weights_hash="sha256:" + ("0" * 64),
        carbon_runtime_hash="sha256:" + ("1" * 64),
        encoder_parameter_count=0,
        encoder_trainable_parameter_count=1,
    )

    assert report["ok"] is False
    assert {blocker["code"] for blocker in report["blockers"]} == {
        "encoder_weights_identity_mismatch",
        "encoder_runtime_identity_mismatch",
        "encoder_parameters_not_frozen",
        "state_dimension_mismatch",
    }


def test_state_contract_audit_rejects_wrong_pooling_coordinate() -> None:
    expected_weights_hash = "sha256:" + ("d" * 64)
    expected_runtime_hash = "sha256:" + ("e" * 64)
    report = build_state_contract_report(
        raw_states=((3.0, 4.0),),
        normalized_states=((0.6, 0.8),),
        sequence_hashes=("a" * 64,),
        commit_sha="b" * 40,
        carbon_model_dir=Path("/carbon"),
        carbon_revision="c" * 40,
        carbon_weights_hash=expected_weights_hash,
        expected_carbon_weights_hash=expected_weights_hash,
        carbon_runtime_hash=expected_runtime_hash,
        expected_carbon_runtime_hash=expected_runtime_hash,
        encoder_parameter_count=500_000_000,
        encoder_trainable_parameter_count=0,
        expected_d_state=2,
        resolved_pool_type="centered_mean",
        resolved_pool_radius=8,
        resolved_center_token=1024,
        expected_center_token=1025,
        execution_device="cpu",
        state_layer=20,
        pool_radius=8,
        dtype="bf16",
        generated_at="2026-07-10T00:00:00Z",
    )

    assert report["ok"] is False
    assert {blocker["code"] for blocker in report["blockers"]} == {"pooling_coordinate_mismatch"}


def test_verify_encoder_weights_fails_closed_on_wrong_mount(tmp_path: Path) -> None:
    (tmp_path / "model.safetensors").write_bytes(b"pinned-carbon-weights")
    observed = sha256_file(tmp_path / "model.safetensors")

    assert verify_encoder_weights(tmp_path, expected_hash=observed) == observed
    with pytest.raises(InputError, match="mounted Carbon weights do not match"):
        verify_encoder_weights(tmp_path, expected_hash="sha256:" + ("0" * 64))


def test_verify_encoder_runtime_fails_closed_on_tokenizer_drift(tmp_path: Path) -> None:
    (tmp_path / "model.safetensors").write_bytes(b"pinned-carbon-weights")
    (tmp_path / "config.json").write_text("{}\n", encoding="utf-8")
    (tmp_path / "dna_config.json").write_text(
        '{"dna_start_id": 151669, "k": 6}\n',
        encoding="utf-8",
    )
    (tmp_path / "tokenizer_config.json").write_text("{}\n", encoding="utf-8")
    tokenizer = tmp_path / "tokenizer.py"
    tokenizer.write_text("# v1\n", encoding="utf-8")
    expected = encoder_runtime_hash(tmp_path)
    assert verify_encoder_runtime(tmp_path, expected_hash=expected) == expected
    tokenizer.write_text("# drift\n", encoding="utf-8")

    with pytest.raises(InputError, match="mounted Carbon runtime does not match"):
        verify_encoder_runtime(tmp_path, expected_hash=expected)
