# SPDX-License-Identifier: Apache-2.0
"""Tests for the shared conftest seed + fixture surface (RFC-0015 §3.6)."""

from __future__ import annotations

import random
from pathlib import Path

from geno_lewm.action.spec import EditSpec
from geno_lewm.provenance.commitment import DtypeConfig, PoolingConfig
from geno_lewm.provenance.receipt import ReceiptOutput


def test_random_seed_fixture_is_a_32_bit_unsigned(random_seed: int) -> None:
    assert isinstance(random_seed, int)
    assert 0 <= random_seed < 2**32


def test_seeded_random_is_a_random_instance(seeded_random: random.Random) -> None:
    """The fixture is a per-test ``random.Random`` (not the module global)."""
    assert isinstance(seeded_random, random.Random)
    # Public contract: identically seeded Randoms produce identical draws.
    s1 = random.Random(42)
    s2 = random.Random(42)
    assert s1.randint(0, 1_000_000) == s2.randint(0, 1_000_000)
    # And the per-test fixture is usable (no crash on a draw).
    seeded_random.random()


def test_synthetic_window_is_4kb_of_acgt(synthetic_window: str) -> None:
    assert len(synthetic_window) == 4096
    assert set(synthetic_window) <= set("ACGTN")


def test_synthetic_edit_spec_is_snv(synthetic_edit_spec: EditSpec) -> None:
    assert synthetic_edit_spec.chrom == "1"
    assert synthetic_edit_spec.pos == 1000
    assert synthetic_edit_spec.ref == "A"
    assert synthetic_edit_spec.alt == "T"


def test_synthetic_pooling_config(synthetic_pooling_config: PoolingConfig) -> None:
    assert synthetic_pooling_config.pool_type == "centered_mean"
    assert synthetic_pooling_config.normalize is True


def test_synthetic_dtype_config(synthetic_dtype_config: DtypeConfig) -> None:
    assert synthetic_dtype_config.encoder_dtype == "bf16"
    assert synthetic_dtype_config.predictor_dtype == "bf16"


def test_synthetic_receipt_output(synthetic_receipt_output: ReceiptOutput) -> None:
    assert synthetic_receipt_output.bucket_id == "coding.missense"
    assert synthetic_receipt_output.low_confidence is False


def test_fixtures_dir_exists(fixtures_dir: Path) -> None:
    assert fixtures_dir.is_dir()
    assert (fixtures_dir / "README.md").is_file()
