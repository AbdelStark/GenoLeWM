# SPDX-License-Identifier: Apache-2.0
"""Shared pytest configuration and fixtures (RFC-0015 §3.6).

Two responsibilities:

1. **Reproducibility seed.** ``PYTEST_RANDOM_SEED`` is honoured if set;
   otherwise the seed is derived from the current git HEAD (``HEAD`` sha
   mod 2**32). The resolved value is exposed as ``random_seed`` fixture
   and printed in the pytest header so a CI failure can be reproduced
   bit-exactly.

2. **Synthetic Phase 1 fixtures.** Lightweight, in-process data fixtures
   (window bytes, edit specs, manifest/receipt builders) that every
   unit / property / api test can consume without touching disk.
"""

from __future__ import annotations

import datetime as _dt
import hashlib
import os
import random
import subprocess
import sys
from collections.abc import Iterator
from pathlib import Path
from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    from geno_lewm.action.spec import EditSpec
    from geno_lewm.attestation.commitment import DtypeConfig, PoolingConfig
    from geno_lewm.attestation.receipt import ReceiptOutput

# Tests need ``tests.fixtures`` importable so they can read the canned
# JSON blobs. The directory ships with an ``__init__.py``; nothing else
# is required.
TESTS_DIR = Path(__file__).resolve().parent
FIXTURES_DIR = TESTS_DIR / "fixtures"


# ---------------------------------------------------------------------------
# Seed
# ---------------------------------------------------------------------------


def _commit_seed() -> int:
    """Derive a 32-bit seed from the current git HEAD; fall back to 0."""
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
            cwd=TESTS_DIR.parent,
        )
    except (FileNotFoundError, subprocess.CalledProcessError):
        return 0
    sha = out.stdout.strip()
    if not sha:
        return 0
    return int(hashlib.sha256(sha.encode("ascii")).hexdigest()[:8], 16)


def _resolve_seed() -> int:
    env = os.environ.get("PYTEST_RANDOM_SEED")
    if env is None:
        return _commit_seed()
    try:
        return int(env) & 0xFFFFFFFF
    except ValueError:
        return _commit_seed()


_SEED = _resolve_seed()


def pytest_report_header(config: pytest.Config) -> str:
    """Surface the resolved seed in the pytest banner."""
    return f"PYTEST_RANDOM_SEED = {_SEED}  (commit-derived; override via env)"


@pytest.fixture(scope="session")
def random_seed() -> int:
    """The resolved per-session seed (override with ``PYTEST_RANDOM_SEED``)."""
    return _SEED


@pytest.fixture()
def seeded_random(random_seed: int) -> Iterator[random.Random]:
    """A per-test ``random.Random`` seeded with the resolved seed.

    Use this fixture in property tests / synthetic generators so the
    randomness is reproducible per-commit. ``random.seed`` is *not*
    called globally because that leaks across tests.
    """
    yield random.Random(random_seed)


# ---------------------------------------------------------------------------
# Phase 1 synthetic fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def synthetic_window() -> str:
    """4 kB ``ACGT``-repeating reference window for commitment tests."""
    return "ACGT" * 1024


@pytest.fixture()
def synthetic_edit_spec() -> EditSpec:
    """A canonical SNV ``EditSpec`` at chr1:1000 A>T."""
    from geno_lewm.action.spec import EditSpec

    return EditSpec(chrom="1", pos=1000, ref="A", alt="T")


@pytest.fixture()
def synthetic_pooling_config() -> PoolingConfig:
    """A canonical ``PoolingConfig`` (centered_mean, layer 20)."""
    from geno_lewm.attestation.commitment import PoolingConfig

    return PoolingConfig(state_layer=20, pool_type="centered_mean", pool_radius=8, normalize=True)


@pytest.fixture()
def synthetic_dtype_config() -> DtypeConfig:
    """A canonical bf16/bf16 ``DtypeConfig``."""
    from geno_lewm.attestation.commitment import DtypeConfig

    return DtypeConfig(encoder_dtype="bf16", predictor_dtype="bf16")


@pytest.fixture()
def synthetic_receipt_output() -> ReceiptOutput:
    """A canonical ``ReceiptOutput`` (coding.missense, high confidence)."""
    from geno_lewm.attestation.receipt import ReceiptOutput

    return ReceiptOutput(
        sigma_raw=0.7321,
        sigma_calibrated=0.812,
        bucket_id="coding.missense",
        confidence=0.94,
        low_confidence=False,
    )


@pytest.fixture()
def fixtures_dir() -> Path:
    """Path to the on-disk fixture corpus (``tests/fixtures/``)."""
    return FIXTURES_DIR


@pytest.fixture()
def utc_now() -> _dt.datetime:
    """A frozen, timezone-aware UTC ``datetime`` for deterministic timestamps."""
    return _dt.datetime(2026, 5, 21, 0, 0, 0, tzinfo=_dt.timezone.utc)


@pytest.fixture()
def stable_isoformat(utc_now: _dt.datetime) -> str:
    """ISO-8601 UTC string derived from :func:`utc_now`."""
    return utc_now.isoformat(timespec="seconds")


# ---------------------------------------------------------------------------
# Diagnostics
# ---------------------------------------------------------------------------


def pytest_configure(config: pytest.Config) -> None:
    """Print the seed to stderr once so it is visible even with `-q`."""
    if os.environ.get("PYTEST_NO_SEED_BANNER") == "1":
        return
    if not getattr(config.option, "quiet", False):
        return
    sys.stderr.write(f"seed: {_SEED}\n")
