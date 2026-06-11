# SPDX-License-Identifier: Apache-2.0
"""pytest-benchmark microbenchmarks for the hot paths (performance budget).

The suite targets functions that the receipt verifier and the action
mutation path invoke in tight loops:

- :func:`geno_lewm.provenance.hashing.canonical_json_sha256`
- :func:`geno_lewm.provenance.hashing.sha256_bytes`
- :func:`geno_lewm.provenance.hashing.sha256_file`
- :func:`geno_lewm.provenance.commitment.compute_input_commitment`
- :func:`geno_lewm.provenance.receipt.compute_output_commitment`
- :func:`geno_lewm.action.apply.apply_edit`
- :func:`geno_lewm.action.apply.apply_edits`
- :class:`geno_lewm.action.spec.EditSpec` validation

Every test carries the ``bench`` marker and is therefore deselected by
the default ``pytest`` invocation (performance budget: nightly only). To run
locally::

    pytest tests/benchmark/ -m bench --benchmark-only

A 5 % regression detector
(:mod:`tools.ci.perf_regression`) consumes the JSON output and gates
the nightly job.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import pytest

from geno_lewm.action.apply import apply_edit, apply_edits
from geno_lewm.action.spec import EditSpec, EditType, RelEdit
from geno_lewm.provenance.commitment import (
    DtypeConfig,
    PoolingConfig,
    compute_input_commitment,
)
from geno_lewm.provenance.hashing import (
    canonical_json_sha256,
    sha256_bytes,
    sha256_file,
)
from geno_lewm.provenance.receipt import ReceiptOutput, compute_output_commitment

if TYPE_CHECKING:  # pragma: no cover - typing only
    from pytest_benchmark.fixture import BenchmarkFixture

pytestmark = pytest.mark.bench


# ---------------------------------------------------------------------------
# Hashing primitives
# ---------------------------------------------------------------------------


def test_bench_canonical_json_sha256_small(benchmark: BenchmarkFixture) -> None:
    payload = {"chrom": "1", "pos": 1000, "ref": "A", "alt": "T", "extra": list(range(8))}
    benchmark(canonical_json_sha256, payload)


def test_bench_canonical_json_sha256_large(benchmark: BenchmarkFixture) -> None:
    payload = {
        "chrom": "1",
        "pos": 1000,
        "ref": "A",
        "alt": "T",
        "extra": list(range(1024)),
        "monitors": {f"m{i}": float(i) for i in range(64)},
    }
    benchmark(canonical_json_sha256, payload)


def test_bench_sha256_bytes_4kb(benchmark: BenchmarkFixture) -> None:
    blob = b"\x00" * 4096
    benchmark(sha256_bytes, blob)


def test_bench_sha256_bytes_64kb(benchmark: BenchmarkFixture) -> None:
    blob = b"\x00" * 65536
    benchmark(sha256_bytes, blob)


def test_bench_sha256_file_4kb(benchmark: BenchmarkFixture, tmp_path: Path) -> None:
    f = tmp_path / "blob.bin"
    f.write_bytes(b"\x00" * 4096)
    benchmark(sha256_file, f)


# ---------------------------------------------------------------------------
# Commitments
# ---------------------------------------------------------------------------


def test_bench_compute_input_commitment(benchmark: BenchmarkFixture) -> None:
    window = "ACGT" * 1024
    edit = EditSpec(chrom="1", pos=1000, ref="A", alt="T")
    pooling = PoolingConfig(
        state_layer=20, pool_type="centered_mean", pool_radius=8, normalize=True
    )
    dtype = DtypeConfig(encoder_dtype="bf16", predictor_dtype="bf16")

    def workload() -> str:
        return compute_input_commitment(
            reference_window=window,
            edit_spec=edit,
            pooling_config=pooling,
            dtype_config=dtype,
        )

    benchmark(workload)


def test_bench_compute_output_commitment(benchmark: BenchmarkFixture) -> None:
    output = ReceiptOutput(
        sigma_raw=0.73,
        sigma_calibrated=0.81,
        bucket_id="coding.missense",
        confidence=0.94,
        low_confidence=False,
    )
    benchmark(compute_output_commitment, output)


# ---------------------------------------------------------------------------
# Action specs / mutation
# ---------------------------------------------------------------------------


def test_bench_edit_spec_validation(benchmark: BenchmarkFixture) -> None:
    """``EditSpec.__init__`` is the validation hot path for VCF ingestion."""

    def workload() -> EditSpec:
        return EditSpec(chrom="1", pos=1000, ref="A", alt="T")

    benchmark(workload)


def test_bench_apply_edit_snv(benchmark: BenchmarkFixture) -> None:
    window = "ACGT" * 1024
    edit = RelEdit(rel_pos=1000, edit_type=EditType.SNV, ref_bases="A", alt_bases="T")
    benchmark(apply_edit, window, edit)


def test_bench_apply_edits_batch16(benchmark: BenchmarkFixture) -> None:
    # ``ACGT`` repeats every 4 bases — only positions ≡ 0 mod 4 hold an ``A``.
    window = "ACGT" * 1024
    edits = tuple(
        RelEdit(rel_pos=100 + 16 * i, edit_type=EditType.SNV, ref_bases="A", alt_bases="T")
        for i in range(16)
    )
    benchmark(apply_edits, window, edits)


def test_bench_apply_edits_batch64(benchmark: BenchmarkFixture) -> None:
    # Step of 8 keeps positions on the ``A`` column and stays inside the window.
    window = "ACGT" * 1024
    edits = tuple(
        RelEdit(rel_pos=100 + 8 * i, edit_type=EditType.SNV, ref_bases="A", alt_bases="T")
        for i in range(64)
    )
    benchmark(apply_edits, window, edits)
