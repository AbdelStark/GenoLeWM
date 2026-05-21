# SPDX-License-Identifier: Apache-2.0
"""Inference-path benchmarks (RFC-0016 §3.4).

Phase 1 surfaces today: the **receipt-mode verifier** is the only
inference hot path in the package. We time the two content-addressed
commitments that every receipt round-trip relies on:

- ``compute_input_commitment`` — canonical-JSON SHA-256 over the input
  payload (window + edit + pooling + dtype).
- ``compute_output_commitment`` — canonical-JSON SHA-256 over the
  output block (sigma + bucket + confidence).

These are the hot path for the verify CLI and the future runtime
attestation receipts.

Predictor / model-forward latency will be added under names like
``inference.predictor.forward`` once the encoder + predictor modules land
(issues #32 and #41); the harness library is reusable as-is.

Usage::

    python -m bench.inference                  # default: 200 iters, write results
    python -m bench.inference --iters 50       # quick smoke
    python -m bench.inference --no-write       # report-only

Result files land at ``bench/results/<machine>/inference.<workload>.json``.
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path

from bench._harness import (
    DEFAULT_RESULTS_DIR,
    BenchResult,
    report_to_stdout,
    time_callable,
    write_result,
)
from geno_lewm.action.spec import EditSpec
from geno_lewm.attestation.commitment import (
    DtypeConfig,
    PoolingConfig,
    compute_input_commitment,
)
from geno_lewm.attestation.receipt import ReceiptOutput, compute_output_commitment


def _bench_input_commitment(iters: int, warmup: int) -> BenchResult:
    window = "ACGT" * 1024  # 4 kB synthetic window
    edit = EditSpec(chrom="1", pos=1000, ref="A", alt="T")
    pooling = PoolingConfig(
        state_layer=20, pool_type="centered_mean", pool_radius=8, normalize=True
    )
    dtype = DtypeConfig(encoder_dtype="bf16", predictor_dtype="bf16")

    def workload() -> None:
        compute_input_commitment(
            reference_window=window,
            edit_spec=edit,
            pooling_config=pooling,
            dtype_config=dtype,
        )

    return time_callable(
        "inference.input_commitment",
        workload,
        iters=iters,
        warmup=warmup,
        dtype="bf16",
        extra={"window_bytes": str(len(window))},
    )


def _bench_output_commitment(iters: int, warmup: int) -> BenchResult:
    output = ReceiptOutput(
        sigma_raw=0.7321,
        sigma_calibrated=0.812,
        bucket_id="coding.missense",
        confidence=0.94,
        low_confidence=False,
    )

    def workload() -> None:
        compute_output_commitment(output)

    return time_callable(
        "inference.output_commitment",
        workload,
        iters=iters,
        warmup=warmup,
        dtype="bf16",
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="bench.inference",
        description="Inference-path benchmarks (RFC-0016 §3.4).",
    )
    parser.add_argument("--iters", type=int, default=200)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--no-write", action="store_true", help="report-only; do not persist")
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=DEFAULT_RESULTS_DIR,
        help="results directory root (default: bench/results)",
    )
    args = parser.parse_args(argv)

    results = [
        _bench_input_commitment(args.iters, args.warmup),
        _bench_output_commitment(args.iters, args.warmup),
    ]
    for r in results:
        report_to_stdout(r)
        if not args.no_write:
            path = write_result(r, out_dir=args.out_dir)
            print(f"  wrote {path}")
    # Round-trip JSON to confirm serialisation is well-formed.
    for r in results:
        json.dumps(r.to_json())
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
