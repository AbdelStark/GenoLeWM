# SPDX-License-Identifier: Apache-2.0
"""Training-path benchmarks (performance budget).

Phase 1 surfaces today: the **action mutation path** is the only piece
of the training data pipeline implemented in the package (the encoder,
predictor, and trainer land in issues #32, #41, #44). We benchmark:

- ``apply_edit`` over a single SNV.
- ``apply_edits`` over batches of synthetic edits at increasing batch
  size.

Full training-step throughput (encoder forward + predictor forward +
loss + backward) will be added under ``training.step`` once those
modules land; the harness library is reusable as-is.
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
from geno_lewm.action.apply import apply_edit, apply_edits
from geno_lewm.action.spec import EditType, RelEdit


def _bench_apply_single(iters: int, warmup: int) -> BenchResult:
    window = "ACGT" * 1024
    edit = RelEdit(rel_pos=1000, edit_type=EditType.SNV, ref_bases="A", alt_bases="T")

    def workload() -> None:
        apply_edit(window, edit)

    return time_callable(
        "training.apply_edit",
        workload,
        iters=iters,
        warmup=warmup,
        extra={"window_bytes": str(len(window))},
    )


def _bench_apply_batch(iters: int, warmup: int, batch_size: int) -> BenchResult:
    window = "ACGT" * 1024
    # Space edits far enough apart to avoid overlap; window has 4096 bases.
    edits = tuple(
        RelEdit(rel_pos=100 + 16 * i, edit_type=EditType.SNV, ref_bases="A", alt_bases="T")
        for i in range(batch_size)
    )

    def workload() -> None:
        apply_edits(window, edits)

    return time_callable(
        f"training.apply_edits_batch{batch_size}",
        workload,
        iters=iters,
        warmup=warmup,
        extra={
            "window_bytes": str(len(window)),
            "batch_size": str(batch_size),
        },
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="bench.training",
        description="Training-path benchmarks (performance budget).",
    )
    parser.add_argument("--iters", type=int, default=200)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--no-write", action="store_true")
    parser.add_argument(
        "--batch-sizes",
        type=int,
        nargs="+",
        default=[1, 4, 16],
        help="batch sizes to sweep (default: 1 4 16)",
    )
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_RESULTS_DIR)
    args = parser.parse_args(argv)

    results = [_bench_apply_single(args.iters, args.warmup)]
    results.extend(_bench_apply_batch(args.iters, args.warmup, b) for b in args.batch_sizes)
    for r in results:
        report_to_stdout(r)
        if not args.no_write:
            path = write_result(r, out_dir=args.out_dir)
            print(f"  wrote {path}")
    for r in results:
        json.dumps(r.to_json())
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
