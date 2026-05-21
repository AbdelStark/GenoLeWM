# SPDX-License-Identifier: Apache-2.0
"""Planning-loop benchmarks (RFC-0016 §3.4).

The planning loop (CEM solver + ActionSampler + cost function) is part
of Phase 2 — issues #59, #60, #61. Until those modules land, this
script is a placeholder that documents the future workloads and emits a
``status=not_implemented`` result so the regression detector can see a
real timeline of when ``planning.*`` benchmarks start appearing.

Planned workloads (RFC-0008 §3.4):

- ``planning.cem_iter.k1`` — one CEM iteration with 1-edit horizon.
- ``planning.cem_iter.k3`` — one CEM iteration with 3-edit horizon.
- ``planning.cem_iter.k10`` — one CEM iteration with 10-edit horizon.

Exit code is always 0 in placeholder mode so the script can be wired
into the nightly job today without spurious red.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path

from bench._harness import (
    DEFAULT_RESULTS_DIR,
    BenchResult,
    collect_metadata,
    write_result,
)


def _placeholder_result(name: str) -> BenchResult:
    """A ``status=not_implemented`` result with empty samples."""
    metadata = collect_metadata(
        dtype="n/a",
        extra={"status": "not_implemented", "rfc": "RFC-0008", "issues": "#59 #60 #61"},
    )
    return BenchResult(
        name=name,
        iters=0,
        warmup=0,
        samples_ns=(),
        median_ns=0,
        p25_ns=0,
        p75_ns=0,
        iqr_ns=0,
        metadata=metadata,
    )


_PLANNED_BENCHMARKS: tuple[str, ...] = (
    "planning.cem_iter.k1",
    "planning.cem_iter.k3",
    "planning.cem_iter.k10",
)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="bench.planning",
        description="Planning-loop benchmarks (RFC-0016 §3.4) — placeholder.",
    )
    parser.add_argument("--no-write", action="store_true")
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_RESULTS_DIR)
    args = parser.parse_args(argv)

    print(
        "[bench] planning module not yet implemented; "
        "writing placeholder results so the timeline tracks status changes.",
        file=sys.stderr,
    )

    for name in _PLANNED_BENCHMARKS:
        result = _placeholder_result(name)
        print(f"[bench] {name}: status=not_implemented")
        if not args.no_write:
            path = write_result(result, out_dir=args.out_dir)
            print(f"  wrote {path}")
        json.dumps(result.to_json())
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
