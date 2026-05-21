# SPDX-License-Identifier: Apache-2.0
"""Profiler entry points (RFC-0016 §3.6).

This script documents the canonical profiler invocations for the
package. It does *not* run anything by default — profiling is a
developer-driven exercise, not a CI gate.

Canonical tools (RFC-0016 §3.6):

| Concern   | Tool                                |
|-----------|-------------------------------------|
| Python CPU| py-spy (sampled), cProfile (deep)   |
| GPU       | torch.profiler, NVIDIA Nsight       |
| Memory    | tracemalloc, nvidia-smi, MST        |
| Microbench| pytest-benchmark                    |

Print the canonical command lines so a contributor can copy-paste::

    python -m bench.profile               # print canonical commands
    python -m bench.profile --run-cprofile-on bench.inference   # run cProfile

The ``--run-cprofile-on`` mode is a thin convenience wrapper around the
stdlib ``cProfile`` module.
"""

from __future__ import annotations

import argparse
import cProfile
import io
import pstats
import runpy
from collections.abc import Sequence

_CANONICAL_COMMANDS: tuple[tuple[str, str], ...] = (
    (
        "py-spy (CPU sampling)",
        "py-spy record -o profile.svg -- python -m bench.inference --iters 500 --no-write",
    ),
    (
        "cProfile (deep CPU)",
        "python -m cProfile -s cumtime -o inference.prof -m bench.inference --iters 200 --no-write",
    ),
    (
        "tracemalloc (Python allocations)",
        "PYTHONTRACEMALLOC=1 python -m bench.inference --iters 100 --no-write",
    ),
    (
        "torch.profiler (Phase 2+, GPU)",
        "python -c 'import torch.profiler; ...'  # see RFC-0016 §3.6",
    ),
)


def _print_canonical_commands() -> None:
    print("# Canonical profiler invocations (RFC-0016 §3.6)\n")
    for label, cmd in _CANONICAL_COMMANDS:
        print(f"## {label}\n  {cmd}\n")


def _run_cprofile_on(target: str) -> int:
    """Run ``runpy.run_module(target)`` under cProfile and print top callers."""
    profiler = cProfile.Profile()
    profiler.enable()
    try:
        runpy.run_module(target, run_name="__main__", alter_sys=True)
    except SystemExit as exc:
        # Per-target main() returns int via SystemExit; preserve the code.
        rc = int(exc.code) if isinstance(exc.code, int) else 1
        profiler.disable()
        _print_stats(profiler)
        return rc
    profiler.disable()
    _print_stats(profiler)
    return 0


def _print_stats(profiler: cProfile.Profile) -> None:
    buf = io.StringIO()
    stats = pstats.Stats(profiler, stream=buf).sort_stats("cumulative")
    stats.print_stats(30)
    print("\n# cProfile top 30 (sorted by cumulative time)\n")
    print(buf.getvalue())


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="bench.profile",
        description="Profiler entry points (RFC-0016 §3.6).",
    )
    parser.add_argument(
        "--run-cprofile-on",
        metavar="MODULE",
        default=None,
        help="run cProfile on `python -m MODULE` and print top callers",
    )
    args = parser.parse_args(argv)

    if args.run_cprofile_on:
        return _run_cprofile_on(args.run_cprofile_on)
    _print_canonical_commands()
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
