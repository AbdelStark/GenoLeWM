# SPDX-License-Identifier: Apache-2.0
"""Shared JSON-report helpers for local data preparation CLIs."""

from __future__ import annotations

import shlex
import sys
from pathlib import Path

from geno_lewm.provenance import sha256_file


def augment_prepare_report(
    payload: dict[str, object],
    *,
    command: str,
    args: list[str],
    input_vcf: Path,
    output_path: Path,
    elapsed_seconds: float,
) -> dict[str, object]:
    """Add command, artifact identity, and process-runtime evidence."""
    enriched = dict(payload)
    enriched["command"] = shlex.join([command, *args])
    enriched["input_vcf"] = _file_identity(input_vcf)
    enriched["output_parquet"] = _file_identity(output_path)
    enriched["runtime"] = {
        "elapsed_seconds": round(max(elapsed_seconds, 0.0), 6),
        "process_peak_rss_bytes": _process_peak_rss_bytes(),
        "peak_memory_note": (
            "Process-level ru_maxrss reported by the host OS; use a dedicated "
            "wrapper for isolated peak-memory benchmarks."
        ),
    }
    return enriched


def _file_identity(path: Path) -> dict[str, object]:
    return {
        "path": str(path),
        "sha256": sha256_file(path),
        "size_bytes": path.stat().st_size,
    }


def _process_peak_rss_bytes() -> int | None:
    try:
        import resource
    except ImportError:  # pragma: no cover - Windows fallback
        return None
    peak_rss = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    if peak_rss <= 0:
        return None
    if _peak_rss_is_bytes():
        return peak_rss
    return peak_rss * 1024


def _peak_rss_is_bytes() -> bool:
    return sys.platform == "darwin"
