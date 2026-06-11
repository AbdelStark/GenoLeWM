# SPDX-License-Identifier: Apache-2.0
"""Measure release dataset tuple-builder throughput."""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections.abc import Mapping
from pathlib import Path
from typing import Final

from geno_lewm.data import (
    SOURCE_CLINVAR,
    SOURCE_GNOMAD_COMMON,
    SOURCE_SYNTHETIC_INDEL,
    SOURCE_SYNTHETIC_SNV,
    GenoLeWMDataset,
    synthetic_indel_provider,
    synthetic_snv_provider,
    variant_provider,
)
from geno_lewm.errors import GenoLeWMError, InputError, exit_code_for
from geno_lewm.provenance import sha256_file
from geno_lewm.training.real import (
    _dataset_fallback_sources,
    _dataset_files,
    _load_clinvar_edits,
    _load_dataset_manifest,
    _load_gnomad_edits,
    _load_windows,
)

SCHEMA_VERSION: Final = "1.0.0"
GENERATED_BY: Final = "tools.data.tuple_throughput"


def measure_tuple_throughput(
    *,
    dataset_dir: Path,
    samples: int,
    seed: int = 0,
    min_tuples_per_second: float | None = None,
) -> dict[str, object]:
    """Measure pure tuple-builder throughput for a packaged release dataset."""
    _require_positive_int("samples", samples)
    if min_tuples_per_second is not None:
        _require_positive_float("min_tuples_per_second", min_tuples_per_second)
    manifest_path = dataset_dir / "dataset_manifest.json"
    manifest = _load_dataset_manifest(dataset_dir)
    dataset_snapshot_id = _required_text(manifest, "snapshot_id")
    files = _dataset_files(manifest)
    windows = tuple(_load_windows(dataset_dir, files))
    if not windows:
        raise InputError("tuple throughput requires at least one active training window")
    gnomad_edits = tuple(_load_gnomad_edits(dataset_dir, files))
    clinvar_edits = tuple(_load_clinvar_edits(dataset_dir, files))
    if not gnomad_edits:
        raise InputError("tuple throughput requires at least one gnomAD edit")

    dataset = GenoLeWMDataset(
        windows,
        {
            SOURCE_GNOMAD_COMMON: variant_provider(gnomad_edits),
            SOURCE_SYNTHETIC_SNV: synthetic_snv_provider,
            SOURCE_SYNTHETIC_INDEL: synthetic_indel_provider,
            SOURCE_CLINVAR: variant_provider(clinvar_edits),
        },
        seed=seed,
        fallback_sources=_dataset_fallback_sources(windows),
    )

    iterator = dataset.iter_with_source_windows()
    start = time.perf_counter()
    observed = 0
    for _ in range(samples):
        try:
            next(iterator)
        except StopIteration as exc:
            raise InputError(
                "tuple dataset exhausted before requested sample count",
                details={"requested": samples, "observed": observed, "windows": len(windows)},
            ) from exc
        observed += 1
    elapsed = max(time.perf_counter() - start, 1e-9)
    tuples_per_second = observed / elapsed
    payload: dict[str, object] = {
        "schema_version": SCHEMA_VERSION,
        "generated_by": GENERATED_BY,
        "dataset_dir": dataset_dir.name,
        "dataset_snapshot_id": dataset_snapshot_id,
        "dataset_manifest": _artifact_identity(manifest_path),
        "seed": seed,
        "requested_samples": samples,
        "samples": observed,
        "elapsed_seconds": elapsed,
        "tuples_per_second": tuples_per_second,
        "windows": len(windows),
        "gnomad_edits": len(gnomad_edits),
        "clinvar_edits": len(clinvar_edits),
    }
    if min_tuples_per_second is not None:
        payload["min_tuples_per_second"] = min_tuples_per_second
        payload["passed_min_tuples_per_second"] = tuples_per_second >= min_tuples_per_second
    return payload


def _artifact_identity(path: Path) -> dict[str, object]:
    return {
        "path": path.name,
        "sha256": sha256_file(path),
        "size_bytes": path.stat().st_size,
    }


def _required_text(payload: Mapping[str, object], key: str) -> str:
    value = payload.get(key)
    if not isinstance(value, str) or not value.strip():
        raise InputError(f"{key} must be a non-empty string")
    return value.strip()


def _require_positive_int(name: str, value: int) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise InputError(f"{name} must be a positive integer", details={name: value})


def _require_positive_float(name: str, value: float) -> None:
    if isinstance(value, bool) or not isinstance(value, int | float) or value <= 0:
        raise InputError(f"{name} must be positive", details={name: value})


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--samples", type=int, default=4096)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--min-tuples-per-second", type=float)
    parser.add_argument("--output", type=Path)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    try:
        payload = measure_tuple_throughput(
            dataset_dir=args.dataset_dir,
            samples=args.samples,
            seed=args.seed,
            min_tuples_per_second=args.min_tuples_per_second,
        )
        if args.output is not None:
            _write_report(args.output, payload)
        min_rate = payload.get("min_tuples_per_second")
        tuples_per_second = payload["tuples_per_second"]
        if not isinstance(tuples_per_second, int | float):
            raise InputError("tuple throughput payload is malformed")
        if (
            isinstance(min_rate, int | float)
            and not isinstance(min_rate, bool)
            and tuples_per_second < min_rate
        ):
            raise InputError(
                "tuple throughput is below the release threshold",
                details={
                    "observed": tuples_per_second,
                    "minimum": min_rate,
                    "samples": payload["samples"],
                    "output": str(args.output) if args.output is not None else None,
                },
            )
    except GenoLeWMError as exc:
        sys.stderr.write(f"error: {exc}\n")
        if exc.details:
            sys.stderr.write(json.dumps(exc.details, sort_keys=True) + "\n")
        return exit_code_for(exc)
    sys.stdout.write(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    return 0


def _write_report(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
