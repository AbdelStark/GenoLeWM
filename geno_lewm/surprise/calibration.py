# SPDX-License-Identifier: Apache-2.0
"""Calibration-table builder and Parquet IO for surprise-scoring contract."""

from __future__ import annotations

import bisect
import hashlib
import math
import random
import warnings
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from geno_lewm.errors import InputError, RuntimeSetupError, SchemaCompatError
from geno_lewm.surprise.context import (
    DEFAULT_MIN_BUCKET_SIZE,
    backoff_chain,
    select_backoff_bucket,
)

__all__ = [
    "CALIBRATION_SCHEMA_VERSION",
    "DEFAULT_CDF_POINTS",
    "DEFAULT_REFERENCE_PER_BUCKET",
    "LOW_CONFIDENCE_BUCKET_SIZE",
    "CalibrationBucket",
    "CalibrationExample",
    "CalibrationTable",
    "CalibrationWarning",
    "build_calibration_table",
    "read_calibration_table",
    "write_calibration_table",
]


CALIBRATION_SCHEMA_VERSION = "1.0.0"
"""On-disk calibration table schema version."""

DEFAULT_CDF_POINTS = 1_001
"""Number of points in each empirical CDF grid."""

DEFAULT_REFERENCE_PER_BUCKET = 10_000
"""Default maximum number of reference variants sampled per bucket."""

LOW_CONFIDENCE_BUCKET_SIZE = 100
"""Buckets below this size are marked low-confidence by surprise-scoring contract."""


@dataclass(frozen=True, slots=True)
class CalibrationExample:
    """One pre-scored reference variant used to build calibration CDFs."""

    bucket_id: str
    sigma_raw: float

    def __post_init__(self) -> None:
        _require_bucket_id(self.bucket_id)
        _require_sigma_raw(self.sigma_raw)


@dataclass(frozen=True, slots=True)
class CalibrationWarning:
    """Sparse-bucket warning emitted while building a calibration table."""

    bucket_id: str
    resolved_bucket_id: str
    n_calibration: int
    min_bucket_size: int
    low_confidence: bool

    def __post_init__(self) -> None:
        _require_bucket_id(self.bucket_id)
        _require_bucket_id(self.resolved_bucket_id)
        _require_non_negative_int("n_calibration", self.n_calibration)
        _require_positive_int("min_bucket_size", self.min_bucket_size)


@dataclass(frozen=True, slots=True)
class CalibrationBucket:
    """One row in ``calibration.parquet``."""

    bucket_id: str
    n_calibration: int
    cdf: tuple[float, ...]
    sigma_grid: tuple[float, ...]
    back_off_to: str | None = None
    schema_version: str = CALIBRATION_SCHEMA_VERSION

    def __post_init__(self) -> None:
        _require_bucket_id(self.bucket_id)
        _require_positive_int("n_calibration", self.n_calibration)
        _require_grid("cdf", self.cdf, min_value=0.0, max_value=1.0)
        _require_grid("sigma_grid", self.sigma_grid, min_value=0.0)
        if len(self.cdf) != len(self.sigma_grid):
            raise InputError(
                "cdf and sigma_grid must have the same length",
                details={"cdf": len(self.cdf), "sigma_grid": len(self.sigma_grid)},
            )
        if self.back_off_to is not None:
            _require_bucket_id(self.back_off_to)
            valid_parents = set(backoff_chain(self.bucket_id)[1:])
            if self.back_off_to not in valid_parents:
                raise InputError(
                    "back_off_to must be a parent bucket in the fixed backoff chain",
                    details={
                        "bucket_id": self.bucket_id,
                        "back_off_to": self.back_off_to,
                        "valid": sorted(valid_parents),
                    },
                )
        if self.schema_version != CALIBRATION_SCHEMA_VERSION:
            raise SchemaCompatError(
                "calibration bucket schema_version mismatch",
                details={
                    "schema_version": self.schema_version,
                    "expected": CALIBRATION_SCHEMA_VERSION,
                },
            )

    @property
    def confidence(self) -> float:
        """Return surprise-scoring contract confidence from this bucket's row count."""
        return min(self.n_calibration / DEFAULT_MIN_BUCKET_SIZE, 1.0)

    @property
    def low_confidence(self) -> bool:
        """Return true when this bucket is below the low-confidence floor."""
        return self.n_calibration < LOW_CONFIDENCE_BUCKET_SIZE


@dataclass(frozen=True, slots=True)
class CalibrationTable:
    """In-memory representation of ``calibration.parquet``."""

    buckets: tuple[CalibrationBucket, ...]
    warnings: tuple[CalibrationWarning, ...] = ()
    schema_version: str = CALIBRATION_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if not self.buckets:
            raise InputError("calibration table must contain at least one bucket")
        bucket_ids = [bucket.bucket_id for bucket in self.buckets]
        duplicates = sorted(
            {bucket_id for bucket_id in bucket_ids if bucket_ids.count(bucket_id) > 1}
        )
        if duplicates:
            raise InputError(
                "calibration table contains duplicate bucket IDs",
                details={"duplicates": duplicates},
            )
        if self.schema_version != CALIBRATION_SCHEMA_VERSION:
            raise SchemaCompatError(
                "calibration table schema_version mismatch",
                details={
                    "schema_version": self.schema_version,
                    "expected": CALIBRATION_SCHEMA_VERSION,
                },
            )

    def get(self, bucket_id: str) -> CalibrationBucket | None:
        """Return a bucket by ID, or ``None`` if absent."""
        _require_bucket_id(bucket_id)
        by_id = {bucket.bucket_id: bucket for bucket in self.buckets}
        return by_id.get(bucket_id)

    def require(self, bucket_id: str) -> CalibrationBucket:
        """Return a bucket by ID, raising ``InputError`` when absent."""
        bucket = self.get(bucket_id)
        if bucket is None:
            raise InputError(
                "calibration bucket is not present",
                details={"bucket_id": bucket_id},
            )
        return bucket

    def resolve(
        self,
        label_or_bucket: str,
        *,
        min_bucket_size: int = DEFAULT_MIN_BUCKET_SIZE,
    ) -> CalibrationBucket:
        """Resolve a sparse bucket through the table's fixed backoff chain."""
        threshold = _require_positive_int("min_bucket_size", min_bucket_size)
        counts = {bucket.bucket_id: bucket.n_calibration for bucket in self.buckets}
        resolved = select_backoff_bucket(label_or_bucket, counts, min_count=threshold)
        return self.require(resolved)


def build_calibration_table(
    examples: Iterable[CalibrationExample],
    *,
    seed: int = 0,
    per_bucket_sample: int = DEFAULT_REFERENCE_PER_BUCKET,
    grid_size: int = DEFAULT_CDF_POINTS,
    min_bucket_size: int = DEFAULT_MIN_BUCKET_SIZE,
    low_confidence_size: int = LOW_CONFIDENCE_BUCKET_SIZE,
    warn_sparse: bool = True,
) -> CalibrationTable:
    """Build deterministic empirical CDF buckets from pre-scored examples."""
    _require_seed(seed)
    sample_limit = _require_positive_int("per_bucket_sample", per_bucket_sample)
    points = _require_positive_int("grid_size", grid_size)
    if points < 2:
        raise InputError("grid_size must be at least 2", details={"grid_size": points})
    min_size = _require_positive_int("min_bucket_size", min_bucket_size)
    low_size = _require_positive_int("low_confidence_size", low_confidence_size)
    if low_size > min_size:
        raise InputError(
            "low_confidence_size must be <= min_bucket_size",
            details={"low_confidence_size": low_size, "min_bucket_size": min_size},
        )

    aggregated: dict[str, list[float]] = {}
    source_buckets: set[str] = set()
    for example in examples:
        if not isinstance(example, CalibrationExample):
            raise InputError(
                "examples must contain CalibrationExample instances",
                details={"type": type(example).__name__},
            )
        source_buckets.add(example.bucket_id)
        for bucket_id in backoff_chain(example.bucket_id):
            aggregated.setdefault(bucket_id, []).append(float(example.sigma_raw))

    if not source_buckets:
        raise InputError("calibration examples must contain at least one row")

    sampled: dict[str, tuple[float, ...]] = {}
    for bucket_id, values in sorted(aggregated.items()):
        sampled[bucket_id] = _sample_bucket_values(
            values,
            seed=seed,
            bucket_id=bucket_id,
            sample_limit=sample_limit,
        )

    counts = {bucket_id: len(values) for bucket_id, values in sampled.items()}
    buckets: list[CalibrationBucket] = []
    for bucket_id, sampled_values in sorted(sampled.items()):
        cdf, sigma_grid = _empirical_cdf(sampled_values, grid_size=points)
        resolved = select_backoff_bucket(bucket_id, counts, min_count=min_size)
        buckets.append(
            CalibrationBucket(
                bucket_id=bucket_id,
                n_calibration=len(sampled_values),
                cdf=cdf,
                sigma_grid=sigma_grid,
                back_off_to=None if resolved == bucket_id else resolved,
            )
        )

    sparse_warnings = _sparse_warnings(
        sorted(source_buckets),
        counts,
        min_bucket_size=min_size,
        low_confidence_size=low_size,
    )
    if warn_sparse:
        for warning in sparse_warnings:
            warnings.warn(
                "calibration bucket remains sparse after backoff: "
                f"{warning.bucket_id} -> {warning.resolved_bucket_id} "
                f"n={warning.n_calibration} min={warning.min_bucket_size}",
                RuntimeWarning,
                stacklevel=2,
            )

    return CalibrationTable(buckets=tuple(buckets), warnings=sparse_warnings)


def write_calibration_table(table: CalibrationTable, path: str | Path) -> Path:
    """Write a calibration table to ``calibration.parquet``."""
    pa, pq = _require_pyarrow()
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    arrow_table = pa.Table.from_pydict(
        {
            "bucket_id": [bucket.bucket_id for bucket in table.buckets],
            "n_calibration": [bucket.n_calibration for bucket in table.buckets],
            "cdf": [list(bucket.cdf) for bucket in table.buckets],
            "sigma_grid": [list(bucket.sigma_grid) for bucket in table.buckets],
            "back_off_to": [bucket.back_off_to for bucket in table.buckets],
            "schema_version": [bucket.schema_version for bucket in table.buckets],
        },
        schema=_arrow_schema(pa),
    )
    pq.write_table(arrow_table, destination, compression="zstd", compression_level=9)
    return destination


def read_calibration_table(path: str | Path) -> CalibrationTable:
    """Read and validate a calibration Parquet file."""
    _pa, pq = _require_pyarrow()
    source = Path(path)
    try:
        arrow_table = pq.read_table(source)
    except Exception as exc:
        raise SchemaCompatError(
            "calibration table could not be read",
            details={"path": str(source), "error": str(exc)},
        ) from exc

    observed = tuple(arrow_table.column_names)
    expected = _column_names()
    if observed != expected:
        raise SchemaCompatError(
            "calibration table columns do not match the documented schema",
            details={"observed": list(observed), "expected": list(expected)},
        )

    return CalibrationTable(
        buckets=tuple(
            CalibrationBucket(
                bucket_id=str(row["bucket_id"]),
                n_calibration=int(row["n_calibration"]),
                cdf=tuple(float(value) for value in row["cdf"]),
                sigma_grid=tuple(float(value) for value in row["sigma_grid"]),
                back_off_to=None if row["back_off_to"] is None else str(row["back_off_to"]),
                schema_version=str(row["schema_version"]),
            )
            for row in arrow_table.to_pylist()
        )
    )


def _sample_bucket_values(
    values: Sequence[float],
    *,
    seed: int,
    bucket_id: str,
    sample_limit: int,
) -> tuple[float, ...]:
    ordered = tuple(sorted(values))
    if len(ordered) <= sample_limit:
        return ordered
    rng = random.Random(_bucket_seed(seed, bucket_id))
    return tuple(sorted(rng.sample(ordered, sample_limit)))


def _bucket_seed(seed: int, bucket_id: str) -> int:
    digest = hashlib.sha256(f"{seed}:{bucket_id}".encode()).digest()
    return int.from_bytes(digest[:8], "big")


def _empirical_cdf(
    values: Sequence[float], *, grid_size: int
) -> tuple[tuple[float, ...], tuple[float, ...]]:
    if not values:
        raise InputError("empirical CDF requires at least one value")
    ordered = tuple(sorted(values))
    if ordered[0] == ordered[-1]:
        sigma_grid = tuple(ordered[0] for _ in range(grid_size))
    else:
        step = (ordered[-1] - ordered[0]) / (grid_size - 1)
        sigma_grid = tuple(ordered[0] + (idx * step) for idx in range(grid_size))
    cdf = tuple(bisect.bisect_right(ordered, sigma) / len(ordered) for sigma in sigma_grid)
    return cdf, sigma_grid


def _sparse_warnings(
    source_buckets: Sequence[str],
    counts: dict[str, int],
    *,
    min_bucket_size: int,
    low_confidence_size: int,
) -> tuple[CalibrationWarning, ...]:
    sparse: list[CalibrationWarning] = []
    for bucket_id in source_buckets:
        resolved = select_backoff_bucket(bucket_id, counts, min_count=min_bucket_size)
        resolved_count = counts.get(resolved, 0)
        if resolved_count < min_bucket_size:
            sparse.append(
                CalibrationWarning(
                    bucket_id=bucket_id,
                    resolved_bucket_id=resolved,
                    n_calibration=resolved_count,
                    min_bucket_size=min_bucket_size,
                    low_confidence=resolved_count < low_confidence_size,
                )
            )
    return tuple(sparse)


def _require_pyarrow() -> tuple[Any, Any]:
    try:
        import pyarrow as pa
        import pyarrow.parquet as pq
    except ImportError as exc:  # pragma: no cover - exercised in packaging-only envs
        raise RuntimeSetupError(
            "calibration table IO requires pyarrow",
            remediation="install geno-lewm[train] or install pyarrow",
        ) from exc
    return pa, pq


def _arrow_schema(pa: Any) -> Any:
    return pa.schema(
        [
            ("bucket_id", pa.string()),
            ("n_calibration", pa.int64()),
            ("cdf", pa.list_(pa.float32())),
            ("sigma_grid", pa.list_(pa.float32())),
            ("back_off_to", pa.string()),
            ("schema_version", pa.string()),
        ]
    )


def _column_names() -> tuple[str, ...]:
    return (
        "bucket_id",
        "n_calibration",
        "cdf",
        "sigma_grid",
        "back_off_to",
        "schema_version",
    )


def _require_bucket_id(bucket_id: str) -> None:
    if not isinstance(bucket_id, str):
        raise InputError(
            "bucket_id must be a string",
            details={"type": type(bucket_id).__name__},
        )
    backoff_chain(bucket_id)


def _require_sigma_raw(value: float) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float) or not math.isfinite(value):
        raise InputError(
            "sigma_raw must be a finite number",
            details={"value": value, "type": type(value).__name__},
        )
    sigma = float(value)
    if sigma < 0.0:
        raise InputError("sigma_raw must be non-negative", details={"sigma_raw": sigma})
    return sigma


def _require_grid(
    name: str,
    values: tuple[float, ...],
    *,
    min_value: float,
    max_value: float | None = None,
) -> None:
    if not values:
        raise InputError(f"{name} must contain at least one value")
    previous: float | None = None
    for idx, value in enumerate(values):
        if (
            isinstance(value, bool)
            or not isinstance(value, int | float)
            or not math.isfinite(value)
        ):
            raise InputError(
                f"{name} values must be finite numbers",
                details={"index": idx, "value": value, "type": type(value).__name__},
            )
        current = float(value)
        if current < min_value or (max_value is not None and current > max_value):
            raise InputError(
                f"{name} value outside supported range",
                details={"index": idx, "value": current},
            )
        if previous is not None and current < previous:
            raise InputError(
                f"{name} values must be non-decreasing",
                details={"index": idx, "previous": previous, "value": current},
            )
        previous = current


def _require_positive_int(name: str, value: int) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise InputError(
            f"{name} must be a positive integer",
            details={name: value, "type": type(value).__name__},
        )
    return value


def _require_non_negative_int(name: str, value: int) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise InputError(
            f"{name} must be a non-negative integer",
            details={name: value, "type": type(value).__name__},
        )
    return value


def _require_seed(seed: int) -> int:
    return _require_non_negative_int("seed", seed)
