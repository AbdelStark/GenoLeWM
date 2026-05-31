# SPDX-License-Identifier: Apache-2.0
"""Surprise-scoring helpers for RFC-0009.

The model-dependent raw scorer is still layered on top of the predictor
runtime. This package currently exposes the pure-Python context
stratification surface used by calibration buckets.
"""

from geno_lewm.surprise.calibration import (
    CALIBRATION_SCHEMA_VERSION,
    DEFAULT_CDF_POINTS,
    DEFAULT_REFERENCE_PER_BUCKET,
    LOW_CONFIDENCE_BUCKET_SIZE,
    CalibrationBucket,
    CalibrationExample,
    CalibrationTable,
    CalibrationWarning,
    build_calibration_table,
    read_calibration_table,
    write_calibration_table,
)
from geno_lewm.surprise.context import (
    DEFAULT_GC_HIGH_CUTOFF,
    DEFAULT_GC_LOW_CUTOFF,
    DEFAULT_MIN_BUCKET_SIZE,
    GC_BINS,
    REGION_CLASSES,
    REPEAT_CLASSES,
    UNKNOWN_BUCKET_ID,
    ContextLabel,
    backoff_chain,
    classify_context,
    classify_gc_bin,
    classify_region,
    classify_repeat,
    gc_fraction,
    make_bucket_id,
    select_backoff_bucket,
)

__all__ = [
    "CALIBRATION_SCHEMA_VERSION",
    "DEFAULT_CDF_POINTS",
    "DEFAULT_GC_HIGH_CUTOFF",
    "DEFAULT_GC_LOW_CUTOFF",
    "DEFAULT_MIN_BUCKET_SIZE",
    "DEFAULT_REFERENCE_PER_BUCKET",
    "GC_BINS",
    "LOW_CONFIDENCE_BUCKET_SIZE",
    "REGION_CLASSES",
    "REPEAT_CLASSES",
    "UNKNOWN_BUCKET_ID",
    "CalibrationBucket",
    "CalibrationExample",
    "CalibrationTable",
    "CalibrationWarning",
    "ContextLabel",
    "backoff_chain",
    "build_calibration_table",
    "classify_context",
    "classify_gc_bin",
    "classify_region",
    "classify_repeat",
    "gc_fraction",
    "make_bucket_id",
    "read_calibration_table",
    "select_backoff_bucket",
    "write_calibration_table",
]
