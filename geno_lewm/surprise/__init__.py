# SPDX-License-Identifier: Apache-2.0
"""Surprise-scoring helpers for RFC-0009.

The model-dependent raw scorer is still layered on top of the predictor
runtime. This package currently exposes the pure-Python context
stratification surface used by calibration buckets.
"""

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
    "DEFAULT_GC_HIGH_CUTOFF",
    "DEFAULT_GC_LOW_CUTOFF",
    "DEFAULT_MIN_BUCKET_SIZE",
    "GC_BINS",
    "REGION_CLASSES",
    "REPEAT_CLASSES",
    "UNKNOWN_BUCKET_ID",
    "ContextLabel",
    "backoff_chain",
    "classify_context",
    "classify_gc_bin",
    "classify_region",
    "classify_repeat",
    "gc_fraction",
    "make_bucket_id",
    "select_backoff_bucket",
]
