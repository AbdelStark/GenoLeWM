# SPDX-License-Identifier: Apache-2.0
"""State-encoder input preparation helpers.

Issue #33 lands the pure-Python windowing and Carbon-tokenizer input
surface. The Carbon model wrapper itself remains a separate issue
because it depends on the optional ML runtime.
"""

from geno_lewm.encoder.cache import (
    CACHE_SCHEMA_VERSION,
    INDEX_DB_NAME,
    CacheReindexReport,
    CacheRepairReport,
    WindowCacheKey,
    WindowCacheRecord,
    default_cache_dir,
    read_embedding,
    reindex_cache,
    repair_cache,
    shard_path_for,
    write_shard,
)
from geno_lewm.encoder.pooling import (
    DEFAULT_POOL_RADIUS_TOKENS,
    POOL_CENTERED_MEAN,
    POOL_GLOBAL_MEAN,
    SUPPORTED_POOL_TYPES,
    PoolingResult,
    centered_mean,
    global_mean,
    pool_hidden_states,
)
from geno_lewm.encoder.windowing import (
    CARBON_DNA_CLOSE_TAG,
    CARBON_DNA_OPEN_TAG,
    CARBON_TOKEN_BP,
    DEFAULT_EDIT_MARGIN_BP,
    DEFAULT_WINDOW_BP,
    SUPPORTED_WINDOW_BP,
    ExtractedWindow,
    canonicalize_dna,
    extract_window,
    pad_for_carbon_tokenizer,
    window_sha256,
    wrap_dna_for_tokenizer,
)

__all__ = [
    "CACHE_SCHEMA_VERSION",
    "CARBON_DNA_CLOSE_TAG",
    "CARBON_DNA_OPEN_TAG",
    "CARBON_TOKEN_BP",
    "DEFAULT_EDIT_MARGIN_BP",
    "DEFAULT_POOL_RADIUS_TOKENS",
    "DEFAULT_WINDOW_BP",
    "INDEX_DB_NAME",
    "POOL_CENTERED_MEAN",
    "POOL_GLOBAL_MEAN",
    "SUPPORTED_POOL_TYPES",
    "SUPPORTED_WINDOW_BP",
    "CacheReindexReport",
    "CacheRepairReport",
    "ExtractedWindow",
    "PoolingResult",
    "WindowCacheKey",
    "WindowCacheRecord",
    "canonicalize_dna",
    "centered_mean",
    "default_cache_dir",
    "extract_window",
    "global_mean",
    "pad_for_carbon_tokenizer",
    "pool_hidden_states",
    "read_embedding",
    "reindex_cache",
    "repair_cache",
    "shard_path_for",
    "window_sha256",
    "wrap_dna_for_tokenizer",
    "write_shard",
]
