# SPDX-License-Identifier: Apache-2.0
"""State-encoder input preparation helpers.

Issue #33 lands the pure-Python windowing and Carbon-tokenizer input
surface. The Carbon model wrapper itself remains a separate issue
because it depends on the optional ML runtime.
"""

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
    "CARBON_DNA_CLOSE_TAG",
    "CARBON_DNA_OPEN_TAG",
    "CARBON_TOKEN_BP",
    "DEFAULT_EDIT_MARGIN_BP",
    "DEFAULT_WINDOW_BP",
    "SUPPORTED_WINDOW_BP",
    "ExtractedWindow",
    "canonicalize_dna",
    "extract_window",
    "pad_for_carbon_tokenizer",
    "window_sha256",
    "wrap_dna_for_tokenizer",
]
