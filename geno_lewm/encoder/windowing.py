# SPDX-License-Identifier: Apache-2.0
"""Window extraction and Carbon tokenizer wrapping.

Defined by RFC-0002 §3.2 and the cache invariant INV-DATA-2. The
helpers in this module are deliberately pure Python: they canonicalize
DNA windows, extract fixed-size windows around an optional edit locus,
right-pad short source sequence with ``A``, and produce the
``<dna>...</dna>`` string consumed by Carbon's tokenizer.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass

from geno_lewm.errors import InputError, OutOfWindowError

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


CARBON_DNA_OPEN_TAG = "<dna>"
CARBON_DNA_CLOSE_TAG = "</dna>"
CARBON_TOKEN_BP = 6
DEFAULT_WINDOW_BP = 12_288
SUPPORTED_WINDOW_BP = (4_096, DEFAULT_WINDOW_BP, 24_576)
DEFAULT_EDIT_MARGIN_BP = 64

_VALID_DNA_BASES = frozenset("ACGTN")
_PAD_BASE = "A"


@dataclass(frozen=True, slots=True)
class ExtractedWindow:
    """A fixed-size DNA window plus its source-coordinate metadata.

    ``start_bp`` and ``end_bp`` are 0-based half-open coordinates in
    the caller's source coordinate system. ``end_bp - start_bp`` always
    equals ``window_bp`` even when the sequence had to be right-padded
    past the available source bases; ``pad_right_bp`` records how many
    trailing ``A`` bases were introduced.
    """

    sequence: str
    start_bp: int
    end_bp: int
    window_bp: int
    edit_locus: int | None = None
    relative_edit_locus: int | None = None
    pad_right_bp: int = 0

    @property
    def untargeted(self) -> bool:
        """Return true when the window was not centered on an edit."""
        return self.edit_locus is None

    @property
    def sha256(self) -> bytes:
        """SHA-256 digest of the canonical window sequence."""
        return window_sha256(self.sequence)

    def as_tokenizer_input(self) -> str:
        """Return the Carbon tokenizer input string for this window."""
        return wrap_dna_for_tokenizer(self.sequence)


def canonicalize_dna(sequence: str) -> str:
    """Return uppercase DNA after validating the supported alphabet.

    The cache hash invariant is based on uppercased window content, so
    callers can hash raw source slices and already-canonical windows
    interchangeably. ``N`` is accepted because reference FASTA and
    edited windows may contain masked bases.
    """
    if not isinstance(sequence, str):
        raise InputError(
            "DNA sequence must be a string",
            details={"type": type(sequence).__name__},
        )
    canonical = sequence.upper()
    bad = sorted(set(canonical) - _VALID_DNA_BASES)
    if bad:
        raise InputError(
            "DNA sequence contains unsupported base(s)",
            details={"bad_chars": bad},
            remediation="provide only A, C, G, T, or N bases",
        )
    return canonical


def window_sha256(sequence: str) -> bytes:
    """Return SHA-256 bytes for the canonicalized DNA sequence."""
    canonical = canonicalize_dna(sequence)
    return hashlib.sha256(canonical.encode("ascii")).digest()


def extract_window(
    source_sequence: str,
    *,
    edit_locus: int | None = None,
    window_bp: int = DEFAULT_WINDOW_BP,
    assume_canonical: bool = False,
) -> ExtractedWindow:
    """Extract a supported-width DNA window from ``source_sequence``.

    ``edit_locus`` is a 0-based offset in ``source_sequence``. When it
    is supplied the window is centered on that locus unless clamped by
    source boundaries. When omitted, the source midpoint is used. If
    the source is shorter than the requested window or the selected
    interval extends past the right edge, trailing ``A`` bases are
    appended per Carbon's tokenizer convention.

    Set ``assume_canonical`` when ``source_sequence`` is already uppercase,
    validated DNA (e.g. a contig from a loaded reference FASTA) to skip the
    O(len) re-validation. Re-validating a whole chromosome once per variant
    otherwise dominates VCF scoring wall-clock.
    """
    _validate_window_bp(window_bp)
    source = source_sequence if assume_canonical else canonicalize_dna(source_sequence)
    if not source:
        raise InputError("source_sequence must be non-empty")

    source_len = len(source)
    center = _center_for(source_len, edit_locus)
    start_bp = _centered_start(source_len, center, window_bp)
    end_bp = start_bp + window_bp

    observed = source[start_bp : min(end_bp, source_len)]
    pad_right_bp = window_bp - len(observed)
    window = observed + (_PAD_BASE * pad_right_bp)

    relative_edit_locus: int | None = None
    if edit_locus is not None:
        relative_edit_locus = edit_locus - start_bp

    return ExtractedWindow(
        sequence=window,
        start_bp=start_bp,
        end_bp=end_bp,
        window_bp=window_bp,
        edit_locus=edit_locus,
        relative_edit_locus=relative_edit_locus,
        pad_right_bp=pad_right_bp,
    )


def pad_for_carbon_tokenizer(sequence: str, *, token_bp: int = CARBON_TOKEN_BP) -> str:
    """Right-pad canonical DNA to Carbon's token multiple."""
    if not isinstance(token_bp, int) or isinstance(token_bp, bool) or token_bp <= 0:
        raise InputError(
            "token_bp must be a positive integer",
            details={"token_bp": token_bp, "type": type(token_bp).__name__},
        )
    canonical = canonicalize_dna(sequence)
    remainder = len(canonical) % token_bp
    if remainder == 0:
        return canonical
    return canonical + (_PAD_BASE * (token_bp - remainder))


def wrap_dna_for_tokenizer(sequence: str) -> str:
    """Return ``<dna>...</dna>`` input with Carbon-compatible padding."""
    padded = pad_for_carbon_tokenizer(sequence)
    return f"{CARBON_DNA_OPEN_TAG}{padded}{CARBON_DNA_CLOSE_TAG}"


def _validate_window_bp(window_bp: int) -> None:
    if not isinstance(window_bp, int) or isinstance(window_bp, bool):
        raise InputError(
            "window_bp must be an integer",
            details={"window_bp": window_bp, "type": type(window_bp).__name__},
        )
    if window_bp not in SUPPORTED_WINDOW_BP:
        raise InputError(
            "unsupported window length",
            details={"window_bp": window_bp, "supported": list(SUPPORTED_WINDOW_BP)},
            remediation="use one of 4096, 12288, or 24576 bp",
        )


def _center_for(source_len: int, edit_locus: int | None) -> int:
    if edit_locus is None:
        return source_len // 2
    if not isinstance(edit_locus, int) or isinstance(edit_locus, bool):
        raise InputError(
            "edit_locus must be an integer offset",
            details={"edit_locus": edit_locus, "type": type(edit_locus).__name__},
        )
    if edit_locus < 0 or edit_locus >= source_len:
        raise OutOfWindowError(
            "edit_locus falls outside source_sequence",
            details={"edit_locus": edit_locus, "source_len": source_len},
            remediation="provide a 0-based locus inside the source sequence",
        )
    return edit_locus


def _centered_start(source_len: int, center: int, window_bp: int) -> int:
    if source_len <= window_bp:
        return 0
    proposed = center - (window_bp // 2)
    return min(max(proposed, 0), source_len - window_bp)
