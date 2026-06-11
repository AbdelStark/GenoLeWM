# SPDX-License-Identifier: Apache-2.0
"""Context stratification labels for surprise-scoring contract calibration buckets."""

from __future__ import annotations

import math
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from geno_lewm.encoder.windowing import canonicalize_dna
from geno_lewm.errors import InputError

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


REGION_CLASSES: tuple[str, ...] = (
    "coding_synonymous",
    "coding_missense",
    "coding_nonsense",
    "splice",
    "utr5",
    "utr3",
    "intron",
    "promoter",
    "enhancer",
    "intergenic",
    "other",
)
"""Canonical surprise-scoring contract ``region_class`` values."""

GC_BINS: tuple[str, ...] = ("low", "mid", "high")
"""Canonical surprise-scoring contract ``gc_bin`` values."""

REPEAT_CLASSES: tuple[str, ...] = (
    "none",
    "simple",
    "low_complexity",
    "transposon",
    "segmental_dup",
)
"""Canonical surprise-scoring contract ``repeat_class`` values."""

UNKNOWN_BUCKET_ID = "*"
"""Catch-all calibration bucket reached after every parent bucket is sparse."""

DEFAULT_GC_LOW_CUTOFF: float = 1.0 / 3.0
"""Inclusive lower-tercile GC cutoff used when no fitted cutpoints are supplied."""

DEFAULT_GC_HIGH_CUTOFF: float = 2.0 / 3.0
"""Inclusive upper-tercile GC cutoff used when no fitted cutpoints are supplied."""

DEFAULT_MIN_BUCKET_SIZE = 1_000
"""surprise-scoring contract default threshold for a well-populated calibration bucket."""

_REGION_ALIAS_GROUPS: tuple[tuple[str, tuple[str, ...]], ...] = (
    (
        "splice",
        (
            "splice",
            "splice_acceptor_variant",
            "splice_donor_variant",
            "splice_region_variant",
            "splice_site",
        ),
    ),
    (
        "coding_nonsense",
        (
            "coding_nonsense",
            "nonsense",
            "stop_gained",
            "stop_lost",
            "stop_retained_variant",
            "start_lost",
            "frameshift_variant",
        ),
    ),
    (
        "coding_missense",
        (
            "coding_missense",
            "missense",
            "missense_variant",
            "protein_altering_variant",
            "inframe_deletion",
            "inframe_insertion",
        ),
    ),
    (
        "coding_synonymous",
        (
            "coding_synonymous",
            "synonymous",
            "synonymous_variant",
            "silent",
            "stop_retained",
        ),
    ),
    ("utr5", ("utr5", "5_prime_utr_variant", "five_prime_utr_variant", "5_utr", "utr_5")),
    ("utr3", ("utr3", "3_prime_utr_variant", "three_prime_utr_variant", "3_utr", "utr_3")),
    (
        "promoter",
        (
            "promoter",
            "promoter_region",
            "promoter_variant",
            "upstream_gene_variant",
            "tf_binding_site_variant",
        ),
    ),
    ("enhancer", ("enhancer", "enhancer_region", "enhancer_variant", "regulatory_region_variant")),
    ("intron", ("intron", "intron_variant", "non_coding_transcript_variant")),
    ("intergenic", ("intergenic", "intergenic_variant", "downstream_gene_variant")),
    ("other", ("other",)),
)

_REPEAT_ALIAS_GROUPS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("none", ("none", "no_repeat", "non_repeat", "not_repetitive")),
    (
        "simple",
        (
            "simple",
            "simple_repeat",
            "microsatellite",
            "short_tandem_repeat",
            "str",
            "tandem_repeat",
            "satellite",
        ),
    ),
    (
        "low_complexity",
        (
            "low_complexity",
            "low_complexity_region",
            "dust",
            "poly_a",
            "poly_t",
            "poly_c",
            "poly_g",
        ),
    ),
    (
        "transposon",
        (
            "transposon",
            "transposable_element",
            "mobile_element",
            "retrotransposon",
            "dna_transposon",
            "line",
            "line1",
            "line_l1",
            "l1",
            "sine",
            "sine_alu",
            "alu",
            "ltr",
            "erv",
        ),
    ),
    (
        "segmental_dup",
        (
            "segmental_dup",
            "segmental_duplication",
            "segdup",
            "genomic_superdup",
            "self_chain",
        ),
    ),
)

_TERM_SEPARATOR = re.compile(r"[,;|&+]+")
_TOKEN_SEPARATOR = re.compile(r"[^a-z0-9]+")
_CALLED_BASES = frozenset("ACGT")
_GC_BASES = frozenset("GC")


@dataclass(frozen=True, slots=True)
class ContextLabel:
    """Canonical surprise-scoring contract context tuple for a single variant locus."""

    region_class: str
    gc_bin: str
    repeat_class: str

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "region_class",
            _require_member("region_class", self.region_class, REGION_CLASSES),
        )
        object.__setattr__(self, "gc_bin", _require_member("gc_bin", self.gc_bin, GC_BINS))
        object.__setattr__(
            self,
            "repeat_class",
            _require_member("repeat_class", self.repeat_class, REPEAT_CLASSES),
        )

    @property
    def bucket_id(self) -> str:
        """Return ``{region_class}|{gc_bin}|{repeat_class}``."""
        return make_bucket_id(self.region_class, self.gc_bin, self.repeat_class)

    def as_tuple(self) -> tuple[str, str, str]:
        """Return the canonical ``(region_class, gc_bin, repeat_class)`` tuple."""
        return (self.region_class, self.gc_bin, self.repeat_class)

    def backoff_chain(self) -> tuple[str, ...]:
        """Return bucket IDs from most specific to catch-all."""
        return backoff_chain(self)


def classify_context(
    *,
    region: str | Sequence[str] | None,
    gc_window: str,
    repeat: str | Sequence[str] | None = None,
    low_gc_cutoff: float = DEFAULT_GC_LOW_CUTOFF,
    high_gc_cutoff: float = DEFAULT_GC_HIGH_CUTOFF,
) -> ContextLabel:
    """Build a canonical context label from annotation terms and a DNA window.

    ``region`` and ``repeat`` accept upstream annotation labels such as
    VEP/SnpEff consequences or repeat-masker class strings. ``gc_window``
    is the sequence window around the variant locus.
    """
    return ContextLabel(
        region_class=classify_region(region),
        gc_bin=classify_gc_bin(
            gc_window,
            low_cutoff=low_gc_cutoff,
            high_cutoff=high_gc_cutoff,
        ),
        repeat_class=classify_repeat(repeat),
    )


def classify_region(annotation: str | Sequence[str] | None) -> str:
    """Return the canonical ``region_class`` for annotation term(s)."""
    terms = _annotation_terms(annotation, field="region")
    if not terms:
        return "other"

    for region_class, aliases in _REGION_ALIAS_GROUPS:
        if any(term == region_class or term in aliases for term in terms):
            return region_class
    return "other"


def classify_repeat(annotation: str | Sequence[str] | None) -> str:
    """Return the canonical ``repeat_class`` for repeat annotation term(s)."""
    terms = _annotation_terms(annotation, field="repeat")
    if not terms:
        return "none"

    for repeat_class, aliases in _REPEAT_ALIAS_GROUPS:
        if any(term == repeat_class or term in aliases for term in terms):
            return repeat_class

    raise InputError(
        "repeat annotation does not map to a known repeat_class",
        details={"annotation": list(terms), "allowed": list(REPEAT_CLASSES)},
        remediation="normalize the repeat track to none/simple/low_complexity/transposon/segmental_dup",
    )


def gc_fraction(sequence: str) -> float:
    """Return GC fraction over called A/C/G/T bases in ``sequence``.

    ``N`` bases are valid in reference windows but are excluded from the
    denominator because their GC status is unknown. A window containing
    no called bases is rejected.
    """
    canonical = canonicalize_dna(sequence)
    called_count = sum(base in _CALLED_BASES for base in canonical)
    if called_count == 0:
        raise InputError(
            "GC window contains no called A/C/G/T bases",
            details={"length": len(canonical)},
        )
    gc_count = sum(base in _GC_BASES for base in canonical)
    return gc_count / called_count


def classify_gc_bin(
    sequence: str,
    *,
    low_cutoff: float = DEFAULT_GC_LOW_CUTOFF,
    high_cutoff: float = DEFAULT_GC_HIGH_CUTOFF,
) -> str:
    """Return ``low``, ``mid``, or ``high`` for a DNA window's GC fraction."""
    low = _validate_cutoff("low_cutoff", low_cutoff)
    high = _validate_cutoff("high_cutoff", high_cutoff)
    if low >= high:
        raise InputError(
            "low_cutoff must be less than high_cutoff",
            details={"low_cutoff": low, "high_cutoff": high},
        )

    fraction = gc_fraction(sequence)
    if fraction <= low:
        return "low"
    if fraction >= high:
        return "high"
    return "mid"


def make_bucket_id(region_class: str, gc_bin: str, repeat_class: str) -> str:
    """Return the stable full calibration bucket ID for a context tuple."""
    return _join_bucket_parts(
        (
            _require_member("region_class", region_class, REGION_CLASSES),
            _require_member("gc_bin", gc_bin, GC_BINS),
            _require_member("repeat_class", repeat_class, REPEAT_CLASSES),
        )
    )


def backoff_chain(label_or_bucket: ContextLabel | str) -> tuple[str, ...]:
    """Return fixed parent-bucket IDs ending in ``*``.

    Full buckets back off as ``region|gc|repeat`` -> ``region|gc`` ->
    ``region`` -> ``*``. Parent buckets can also be passed directly.
    """
    parts = _bucket_parts(label_or_bucket)
    if not parts:
        return (UNKNOWN_BUCKET_ID,)
    parents = tuple(_join_bucket_parts(parts[:i]) for i in range(len(parts), 0, -1))
    return (*parents, UNKNOWN_BUCKET_ID)


def select_backoff_bucket(
    label_or_bucket: ContextLabel | str,
    bucket_sizes: Mapping[str, int],
    *,
    min_count: int = DEFAULT_MIN_BUCKET_SIZE,
) -> str:
    """Return the first bucket in the backoff chain with enough calibration rows.

    If every specific parent is sparse, the catch-all ``*`` bucket is
    returned. Downstream calibration code can still report low
    confidence based on that bucket's own count.
    """
    threshold = _validate_positive_int("min_count", min_count)
    counts = _validate_bucket_sizes(bucket_sizes)
    chain = backoff_chain(label_or_bucket)
    for bucket_id in chain[:-1]:
        if counts.get(bucket_id, 0) >= threshold:
            return bucket_id
    return chain[-1]


def _annotation_terms(annotation: str | Sequence[str] | None, *, field: str) -> tuple[str, ...]:
    if annotation is None:
        return ()
    if isinstance(annotation, str):
        raw_values: Sequence[str] = (annotation,)
    elif isinstance(annotation, Sequence):
        raw_values = annotation
    else:
        raise InputError(
            f"{field} annotation must be a string, sequence of strings, or None",
            details={"type": type(annotation).__name__},
        )

    terms: list[str] = []
    for raw_value in raw_values:
        if not isinstance(raw_value, str):
            raise InputError(
                f"{field} annotation entries must be strings",
                details={"type": type(raw_value).__name__},
            )
        for raw_term in _TERM_SEPARATOR.split(raw_value):
            term = _normalize_annotation(raw_term)
            if term:
                terms.append(term)
    return tuple(terms)


def _normalize_annotation(value: str) -> str:
    return "_".join(part for part in _TOKEN_SEPARATOR.split(value.strip().lower()) if part)


def _require_member(field: str, value: str, allowed: tuple[str, ...]) -> str:
    if not isinstance(value, str):
        raise InputError(
            f"{field} must be a string",
            details={"field": field, "type": type(value).__name__},
        )
    if value not in allowed:
        raise InputError(
            f"{field} is not a supported value",
            details={"field": field, "value": value, "allowed": list(allowed)},
        )
    return value


def _validate_cutoff(name: str, value: float) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float) or not math.isfinite(value):
        raise InputError(
            f"{name} must be a finite number",
            details={name: value, "type": type(value).__name__},
        )
    cutoff = float(value)
    if cutoff < 0.0 or cutoff > 1.0:
        raise InputError(
            f"{name} must be in [0, 1]",
            details={name: cutoff},
        )
    return cutoff


def _validate_positive_int(name: str, value: int) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise InputError(
            f"{name} must be a positive integer",
            details={name: value, "type": type(value).__name__},
        )
    return value


def _validate_bucket_sizes(bucket_sizes: Mapping[str, int]) -> dict[str, int]:
    if not isinstance(bucket_sizes, Mapping):
        raise InputError(
            "bucket_sizes must be a mapping",
            details={"type": type(bucket_sizes).__name__},
        )

    normalized: dict[str, int] = {}
    for bucket_id, count in bucket_sizes.items():
        if not isinstance(bucket_id, str):
            raise InputError(
                "bucket_sizes keys must be bucket ID strings",
                details={"type": type(bucket_id).__name__},
            )
        _bucket_parts(bucket_id)
        if not isinstance(count, int) or isinstance(count, bool) or count < 0:
            raise InputError(
                "bucket_sizes counts must be non-negative integers",
                details={"bucket_id": bucket_id, "count": count, "type": type(count).__name__},
            )
        normalized[bucket_id] = count
    return normalized


def _bucket_parts(label_or_bucket: ContextLabel | str) -> tuple[str, ...]:
    if isinstance(label_or_bucket, ContextLabel):
        return label_or_bucket.as_tuple()

    if not isinstance(label_or_bucket, str):
        raise InputError(
            "bucket must be a ContextLabel or bucket ID string",
            details={"type": type(label_or_bucket).__name__},
        )

    bucket_id = label_or_bucket.strip()
    if bucket_id == UNKNOWN_BUCKET_ID:
        return ()
    parts = tuple(bucket_id.split("|"))
    if not 1 <= len(parts) <= 3 or any(part == "" for part in parts):
        raise InputError(
            "bucket ID must be region, region|gc, region|gc|repeat, or *",
            details={"bucket_id": label_or_bucket},
        )

    region = _require_member("region_class", parts[0], REGION_CLASSES)
    if len(parts) == 1:
        return (region,)
    gc_bin = _require_member("gc_bin", parts[1], GC_BINS)
    if len(parts) == 2:
        return (region, gc_bin)
    repeat_class = _require_member("repeat_class", parts[2], REPEAT_CLASSES)
    return (region, gc_bin, repeat_class)


def _join_bucket_parts(parts: tuple[str, ...]) -> str:
    if not parts:
        return UNKNOWN_BUCKET_ID
    return "|".join(parts)
