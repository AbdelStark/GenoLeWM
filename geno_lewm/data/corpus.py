# SPDX-License-Identifier: Apache-2.0
"""Carbon pretraining corpus records and RFC-0006 window sampling."""

from __future__ import annotations

import hashlib
import math
import random
from collections.abc import Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from geno_lewm.encoder.windowing import DEFAULT_WINDOW_BP, canonicalize_dna, window_sha256
from geno_lewm.errors import InputError, RuntimeSetupError

__all__ = [
    "CARBON_SUBMIX",
    "DEFAULT_CORPUS_MARGIN_BP",
    "DEFAULT_CORPUS_STRIDE_BP",
    "DEFAULT_PHASE1_SUBSET_FRACTION",
    "DEFAULT_SEQUENCE_FIELD",
    "DEFAULT_SOURCE_FIELD",
    "DEFAULT_SOURCE_ID_FIELD",
    "CarbonCorpusConfig",
    "CarbonRecord",
    "CarbonSourceMix",
    "CarbonWindow",
    "draw_source_counts",
    "iter_carbon_records",
    "iter_record_windows",
    "iter_window_starts",
    "load_hf_carbon_records",
    "normalize_source_label",
    "sample_source",
    "stable_subset_includes",
]

DEFAULT_CARBON_DATASET_ID = "HuggingFaceBio/carbon-pretraining-corpus"
DEFAULT_PHASE1_SUBSET_FRACTION = 0.10
DEFAULT_CORPUS_MARGIN_BP = 256
DEFAULT_CORPUS_STRIDE_BP = 8_192
DEFAULT_SEQUENCE_FIELD = "sequence"
DEFAULT_SOURCE_FIELD = "source"
DEFAULT_SOURCE_ID_FIELD = "id"


@dataclass(frozen=True, slots=True)
class CarbonSourceMix:
    """One source bucket in the RFC-0006 Carbon sub-mix."""

    source: str
    fraction: float


CARBON_SUBMIX: tuple[CarbonSourceMix, ...] = (
    CarbonSourceMix("eukaryotic_genes", 0.50),
    CarbonSourceMix("mrna", 0.25),
    CarbonSourceMix("splice_mrna", 0.10),
    CarbonSourceMix("gtdb", 0.15),
)


@dataclass(frozen=True, slots=True)
class CarbonCorpusConfig:
    """Configuration for reading and windowing the Carbon pretraining corpus."""

    dataset_id: str = DEFAULT_CARBON_DATASET_ID
    dataset_config: str | None = None
    revision: str | None = None
    split: str = "train"
    streaming: bool = True
    subset_fraction: float = DEFAULT_PHASE1_SUBSET_FRACTION
    subset_seed: int = 0
    sequence_field: str = DEFAULT_SEQUENCE_FIELD
    source_field: str = DEFAULT_SOURCE_FIELD
    source_id_field: str = DEFAULT_SOURCE_ID_FIELD
    window_bp: int = DEFAULT_WINDOW_BP
    margin_bp: int = DEFAULT_CORPUS_MARGIN_BP
    stride_bp: int = DEFAULT_CORPUS_STRIDE_BP

    def __post_init__(self) -> None:
        _require_nonempty_str("dataset_id", self.dataset_id)
        _require_nonempty_str("split", self.split)
        _require_nonempty_str("sequence_field", self.sequence_field)
        _require_nonempty_str("source_field", self.source_field)
        _require_nonempty_str("source_id_field", self.source_id_field)
        _validate_fraction("subset_fraction", self.subset_fraction)
        _require_nonnegative_int("subset_seed", self.subset_seed)
        _require_positive_int("window_bp", self.window_bp)
        _require_nonnegative_int("margin_bp", self.margin_bp)
        _require_positive_int("stride_bp", self.stride_bp)
        if self.dataset_config is not None:
            _require_nonempty_str("dataset_config", self.dataset_config)
        if self.revision is not None:
            _require_nonempty_str("revision", self.revision)


@dataclass(frozen=True, slots=True)
class CarbonRecord:
    """Canonicalized source sequence record from the Carbon corpus."""

    record_id: str
    source: str
    sequence: str

    @property
    def length_bp(self) -> int:
        """Return the canonical DNA sequence length in base pairs."""
        return len(self.sequence)


@dataclass(frozen=True, slots=True)
class CarbonWindow:
    """A fixed-width training window sampled from a Carbon corpus record."""

    record_id: str
    source: str
    start_bp: int
    end_bp: int
    sequence: str

    @property
    def window_bp(self) -> int:
        """Return the window length in base pairs."""
        return self.end_bp - self.start_bp

    @property
    def window_id(self) -> str:
        """Return the content-addressed window hash as lowercase hex."""
        return window_sha256(self.sequence).hex()


_SOURCE_ALIASES: dict[str, str] = {
    "eukaryotic_genes": "eukaryotic_genes",
    "eukaryotic genes": "eukaryotic_genes",
    "eukaryotic_gene": "eukaryotic_genes",
    "eukaryotic gene": "eukaryotic_genes",
    "generator": "eukaryotic_genes",
    "generator-style annotated": "eukaryotic_genes",
    "mrna": "mrna",
    "mrna transcripts": "mrna",
    "transcript": "mrna",
    "transcripts": "mrna",
    "splice_mrna": "splice_mrna",
    "splice mrna": "splice_mrna",
    "splice-enriched mrna": "splice_mrna",
    "splice enriched mrna": "splice_mrna",
    "gtdb": "gtdb",
    "gtdb bacterial genomes": "gtdb",
    "bacterial genomes": "gtdb",
}


def normalize_source_label(value: object) -> str:
    """Normalize a Carbon corpus source label to the RFC-0006 source key."""
    if not isinstance(value, str) or not value.strip():
        raise InputError(
            "source label must be a non-empty string",
            details={"value": value, "type": type(value).__name__},
        )
    key = value.strip().lower().replace("-", " ").replace("/", " ")
    key = " ".join(key.split())
    normalized = _SOURCE_ALIASES.get(key)
    if normalized is None:
        raise InputError(
            "unsupported Carbon corpus source label",
            details={"source": value, "known_sources": [entry.source for entry in CARBON_SUBMIX]},
        )
    return normalized


def sample_source(
    rng: random.Random,
    *,
    mix: Sequence[CarbonSourceMix] = CARBON_SUBMIX,
) -> str:
    """Sample one source key from the configured RFC-0006 sub-mix."""
    return _sample_source_from_entries(rng, _validate_mix(mix))


def _sample_source_from_entries(rng: random.Random, entries: Sequence[CarbonSourceMix]) -> str:
    total = sum(entry.fraction for entry in entries)
    draw = rng.random() * total
    cumulative = 0.0
    for entry in entries:
        cumulative += entry.fraction
        if draw < cumulative:
            return entry.source
    return entries[-1].source


def draw_source_counts(
    n: int,
    *,
    rng: random.Random,
    mix: Sequence[CarbonSourceMix] = CARBON_SUBMIX,
) -> dict[str, int]:
    """Draw ``n`` source samples and return counts by normalized source key."""
    _require_nonnegative_int("n", n)
    entries = _validate_mix(mix)
    counts = {entry.source: 0 for entry in entries}
    for _ in range(n):
        counts[_sample_source_from_entries(rng, entries)] += 1
    return counts


def stable_subset_includes(record_id: str, *, fraction: float, seed: int = 0) -> bool:
    """Return whether ``record_id`` belongs to a deterministic corpus subset."""
    _require_nonempty_str("record_id", record_id)
    _validate_fraction("fraction", fraction)
    _require_nonnegative_int("seed", seed)
    digest = hashlib.sha256(f"{seed}:{record_id}".encode()).digest()
    value = int.from_bytes(digest[:8], byteorder="big") / float(1 << 64)
    return value < fraction


def iter_window_starts(
    sequence_length: int,
    *,
    window_bp: int = DEFAULT_WINDOW_BP,
    margin_bp: int = DEFAULT_CORPUS_MARGIN_BP,
    stride_bp: int = DEFAULT_CORPUS_STRIDE_BP,
    rng: random.Random | None = None,
) -> Iterator[int]:
    """Yield RFC-0006 window starts respecting margin and stride constraints."""
    _require_nonnegative_int("sequence_length", sequence_length)
    _require_positive_int("window_bp", window_bp)
    _require_nonnegative_int("margin_bp", margin_bp)
    _require_positive_int("stride_bp", stride_bp)

    required = window_bp + (2 * margin_bp)
    if sequence_length < required:
        return

    min_start = margin_bp
    max_start = sequence_length - window_bp - margin_bp
    phase_span = min(stride_bp, max_start - min_start + 1)
    offset = rng.randrange(phase_span) if rng is not None and phase_span > 1 else 0
    start = min_start + offset
    while start <= max_start:
        yield start
        start += stride_bp


def iter_record_windows(
    record: CarbonRecord,
    *,
    window_bp: int = DEFAULT_WINDOW_BP,
    margin_bp: int = DEFAULT_CORPUS_MARGIN_BP,
    stride_bp: int = DEFAULT_CORPUS_STRIDE_BP,
    rng: random.Random | None = None,
) -> Iterator[CarbonWindow]:
    """Yield canonical windows for one Carbon corpus record."""
    for start in iter_window_starts(
        record.length_bp,
        window_bp=window_bp,
        margin_bp=margin_bp,
        stride_bp=stride_bp,
        rng=rng,
    ):
        end = start + window_bp
        yield CarbonWindow(
            record_id=record.record_id,
            source=record.source,
            start_bp=start,
            end_bp=end,
            sequence=record.sequence[start:end],
        )


def iter_carbon_records(
    rows: Iterable[Mapping[str, Any]],
    *,
    sequence_field: str = DEFAULT_SEQUENCE_FIELD,
    source_field: str = DEFAULT_SOURCE_FIELD,
    source_id_field: str = DEFAULT_SOURCE_ID_FIELD,
    subset_fraction: float = 1.0,
    subset_seed: int = 0,
) -> Iterator[CarbonRecord]:
    """Yield canonical Carbon records from HF-style row mappings."""
    _require_nonempty_str("sequence_field", sequence_field)
    _require_nonempty_str("source_field", source_field)
    _require_nonempty_str("source_id_field", source_id_field)
    _validate_fraction("subset_fraction", subset_fraction)
    _require_nonnegative_int("subset_seed", subset_seed)

    for row_idx, row in enumerate(rows):
        sequence_value = row.get(sequence_field)
        if not isinstance(sequence_value, str):
            raise InputError(
                "Carbon corpus row is missing a DNA sequence string",
                details={"row": row_idx, "sequence_field": sequence_field},
            )
        source = normalize_source_label(row.get(source_field))
        sequence = canonicalize_dna(sequence_value)
        raw_record_id = row.get(source_id_field)
        record_id = (
            str(raw_record_id) if raw_record_id not in (None, "") else _fallback_id(sequence)
        )
        if not stable_subset_includes(record_id, fraction=subset_fraction, seed=subset_seed):
            continue
        yield CarbonRecord(record_id=record_id, source=source, sequence=sequence)


def load_hf_carbon_records(
    config: CarbonCorpusConfig | None = None,
) -> Iterator[CarbonRecord]:
    """Load Carbon corpus records through Hugging Face ``datasets`` lazily."""
    if config is None:
        config = CarbonCorpusConfig()
    try:
        from datasets import load_dataset  # type: ignore[import-not-found]
    except ImportError as exc:
        raise RuntimeSetupError(
            "Carbon corpus loading requires Hugging Face datasets",
            remediation="install geno-lewm[train] or install datasets",
        ) from exc

    args: tuple[str, ...]
    if config.dataset_config is None:
        args = (config.dataset_id,)
    else:
        args = (config.dataset_id, config.dataset_config)
    dataset = load_dataset(
        *args,
        split=config.split,
        streaming=config.streaming,
        revision=config.revision,
    )
    return iter_carbon_records(
        dataset,
        sequence_field=config.sequence_field,
        source_field=config.source_field,
        source_id_field=config.source_id_field,
        subset_fraction=config.subset_fraction,
        subset_seed=config.subset_seed,
    )


def _fallback_id(sequence: str) -> str:
    return hashlib.sha256(sequence.encode("ascii")).hexdigest()[:16]


def _validate_mix(mix: Sequence[CarbonSourceMix]) -> tuple[CarbonSourceMix, ...]:
    if not mix:
        raise InputError("source mix must contain at least one entry")
    entries: list[CarbonSourceMix] = []
    seen: set[str] = set()
    for entry in mix:
        normalized = normalize_source_label(entry.source)
        if normalized in seen:
            raise InputError(
                "source mix contains duplicate source labels", details={"source": entry.source}
            )
        seen.add(normalized)
        _validate_fraction("source fraction", entry.fraction)
        entries.append(CarbonSourceMix(normalized, entry.fraction))
    return tuple(entries)


def _validate_fraction(name: str, value: float) -> None:
    if (
        isinstance(value, bool)
        or not isinstance(value, int | float)
        or not math.isfinite(float(value))
        or value <= 0.0
        or value > 1.0
    ):
        raise InputError(
            f"{name} must be in the interval (0, 1]",
            details={"field": name, "value": value, "type": type(value).__name__},
        )


def _require_nonempty_str(name: str, value: str) -> None:
    if not isinstance(value, str) or not value:
        raise InputError(
            f"{name} must be a non-empty string",
            details={"field": name, "value": value, "type": type(value).__name__},
        )


def _require_positive_int(name: str, value: int) -> None:
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise InputError(
            f"{name} must be a positive integer",
            details={"field": name, "value": value, "type": type(value).__name__},
        )


def _require_nonnegative_int(name: str, value: int) -> None:
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise InputError(
            f"{name} must be a non-negative integer",
            details={"field": name, "value": value, "type": type(value).__name__},
        )
