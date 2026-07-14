# SPDX-License-Identifier: Apache-2.0
"""Training tuple builder for the data-pipeline contract data pipeline.

This module owns the dependency-free boundary between prepared data
sources and the eventual PyTorch trainer. It does not download gnomAD,
ClinVar, or Carbon data. Instead, callers provide edit-source providers
that are easy to unit-test with fixtures and later wire to real shards.
"""

from __future__ import annotations

import importlib
import random
from bisect import bisect_right
from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, cast

from geno_lewm.action import (
    DEFAULT_EDGE_MARGIN,
    EditSpec,
    EditType,
    RelEdit,
    apply_edit,
    indel,
    uniform_snv,
)
from geno_lewm.encoder.windowing import canonicalize_dna, window_sha256
from geno_lewm.errors import InputError
from geno_lewm.provenance import canonical_json_sha256

__all__ = [
    "DEFAULT_EDIT_SOURCE_COUNTS",
    "DEFAULT_SOURCE_FALLBACKS",
    "SOURCE_CLINVAR",
    "SOURCE_GNOMAD_COMMON",
    "SOURCE_SYNTHETIC_INDEL",
    "SOURCE_SYNTHETIC_SNV",
    "EditSourceCount",
    "GenoLeWMDataset",
    "HoldoutInterval",
    "HoldoutPolicy",
    "TrainingDatasetItem",
    "TrainingTuple",
    "WindowContext",
    "build_training_tuples",
    "synthetic_indel_provider",
    "synthetic_snv_provider",
    "variant_provider",
]

SOURCE_GNOMAD_COMMON = "gnomad_common"
SOURCE_SYNTHETIC_SNV = "synthetic_snv"
SOURCE_SYNTHETIC_INDEL = "synthetic_indel"
SOURCE_CLINVAR = "clinvar"


def _require_nonempty_str(name: str, value: str) -> None:
    if not isinstance(value, str) or not value:
        raise InputError(
            f"{name} must be a non-empty string",
            details={"field": name, "value": value, "type": type(value).__name__},
        )


def _require_nonnegative_int(name: str, value: int) -> None:
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise InputError(
            f"{name} must be a non-negative integer",
            details={"field": name, "value": value, "type": type(value).__name__},
        )


@dataclass(frozen=True, slots=True)
class EditSourceCount:
    """Number of edits to draw from one data-pipeline contract source per window."""

    source: str
    count: int

    def __post_init__(self) -> None:
        _require_nonempty_str("source", self.source)
        _require_nonnegative_int("count", self.count)


DEFAULT_EDIT_SOURCE_COUNTS: tuple[EditSourceCount, ...] = (
    EditSourceCount(SOURCE_GNOMAD_COMMON, 3),
    EditSourceCount(SOURCE_SYNTHETIC_SNV, 3),
    EditSourceCount(SOURCE_SYNTHETIC_INDEL, 1),
    EditSourceCount(SOURCE_CLINVAR, 1),
)
"""data-pipeline contract per-window source allocation for ``N_edits = 8``."""

DEFAULT_SOURCE_FALLBACKS: dict[str, str] = {
    SOURCE_CLINVAR: SOURCE_SYNTHETIC_SNV,
    SOURCE_GNOMAD_COMMON: SOURCE_SYNTHETIC_SNV,
}
"""Default fallback when an absolute VCF edit is unavailable for a window.

ClinVar hard-negatives and gnomAD common variants are placed (absolute) sources:
they only apply to windows that carry genome coordinates. On unplaced windows
(the synthetic Carbon pretraining corpus) the absolute providers yield nothing
and the builder draws synthetic SNVs instead, so pretraining-corpus windows
still produce full edit tuples."""


@dataclass(frozen=True, slots=True)
class WindowContext:
    """One reference window plus source coordinates for tuple building.

    Coordinates are 0-based half-open: ``start_bp`` is inclusive and
    ``end_bp`` is exclusive. ``chrom`` is required for absolute variant
    providers and chromosome/interval holdouts, but synthetic providers
    can operate on unplaced Carbon windows.
    """

    record_id: str
    source: str
    sequence: str
    start_bp: int = 0
    chrom: str | None = None

    def __post_init__(self) -> None:
        _require_nonempty_str("record_id", self.record_id)
        _require_nonempty_str("source", self.source)
        _require_nonnegative_int("start_bp", self.start_bp)
        if self.chrom is not None:
            _require_nonempty_str("chrom", self.chrom)
        sequence = canonicalize_dna(self.sequence)
        if not sequence:
            raise InputError("window sequence must be non-empty")
        object.__setattr__(self, "sequence", sequence)

    @property
    def end_bp(self) -> int:
        """Return the 0-based exclusive end coordinate."""
        return self.start_bp + len(self.sequence)

    @property
    def window_id(self) -> str:
        """Return the content hash used for cache lookup."""
        return window_sha256(self.sequence).hex()


@dataclass(frozen=True, slots=True)
class HoldoutInterval:
    """0-based half-open genomic interval excluded from training."""

    chrom: str
    start_bp: int
    end_bp: int

    def __post_init__(self) -> None:
        _require_nonempty_str("chrom", self.chrom)
        _require_nonnegative_int("start_bp", self.start_bp)
        _require_nonnegative_int("end_bp", self.end_bp)
        if self.end_bp <= self.start_bp:
            raise InputError(
                "holdout interval end_bp must be greater than start_bp",
                details={"chrom": self.chrom, "start_bp": self.start_bp, "end_bp": self.end_bp},
            )

    def intersects(self, chrom: str | None, start_bp: int, end_bp: int) -> bool:
        """Return whether ``[start_bp, end_bp)`` intersects this interval."""
        if chrom != self.chrom:
            return False
        return start_bp < self.end_bp and self.start_bp < end_bp

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-serializable holdout interval payload."""
        return {
            "chrom": self.chrom,
            "start_bp": self.start_bp,
            "end_bp": self.end_bp,
        }


@dataclass(frozen=True, slots=True)
class HoldoutPolicy:
    """Holdout exclusions enforced before a tuple reaches the trainer."""

    holdout_chroms: tuple[str, ...] = ()
    intervals: tuple[HoldoutInterval, ...] = ()
    edit_keys: tuple[str, ...] = ()
    record_ids: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "holdout_chroms",
            _normalize_nonempty_tuple("holdout_chroms", self.holdout_chroms),
        )
        object.__setattr__(
            self, "edit_keys", _normalize_nonempty_tuple("edit_keys", self.edit_keys)
        )
        object.__setattr__(
            self,
            "record_ids",
            _normalize_nonempty_tuple("record_ids", self.record_ids),
        )
        for interval in self.intervals:
            if not isinstance(interval, HoldoutInterval):
                raise InputError(
                    "holdout intervals must contain HoldoutInterval values",
                    details={"type": type(interval).__name__},
                )

    def excludes_window(self, window: WindowContext) -> bool:
        """Return whether the entire source window is in a holdout."""
        if not isinstance(window, WindowContext):
            raise InputError(
                "window must be a WindowContext",
                details={"type": type(window).__name__},
            )
        if window.record_id in self.record_ids:
            return True
        if window.chrom in self.holdout_chroms:
            return True
        return any(
            interval.intersects(window.chrom, window.start_bp, window.end_bp)
            for interval in self.intervals
        )

    def excludes_edit(self, window: WindowContext, edit: RelEdit) -> bool:
        """Return whether one relative edit intersects an edit-level holdout."""
        if window.chrom is None:
            return False
        if window.chrom in self.holdout_chroms:
            return True
        edit_start = window.start_bp + edit.rel_pos
        edit_end = edit_start + len(edit.ref_bases)
        if any(
            interval.intersects(window.chrom, edit_start, edit_end) for interval in self.intervals
        ):
            return True
        return _edit_key(window.chrom, edit_start + 1, edit.ref_bases, edit.alt_bases) in set(
            self.edit_keys
        )

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-serializable holdout policy payload."""
        return {
            "schema_version": "1.0.0",
            "holdout_chroms": list(self.holdout_chroms),
            "intervals": [interval.to_dict() for interval in self.intervals],
            "edit_keys": list(self.edit_keys),
            "record_ids": list(self.record_ids),
        }

    def identity(self) -> str:
        """Return the canonical SHA-256 identity of this holdout policy."""
        return canonical_json_sha256(self.to_dict())


@dataclass(frozen=True, slots=True)
class TrainingTuple:
    """One data-pipeline contract ``(window_id, action, target_window)`` training item."""

    window_id: str
    source_record_id: str
    edit_source: str
    rel_edits: tuple[RelEdit, ...]
    target_window: str
    window_start_bp: int
    window_end_bp: int

    def __post_init__(self) -> None:
        _require_nonempty_str("window_id", self.window_id)
        _require_nonempty_str("source_record_id", self.source_record_id)
        _require_nonempty_str("edit_source", self.edit_source)
        _require_nonnegative_int("window_start_bp", self.window_start_bp)
        _require_nonnegative_int("window_end_bp", self.window_end_bp)
        if self.window_end_bp <= self.window_start_bp:
            raise InputError("window_end_bp must be greater than window_start_bp")
        if not self.rel_edits:
            raise InputError("rel_edits must contain at least one edit")
        for edit in self.rel_edits:
            if not isinstance(edit, RelEdit):
                raise InputError(
                    "rel_edits must contain RelEdit values",
                    details={"type": type(edit).__name__},
                )
        object.__setattr__(self, "target_window", canonicalize_dna(self.target_window))


_EditProvider = Callable[[WindowContext, int, random.Random], Iterable[RelEdit]]
_WindowSource = Iterable[WindowContext] | Callable[[], Iterable[WindowContext]]


@dataclass(frozen=True, slots=True)
class _DatasetWorkerInfo:
    """Worker partition used by ``GenoLeWMDataset``."""

    id: int
    num_workers: int


def _load_iterable_dataset_base() -> type[Any]:
    try:  # pragma: no cover - optional runtime branch.
        torch_data = importlib.import_module("torch.utils.data")
    except ImportError:  # pragma: no cover - covered by torch-absent imports.
        return object
    return cast(type[Any], torch_data.__dict__["IterableDataset"])


@dataclass(frozen=True, slots=True)
class TrainingDatasetItem:
    """One stream item with the source window needed for trainer encoding."""

    source_window: WindowContext
    training_tuple: TrainingTuple


class GenoLeWMDataset(_load_iterable_dataset_base()):  # type: ignore[misc]
    """Deterministic iterable dataset over windows and edit-source providers.

    The class subclasses ``torch.utils.data.IterableDataset`` when torch
    is installed, but falls back to a plain Python iterable in core/dev
    environments. That keeps the data contract testable without pulling
    in the full training extra.
    """

    def __init__(
        self,
        windows: _WindowSource,
        providers: Mapping[str, _EditProvider],
        *,
        seed: int,
        mix: Sequence[EditSourceCount] = DEFAULT_EDIT_SOURCE_COUNTS,
        holdouts: HoldoutPolicy | None = None,
        fallback_sources: Mapping[str, str] | None = DEFAULT_SOURCE_FALLBACKS,
        preserve_length: bool = True,
    ) -> None:
        _require_nonnegative_int("seed", seed)
        if not providers:
            raise InputError("providers must contain at least one edit source")
        self.windows = windows
        self.providers = dict(providers)
        self.seed = seed
        self.mix = _normalize_mix(mix)
        self.holdouts = holdouts
        self.fallback_sources = dict(fallback_sources or {})
        self.preserve_length = preserve_length

    def __iter__(self) -> Iterator[TrainingTuple]:
        """Yield training tuples suitable for a PyTorch DataLoader."""
        for item in self.iter_with_source_windows():
            yield item.training_tuple

    def iter_with_source_windows(self) -> Iterator[TrainingDatasetItem]:
        """Yield tuples together with their source windows for trainer encoding."""
        worker = _torch_worker_info()
        rng = random.Random(self.seed + worker.id)
        for index, window in enumerate(_iter_window_source(self.windows)):
            if index % worker.num_workers != worker.id:
                continue
            if not isinstance(window, WindowContext):
                raise InputError(
                    "window source must yield WindowContext values",
                    details={"type": type(window).__name__},
                )
            for item in build_training_tuples(
                window,
                self.providers,
                rng=rng,
                mix=self.mix,
                holdouts=self.holdouts,
                fallback_sources=self.fallback_sources,
                preserve_length=self.preserve_length,
            ):
                yield TrainingDatasetItem(source_window=window, training_tuple=item)


def build_training_tuples(
    window: WindowContext,
    providers: Mapping[str, _EditProvider],
    *,
    rng: random.Random,
    mix: Sequence[EditSourceCount] = DEFAULT_EDIT_SOURCE_COUNTS,
    holdouts: HoldoutPolicy | None = None,
    fallback_sources: Mapping[str, str] | None = DEFAULT_SOURCE_FALLBACKS,
    preserve_length: bool = True,
) -> tuple[TrainingTuple, ...]:
    """Build per-window training tuples with source mix and holdout checks.

    ``providers`` map source names to callables returning relative edits
    for that window. The default mix encodes the data-pipeline contract's 3/3/1/1
    gnomAD/synthetic-SNV/synthetic-indel/ClinVar allocation. If a source
    cannot produce enough edits, only explicitly configured fallbacks are
    used; missing gnomAD data therefore fails instead of silently turning
    the training stream synthetic.
    """
    if not isinstance(window, WindowContext):
        raise InputError("window must be a WindowContext")
    if not isinstance(rng, random.Random):
        raise InputError("rng must be a random.Random instance")
    active_holdouts = holdouts if holdouts is not None else HoldoutPolicy()
    if active_holdouts.excludes_window(window):
        return ()

    source_mix = _normalize_mix(mix)
    fallbacks = dict(fallback_sources or {})
    tuples: list[TrainingTuple] = []
    for entry in source_mix:
        if entry.count == 0:
            continue
        edits = _sample_edits(
            source=entry.source,
            count=entry.count,
            window=window,
            providers=providers,
            rng=rng,
            holdouts=active_holdouts,
            fallback_sources=fallbacks,
        )
        tuples.extend(
            _tuple_for_edit(
                window,
                edit,
                source=source,
                preserve_length=preserve_length,
            )
            for source, edit in edits
        )
    return tuple(tuples)


def synthetic_snv_provider(
    window: WindowContext, count: int, rng: random.Random
) -> tuple[RelEdit, ...]:
    """Provider for data-pipeline contract uniform synthetic SNVs."""
    _require_nonnegative_int("count", count)
    return tuple(uniform_snv(window.sequence, count, rng=rng))


def synthetic_indel_provider(
    window: WindowContext, count: int, rng: random.Random
) -> tuple[RelEdit, ...]:
    """Provider for data-pipeline contract synthetic indels."""
    _require_nonnegative_int("count", count)
    return tuple(indel(window.sequence, count, rng=rng))


@dataclass(frozen=True, slots=True)
class _VariantProvider:
    by_chrom: Mapping[str, tuple[tuple[int, ...], tuple[EditSpec, ...]]]

    def __call__(
        self,
        window: WindowContext,
        count: int,
        rng: random.Random,
    ) -> tuple[RelEdit, ...]:
        return self.sample(window, count, rng=rng, holdouts=None)

    def sample(
        self,
        window: WindowContext,
        count: int,
        *,
        rng: random.Random,
        holdouts: HoldoutPolicy | None,
    ) -> tuple[RelEdit, ...]:
        _require_nonnegative_int("count", count)
        if count == 0:
            return ()
        candidates = list(self._candidates(window))
        if holdouts is not None:
            candidates = [edit for edit in candidates if not holdouts.excludes_edit(window, edit)]
        rng.shuffle(candidates)
        return tuple(candidates[:count])

    def available_count(self, window: WindowContext, *, holdouts: HoldoutPolicy) -> int:
        return sum(
            1 for edit in self._candidates(window) if not holdouts.excludes_edit(window, edit)
        )

    def _candidates(self, window: WindowContext) -> tuple[RelEdit, ...]:
        if window.chrom is None:
            # Unplaced windows (e.g. the synthetic Carbon pretraining corpus)
            # carry no genome coordinates, so absolute VCF variants cannot be
            # mapped onto them. Yield nothing and let the source fallback supply
            # synthetic edits (see DEFAULT_SOURCE_FALLBACKS). Placed windows with
            # a chrom still receive their real gnomAD/ClinVar variants.
            return ()
        indexed = self.by_chrom.get(window.chrom)
        if indexed is None:
            return ()
        positions, chrom_variants = indexed
        start = bisect_right(positions, window.start_bp)
        stop = bisect_right(positions, window.end_bp)
        return tuple(
            variant.relative_to(window.start_bp, window.end_bp - 1)
            for variant in chrom_variants[start:stop]
            if variant.pos - 1 + len(variant.ref) <= window.end_bp
        )


def variant_provider(variants: Sequence[EditSpec]) -> _EditProvider:
    """Return a provider backed by absolute VCF-style variants."""
    normalized = tuple(_require_edit_spec(value) for value in variants)
    by_chrom: dict[str, tuple[tuple[int, ...], tuple[EditSpec, ...]]] = {}
    chroms = sorted({variant.chrom for variant in normalized})
    for chrom in chroms:
        ordered = tuple(sorted((item for item in normalized if item.chrom == chrom), key=_edit_pos))
        by_chrom[chrom] = (tuple(item.pos for item in ordered), ordered)
    return _VariantProvider(by_chrom)


def _provider_available_edit_count(
    provider: _EditProvider,
    window: WindowContext,
    count: int,
    *,
    holdouts: HoldoutPolicy,
) -> int:
    """Return seed-independent availability for supported release providers."""
    if isinstance(provider, _VariantProvider):
        return provider.available_count(window, holdouts=holdouts)
    if provider is synthetic_snv_provider or provider is synthetic_indel_provider:
        if provider is synthetic_indel_provider and holdouts.edit_keys:
            raise InputError(
                "schema-1.1 training does not support synthetic-indel edit-key holdouts",
                remediation=(
                    "remove edit-key holdouts from the synthetic-indel source or provide "
                    "a deterministic provider with complete availability semantics"
                ),
            )
        content_end = len(window.sequence) - DEFAULT_EDGE_MARGIN
        has_editable_anchor = any(
            window.sequence[position] in "ACGT"
            for position in range(DEFAULT_EDGE_MARGIN, content_end)
        )
        if not has_editable_anchor:
            return 0
        # Supported release holdouts exclude whole windows unless explicit edit
        # keys are present. Avoid materializing the synthetic action space for
        # the common whole-window-only policy.
        if not holdouts.edit_keys:
            return count
        candidates = _eligible_synthetic_snv_candidates(window, holdouts=holdouts)
        return count if candidates else 0
    raise InputError(
        "schema-1.1 training requires deterministic provider availability",
        details={"provider": type(provider).__name__},
    )


def _eligible_synthetic_snv_candidates(
    window: WindowContext,
    *,
    holdouts: HoldoutPolicy,
) -> tuple[RelEdit, ...]:
    candidates: list[RelEdit] = []
    content_end = len(window.sequence) - DEFAULT_EDGE_MARGIN
    for position in range(DEFAULT_EDGE_MARGIN, content_end):
        anchor = window.sequence[position]
        if anchor not in "ACGT":
            continue
        candidates.extend(
            RelEdit(
                rel_pos=position,
                edit_type=EditType.SNV,
                ref_bases=anchor,
                alt_bases=alternate,
            )
            for alternate in "ACGT"
            if alternate != anchor
        )
    return tuple(edit for edit in candidates if not holdouts.excludes_edit(window, edit))


def _edit_pos(value: EditSpec) -> int:
    return value.pos


def _torch_worker_info() -> _DatasetWorkerInfo:
    try:  # pragma: no cover - optional runtime branch.
        torch_data = importlib.import_module("torch.utils.data")
    except ImportError:  # pragma: no cover - covered by torch-absent imports.
        return _DatasetWorkerInfo(id=0, num_workers=1)
    worker = torch_data.__dict__["get_worker_info"]()
    if worker is None:
        return _DatasetWorkerInfo(id=0, num_workers=1)
    return _DatasetWorkerInfo(id=int(worker.id), num_workers=int(worker.num_workers))


def _iter_window_source(windows: _WindowSource) -> Iterator[WindowContext]:
    source: object = windows() if callable(windows) else windows
    if isinstance(source, str | bytes) or not isinstance(source, Iterable):
        raise InputError(
            "windows must be an iterable or a callable returning an iterable",
            details={"type": type(source).__name__},
        )
    return iter(cast(Iterable[WindowContext], source))


def _sample_edits(
    *,
    source: str,
    count: int,
    window: WindowContext,
    providers: Mapping[str, _EditProvider],
    rng: random.Random,
    holdouts: HoldoutPolicy,
    fallback_sources: Mapping[str, str],
) -> list[tuple[str, RelEdit]]:
    edits = _provider_edits(source, count, window, providers, rng, holdouts)
    if len(edits) >= count:
        return [(source, edit) for edit in edits[:count]]

    fallback = fallback_sources.get(source)
    if fallback is None:
        raise InputError(
            "edit source did not produce enough edits and has no fallback",
            details={"source": source, "needed": count, "observed": len(edits)},
        )
    missing = count - len(edits)
    fallback_edits = _provider_edits(fallback, missing, window, providers, rng, holdouts)
    if len(fallback_edits) < missing:
        raise InputError(
            "fallback edit source did not produce enough edits",
            details={
                "source": source,
                "fallback": fallback,
                "needed": missing,
                "observed": len(fallback_edits),
            },
        )
    return [(source, edit) for edit in edits] + [
        (fallback, edit) for edit in fallback_edits[:missing]
    ]


def _provider_edits(
    source: str,
    count: int,
    window: WindowContext,
    providers: Mapping[str, _EditProvider],
    rng: random.Random,
    holdouts: HoldoutPolicy,
) -> list[RelEdit]:
    provider = providers.get(source)
    if provider is None:
        raise InputError(
            "missing edit provider",
            details={"source": source, "known_sources": sorted(providers)},
        )
    if isinstance(provider, _VariantProvider):
        observed = list(provider.sample(window, count, rng=rng, holdouts=holdouts))
    else:
        observed = list(provider(window, count, rng))
    edits: list[RelEdit] = []
    for edit in observed:
        if not isinstance(edit, RelEdit):
            raise InputError(
                "edit providers must return RelEdit values",
                details={"source": source, "type": type(edit).__name__},
            )
        if not holdouts.excludes_edit(window, edit):
            edits.append(edit)
    if len(edits) < count and provider is synthetic_snv_provider:
        candidates = _eligible_synthetic_snv_candidates(window, holdouts=holdouts)
        while len(edits) < count and candidates:
            edits.append(rng.choice(candidates))
    return edits


def _tuple_for_edit(
    window: WindowContext,
    edit: RelEdit,
    *,
    source: str,
    preserve_length: bool,
) -> TrainingTuple:
    target = apply_edit(window.sequence, edit, preserve_length=preserve_length)
    return TrainingTuple(
        window_id=window.window_id,
        source_record_id=window.record_id,
        edit_source=source,
        rel_edits=(edit,),
        target_window=target,
        window_start_bp=window.start_bp,
        window_end_bp=window.end_bp,
    )


def _normalize_mix(mix: Sequence[EditSourceCount]) -> tuple[EditSourceCount, ...]:
    if not mix:
        raise InputError("edit source mix must contain at least one source")
    seen: set[str] = set()
    normalized: list[EditSourceCount] = []
    for entry in mix:
        if not isinstance(entry, EditSourceCount):
            raise InputError(
                "edit source mix entries must be EditSourceCount values",
                details={"type": type(entry).__name__},
            )
        if entry.source in seen:
            raise InputError(
                "edit source mix contains duplicate sources",
                details={"source": entry.source},
            )
        seen.add(entry.source)
        normalized.append(entry)
    if all(entry.count == 0 for entry in normalized):
        raise InputError("edit source mix must request at least one edit")
    return tuple(normalized)


def _normalize_nonempty_tuple(name: str, values: Sequence[str]) -> tuple[str, ...]:
    normalized: list[str] = []
    seen: set[str] = set()
    for value in values:
        _require_nonempty_str(name, value)
        if value not in seen:
            normalized.append(value)
            seen.add(value)
    return tuple(normalized)


def _require_edit_spec(value: EditSpec) -> EditSpec:
    if not isinstance(value, EditSpec):
        raise InputError(
            "variants must contain EditSpec values",
            details={"type": type(value).__name__},
        )
    return value


def _edit_key(chrom: str, pos: int, ref: str, alt: str) -> str:
    return f"{chrom}:{pos}:{ref}:{alt}"
