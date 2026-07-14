# SPDX-License-Identifier: Apache-2.0
"""Prepared finite data stream shared by real training and trace evidence.

The boundary deliberately prepares the finite window set before constructing
the epoch RNG.  Schema-1.1 release snapshots can therefore exclude windows
that cannot satisfy a required, non-fallback edit source without shifting the
random draws of any remaining window.
"""

from __future__ import annotations

import sys
from collections.abc import Callable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

from geno_lewm.data import (
    EditSourceCount,
    GenoLeWMDataset,
    HoldoutPolicy,
    TrainingDatasetItem,
    WindowContext,
)
from geno_lewm.data.builder import _provider_available_edit_count
from geno_lewm.errors import InputError

if TYPE_CHECKING:
    from geno_lewm.config import GenoLeWMConfig

_SCHEMA_1_1 = "1.1.0"


@dataclass(frozen=True, slots=True)
class PreparedWindowExclusion:
    """One finite release window rejected before epoch RNG construction."""

    record_id: str
    window_id: str
    chrom: str | None
    start_bp: int
    end_bp: int
    reason: str
    source: str | None
    required_count: int
    available_count: int

    def to_dict(self) -> dict[str, object]:
        return {
            "record_id": self.record_id,
            "window_id": self.window_id,
            "chrom": self.chrom,
            "start_bp": self.start_bp,
            "end_bp": self.end_bp,
            "reason": self.reason,
            "source": self.source,
            "required_count": self.required_count,
            "available_count": self.available_count,
        }


class PreparedTrainingStream:
    """Finite prepared windows plus the exact deterministic epoch contract."""

    def __init__(
        self,
        *,
        dataset_snapshot_id: str,
        schema_version: str,
        input_windows: tuple[WindowContext, ...],
        usable_windows: tuple[WindowContext, ...],
        exclusions: tuple[PreparedWindowExclusion, ...],
        providers: Mapping[str, Any],
        mix: tuple[EditSourceCount, ...],
        fallback_sources: Mapping[str, str],
        holdouts: HoldoutPolicy | None,
        membership_identity: Mapping[str, object] | None,
        seed: int,
        close_callback: Callable[[], None] | None = None,
    ) -> None:
        self.dataset_snapshot_id = dataset_snapshot_id
        self.schema_version = schema_version
        self.input_windows = input_windows
        self.usable_windows = usable_windows
        self.exclusions = exclusions
        self.providers = dict(providers)
        self.mix = mix
        self.fallback_sources = dict(fallback_sources)
        self.holdouts = holdouts
        self.membership_identity = (
            None if membership_identity is None else dict(membership_identity)
        )
        self.seed = seed
        self._close_callback = close_callback
        self._closed = False

    @classmethod
    def open(
        cls,
        *,
        dataset_dir: Path,
        config: GenoLeWMConfig,
        require_membership: bool = True,
    ) -> PreparedTrainingStream:
        """Open and verify one packaged dataset into the production stream."""
        # Import lazily so ``real`` can consume this module without a cycle.
        from geno_lewm.training import real as real_module

        manifest = real_module._load_dataset_manifest(dataset_dir)
        dataset_snapshot_id = real_module._required_text(manifest, "snapshot_id")
        schema_version = real_module._required_text(manifest, "schema_version")
        files = real_module._dataset_files(manifest)
        windows = tuple(
            real_module._load_windows(dataset_dir, files, schema_version=schema_version)
        )
        if not windows:
            raise InputError("Carbon training requires at least one source window")
        gnomad_edits = tuple(
            real_module._load_gnomad_edits(dataset_dir, files, schema_version=schema_version)
        )
        clinvar_edits = tuple(
            real_module._load_clinvar_edits(dataset_dir, files, schema_version=schema_version)
        )
        if not gnomad_edits:
            raise InputError("Carbon training requires at least one gnomAD edit")

        holdout_context = real_module._membership_holdout_policy(dataset_dir, manifest)
        holdouts = holdout_context.__enter__()
        try:
            membership_identity = real_module._membership_runtime_identity(manifest, holdouts)
            if require_membership and membership_identity is None:
                raise InputError(
                    "schema-1.1 training trace requires verified membership and split evidence"
                )
            providers, mix = real_module._training_edit_contract(
                config,
                gnomad_edits=gnomad_edits,
                clinvar_edits=clinvar_edits,
            )
            closed = False

            def _close() -> None:
                nonlocal closed
                if not closed:
                    closed = True
                    holdout_context.__exit__(None, None, None)

            return cls.from_components(
                dataset_snapshot_id=dataset_snapshot_id,
                schema_version=schema_version,
                windows=windows,
                providers=providers,
                mix=mix,
                fallback_sources=real_module._dataset_fallback_sources(windows),
                holdouts=holdouts,
                membership_identity=membership_identity,
                seed=config.seed,
                close_callback=_close,
            )
        except BaseException:
            holdout_context.__exit__(*sys.exc_info())
            raise

    @classmethod
    def from_components(
        cls,
        *,
        dataset_snapshot_id: str,
        schema_version: str,
        windows: Sequence[WindowContext],
        providers: Mapping[str, Any],
        mix: Sequence[EditSourceCount],
        fallback_sources: Mapping[str, str],
        holdouts: HoldoutPolicy | None,
        membership_identity: Mapping[str, object] | None,
        seed: int,
        close_callback: Callable[[], None] | None = None,
    ) -> PreparedTrainingStream:
        """Prepare a stream from already verified finite dataset components."""
        normalized_windows = tuple(windows)
        normalized_mix = tuple(mix)
        if schema_version == _SCHEMA_1_1:
            usable, exclusions = _filter_windows_before_rng(
                normalized_windows,
                providers=providers,
                mix=normalized_mix,
                fallback_sources=fallback_sources,
                holdouts=holdouts,
            )
        else:
            usable, exclusions = normalized_windows, ()
        if not usable:
            raise InputError(
                "prepared training stream contains no usable source windows",
                details={
                    "input_windows": len(normalized_windows),
                    "excluded_windows": len(exclusions),
                },
            )
        return cls(
            dataset_snapshot_id=dataset_snapshot_id,
            schema_version=schema_version,
            input_windows=normalized_windows,
            usable_windows=usable,
            exclusions=exclusions,
            providers=providers,
            mix=normalized_mix,
            fallback_sources=fallback_sources,
            holdouts=holdouts,
            membership_identity=membership_identity,
            seed=seed,
            close_callback=close_callback,
        )

    @property
    def input_window_count(self) -> int:
        return len(self.input_windows)

    @property
    def usable_window_count(self) -> int:
        return len(self.usable_windows)

    def iter_epoch(self, epoch: int) -> Iterator[TrainingDatasetItem]:
        """Yield one exact epoch, preserving trainer seed and source-mix order."""
        if self._closed:
            raise InputError("prepared training stream is closed")
        if isinstance(epoch, bool) or not isinstance(epoch, int) or epoch < 0:
            raise InputError("epoch must be a non-negative integer")
        dataset = GenoLeWMDataset(
            self.usable_windows,
            self.providers,
            seed=self.seed + epoch,
            fallback_sources=self.fallback_sources,
            mix=self.mix,
            holdouts=self.holdouts,
        )
        return dataset.iter_with_source_windows()

    def iter_repeated(self) -> Iterator[TrainingDatasetItem]:
        """Yield deterministic repeated finite epochs for the real trainer."""
        epoch = 0
        while True:
            produced = 0
            for item in self.iter_epoch(epoch):
                produced += 1
                yield item
            if produced == 0:
                raise InputError(
                    "training dataset epoch produced no usable tuples",
                    details={"epoch": epoch, "window_count": self.usable_window_count},
                )
            epoch += 1

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        if self._close_callback is not None:
            self._close_callback()

    def __enter__(self) -> PreparedTrainingStream:
        if self._closed:
            raise InputError("prepared training stream is closed")
        return self

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> None:
        del exc_type, exc, traceback
        self.close()


def _filter_windows_before_rng(
    windows: tuple[WindowContext, ...],
    *,
    providers: Mapping[str, Any],
    mix: tuple[EditSourceCount, ...],
    fallback_sources: Mapping[str, str],
    holdouts: HoldoutPolicy | None,
) -> tuple[tuple[WindowContext, ...], tuple[PreparedWindowExclusion, ...]]:
    active_holdouts = holdouts if holdouts is not None else HoldoutPolicy()
    required_sources = tuple(
        entry for entry in mix if entry.count > 0 and entry.source not in fallback_sources
    )
    usable: list[WindowContext] = []
    exclusions: list[PreparedWindowExclusion] = []
    for window in windows:
        if active_holdouts.excludes_window(window):
            exclusions.append(
                _exclusion(
                    window,
                    reason="holdout_policy",
                    source=None,
                    required_count=0,
                    available_count=0,
                )
            )
            continue
        rejected: PreparedWindowExclusion | None = None
        for entry in required_sources:
            provider = providers.get(entry.source)
            if not callable(provider):
                raise InputError(
                    "training edit provider is missing",
                    details={"source": entry.source},
                )
            available = _provider_available_edit_count(
                provider,
                window,
                entry.count,
                holdouts=active_holdouts,
            )
            if available < entry.count:
                rejected = _exclusion(
                    window,
                    reason="required_source_insufficient",
                    source=entry.source,
                    required_count=entry.count,
                    available_count=available,
                )
                break
        if rejected is None:
            usable.append(window)
        else:
            exclusions.append(rejected)
    return tuple(usable), tuple(exclusions)


def _exclusion(
    window: WindowContext,
    *,
    reason: str,
    source: str | None,
    required_count: int,
    available_count: int,
) -> PreparedWindowExclusion:
    return PreparedWindowExclusion(
        record_id=window.record_id,
        window_id=window.window_id,
        chrom=window.chrom,
        start_bp=window.start_bp,
        end_bp=window.end_bp,
        reason=reason,
        source=source,
        required_count=required_count,
        available_count=available_count,
    )
