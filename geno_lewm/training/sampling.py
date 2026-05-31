# SPDX-License-Identifier: Apache-2.0
"""RFC-0005 edit-balanced and rollout-length samplers."""

from __future__ import annotations

import math
import random
from collections.abc import Sequence
from dataclasses import dataclass
from typing import TypeVar

from geno_lewm.action import EditType
from geno_lewm.errors import InputError

__all__ = [
    "DEFAULT_EDIT_TYPE_WEIGHTS",
    "DEFAULT_ROLLOUT_STEP_MIX",
    "EditTypeWeight",
    "RolloutStepWeight",
    "draw_edit_type_counts",
    "draw_rollout_step_counts",
    "sample_edit_type",
    "sample_rollout_steps",
]


def _validate_weight(name: str, value: float) -> None:
    if (
        isinstance(value, bool)
        or not isinstance(value, int | float)
        or not math.isfinite(float(value))
        or value <= 0.0
    ):
        raise InputError(
            f"{name} must be a positive finite number",
            details={"field": name, "value": value, "type": type(value).__name__},
        )


@dataclass(frozen=True, slots=True)
class EditTypeWeight:
    """One RFC-0005 edit-type sampling weight."""

    edit_type: EditType
    weight: float

    def __post_init__(self) -> None:
        if not isinstance(self.edit_type, EditType):
            try:  # type: ignore[unreachable]
                object.__setattr__(self, "edit_type", EditType(int(self.edit_type)))
            except (TypeError, ValueError) as exc:
                raise InputError(
                    "edit_type must be a supported EditType",
                    details={"edit_type": self.edit_type},
                ) from exc
        _validate_weight("weight", self.weight)


@dataclass(frozen=True, slots=True)
class RolloutStepWeight:
    """One rollout-step-count sampling weight."""

    steps: int
    weight: float

    def __post_init__(self) -> None:
        if not isinstance(self.steps, int) or isinstance(self.steps, bool) or self.steps < 1:
            raise InputError(
                "steps must be a positive integer",
                details={"steps": self.steps, "type": type(self.steps).__name__},
            )
        _validate_weight("weight", self.weight)


_WeightedEntry = TypeVar("_WeightedEntry", EditTypeWeight, RolloutStepWeight)


DEFAULT_EDIT_TYPE_WEIGHTS: tuple[EditTypeWeight, ...] = (
    EditTypeWeight(EditType.SNV, 0.40),
    EditTypeWeight(EditType.INS, 0.20),
    EditTypeWeight(EditType.DEL, 0.20),
    EditTypeWeight(EditType.MNV, 0.10),
    EditTypeWeight(EditType.INDEL, 0.10),
)

DEFAULT_ROLLOUT_STEP_MIX: tuple[RolloutStepWeight, ...] = (
    RolloutStepWeight(1, 0.90),
    RolloutStepWeight(2, 0.05),
    RolloutStepWeight(3, 0.05),
)


def sample_edit_type(
    rng: random.Random,
    *,
    weights: Sequence[EditTypeWeight] = DEFAULT_EDIT_TYPE_WEIGHTS,
) -> EditType:
    """Sample one edit type from the RFC-0005 edit-balanced distribution."""
    return _sample_weighted(rng, _validate_edit_type_weights(weights)).edit_type


def draw_edit_type_counts(
    n: int,
    *,
    rng: random.Random,
    weights: Sequence[EditTypeWeight] = DEFAULT_EDIT_TYPE_WEIGHTS,
) -> dict[EditType, int]:
    """Draw ``n`` edit types and return counts by :class:`EditType`."""
    _require_nonnegative_int("n", n)
    entries = _validate_edit_type_weights(weights)
    counts = {entry.edit_type: 0 for entry in entries}
    for _ in range(n):
        counts[_sample_weighted(rng, entries).edit_type] += 1
    return counts


def sample_rollout_steps(
    rng: random.Random,
    *,
    mix: Sequence[RolloutStepWeight] = DEFAULT_ROLLOUT_STEP_MIX,
) -> int:
    """Sample a rollout length ``K`` from the Phase-1 RFC-0005 mix."""
    return _sample_weighted(rng, _validate_rollout_mix(mix)).steps


def draw_rollout_step_counts(
    n: int,
    *,
    rng: random.Random,
    mix: Sequence[RolloutStepWeight] = DEFAULT_ROLLOUT_STEP_MIX,
) -> dict[int, int]:
    """Draw ``n`` rollout lengths and return counts by step count."""
    _require_nonnegative_int("n", n)
    entries = _validate_rollout_mix(mix)
    counts = {entry.steps: 0 for entry in entries}
    for _ in range(n):
        counts[_sample_weighted(rng, entries).steps] += 1
    return counts


def _sample_weighted(rng: random.Random, entries: Sequence[_WeightedEntry]) -> _WeightedEntry:
    total = sum(entry.weight for entry in entries)
    draw = rng.random() * total
    cumulative = 0.0
    for entry in entries:
        cumulative += entry.weight
        if draw < cumulative:
            return entry
    return entries[-1]


def _validate_edit_type_weights(weights: Sequence[EditTypeWeight]) -> tuple[EditTypeWeight, ...]:
    if not weights:
        raise InputError("edit type weights must contain at least one entry")
    entries: list[EditTypeWeight] = []
    seen: set[EditType] = set()
    for entry in weights:
        normalized = EditTypeWeight(entry.edit_type, entry.weight)
        if normalized.edit_type is EditType.SV:
            raise InputError(
                "SV edits are outside the v0.1 edit-balanced sampler",
                details={"edit_type": int(normalized.edit_type)},
            )
        if normalized.edit_type in seen:
            raise InputError(
                "edit type weights contain duplicate edit types",
                details={"edit_type": int(normalized.edit_type)},
            )
        seen.add(normalized.edit_type)
        entries.append(normalized)
    return tuple(entries)


def _validate_rollout_mix(mix: Sequence[RolloutStepWeight]) -> tuple[RolloutStepWeight, ...]:
    if not mix:
        raise InputError("rollout step mix must contain at least one entry")
    entries: list[RolloutStepWeight] = []
    seen: set[int] = set()
    for entry in mix:
        normalized = RolloutStepWeight(entry.steps, entry.weight)
        if normalized.steps in seen:
            raise InputError(
                "rollout step mix contains duplicate step counts",
                details={"steps": normalized.steps},
            )
        seen.add(normalized.steps)
        entries.append(normalized)
    return tuple(entries)


def _require_nonnegative_int(name: str, value: int) -> None:
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise InputError(
            f"{name} must be a non-negative integer",
            details={"field": name, "value": value, "type": type(value).__name__},
        )
