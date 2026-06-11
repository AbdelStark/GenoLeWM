# SPDX-License-Identifier: Apache-2.0
"""Factored edit sampler for planning contract latent planning."""

from __future__ import annotations

import math
import random
from collections.abc import Mapping, Sequence

from geno_lewm.action import DEFAULT_EDGE_MARGIN, V1_MAX_LEN, EditType, RelEdit
from geno_lewm.errors import InputError
from geno_lewm.training.sampling import DEFAULT_EDIT_TYPE_WEIGHTS, EditTypeWeight

__all__ = [
    "DEFAULT_ACTION_TYPE_WEIGHTS",
    "ActionSampler",
]

DEFAULT_ACTION_TYPE_WEIGHTS: tuple[EditTypeWeight, ...] = DEFAULT_EDIT_TYPE_WEIGHTS
"""Initial planner proposal over ``{SNV, INS, DEL, MNV, INDEL}``."""

_BASES: tuple[str, ...] = ("A", "C", "G", "T")
_OTHER_BASE: dict[str, tuple[str, str, str]] = {
    "A": ("C", "G", "T"),
    "C": ("A", "G", "T"),
    "G": ("A", "C", "T"),
    "T": ("A", "C", "G"),
}


class ActionSampler:
    """Sample valid ``RelEdit`` actions from a factored proposal.

    The proposal follows planning contract: edit type is categorical,
    position is uniform or binned-categorical over the window interior,
    and bases are sampled conditionally on the chosen edit type.
    """

    def __init__(
        self,
        window: str,
        *,
        seed: int | None = None,
        rng: random.Random | None = None,
        edge_margin: int = DEFAULT_EDGE_MARGIN,
        type_weights: Sequence[EditTypeWeight] = DEFAULT_ACTION_TYPE_WEIGHTS,
        length_dist: Mapping[int, float] | Sequence[float] | None = None,
        position_bin_bp: int = 8,
        position_weights: Mapping[int, float] | Sequence[float] | None = None,
        max_attempts: int = 256,
    ) -> None:
        if rng is not None and seed is not None:
            raise InputError("pass either rng or seed, not both")
        _validate_window(window, edge_margin)
        _require_positive_int("position_bin_bp", position_bin_bp)
        _require_positive_int("max_attempts", max_attempts)

        self.window = window
        self.edge_margin = edge_margin
        self.position_bin_bp = position_bin_bp
        self.max_attempts = max_attempts
        self._rng = rng if rng is not None else random.Random(seed)
        self._type_weights = _normalize_type_weights(type_weights)
        self._length_dist = length_dist
        self._position_weights = _normalize_position_weights(position_weights)

    def sample_edit(self, edit_type: EditType | int | None = None) -> RelEdit:
        """Sample one shape-consistent edit inside the configured window."""
        normalized_type = self._sample_type() if edit_type is None else _normalize_type(edit_type)

        for _attempt in range(self.max_attempts):
            edit = self._sample_for_type(normalized_type)
            if edit is not None:
                return edit

        raise InputError(
            "could not sample a valid edit from the window interior",
            details={
                "window_len": len(self.window),
                "edge_margin": self.edge_margin,
                "edit_type": int(normalized_type),
                "max_attempts": self.max_attempts,
            },
            remediation="provide a longer ACGT window or reduce edge_margin",
        )

    def sample_sequence(self, horizon: int) -> tuple[RelEdit, ...]:
        """Sample a candidate edit sequence of length ``horizon``."""
        _require_nonnegative_int("horizon", horizon)
        return tuple(self.sample_edit() for _ in range(horizon))

    def sample_sequences(self, n: int, horizon: int) -> tuple[tuple[RelEdit, ...], ...]:
        """Sample ``n`` candidate edit sequences."""
        _require_nonnegative_int("n", n)
        _require_nonnegative_int("horizon", horizon)
        return tuple(self.sample_sequence(horizon) for _ in range(n))

    def _sample_type(self) -> EditType:
        total = sum(entry.weight for entry in self._type_weights)
        draw = self._rng.random() * total
        cumulative = 0.0
        for entry in self._type_weights:
            cumulative += entry.weight
            if draw < cumulative:
                return entry.edit_type
        return self._type_weights[-1].edit_type

    def _sample_for_type(self, edit_type: EditType) -> RelEdit | None:
        if edit_type is EditType.SNV:
            return self._sample_snv()
        if edit_type is EditType.INS:
            return self._sample_ins()
        if edit_type is EditType.DEL:
            return self._sample_del()
        if edit_type is EditType.MNV:
            return self._sample_mnv()
        if edit_type is EditType.INDEL:
            return self._sample_indel()
        raise InputError(
            "SV edits are outside the v1 planning sampler",
            details={"edit_type": int(edit_type)},
        )

    def _sample_snv(self) -> RelEdit | None:
        pos = self._pick_position(ref_len=1)
        if pos is None:
            return None
        ref = self.window[pos]
        if ref not in _OTHER_BASE:
            return None
        alt = self._rng.choice(_OTHER_BASE[ref])
        return RelEdit(rel_pos=pos, edit_type=EditType.SNV, ref_bases=ref, alt_bases=alt)

    def _sample_ins(self) -> RelEdit | None:
        pos = self._pick_position(ref_len=1)
        if pos is None:
            return None
        ref = self.window[pos]
        if ref not in _OTHER_BASE:
            return None
        event_len = self._draw_length(min_len=1, max_len=V1_MAX_LEN - 1)
        alt = ref + self._rand_bases(event_len)
        return RelEdit(rel_pos=pos, edit_type=EditType.INS, ref_bases=ref, alt_bases=alt)

    def _sample_del(self) -> RelEdit | None:
        event_len = self._draw_length(min_len=1, max_len=V1_MAX_LEN - 1)
        ref_len = 1 + event_len
        pos = self._pick_position(ref_len=ref_len)
        if pos is None:
            return None
        ref = self._window_segment(pos, ref_len)
        if ref is None:
            return None
        return RelEdit(rel_pos=pos, edit_type=EditType.DEL, ref_bases=ref, alt_bases=ref[0])

    def _sample_mnv(self) -> RelEdit | None:
        length = self._draw_length(min_len=2, max_len=V1_MAX_LEN)
        pos = self._pick_position(ref_len=length)
        if pos is None:
            return None
        ref = self._window_segment(pos, length)
        if ref is None:
            return None
        alt = "".join(self._rng.choice(_OTHER_BASE[base]) for base in ref)
        return RelEdit(rel_pos=pos, edit_type=EditType.MNV, ref_bases=ref, alt_bases=alt)

    def _sample_indel(self) -> RelEdit | None:
        ref_event_len = self._draw_length(min_len=1, max_len=V1_MAX_LEN - 1)
        alt_event_len = self._draw_distinct_length(
            ref_event_len,
            min_len=1,
            max_len=V1_MAX_LEN - 1,
        )
        ref_len = 1 + ref_event_len
        pos = self._pick_position(ref_len=ref_len)
        if pos is None:
            return None
        ref = self._window_segment(pos, ref_len)
        if ref is None:
            return None
        alt = ref[0] + self._rand_bases(alt_event_len)
        return RelEdit(rel_pos=pos, edit_type=EditType.INDEL, ref_bases=ref, alt_bases=alt)

    def _pick_position(self, *, ref_len: int) -> int | None:
        lo = self.edge_margin
        hi = len(self.window) - self.edge_margin - ref_len
        if hi < lo:
            return None

        if self._position_weights is None:
            return self._rng.randint(lo, hi)

        for _attempt in range(self.max_attempts):
            bin_index = self._sample_position_bin()
            bin_lo = self.edge_margin + bin_index * self.position_bin_bp
            bin_hi = min(bin_lo + self.position_bin_bp - 1, hi)
            if bin_lo <= hi and bin_hi >= lo:
                return self._rng.randint(max(lo, bin_lo), bin_hi)

        return None

    def _sample_position_bin(self) -> int:
        if self._position_weights is None:
            raise InputError("position weights are not configured")

        total = sum(weight for _bin_index, weight in self._position_weights)
        draw = self._rng.random() * total
        cumulative = 0.0
        for bin_index, weight in self._position_weights:
            cumulative += weight
            if draw < cumulative:
                return bin_index
        return self._position_weights[-1][0]

    def _draw_length(self, *, min_len: int, max_len: int) -> int:
        return _draw_length(
            self._rng,
            self._length_dist,
            min_len=min_len,
            max_len=max_len,
        )

    def _draw_distinct_length(self, current: int, *, min_len: int, max_len: int) -> int:
        if min_len == max_len:
            raise InputError(
                "cannot sample a distinct length from a singleton distribution",
                details={"length": current},
            )
        for _attempt in range(self.max_attempts):
            candidate = self._draw_length(min_len=min_len, max_len=max_len)
            if candidate != current:
                return candidate
        raise InputError(
            "length_dist has no distinct positive mass for INDEL sampling",
            details={"current": current, "min_len": min_len, "max_len": max_len},
        )

    def _window_segment(self, pos: int, length: int) -> str | None:
        segment = self.window[pos : pos + length]
        if len(segment) != length or any(base not in _OTHER_BASE for base in segment):
            return None
        return segment

    def _rand_bases(self, n: int) -> str:
        return "".join(self._rng.choice(_BASES) for _ in range(n))


def _validate_window(window: str, edge_margin: int) -> None:
    if not isinstance(window, str) or not window:
        raise InputError(
            "window must be a non-empty string",
            details={"type": type(window).__name__},
        )
    if window != window.upper():
        raise InputError("window must be uppercase", remediation="call window.upper() first")
    _require_nonnegative_int("edge_margin", edge_margin)
    if 2 * edge_margin >= len(window):
        raise InputError(
            "edge_margin leaves no interior region",
            details={"window_len": len(window), "edge_margin": edge_margin},
            remediation="reduce edge_margin or use a longer window",
        )


def _normalize_type_weights(weights: Sequence[EditTypeWeight]) -> tuple[EditTypeWeight, ...]:
    if not weights:
        raise InputError("edit type weights must contain at least one entry")

    normalized: list[EditTypeWeight] = []
    seen: set[EditType] = set()
    for entry in weights:
        item = EditTypeWeight(entry.edit_type, entry.weight)
        if item.edit_type is EditType.SV:
            raise InputError(
                "SV edits are outside the v1 planning sampler",
                details={"edit_type": int(item.edit_type)},
            )
        if item.edit_type in seen:
            raise InputError(
                "edit type weights contain duplicate edit types",
                details={"edit_type": int(item.edit_type)},
            )
        seen.add(item.edit_type)
        normalized.append(item)
    return tuple(normalized)


def _normalize_position_weights(
    weights: Mapping[int, float] | Sequence[float] | None,
) -> tuple[tuple[int, float], ...] | None:
    if weights is None:
        return None

    if isinstance(weights, Mapping):
        raw_entries = tuple((bin_index, weight) for bin_index, weight in weights.items())
    else:
        raw_entries = tuple((idx, weight) for idx, weight in enumerate(weights))

    if not raw_entries:
        raise InputError("position weights must contain at least one entry")

    normalized: list[tuple[int, float]] = []
    seen: set[int] = set()
    for raw_bin, raw_weight in raw_entries:
        if not isinstance(raw_bin, int) or isinstance(raw_bin, bool) or raw_bin < 0:
            raise InputError(
                "position bin index must be a non-negative integer",
                details={"bin_index": raw_bin, "type": type(raw_bin).__name__},
            )
        if raw_bin in seen:
            raise InputError(
                "position weights contain duplicate bins",
                details={"bin_index": raw_bin},
            )
        seen.add(raw_bin)
        normalized.append((raw_bin, _validate_nonnegative_weight("position weight", raw_weight)))

    if sum(weight for _bin, weight in normalized) <= 0.0:
        raise InputError("position weights must sum to a positive value")
    return tuple(sorted(normalized))


def _draw_length(
    rng: random.Random,
    length_dist: Mapping[int, float] | Sequence[float] | None,
    *,
    min_len: int,
    max_len: int,
) -> int:
    if min_len < 1 or max_len < min_len:
        raise InputError(
            "invalid length bounds",
            details={"min_len": min_len, "max_len": max_len},
        )

    if length_dist is None:
        entries = tuple((length, 0.5**length) for length in range(min_len, max_len + 1))
    elif isinstance(length_dist, Mapping):
        entries = _mapping_length_entries(length_dist, min_len=min_len, max_len=max_len)
    else:
        entries = _sequence_length_entries(length_dist, min_len=min_len, max_len=max_len)

    total = sum(weight for _length, weight in entries)
    if total <= 0.0:
        raise InputError(
            "length_dist has no positive mass in the requested bounds",
            details={"min_len": min_len, "max_len": max_len},
        )

    draw = rng.random() * total
    cumulative = 0.0
    for length, weight in entries:
        cumulative += weight
        if draw < cumulative:
            return length
    return entries[-1][0]


def _mapping_length_entries(
    length_dist: Mapping[int, float],
    *,
    min_len: int,
    max_len: int,
) -> tuple[tuple[int, float], ...]:
    entries: list[tuple[int, float]] = []
    for raw_length, raw_weight in length_dist.items():
        length = _normalize_length(raw_length)
        weight = _validate_nonnegative_weight("length weight", raw_weight)
        if min_len <= length <= max_len:
            entries.append((length, weight))
    if not entries:
        raise InputError(
            "length_dist does not overlap the requested bounds",
            details={"min_len": min_len, "max_len": max_len},
        )
    return tuple(sorted(entries))


def _sequence_length_entries(
    length_dist: Sequence[float],
    *,
    min_len: int,
    max_len: int,
) -> tuple[tuple[int, float], ...]:
    entries: list[tuple[int, float]] = []
    for idx, raw_weight in enumerate(length_dist, start=1):
        weight = _validate_nonnegative_weight("length weight", raw_weight)
        if min_len <= idx <= max_len:
            entries.append((idx, weight))
    if not entries:
        raise InputError(
            "length_dist does not overlap the requested bounds",
            details={"min_len": min_len, "max_len": max_len},
        )
    return tuple(entries)


def _normalize_type(value: EditType | int) -> EditType:
    if isinstance(value, EditType):
        edit_type = value
    else:
        try:
            edit_type = EditType(int(value))
        except (TypeError, ValueError) as exc:
            raise InputError(
                "edit_type must be an EditType member",
                details={"edit_type": value},
            ) from exc
    if edit_type is EditType.SV:
        raise InputError(
            "SV edits are outside the v1 planning sampler",
            details={"edit_type": int(edit_type)},
        )
    return edit_type


def _normalize_length(value: int) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < 1:
        raise InputError(
            "length_dist keys must be positive integers",
            details={"length": value, "type": type(value).__name__},
        )
    return value


def _validate_nonnegative_weight(name: str, value: float) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, int | float)
        or not math.isfinite(float(value))
        or value < 0.0
    ):
        raise InputError(
            f"{name} must be a non-negative finite number",
            details={"value": value, "type": type(value).__name__},
        )
    return float(value)


def _require_nonnegative_int(name: str, value: int) -> None:
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise InputError(
            f"{name} must be a non-negative integer",
            details={"field": name, "value": value, "type": type(value).__name__},
        )


def _require_positive_int(name: str, value: int) -> None:
    if not isinstance(value, int) or isinstance(value, bool) or value < 1:
        raise InputError(
            f"{name} must be a positive integer",
            details={"field": name, "value": value, "type": type(value).__name__},
        )
