# SPDX-License-Identifier: Apache-2.0
"""Edit-sequence cost functions for RFC-0008 planning."""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping, Sequence
from types import MappingProxyType

from geno_lewm.action import EditType, RelEdit
from geno_lewm.errors import InputError

__all__ = [
    "DEFAULT_TYPE_COSTS",
    "bp_cost",
    "count_cost",
    "custom_cost",
    "edit_bp_cost",
    "weighted_type_cost",
]

_DEFAULT_TYPE_COSTS: dict[EditType, float] = {
    EditType.SNV: 1.0,
    EditType.INS: 2.0,
    EditType.DEL: 2.0,
    EditType.MNV: 2.0,
    EditType.INDEL: 3.0,
}

DEFAULT_TYPE_COSTS: Mapping[EditType, float] = MappingProxyType(_DEFAULT_TYPE_COSTS)
"""Default non-negative type costs for ``weighted_type_cost``.

The defaults keep SNVs cheapest, assign a higher penalty to simple
indels and MNVs, and make mixed indels the most expensive v1 edit
class. Structural variants are outside the v1 planner surface.
"""


def count_cost(edits: Sequence[RelEdit]) -> float:
    """Return the number of edits in a candidate sequence."""
    return float(len(edits))


def edit_bp_cost(edit: RelEdit) -> float:
    """Return the base-pair cost contribution of a single edit.

    SNVs cost one base. Insertions and deletions cost their event
    length excluding the VCF anchor base. MNVs cost their substituted
    span. Mixed indels cost the larger touched span because both the
    deleted and inserted sequence are material to the action.
    """
    _require_shape_consistent(edit)
    ref_len = len(edit.ref_bases)
    alt_len = len(edit.alt_bases)

    if edit.edit_type is EditType.SNV:
        return 1.0
    if edit.edit_type is EditType.INS:
        return float(alt_len - ref_len)
    if edit.edit_type is EditType.DEL:
        return float(ref_len - alt_len)
    if edit.edit_type is EditType.MNV:
        return float(ref_len)
    if edit.edit_type is EditType.INDEL:
        return float(max(ref_len, alt_len))

    raise InputError(
        "SV edits are outside the v1 planning cost surface",
        details={"edit_type": int(edit.edit_type)},
    )


def bp_cost(edits: Sequence[RelEdit]) -> float:
    """Return the total base-pair cost for ``edits``."""
    return sum(edit_bp_cost(edit) for edit in edits)


def weighted_type_cost(
    edits: Sequence[RelEdit],
    weights: Mapping[EditType, float] = DEFAULT_TYPE_COSTS,
) -> float:
    """Return the sum of per-edit-type costs for ``edits``."""
    normalized = _validate_type_costs(weights)
    total = 0.0
    for edit in edits:
        _require_shape_consistent(edit)
        try:
            total += normalized[edit.edit_type]
        except KeyError as exc:
            raise InputError(
                "missing cost weight for edit type",
                details={"edit_type": int(edit.edit_type)},
                remediation="provide a non-negative finite cost for every sampled edit type",
            ) from exc
    return total


def custom_cost(edits: Sequence[RelEdit], cost_fn: Callable[[Sequence[RelEdit]], float]) -> float:
    """Evaluate and validate a user-provided cost function."""
    value = cost_fn(tuple(edits))
    return _validate_cost_value("custom cost", value)


def _validate_type_costs(weights: Mapping[EditType, float]) -> dict[EditType, float]:
    if not weights:
        raise InputError("type costs must contain at least one entry")

    normalized: dict[EditType, float] = {}
    for raw_type, raw_weight in weights.items():
        edit_type = _normalize_edit_type(raw_type)
        if edit_type is EditType.SV:
            raise InputError(
                "SV edits are outside the v1 planning cost surface",
                details={"edit_type": int(edit_type)},
            )
        normalized[edit_type] = _validate_cost_value("type cost", raw_weight)
    return normalized


def _validate_cost_value(name: str, value: float) -> float:
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


def _normalize_edit_type(value: EditType | int) -> EditType:
    if isinstance(value, EditType):
        return value
    try:
        return EditType(int(value))
    except (TypeError, ValueError) as exc:
        raise InputError(
            "edit type must be an EditType member",
            details={"edit_type": value},
        ) from exc


def _require_shape_consistent(edit: RelEdit) -> None:
    if edit.ref_bases == edit.alt_bases:
        raise InputError(
            "planning costs require non-identity edits",
            details={"rel_pos": edit.rel_pos, "edit_type": int(edit.edit_type)},
        )

    ref_len = len(edit.ref_bases)
    alt_len = len(edit.alt_bases)
    expected = {
        EditType.SNV: ref_len == 1 and alt_len == 1,
        EditType.INS: ref_len == 1 and alt_len > 1,
        EditType.DEL: ref_len > 1 and alt_len == 1,
        EditType.MNV: ref_len == alt_len and ref_len > 1,
        EditType.INDEL: ref_len > 1 and alt_len > 1 and ref_len != alt_len,
    }
    if edit.edit_type is EditType.SV or not expected.get(edit.edit_type, False):
        raise InputError(
            "edit_type does not match ref/alt lengths",
            details={
                "edit_type": int(edit.edit_type),
                "ref_len": ref_len,
                "alt_len": alt_len,
            },
        )
