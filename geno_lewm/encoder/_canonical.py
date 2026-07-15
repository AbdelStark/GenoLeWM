# SPDX-License-Identifier: Apache-2.0
"""Exact canonical representation for live and persisted encoder states."""

from __future__ import annotations

import math
import struct
from typing import cast

from geno_lewm.errors import InputError


def canonical_fp32(value: float, *, field: str = "state") -> float:
    """Round one finite coordinate to IEEE-754 binary32, preserving its bits."""
    if type(value) is not float or not math.isfinite(value):
        raise InputError(
            f"{field} values must be finite floating-point values",
            details={"value": repr(value)},
        )
    try:
        return cast(float, struct.unpack("<f", struct.pack("<f", value))[0])
    except (OverflowError, struct.error) as exc:
        raise InputError(
            f"{field} values must be representable as canonical fp32",
            details={"value": value},
        ) from exc


def canonical_state_fp32(
    values: tuple[float, ...],
    *,
    field: str = "state",
) -> tuple[float, ...]:
    """Return the canonical binary32 representation of a complete state."""
    return tuple(canonical_fp32(value, field=field) for value in values)
