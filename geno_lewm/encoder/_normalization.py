# SPDX-License-Identifier: Apache-2.0
"""Internal state-vector normalization shared by live and cached encoders."""

from __future__ import annotations

import math
from collections.abc import Sequence

from geno_lewm.errors import InputError


def l2_normalize_state(
    vector: Sequence[float],
    *,
    item_index: int | None = None,
) -> tuple[float, ...]:
    """Return a unit-norm state vector and reject undefined normalization."""
    values = tuple(float(value) for value in vector)
    norm = math.hypot(*values)
    if not math.isfinite(norm) or norm == 0.0:
        details: dict[str, object] = {"norm": norm}
        if item_index is not None:
            details["item_index"] = item_index
        raise InputError(
            "pooled encoder state must have a finite non-zero norm",
            details=details,
        )
    return tuple(value / norm for value in values)
