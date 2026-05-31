# SPDX-License-Identifier: Apache-2.0
"""Pooling strategies for Carbon hidden states.

Defined by RFC-0002 §3.4. This module is intentionally independent of
torch so the pooling contract, cache metadata, and downstream schema
behavior can be validated before the Carbon runtime wrapper lands.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Literal

from geno_lewm.encoder.windowing import CARBON_TOKEN_BP
from geno_lewm.errors import InputError

__all__ = [
    "DEFAULT_POOL_RADIUS_TOKENS",
    "POOL_CENTERED_MEAN",
    "POOL_GLOBAL_MEAN",
    "SUPPORTED_POOL_TYPES",
    "PoolingResult",
    "centered_mean",
    "global_mean",
    "pool_hidden_states",
]


POOL_CENTERED_MEAN: Literal["centered_mean"] = "centered_mean"
POOL_GLOBAL_MEAN: Literal["global_mean"] = "global_mean"
SUPPORTED_POOL_TYPES: tuple[Literal["centered_mean"], Literal["global_mean"]] = (
    POOL_CENTERED_MEAN,
    POOL_GLOBAL_MEAN,
)
DEFAULT_POOL_RADIUS_TOKENS = 256


@dataclass(frozen=True, slots=True)
class PoolingResult:
    """Pooled state vector plus cache-key metadata."""

    vector: tuple[float, ...]
    pool_type: Literal["centered_mean", "global_mean"]
    pool_radius: int
    untargeted: bool
    center_token: int | None
    token_count: int

    @property
    def d_state(self) -> int:
        """Return the pooled vector width."""
        return len(self.vector)

    def as_cache_fields(self) -> Mapping[str, object]:
        """Return fields shared with the window-cache schema."""
        return {
            "pool_type": self.pool_type,
            "pool_radius": self.pool_radius,
            "untargeted": self.untargeted,
        }


def global_mean(hidden_states: Sequence[Sequence[float]]) -> tuple[float, ...]:
    """Mean-pool every token vector in ``hidden_states``."""
    rows = _coerce_hidden_states(hidden_states)
    return _mean_rows(rows)


def centered_mean(
    hidden_states: Sequence[Sequence[float]],
    *,
    center_token: int,
    pool_radius: int = DEFAULT_POOL_RADIUS_TOKENS,
) -> tuple[float, ...]:
    """Mean-pool the inclusive token span ``center_token ± pool_radius``."""
    rows = _coerce_hidden_states(hidden_states)
    center = _validate_center_token(center_token, len(rows))
    radius = _validate_pool_radius(pool_radius)

    start = max(0, center - radius)
    end = min(len(rows), center + radius + 1)
    return _mean_rows(rows[start:end])


def pool_hidden_states(
    hidden_states: Sequence[Sequence[float]],
    *,
    edit_locus: int | None = None,
    pool_type: Literal["centered_mean", "global_mean"] = POOL_CENTERED_MEAN,
    pool_radius: int = DEFAULT_POOL_RADIUS_TOKENS,
    token_bp: int = CARBON_TOKEN_BP,
) -> PoolingResult:
    """Pool token-level hidden states into a state vector.

    ``edit_locus`` is a 0-based base-pair offset within the encoder
    window. When it is absent, RFC-0002 requires a global-mean fallback
    tagged as ``untargeted=True`` so cache consumers do not mix arbitrary
    reference-window embeddings with edit-local embeddings.
    """
    rows = _coerce_hidden_states(hidden_states)
    requested_type = _validate_pool_type(pool_type)
    radius = _validate_pool_radius(pool_radius)

    if edit_locus is None:
        return PoolingResult(
            vector=_mean_rows(rows),
            pool_type=POOL_GLOBAL_MEAN,
            pool_radius=0,
            untargeted=True,
            center_token=None,
            token_count=len(rows),
        )

    center_token = _edit_locus_to_token(edit_locus, token_count=len(rows), token_bp=token_bp)
    if requested_type == POOL_GLOBAL_MEAN:
        return PoolingResult(
            vector=_mean_rows(rows),
            pool_type=POOL_GLOBAL_MEAN,
            pool_radius=0,
            untargeted=False,
            center_token=None,
            token_count=len(rows),
        )

    return PoolingResult(
        vector=centered_mean(rows, center_token=center_token, pool_radius=radius),
        pool_type=POOL_CENTERED_MEAN,
        pool_radius=radius,
        untargeted=False,
        center_token=center_token,
        token_count=len(rows),
    )


def _coerce_hidden_states(
    hidden_states: Sequence[Sequence[float]],
) -> tuple[tuple[float, ...], ...]:
    rows: list[tuple[float, ...]] = []
    width: int | None = None
    for row_index, row in enumerate(hidden_states):
        values = tuple(_finite_float(value, row_index=row_index) for value in row)
        if not values:
            raise InputError(
                "hidden state rows must be non-empty",
                details={"row_index": row_index},
            )
        if width is None:
            width = len(values)
        elif len(values) != width:
            raise InputError(
                "hidden state rows must have a consistent width",
                details={"row_index": row_index, "expected": width, "observed": len(values)},
            )
        rows.append(values)

    if not rows:
        raise InputError("hidden_states must contain at least one token vector")
    return tuple(rows)


def _finite_float(value: float, *, row_index: int) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError) as exc:
        raise InputError(
            "hidden state values must be numeric",
            details={"row_index": row_index, "value": repr(value)},
        ) from exc
    if not math.isfinite(out):
        raise InputError(
            "hidden state values must be finite",
            details={"row_index": row_index, "value": out},
        )
    return out


def _mean_rows(rows: Sequence[Sequence[float]]) -> tuple[float, ...]:
    width = len(rows[0])
    totals = [0.0] * width
    for row in rows:
        for col, value in enumerate(row):
            totals[col] += value
    denom = float(len(rows))
    return tuple(total / denom for total in totals)


def _validate_center_token(center_token: int, token_count: int) -> int:
    if not isinstance(center_token, int) or isinstance(center_token, bool):
        raise InputError(
            "center_token must be an integer",
            details={"center_token": center_token, "type": type(center_token).__name__},
        )
    if center_token < 0 or center_token >= token_count:
        raise InputError(
            "center_token falls outside hidden_states",
            details={"center_token": center_token, "token_count": token_count},
        )
    return center_token


def _validate_pool_radius(pool_radius: int) -> int:
    if not isinstance(pool_radius, int) or isinstance(pool_radius, bool):
        raise InputError(
            "pool_radius must be an integer",
            details={"pool_radius": pool_radius, "type": type(pool_radius).__name__},
        )
    if pool_radius < 0:
        raise InputError(
            "pool_radius must be non-negative",
            details={"pool_radius": pool_radius},
        )
    return pool_radius


def _validate_pool_type(pool_type: str) -> Literal["centered_mean", "global_mean"]:
    if pool_type == POOL_CENTERED_MEAN:
        return POOL_CENTERED_MEAN
    if pool_type == POOL_GLOBAL_MEAN:
        return POOL_GLOBAL_MEAN
    raise InputError(
        "unsupported pool_type",
        details={"pool_type": pool_type, "supported": list(SUPPORTED_POOL_TYPES)},
        remediation="use centered_mean or global_mean; attention pooling is deferred",
    )


def _edit_locus_to_token(edit_locus: int, *, token_count: int, token_bp: int) -> int:
    if not isinstance(edit_locus, int) or isinstance(edit_locus, bool):
        raise InputError(
            "edit_locus must be an integer offset",
            details={"edit_locus": edit_locus, "type": type(edit_locus).__name__},
        )
    if edit_locus < 0:
        raise InputError(
            "edit_locus must be non-negative",
            details={"edit_locus": edit_locus},
        )
    if not isinstance(token_bp, int) or isinstance(token_bp, bool) or token_bp <= 0:
        raise InputError(
            "token_bp must be a positive integer",
            details={"token_bp": token_bp, "type": type(token_bp).__name__},
        )
    token = edit_locus // token_bp
    if token >= token_count:
        raise InputError(
            "edit_locus maps outside hidden_states",
            details={"edit_locus": edit_locus, "token_bp": token_bp, "token_count": token_count},
        )
    return token
