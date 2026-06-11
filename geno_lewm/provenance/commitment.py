# SPDX-License-Identifier: Apache-2.0
"""Input commitments for scoring calls (artifact-provenance contract).

For every inference call the inputs are committed via:

    input_commitment = SHA-256(canonical_serialize(
        reference_window || edit_spec || pooling_config || dtype_config
    ))

Two runs with identical inputs produce identical commitments; any
difference — even nominally-equivalent (different state layer,
different pool radius, different dtype) — produces a distinct one.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from geno_lewm.action.spec import EditSpec
from geno_lewm.errors import InputError
from geno_lewm.provenance.hashing import canonical_json_sha256

__all__ = ["DtypeConfig", "PoolingConfig", "compute_input_commitment"]


@dataclass(frozen=True, slots=True)
class PoolingConfig:
    """State-encoder pooling configuration commit shape (encoder contract)."""

    state_layer: int
    pool_type: str  # "centered_mean" | "mean" | "max" | …
    pool_radius: int
    normalize: bool

    def __post_init__(self) -> None:
        if self.state_layer < 0:
            raise InputError(
                "state_layer must be non-negative",
                details={"state_layer": self.state_layer},
            )
        if not self.pool_type:
            raise InputError("pool_type must be non-empty", details={"pool_type": self.pool_type})
        if self.pool_radius < 0:
            raise InputError(
                "pool_radius must be non-negative",
                details={"pool_radius": self.pool_radius},
            )


@dataclass(frozen=True, slots=True)
class DtypeConfig:
    """Numerical-precision commit shape."""

    encoder_dtype: str  # "bf16" | "fp16" | "fp32" | "int8" | "int4"
    predictor_dtype: str

    def __post_init__(self) -> None:
        if not self.encoder_dtype:
            raise InputError(
                "encoder_dtype must be non-empty",
                details={"encoder_dtype": self.encoder_dtype},
            )
        if not self.predictor_dtype:
            raise InputError(
                "predictor_dtype must be non-empty",
                details={"predictor_dtype": self.predictor_dtype},
            )


def _editspec_to_commit_dict(e: EditSpec) -> dict[str, Any]:
    return {
        "chrom": e.chrom,
        "pos": e.pos,
        "ref": e.ref,
        "alt": e.alt,
        # ``edit_type`` is derived from ref/alt, but include it so a
        # bug in the deriver flips the commitment loud.
        "edit_type": int(e.edit_type),
    }


def compute_input_commitment(
    reference_window: str,
    edit_spec: EditSpec,
    pooling_config: PoolingConfig,
    dtype_config: DtypeConfig,
) -> str:
    """Return the ``"sha256:<hex>"`` input commitment for a scoring call.

    The canonical payload is a dict with fixed keys; canonical-JSON
    encoding handles ordering and stability.
    """
    if not isinstance(reference_window, str):
        raise InputError(
            "reference_window must be a string of bases",
            details={"type": type(reference_window).__name__},
        )
    if not reference_window:
        raise InputError(
            "reference_window must be non-empty",
            details={"len": 0},
        )

    payload = {
        "reference_window": reference_window,
        "edit_spec": _editspec_to_commit_dict(edit_spec),
        "pooling_config": {
            "state_layer": pooling_config.state_layer,
            "pool_type": pooling_config.pool_type,
            "pool_radius": pooling_config.pool_radius,
            "normalize": pooling_config.normalize,
        },
        "dtype_config": {
            "encoder_dtype": dtype_config.encoder_dtype,
            "predictor_dtype": dtype_config.predictor_dtype,
        },
        "version": 1,
    }
    return canonical_json_sha256(payload)
