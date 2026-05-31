# SPDX-License-Identifier: Apache-2.0
"""Prediction and LeJEPA monitoring losses for GenoLeWM predictors."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Literal

from geno_lewm.errors import InputError, RuntimeSetupError

__all__ = [
    "PredictionLossResult",
    "lejepa_kl_regularizer",
    "prediction_loss",
    "predictor_loss",
]

try:  # pragma: no cover - exercised by optional-runtime tests with torch installed.
    import torch  # type: ignore[import-not-found]
    from torch import Tensor
    from torch.nn import functional  # type: ignore[import-not-found]
except ImportError:  # pragma: no cover - covered by missing-runtime tests.
    torch = None
    functional = None
    Tensor = Any


@dataclass(frozen=True, slots=True)
class PredictionLossResult:
    """Phase-aware predictor loss components."""

    loss: Tensor
    pred_loss: Tensor
    kl_reg: Tensor
    phase: Literal["phase1", "phase2"]


def prediction_loss(
    prediction: Tensor,
    target: Tensor,
    *,
    alpha: float = 1.0,
    beta: float = 0.1,
    mask: Tensor | None = None,
    eps: float = 1.0e-8,
) -> Tensor:  # pragma: no cover - optional torch runtime is tested separately.
    """Return RFC-0005 ``alpha * (1 - cos) + beta * MSE / d_state``."""
    _require_torch("prediction_loss")
    _validate_loss_inputs(prediction, target, mask=mask)
    _require_nonnegative("alpha", alpha)
    _require_nonnegative("beta", beta)
    _require_positive_float("eps", eps)

    cosine = functional.cosine_similarity(prediction, target, dim=-1, eps=eps)
    squared = (prediction - target).pow(2).sum(dim=-1) / prediction.shape[-1]
    per_step = alpha * (1.0 - cosine) + beta * squared
    return _masked_mean(per_step, mask)


def lejepa_kl_regularizer(
    states: Tensor, *, eps: float = 1.0e-6
) -> Tensor:  # pragma: no cover - optional torch runtime is tested separately.
    """Return the closed-form KL from empirical state distribution to ``N(0, I)``."""
    _require_torch("lejepa_kl_regularizer")
    _require_positive_float("eps", eps)
    if states.ndim < 2:
        raise InputError(
            "states must have shape (..., d_state)",
            details={"shape": tuple(states.shape)},
        )
    if not states.is_floating_point():
        raise InputError("states must be a floating-point tensor")
    d_state = states.shape[-1]
    if d_state <= 0:
        raise InputError("states must have a non-empty feature dimension")

    flat = states.reshape(-1, d_state).to(dtype=torch.float32)
    if flat.shape[0] == 0:
        raise InputError("states must contain at least one sample")
    mean = flat.mean(dim=0)
    centered = flat - mean
    covariance = centered.T @ centered / flat.shape[0]
    eye = torch.eye(d_state, dtype=covariance.dtype, device=covariance.device)
    sign, logdet = torch.linalg.slogdet(covariance + eps * eye)
    if torch.any(sign <= 0):
        raise InputError("stabilized covariance must be positive definite")
    return 0.5 * (mean.pow(2).sum() + torch.trace(covariance) - logdet - d_state)


def predictor_loss(
    prediction: Tensor,
    target: Tensor,
    *,
    phase: Literal["phase1", "phase2"] = "phase1",
    alpha: float = 1.0,
    beta: float = 0.1,
    gamma: float = 0.5,
    mask: Tensor | None = None,
    regularizer_states: Tensor | None = None,
    eps: float = 1.0e-6,
) -> PredictionLossResult:  # pragma: no cover - optional torch runtime is tested separately.
    """Return phase-conditional total loss and monitorable components."""
    _require_torch("predictor_loss")
    if phase not in ("phase1", "phase2"):
        raise InputError("phase must be either 'phase1' or 'phase2'", details={"phase": phase})
    _require_nonnegative("gamma", gamma)
    pred = prediction_loss(prediction, target, alpha=alpha, beta=beta, mask=mask)
    reg_source = target if regularizer_states is None else regularizer_states
    kl_reg = lejepa_kl_regularizer(reg_source, eps=eps)
    total = pred if phase == "phase1" else pred + gamma * kl_reg
    return PredictionLossResult(loss=total, pred_loss=pred, kl_reg=kl_reg, phase=phase)


def _require_torch(function_name: str) -> None:  # pragma: no cover - covered by public wrappers.
    if torch is None:
        raise RuntimeSetupError(
            f"{function_name} requires PyTorch",
            remediation="install geno-lewm[train] or install torch",
        )


def _validate_loss_inputs(  # pragma: no cover - optional torch runtime is tested separately.
    prediction: Tensor, target: Tensor, *, mask: Tensor | None
) -> None:
    if prediction.shape != target.shape:
        raise InputError(
            "prediction and target shapes must match",
            details={
                "prediction_shape": tuple(prediction.shape),
                "target_shape": tuple(target.shape),
            },
        )
    if prediction.ndim < 2:
        raise InputError(
            "prediction and target must have shape (..., d_state)",
            details={"shape": tuple(prediction.shape)},
        )
    if prediction.shape[-1] <= 0:
        raise InputError("prediction and target must have a non-empty feature dimension")
    if not prediction.is_floating_point() or not target.is_floating_point():
        raise InputError("prediction and target must be floating-point tensors")
    if prediction.device != target.device:
        raise InputError(
            "prediction and target must be on the same device",
            details={
                "prediction_device": str(prediction.device),
                "target_device": str(target.device),
            },
        )
    if mask is not None and mask.shape != prediction.shape[:-1]:
        raise InputError(
            "mask must match prediction and target leading dimensions",
            details={
                "mask_shape": tuple(mask.shape),
                "expected_shape": tuple(prediction.shape[:-1]),
            },
        )


def _masked_mean(
    values: Tensor, mask: Tensor | None
) -> Tensor:  # pragma: no cover - optional torch runtime is tested separately.
    if mask is None:
        return values.mean()
    if mask.dtype != torch.bool and not torch.all((mask == 0) | (mask == 1)):
        raise InputError("mask must contain boolean or 0/1 values")
    valid = mask.to(device=values.device, dtype=torch.bool)
    count = valid.sum()
    if int(count.item()) == 0:
        raise InputError("mask must include at least one valid element")
    return values.masked_fill(~valid, 0.0).sum() / count


def _require_nonnegative(
    name: str, value: float
) -> None:  # pragma: no cover - optional torch runtime is tested separately.
    if (
        isinstance(value, bool)
        or not isinstance(value, int | float)
        or not math.isfinite(float(value))
        or value < 0.0
    ):
        raise InputError(
            f"{name} must be non-negative",
            details={"field": name, "value": value, "type": type(value).__name__},
        )


def _require_positive_float(
    name: str, value: float
) -> None:  # pragma: no cover - optional torch runtime is tested separately.
    if (
        isinstance(value, bool)
        or not isinstance(value, int | float)
        or not math.isfinite(float(value))
        or value <= 0.0
    ):
        raise InputError(
            f"{name} must be positive",
            details={"field": name, "value": value, "type": type(value).__name__},
        )
