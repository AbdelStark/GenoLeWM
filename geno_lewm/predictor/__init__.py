# SPDX-License-Identifier: Apache-2.0
"""Action-conditioned predictor modules for GenoLeWM."""

from typing import Any

from geno_lewm.predictor.ar import ARPredictor
from geno_lewm.predictor.losses import (
    PredictionLossResult,
    lejepa_kl_regularizer,
    prediction_loss,
    predictor_loss,
)
from geno_lewm.predictor.model import Predictor

__all__ = [
    "ARPredictor",
    "PredictionLossResult",
    "Predictor",
    "build_predictor",
    "lejepa_kl_regularizer",
    "prediction_loss",
    "predictor_loss",
]


def build_predictor(config: Any) -> Predictor:
    """Construct the predictor from a ``GenoLeWMConfig``.

    Single source of truth shared by training (:mod:`geno_lewm.training.real`)
    and the deploy runtime (:mod:`geno_lewm.deploy.runtime`) so a trained
    checkpoint always loads back into an identically-shaped predictor. The two
    call sites previously built the predictor with different hyperparameters
    (d_hidden / n_cross_layers / ffn_dim), so a real exported checkpoint could
    not be loaded for scoring. Keep both on this builder.
    """
    return Predictor(
        d_state=config.predictor.d_state,
        d_action=config.action.d_action,
        n_heads=config.predictor.n_heads,
        n_cross_layers=config.predictor.n_layers,
        max_actions=getattr(config.action, "max_len", 16),
    )
