"""Unit tests for predictor losses."""

from __future__ import annotations

import importlib.util
import math

import pytest

from geno_lewm.errors import InputError, RuntimeSetupError


def test_predictor_losses_report_missing_torch_runtime() -> None:
    if importlib.util.find_spec("torch") is not None:
        pytest.skip("torch is installed in this environment")
    from geno_lewm.predictor import prediction_loss

    with pytest.raises(RuntimeSetupError):
        prediction_loss(None, None)


def test_prediction_loss_matches_known_cosine_and_mse_values() -> None:
    torch = pytest.importorskip("torch")
    from geno_lewm.predictor import prediction_loss

    prediction = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    target = torch.tensor([[1.0, 0.0], [1.0, 0.0]])

    loss = prediction_loss(prediction, target, alpha=1.0, beta=0.1)

    assert loss.item() == pytest.approx(0.55)


def test_prediction_loss_masks_padded_steps() -> None:
    torch = pytest.importorskip("torch")
    from geno_lewm.predictor import prediction_loss

    prediction = torch.tensor([[[1.0, 0.0], [0.0, 1.0]]])
    target = torch.tensor([[[1.0, 0.0], [1.0, 0.0]]])
    mask = torch.tensor([[1, 0]])

    loss = prediction_loss(prediction, target, mask=mask)

    assert loss.item() == pytest.approx(0.0)


def test_lejepa_kl_regularizer_matches_closed_form_with_stabilized_logdet() -> None:
    torch = pytest.importorskip("torch")
    from geno_lewm.predictor import lejepa_kl_regularizer

    states = torch.tensor([[-1.0, 0.0], [1.0, 0.0]])
    eps = 1.0e-6

    kl = lejepa_kl_regularizer(states, eps=eps)
    expected = 0.5 * (1.0 - math.log((1.0 + eps) * eps) - 2.0)

    assert kl.item() == pytest.approx(expected, rel=1e-6)


def test_predictor_loss_dispatches_phase_conditionally_and_always_reports_kl() -> None:
    torch = pytest.importorskip("torch")
    from geno_lewm.predictor import predictor_loss

    prediction = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    target = torch.tensor([[1.0, 0.0], [1.0, 0.0]])

    phase1 = predictor_loss(prediction, target, phase="phase1", gamma=0.5)
    phase2 = predictor_loss(prediction, target, phase="phase2", gamma=0.5)

    assert phase1.phase == "phase1"
    assert phase2.phase == "phase2"
    assert phase1.pred_loss.item() == pytest.approx(0.55)
    assert phase1.kl_reg.item() > 0.0
    assert phase1.loss.item() == pytest.approx(phase1.pred_loss.item())
    assert phase2.loss.item() == pytest.approx(phase2.pred_loss.item() + 0.5 * phase2.kl_reg.item())


def test_predictor_losses_reject_invalid_inputs() -> None:
    torch = pytest.importorskip("torch")
    from geno_lewm.predictor import lejepa_kl_regularizer, prediction_loss, predictor_loss

    with pytest.raises(InputError):
        prediction_loss(torch.ones(2, 2), torch.ones(2, 3))
    with pytest.raises(InputError):
        prediction_loss(torch.ones(2, 2), torch.ones(2, 2), mask=torch.zeros(2, dtype=torch.bool))
    with pytest.raises(InputError):
        prediction_loss(torch.ones(2, 2), torch.ones(2, 2), alpha=-1.0)
    with pytest.raises(InputError):
        lejepa_kl_regularizer(torch.ones(2))
    with pytest.raises(InputError):
        predictor_loss(torch.ones(2, 2), torch.ones(2, 2), phase="phase3")  # type: ignore[arg-type]
