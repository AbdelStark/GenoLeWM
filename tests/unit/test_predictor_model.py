"""Unit tests for the optional PyTorch predictor."""

from __future__ import annotations

import importlib.util
from itertools import pairwise

import pytest

from geno_lewm.errors import InputError, RuntimeSetupError


def test_predictor_reports_missing_torch_runtime() -> None:
    if importlib.util.find_spec("torch") is not None:
        pytest.skip("torch is installed in this environment")
    from geno_lewm.predictor import Predictor

    with pytest.raises(RuntimeSetupError):
        Predictor()


def test_predictor_default_shape_identity_and_parameter_budget() -> None:
    torch = pytest.importorskip("torch")
    from geno_lewm.predictor import Predictor

    torch.manual_seed(0)
    predictor = Predictor()
    state = torch.nn.functional.normalize(torch.randn(2, 1024), dim=-1)
    actions = torch.randn(2, 3, 512)
    action_mask = torch.tensor([[1, 1, 1], [1, 0, 0]], dtype=torch.bool)

    output = predictor(state, actions, action_mask)
    params = sum(param.numel() for param in predictor.parameters())

    assert output.shape == (2, 3, 1024)
    assert 19_800_000 <= params <= 24_200_000
    torch.testing.assert_close(output[0], state[0].expand(3, -1), atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(output[1, 0], state[1], atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(output[1, 1:], torch.zeros(2, 1024), atol=0.0, rtol=0.0)
    torch.testing.assert_close(output[0].norm(dim=-1), torch.ones(3), atol=1e-5, rtol=1e-5)


def test_predictor_rejects_invalid_shapes_and_dimensions() -> None:
    torch = pytest.importorskip("torch")
    from geno_lewm.predictor import Predictor

    predictor = Predictor(d_state=16, d_action=8, d_hidden=16, n_heads=4, max_actions=2)
    state = torch.randn(2, 16)
    actions = torch.randn(2, 2, 8)
    mask = torch.ones(2, 2, dtype=torch.bool)

    with pytest.raises(InputError):
        Predictor(d_hidden=15, n_heads=4)
    with pytest.raises(InputError):
        predictor(state[:, :8], actions, mask)
    with pytest.raises(InputError):
        predictor(state, actions[:, :, :4], mask)
    with pytest.raises(InputError):
        predictor(state, torch.randn(2, 3, 8), torch.ones(2, 3, dtype=torch.bool))
    with pytest.raises(InputError):
        predictor(state, actions, torch.zeros(2, 2, dtype=torch.bool))
    with pytest.raises(InputError):
        predictor(state, actions, torch.full((2, 2), 0.5))


def test_predictor_loss_decreases_on_tiny_fixed_minibatch() -> None:
    torch = pytest.importorskip("torch")
    from geno_lewm.predictor import Predictor

    torch.manual_seed(3)
    predictor = Predictor(
        d_state=16,
        d_action=8,
        d_hidden=16,
        n_heads=4,
        n_cross_layers=2,
        n_self_layers=1,
        ffn_dim=32,
        max_actions=3,
    )
    state = torch.nn.functional.normalize(torch.randn(4, 16), dim=-1)
    actions = torch.randn(4, 3, 8)
    mask = torch.ones(4, 3, dtype=torch.bool)
    direction = torch.nn.functional.normalize(torch.randn(4, 3, 16), dim=-1)
    target = torch.nn.functional.normalize(
        state.unsqueeze(1) + 0.2 * direction,
        dim=-1,
    )

    optimizer = torch.optim.AdamW(predictor.parameters(), lr=5e-3, weight_decay=0.0)
    losses: list[float] = []
    for _ in range(100):
        optimizer.zero_grad(set_to_none=True)
        prediction = predictor(state, actions, mask)
        loss = torch.nn.functional.mse_loss(prediction, target)
        loss.backward()
        optimizer.step()
        losses.append(float(loss.detach()))

    assert losses[-1] < losses[0] * 0.2
    assert all(after <= before + 1e-6 for before, after in pairwise(losses))
