# SPDX-License-Identifier: Apache-2.0
"""Optional torch smoke tests for the predictor/trainer boundary."""

from __future__ import annotations

import pytest


def test_tiny_predictor_identity_at_init_and_loss_decrease() -> None:
    """Catch predictor initialization or tiny-batch learning regressions.

    The hosted ML smoke gate installs only public fixture dependencies by
    default, so this test skips explicitly when torch is unavailable. In
    `geno-lewm[train]` environments it exercises a tiny predictor on CPU.
    """
    torch = pytest.importorskip("torch")
    from geno_lewm.predictor import Predictor

    torch.manual_seed(3)
    predictor = Predictor(
        d_state=16,
        d_action=8,
        d_hidden=16,
        n_heads=4,
        n_cross_layers=1,
        n_self_layers=1,
        ffn_dim=32,
        max_actions=3,
    )
    state = torch.nn.functional.normalize(torch.randn(4, 16), dim=-1)
    actions = torch.randn(4, 3, 8)
    mask = torch.tensor([[1, 1, 1], [1, 0, 0], [1, 1, 0], [1, 1, 1]], dtype=torch.bool)
    initial = predictor(state, actions, mask)

    torch.testing.assert_close(initial[0], state[0].expand(3, -1), atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(initial[1, 0], state[1], atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(initial[1, 1:], torch.zeros(2, 16), atol=0.0, rtol=0.0)

    direction = torch.nn.functional.normalize(torch.randn(4, 3, 16), dim=-1)
    target = torch.nn.functional.normalize(state.unsqueeze(1) + 0.2 * direction, dim=-1)
    optimizer = torch.optim.AdamW(predictor.parameters(), lr=5e-3, weight_decay=0.0)
    losses: list[float] = []
    for _ in range(30):
        optimizer.zero_grad(set_to_none=True)
        prediction = predictor(state, actions, mask)
        loss = torch.nn.functional.mse_loss(prediction[mask], target[mask])
        loss.backward()
        optimizer.step()
        losses.append(float(loss.detach()))

    assert losses[-1] < losses[0]
    assert torch.isfinite(torch.tensor(losses)).all()
