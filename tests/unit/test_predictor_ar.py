"""Unit tests for autoregressive predictor rollout."""

from __future__ import annotations

import importlib.util

import pytest

from geno_lewm.errors import InputError, RuntimeSetupError


def test_ar_predictor_reports_missing_torch_runtime() -> None:
    if importlib.util.find_spec("torch") is not None:
        pytest.skip("torch is installed in this environment")
    from geno_lewm.predictor import ARPredictor

    with pytest.raises(RuntimeSetupError):
        ARPredictor(object())


def test_ar_rollout_matches_repeated_one_step_forward() -> None:
    torch = pytest.importorskip("torch")
    from geno_lewm.predictor import ARPredictor, Predictor

    torch.manual_seed(11)
    predictor = Predictor(
        d_state=12,
        d_action=6,
        d_hidden=12,
        n_heads=3,
        n_cross_layers=2,
        n_self_layers=1,
        ffn_dim=24,
        max_actions=5,
    )
    torch.nn.init.normal_(predictor.output_mlp[-1].weight, mean=0.0, std=0.03)
    torch.nn.init.normal_(predictor.output_mlp[-1].bias, mean=0.0, std=0.03)

    state = torch.nn.functional.normalize(torch.randn(2, 12), dim=-1)
    actions = torch.randn(2, 4, 6)
    rollout = ARPredictor(predictor)

    observed = rollout.rollout(state, actions)
    current = state
    expected = []
    step_mask = torch.ones(2, 1, dtype=torch.bool)
    for step in range(actions.shape[1]):
        pred = predictor(current, actions[:, step : step + 1, :], step_mask)[:, 0, :]
        expected.append(pred)
        current = pred

    assert len(observed) == 4
    for got, want in zip(observed, expected, strict=True):
        torch.testing.assert_close(got, want, atol=1e-6, rtol=1e-6)

    torch.testing.assert_close(
        rollout.rollout_tensor(state, actions),
        torch.stack(expected, dim=1),
        atol=1e-6,
        rtol=1e-6,
    )
    torch.testing.assert_close(
        rollout.predict_haplotype(state, actions),
        expected[-1],
        atol=1e-6,
        rtol=1e-6,
    )


def test_ar_rollout_accepts_sequence_input_and_masks_padding() -> None:
    torch = pytest.importorskip("torch")
    from geno_lewm.predictor import ARPredictor, Predictor

    torch.manual_seed(13)
    predictor = Predictor(
        d_state=8,
        d_action=4,
        d_hidden=8,
        n_heads=2,
        n_cross_layers=2,
        n_self_layers=1,
        ffn_dim=16,
        max_actions=3,
    )
    torch.nn.init.normal_(predictor.output_mlp[-1].weight, mean=0.0, std=0.02)

    state = torch.nn.functional.normalize(torch.randn(2, 8), dim=-1)
    actions = [torch.randn(2, 4) for _ in range(3)]
    mask = torch.tensor([[1, 1, 1], [1, 0, 0]], dtype=torch.bool)
    rollout = ARPredictor(predictor)

    trajectory = rollout.rollout(state, actions, mask)
    assert len(trajectory) == 3
    torch.testing.assert_close(trajectory[1][1], torch.zeros(8), atol=0.0, rtol=0.0)
    torch.testing.assert_close(trajectory[2][1], torch.zeros(8), atol=0.0, rtol=0.0)
    torch.testing.assert_close(
        rollout.predict_haplotype(state, actions, mask)[1],
        trajectory[0][1],
        atol=1e-6,
        rtol=1e-6,
    )
    torch.testing.assert_close(
        rollout.predict_single(state, actions[0]),
        trajectory[0],
        atol=1e-6,
        rtol=1e-6,
    )


def test_ar_rollout_rejects_invalid_inputs() -> None:
    torch = pytest.importorskip("torch")
    from geno_lewm.predictor import ARPredictor, Predictor

    predictor = Predictor(d_state=8, d_action=4, d_hidden=8, n_heads=2, max_actions=2)
    rollout = ARPredictor(predictor)
    state = torch.randn(2, 8)
    actions = torch.randn(2, 2, 4)
    mask = torch.ones(2, 2, dtype=torch.bool)

    with pytest.raises(InputError):
        ARPredictor(object())  # type: ignore[arg-type]
    with pytest.raises(InputError):
        rollout.rollout(state[:, :4], actions)
    with pytest.raises(InputError):
        rollout.rollout(state, actions[:, :, :2])
    with pytest.raises(InputError):
        rollout.rollout(state, torch.randn(2, 3, 4))
    with pytest.raises(InputError):
        rollout.rollout(state, actions, mask[:, :1])
    with pytest.raises(InputError):
        rollout.rollout(state, actions, torch.zeros(2, 2, dtype=torch.bool))
    with pytest.raises(InputError):
        rollout.rollout(state, actions, torch.tensor([[0, 1], [1, 1]], dtype=torch.bool))
    with pytest.raises(InputError):
        rollout.rollout(state, actions, torch.full((2, 2), 0.5))
    with pytest.raises(InputError):
        rollout.predict_single(state, actions)
