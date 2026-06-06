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


def test_ar_rollout_reuses_cached_action_projection() -> None:
    torch = pytest.importorskip("torch")
    from geno_lewm.predictor import ARPredictor, Predictor

    torch.manual_seed(17)
    predictor = Predictor(
        d_state=10,
        d_action=5,
        d_hidden=10,
        n_heads=2,
        n_cross_layers=2,
        n_self_layers=1,
        ffn_dim=20,
        max_actions=5,
    )
    torch.nn.init.normal_(predictor.output_mlp[-1].weight, mean=0.0, std=0.02)
    torch.nn.init.normal_(predictor.output_mlp[-1].bias, mean=0.0, std=0.02)

    state = torch.nn.functional.normalize(torch.randn(3, 10), dim=-1)
    actions = torch.randn(3, 5, 5)
    current = state
    expected = []
    step_mask = torch.ones(3, 1, dtype=torch.bool)
    for step in range(actions.shape[1]):
        pred = predictor(current, actions[:, step : step + 1, :], step_mask)[:, 0, :]
        expected.append(pred)
        current = pred

    calls = 0
    cache_calls = 0
    state_bias_calls = 0
    state_step_calls = 0
    original_forward = predictor.action_projection.forward
    original_cache = predictor._precompute_rollout_action_cache
    original_state_bias = predictor._rollout_state_token_bias
    original_state_step = predictor._forward_one_step_unmasked_state_from_action_token

    def counted_forward(input_tensor: torch.Tensor) -> torch.Tensor:
        nonlocal calls
        calls += 1
        return original_forward(input_tensor)

    def counted_cache(action_tokens: torch.Tensor) -> object:
        nonlocal cache_calls
        cache_calls += 1
        return original_cache(action_tokens)

    def counted_state_bias(state_tensor: torch.Tensor) -> torch.Tensor:
        nonlocal state_bias_calls
        state_bias_calls += 1
        return original_state_bias(state_tensor)

    def counted_state_step(
        state_tensor: torch.Tensor,
        action_token: torch.Tensor,
        action_cache: object | None = None,
        *,
        state_token_bias: torch.Tensor | None = None,
        upcast_output_mlp: bool = False,
    ) -> torch.Tensor:
        nonlocal state_step_calls
        state_step_calls += 1
        assert state_token_bias is not None
        return original_state_step(
            state_tensor,
            action_token,
            action_cache,
            state_token_bias=state_token_bias,
            upcast_output_mlp=upcast_output_mlp,
        )

    predictor.action_projection.forward = counted_forward  # type: ignore[method-assign]
    predictor._precompute_rollout_action_cache = counted_cache  # type: ignore[method-assign]
    predictor._rollout_state_token_bias = counted_state_bias  # type: ignore[method-assign]
    predictor._forward_one_step_unmasked_state_from_action_token = counted_state_step  # type: ignore[method-assign]
    observed = ARPredictor(predictor).rollout_tensor(state, actions)

    assert calls == 1
    assert cache_calls == 1
    assert state_bias_calls == 1
    assert state_step_calls == actions.shape[1]
    torch.testing.assert_close(
        observed,
        torch.stack(expected, dim=1),
        atol=1e-6,
        rtol=1e-6,
    )


def test_predict_haplotype_without_mask_uses_unmasked_cache_path() -> None:
    torch = pytest.importorskip("torch")
    from geno_lewm.predictor import ARPredictor, Predictor

    torch.manual_seed(23)
    predictor = Predictor(
        d_state=10,
        d_action=5,
        d_hidden=10,
        n_heads=2,
        n_cross_layers=2,
        n_self_layers=1,
        ffn_dim=20,
        max_actions=4,
    )
    torch.nn.init.normal_(predictor.output_mlp[-1].weight, mean=0.0, std=0.02)
    torch.nn.init.normal_(predictor.output_mlp[-1].bias, mean=0.0, std=0.02)

    state = torch.nn.functional.normalize(torch.randn(2, 10), dim=-1)
    actions = torch.randn(2, 4, 5)
    current = state
    step_mask = torch.ones(2, 1, dtype=torch.bool)
    for step in range(actions.shape[1]):
        current = predictor(current, actions[:, step : step + 1, :], step_mask)[:, 0, :]
    expected = current

    cache_calls = 0
    state_step_calls = 0
    original_cache = predictor._precompute_rollout_action_cache
    original_state_step = predictor._forward_one_step_unmasked_state_from_action_token

    def counted_cache(action_tokens: torch.Tensor) -> object:
        nonlocal cache_calls
        cache_calls += 1
        return original_cache(action_tokens)

    def counted_state_step(
        state_tensor: torch.Tensor,
        action_token: torch.Tensor,
        action_cache: object | None = None,
        *,
        state_token_bias: torch.Tensor | None = None,
        upcast_output_mlp: bool = False,
    ) -> torch.Tensor:
        nonlocal state_step_calls
        state_step_calls += 1
        assert action_cache is not None
        return original_state_step(
            state_tensor,
            action_token,
            action_cache,
            state_token_bias=state_token_bias,
            upcast_output_mlp=upcast_output_mlp,
        )

    def masked_step_is_not_expected(
        state_tensor: torch.Tensor,
        action_token: torch.Tensor,
        action_mask: torch.Tensor,
        *,
        upcast_output_mlp: bool = False,
    ) -> torch.Tensor:
        del state_tensor, action_token, action_mask, upcast_output_mlp
        raise AssertionError("unmasked predict_haplotype should not use the masked step path")

    predictor._precompute_rollout_action_cache = counted_cache  # type: ignore[method-assign]
    predictor._forward_one_step_unmasked_state_from_action_token = counted_state_step  # type: ignore[method-assign]
    predictor._forward_one_step_from_action_token = masked_step_is_not_expected  # type: ignore[method-assign]

    observed = ARPredictor(predictor).predict_haplotype(state, actions)

    assert cache_calls == 1
    assert state_step_calls == actions.shape[1]
    torch.testing.assert_close(observed, expected, atol=1e-6, rtol=1e-6)


def test_ar_rollout_requests_fp32_output_path_for_real_predictor_long_rollout() -> None:
    torch = pytest.importorskip("torch")
    from geno_lewm.predictor import ARPredictor, Predictor

    torch.manual_seed(19)
    predictor = Predictor(
        d_state=6,
        d_action=3,
        d_hidden=6,
        n_heads=2,
        n_cross_layers=2,
        n_self_layers=1,
        ffn_dim=12,
        max_actions=21,
    )
    state = torch.nn.functional.normalize(torch.randn(1, 6), dim=-1)
    actions = torch.randn(1, 21, 3)
    flags: list[bool] = []
    original_output_delta = predictor._output_delta

    def counted_output_delta(
        action_output: torch.Tensor,
        *,
        upcast_output_mlp: bool,
    ) -> torch.Tensor:
        flags.append(upcast_output_mlp)
        return original_output_delta(
            action_output,
            upcast_output_mlp=upcast_output_mlp,
        )

    predictor._output_delta = counted_output_delta  # type: ignore[method-assign]

    trajectory = ARPredictor(predictor).rollout(state, actions)

    assert len(trajectory) == 21
    assert flags == [True] * 21


def test_ar_rollout_requests_fp32_output_path_for_long_rollouts() -> None:
    torch = pytest.importorskip("torch")
    from geno_lewm.predictor import ARPredictor

    class CachedPredictor:
        d_state = 4
        d_action = 2
        max_actions = 24

        def __init__(self) -> None:
            self.upcast_flags: list[bool] = []

        def __call__(
            self,
            state: torch.Tensor,
            actions: torch.Tensor,
            action_mask: torch.Tensor,
        ) -> torch.Tensor:
            raise AssertionError("cached rollout should not call the fallback forward path")

        def _encode_rollout_actions(self, actions: torch.Tensor) -> torch.Tensor:
            return torch.zeros(actions.shape[0], actions.shape[1], self.d_state)

        def _forward_one_step_from_action_token(
            self,
            state: torch.Tensor,
            action_token: torch.Tensor,
            action_mask: torch.Tensor,
            *,
            upcast_output_mlp: bool = False,
        ) -> torch.Tensor:
            del action_token, action_mask
            self.upcast_flags.append(upcast_output_mlp)
            return torch.nn.functional.normalize(state + 0.01, dim=-1).unsqueeze(1)

    predictor = CachedPredictor()
    state = torch.nn.functional.normalize(torch.randn(1, 4), dim=-1)
    actions = torch.randn(1, 21, 2)

    trajectory = ARPredictor(predictor).rollout(state, actions)

    assert len(trajectory) == 21
    assert predictor.upcast_flags == [True] * 21


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
