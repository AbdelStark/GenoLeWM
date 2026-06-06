# SPDX-License-Identifier: Apache-2.0
"""Autoregressive rollout wrapper for action-conditioned predictors."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any, TypeVar, cast

from geno_lewm.errors import InputError, RuntimeSetupError

__all__ = ["ARPredictor"]

_F = TypeVar("_F", bound=Callable[..., Any])

try:  # pragma: no cover - exercised by optional-runtime tests with torch installed.
    import torch  # type: ignore[import-not-found]
    from torch import Tensor, nn
except ImportError:  # pragma: no cover - covered through the lightweight fallback class.
    torch = None
    Tensor = Any
    nn = None


if nn is None:

    class ARPredictor:
        """Placeholder that reports the missing optional training runtime."""

        def __init__(self, predictor: object) -> None:
            del predictor
            raise RuntimeSetupError(
                "ARPredictor requires PyTorch",
                remediation="install geno-lewm[train] or install torch",
            )


else:  # pragma: no cover - optional torch runtime is validated outside base CI.
    _INFERENCE_MODE = cast(Callable[[_F], _F], torch.inference_mode())

    class ARPredictor(nn.Module):  # type: ignore[no-redef,misc]
        """Inference-time autoregressive rollout over a base ``Predictor``.

        The wrapper defines the public RFC-0004 rollout contract: each
        action is scored against the state predicted by the previous
        action, producing ``[s_hat[t+1], ..., s_hat[t+K]]``. When the
        wrapped predictor exposes rollout-cache hooks, static action
        projections are encoded once before the autoregressive loop;
        otherwise the wrapper falls back to repeated ``forward`` calls.
        """

        def __init__(self, predictor: object) -> None:
            super().__init__()
            if not callable(predictor):
                raise InputError("predictor must be callable")
            self.predictor: Any = predictor
            self.d_state = _required_positive_int(predictor, "d_state")
            self.d_action = _required_positive_int(predictor, "d_action")
            self.max_actions = _required_positive_int(predictor, "max_actions")

        @_INFERENCE_MODE
        def rollout(
            self,
            state: Tensor,
            action_sequence: Tensor | Sequence[Tensor],
            action_mask: Tensor | None = None,
        ) -> tuple[Tensor, ...]:
            """Return one predicted state per autoregressive action step."""
            return tuple(self.rollout_tensor(state, action_sequence, action_mask).unbind(dim=1))

        @_INFERENCE_MODE
        def rollout_tensor(
            self,
            state: Tensor,
            action_sequence: Tensor | Sequence[Tensor],
            action_mask: Tensor | None = None,
        ) -> Tensor:
            """Return autoregressive rollout as ``(batch, steps, d_state)``."""
            actions = self._normalize_actions(action_sequence)
            self._validate_state(state, actions)

            current = state
            call_mask = torch.ones((actions.shape[0], 1), dtype=torch.bool, device=actions.device)
            action_tokens = self._cached_action_tokens(actions)
            action_cache = (
                self._cached_rollout_action_cache(action_tokens) if action_mask is None else None
            )
            state_token_bias = self._cached_state_token_bias(state)
            upcast_output_mlp = actions.shape[1] > 20
            output_dtype = torch.float32 if upcast_output_mlp else state.dtype
            outputs = state.new_empty(
                (actions.shape[0], actions.shape[1], self.d_state),
                dtype=output_dtype,
            )
            if action_mask is None:
                if action_tokens is None:
                    for step in range(actions.shape[1]):
                        prediction = self.predictor(
                            current,
                            actions[:, step : step + 1, :],
                            call_mask,
                        )
                        current = prediction[:, 0, :]
                        outputs[:, step, :] = current
                    return outputs

                one_step_state = getattr(
                    self.predictor,
                    "_forward_one_step_unmasked_state_from_action_token",
                    None,
                )

                one_step_unmasked = getattr(
                    self.predictor,
                    "_forward_one_step_unmasked_from_action_token",
                    None,
                )
                select_action_cache = getattr(
                    self.predictor,
                    "_slice_rollout_action_cache",
                    None,
                )
                if callable(one_step_state):
                    for step in range(actions.shape[1]):
                        step_cache = (
                            select_action_cache(action_cache, step)
                            if action_cache is not None and callable(select_action_cache)
                            else None
                        )
                        current = one_step_state(
                            current,
                            action_tokens[:, step, :],
                            step_cache,
                            state_token_bias=state_token_bias,
                            upcast_output_mlp=upcast_output_mlp,
                        )
                        outputs[:, step, :] = current
                    return outputs

                if callable(one_step_unmasked):
                    for step in range(actions.shape[1]):
                        step_cache = (
                            select_action_cache(action_cache, step)
                            if action_cache is not None and callable(select_action_cache)
                            else None
                        )
                        prediction = one_step_unmasked(
                            current,
                            action_tokens[:, step, :],
                            step_cache,
                            upcast_output_mlp=upcast_output_mlp,
                        )
                        current = prediction[:, 0, :]
                        outputs[:, step, :] = current
                    return outputs

                for step in range(actions.shape[1]):
                    prediction = self.predictor._forward_one_step_from_action_token(
                        current,
                        action_tokens[:, step, :],
                        call_mask,
                        upcast_output_mlp=upcast_output_mlp,
                    )
                    current = prediction[:, 0, :]
                    outputs[:, step, :] = current
                return outputs

            mask = self._normalize_mask(actions, action_mask)
            for step in range(actions.shape[1]):
                active = mask[:, step].unsqueeze(-1)
                if action_tokens is None:
                    prediction = self.predictor(
                        current,
                        actions[:, step : step + 1, :],
                        call_mask,
                    )
                else:
                    prediction = self.predictor._forward_one_step_from_action_token(
                        current,
                        action_tokens[:, step, :],
                        call_mask,
                        upcast_output_mlp=upcast_output_mlp,
                    )
                next_state = prediction[:, 0, :]
                current = torch.where(active, next_state, current)
                outputs[:, step, :] = torch.where(
                    active,
                    next_state,
                    torch.zeros_like(next_state),
                )
            return outputs

        @_INFERENCE_MODE
        def predict_single(self, state: Tensor, action: Tensor) -> Tensor:
            """Return the one-step predicted state for ``action``."""
            actions = self._normalize_actions(action.unsqueeze(1) if action.ndim == 2 else action)
            if actions.shape[1] != 1:
                raise InputError(
                    "predict_single expects exactly one action",
                    details={"steps": actions.shape[1]},
                )
            return self.rollout(state, actions)[0]

        @_INFERENCE_MODE
        def predict_trajectory(
            self,
            state: Tensor,
            action_sequence: Tensor | Sequence[Tensor],
            action_mask: Tensor | None = None,
        ) -> tuple[Tensor, ...]:
            """Alias for :meth:`rollout` matching RFC-0004 terminology."""
            return self.rollout(state, action_sequence, action_mask)

        @_INFERENCE_MODE
        def predict_haplotype(
            self,
            state: Tensor,
            action_sequence: Tensor | Sequence[Tensor],
            action_mask: Tensor | None = None,
        ) -> Tensor:
            """Return the final predicted state after all valid actions."""
            actions = self._normalize_actions(action_sequence)
            mask = self._normalize_mask(actions, action_mask)
            trajectory = self.rollout_tensor(state, actions, mask)
            indices = mask.sum(dim=1) - 1
            rows = torch.arange(actions.shape[0], device=actions.device)
            return trajectory[rows, indices]

        def _normalize_actions(self, action_sequence: Tensor | Sequence[Tensor]) -> Tensor:
            if isinstance(action_sequence, torch.Tensor):
                actions = action_sequence
            else:
                if not action_sequence:
                    raise InputError("action_sequence must contain at least one action")
                actions = torch.stack(tuple(action_sequence), dim=1)

            if actions.ndim != 3:
                raise InputError(
                    "action_sequence must have shape (batch, steps, d_action)",
                    details={"shape": tuple(actions.shape)},
                )
            if actions.shape[1] == 0:
                raise InputError("action_sequence must contain at least one action")
            if actions.shape[1] > self.max_actions:
                raise InputError(
                    "action sequence length exceeds max_actions",
                    details={"max_actions": self.max_actions, "actual": actions.shape[1]},
                )
            if actions.shape[2] != self.d_action:
                raise InputError(
                    "action feature dimension must equal d_action",
                    details={"expected": self.d_action, "actual": actions.shape[2]},
                )
            if not actions.is_floating_point():
                raise InputError("action_sequence must contain floating-point tensors")
            return actions

        def _normalize_mask(self, actions: Tensor, action_mask: Tensor | None) -> Tensor:
            if action_mask is None:
                return torch.ones(
                    actions.shape[:2],
                    dtype=torch.bool,
                    device=actions.device,
                )
            if action_mask.ndim != 2 or action_mask.shape != actions.shape[:2]:
                raise InputError(
                    "action_mask must have shape (batch, steps)",
                    details={
                        "expected": tuple(actions.shape[:2]),
                        "actual": tuple(action_mask.shape),
                    },
                )
            if action_mask.device != actions.device:
                raise InputError(
                    "action_mask and action_sequence must be on the same device",
                    details={
                        "action_mask_device": str(action_mask.device),
                        "action_device": str(actions.device),
                    },
                )
            if action_mask.dtype != torch.bool and not torch.all(
                (action_mask == 0) | (action_mask == 1)
            ):
                raise InputError("action_mask must contain boolean or 0/1 values")
            mask = action_mask.to(dtype=torch.bool)
            if not torch.all(mask.any(dim=1)):
                raise InputError("each batch item must include at least one valid action")
            if torch.any(mask[:, 1:] & ~mask[:, :-1]):
                raise InputError("action_mask valid actions must be left-aligned before padding")
            return mask

        def _validate_state(self, state: Tensor, actions: Tensor) -> None:
            if state.ndim != 2:
                raise InputError(
                    "state must have shape (batch, d_state)",
                    details={"shape": tuple(state.shape)},
                )
            if state.shape[0] != actions.shape[0]:
                raise InputError(
                    "state and action_sequence batch dimensions must agree",
                    details={"state_batch": state.shape[0], "action_batch": actions.shape[0]},
                )
            if state.shape[1] != self.d_state:
                raise InputError(
                    "state feature dimension must equal d_state",
                    details={"expected": self.d_state, "actual": state.shape[1]},
                )
            if state.device != actions.device:
                raise InputError(
                    "state and action_sequence must be on the same device",
                    details={
                        "state_device": str(state.device),
                        "action_device": str(actions.device),
                    },
                )
            if not state.is_floating_point():
                raise InputError("state must be a floating-point tensor")

        def _cached_action_tokens(self, actions: Tensor) -> Tensor | None:
            encode = getattr(self.predictor, "_encode_rollout_actions", None)
            step = getattr(self.predictor, "_forward_one_step_from_action_token", None)
            if not callable(encode) or not callable(step):
                return None
            return encode(actions)

        def _cached_rollout_action_cache(self, action_tokens: Tensor | None) -> Any:
            if action_tokens is None:
                return None
            encode = getattr(self.predictor, "_precompute_rollout_action_cache", None)
            if not callable(encode):
                return None
            return encode(action_tokens)

        def _cached_state_token_bias(self, state: Tensor) -> Tensor | None:
            encode = getattr(self.predictor, "_rollout_state_token_bias", None)
            if not callable(encode):
                return None
            return encode(state)


def _required_positive_int(obj: object, name: str) -> int:
    value = getattr(obj, name, None)
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise InputError(
            f"predictor must expose positive integer {name}",
            details={"field": name, "value": value, "type": type(value).__name__},
        )
    return value
