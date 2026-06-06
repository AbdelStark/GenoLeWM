# SPDX-License-Identifier: Apache-2.0
"""Cross-attention predictor for action-conditioned latent transitions.

The PyTorch runtime is optional. Importing :mod:`geno_lewm.predictor`
is lightweight; instantiating :class:`Predictor` requires a training
environment with PyTorch installed.
"""

from __future__ import annotations

import math
from typing import Any

from geno_lewm.errors import InputError, RuntimeSetupError

__all__ = ["Predictor"]

try:  # pragma: no cover - exercised by optional-runtime tests with torch installed.
    import torch  # type: ignore[import-not-found]
    from torch import Tensor, nn
    from torch.nn import functional  # type: ignore[import-not-found]
except ImportError:  # pragma: no cover - covered through the lightweight fallback class.
    torch = None
    functional = None
    Tensor = Any
    nn = None


if nn is None:

    class Predictor:
        """Placeholder that reports the missing optional training runtime."""

        def __init__(
            self,
            *,
            d_state: int = 1024,
            d_action: int = 512,
            d_hidden: int = 768,
            n_heads: int = 8,
            n_cross_layers: int = 4,
            n_self_layers: int = 2,
            ffn_dim: int = 768,
            max_actions: int = 16,
        ) -> None:
            del (
                d_state,
                d_action,
                d_hidden,
                n_heads,
                n_cross_layers,
                n_self_layers,
                ffn_dim,
                max_actions,
            )
            raise RuntimeSetupError(
                "Predictor requires PyTorch",
                remediation="install geno-lewm[train] or install torch",
            )

else:  # pragma: no cover - optional torch runtime is validated outside base CI.

    class Predictor(nn.Module):  # type: ignore[no-redef,misc]
        """Cross-attention Transformer predictor from RFC-0004.

        The default keeps the RFC-0004 4-cross/2-self topology and
        Carbon-compatible ``d_state=1024`` output while using the RFC's
        target-size ``d_hidden=768`` variant so the trainable budget is
        close to the documented ~22M target.
        """

        def __init__(
            self,
            *,
            d_state: int = 1024,
            d_action: int = 512,
            d_hidden: int = 768,
            n_heads: int = 8,
            n_cross_layers: int = 4,
            n_self_layers: int = 2,
            ffn_dim: int = 768,
            max_actions: int = 16,
        ) -> None:
            super().__init__()
            _require_positive("d_state", d_state)
            _require_positive("d_action", d_action)
            _require_positive("d_hidden", d_hidden)
            _require_positive("n_heads", n_heads)
            _require_positive("n_cross_layers", n_cross_layers)
            _require_positive("n_self_layers", n_self_layers)
            _require_positive("ffn_dim", ffn_dim)
            _require_positive("max_actions", max_actions)
            if d_hidden % n_heads != 0:
                raise InputError(
                    "d_hidden must be divisible by n_heads",
                    details={"d_hidden": d_hidden, "n_heads": n_heads},
                )

            self.d_state = d_state
            self.d_action = d_action
            self.d_hidden = d_hidden
            self.max_actions = max_actions
            self.state_projection = (
                nn.Identity() if d_state == d_hidden else nn.Linear(d_state, d_hidden)
            )
            self.action_projection = nn.Linear(d_action, d_hidden)
            self.token_type_embedding = nn.Embedding(2, d_hidden)
            self.step_position_embedding = nn.Embedding(max_actions + 1, d_hidden)
            self.cross_blocks = nn.ModuleList(
                _StateToActionCrossBlock(
                    d_hidden=d_hidden,
                    n_heads=n_heads,
                    ffn_dim=ffn_dim,
                )
                if index % 2 == 0
                else _ActionToStateCrossBlock(
                    d_hidden=d_hidden,
                    n_heads=n_heads,
                    ffn_dim=ffn_dim,
                )
                for index in range(n_cross_layers)
            )
            self.self_blocks = nn.ModuleList(
                _SelfAttentionBlock(d_hidden=d_hidden, n_heads=n_heads, ffn_dim=ffn_dim)
                for _ in range(n_self_layers)
            )
            self.output_mlp = nn.Sequential(
                nn.Linear(d_hidden, d_hidden),
                nn.GELU(),
                nn.LayerNorm(d_hidden),
                nn.Linear(d_hidden, d_state),
            )
            self.reset_parameters()

        def reset_parameters(self) -> None:
            """Initialize layers per RFC-0004 §3.4."""
            for module in self.modules():
                if isinstance(module, nn.MultiheadAttention):
                    std = math.sqrt(2.0 / float(module.embed_dim))
                    nn.init.trunc_normal_(module.in_proj_weight, std=std, a=-2 * std, b=2 * std)
                    if module.in_proj_bias is not None:
                        nn.init.zeros_(module.in_proj_bias)
                elif isinstance(module, nn.Linear):
                    std = math.sqrt(2.0 / float(module.in_features))
                    nn.init.trunc_normal_(module.weight, std=std, a=-2 * std, b=2 * std)
                    if module.bias is not None:
                        nn.init.zeros_(module.bias)
                elif isinstance(module, nn.LayerNorm):
                    nn.init.ones_(module.weight)
                    nn.init.zeros_(module.bias)
                elif isinstance(module, nn.Embedding):
                    nn.init.normal_(module.weight, std=0.02)

            final = self.output_mlp[-1]
            if isinstance(final, nn.Linear):
                nn.init.zeros_(final.weight)
                nn.init.zeros_(final.bias)

        def forward(
            self,
            state: Tensor,
            actions: Tensor,
            action_mask: Tensor,
        ) -> Tensor:
            """Return per-action next-state predictions of shape ``(B, K, d_state)``."""
            mask = self._validate_inputs(state, actions, action_mask)
            state_token = self._encode_state_token(state)
            action_tokens = self._encode_forward_actions(actions)
            return self._predict_from_tokens(
                state,
                state_token,
                action_tokens,
                mask,
                upcast_output_mlp=actions.shape[1] > 20,
            )

        def _encode_state_token(self, state: Tensor) -> Tensor:
            batch_size = state.shape[0]
            device = state.device
            state_type = torch.zeros((batch_size, 1), dtype=torch.long, device=device)
            state_position = torch.zeros((batch_size, 1), dtype=torch.long, device=device)
            state_token = self.state_projection(state).unsqueeze(1)
            state_token = state_token + self.token_type_embedding(state_type)
            state_token = state_token + self.step_position_embedding(state_position)
            return state_token

        def _encode_forward_actions(self, actions: Tensor) -> Tensor:
            batch_size, steps, _d_action = actions.shape
            device = actions.device
            action_type = torch.ones((batch_size, steps), dtype=torch.long, device=device)
            action_positions = torch.arange(1, steps + 1, dtype=torch.long, device=device)
            action_positions = action_positions.unsqueeze(0).expand(batch_size, -1)
            action_tokens = self.action_projection(actions)
            action_tokens = action_tokens + self.token_type_embedding(action_type)
            action_tokens = action_tokens + self.step_position_embedding(action_positions)
            return action_tokens

        def _encode_rollout_actions(self, actions: Tensor) -> Tensor:
            """Encode actions once for repeated single-step rollout calls.

            ``ARPredictor`` is logically equivalent to calling
            ``forward(current_state, action[:, step:step+1], ones)`` for
            every step. In that convention each action is the first and
            only action token for its call, so all cached actions use the
            single-step position embedding rather than absolute rollout
            positions.
            """
            batch_size, steps, _d_action = actions.shape
            device = actions.device
            action_type = torch.ones((batch_size, steps), dtype=torch.long, device=device)
            action_positions = torch.ones((batch_size, steps), dtype=torch.long, device=device)
            action_tokens = self.action_projection(actions)
            action_tokens = action_tokens + self.token_type_embedding(action_type)
            action_tokens = action_tokens + self.step_position_embedding(action_positions)
            return action_tokens

        def _precompute_rollout_action_cache(
            self,
            action_tokens: Tensor,
        ) -> tuple[Tensor | None, tuple[Tensor, Tensor, Tensor] | None] | None:
            first_state_attention = None
            first_action_attention = None
            if self.cross_blocks:
                precompute_state = getattr(
                    self.cross_blocks[0],
                    "precompute_one_step_state_attention",
                    None,
                )
                if callable(precompute_state):
                    first_state_attention = precompute_state(action_tokens)
            if len(self.cross_blocks) > 1:
                precompute_action = getattr(
                    self.cross_blocks[1],
                    "precompute_one_step_action_attention",
                    None,
                )
                if callable(precompute_action):
                    first_action_attention = precompute_action(action_tokens)
            if first_state_attention is None and first_action_attention is None:
                return None
            return first_state_attention, first_action_attention

        def _slice_rollout_action_cache(
            self,
            action_cache: tuple[Tensor | None, tuple[Tensor, Tensor, Tensor] | None],
            step: int,
        ) -> tuple[Tensor | None, tuple[Tensor, Tensor, Tensor] | None]:
            first_state_attention, first_action_attention = action_cache
            return (
                None if first_state_attention is None else first_state_attention[:, step, :],
                None
                if first_action_attention is None
                else (
                    first_action_attention[0][:, step : step + 1, :],
                    first_action_attention[1][:, step : step + 1, :],
                    first_action_attention[2][:, step : step + 1, :],
                ),
            )

        def _forward_one_step_from_action_token(
            self,
            state: Tensor,
            action_token: Tensor,
            action_mask: Tensor,
            *,
            upcast_output_mlp: bool = False,
        ) -> Tensor:
            if not torch.all(action_mask):
                return self._predict_from_tokens(
                    state,
                    self._encode_state_token(state),
                    action_token.unsqueeze(1),
                    action_mask,
                    upcast_output_mlp=upcast_output_mlp,
                )
            return self._forward_one_step_unmasked_from_action_token(
                state,
                action_token,
                upcast_output_mlp=upcast_output_mlp,
            )

        def _forward_one_step_unmasked_from_action_token(
            self,
            state: Tensor,
            action_token: Tensor,
            action_cache: tuple[Tensor | None, tuple[Tensor, Tensor, Tensor] | None] | None = None,
            *,
            upcast_output_mlp: bool = False,
        ) -> Tensor:
            first_state_attention = None
            first_action_attention = None
            if action_cache is not None:
                first_state_attention, first_action_attention = action_cache
            action_tokens = action_token.unsqueeze(1)
            state_token = self._encode_state_token(state)
            for block_index, block in enumerate(self.cross_blocks):
                if block_index == 0 and first_state_attention is not None:
                    state_token, action_tokens = block.forward_one_step_from_state_attention(
                        state_token,
                        action_tokens,
                        first_state_attention.unsqueeze(1),
                    )
                elif block_index == 1 and first_action_attention is not None:
                    state_token, action_tokens = block.forward_one_step_from_action_attention(
                        state_token,
                        action_tokens,
                        first_action_attention,
                    )
                else:
                    state_token, action_tokens = block.forward_one_step(state_token, action_tokens)

            tokens = torch.cat((state_token, action_tokens), dim=1)
            for block in self.self_blocks[:-1]:
                tokens = block.forward_one_step(tokens)
            if self.self_blocks:
                action_output = self.self_blocks[-1].forward_one_step_action_only(tokens)
            else:
                action_output = action_tokens

            delta = self._output_delta(action_output, upcast_output_mlp=upcast_output_mlp)
            if upcast_output_mlp:
                prediction = functional.normalize(
                    state.unsqueeze(1).float() + delta,
                    p=2.0,
                    dim=-1,
                    eps=1.0e-12,
                )
            else:
                prediction = functional.normalize(
                    state.unsqueeze(1) + delta,
                    p=2.0,
                    dim=-1,
                    eps=1.0e-12,
                )
            return prediction

        def _predict_from_tokens(
            self,
            state: Tensor,
            state_token: Tensor,
            action_tokens: Tensor,
            mask: Tensor,
            *,
            upcast_output_mlp: bool,
        ) -> Tensor:
            for block in self.cross_blocks:
                state_token, action_tokens = block(state_token, action_tokens, mask)

            batch_size, steps, _d_hidden = action_tokens.shape
            device = action_tokens.device
            tokens = torch.cat((state_token, action_tokens), dim=1)
            token_mask = torch.cat(
                (
                    torch.ones((batch_size, 1), dtype=torch.bool, device=device),
                    mask,
                ),
                dim=1,
            )
            for block in self.self_blocks:
                tokens = block(tokens, token_mask)

            action_output = tokens[:, 1:, :]
            delta = self._output_delta(action_output, upcast_output_mlp=upcast_output_mlp)
            base = state.unsqueeze(1).expand(-1, steps, -1)
            if upcast_output_mlp:
                prediction = functional.normalize(
                    base.float() + delta,
                    p=2.0,
                    dim=-1,
                    eps=1.0e-12,
                )
            else:
                prediction = functional.normalize(base + delta, p=2.0, dim=-1, eps=1.0e-12)
            return prediction.masked_fill(~mask.unsqueeze(-1), 0.0)

        def _output_delta(self, action_output: Tensor, *, upcast_output_mlp: bool) -> Tensor:
            if not upcast_output_mlp:
                return self.output_mlp(action_output)

            first = self.output_mlp[0]
            norm = self.output_mlp[2]
            final = self.output_mlp[3]
            if not (
                isinstance(first, nn.Linear)
                and isinstance(norm, nn.LayerNorm)
                and isinstance(final, nn.Linear)
            ):  # pragma: no cover - protects future output_mlp edits.
                raise RuntimeSetupError("Predictor output_mlp has an unsupported layout")

            hidden = functional.linear(
                action_output.float(),
                first.weight.float(),
                None if first.bias is None else first.bias.float(),
            )
            hidden = functional.gelu(hidden)
            hidden = functional.layer_norm(
                hidden,
                norm.normalized_shape,
                None if norm.weight is None else norm.weight.float(),
                None if norm.bias is None else norm.bias.float(),
                norm.eps,
            )
            return functional.linear(
                hidden,
                final.weight.float(),
                None if final.bias is None else final.bias.float(),
            )

        def _validate_inputs(self, state: Tensor, actions: Tensor, action_mask: Tensor) -> Tensor:
            if state.ndim != 2:
                raise InputError(
                    "state must have shape (batch, d_state)",
                    details={"shape": tuple(state.shape)},
                )
            if actions.ndim != 3:
                raise InputError(
                    "actions must have shape (batch, steps, d_action)",
                    details={"shape": tuple(actions.shape)},
                )
            if action_mask.ndim != 2:
                raise InputError(
                    "action_mask must have shape (batch, steps)",
                    details={"shape": tuple(action_mask.shape)},
                )
            if state.shape[0] != actions.shape[0] or action_mask.shape != actions.shape[:2]:
                raise InputError(
                    "state, actions, and action_mask batch/step dimensions must agree",
                    details={
                        "state_shape": tuple(state.shape),
                        "actions_shape": tuple(actions.shape),
                        "action_mask_shape": tuple(action_mask.shape),
                    },
                )
            if state.shape[1] != self.d_state:
                raise InputError(
                    "state feature dimension must equal d_state",
                    details={"expected": self.d_state, "actual": state.shape[1]},
                )
            if actions.shape[2] != self.d_action:
                raise InputError(
                    "action feature dimension must equal d_action",
                    details={"expected": self.d_action, "actual": actions.shape[2]},
                )
            if actions.shape[1] == 0:
                raise InputError("actions must contain at least one step")
            if actions.shape[1] > self.max_actions:
                raise InputError(
                    "action sequence length exceeds max_actions",
                    details={"max_actions": self.max_actions, "actual": actions.shape[1]},
                )
            if state.device != actions.device:
                raise InputError(
                    "state and actions must be on the same device",
                    details={
                        "state_device": str(state.device),
                        "actions_device": str(actions.device),
                    },
                )
            if not state.is_floating_point() or not actions.is_floating_point():
                raise InputError("state and actions must be floating-point tensors")

            if action_mask.dtype != torch.bool and not torch.all(
                (action_mask == 0) | (action_mask == 1)
            ):
                raise InputError("action_mask must contain boolean or 0/1 values")
            mask = action_mask.to(device=actions.device, dtype=torch.bool)
            if not torch.all(mask.any(dim=1)):
                raise InputError("each batch item must include at least one valid action")
            return mask

    class _StateToActionCrossBlock(nn.Module):  # type: ignore[misc]
        def __init__(self, *, d_hidden: int, n_heads: int, ffn_dim: int) -> None:
            super().__init__()
            self.attn_norm = nn.LayerNorm(d_hidden)
            self.attn = nn.MultiheadAttention(
                d_hidden,
                n_heads,
                dropout=0.0,
                batch_first=True,
            )
            self.ffn_norm = nn.LayerNorm(d_hidden)
            self.ffn = _FeedForward(d_hidden=d_hidden, ffn_dim=ffn_dim)

        def forward(
            self,
            state_token: Tensor,
            action_tokens: Tensor,
            action_mask: Tensor,
        ) -> tuple[Tensor, Tensor]:
            query = self.attn_norm(state_token)
            key_value = self.attn_norm(action_tokens)
            attended = self.attn(
                query,
                key_value,
                key_value,
                key_padding_mask=~action_mask,
                need_weights=False,
            )[0]
            state_token = state_token + attended
            state_token = state_token + self.ffn(self.ffn_norm(state_token))
            return state_token, action_tokens

        def forward_one_step(
            self,
            state_token: Tensor,
            action_tokens: Tensor,
        ) -> tuple[Tensor, Tensor]:
            attended = _mha_single_key_value_only(
                self.attn,
                self.attn_norm(action_tokens),
            )
            state_token = state_token + attended
            state_token = state_token + self.ffn(self.ffn_norm(state_token))
            return state_token, action_tokens

        def precompute_one_step_state_attention(self, action_tokens: Tensor) -> Tensor:
            return _mha_single_key_value_only(
                self.attn,
                self.attn_norm(action_tokens),
            )

        def forward_one_step_from_state_attention(
            self,
            state_token: Tensor,
            action_tokens: Tensor,
            attended: Tensor,
        ) -> tuple[Tensor, Tensor]:
            state_token = state_token + attended
            state_token = state_token + self.ffn(self.ffn_norm(state_token))
            return state_token, action_tokens

    class _ActionToStateCrossBlock(nn.Module):  # type: ignore[misc]
        def __init__(self, *, d_hidden: int, n_heads: int, ffn_dim: int) -> None:
            super().__init__()
            self.query_norm = nn.LayerNorm(d_hidden)
            self.memory_norm = nn.LayerNorm(d_hidden)
            self.attn = nn.MultiheadAttention(
                d_hidden,
                n_heads,
                dropout=0.0,
                batch_first=True,
            )
            self.ffn_norm = nn.LayerNorm(d_hidden)
            self.ffn = _FeedForward(d_hidden=d_hidden, ffn_dim=ffn_dim)

        def forward(
            self,
            state_token: Tensor,
            action_tokens: Tensor,
            action_mask: Tensor,
        ) -> tuple[Tensor, Tensor]:
            steps = action_tokens.shape[1]
            memory = torch.cat((state_token, action_tokens), dim=1)
            memory_mask = torch.cat(
                (
                    torch.ones(
                        (action_mask.shape[0], 1), dtype=torch.bool, device=action_mask.device
                    ),
                    action_mask,
                ),
                dim=1,
            )
            attended = self.attn(
                self.query_norm(action_tokens),
                self.memory_norm(memory),
                self.memory_norm(memory),
                attn_mask=_causal_cross_mask(steps, action_tokens.device),
                key_padding_mask=~memory_mask,
                need_weights=False,
            )[0]
            updated = action_tokens + attended
            updated = updated + self.ffn(self.ffn_norm(updated))
            action_tokens = torch.where(action_mask.unsqueeze(-1), updated, action_tokens)
            return state_token, action_tokens

        def forward_one_step(
            self,
            state_token: Tensor,
            action_tokens: Tensor,
        ) -> tuple[Tensor, Tensor]:
            memory = torch.cat((state_token, action_tokens), dim=1)
            attended = _mha_no_mask(
                self.attn,
                self.query_norm(action_tokens),
                self.memory_norm(memory),
            )
            action_tokens = action_tokens + attended
            action_tokens = action_tokens + self.ffn(self.ffn_norm(action_tokens))
            return state_token, action_tokens

        def precompute_one_step_action_attention(
            self,
            action_tokens: Tensor,
        ) -> tuple[Tensor, Tensor, Tensor]:
            query = self.query_norm(action_tokens)
            memory = self.memory_norm(action_tokens)
            return _mha_project_qkv(self.attn, query, memory)

        def forward_one_step_from_action_attention(
            self,
            state_token: Tensor,
            action_tokens: Tensor,
            action_attention: tuple[Tensor, Tensor, Tensor],
        ) -> tuple[Tensor, Tensor]:
            key_state_proj, value_state_proj = _mha_project_kv(
                self.attn,
                self.memory_norm(state_token),
            )
            query_proj, key_action_proj, value_action_proj = action_attention
            attended = _mha_no_mask_from_projected(
                self.attn,
                query_proj,
                torch.cat((key_state_proj, key_action_proj), dim=1),
                torch.cat((value_state_proj, value_action_proj), dim=1),
            )
            action_tokens = action_tokens + attended
            action_tokens = action_tokens + self.ffn(self.ffn_norm(action_tokens))
            return state_token, action_tokens

    class _SelfAttentionBlock(nn.Module):  # type: ignore[misc]
        def __init__(self, *, d_hidden: int, n_heads: int, ffn_dim: int) -> None:
            super().__init__()
            self.attn_norm = nn.LayerNorm(d_hidden)
            self.attn = nn.MultiheadAttention(
                d_hidden,
                n_heads,
                dropout=0.0,
                batch_first=True,
            )
            self.ffn_norm = nn.LayerNorm(d_hidden)
            self.ffn = _FeedForward(d_hidden=d_hidden, ffn_dim=ffn_dim)

        def forward(self, tokens: Tensor, token_mask: Tensor) -> Tensor:
            attended = self.attn(
                self.attn_norm(tokens),
                self.attn_norm(tokens),
                self.attn_norm(tokens),
                key_padding_mask=~token_mask,
                need_weights=False,
            )[0]
            tokens = tokens + attended
            tokens = tokens + self.ffn(self.ffn_norm(tokens))
            return tokens

        def forward_one_step(self, tokens: Tensor) -> Tensor:
            attended = _mha_no_mask(
                self.attn,
                self.attn_norm(tokens),
                self.attn_norm(tokens),
            )
            tokens = tokens + attended
            tokens = tokens + self.ffn(self.ffn_norm(tokens))
            return tokens

        def forward_one_step_action_only(self, tokens: Tensor) -> Tensor:
            normalized = self.attn_norm(tokens)
            action_tokens = tokens[:, 1:, :]
            attended = _mha_no_mask(
                self.attn,
                normalized[:, 1:, :],
                normalized,
            )
            action_tokens = action_tokens + attended
            action_tokens = action_tokens + self.ffn(self.ffn_norm(action_tokens))
            return action_tokens

    class _FeedForward(nn.Module):  # type: ignore[misc]
        def __init__(self, *, d_hidden: int, ffn_dim: int) -> None:
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(d_hidden, ffn_dim),
                nn.GELU(),
                nn.Linear(ffn_dim, d_hidden),
            )

        def forward(self, x: Tensor) -> Tensor:
            return self.net(x)

    def _causal_cross_mask(steps: int, device: torch.device) -> Tensor:
        mask = torch.ones((steps, steps + 1), dtype=torch.bool, device=device)
        mask[:, 0] = False
        for step in range(steps):
            mask[step, 1 : step + 2] = False
        return mask

    def _mha_single_key_value_only(attn: nn.MultiheadAttention, key_value: Tensor) -> Tensor:
        embed_dim = key_value.shape[-1]
        value_weight = attn.in_proj_weight[2 * embed_dim :, :]
        value_bias = None if attn.in_proj_bias is None else attn.in_proj_bias[2 * embed_dim :]
        value = functional.linear(key_value, value_weight, value_bias)
        return attn.out_proj(value)

    def _mha_no_mask(
        attn: nn.MultiheadAttention,
        query: Tensor,
        key_value: Tensor,
    ) -> Tensor:
        query_proj, key_proj, value_proj = _mha_project_qkv(attn, query, key_value)
        return _mha_no_mask_from_projected(attn, query_proj, key_proj, value_proj)

    def _mha_project_qkv(
        attn: nn.MultiheadAttention,
        query: Tensor,
        key_value: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor]:
        embed_dim = query.shape[-1]
        in_bias = attn.in_proj_bias
        query_proj = functional.linear(
            query,
            attn.in_proj_weight[:embed_dim, :],
            None if in_bias is None else in_bias[:embed_dim],
        )
        key_proj = functional.linear(
            key_value,
            attn.in_proj_weight[embed_dim : 2 * embed_dim, :],
            None if in_bias is None else in_bias[embed_dim : 2 * embed_dim],
        )
        value_proj = functional.linear(
            key_value,
            attn.in_proj_weight[2 * embed_dim :, :],
            None if in_bias is None else in_bias[2 * embed_dim :],
        )
        return query_proj, key_proj, value_proj

    def _mha_project_kv(
        attn: nn.MultiheadAttention,
        key_value: Tensor,
    ) -> tuple[Tensor, Tensor]:
        embed_dim = key_value.shape[-1]
        in_bias = attn.in_proj_bias
        key_proj = functional.linear(
            key_value,
            attn.in_proj_weight[embed_dim : 2 * embed_dim, :],
            None if in_bias is None else in_bias[embed_dim : 2 * embed_dim],
        )
        value_proj = functional.linear(
            key_value,
            attn.in_proj_weight[2 * embed_dim :, :],
            None if in_bias is None else in_bias[2 * embed_dim :],
        )
        return key_proj, value_proj

    def _mha_no_mask_from_projected(
        attn: nn.MultiheadAttention,
        query_proj: Tensor,
        key_proj: Tensor,
        value_proj: Tensor,
    ) -> Tensor:
        embed_dim = query_proj.shape[-1]
        head_count = attn.num_heads
        head_dim = embed_dim // head_count
        batch_size = query_proj.shape[0]
        query_heads = query_proj.view(
            batch_size,
            query_proj.shape[1],
            head_count,
            head_dim,
        ).transpose(1, 2)
        key_heads = key_proj.view(batch_size, key_proj.shape[1], head_count, head_dim).transpose(
            1, 2
        )
        value_heads = value_proj.view(
            batch_size,
            value_proj.shape[1],
            head_count,
            head_dim,
        ).transpose(1, 2)
        attended = functional.scaled_dot_product_attention(
            query_heads,
            key_heads,
            value_heads,
            dropout_p=0.0,
        )
        attended = (
            attended.transpose(1, 2)
            .contiguous()
            .view(
                batch_size,
                query_proj.shape[1],
                embed_dim,
            )
        )
        return attn.out_proj(attended)

    def _require_positive(name: str, value: int) -> None:
        if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
            raise InputError(
                f"{name} must be a positive integer",
                details={"field": name, "value": value, "type": type(value).__name__},
            )
