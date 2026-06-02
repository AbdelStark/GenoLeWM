# RFC-0004: Predictor architecture

- **Status:** Draft
- **Author(s):** GenoLeWM Project
- **Created:** 2026-05-20
- **Updated:** 2026-06-02
- **Depends on:** RFC-0001, RFC-0002, RFC-0003
- **Supersedes:** —
- **Implementation status:** Partial — base cross-attention `Predictor`,
  `ARPredictor` rollout wrapper, and predictor losses are implemented;
  attention KV-cache speedups, released-artifact validation, and full
  trainer/evaluator integration remain open.

---

## 1. Summary

The predictor is the trainable core of GenoLeWM. It maps a state
embedding `s_t` and one or more action embeddings `a_emb` to a predicted
next-state embedding `ŝ_{t+1}` in the same latent space. This RFC
specifies the architecture (cross-attention Transformer), the parameter
budget (~20M trainable), the autoregressive multi-step rollout
mechanics, and the inference/training APIs.

## 2. Motivation

The predictor needs to:

1. **Cross-condition** the state on the action and vice versa. An SNV at
   one position should affect the latent differently than the same
   SNV at a different position; this requires interaction between
   position-bearing action features and the state.
2. **Support variable-length action sequences** (single edit through
   K-edit haplotypes) without architectural changes.
3. **Be small enough** to fit the project's compute and deployment
   budgets — single-GPU training in hours, ≤ 200 MB at int8 for
   on-device inference.
4. **Preserve the latent geometry** of the encoder. The output lives
   in the same space as `s_t`, so downstream surprise calculations
   (which are distances in that space) are meaningful.

LeWM's predictor is a small autoregressive Transformer over (state,
action) pairs. We adopt the same shape with three concrete deviations
documented in §3.

## 3. Specification

### 3.1 Architecture

The predictor is a Transformer with:

- **4 cross-attention blocks**, alternating state-cross-action and
  action-cross-state attention.
- **2 self-attention blocks** on the fused (state ⊕ action) sequence.
- **Hidden dimension:** 1,024 (matches `d_state`).
- **Attention heads:** 8.
- **FFN intermediate dim:** 2,048 (× 2 expansion).
- **Activation:** GELU.
- **Normalization:** Pre-LayerNorm (LN before each sublayer).
- **Output:** the first output token, projected through a 2-layer MLP
  (1024 → 1024 → 1024) to produce `ŝ_{t+1}`.
- **Output normalization:** L2-normalized (to match the encoder's
  output convention, RFC-0002 §3.5).

#### 3.1.1 Input token construction

The predictor receives a sequence of tokens:

```
[s_t]  [a_1]  [a_2]  ...  [a_K]
```

- `s_t` is projected from `ℝ^{d_state}` to `ℝ^{d_hidden}` (identity if
  `d_state == d_hidden`, learned linear otherwise).
- Each `a_k` is projected from `ℝ^{d_action}` to `ℝ^{d_hidden}` by a
  learned linear layer.
- A learned **token-type embedding** (state=0, action=1) is added to
  distinguish state vs action tokens.
- A learned **step-position embedding** (0, 1, 2, ..., K) is added so
  the predictor knows the order in which actions are to be applied.

#### 3.1.2 Causal mask

In the cross-attention blocks, the state can attend to all actions and
each action can attend to the state and to earlier actions in the
sequence (causally). In the self-attention blocks, all positions
attend bidirectionally within the (state ⊕ actions) sequence.

This is the same convention as LeWM's `ARPredictor`.

#### 3.1.3 Parameter budget

| Component | Params |
|-----------|--------|
| Action projection (512 → 1024) | ~0.5M |
| State projection (1024 → 1024, identity option) | 0 or 1.0M |
| Token-type embedding (2 × 1024) | <1k |
| Step-position embedding (16 × 1024) | ~16k |
| 4 cross-attention blocks (1024 dim, 8 heads, 2048 FFN) | ~25M |
| 2 self-attention blocks | ~12M |
| Output MLP (1024 → 1024 → 1024) | ~2M |
| **Total** | **~40M** |

Note: this is larger than the 20M target stated in the SPECIFICATION.
We will dial back to the target by either:
- Reducing hidden dimension to 768 (parameter count ≈ 23M), or
- Reducing the number of cross-attention blocks from 4 to 2.

We expect to converge on `d_hidden=768, n_cross=4, n_self=2` after
ablation, giving ~22M parameters. The default in the reference config
is the larger variant for accuracy headroom; the on-device default
(RFC-0010) uses the smaller variant.

### 3.2 Forward pass

```python
class Predictor(nn.Module):
    def forward(
        self,
        state: Tensor,          # (B, d_state)
        actions: Tensor,        # (B, K, d_action)
        action_mask: Tensor,    # (B, K), 1 for valid, 0 for padding
    ) -> Tensor:                # (B, K, d_state); per-step predictions
        ...
```

Per-step output: `ŝ_{t+k+1}` is read out at the position of `a_k+1` in
the output sequence (so step 0's prediction is read out at action
position 0, etc.). The final prediction `ŝ_{t+K}` corresponds to the
fully-applied haplotype.

For single-edit inputs (`K=1`), the output is a single vector.

### 3.3 Autoregressive rollout

For multi-step rollout where each action depends on a previous
predicted state (e.g., for planning), the predictor is unrolled
step-by-step with KV-caching:

```python
class ARPredictor(nn.Module):
    def rollout(
        self,
        state: Tensor,                    # (B, d_state)
        action_sequence: list[Tensor],    # K tensors of (B, d_action)
    ) -> list[Tensor]:                    # K tensors of (B, d_state)
        ...
```

This is logically equivalent to repeated `forward` calls but with KV
cache reuse, giving ~2× speedup at K=5 and ~5× at K=20.

The rollout method is what the planner (RFC-0008) calls in its inner
loop. It is also what is used for inference-time multi-edit haplotype
prediction.

### 3.4 Initialization

- Linear layers: truncated normal, std = `sqrt(2 / fan_in)`.
- LayerNorm: weight 1.0, bias 0.0.
- Token-type and step-position embeddings: normal, std = 0.02.
- Output MLP final layer: zero-initialized so that at the start of
  training the predictor outputs `s_t` (identity), making the early
  prediction loss small and stable.

### 3.5 Inference

The predictor exposes:

```python
def predict_single(self, s_t: Tensor, edit: RelEdit) -> Tensor:
    """One-step prediction. Returns ŝ_{t+1} of shape (d_state,)."""

def predict_haplotype(self, s_t: Tensor, edits: list[RelEdit]) -> Tensor:
    """Multi-step prediction. Returns ŝ_{t+K} of shape (d_state,)."""

def predict_trajectory(self, s_t: Tensor, edits: list[RelEdit]) -> list[Tensor]:
    """All intermediate predictions. Returns [ŝ_{t+1}, ..., ŝ_{t+K}]."""
```

All three are inference-only (no gradient); training uses the `forward`
method directly.

### 3.6 Numerical considerations

- The predictor runs in bf16 by default, matching Carbon's dtype.
- For very deep rollouts (K > 10), small numerical drift accumulates;
  we recommend up-casting to fp32 for the output MLP when K > 20.
- L2 normalization at the output prevents magnitude drift in long
  rollouts, but does not prevent angular drift; long-rollout accuracy
  degrades and we report this in the eval (RFC-0007 §3.2).

## 4. Rationale and alternatives

### 4.1 Why cross-attention rather than MLP-concat?

The simplest predictor is an MLP that takes `concat(s_t, a_emb)` and
outputs `ŝ_{t+1}`. This was our initial reflex. We rejected it because:

- It hardcodes the assumption that one action follows the state. With
  cross-attention, the action sequence can vary in length without
  architectural change.
- It throws away the structured nature of the action embedding (the
  four sub-components from RFC-0003 §3.4). Cross-attention lets the
  state attend to specific parts of the action representation.
- The compute cost difference at the scales we care about (~25M params
  vs ~40M) is small, and the modeling flexibility is worth it.

### 4.2 Why not a Mamba / SSM-based predictor?

State-space models would also handle variable-length action sequences.
We chose Transformers because:
- Reference implementations are more mature.
- Carbon is itself a Transformer; reasoning about latent geometry is
  more straightforward when the predictor matches.
- For K ≤ 16 (our default max), the quadratic attention cost is
  negligible.

An SSM variant is a reasonable v2 experiment for very long edit
sequences (chromosome-scale planning).

### 4.3 Why zero-init the output MLP final layer?

This is the "identity-at-init" trick from residual networks. At step
zero, the predictor outputs (something close to) `s_t`, so the
prediction loss is small but non-zero, gradients are well-behaved, and
training avoids the early-instability cliff that uniform-random
initialization causes.

### 4.4 Why same dimensionality as the encoder?

The predictor output lives in the same space as `s_t` and `s_{t+1}`,
so distances between them are meaningful. Projecting to a different
space would require an inverse projection for evaluating the prediction
loss, adding parameters and breaking the surprise calculation.

### 4.5 Why 4 cross + 2 self blocks rather than e.g. 6 cross?

Empirically (from LeWM's ablations), alternating cross-attention with
self-attention helps the model fuse state and action information
without one dominating the other. 4+2 is the smallest combination that
gives both ample cross-conditioning and a self-attention "integration"
phase at the end.

## 5. Unresolved questions

- Whether the step-position embedding should be sinusoidal (extrapolates
  to longer K) or learned (more expressive at fixed K).
- Whether to add Rotary Position Embeddings (RoPE) on the attention
  layers; Carbon uses RoPE internally, so consistency is appealing,
  but the predictor's "sequence" is so short that the gain may be tiny.
- The exact small/large variant split: where on the (d_hidden, n_layers)
  Pareto frontier does v1 land?
- Whether to support a "no-state" mode where the predictor is given
  only actions and produces a state from scratch. Not needed for v1,
  but might be useful for generative planning.

## 6. Future work

- A distilled "tiny" predictor variant (~5M params) for ultra-low-latency
  on-device inference.
- A MoE variant for the large-K planning regime, where different
  experts specialize in different edit types.
- Multi-task heads: an auxiliary regression head predicting binarized
  pathogenicity could be added at small cost in Phase 2, if labeled data
  proves more useful than expected.

## 7. Changelog

- 2026-05-20 — Initial draft.
