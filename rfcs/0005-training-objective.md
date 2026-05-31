# RFC-0005: Training objective

- **Status:** Draft
- **Author(s):** GenoLeWM Project
- **Created:** 2026-05-20
- **Updated:** 2026-05-20
- **Depends on:** RFC-0002, RFC-0003, RFC-0004
- **Supersedes:** —
- **Implementation status:** Partial — prediction loss and LeJEPA KL
  regularizer implemented; optimizer, schedule, and trainer integration
  remain open.

---

## 1. Summary

This RFC defines the training loss, the optimizer configuration, the
learning-rate schedule, and the collapse-monitoring protocol. The loss
follows the LeWorldModel two-loss recipe with one practical adaptation:
in Phase 1, because the encoder is frozen, only the prediction loss is
trained. The Gaussian regularizer becomes a live training term in
Phase 2 when LoRA adaptation is enabled and collapse becomes possible.

## 2. Motivation

The classic JEPA failure mode is representation collapse: the encoder
maps every input to (near) the same vector, making the prediction loss
trivially small. LeWM avoids this with two losses: a prediction loss
in latent space, and an isotropic-Gaussian regularizer on the encoder
output distribution (the LeJEPA contribution).

In GenoLeWM Phase 1, the encoder (Carbon-500M) is **frozen**. The
prediction targets `s_{t+1}` are therefore fixed by Carbon and cannot
collapse. The predictor alone cannot drive the encoder to a collapsed
representation. So Phase 1 needs only the prediction loss.

In Phase 2, when LoRA adaptation modifies the encoder, the standard
LeJEPA term becomes a real training loss. We specify both phases here
so the transition is mechanical.

## 3. Specification

### 3.1 Prediction loss (Phase 1 + Phase 2)

```
L_pred(ŝ, s) = α · (1 - cos(ŝ, s)) + β · ||ŝ - s||²₂ / d_state
```

with defaults:

- `α = 1.0` (cosine term)
- `β = 0.1` (scaled-MSE term)
- `d_state = 1024` (Carbon-500M hidden size)

The MSE term is normalized by `d_state` so that the two terms have
comparable scale at initialization.

**Why both?** Cosine alone is invariant to vector magnitude, which can
cause the predictor to ignore calibration; MSE alone is sensitive to
nuisance magnitude variance in the targets. The combination is more
robust empirically (this is also LeWM's published recipe with their
loss reformulated for L2-normalized embeddings).

**Per-step loss for multi-step rollout:**

```
L_pred,total = (1 / K) Σ_{k=1..K} L_pred(ŝ_{t+k}, s_{t+k})
```

i.e., a uniform-weight average over rollout steps. We considered
geometric weighting (later steps weighted more, since they are harder
to predict); rejected for v1 in favor of the simpler uniform weighting.

### 3.2 Gaussian regularizer (Phase 2 only)

When the encoder is LoRA-adapted, we add the LeJEPA isotropic-Gaussian
regularizer on encoder outputs:

```
L_reg = γ · D_KL( N(μ_batch, Σ_batch) || N(0, I) )
```

where `(μ_batch, Σ_batch)` are the empirical mean and covariance of
encoder outputs over the current batch, and `γ = 0.5` is the default
weight.

In practice we compute this as:

```
L_reg = γ · (0.5 · (||μ_batch||²₂ + tr(Σ_batch) - log det(Σ_batch) - d))
```

with numerical stabilization on the log-determinant (add small
diagonal `ε I` with `ε = 1e-6`). This is the standard closed-form
KL between a multivariate Gaussian and `N(0, I)`.

**Important:** in Phase 1, `L_reg` is computed for **monitoring only**
(logged to wandb) but not added to the training loss.

### 3.3 Phase-conditional total loss

```
Phase 1 (frozen encoder):     L = L_pred,total
Phase 2 (LoRA-adapted encoder): L = L_pred,total + L_reg
```

### 3.4 Optimizer

AdamW with:

- `β₁ = 0.9`
- `β₂ = 0.95` (lower than the typical 0.999 for stability with small
  batches over high-dimensional latents)
- `ε = 1e-8`
- `weight_decay = 0.05`
- `grad_clip = 1.0`

Separate parameter groups:

| Group | LR | Weight decay |
|-------|------|--------------|
| Predictor | 3e-4 | 0.05 |
| Action encoder | 3e-4 | 0.05 |
| LoRA adapters (Phase 2) | 1e-5 | 0.0 |
| Embedding tables (token-type, step-position) | 3e-4 | 0.0 |
| LayerNorm parameters | 3e-4 | 0.0 |

### 3.5 Learning-rate schedule

WSD (warmup-stable-decay):

- **Warmup:** 2,000 steps linear from 0 to peak LR.
- **Stable:** held at peak LR for the next 80% of training.
- **Decay:** linear from peak LR to peak LR × 0.1 over the final 18%
  (i.e., 2,000 + 80% + 18% = ~98% then a final 2% taper to peak × 0.01).

We adopt WSD over cosine because it gives the option to checkpoint at
the end of the stable phase and continue training without restart
artifacts — useful for the eventual phase transition from Phase 1
to Phase 2.

### 3.6 Collapse monitoring

Even in Phase 1, where collapse is impossible by construction, we
monitor the following metrics on a held-out validation batch every
500 steps:

| Metric | Notes |
|--------|-------|
| `pred_cos_mean` | mean cosine similarity between `ŝ` and `s` |
| `pred_l2_mean` | mean L2 distance |
| `target_var_per_dim` | per-dimension variance of `s_{t+1}` (should stay roughly constant; frozen encoder) |
| `pred_var_per_dim` | per-dimension variance of `ŝ_{t+1}` (collapse → variance → 0) |
| `pred_target_corr` | correlation between predicted and target dimensions (collapse → 0) |
| `kl_reg` | the L_reg value (monitored, not trained on, in Phase 1) |
| `pairwise_pred_dist_mean` | mean pairwise distance between predicted vectors in a batch (collapse → 0) |

Phase 2 makes `kl_reg` an active training term but does not change the
monitoring set.

**Alert criteria** during training (auto-log to wandb as warnings):
- `pred_var_per_dim < 0.5 × target_var_per_dim`
- `pairwise_pred_dist_mean < 0.5 × initial_value`
- `kl_reg > 10` (encoder distribution drifted strongly from `N(0, I)`)

### 3.7 Batching and gradient accumulation

- **Batch size:** 256 effective. On a single H100 with bf16, this fits
  as `microbatch=16, accum=16` for the default predictor
  (`d_hidden=1024`).
- **Edit-balanced sampling:** each batch contains roughly equal counts
  of each edit type. Implementation: a weighted sampler with weights
  `[0.4, 0.2, 0.2, 0.1, 0.1]` for `[SNV, INS, DEL, MNV, INDEL]`,
  chosen to balance training signal across types while leaving SNVs
  as the plurality (matching their real-world prevalence).
- **Multi-step rollout in batch:** each batch contains a mix of K=1
  (90%) and K∈{2,3} (10%) samples, drawn from gnomAD's phased multi-edit
  data. K is bumped progressively in Phase 2 with curriculum
  (K_max: 1 → 3 → 5 → 8 over the first 30% of Phase 2 training).

### 3.8 Training compute budget

| Phase | Tokens (target embeddings produced) | Wall clock (single H100) |
|-------|-------------------------------------|---------------------------|
| Phase 1 baseline | ~10B target embeddings | ~24 h |
| Phase 1 full | ~50B target embeddings | ~5 days |
| Phase 2 (LoRA + full edit suite) | ~100B target embeddings | ~10 days |

These are estimates with the default predictor (~40M params). With the
smaller variant (~22M params), wall-clock drops by ~30%.

### 3.9 Reproducibility

- **Random seeds:** the data sampler, predictor init, and LoRA init
  all consume distinct seeds, recorded in the training config.
- **Deterministic mode:** when `--deterministic` is set, PyTorch's
  deterministic algorithms are enabled and CuBLAS workspace is
  configured. Throughput drops ~15%, but bit-exact reproduction is
  possible.
- **Logging:** wandb is the default; logs include all metrics in §3.6,
  the optimizer state hash every 10k steps, and the final eval report.

## 4. Rationale and alternatives

### 4.1 Why two loss components instead of just cosine?

Cosine alone leaves magnitude unconstrained. In a frozen-encoder
regime, the target magnitudes are roughly constant (Carbon's
representations have a consistent scale), but the predictor under
pure-cosine training can output vectors of any magnitude that happen
to have the right angle. This degrades downstream uses (the surprise
score in RFC-0009 is a distance, which is magnitude-sensitive).

Adding a small MSE term (`β = 0.1`) preserves magnitude calibration at
low cost. We will ablate `β ∈ {0.0, 0.05, 0.1, 0.5, 1.0}` in Phase 1.

### 4.2 Why not L1 (Smooth L1) for the magnitude term?

L1 is more robust to outliers, but in latent space, outlier targets
are usually *interesting* (rare biology), so we want the predictor to
attend to them. L2 (MSE) is the right choice.

### 4.3 Why WSD instead of cosine LR?

WSD is friendlier to phase transitions: we can train Phase 1 to the end
of its stable phase, save a checkpoint, then begin Phase 2 from there
with a fresh decay schedule. With cosine LR, the optimizer state would
be in an awkward late-decay regime when LoRA training begins.

### 4.4 Why is `L_reg` not active in Phase 1?

Two reasons.

1. **Mechanically impossible to need it.** Targets are fixed by a
   frozen encoder; the predictor cannot drive encoder outputs.
2. **Computational simplicity.** The covariance computation is non-trivial
   for `d_state = 1024`; skipping it in Phase 1 saves 5–10% of training
   time.

We do compute it for monitoring, so we can ensure it does not drift
unexpectedly (which would indicate, e.g., a bug in encoder caching).

### 4.5 Why batch size 256?

LeWM uses 256 globally. With Carbon-500M targets, batch 256 produces
~256 unique 1024-dim targets per step, which is enough for stable
covariance estimation in Phase 2. Smaller batches risk noisy `L_reg`
estimates.

### 4.6 Why edit-balanced sampling?

A realistic distribution would be ~95% SNV / 5% indel, matching gnomAD.
That gives the predictor almost no indel signal. Up-weighting indels
during training is a standard rebalance; the predictor still sees more
SNVs (40%) than any other type, matching their importance.

## 5. Unresolved questions

- The exact `(α, β)` ratio. The 1.0 / 0.1 starting point is a guess.
- Whether to use a per-edit-type loss weight (e.g., harder edit types
  contribute more loss). Probably not — we want even per-step loss.
- Whether to add a per-position consistency loss that says "the
  prediction at position p should change linearly with the encoder
  output at position p when edits are local". This would encode the
  spatial inductive bias but is non-trivial to implement; deferred to
  a possible v2 RFC.
- Whether to use SGD with momentum instead of AdamW. AdamW is the safe
  default; SGD might generalize better in the long Phase 2 runs.

## 6. Future work

- **Curriculum on edit difficulty.** Start with SNVs in well-conserved
  regions, advance to indels in repeat regions. Plausibly faster
  convergence.
- **Contrastive auxiliary.** Add a SimCLR-style contrastive term across
  edits in the same window. Adds compute; unclear payoff.
- **Reward-weighted training.** If we have a downstream evaluator
  (e.g., pathogenicity AUROC), use it as a reward signal to bias
  training. Reserved for late v2.

## 7. Changelog

- 2026-05-20 — Initial draft.
