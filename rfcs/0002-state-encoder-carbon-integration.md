# RFC-0002: State encoder — Carbon integration

- **Status:** Draft
- **Author(s):** GenoLeWM Project
- **Created:** 2026-05-20
- **Updated:** 2026-06-02
- **Depends on:** RFC-0001
- **Supersedes:** —
- **Implementation status:** Partial — local windowing, tokenizer/model
  wrapping, pooling, cache schema/read/write/reindex/repair primitives,
  and lazy `CarbonStateEncoder` with injected-component tests exist.
  Clean-machine validation against pinned Carbon weights and full
  selected-corpus cache-build throughput evidence remain open.

---

## 1. Summary

This RFC defines how GenoLeWM extracts state vectors from Carbon. It
specifies the encoder identity (default Carbon-500M), the input window
format (length, alignment, tokenization), the hidden-layer selection,
the pooling strategy, the output dimensionality, the caching contract,
and the (optional, deferred to Phase 2) LoRA adaptation strategy.

## 2. Motivation

The state encoder is the heaviest component in the system. Every
choice in this RFC has direct consequences for training cost, inference
cost, and downstream metric quality. We want the choices to be:

- **Reproducible.** A specific Carbon revision is pinned.
- **Cacheable.** Reference embeddings are computed once and reused.
- **Edit-aware.** Pooling concentrates around the edit locus so that
  the state vector is sensitive to local context (where edits act)
  rather than diluted across a long window.
- **Encoder-version-aware.** Every checkpoint identifies which Carbon
  variant produced its training targets.

## 3. Specification

### 3.1 Encoder identity

The default encoder is:

```
encoder = "HuggingFaceBio/Carbon-500M"
revision = "main"   # pinned to a specific commit SHA in config
```

Carbon-3B and Carbon-8B are supported alternatives, selected via
configuration. The encoder identity (model id, commit SHA, dtype) is
recorded in the GenoLeWM checkpoint's `config.json` and in the
`encoder_hash.txt` file.

The encoder is loaded in bf16 by default, with fp16 and fp32 supported
for environments without bf16 (consumer GPUs predating Ampere, some
ARM SoCs).

### 3.2 Window format

A GenoLeWM input window is:

- **Length:** 2,048 6-mer tokens = **12,288 base pairs** (default).
- **Alignment:** the window length in base pairs is constrained to a
  multiple of 6 (Carbon's 6-mer tokenizer requirement).
- **Centering:** the window is centered on the edit locus when one is
  specified. For windows extracted during pretraining (where no edit
  yet exists), the center is the midpoint.
- **DNA tag:** every window is wrapped in `<dna>...</dna>` exactly
  per Carbon's tokenizer requirements (Carbon-3B README §3.1).
- **Strand:** forward strand only in v1. Reverse-complement augmentation
  is reserved for Phase 2.
- **Padding:** if the source sequence is shorter than 12,288 bp at the
  chosen genomic locus, the window is right-padded with `A`s (matching
  Carbon's tokenizer's documented behavior).

Smaller (4,096 bp / 1,024 6-mer tokens) and larger (24,576 bp / 4,096
6-mer tokens) window sizes are supported as configuration overrides for
ablation studies. The default of 12,288 bp is chosen because it (a)
fits 32 windows per sequence into a typical Carbon-pretraining-corpus
shard, (b) covers most exons and many full short genes, and (c)
keeps Carbon-500M inference under ~80 ms per window on an H100.

### 3.3 Hidden-layer selection

The state vector is derived from one of Carbon's transformer hidden
states.

```python
state_layer = -1   # default: final layer
state_layer ∈ {-1, -2, -3, -4}   # supported values
```

Rationale for last-layer default:
- Carbon's training objective (cross-entropy → factorized nucleotide
  supervision) makes the final layer's representation most directly
  next-token-predictive, which we hypothesize correlates with sensitivity
  to local sequence context.
- This matches the LeWM convention of using the encoder's final
  representation.

The penultimate layer is hypothesized to retain more general-purpose
information; this is the first ablation in Phase 1.

### 3.4 Pooling

Pooling collapses Carbon's per-token hidden states (`seq_len × hidden`)
into a single state vector (`hidden`). Three pooling strategies are
supported:

```
pool_type ∈ {"centered_mean", "global_mean", "attention"}
```

- **`centered_mean` (default).** Compute the mean of hidden states over
  the ± `pool_radius` tokens centered on the edit locus. Default
  `pool_radius = 256` tokens = ± 1,536 bp = a 3 kbp window of
  attention. This is the recommended default because it makes the state
  vector edit-local: an edit's downstream effect on the state vector
  is concentrated where the edit happens.
- **`global_mean`.** Mean over the entire window. Use this for windows
  with no specified edit locus (e.g., during pretraining encoding of
  arbitrary reference windows).
- **`attention`.** A learned single-head attention pool with the edit
  locus as the query position. This adds ~1M parameters and is
  reserved for ablation; not in the default path.

When no edit locus is specified, `global_mean` is used as a fallback,
and the resulting state is tagged `untargeted=True` in the cache so
downstream consumers do not mix targeted and untargeted embeddings.

### 3.5 Output

The state vector is:

```
s_t ∈ ℝ^{d_state}
d_state = 1024   # Carbon-500M hidden size
```

For Carbon-3B and Carbon-8B, `d_state` is 3072 and 4096 respectively.

The output is **L2-normalized by default** before being passed to the
predictor. Normalization is configured at the encoder level, not the
predictor level, so that any downstream consumer (rollout, surprise)
operates in a consistent geometry.

### 3.6 Caching

Reference-window embeddings are cached on disk to avoid re-encoding.

**Format:** Parquet, one shard per (chromosome × `cache_stride_bp`)
block.

**Schema:**

| column            | type             | notes |
|-------------------|------------------|-------|
| `chrom`           | string           | chromosome name |
| `start_bp`        | int64            | window start (inclusive) |
| `end_bp`          | int64            | window end (exclusive) |
| `window_hash`     | bytes (32)       | SHA-256 of the window DNA string |
| `encoder_hash`    | bytes (32)       | SHA-256 of the encoder weights file |
| `state_layer`    | int8             | layer index used |
| `pool_type`       | string           | pooling strategy used |
| `pool_radius`     | int32            | pooling radius (tokens) |
| `embedding`       | list\<float16\>  | the state vector |

The cache is **content-addressed**: a row is uniquely identified by
`(window_hash, encoder_hash, state_layer, pool_type, pool_radius)`.
Changing any of these fields invalidates the cached entry.

**Cache lifecycle:**
- Reference windows are encoded eagerly at training start and stored.
- Edited windows are encoded on-the-fly during training (since edits
  vary per epoch) and **not** cached.
- At inference time, a small LRU cache (default 10,000 entries) holds
  the most recently-used edited-window embeddings.

**Cache size budget:** with 12,288 bp windows striding by 8,192 bp over
the human genome (~3 Gbp), we get ~370k windows × 1024 dim × 2 bytes
(fp16) ≈ **750 MB per encoder configuration**. That is the budget for
the default config.

### 3.7 LoRA adaptation (Phase 2, optional)

In Phase 1, the encoder is **frozen**. In Phase 2, optional
LoRA adaptation is supported.

```
lora_rank = 16
lora_alpha = 32
lora_target_modules = ["q_proj", "k_proj", "v_proj", "o_proj"]
lora_layers = "last_8"   # only the top 8 transformer layers
```

Rationale: targeting only the top layers (a) reduces parameter count to
~5M, (b) preserves Carbon's pretrained early-layer features which are
the most general-purpose, and (c) lets the encoder specialize its
late-layer outputs toward the GenoLeWM prediction loss.

When LoRA is active, the **L_reg** (LeJEPA isotropic-Gaussian)
regularizer becomes a live training term (see RFC-0005 §4); without
it, the encoder can collapse to make the predictor's job trivial.

### 3.8 Encoder API

The module `geno_lewm.encoder.carbon` exposes:

```python
class CarbonStateEncoder:
    def __init__(self, model_id: str, revision: str,
                 dtype: str = "bf16",
                 state_layer: int = -1,
                 pool_type: str = "centered_mean",
                 pool_radius: int = 256,
                 normalize: bool = True,
                 lora_config: LoRAConfig | None = None) -> None: ...

    def encode(self, window: str, edit_locus: int | None = None) -> Tensor:
        """Encode a single DNA window. Returns shape (d_state,)."""

    def encode_batch(self, windows: list[str],
                     edit_loci: list[int | None]) -> Tensor:
        """Batched encode. Returns shape (B, d_state)."""

    @property
    def encoder_hash(self) -> bytes: ...

    @property
    def d_state(self) -> int: ...
```

The encoder is a `torch.nn.Module`; in Phase 1 its parameters are
frozen via `requires_grad_(False)`, so gradient computation skips it
automatically.

## 4. Rationale and alternatives

### 4.1 Why centered-mean pooling, not [CLS]-style pooling?

Carbon does not emit a `[CLS]` token; its tokenizer wraps in `<dna>`
but the model is a causal autoregressive LM, not a bidirectional
encoder. Pooling is therefore the natural operation.

We considered:
- **Last-token pooling.** Common for autoregressive LMs. Rejected
  because for genomics the "last token" of a window is an arbitrary
  position, not semantically meaningful.
- **Global mean.** Simple. Used as a fallback. Rejected as default
  because it dilutes the edit's effect across 2,048 tokens.
- **Centered mean (chosen).** Edit-local, simple, no extra parameters.
- **Attention pooling.** More expressive but adds parameters and a
  failure mode (the attention head can collapse to a constant).

### 4.2 Why freeze the encoder in Phase 1?

Three reasons.

1. **Cost.** Carbon-500M is 500M parameters. Fine-tuning it for every
   GenoLeWM experiment is expensive; freezing makes the project
   trainable on a single H100.
2. **Stability.** With a frozen encoder, the prediction targets
   `s_{t+1}` are fixed; this rules out collapse modes that LeJEPA was
   designed to address. Phase 1 thus has a simpler loss (RFC-0005 §3).
3. **Reproducibility.** A frozen encoder means GenoLeWM checkpoints
   are interpretable as predictor heads. Multiple GenoLeWM versions
   can share the same encoder cache.

### 4.3 Why Carbon-500M as default, not Carbon-3B?

500M is the smallest variant that meets Carbon's quality bar (the model
card shows competitive numbers across the eval suite). It is the only
variant that fits in 16 GB at bf16 alongside a small predictor on
consumer hardware, which is a Phase 3 requirement. Choosing 500M now
saves a re-port later.

We will also train against 3B and 8B in Phase 2 for benchmark
comparison; the architecture is encoder-agnostic.

### 4.4 Why a 12,288 bp window?

This corresponds to 2,048 6-mer tokens. Empirical justification will
come from ablation in Phase 1. A priori reasoning:

- Most exons are < 1 kbp; most short genes are < 10 kbp. A 12 kbp
  window covers them with margin.
- 2,048 tokens is the "middle ground" for transformer attention cost:
  short enough to run cheaply (~80 ms / window on H100 with bf16),
  long enough to capture useful sequence context.
- Carbon's native context is 32 k tokens; we use a fraction of that
  for per-edit work to keep encoding cost manageable.

### 4.5 Why L2-normalize state vectors?

Two reasons.

1. The dominant prediction loss (`L_pred`, RFC-0005) combines cosine
   and MSE terms. Normalized states make the cosine term invariant to
   magnitude and the MSE term operate on directional residuals.
2. L2-normalized embeddings produce more stable distances in the
   surprise calculation (RFC-0009), where percentile calibration depends
   on a consistent embedding scale.

We acknowledge this is a non-trivial choice; an ablation comparing
normalized vs un-normalized is in the Phase 1 plan.

## 5. Unresolved questions

- **State layer selection.** Last vs penultimate vs concat-of-last-4.
  Ablate in Phase 1.
- **Pooling radius.** ±256 tokens (≈ 1.5 kbp) is a guess. Optimal
  radius likely depends on edit type (SNV vs large indel).
- **Reverse complement.** Whether to encode both strands and average,
  or pick one. Carbon was trained on forward strand only; we follow.
  Phase 2 ablation will check whether RC augmentation helps.
- **Cache invalidation policy.** When Carbon publishes a new revision,
  the encoder hash changes and all caches invalidate. We need a
  graceful migration path.

## 6. Future work

- Multi-encoder support (Evo2, Generator-v2, Nucleotide Transformer)
  as alternative encoders behind the same API.
- Distillation of Carbon-3B-derived states into a Carbon-500M-cached
  representation, for users who want 3B quality at 500M cost.
- Per-cell-type encoder conditioning, if Carbon or a successor adds
  this capability.

## 7. Changelog

- 2026-06-01 — Added lazy `CarbonStateEncoder` implementation status:
  local Transformers loading defaults to `local_files_only=True`, and
  tests can inject model/tokenizer objects without the optional ML
  runtime.
- 2026-05-20 — Initial draft.
