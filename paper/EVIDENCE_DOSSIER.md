# GenoLeWM — Consolidated Evidence Dossier

**Status:** Single source of truth for the v0.2.1-r1 negative-results + systems/reproducibility paper.
**Honest framing (non-negotiable):** GenoLeWM does NOT beat Carbon zero-shot broadly. No clinical, privacy, runtime-assurance, or deployment claims. The strongest positive claim is that the repository can train / eval / benchmark / replay an action-conditioned genomic-edit latent world model with content-addressed (checksum-receipt) evidence. Never inflate negatives into wins.

---

## CANONICAL GROUND-TRUTH NUMBERS (v0.2.1-r1) — AUTHORITATIVE, DO NOT ALTER

**Identity**
- `model_release = geno-lewm-v0.2.1-r1`
- `model_id = sha256:cddb8f3b9671090201370b9824b9da741b933ff296b651238f022df5f3ed6af4`
- `dataset_snapshot = geno-lewm-data-v0.2.1-r1`
- `commit = d9b06815cf8e64860f51d236b8db6ba55aa4154d`
- `hardware = NVIDIA H200 (Linux x86_64, glibc2.35)`

**VEP track**
- `clinvar_coding` (status pass): accuracy=0.75, AUROC=0.734375, average_precision=0.852976, balanced_accuracy=0.75 ; deltas vs Carbon zero-shot: accuracy=+0.0625, AUROC=-0.1875, AP=-0.098947, balanced_accuracy=+0.0625 ; issues #53,#55,#56,#197.
- `clinvar_noncoding` (pass): accuracy=0.4375, AUROC=0.5625, AP=0.605456, balanced_accuracy=0.4375 ; deltas: accuracy=-0.25, AUROC=-0.3125, AP=-0.308967, balanced_accuracy=-0.25.
- `brca2_saturation` (pass): Spearman_rho=0.149194 ; delta=-0.327713.
- `traitgym_mendelian` (pass): Spearman_rho=-0.0279645 ; delta=+0.055929.

**Latent rollout fidelity** (baseline = source-state s_t, i.e. predicting "no change")
- `rollout_phased_haplotypes` (pass): cosine_mean=0.288861, l2_mean=33.3197, recall_at_k=1.0 ; deltas vs source-state: cosine=-0.70897, l2=+31.1929, recall_at_k=0.0.
- `rollout_synthetic_edit_chains` (pass): cosine_mean=0.301608, l2_mean=28.8029, recall_at_k=1.0 ; deltas: cosine=-0.689631, l2=+25.6371, recall_at_k=0.0.

**Inference efficiency** (single released score path, H200)
- `single_variant_latency_ms = 115262.939968` (~115 s)
- `batched_throughput_variants_per_s = 0.3095340544239052` (~0.31/s)
- `peak_memory_bytes = 1966149632` (~1.83 GiB / 1966 MB)

**AR rollout speed** (status rescoped, #42 open)
- `K5_speedup = 2.41386`, `K20_speedup = 2.47322` ; RFC-0004 target was 5.0x at K=20.

**Planning demo**
- `best_distance = 23.656930390534644` (~23.66), `n_evaluations = 384`, `elapsed_seconds ≈ 15.34`, `stopped_reason = patience`.

**Artifact provenance**
- `release_inputs` status pass, all 28 suite outputs verified.

**Sample sizes (per HF model card)**
| Slice | N |
|---|---|
| clinvar_coding | 16 |
| clinvar_noncoding | 16 |
| brca2_saturation | 32 |
| traitgym_mendelian | 32 |
| rollout_phased_haplotypes | 8 |
| rollout_synthetic_edit_chains | 8 |

ClinVar deltas/metrics are quantized in steps of 0.0625 = 1/16, confirming N=16 per ClinVar slice (statistically underpowered).

---
---

# PART I — EVIDENCE CARDS (verbatim from subsystem readers)

---

## EVIDENCE CARD: R1-encoder-action

### Summary

The GenoLeWM state encoder wraps Carbon-500M — a 500M-parameter, llama-architecture causal DNA language model (hidden size 1024) from HuggingFaceBio — and extracts hidden states from a single selected transformer layer (default: layer 20, pinned to commit `5d31d59b3c845b288a13aedb1358934196852eec`). Input to Carbon is a 12,288 bp (2,048 six-mer-token) window centered on the edit locus, wrapped in `<dna>...</dna>` tags required by Carbon's custom `HybridDNATokenizer` (`trust_remote_code=True`). Per-token hidden states are collapsed via centered-mean pooling over a ±`pool_radius`-token window around the edit locus (default `pool_radius=8` tokens = ±48 bp in code; the RFC specifies ±256 tokens). The encoder is frozen in Phase 1; its output is L2-normalized before being passed to the predictor, yielding a state vector `s_t ∈ ℝ^{1024}`. Reference-window embeddings are cached in content-addressed Parquet shards indexed by a SQLite index; edited windows are encoded on-the-fly. The action subsystem encodes a genomic edit — represented as `EditSpec` (absolute VCF coordinates) or its window-relative form `RelEdit` — into a fixed-size action embedding through four sub-encoders (sinusoidal position, learned type table, shared SeqMicroEncoder for ref/alt bases) fed into a 2-layer MLP projection, targeting `d_action=64` in the paper's first experiment (not the `d_action=512` stated in the RFC).

### Key facts

#### State Encoder

- **Carbon model identity:** `HuggingFaceBio/Carbon-500M`, described as "llama arch" in config comment, `hidden_size=1024`; loaded via `AutoModel` + `AutoTokenizer` with `trust_remote_code=True`. [VERIFIED: `configs/first_experiment/train-carbon-500m-snv.yaml:20-31`, `geno_lewm/encoder/carbon.py:232-237`]
- **Pinned revision:** SHA `5d31d59b3c845b288a13aedb1358934196852eec`. [VERIFIED: `configs/first_experiment/train-carbon-500m-snv.yaml:22`]
- **Encoder dtype:** `bf16` (default; `fp16` and `fp32` supported). [VERIFIED: `carbon.py:29`, `train-carbon-500m-snv.yaml:23`]
- **Tokenizer type:** `HybridDNATokenizer` (Carbon custom tokenizer, requires `trust_remote_code=True`). Window must be wrapped as `<dna>SEQUENCE</dna>` and length must be a multiple of 6 bp (6-mer tokenizer). [VERIFIED: `windowing.py:34-36`, config comment line 28-30]
- **Window size:** Default `DEFAULT_WINDOW_BP = 12_288` bp = 2,048 6-mer tokens. Supported sizes: `(4096, 12288, 24576)` bp. Centering: on edit locus when supplied, otherwise on source midpoint. Right-padding with `A` when source sequence is shorter than window. [VERIFIED: `windowing.py:37-38`, `windowing.py:109-156`]
- **Hidden-layer selection (state_layer):** Config pins `state_layer: 20` (an explicit positive integer index, not the final layer). [VERIFIED: `train-carbon-500m-snv.yaml:24`]. RFC-0002 §3.3 specifies the default as `state_layer = -1` (final layer). **Discrepancy — see Reconcile section.**
- **Pool type:** `centered_mean` (default constant `POOL_CENTERED_MEAN`). Fallback to `global_mean` when `edit_locus=None`, tagged `untargeted=True`. `attention` pooling is not implemented (raises InputError). [VERIFIED: `pooling.py:31-36`, `pooling.py:106-114`, `pooling.py:222-226`, config line 25]
- **Pool radius:** `DEFAULT_POOL_RADIUS_TOKENS = 256` tokens (defined in `pooling.py:37`). Centered-mean spans tokens `[center - radius, center + radius]` inclusive. [VERIFIED: `pooling.py:37`, `pooling.py:71-84`]. However, config pins `pool_radius: 8`. **Discrepancy — see Reconcile section.**
- **Edit-locus to token conversion:** `token = edit_locus // token_bp` where `token_bp = 6`. [VERIFIED: `pooling.py:245`]
- **Normalize:** L2-normalization applied by `CarbonStateEncoder` when `normalize=True` (the constructor stores the flag; `encode_batch` returns pooled tuples — normalization is applied at a layer above this module; the config sets `normalize: true`). [VERIFIED: `train-carbon-500m-snv.yaml:27`, `carbon.py:46`; actual L2 normalization call not found in `carbon.py` itself — see Caveats]
- **Encoder frozen in Phase 1:** `lora_config=None` is enforced (any non-None `lora_config` raises `RuntimeSetupError`); model loaded and called with `torch.no_grad()` via `torch_inference_context()`. [VERIFIED: `carbon.py:86-90`, `carbon.py:159`]
- **State vector dimension:** `d_state = 1024` (Carbon-500M hidden size; auto-read from `model.config.hidden_size`). [VERIFIED: `carbon.py:122-125`, `train-carbon-500m-snv.yaml:40`]
- **Cache format:** Parquet shards, zstd compression level 9, one shard per (contig × stride_block). Schema fields: `chrom`, `start_bp`, `end_bp`, `window_hash` (SHA-256, 32 bytes binary), `encoder_hash` (32 bytes binary), `state_layer` (int8), `pool_type`, `pool_radius` (int32), `dtype`, `embedding` (list\<float16\>), `untargeted` (bool), `created_at` (int64 ns), `schema_version`. SQLite index at `embeddings/index.sqlite` keyed on `(window_hash, encoder_hash, state_layer, pool_type, pool_radius, dtype)`. [VERIFIED: `cache.py:412-429`, `cache.py:575-591`]
- **Cache size estimate (RFC):** ~370k windows × 1024 × 2 bytes ≈ 750 MB for the default human-genome config. [SPEC: RFC-0002 §3.6]
- **Encoder hash:** SHA-256 of the encoder weights file, 32 bytes; stored as `encoder_hash.txt` in the GenoLeWM checkpoint. [SPEC: RFC-0002 §3.1, VERIFIED: `cache.py:52-65`]
- **Supported window sizes (code):** `(4096, 12288, 24576)`. These exactly correspond to `(682, 2048, 4096)` 6-mer tokens. [VERIFIED: `windowing.py:38`]

#### Action Representation

- **Edit data model — `EditSpec`:** Frozen dataclass with `chrom` (str), `pos` (int, 1-based VCF), `ref` (str, uppercase ACGT only, ≥1 bp), `alt` (str, uppercase ACGT only, ≥1 bp), `edit_type` (derived `EditType`, not user-supplied). `ref != alt` enforced. [VERIFIED: `spec.py:95-153`]
- **`V1_MAX_LEN = 16`:** Maximum allowed `len(ref)` or `len(alt)`; edits with either exceeding 16 raise `UnsupportedEditError`. [VERIFIED: `spec.py:35`]
- **Edit types (IntEnum):** `SNV=0` (|ref|=1, |alt|=1), `INS=1` (|ref|=1, |alt|>1), `DEL=2` (|ref|>1, |alt|=1), `MNV=3` (|ref|=|alt|>1), `INDEL=4` (|ref|≠|alt|, both>1), `SV=5` (any dimension > 16). Derived automatically from (ref, alt) lengths at construction. [VERIFIED: `spec.py:38-69`]
- **`RelEdit`:** Frozen dataclass: `rel_pos` (int, 0-based bp offset within window), `edit_type`, `ref_bases`, `alt_bases`. Conversion from `EditSpec`: `rel_pos = pos - 1 - window_start_bp` (1-based VCF → 0-based). [VERIFIED: `spec.py:155-186`, `spec.py:189-223`]
- **ActionEncoder defaults:** `d_action=512`, `d_pos=128`, `d_type=64`, `d_seq=256`, `max_window_bp=12_288`. [VERIFIED: `encoder.py:96-102`]. Config pins `d_action=64`. **Discrepancy — see Reconcile.**
- **Positional encoding:** Custom sinusoidal at base-pair resolution. Formula: for position `p` and half-dim `h = d_pos // 2`, scale `i` in `[0, h)`: `scale_i = exp(i * (-log(max_bp) / max(h-1, 1)))`, then `sin(p * scale_i)` concat `cos(p * scale_i)`. [VERIFIED: `encoder.py:190-202`]. Note: this is a frequency schedule using `max_bp` as the base, not the standard `10000` of vanilla transformers.
- **Edit-type embedding:** `nn.Embedding(len(EditType)=6, d_type=64)`. [VERIFIED: `encoder.py:115`]
- **SeqMicroEncoder:** Shared for ref and alt paths. Token embedding: `nn.Embedding(vocab_size=4^6+1=4097, token_dim=min(128, d_seq))`. 2-layer `TransformerEncoder` with `d_model=d_seq=256`, `nhead=4`, `dim_feedforward=d_seq`, GELU, `batch_first=True`, `dropout=0.0`. Mean-pooled over tokens. [VERIFIED: `encoder.py:63-89`]
- **Short DNA tokenization (SeqMicroEncoder input):** Sequence is chunked into `_SEQ_TOKENS=4` 6-mer slots; each 6-mer → integer ID via base-4 encoding (`A=0,C=1,G=2,T=3`); trailing incomplete k-mers are `A`-padded to 6 bases; empty slots get `_OOV_TOKEN_ID=4096`. This results in a 4-slot integer sequence per ref or alt string. [VERIFIED: `encoder.py:56-62`, `encoder.py:204-217`]
- **Projection MLP:** `Linear(projection_in=d_pos+d_type+2*d_seq, 1024) → GELU → LayerNorm(1024) → Linear(1024, d_action)`. Input dim: `128 + 64 + 2×256 = 704`. Output is NOT L2-normalized. [VERIFIED: `encoder.py:117-123`]
- **Padding embedding:** Learned `nn.Parameter(zeros(d_action))` for right-padding shorter sequences in a batch. [VERIFIED: `encoder.py:124`]
- **Multi-edit ordering:** Batch sorted ascending by `rel_pos`; applied right-to-left (descending `rel_pos`) via `apply_edits`. [VERIFIED: `apply.py:118`]
- **Overlap check:** Raises `OverlappingEditsError` if `[e1.rel_pos, e1.rel_pos+len(ref1)) ∩ [e2.rel_pos, e2.rel_pos+len(ref2)) ≠ ∅`. [VERIFIED: `apply.py:134-151`]
- **apply_edit semantics:** `edited = window[:rel_pos] + alt_bases + window[rel_pos+len(ref_bases):]`. Reference bases verified case-insensitively before substitution. `preserve_length=True` truncates/pads on the far side from the edit locus; padding char is `N`. [VERIFIED: `apply.py:40-85`]
- **Synthetic samplers:**
  - `uniform_snv(window, n, rng, edge_margin=64)`: uniform position in `[edge_margin, len-edge_margin-1]`; alt uniform from 3 non-ref bases; retries up to 10× for N positions. [VERIFIED: `synthetic.py:127-168`]
  - `indel(window, n, rng, length_dist=None, type_mix=(0.5,0.5), edge_margin=64)`: truncated geometric over `[1, V1_MAX_LEN-1=15]` with `p=0.5`; 50/50 INS/DEL; if deletion would cross right margin, converts to INS; max `n*16+16` attempts before error. [VERIFIED: `synthetic.py:171-270`]
  - `mnv(window, n, rng, length_dist=None, edge_margin=64)`: uniform length over `[2, 8]` by default; every base perturbed to a non-self draw. [VERIFIED: `synthetic.py:273-318`]
- **Edge margin (synthetic samplers):** `DEFAULT_EDGE_MARGIN = 64` bp. [VERIFIED: `synthetic.py:34`]

### Equations & notation

**State encoder pipeline:**
```
w_t  ∈ {A,C,G,T,N}^{12288}       # DNA window, 12288 bp
T(w_t) = <dna>w_t</dna>           # wrapped tokenizer input
H = Carbon(T(w_t))[L]             # H ∈ R^{2048 × 1024}, layer L=20 hidden states
c = floor(locus_bp / 6)           # edit locus → token index (6 bp per token)
s_t = mean(H[max(0,c-r) : c+r+1]) # centered mean, r=pool_radius tokens
s_t = s_t / ||s_t||_2             # L2 normalize (normalize=true)
s_t ∈ R^{d_state},  d_state=1024
```

**Centered-mean pooling (inclusive):**
```
center_token c = floor(edit_locus_bp / token_bp)
start = max(0, c - pool_radius)
end   = min(|H|, c + pool_radius + 1)
s_t   = (1/(end-start)) * sum_{i=start}^{end-1} H[i]
```

**Action encoding pipeline:**
```
# Positional embedding (sinusoidal, h = d_pos/2 = 64)
scale_i = exp(i * (- log(max_bp) / max(h-1, 1)))   for i in [0, h)
p_emb = [sin(pos * scale_i) for i] ++ [cos(pos * scale_i) for i]
p_emb ∈ R^{128}

# Type embedding
t_emb = E_type[edit_type],  E_type ∈ R^{6 × 64}
t_emb ∈ R^{64}

# SeqMicroEncoder (shared for ref and alt)
tokens_ref = 6-mer tokenize(ref_bases), padded to 4 slots ∈ Z^4
r_emb = mean_pool(TransformerEncoder_2L(token_embed(tokens_ref))) ∈ R^{256}
v_emb = mean_pool(TransformerEncoder_2L(token_embed(tokens_alt))) ∈ R^{256}

# Concatenation + MLP projection
concat = [p_emb; t_emb; r_emb; v_emb] ∈ R^{704}
a_emb = Linear(1024, d_action)(GELU(LayerNorm(Linear(704, 1024)(concat))))
a_emb ∈ R^{d_action},  d_action=64  (first experiment)
```

**6-mer ID:**
```
kmer_id("b1 b2 b3 b4 b5 b6") = sum_k base4(b_k) * 4^(5-k)
where base4: A->0, C->1, G->2, T->3
OOV = 4^6 = 4096
```

**Edit application:**
```
apply_edit(w, e) = w[0:e.rel_pos] + e.alt_bases + w[e.rel_pos + len(e.ref_bases):]
```

**Overlap predicate:**
```
overlap(e1, e2) ⟺ max(e1.rel_pos, e2.rel_pos) < min(e1.rel_pos+|e1.ref|, e2.rel_pos+|e2.ref|)
```

### Reconcile

| Parameter | RFC-0002 / RFC-0003 spec value | Code / Config value | Source |
|---|---|---|---|
| `state_layer` default | `-1` (final layer) | `20` in production config | RFC-0002 §3.3 vs `train-carbon-500m-snv.yaml:24` |
| `pool_radius` default | `256` tokens (RFC §3.4 text, ±1,536 bp) | `DEFAULT_POOL_RADIUS_TOKENS = 256` in `pooling.py:37`, but config pins `pool_radius: 8` (±48 bp) | RFC-0002 §3.4 vs `train-carbon-500m-snv.yaml:26` and `pooling.py:37` |
| `d_action` default | `512` (RFC-0003 §3.4) | `64` in production config and in `ActionEncoder.__init__` default in code | RFC-0003 §3.4, `encoder.py:97` vs `train-carbon-500m-snv.yaml:44` |
| Action concat dim | RFC: `concat ∈ R^{704}`, MLP `704→1024→512` | Code: `projection_in = 128+64+512 = 704`, MLP `704→1024→d_action(=512 default, 64 in config)` | RFC-0003 §3.4 vs `encoder.py:117-123` — math consistent |
| L2 normalization site | RFC: encoder normalizes before predictor | `CarbonStateEncoder` stores `self.normalize` but `encode_batch` returns raw pooled tuples | RFC-0002 §3.5 vs `carbon.py:127-179` |
| `pool_type=attention` | RFC lists as supported (deferred) | `pooling.py` raises `InputError`; `carbon.py:68-74` only accepts `centered_mean`/`global_mean` | RFC-0002 §3.4 vs `pooling.py:222-226` |
| `state_layer` supported values | RFC: `{-1, -2, -3, -4}` | Code: any integer; `state_layer=20` in config | RFC-0002 §3.3 vs `carbon.py:63-67` |
| Synthetic MNV length dist | RFC: "uniform over [2, 8]" | Code: `range(2,9)` = [2..8] inclusive. Consistent. | RFC-0003 §3.8 vs `synthetic.py:293` |
| Synthetic indel length dist | RFC: "truncated geometric over [1, 16]" | Code: truncated geometric `p=0.5`, event length clipped to `V1_MAX_LEN-1=15` | RFC-0003 §3.8 vs `synthetic.py:66-85` |
| `trust_remote_code` default | RFC does not mention it | Code default `False`; config sets `true` | `carbon.py:51` vs `train-carbon-500m-snv.yaml:30` |

### Open questions / caveats

1. **L2 normalization not found in `carbon.py`:** `normalize=True` flag stored and validated, but the `F.normalize` call does not appear inside `CarbonStateEncoder.encode_batch`. Either happens in an outer training loop, or is an unimplemented stub.
2. **state_layer=20 vs. 24-layer Carbon-500M:** Layer index 20 (0-based) would be the 21st layer, plausibly penultimate. Total layer count not confirmed in read files; verify `model.config.num_hidden_layers`.
3. **pool_radius=8 tokens = ±48 bp:** Dramatically tighter than RFC default ±256 tokens (±1,536 bp). The first-experiment config is an ablation point, not the RFC default. Any paper table must distinguish.
4. **SeqMicroEncoder shares weights for ref and alt** but the RFC diagram is slightly ambiguous; code confirms sharing.
5. **Sinusoidal position frequency base:** uses `max_bp` (12,288) as the period base rather than 10,000. Non-standard; extrapolation depends on this formula.
6. **SV type never constructible via `EditSpec`:** `EditType.SV=5` is in the enum but `EditSpec.__post_init__` raises `UnsupportedEditError` first. The extra embedding row is dead in v1.
7. **RFC-0002 "implementation status: Partial":** clean-machine validation against pinned Carbon weights and full cache-build throughput evidence explicitly open.
8. **`max_len: 16` in action config** matches `V1_MAX_LEN=16` (per-sequence max ref/alt length), not the multi-edit haplotype length.
9. **`ActionEncoder.forward` signature** in code more permissive than RFC (accepts `Sequence[RelEdit] | Sequence[Sequence[RelEdit]]`).

---

## EVIDENCE CARD: R2-predictor-training

### Summary

GenoLeWM's PREDICTOR subsystem is a cross-attention Transformer that maps a frozen Carbon-500M state embedding `s_t` and one or more action embeddings `a_emb` to a predicted next-state embedding `ŝ_{t+1}` in the same 1024-dimensional latent space. The architecture alternates state-cross-action and action-cross-state attention blocks (`n_cross_layers`), then applies self-attention blocks (`n_self_layers`) over the fused sequence, and finally reads out a per-step prediction through a 2-layer output MLP followed by L2 normalization. The TRAINING subsystem is a Phase-1-only frozen-encoder loop: only the prediction loss `L_pred = alpha*(1-cos) + beta*||delta||^2/d` is active; the LeJEPA Gaussian KL regularizer is computed for monitoring but is not added to the gradient until Phase 2. Optimization uses AdamW with beta2=0.95 and a WSD schedule; collapse is monitored every 500 steps via seven scalar diagnostics. The first-experiment paper run (config `train-carbon-500m-snv.yaml`) is configured for `max_steps=20000`, `seed=104729`, `batch_size=8` (physical), `warmup_steps=1000`, and `d_state=1024`.

### Key Facts

#### Architecture
- **Architecture type:** Cross-attention Transformer with alternating `_StateToActionCrossBlock` and `_ActionToStateCrossBlock`. [VERIFIED] `predictor/model.py:115-128`
- **Default hidden dimension (`d_hidden`):** 768. [VERIFIED] `predictor/model.py:46,83`
- **State dimension (`d_state`):** 1024, matching Carbon-500M `hidden_size`. [VERIFIED] `configs/first_experiment/train-carbon-500m-snv.yaml:39`
- **Default action dimension (`d_action`):** 64 (first-experiment config). [VERIFIED] `yaml:41`; `schema.py:76`
- **Attention heads (`n_heads`):** 8. [VERIFIED] `model.py:87`; config `yaml:37`
- **FFN intermediate dimension (`ffn_dim`):** Default 768 (equals `d_hidden`). NOT 2048 as stated in the RFC. [VERIFIED] `model.py:49,88`
- **FFN activation:** GELU. [VERIFIED] `model.py:700-710`
- **Normalization:** Pre-LayerNorm. [VERIFIED] `model.py:505-533`
- **Output MLP structure:** `Linear(768,768) → GELU → LayerNorm(768) → Linear(768,1024)`. [VERIFIED] `model.py:133-138` (extra LayerNorm not in RFC)
- **Output normalization:** L2-normalized via `F.normalize(base + delta, p=2, dim=-1, eps=1e-12)`. Output is `state + MLP_delta` then normalized. [VERIFIED] `model.py:396-406`
- **Identity-at-init:** final `nn.Linear(d_hidden, d_state)` weight and bias zero-initialized. [VERIFIED] `model.py:160-163`
- **Other init:** Truncated normal `std=sqrt(2/fan_in)` for linear/attn; LayerNorm weight=1 bias=0; embeddings normal std=0.02. [VERIFIED] `model.py:141-163`
- **Token-type embedding:** `nn.Embedding(2, d_hidden)` (state=0, action=1). [VERIFIED] `model.py:113`
- **Step-position embedding:** `nn.Embedding(max_actions+1, d_hidden)`, max_actions=16 default. [VERIFIED] `model.py:114`; config `yaml:45`
- **State projection:** `nn.Identity()` when `d_state==d_hidden`; else `nn.Linear(d_state, d_hidden)`. Since 1024≠768 in paper run, a learned linear projection is used. [VERIFIED] `model.py:109-111`
- **Action projection:** `nn.Linear(d_action, d_hidden)` = `Linear(64, 768)`. [VERIFIED] `model.py:112`
- **Causal mask:** `_ActionToStateCrossBlock` causal cross-mask (each action attends to state + earlier actions); `_StateToActionCrossBlock` full attention. [VERIFIED] `model.py:597-605, 711-716`
- **Dropout:** 0.0 in all attention. [VERIFIED] `model.py:508-511, 571-573, 655-659`
- **Parameter budget:** RFC §3.1.3 gives ~40M for `d_hidden=1024`, ~22M for `d_hidden=768`. Code default `d_hidden=768`, `ffn_dim=768` (not 2048), so actual count lower than ~22M. [SPEC]

#### Autoregressive Rollout (ARPredictor)
- **ARPredictor wraps a base Predictor** and unrolls step-by-step. [VERIFIED] `predictor/ar.py:45-195`
- **KV-cache optimization:** action tokens encoded once before loop and cached attention projections reused. [VERIFIED] `ar.py:88-142`
- **`rollout_tensor` returns shape `(B, K, d_state)`.** [VERIFIED] `ar.py:65-82`
- **Upcast to fp32 when K > 20:** `upcast_output_mlp=True` for `actions.shape[1] > 20`. [VERIFIED] `model.py:181`, `ar.py:93`

#### Loss
- **Prediction loss formula:** `L_pred = alpha*(1 - cos(hat_s, s)) + beta*sum((hat_s - s)^2)/d_state`. [VERIFIED] `predictor/losses.py:60-62`
- **Default alpha:** 1.0. [VERIFIED] `losses.py:48`
- **Default beta:** 0.1. [VERIFIED] `losses.py:49`
- **d_state for normalization:** uses `prediction.shape[-1]` dynamically (=1024). [VERIFIED] `losses.py:61`
- **Per-step averaging:** masked uniform-weight mean over valid `(batch, step)`. [VERIFIED] `losses.py:63, 175-186`
- **Phase-conditional total loss:** Phase 1 `L = L_pred`; Phase 2 `L = L_pred + gamma*kl_reg`. [VERIFIED] `losses.py:125`
- **Default gamma:** 0.5. [VERIFIED] `losses.py:111`
- **KL regularizer in Phase 1:** computed from target states for monitoring only. [VERIFIED] `losses.py:123-125`; `trainer.py:179-181`
- **KL formula:** `eigvalsh` of empirical covariance in float64; `KL = 0.5*(||mu||^2 + tr(Sigma) - logdet(Sigma + eps*I) - d)`. [VERIFIED] `losses.py:87-101`
- **KL stabilizer eps:** 1e-6. [VERIFIED] `losses.py:67,115`

#### Optimizer and Schedule
- **Optimizer:** AdamW. [VERIFIED] `trainer.py:344`
- **beta1:** 0.9. [VERIFIED] `yaml:58`; `schema.py:109`
- **beta2:** 0.95. [VERIFIED] `yaml:59`; `schema.py:110`; `trainer.py:346`
- **eps:** 1e-8 hardcoded. [VERIFIED] `trainer.py:347`
- **weight_decay:** 0.1. [VERIFIED] `yaml:60`; `schema.py:111`
- **grad_clip:** 1.0. [VERIFIED] `yaml:61`; `trainer.py:201-203`
- **Parameter groups:** Two groups (decay / no_decay), same lr & wd. RFC's fine-grained per-component LR groups NOT implemented. [VERIFIED] `trainer.py:422-470`
- **WSD schedule:** warmup linear; stable to 80% of post-warmup; decay peak→0.1*peak over 80%→98%; taper to 0.01*peak over 98%→100%. [VERIFIED] `trainer.py:351-391`
- **Warmup steps (config):** 1000. [VERIFIED] `yaml:64`. RFC: 2000. [SPEC]

#### Training Config (first experiment)
- **max_steps:** 20000. [VERIFIED] `yaml:52`
- **seed:** 104729. [VERIFIED] `yaml:14`
- **Seeds:** `data_seed=seed`, `predictor_seed=seed+1`, `lora_seed=seed+2`. [VERIFIED] `trainer.py:67-70`
- **phase:** phase1 (frozen encoder). [VERIFIED] `yaml:15`
- **batch_size (config):** 8 physical. [VERIFIED] `yaml:71`. RFC: 256 effective (microbatch=16, accum=16). [SPEC]
- **Collapse monitoring interval:** every 500 steps. [VERIFIED] `yaml:53`; `collapse.py:94`
- **Deterministic mode:** true. [VERIFIED] `yaml:16`
- **Device:** cuda. [VERIFIED] `yaml:93`
- **dtype:** bf16 throughout. [VERIFIED] `yaml:22,41`

#### Collapse Monitoring Metrics
Seven metrics every 500 steps: `pred_cos_mean`, `pred_l2_mean`, `target_var_per_dim`, `pred_var_per_dim`, `pred_target_corr`, `pairwise_pred_dist_mean`, `kl_reg`. [VERIFIED] `collapse.py:37-46`
Alert thresholds: `pred_var_per_dim < 0.5*target_var_per_dim`; `pairwise_pred_dist_mean < 0.5*initial`; `kl_reg > 10`. [VERIFIED] `collapse.py:49-59, 193-217`

#### Edit-Balanced Sampling
Default weights: SNV=0.40, INS=0.20, DEL=0.20, MNV=0.10, INDEL=0.10. [VERIFIED] `sampling.py:78-84`
Default rollout step mix: K=1 p=0.90, K=2 p=0.05, K=3 p=0.05. [VERIFIED] `sampling.py:86-90`

### Equations and Notation

```
L_pred(hat_s, s) = alpha*(1 - cos(hat_s, s)) + beta*sum_j((hat_s_j - s_j)^2)/d_state
cos(u,v) = (u.v)/(||u||*||v||);  d_state=1024; alpha=1.0; beta=0.1
L_pred,total = (1/K) * sum_{k=1}^{K} L_pred(hat_s_{t+k}, s_{t+k})   # uniform, masked-mean
Phase 1: L = L_pred,total ;  Phase 2: L = L_pred,total + gamma*L_reg ;  gamma=0.5
L_reg = 0.5*(||mu_batch||^2 + tr(Sigma_batch) - logdet(Sigma_batch + eps*I) - d) ;  eps=1e-6 float64 eigvalsh
hat_s_{t+k} = normalize(s_t + MLP(action_output_k))   # MLP: Linear→GELU→LayerNorm→Linear (final zero-init)
WSD: warmup t/warmup ; stable 1.0 to ~80% ; decay 1.0→0.1 ; taper 0.1→0.01
```

### Reconcile

| Item | RFC / Spec | Config / Code | Source |
|------|-----------|---------------|--------|
| FFN intermediate dim | 2048 (RFC-0004 §3.1) | **768** (=d_hidden) | RFC vs `model.py:49,88` |
| Parameter budget | ~40M (1024), ~22M (768 w/ 2048 FFN) | lower than ~22M (ffn=768) | RFC vs code |
| n_layers interpretation | 4 cross + 2 self = 6 | config `n_layers:6` → `n_cross_layers=6`, plus hardcoded `n_self_layers=2` ⇒ **6 cross + 2 self = 8** | `predictor/__init__.py:40`; `model.py:85-86` |
| Batch size | 256 effective (16×16) | `batch_size:8` physical, no accum | RFC vs `yaml:71` |
| Warmup steps | 2000 | 1000 | RFC vs `yaml:64`; `schema.py:113` |
| weight_decay | 0.05 | **0.1** | RFC vs `yaml:60` |
| d_state schema default | 1024 | schema default **512**; config overrides to 1024 | `schema.py:75` |
| Per-component LR groups | separate (predictor 3e-4, LoRA 1e-5, ...) | two groups only, same LR; LoRA not implemented | RFC vs `trainer.py:422-470` |
| KL regularizer input Phase 1 | "encoder outputs over current batch" | computed on target states (correct, deterministic) | `losses.py:123` |
| Output MLP LayerNorm | "1024→1024→1024" no LayerNorm | `Linear→GELU→LayerNorm→Linear` | RFC vs `model.py:133-138` |

### Open Questions / Caveats
1. **n_layers=6 means 6 cross + 2 self (not 4+2).** Significant architectural discrepancy; RFC says 4+2, first-experiment config instantiates 6+2.
2. **Effective batch size unknown.** `batch_size:8`, no accum config. RFC targets 256 effective.
3. **ffn_dim=768 (not 2048).** 2× expansion in RFC not reflected in code.
4. **~22M parameter count overstated.** With ffn=768, actual ~15-17M (not counted in code).
5. **Phase-1 KL on target states is numerically valid** but cannot drift (encoder frozen) — correct.
6. **WSD decay boundaries fractional-step**; correct for max_steps=20000.
7. **Attention impl:** `F.scaled_dot_product_attention` (cached path) + `nn.MultiheadAttention` (standard); no RoPE.
8. **d_action=64 vs RFC table 512→1024**; first experiment SNV-only `Linear(64,768)`.

---

## EVIDENCE CARD: R3-data-eval-protocol

### Summary

GenoLeWM is a systems-and-reproducibility research project that trains and evaluates an action-conditioned latent world model (LeWM) over genomic edits. The data pipeline (RFC-0006) feeds a frozen Carbon-500M encoder with windows drawn from `HuggingFaceBio/carbon-pretraining-corpus`, applies a 4-source edit mix (gnomAD/synthetic SNV/synthetic indel/ClinVar P/LP), and emits `(w_ref, action, w_alt)` training tuples. The evaluation suite (RFC-0007) covers three tracks: binary VEP classification on ClinVar coding/noncoding (AUROC, AP, balanced accuracy), continuous Spearman correlation on BRCA2 saturation mutagenesis and TraitGym Mendelian benchmarks, and latent rollout fidelity (cosine similarity, L2 distance, Recall@k against a source-state naive baseline). Baselines are Carbon-500M zero-shot log-likelihood ratio and — for rollout — predicting no-change (`s_t` unchanged). The canonical v0.2.1-r1 results are uniformly mixed-to-negative. The paper is explicitly framed as a negative-results and systems/reproducibility contribution.

### Key Facts

#### Training Corpus
- **Primary dataset:** `HuggingFaceBio/carbon-pretraining-corpus`; ~180M sequences, predominantly eukaryotic. HF revision `cb4c13a78102933b3a6ac65734d326f7b431d9b7`. [VERIFIED: RFC-0006 §3.1; `dataset-snapshot-snv.json:62`]
- **Phase 1 subset fraction:** 10% (`DEFAULT_PHASE1_SUBSET_FRACTION = 0.10`), ~18M sequences. [VERIFIED: `corpus.py:40`]
- **Sub-mix:** eukaryotic_genes 50%, mrna 25%, splice_mrna 10%, gtdb 15%. [VERIFIED: `corpus.py:53-59` `CARBON_SUBMIX`]
- **Subset selection:** SHA-256 stable-hash of `"{seed}:{record_id}"`, fraction 0.10, seed 0. [VERIFIED: `corpus.py stable_subset_includes`]

#### Window Sampling
- **Window size:** `DEFAULT_WINDOW_BP = 12,288` bp; `SUPPORTED_WINDOW_BP = (4096, 12288, 24576)`. [VERIFIED: `windowing.py:37-38`]
  - **Exception:** placed-window training windows in the dataset snapshot use `window_bp = 4096`. [VERIFIED: `dataset-snapshot-snv.json placed_windows.window_bp=4096`]
- **Margin:** `DEFAULT_CORPUS_MARGIN_BP = 256` bp. [VERIFIED: `corpus.py:41`]
- **Stride:** `DEFAULT_CORPUS_STRIDE_BP = 8,192` bp → 67% overlap. [VERIFIED: `corpus.py:42`]
- **Skip:** sequences < `window_bp + 2×margin = 12,800` bp skipped. [VERIFIED: `corpus.py iter_window_starts`]
- **Edits per window:** `N_edits = 8`. [VERIFIED: `builder.py DEFAULT_EDIT_SOURCE_COUNTS` 3+3+1+1=8]

#### Edit Sources
- **Default per-window allocation (code):** gnomad_common 3 (37.5%), synthetic_snv 3 (37.5%), synthetic_indel 1 (12.5%), clinvar 1 (12.5%). [VERIFIED: `builder.py:77-83`]. RFC-0006 §3.3 states 40/30/20/10 — **not implemented; code uses 3/3/1/1**.
- **gnomAD:** v4.1, PASS-filtered, `min_af=0.01`, `max_allele_len=16`. Output `DIR/gnomad/v4.1/variants.parquet`. [VERIFIED: `gnomad.py:30,117-127`]
- **gnomAD populations:** afr, ami, amr, asj, eas, fin, nfe, oth, sas. [VERIFIED: `gnomad.py:31-41`]
- **ClinVar:** pinned 2026-04-15; monthly NCBI VCF → Parquet. [VERIFIED: `clinvar.py`; `dataset-snapshot-snv.json:14`]
- **ClinVar label mapping:** CLNSIG → {P, LP, B, LB, VUS, OTHER}. VUS/OTHER excluded; P/LP positive; B/LB negative. [VERIFIED: `clinvar.py:201-215`; `evaluation.py:39-40`]
- **Allele length cap:** `max_allele_len=16`, ACGT-only. [VERIFIED: `_vcf.py:131-133`]
- **Fallback:** insufficient ClinVar/gnomAD → `synthetic_snv`. [VERIFIED: `builder.py:85-95`]
- **Synthetic indels:** 50% INS / 50% DEL, `geometric(p=0.5)` truncated `[1,16]`. [SPEC: RFC-0006 §3.4]

#### Tuple Builder
- **`TrainingTuple`:** `window_id` (SHA-256 hex of ref window), `source_record_id`, `edit_source`, `rel_edits`, `target_window`, `window_start_bp`, `window_end_bp`. [VERIFIED: `builder.py:224-253`]
- **`GenoLeWMDataset`:** `IterableDataset`; deterministic per worker `seed + worker_id`. [VERIFIED: `builder.py:283-341`]
- **Target encoding:** `apply_edit(window.sequence, edit, preserve_length=True)`. [VERIFIED: `builder.py:539-555`]

#### Holdout Policy
Four holdout types in `HoldoutPolicy`: `holdout_chroms`, `intervals`, `edit_keys` (`chrom:pos:ref:alt`), `record_ids`. RFC-0006 §3.8: `holdout-chr` (chr21), `holdout-clinvar` (ClinVar P/LP in eval), `holdout-haplotypes` (gnomAD haplotype blocks ≥2 variants in 1 kbp). [VERIFIED: `builder.py:161-221`; SPEC RFC-0006 §3.8]

#### VCF → Parquet Builders
- **gnomAD:** `prepare_gnomad_shard(...)` filters PASS, ACGT ≤16 bp, AF_global ≥0.01; PyArrow 100k-row batches; atomic write. Schema: chrom, pos, ref, alt, af_global, af_{pop}×9, filter, schema_version="1.0.0". [VERIFIED: `gnomad.py`]
- **ClinVar:** `prepare_clinvar_shard(...)` no AF filter; allele filter; CLNSIG → {P,LP,B,LB,VUS,OTHER}. Schema: chrom, pos, ref, alt, clinical_significance, review_status, gene_symbol, clinvar_id, schema_version="1.0.0". [VERIFIED: `clinvar.py`]

#### Evaluation — VEP Track
- **AUROC:** `(W_+ - n_+(n_++1)/2)/(n_+·n_-)` via average-rank ties. [VERIFIED: `evaluation.py:1178-1195`]
- **Average Precision:** `AP = sum_k Prec(k)·1[label_k=1]/n_+`. [VERIFIED: `evaluation.py:1198-1208`]
- **Balanced Accuracy:** `BA = (TPR+TNR)/2` at `DEFAULT_EVAL_THRESHOLD=0.5`. [VERIFIED: `evaluation.py:828-862`]
- **Confidence Intervals:** stratified bootstrap 1,000 resamples, seed 0, 95% CI. [VERIFIED: `evaluation.py:865-896`]
- **Score Field:** `DEFAULT_EVAL_SCORE_FIELD = "sigma_calibrated"`. [VERIFIED: `evaluation.py:34`]

#### Evaluation — Continuous Track
- **Spearman rho:** `Pearson(rank(y_true), rank(y_score))`, average-rank ties. Label field `functional_score` (BRCA2, TraitGym). [VERIFIED: `evaluation.py:1211-1218`]

#### Evaluation — Rollout Fidelity Track
Three metrics over n instances (each = one `(source_window, edit_list)` pair): cosine_similarity_mean, l2_distance_mean, recall_at_k. `ŝ_{t+K}` = predictor final state; `s*_{t+K}` = encoder of fully edited window; rank = position of `s*` among K-nearest cached refs. Default `DEFAULT_RECALL_K = 10`. [VERIFIED: `cli/rollout.py:26, 383-394`]
**Source-state baseline:** `ŝ_{t+K} = s_t` (no-change); naive cosine `cos(s_t, enc(apply_all(w, edits)))`. [VERIFIED: `cli/rollout.py:386-393`]
**DISCREPANCY:** README reports "Recall@4"; template + CLI default `recall_k=10`. Canonical recall_at_k=1.0 trivially holds for any small k at N=8.

#### Carbon Zero-Shot Baseline
For each variant, 12,288 bp reference window centered on variant. `score = logP_ref - logP_alt` (higher → more pathogenic). Standard AR LM log-likelihood (attention-mask weighted). [VERIFIED: `carbon_zero_shot.py:316-328, 442-460`]
**Model:** `HuggingFaceBio/Carbon-500M`, `AutoModelForCausalLM`, bf16. [VERIFIED: `carbon_zero_shot.py:180-221`; template `carbon_model_dir="carbon/500m"`]
**Caching:** SHA-256-keyed JSONL by `(carbon_model, carbon_revision, sequence_sha256)`. [VERIFIED]

#### Sample Sizes (N per Eval Slice)
| Slice | N |
|---|---|
| clinvar_coding | **16** |
| clinvar_noncoding | **16** |
| brca2_saturation | **32** |
| traitgym_mendelian | **32** |
| rollout_phased_haplotypes | **8** |
| rollout_synthetic_edit_chains | **8** |
[VERIFIED: `docs/release/huggingface-model-card.md:231-240`]
Step size 1/16 = 0.0625 confirms N=16 for ClinVar. All AUROC/AP/balanced accuracy heavily quantized; bootstrap CIs statistically meaningless.

### Equations & Notation
| Symbol | Definition |
|---|---|
| `w_ref` | Reference window, L=12,288 bp |
| `a_v` | Action encoding of edit `v=(chrom,pos,ref,alt)` |
| `w_alt` | `apply_edit(w_ref, rel_edit, preserve_length=True)` |
| `s_t` | `enc(w_ref)`; encoder = Carbon-500M |
| `s*_{t+1}` | `enc(w_alt)` |
| `ŝ_{t+1}` | `g(s_t, a_v)` |
| `surprise(v)` | `||ŝ_{t+1} - s*_{t+1}||_2` |
| `displacement(v)` | `1 - cos(ŝ_{t+1}, s_t)` |
| `score_carbon(v)` | `logP_Carbon(w_ref) - logP_Carbon(w_alt)` |
| `rho` | Spearman rank corr (scores vs functional labels) |
| `k` | Recall@k; default 10; config 10; README says 4 |
| `n_+, n_-` | # positive (P/LP), # negative (B/LB) |

### Reconcile
1. **Edit-source mix:** RFC 40/30/20/10 vs code 3/3/1/1 (37.5/37.5/12.5/12.5).
2. **Placed-window window_bp:** RFC 12,288 vs snapshot 4,096 (chr22).
3. **Recall@k label:** README "Recall@4" vs config/CLI `recall_k=10`; both give 1.0 at N=8.
4. **ClinVar eval size:** RFC ~50k coding / ~30k noncoding vs actual N=16.
5. **Carbon tokenizer wrapping:** `wrap_dna_for_tokenizer`; shifted-label LM accumulation, mask-weighted.

### Open Questions / Caveats
1. **Extreme under-powering:** ClinVar N=16; fully quantized 1/16; bootstrap CIs unreliable.
2. **Single chromosome training:** placed windows only chr22.
3. **Holdout chr21 not confirmed in first-experiment config.**
4. **Carbon zero-shot window-centering edge behavior** relies on full `windowing.py` (not fully traced).
5. **Recall@k discrepancy unresolved**; trivially 1.0 at N=8.
6. **Rollout fidelity vs actual haplotypes** unclear (N=8).
7. **Throughput single-run, no warmup** (not a serving benchmark).
8. **Multi-edit training tuple coverage** for v0.2.1 run not confirmed.
9. **Encoder caching status "Partial"** in RFC-0006.
10. **Carbon-3B / Evo2-7B** planned baselines; only Carbon-500M measured.

---

## EVIDENCE CARD: R4-surprise-planning-rollout

### Summary

GenoLeWM implements three interlocking subsystems: the **surprise scorer** (RFC-0009) converts predictor residual L2 error into a calibrated, context-stratified percentile score (unsupervised pathogenicity signal). The **CEM planner** (RFC-0008) performs model-predictive control in latent space — iteratively sampling, evaluating, refitting a factored proposal distribution over discrete edit sequences, never calling Carbon during search. The **AR rollout wrapper** (`ARPredictor`, RFC-0004) pre-encodes static action projections and reuses them across the autoregressive loop. v0.2.1-r1: AR cache 2.41x at K=5 (passing local 2x), 2.47x at K=20 (failing RFC-0004 5x; #42 open, rescoped). Planning demo ran 384 evaluations on a synthetic task, stopped on patience, `best_distance=23.66` on a non-learned proxy. Systems/reproducibility evidence, not useful-planning evidence.

### Key Facts

#### Surprise Scoring
- **Raw surprise:** `sigma_raw(v) = ||g(s_t, a_v) - enc(apply(v, w_ref))||_2`. [VERIFIED] `surprise/score.py:138-184`
- **Distance impl:** pure Python `sqrt(sum((a-b)^2))`. [VERIFIED] `score.py:614-615`
- **Multi-window aggregation:** mean (default)/max/median over non-overlapping windows. [VERIFIED] `score.py:591-611`
- **Context stratification:** `(region_class, gc_bin, repeat_class)` = 11×3×5 = 165 buckets; id `{region}|{gc}|{repeat}`. [VERIFIED] `context.py:34-68`
- **GC bins:** exact thirds `1/3`, `2/3` (not data-derived terciles). [VERIFIED] `context.py:64-68`
- **Backoff chain:** full → drop repeat → drop gc → `*`. Well-populated ≥1,000 rows; low-confidence <100. [VERIFIED] `context.py:70`; `calibration.py:41-48`
- **Calibration:** ≤10,000 gnomAD common variants/bucket; 1,001-point empirical CDF; values propagate up backoff chain. [VERIFIED] `calibration.py:41-45, 224-225`
- **Calibrated score:** `sigma_calibrated(v) = F_{bucket}(sigma_raw(v))` ∈ [0,1], linear interp. [VERIFIED] `calibration.py:347-359`
- **Confidence:** `min(n_calibration/1000, 1.0)`. [VERIFIED] `calibration.py:123-125`
- **Parquet schema:** bucket_id, n_calibration, cdf, sigma_grid, back_off_to, schema_version "1.0.0"; zstd-9. [VERIFIED] `calibration.py:38, 398-408`
- **Cosine raw-surprise variant:** Not implemented (L2 only in `score.py`); cosine exists only in planner. [VERIFIED]
- **Bayesian/MC-dropout variant:** future work. [SPEC] RFC-0009 §6
- **Output (JSONL):** sigma_raw, sigma_calibrated, bucket_id, confidence, low_confidence + VCF coords. [VERIFIED] `score.py:272-286`
- **`score_vcf` defaults:** aggregation="mean", batch_size=64. [VERIFIED] `score.py:228-237`

#### CEM Planner
- **Objective:** `argmin d(g^K(s_0, a_{1:K}), s_target) + lambda*c(a_{1:K})`, lambda=0 default. [VERIFIED] `cem.py:359`
- **CEM defaults:** horizon=5, n_iterations=5, n_samples=1024, n_elite=64, cost_weight=0.0, stopping_eps=0.05, patience=2, smoothing=0.1. [VERIFIED] `cem.py:48-88`
- **Total predictor calls (theoretical):** 5×1024 = 5,120 K-step rollouts → 25,600 calls if run to completion. [SPEC] RFC-0008 §3.4
- **Planning demo (published):** best_distance=23.656930390534644, n_evaluations=384, elapsed≈15.34s, stopped_reason=patience. NOT 5,120 — patience fired early on synthetic task. [SPEC]
- **Latent-only:** Carbon called once to encode initial/target; CEM inner loop only predictor. [VERIFIED] RFC-0008 §2
- **Refit:** `new_weight[t] = smoothing*prior_prob[t] + (1-smoothing)*mle[t]`, smoothing=0.1. [VERIFIED] `cem.py:429-478`
- **Stopping reasons:** max_iterations, distance_threshold (<0.05), patience (no improve for 2). [VERIFIED] `cem.py:229, 267-271`
- **Distance functions:** l2, cosine, region, projection. [VERIFIED] `cem.py:297-345`
- **Cost functions:** count_cost, bp_cost, weighted_type_cost (SNV1/INS2/DEL2/MNV2/INDEL3), custom. [VERIFIED] `costs.py:22-75`
- **ActionSampler:** factored type/position/bases; edge margin 64 bp. [VERIFIED] `sampling.py:22-66, 124-182`
- **Length dist default:** geometric `0.5^L`. [VERIFIED] `sampling.py:336-337`
- **SV excluded:** raises `InputError`. [VERIFIED] `sampling.py:409-413`
- **Hardware targets (bench):** H100 ≤1s; M3 Max ≤30s; not confirmed in artifacts. [VERIFIED code exists]

#### AR Rollout Speed
- **Benchmark:** `ARPredictor.rollout_tensor` (cached) vs `_naive_rollout_tensor` (loop). Speedup = naive.median_ns / cached.median_ns. [VERIFIED] `bench/rollout.py:163-175`
- **Timing:** perf_counter_ns, iters=30, warmup=5, median. [VERIFIED] `bench/rollout.py:46-49`
- **Default params:** batch_size=4, d_state=64, d_action=32, d_hidden=64, n_heads=4, n_cross_layers=2, n_self_layers=1, ffn_dim=128. **Toy synthetic dims.** [VERIFIED] `bench/rollout.py:34-51`
- **Horizons:** K=5, K=20. [VERIFIED]
- **RFC-0004 targets:** 5.0 if K≥20, 2.0 if K≥5. [VERIFIED] `bench/rollout.py:178-184`
- **Measured:** K=5 2.41386 (passes); K=20 2.47322 (fails 5.0; #42 open, rescoped). [SPEC]
- **RFC prose:** "~2× at K=5 and ~5× at K=20." [SPEC] RFC-0004 §3.3
- **Cache mechanism:** `_encode_rollout_actions` once, per-step `_forward_one_step_from_action_token`; falls back to full forward. [VERIFIED] `ar.py:88-170`
- **upcast_output_mlp:** K>20 forces fp32. [VERIFIED] `ar.py:93-94`

### Equations & Notation
```
sigma_raw(v) = || g(s_t, a_v) - enc(apply(v, w_ref)) ||_2
sigma_calibrated(v) = F_{bucket(v)}( sigma_raw(v) )   in [0, 1]
conf(v) = min( N_{bucket(v)} / 1000 , 1.0 )
bucket(v) = region_class(v) | gc_bin(v) | repeat_class(v)
a*_{1:K} = argmin_{a_{1:K}}  d( g^K(s_0, a_{1:K}), s_target ) + lambda*c(a_{1:K})
objective = distance + cost_weight*cost
new_weight[t] = smoothing*prior_prob[t] + (1-smoothing)*MLE[t]   # smoothing=0.1
speedup(K) = median_latency_naive(K) / median_latency_cached(K)
```

### Reconcile
- CEM defaults (horizon/iters/samples/elite/smoothing/cost_weight) all MATCH RFC-0008. [VERIFIED]
- `PlanningResult.n_predictor_calls` = alias for `n_evaluations`. [VERIFIED]
- **RFC-0008 §3.8 `PlanningConfig.distance` field:** RFC specifies it; code makes distance implicit in evaluator closure. **MISMATCH.** [VERIFIED]
- AR targets K=5(2.0)/K=20(5.0) MATCH; measured K=20 2.47x FAILS target.
- CDF grid 1,001 MATCH; reference per bucket 10,000 MATCH.
- Cosine surprise planner-only MATCH; Bayesian future work MATCH.
- `score_variant` API extends RFC (extra kwargs); backwards-compatible.

### Open Questions / Caveats
1. **K=20 AR speedup fails RFC-0004 (5x); measured 2.47x; #42 open.** Benchmark on toy model, not Carbon-scale.
2. **Planning demo uses synthetic proxy evaluator, not learned predictor.** `best_distance=23.66`, `n_evaluations=384` measure CEM on a convex proxy.
3. **Planning demo stopped at 384 evaluations (not 5,120)** — patience.
4. **ClinVar N≈16** quantized 1/16; AUROC 0.734375 with N≈16 has SE ≈ ±0.13; noncoding delta -0.1875 uninterpretable.
5. **Calibration table not validated on released model.** [SPEC RFC-0009 status]
6. **GC bins hard-coded thirds**, not data-derived.
7. **`PlanningConfig` lacks `distance` field** despite RFC §3.8.
8. **AR benchmark toy dims** — speedups not predictive at Carbon scale.
9. **`upcast_output_mlp` at K>20** silently forces fp32.

**Source files verified:** `rfcs/0009`, `rfcs/0008`, `rfcs/0004` (140-155); `surprise/score.py`, `surprise/context.py`, `surprise/calibration.py`; `planning/cem.py`, `planning/costs.py`, `planning/sampling.py`; `predictor/ar.py`; `bench/rollout.py`, `bench/planning.py`, `bench/_harness.py`.

---

## EVIDENCE CARD: R5-systems-provenance

### Summary

GenoLeWM's reproducibility/evidence-bound release subsystem is a multi-layer content-addressed provenance chain. A SHA-256-over-canonical-JSON scheme produces a `model_id` that is literally the hash of the manifest (which itself hashes every artifact file); every scoring call produces a cryptographic receipt binding model identity, input commitment, calibrated score, and runtime metadata. The on-device runtime enforces a fail-closed network guard patching five socket/HTTP entry points during inference. Personal-genome importers (23andMe, AncestryDNA, MyHeritage, Sequencing.com JSON) operate entirely locally. The release toolchain implements an evidence-bound paper path: a benchmark suite orchestrator verifies content-addressed output files per step, a readiness report gates all required benchmark rows (including mandatory negative findings), and `serious_completion_paper.py` re-renders the entire manuscript from machine-readable artifacts at verification time, rejecting stale/placeholder/mutated text. Negative findings are structurally auditable: the paper cannot pass verification unless the negative deltas are present in the artifacts that drive it.

### Key Facts

#### Checksum Receipt Schema
- Receipt schema version `"1.0.0"`. [VERIFIED `receipt.py:38`]
- Supported provenance kinds: `frozenset({"checksum_only"})`. [VERIFIED `receipt.py:41,101-108`]
- Required keys: schema_version, model_id, input_commitment, output, output_commitment, calibration_hash, runtime, timestamp, provenance. [VERIFIED `receipt.py:169-192`]
- Output sub-keys: sigma_raw, sigma_calibrated, bucket_id, confidence, low_confidence. [VERIFIED]
- Runtime sub-keys: backend, device, geno_lewm_version, carbon_revision. [VERIFIED]
- On-disk: canonical JSON, round-trip stable. [VERIFIED `receipt.py:161-168`]
- VCF batch receipts: JSONL sidecars (one v1 receipt per scored alt). [VERIFIED `deploy/runtime.py:228-231`]

#### Canonical JSON and Hashing
- Keys sorted, compact `(",", ":")`, UTF-8, NaN/Inf rejected, bytes rejected. [VERIFIED `hashing.py:73-94`]
- SHA-256 format `"sha256:<64-hex>"`. [VERIFIED `hashing.py:32-33,98-99`]
- File hashing streams 1 MiB chunks. [VERIFIED `hashing.py:34,107-121`]
- `looks_like_sha256` validator. [VERIFIED `hashing.py:124-130`]

#### Input Commitment
- `input_commitment = SHA-256(canonical_json({reference_window, edit_spec, pooling_config, dtype_config, version:1}))`. [VERIFIED `commitment.py:105-120`]
- edit_spec committed fields: chrom, pos, ref, alt, edit_type (int). [VERIFIED `commitment.py:71-80`]
- pooling_config: state_layer, pool_type, pool_radius, normalize. [VERIFIED]
- dtype_config: encoder_dtype, predictor_dtype. [VERIFIED]
- Privacy: reference window committed but not in receipt top-level JSON. [VERIFIED RFC-0011 §4]

#### Model Manifest
- Schema version "1.0.0". [VERIFIED `manifest.py:45`]
- `model_id = SHA-256(canonical_json(asdict(manifest)))`. [VERIFIED `manifest.py:143-145`]
- Required: schema_version, model_name, model_version, release_id, encoder, predictor, action_encoder, calibration, training, eval. [VERIFIED `manifest.py:104-129`]
- Strict validation; all artifact hashes `sha256:<64hex>`. [VERIFIED `manifest.py:48-53`]

#### Runtime Manifest Verification
- `_verify_manifest_artifacts` hashes predictor/action_encoder/calibration/eval/config_file. [VERIFIED `runtime.py:1065-1079`]
- `ManifestHashMismatchError` on mismatch. [VERIFIED `runtime.py:1091-1099`]
- Absolute paths and `..` rejected. [VERIFIED `runtime.py:1102-1109`]
- Verified at `GenoLeWMRuntime.__init__`. [VERIFIED `runtime.py:140-143`]

#### Fail-Closed Network Guard
- `unittest.mock.patch` ExitStack patches 5 targets: socket.create_connection, socket.socket.connect, urllib.request.urlopen, http.client.HTTP(S)Connection.connect. [VERIFIED `runtime.py:366-384`]
- Raises `NetworkCallProhibitedError`. [VERIFIED `runtime.py:369-373`]
- Applied to score_variant, score_vcf, encode_window, predict. [VERIFIED]
- Not applied to setup/update (weight download). [VERIFIED RFC-0010 §3.7]

#### Local-First Importers
- 23andMe, AncestryDNA, MyHeritage, Sequencing.com JSON; local-only. [VERIFIED]
- Array formats lack REF — require explicit `(chrom,pos)->ref` map, fail closed. [VERIFIED RFC-0010 §3.9]
- `_MISSING_GENOTYPES` skipped. [VERIFIED `_common.py:13`]
- Backend priority: CoreML → CUDA → ONNX → CPU. [VERIFIED `runtime.py:78-83`]

#### Redaction Filter (RFC-0013)
- DNA pattern `^[ACGTNacgtn]{20,}$` dropped regardless of key. [VERIFIED `_redaction.py:67`]
- Deny-list: vcf_content, genotype, sample_id, user_email, email, phone, address, dob, birthdate. [VERIFIED `_redaction.py:49-61`]
- Strict mode default `GENO_LEWM_REDACTION_STRICT=1` raises `InvariantViolation`. [VERIFIED]
- Single chokepoint at logger boundary. [VERIFIED `observability.py:590-596`]

#### Benchmark Suite and Readiness
- Step kinds: vep_score, carbon_baseline, vep_eval, rollout_state_examples, rollout_state_generation, rollout_eval, aggregate_eval, v02_readiness. [VERIFIED]
- VEP benchmarks (4): clinvar_coding, clinvar_noncoding, brca2_saturation, traitgym_mendelian. [VERIFIED]
- Rollout benchmarks (2): rollout_phased_haplotypes, rollout_synthetic_edit_chains. [VERIFIED]
- Required VEP metrics: auroc, average_precision, balanced_accuracy, accuracy (+spearman_rho for BRCA2/TraitGym). [VERIFIED]
- Required rollout metrics: cosine_similarity_mean, l2_distance_mean, recall_at_k. [VERIFIED]
- All 4 VEP require carbon_zero_shot baseline w/ CIs. [VERIFIED]
- Rollout speed required: k5_speedup, k20_speedup. [VERIFIED]
- `ok=true` gating: `"ok": execute and not failed`. [VERIFIED `v02_benchmark_suite.py:210`]
- Each step records `output_identities` (path, sha256, size_bytes). [VERIFIED]
- Duplicate output detection. [VERIFIED]
- Suite template declared outputs: 29 step-declared output files (per code logic).

#### Evidence-Bound Paper (serious_completion_paper.py)
- Paper kind `"serious_completion_v0.2"`. [VERIFIED `:40`]
- Default title `"GenoLeWM: Evidence-Bound Genomic Edit World Models, Benchmarks, and Negative Results"`. [VERIFIED `:41-43`]
- Re-render-to-verify: raises `serious_paper.stale` if `text != expected`. [VERIFIED `:900-916`]
- Stale eval report detection → InputError. [VERIFIED `:361-363`]
- Placeholder rejection `PLACEHOLDER_RE`: tbd|todo|placeholder|coming soon|fake|dummy|lorem ipsum|go here. [VERIFIED `:71-74,210`]
- `claim_boundary` enforcement: each artifact must contain one of clinical/privacy/runtime/model-quality/release-readiness/deployment/"evidence only". [VERIFIED `:1497-1509`]
- Mandatory non-empty `negative_findings` on readiness, rollout speed scope, planning manifest. [VERIFIED `:950,1015,1031`]
- Hardcoded negative requirements: negative AUROC delta on clinvar_noncoding, negative Spearman delta on brca2_saturation, rollout weakness on both splits, K20 speedup < 5.0. [VERIFIED `:968-982`]
- K20 scope: `ar_rollout_speed` row "rescoped" + `scope_decision.status == "accepted"`. [VERIFIED `:983-985`]
- Required paper sections (20). [VERIFIED `:831-851`]
- Required patterns: Carbon-500M, Joint-Embedding Predictive Architecture, DNABERT.*HyenaDNA.*Nucleotide Transformer, K20.*#42, "does not prove useful planning behavior", "negative-results and systems". [VERIFIED `:856-876`]
- 9 required benchmark rows: clinvar_coding, clinvar_noncoding, brca2_saturation, traitgym_mendelian, rollout_phased_haplotypes, rollout_synthetic_edit_chains, inference_efficiency, ar_rollout_speed, release_inputs. [VERIFIED `:60-70`]

#### Performance Targets (RFC-0016 / RFC-0010)
- Single-variant warm, M3 Max: < 200 ms. [VERIFIED RFC-0010 §3.5, RFC-0016 §3.2]
- 100k-variant VCF, M3 Max: < 30 min. [VERIFIED]
- Peak resident memory (loaded), M3 Max: < 8 GB. [VERIFIED]
- Quant budget int8 predictor: max 1.0 AUROC drop, max 0.02 cosine drop. [VERIFIED]
- Quant budget int4 Carbon + int8 predictor: max 2.0 AUROC drop, max 0.05 cosine drop. [VERIFIED]
- Artifact size: ~120 MB bf16, ~40 MB int8; Carbon ~1 GB separate. [VERIFIED]
- Regression threshold > 5% opens P1 issue. [VERIFIED RFC-0016 §3.5,3.7]

### Equations & Notation
```
model_id     = SHA-256( canonical_json( manifest ) )
input_commitment = SHA-256( canonical_json({
    dtype_config:{encoder_dtype, predictor_dtype},
    edit_spec:{alt, chrom, edit_type, pos, ref},
    pooling_config:{normalize, pool_radius, pool_type, state_layer},
    reference_window, version:1 }) )
output_commitment = SHA-256( canonical_json( asdict(ReceiptOutput) ) )
artifact_hash_i = SHA-256( bytes(artifact_file_i) )   # 1 MiB streaming
```

### Reconcile
1. Receipt `provenance.details`: RFC shows `null`; code always sets non-null details (backward compatible).
2. RFC-0011 narrowed to checksum-only; "runtime_attested" removed.
3. `score_variant` signature adds `window_start_bp`, `receipt_path`; raises InputError instead of FASTA auto-fetch (auto-fetch only via `score_vcf`).
4. int4 export/quantization: RFC default; runtime currently loads safetensors directly, no quantization. Export "open".
5. Crash-log sanitization: only in RFC, not found in repo code.
6. **"28 verified suite outputs"** from task framing; template produces 29 declared step-level outputs (4×4 + 2×5 + 2 aggregate + 1 readiness = 29). Number 28 not found in repo.

### Open Questions / Caveats
1. **No public release-run artifacts in the repo** (HF runs tree `geno-lewm-v021-strong-4f36eef-10k-r1`). Reviewer cannot re-run without private data/checkpoint.
2. **`checksum_only` provenance is integrity, not attestation.** No TEE/SNARK; cannot prove correct execution.
3. **Network guard uses `unittest.mock.patch` in production** — unconventional; indirect socket import could bypass.
4. **VCF JSONL per-row receipts**; no aggregate; scalability gap at WGS scale.
5. **K20 rollout speed open (#42)**; verifier hardcodes K20 < 5.0 to pass.
6. **ClinVar noncoding AUROC + BRCA2 Spearman require negative deltas** — verifier structurally enforces model is worse than Carbon on these splits.
7. **Export pipeline (ONNX/CoreML/GGUF) unimplemented.**
8. **Observability event/metric registry linting** end-to-end CI enforcement not verifiable from RFC text.

---

## EVIDENCE CARD: R6-chronology-learnings

### Summary

GenoLeWM is an action-conditioned Joint-Embedding Predictive Architecture (JEPA) over DNA, treating a genomic edit as an explicit action in a latent state space. A frozen Carbon-500M DNA foundation model encodes a contiguous reference window into a state vector; a small trainable predictor head and action encoder, trained in the style of LeWorldModel (Maes et al. 2026), estimate the post-edit latent state without re-running the full encoder. The system supports single-variant scoring, multi-edit haplotype rollout, latent planning via CEM, and unsupervised surprise-based pathogenicity scoring. This is a **negative-results and systems/reproducibility paper**: all four evaluated tracks either fail to beat Carbon-500M zero-shot or produce near-random metrics, the K=20 AR rollout speed target remains unmet, and the planning demo does not demonstrate useful planning behavior. Strongest published claim: the repository can train, package, evaluate, benchmark, and content-addressedly replay a genomic-edit world-model pipeline. History: 2026-05-20 scaffold-only `0.1.0-draft` → v0.1 terminal-demo (chr21 ClinVar, N=3,000, near-chance AUROC 0.519) → v0.2.1 "serious-completion" (2026-06-09).

### Key Facts

#### Project Thesis and Architecture
- **Core equation:** `ŝ_{t+1} = g(s_t, a)` where `s_t = enc(w_ref)`, `a = action(EditSpec)`, `enc` frozen. [VERIFIED `docs/spec/00-overview.md:19-20`]
- **State encoder:** Carbon-500M, frozen Phase 1; smallest model meeting quality bar, single-GPU when frozen. [VERIFIED `docs/design-decisions.md:17-22`]
- **Window length:** 12,288 bp (2,048 6-mer tokens). [VERIFIED `:33-35`]
- **Pooling:** centered-mean ±256 tokens around edit locus. [VERIFIED `:38-41`] (Note: deployed config uses pool_radius=8 — see R1/R7.)
- **State vectors:** L2-normalized at encoder output. [VERIFIED `:26-30`]
- **Predictor (RFC-0004 doc):** cross-attention Transformer, 4 cross + 2 self, `d_hidden=1024`, 8 heads, 2048 FFN, GELU, pre-LN. [VERIFIED `rfcs/0004:50-62`] (Note: code defaults differ — see R2/R7.)
- **Predictor params:** ~40M default; ~22M for d_hidden=768. RFC-0001 target was 20M. [VERIFIED `rfcs/0004:93-109`]
- **Action encoder:** 4 sub-encoders; ref/alt share SeqMicroEncoder; ~2.5M params. [VERIFIED]
- **v1 edit-length cap:** ≤16 bp. [VERIFIED]
- **Output MLP init:** final layer zero-init (identity-at-init). [VERIFIED `rfcs/0004:157-163`]
- **Why JEPA not discriminative:** (A) supervised classifier (needs labels, no rollout), (B) regression on ΔlogP (distillation, no rollout), (C) action-conditioned JEPA — selected for scoring/rollout/planning/surprise. [VERIFIED `rfcs/0001:169-185`]
- **Why action-conditioned not masked-autoencoding:** edits as conditioning unlock rollout/planning/surprise. [VERIFIED `docs/faq.md:18-26`]

#### Training Recipe
- **Loss:** `L = α(1−cos) + β·MSE/d_state`. [VERIFIED `docs/design-decisions.md:78-83`]
- **LeJEPA regularizer:** monitored-only Phase 1 (frozen encoder ⇒ collapse impossible). [VERIFIED `:86-91`]
- **Optimizer:** AdamW β₂=0.95, WSD schedule. [VERIFIED `:94-108`]
- **Batch size:** 256, edit-balanced [0.4,0.2,0.2,0.1,0.1]. [VERIFIED `:110-115`]
- **Edit source mix:** 40% gnomAD / 30% synthetic SNV / 20% synthetic indel / 10% ClinVar. [VERIFIED `:128-132`]
- **Holdouts:** chr21, ClinVar P/LP, gnomAD phased multi-edit. [VERIFIED `:142-148`]
- **Window overlap:** 67%, stride 8,192. [VERIFIED `:135-139`]
- **v0.1 run:** 160,000 samples, 20,000 steps, final loss 0.36124, run id `first-snv-carbon-500m-r1`, commit `cd2bfccb...`. [VERIFIED `model-card:187-194`]
- **Hardware:** NVIDIA H200. [VERIFIED]

#### Non-Goals (citable)
No new DNA FM pretraining; no DNA generation; no clinical decision support; no protein structure; no hosted service; no Carbon-likelihood replacement; no multi-omics until v2. [VERIFIED `rfcs/0001:80-85`; `docs/spec/00-overview.md:55`]

#### Success Criteria (stated, NOT met)
- Trained checkpoint on HF Hub: partially met (negative evidence).
- ClinVar coding AUROC ≥ Carbon zero-shot at ≥10× lower latency: **Not met** — 0.734375 vs 0.921875 (Carbon), delta −0.1875.
- 3-edit haplotype cosine ≥0.80: **Not met** — phased 0.289 vs source baseline ~0.997.
- Predictor <200 MB int8: not measured. [VERIFIED `rfcs/0001:114-124`]

#### Chronology
- **2026-05-20** `v0.1.0-draft`: scaffold, 19 RFCs (0001–0019), Phase 1 infra. No training. [VERIFIED `CHANGELOG.md:511-548`]
- **v0.1:** first Carbon training (`first-snv-carbon-500m-r1`, 20k steps, loss 0.361). Terminal demo on 32-row chr21 VCF. Eval 3,000 chr21 ClinVar, AUROC 0.5192 (CI 0.491–0.547), AP 0.1652, balanced accuracy 0.500. "Near-chance." [VERIFIED `model-card:202-211`]
- **2026-06-08** June 8 v0.2 readiness run: broader suite + Carbon baselines; mixed/negative. [VERIFIED `docs/faq.md:90-97`]
- **2026-06-09** `v0.2.1` "serious-completion": #202 checkpoint, #203 suite rerun, #204 planning demo, #205 paper. Preserves negative framing. Readiness `ok=true` for coverage; AR speed `rescoped`. [VERIFIED `CHANGELOG.md:15-27`]

#### Issue Number Meanings
- **#42:** AR rollout speed K=20 5× target — open. [VERIFIED]
- **#53, #55:** ClinVar coding/noncoding VEP. [VERIFIED]
- **#56:** All four VEP benchmarks. [VERIFIED]
- **#57:** Rollout fidelity benchmarks. [VERIFIED]
- **#58:** Paired with #56 (release-inputs). [VERIFIED]
- **#197:** All-up v0.2 readiness gate. [VERIFIED]
- **#202:** Post-v0.2 checkpoint lineage. **#203:** June 9 suite rerun. **#204:** planning demo. **#205:** paper package. [VERIFIED]

#### AR Rollout Speed Rescope
- RFC-0004 target 5.0× at K=20, ~2× at K=5. [VERIFIED `rfcs/0004:151-152`]
- Measured K=5 2.41386×; K=20 2.47322×. [VERIFIED]
- Decision `rescope_rfc0004_speed_target` status `accepted`; refs #42, #197; row `rescoped`. "Not rollout-speed evidence." [VERIFIED `rollout_speed_scope.py:24-25,73-85`]
- Accepted in 2026-06-08/09 window; no persisted scope JSON in repo. [VERIFIED tooling; scope JSON not found]

#### Canonical Numbers — see top of dossier (identical).

**Compare v0.1 efficiency (same H200):** 494 ms single-variant, 2.024 variants/s, ~1.1 GiB peak. v0.2.1 latency ~233× slower — likely batch/warmup difference, not regression. [VERIFIED `model-card:218-221`]

#### Explicit "What We Learned" (citable quotations)
1. "GenoLeWM does not broadly beat Carbon." [`CHANGELOG.md:23-24`; `AGENTS.md:55`; `README.md:89`]
2. "K20 rollout speed remains below the RFC-0004 target." [`CHANGELOG.md:24-25`; `README.md:316`]
3. "The released planning demo exercises the manifest-backed model path but does not prove useful planning behavior." [`AGENTS.md:58-59`; `ROADMAP.md:23`]
4. "The current evidence is useful systems evidence with mixed or negative model-quality results. Do not cite it as broad superiority over Carbon." [`README.md:74-75`]
5. "The strongest current claim is that the repository can train, package, evaluate, benchmark, and replay a genomic-edit world-model pipeline with content-addressed evidence." [`README.md:89-91`]
6. "The current generated paper is negative-results/systems evidence." [`ROADMAP.md:100`]
7. "Fixture smoke outputs are CI evidence, not model results." [`AGENTS.md:62`]
8. v0.1: "near-chance first-release metrics, not a robust comparison against those scorers." [`docs/faq.md:137-139`]
9. "the non-coding row preserved as a negative finding." [`docs/faq.md:149`]
10. "Checksum receipts prove artifact and output identity; they do not certify runtime behavior." [`README.md:323-324`]

### Equations & Notation
```
ŝ_{t+1} = g(s_t, a)
s_t = enc(w_ref)        # L2-norm, centered-mean pooled, d_state=1024
a = action(EditSpec)    # d_action=512 (RFC) / 64 (config) projected to d_hidden
L = α(1 − cos(ŝ_{t+1}, s_{t+1})) + β·MSE(ŝ_{t+1}, s_{t+1})/d_state
carbon_zero_shot_score = -(log_lik_alt - log_lik_ref)
AdamW β₂ = 0.95
K_speedup = T_naive(K) / T_cached(K)
```

### Reconcile
| Item | RFC/Spec | Implementation |
|------|----------|----------------|
| Predictor params | ~20M target (SPEC) | ~40M default (RFC-0004); ~22M variant. Inconsistent. |
| K=20 AR speedup | ~5× | 2.47×. Rescoped. |
| v0.1 latency | <200 ms | 494 ms (v0.1); 115,262 ms (v0.2.1). Neither met. |
| RFC-0001 status | "Draft" | All 19 RFCs remain Draft despite releases. |
| Receipts | future runtime modes | narrowed to checksum_only. |
| Training corpus | carbon-pretraining-corpus | rev cb4c13a... + gnomAD v4.1 + ClinVar 2026-04-15. Consistent. |
| PyPI publishing | Trusted Publishing (OIDC) | published via maintainer-token fallback (#201). |

### Open Questions / Caveats
1. **ClinVar N=16** extremely small; metrics 1/16-quantized; no meaningful CI.
2. **Rollout cosine ~0.29–0.30** vs source baseline ~0.997 (predictor nearly orthogonal to target). recall_at_k=1.0 suspicious (N=8 candidate set).
3. **v0.2.1 latency 115 s vs v0.1 494 ms** unexplained; 233× on same H200; "one sample, no warmup."
4. **K=20 mechanism unclear**; no no-cache vs with-cache ablation at K=20.
5. **No RFC promoted from Draft.** Spec success criterion ("all RFCs Accepted/Superseded") not met.
6. **Planning best_distance=23.66 has no scale.** stopped_reason=patience with 384 evals may be premature.
7. **Population calibration not validated** ("not a population-general reliability claim").
8. **No persisted rollout_speed_scope.json** in repo checkout.
9. **"serious-completion" dataset uses chr22, not chr21.** Train/eval chromosome split must be clarified.
10. **Deterministic vs non-deterministic training**: `--deterministic` (~15% slower); repeated-run reproducibility listed as open gap.

---

## EVIDENCE CARD: R7-efficiency-analysis

### Summary

GenoLeWM is a latent world model built on a frozen Carbon-500M. The **efficiency thesis**: pay one Carbon forward pass per reference window, then a cheap (~25–40M-param) predictor supports thousands of downstream latent queries (rollout, planning, VEP) without re-calling Carbon. The "pay once, query many" argument is formally correct for multi-edit rollout and planning. However, VEP scoring as implemented **always** requires a second Carbon call on the edited window, so single-variant benefit is at most a halving when the reference cache is warm — not elimination. The released `--release-efficiency` benchmark measures end-to-end wall-clock of the `geno-lewm-score` subprocess (cold CLI start, model loading, exactly two Carbon forwards); it does not exercise warm-cache or rollout paths. The measured ~115 s latency on H200 is dominated by model loading and cold-start, not pure inference. The only measured latent-cheapness evidence is the AR rollout speedup (2.41×/2.47×), both below RFC-0004 targets (K=20 badly missed). Predictive quality vs Carbon zero-shot is mixed-to-negative on all VEP tracks. Main contribution: reproducible pipeline + content-addressed evidence, not a model that beats its encoder.

### Key facts

**Architecture and parameter budget**
- Carbon-500M encoder ~500M frozen, bf16, `d_state=1024`. [VERIFIED ARCHITECTURE.md:39-41; RFC-0002 §3.5]
- Predictor code defaults: d_state=1024, d_hidden=768, n_heads=8, n_cross_layers=4, n_self_layers=2, ffn_dim=768. [VERIFIED `model.py:43-49`]
- RFC-0004 §3.1: d_hidden=1024, 4 cross+2 self, FFN=2048, ~40M params. [SPEC]
- **YAML config defaults use `d_state=512` and `n_layers=6`** — further reduction not in RFC-0004. [VERIFIED `defaults/train.yaml`, `score.yaml`]
- Action encoder ~25–30M; default `d_action=64`. [VERIFIED]

**Window/encoder hyperparameters**
- `DEFAULT_WINDOW_BP=12,288` = 2,048 6-mer tokens. [VERIFIED]
- RFC pooling `centered_mean`, `pool_radius=256` tokens. [SPEC]
- **Actual config default `pool_radius=8` tokens** (not 256) in all YAML + `bench/inference.py:71`. [VERIFIED — discrepancy]
- `state_layer=20` in all four YAML defaults. [VERIFIED]
- RFC-0002 §3.2: Carbon forward on 12,288 bp ~80 ms H100 bf16. [SPEC]

**VEP scoring data flow (two-Carbon-call path)**
- ARCHITECTURE.md §2.1: (1) ref→Carbon→s_t (or cache); (2) action→a_emb; (3) ŝ=g(s_t,a); (4) edited→Carbon→s_{t+1}; (5) `||ŝ−s_{t+1}||_2`. Total Carbon calls: 2 uncached / 1 warm. [VERIFIED]
- §5 diagram: Carbon on edited window mandatory regardless of cache. [VERIFIED]

**Release-efficiency benchmark methodology**
- `bench/inference.py --release-efficiency` spawns `geno-lewm-score` subprocess; `time.perf_counter_ns`. [VERIFIED `:178-195, 549-550`]
- Defaults: 100 samples, 10 warmup batches. [VERIFIED `:63-64`]
- Reported latency = **median** post-warmup ns / 1e6. [VERIFIED `:194`]
- Self-documents: "Subprocess wall-clock timing includes CLI startup and artifact loading overhead." [VERIFIED `:226`]
- Throughput = variant_count / median_batch_seconds. [VERIFIED `:193-196`]
- Peak memory: `RUSAGE_CHILDREN.ru_maxrss`, best-effort. [VERIFIED `:562-574`]
- Does NOT measure warm-cache, pipeline-batched, or rollout/planning latency.

**Canonical numbers (v0.2.1-r1, H200):** see top of dossier.

**Cache architecture**
- On-disk Parquet shards per (chrom × stride_block); content-addressed by `(window_hash, encoder_hash, state_layer, pool_type, pool_radius, dtype)`. [VERIFIED `cache.py:49-58`]
- SQLite index. [VERIFIED `cache.py:575-592`]
- Embedding `list<float16>`. [VERIFIED `cache.py:424`]
- **LRU edited-window cache: RFC-0002 §3.6 specifies 10,000-entry; NOT implemented in cache.py** (only reference-window disk cache). [VERIFIED]
- RFC §3.6 reference cache ≈ 750 MB. [SPEC]
- ARCHITECTURE.md:200 "8 workers, batch 256, ~150 steps/sec" cache-warm. [SPEC]
- RFC-0006 §6 "cache hit-rate budget" referenced but §6 is "Future work" with no numbers. [VERIFIED]

**Planning demo:** best_distance≈23.67, n_evaluations=384, elapsed≈15.34s, patience; no Carbon calls during search. [VERIFIED ARCHITECTURE §2.3, §5]

### Equations and notation
```
s_t = L2Normalize( centered_mean_pool( Carbon-500M(w_ref, layer=L) ) )
s_{t+1} = L2Normalize( centered_mean_pool( Carbon-500M(w_alt, layer=L) ) )
  L = state_layer (20 deployed; RFC default final)
  pool = centered_mean ±pool_radius (256 RFC; 8 deployed)
a_emb = ActionEncoder(chrom, pos, ref, alt) ∈ R^{d_action}   (64 deployed)
hat_s_{t+1} = g(s_t, a_emb) = L2Normalize( OutputMLP( CrossAttention(s_t, a_emb) ) )   (d_state 512 deployed / 1024 RFC)
sigma = || hat_s_{t+1} - s_{t+1} ||_2
L_pred = lambda_cos*(1 - cos) + lambda_mse*|| · ||^2
# AR rollout (K steps, 0 Carbon after step 0): K-1 predictor calls
speedup(K) = naive_median_ns / cached_median_ns
```

### Reconcile
| Parameter | RFC/SPEC | Code/config | Location |
|---|---|---|---|
| pool_radius | 256 tokens | **8 tokens** | all YAML; `bench/inference.py:71` |
| state_layer | "final" (−1) | **20** | all YAML |
| d_state | 1024 | **512** | all YAML |
| Predictor d_hidden | 1024 (RFC large); 768 target | **768** | `model.py:43-49` |
| Predictor n_layers | 4 cross + 2 self | `n_layers:6` → cross=4(default)/6(first-exp), self=2 — see R2 | varies |
| LRU edited-window cache | 10,000-entry | **Not present** | `cache.py` |
| RFC-0006 §6 cache hit-rate budget | referenced | **§6 is "Future work", no budget** | rfcs/0006 |
| AR K=20 speedup target | 5.0× | **2.47×** (missed) | `bench/rollout.py:182` |
| RFC-0016 §3.2 single-variant warm H100 | <50 ms cold / <5 ms warm | **115,262 ms** (cold subprocess, 2 Carbon, model load) | perf budget vs canonical |

### Open questions / caveats
1. **ClinVar N≈16** (1/16-quantized); no CIs; single misclassification = 6.25 pp.
2. **pool_radius discrepancy load-bearing:** ±48 bp deployed vs ±1,536 bp RFC. Changes what state captures. Paper must state deployed value.
3. **d_state discrepancy:** deployed 512 vs RFC 1024 ⇒ 1024→512 projection alters geometry. Impacts surprise/rollout claims. (NOTE: R1/R2/R6 evidence cards report d_state=1024 from the first-experiment config `train-carbon-500m-snv.yaml:39`; R7 reports the package-level `defaults/*.yaml` default of 512. The first-experiment config OVERRIDES to 1024. Treat d_state=1024 as the canonical released value; flag the 512 default as a config-layering caveat.)
4. **LRU edited-window cache unimplemented** — every score re-encodes edited window.
5. **Rollout benchmark toy dims** (d_state=64, CPU, fp32). 2.41×/2.47× not predictive of real model on GPU.
6. **No warm-cache single-variant latency measured.** <5 ms warm / <50 ms cold never demonstrated.
7. **Predictive quality rests on negative results.** Only positive (coding accuracy +0.0625 = 1/16) within quantization step; no permutation test/CI.
8. **Hardware mismatch:** benchmark H200, budget defined for H100. 115 s on H200 implies worse on H100.

### Efficiency analysis — three regimes

**Regime 1 — Cold single-variant VEP (what the benchmark measures):** subprocess = interpreter start + model loading (predictor + action-encoder safetensors + Carbon ~1 GB) + 2 Carbon forwards (~80 ms each H100) + predictor (~1 ms) + surprise. The 115,262 ms median is dominated by model loading, not compute. Confirmed by 0.31 variants/s throughput.

**Regime 2 — Warm reference cache, in-process, single variant:** cache lookup → predictor (~1 ms) → Carbon on edited window (~80 ms) → surprise. Expected ~80–100 ms. **Not measured. Not reported.** <50 ms cold target never validated for this checkpoint.

**Regime 3 — Latent rollout/planning (the thesis proper):** after initial encode, K-step rollout = 0 additional Carbon, K predictor calls; CEM 1024×5 = 25,600 predictor calls, 0 Carbon (demo 384 evals ~15 s, latent only). 2.41×/2.47× KV-cache speedups real (toy model, CPU) but below RFC-0004 5× at K=20.

**Honest verdict:** the efficiency thesis is **architecturally sound but operationally unmeasured in the regime it describes, and irrelevant to the measured regime.** Correct for rollout/planning; partial (2× Carbon-call reduction at best) for warm-cache VEP; not applicable to the cold-start benchmark as reported. The 115 s is evidence of expensive cold-process model loading — neither evidence GenoLeWM is slow per token nor evidence it is fast vs Carbon. The correct comparison (warm-cache GenoLeWM, one Carbon call, in-process vs Carbon zero-shot one Carbon call) has not been made. Conditions for the thesis to fully hold (pre-built ref cache; resident model; multi-variant amortization; KV-cache reaching 5× at K=20; rollout/planning-heavy workloads) — none of 1–4 demonstrated in v0.2.1-r1.

---
---

# PART II — VERIFIED CITATIONS

## (A) Verified BibTeX

```bibtex
@misc{carbon500m,
  title        = {Carbon-500M Model Card},
  author       = {{Hugging Face Biology Research and Zhongguancun Academy and TIGEM/University of Naples Federico II}},
  howpublished = {Hugging Face model card},
  year         = {2025},
  note         = {500M-parameter decoder-only autoregressive DNA model, Carbon family},
  url          = {https://huggingface.co/HuggingFaceBio/Carbon-500M}
}

@inproceedings{jepa2023,
  title     = {Self-Supervised Learning from Images with a Joint-Embedding Predictive Architecture},
  author    = {Assran, Mahmoud and Duval, Quentin and Misra, Ishan and Bojanowski, Piotr and Vincent, Pascal and Rabbat, Michael and LeCun, Yann and Ballas, Nicolas},
  booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
  pages     = {15619--15629},
  year      = {2023},
  eprint    = {2301.08243},
  archivePrefix = {arXiv},
  primaryClass  = {cs.CV},
  url       = {https://arxiv.org/abs/2301.08243}
}

@article{dnabert2021,
  title   = {{DNABERT}: pre-trained Bidirectional Encoder Representations from Transformers model for {DNA}-language in genome},
  author  = {Ji, Yanrong and Zhou, Zhihan and Liu, Han and Davuluri, Ramana V.},
  journal = {Bioinformatics},
  volume  = {37},
  number  = {15},
  pages   = {2112--2120},
  year    = {2021},
  doi     = {10.1093/bioinformatics/btab083},
  url     = {https://doi.org/10.1093/bioinformatics/btab083}
}

@inproceedings{hyenadna2023,
  title     = {{HyenaDNA}: Long-Range Genomic Sequence Modeling at Single Nucleotide Resolution},
  author    = {Nguyen, Eric and Poli, Michael and Faizi, Marjan and Thomas, Armin W. and Birch-Sykes, Callum and Wornow, Michael and Patel, Aman and Rabideau, Clayton and Massaroli, Stefano and Bengio, Yoshua and Ermon, Stefano and R{\'e}, Christopher and Baccus, Stephen},
  booktitle = {Advances in Neural Information Processing Systems (NeurIPS)},
  year      = {2023},
  eprint    = {2306.15794},
  archivePrefix = {arXiv},
  primaryClass  = {cs.LG},
  url       = {https://arxiv.org/abs/2306.15794}
}

@article{nucleotidetransformer2025,
  title   = {Nucleotide Transformer: building and evaluating robust foundation models for human genomics},
  author  = {Dalla-Torre, Hugo and Gonzalez, Liam and Mendoza-Revilla, Javier and Lopez Carranza, Nicolas and Grzywaczewski, Adam Henryk and Oteri, Francesco and Dallago, Christian and Trop, Evan and de Almeida, Bernardo P. and Sirelkhatim, Hassan and others},
  journal = {Nature Methods},
  volume  = {22},
  number  = {2},
  pages   = {287--297},
  year    = {2025},
  doi     = {10.1038/s41592-024-02523-z},
  url     = {https://doi.org/10.1038/s41592-024-02523-z}
}

@article{clinvar2014,
  title   = {{ClinVar}: public archive of relationships among sequence variation and human phenotype},
  author  = {Landrum, Melissa J. and Lee, Jennifer M. and Riley, George R. and Jang, Wonhee and Rubinstein, Wendy S. and Church, Deanna M. and Maglott, Donna R.},
  journal = {Nucleic Acids Research},
  volume  = {42},
  number  = {D1},
  pages   = {D980--D985},
  year    = {2014},
  doi     = {10.1093/nar/gkt1113},
  url     = {https://academic.oup.com/nar/article/42/D1/D980/1051029}
}

@article{gnomad2020,
  title   = {The mutational constraint spectrum quantified from variation in 141,456 humans},
  author  = {Karczewski, Konrad J. and Francioli, Laurent C. and Tiao, Grace and Cummings, Beryl B. and Alf{\"o}ldi, Jessica and Wang, Qingbo and Collins, Ryan L. and Laricchia, Kristen M. and Ganna, Andrea and Birnbaum, Daniel P. and others},
  journal = {Nature},
  volume  = {581},
  pages   = {434--443},
  year    = {2020},
  doi     = {10.1038/s41586-020-2308-7},
  url     = {https://doi.org/10.1038/s41586-020-2308-7}
}

@article{traitgym2025,
  title   = {Benchmarking {DNA} Sequence Models for Causal Regulatory Variant Prediction in Human Genetics},
  author  = {Benegas, Gonzalo and Eraslan, G{\"o}k{\c{c}}en and Song, Yun S.},
  journal = {bioRxiv},
  year    = {2025},
  doi     = {10.1101/2025.02.11.637758},
  note    = {TraitGym benchmark},
  url     = {https://doi.org/10.1101/2025.02.11.637758}
}

@article{alphamissense2023,
  title   = {Accurate proteome-wide missense variant effect prediction with {AlphaMissense}},
  author  = {Cheng, Jun and Novati, Guido and Pan, Joshua and Bycroft, Clare and {\v{Z}}emgulyt{\.e}, Akvil{\.e} and Applebaum, Taylor and Pritzel, Alexander and Wong, Lai Hong and Zielinski, Michal and Sargeant, Tobias and Schneider, Rosalia G. and Senior, Andrew W. and Jumper, John and Hassabis, Demis and Kohli, Pushmeet and Avsec, {\v{Z}}iga},
  journal = {Science},
  volume  = {381},
  number  = {6664},
  pages   = {eadg7492},
  year    = {2023},
  doi     = {10.1126/science.adg7492},
  url     = {https://doi.org/10.1126/science.adg7492}
}

@article{lejepa2025,
  title   = {{LeJEPA}: Provable and Scalable Self-Supervised Learning Without the Heuristics},
  author  = {Balestriero, Randall and LeCun, Yann},
  journal = {arXiv preprint arXiv:2511.08544},
  year    = {2025},
  eprint  = {2511.08544},
  archivePrefix = {arXiv},
  primaryClass  = {cs.LG},
  url     = {https://arxiv.org/abs/2511.08544}
}

@article{worldmodels2018,
  title   = {World Models},
  author  = {Ha, David and Schmidhuber, J{\"u}rgen},
  journal = {arXiv preprint arXiv:1803.10122},
  year    = {2018},
  eprint  = {1803.10122},
  archivePrefix = {arXiv},
  primaryClass  = {cs.LG},
  doi     = {10.5281/zenodo.1207631},
  url     = {https://arxiv.org/abs/1803.10122}
}

@article{dreamerv3,
  title   = {Mastering diverse control tasks through world models},
  author  = {Hafner, Danijar and Pasukonis, Jurgis and Ba, Jimmy and Lillicrap, Timothy},
  journal = {Nature},
  volume  = {640},
  pages   = {647--653},
  year    = {2025},
  doi     = {10.1038/s41586-025-08744-2},
  url     = {https://doi.org/10.1038/s41586-025-08744-2}
}

@inproceedings{pets2018,
  title     = {Deep Reinforcement Learning in a Handful of Trials using Probabilistic Dynamics Models},
  author    = {Chua, Kurtland and Calandra, Roberto and McAllister, Rowan and Levine, Sergey},
  booktitle = {Advances in Neural Information Processing Systems (NeurIPS)},
  year      = {2018},
  eprint    = {1805.12114},
  archivePrefix = {arXiv},
  primaryClass  = {cs.LG},
  note      = {Introduces PETS; uses the cross-entropy method (CEM) for model-predictive control},
  url       = {https://arxiv.org/abs/1805.12114}
}

@inproceedings{caduceus2024,
  title     = {Caduceus: Bi-Directional Equivariant Long-Range {DNA} Sequence Modeling},
  author    = {Schiff, Yair and Kao, Chia-Hsiang and Gokaslan, Aaron and Dao, Tri and Gu, Albert and Kuleshov, Volodymyr},
  booktitle = {Proceedings of the 41st International Conference on Machine Learning (ICML)},
  year      = {2024},
  eprint    = {2403.03234},
  archivePrefix = {arXiv},
  primaryClass  = {q-bio.GN},
  url       = {https://arxiv.org/abs/2403.03234}
}

@article{evo2024,
  title   = {Sequence modeling and design from molecular to genome scale with Evo},
  author  = {Nguyen, Eric and Poli, Michael and Durrant, Matthew G. and Kang, Brian and Katrekar, Dhruva and Li, David B. and Bartie, Liam J. and Thomas, Armin W. and King, Samuel H. and Brixi, Garyk and Sullivan, Jeremy and Ng, Madelena Y. and Lewis, Ashley and Lou, Aaron and Ermon, Stefano and Baccus, Stephen A. and Hernandez-Boussard, Tina and R{\'e}, Christopher and Hsu, Patrick D. and Hie, Brian L.},
  journal = {Science},
  volume  = {386},
  number  = {6723},
  pages   = {eado9336},
  year    = {2024},
  doi     = {10.1126/science.ado9336},
  url     = {https://doi.org/10.1126/science.ado9336}
}

@article{evo2_2025,
  title   = {Genome modeling and design across all domains of life with Evo 2},
  author  = {Brixi, Garyk and Durrant, Matthew G. and Ku, Jerome and Poli, Michael and others},
  journal = {bioRxiv},
  year    = {2025},
  doi     = {10.1101/2025.02.18.638918},
  url     = {https://doi.org/10.1101/2025.02.18.638918}
}

@article{alphagenome2025,
  title   = {{AlphaGenome}: advancing regulatory variant effect prediction with a unified {DNA} sequence model},
  author  = {Avsec, {\v{Z}}iga and Latysheva, Natasha and Cheng, Jun and Novati, Guido and Taylor, Kyle R. and Ward, Tom and Bycroft, Clare and Nicolaisen, Lauren and Arvaniti, Eirini and Pan, Joshua and others},
  journal = {bioRxiv},
  year    = {2025},
  doi     = {10.1101/2025.06.25.661532},
  note    = {Published version: Nature (2025), doi:10.1038/s41586-025-10014-0},
  url     = {https://doi.org/10.1101/2025.06.25.661532}
}

@article{reproducibility2021,
  title   = {Improving Reproducibility in Machine Learning Research (A Report from the {NeurIPS} 2019 Reproducibility Program)},
  author  = {Pineau, Joelle and Vincent-Lamarre, Philippe and Sinha, Koustuv and Larivi{\`e}re, Vincent and Beygelzimer, Alina and d'Alch{\'e}-Buc, Florence and Fox, Emily and Larochelle, Hugo},
  journal = {Journal of Machine Learning Research},
  volume  = {22},
  number  = {164},
  pages   = {1--20},
  year    = {2021},
  url     = {https://www.jmlr.org/papers/v22/20-303.html}
}
```

## (B) Could-not-fully-verify caveats (NOT fabricated)

- **`carbon500m`** — Model card page VERIFIED (HuggingFaceBio/Carbon-500M, 500M decoder-only AR DNA model, Apache-2.0). Card mentions a "technical report" but exposes no formal author list/title/citation. Author field = org collaboration. Any individual-author Carbon citation = PLACEHOLDER until tech report located.
- **`alphagenome2025`** — VERIFIED real (bioRxiv 10.1101/2025.06.25.661532; Nature 10.1038/s41586-025-10014-0). Cite bioRxiv. Nature volume/pages NOT verified.
- **`gnomad2020`** — DOI/title/authors/year VERIFIED; volume 581 / pages 434–443 INFERRED (final page-number check advised).
- **`clinvar2014`** — NAR 42(D1):D980–D985, 2014 VERIFIED; DOI `10.1093/nar/gkt1113` INFERRED. (Later ClinVar 2018 paper exists: Landrum et al., NAR 46(D1):D1062, 10.1093/nar/gkx1153 — 2014 is correct for "first archive.")

All others (jepa2023, dnabert2021, hyenadna2023, nucleotidetransformer2025, traitgym2025, alphamissense2023, lejepa2025, worldmodels2018, dreamerv3, pets2018, caduceus2024, evo2024, evo2_2025, reproducibility2021) fully VERIFIED.

## (C) "Supports which claim" map

- **carbon500m** — base genomic FM whose representations GenoLeWM builds on (system-under-study).
- **jepa2023** — origin of the JEPA latent-prediction objective.
- **dnabert2021** — prior DNA LM; transformer-for-genomics paradigm.
- **hyenadna2023** — long-context genomic baseline.
- **nucleotidetransformer2025** — FM genomics baseline + standard downstream tasks.
- **clinvar2014** — ground-truth P/B variant labels.
- **gnomad2020** — population constraint data; surprise-calibration null model.
- **traitgym2025** — causal regulatory variant benchmark.
- **alphamissense2023** — SOTA supervised VEP upper-bound context.
- **lejepa2025** — isotropic-Gaussian/SIGReg anti-collapse regularizer; collapse-prevention design.
- **worldmodels2018** — foundational world-model framing.
- **dreamerv3** — SOTA latent world model + imagination planning; positioning.
- **pets2018** — CEM-based MPC reference; planning method choice.
- **caduceus2024** — bi-directional Mamba genomic FM; related work.
- **evo2024 / evo2_2025** — genome-scale FMs; frontier positioning.
- **alphagenome2025** — unified regulatory VEP; contemporary baseline context.
- **reproducibility2021** — ML reproducibility checklist; methodology framing.

---
---

# PART III — LITERATURE / POSITIONING SCAFFOLD

*Prose, methodologically organized. Citations by author/short-name. Works flagged `[VERIFY EXISTENCE]` are uncertain.*

## 1. RELATED WORK

Organized around the *question* GenoLeWM asks — *can a learned model predict the latent consequence of an explicit genomic edit cheaply enough to support scoring, rollout, and planning?* Four research programs converge: DNA foundation models supply the encoder; variant-effect prediction supplies the evaluation and the foil; JEPA/anti-collapse SSL supplies the objective and the principal failure mode; latent world models supply the action-conditioned framing and planning interface. GenoLeWM is the intersection, and its negative result is most legible against all four.

### 1.1 DNA and genomic foundation models: encoders, not edit predictors

First wave ported MLM/causal-LM to nucleotides: **DNABERT** (Ji et al.) and DNABERT-2 (BERT-style masked k-mer/BPE); **Nucleotide Transformer** (Dalla-Torre et al., multi-species billion-param); **HyenaDNA** (Nguyen et al., implicit long convolutions, single-nucleotide resolution, hundred-kb–Mb context); **Caduceus** (Schiff et al., reverse-complement equivariant, Mamba/SSM); **Evo** and **Evo 2** (Nguyen et al.; Brixi et al. — *Evo 2 attribution/date* `[VERIFY EXISTENCE]`) pushed generative genomic LMs to genome scale (StripedHyena), demonstrating zero-shot variant-effect from sequence likelihood.

Unifying property: output is a representation or likelihood; where they "predict," they predict *tokens*, not *consequences of an intervention*. **Carbon-500M** (HuggingFaceBio — release date/attribution `[VERIFY EXISTENCE]`) is GenoLeWM's chosen instance: permissively licensed, consumer-hardware-runnable, competitive VEP. GenoLeWM uses it strictly as a **frozen state encoder** (RFC-0002): pay once per reference window, never fine-tune in Phase 1.

Sharp distinction: these answer *"what is the representation/likelihood of this sequence?"* GenoLeWM trains a small head to answer *"given the frozen representation of a reference window and an explicit edit action, what is the representation of the edited window?"* — an **edit-conditioned latent transition operator**, not a sequence scorer. The contrast to **GeneJepa** (`[VERIFY EXISTENCE]`, RFC-0001 §4.2) is sharpest: GeneJepa does masked-token JEPA over gene structure (no explicit action variable); GenoLeWM makes **the edit a conditioning input, not a masked perturbation of the target**. That decision is what would, in principle, license rollout and planning — and what the negative result interrogates.

### 1.2 Variant-effect prediction and resources: the evaluation, and the precise sense in which GenoLeWM is *not* a clinical predictor

Standard VEP ecosystem (RFC-0007 §3.1): **ClinVar** (Landrum et al.) clinical interpretations (P/LP vs B/LB, VUS excluded); **gnomAD** (Karczewski et al.) population frequency, repurposed as a per-context *null model* for surprise calibration (RFC-0009 §3.4); **TraitGym** (Benegas et al. — *attribution* `[VERIFY EXISTENCE]`) causal regulatory/Mendelian; **BRCA2 saturation genome editing** (Findlay et al.) continuous functional scores; **AlphaMissense** (Cheng et al.) structure-aware supervised missense, anchoring the high end.

GenoLeWM is **not** a clinical predictor, precise along four axes:
1. **Label-free by construction.** Surprise is an unsupervised side effect of predictor residual error (RFC-0009 §2); no pathogenicity label enters training. Cannot claim calibrated clinical-grade discrimination.
2. **Signal, not decision.** Every output framed as research signal; disclaims clinical decision support, refuses germline/reproductive use (RFC-0001 §3.2, §3.5).
3. **Comparison is to the *encoder*, not SOTA clinical models.** Baseline = Carbon-500M zero-shot likelihood (`-ΔlogLik`). Question is narrow/internal: *does action-conditioned latent prediction add evidence over the encoder it sits on?* NOT *does GenoLeWM beat AlphaMissense?* Conflating them = central overclaim to avoid.
4. **The honest finding is that it does not, yet.** Trails Carbon zero-shot on most VEP rows; only narrow coding-ClinVar balanced-accuracy positive. So GenoLeWM is not even a *Carbon-beating* scorer on this evidence — making "not a clinical predictor" an accurate scope description, not modesty.

This section earns trust: adopt the field's benchmarks for *comparability* while refusing the field's *claims*.

### 1.3 JEPAs and anti-collapse SSL: the objective, and why the frozen encoder sidesteps collapse but springs a different trap

Lineage = **Joint-Embedding Predictive Architecture**. **I-JEPA** (Assran et al.) established predicting *in representation space* yields strong features without pixel reconstruction. Genomic analogue is direct: GenoLeWM predicts Carbon-encoded window embeddings, never tokens.

Defining hazard of any predict-in-latent-space objective: **representation collapse** (RFC-0005 §2). SSL anti-collapse menu: stop-gradient/EMA (BYOL, I-JEPA), variance-covariance regularizers (**VICReg**, Bardes et al.; **PLDM** `[VERIFY EXISTENCE]`), contrastive negatives, distribution-matching. **LeWorldModel / LeJEPA** (Maes, Le Lidec, Scieur, LeCun, Balestriero `[VERIFY EXISTENCE]`) — direct parent — is distribution-matching: prediction + isotropic-Gaussian regularizer, no EMA/teacher, via **SIGReg** (random 1-D projections + Epps–Pulley normality test, Cramér–Wold). GenoLeWM inherits the *form*: Phase-2 regularizer = closed-form KL between batch-empirical Gaussian and `N(0,I)` (RFC-0005 §3.2).

The pivot: **In Phase 1, GenoLeWM freezes the encoder**, changing the collapse calculus entirely (RFC-0005 §2–§3.3). Frozen targets `s_{t+1}` are fixed; the predictor cannot drive the encoder to a constant; collapse is **mechanically impossible**; the Gaussian regularizer is monitored not trained. Legitimate simplification — same move **DINO-WM** (Zhou et al. `[VERIFY ATTRIBUTION]`) makes on frozen DINOv2.

But the price is the **latent-residual baseline trap**: with a fixed pre-trained target manifold and a single SNV moving the 1024-d frozen embedding only slightly, the trivial predictor `ŝ_{t+1} := s_t` (copy source state) becomes a *strong* baseline — not because it is good, but because the quantity to predict is small relative to embedding scale. LeWM/PLDM never face this fully because their encoders are *trained jointly*. By freezing Carbon, GenoLeWM forgoes that adaptation. This is why the eval suite was designed *a priori* with a **naive source-state baseline** (RFC-0007 §3.2.3), and why the central negative finding is rollout cosine *below* that baseline. §2.1 develops this.

### 1.4 Latent world models and planning: GenoLeWM as a world model whose action is a discrete genomic edit

Precise architectural claim (RFC-0001 §4.4). Lineage: **Ha & Schmidhuber "World Models"** (VAE encoder + recurrent latent dynamics + controller in imagination) → **Dreamer/DreamerV2/DreamerV3** (Hafner et al., latent recurrent SSM + policy via imagined rollouts) → **TD-MPC/TD-MPC2** (Hansen et al. `[VERIFY ATTRIBUTION]`, task-oriented latent dynamics + sampling-based MPC). GenoLeWM's planning = **CEM** in latent space with MPC replanning, same family as PlaNet/Dreamer and LeWM's CEM-over-action-sequences (RFC-0008 §3.4).

Distinctive move: the *type of action*. In all the above, action is a continuous control vector. In GenoLeWM, **the action is a discrete genomic edit** — `(position, type, ref, alt)` over `{SNV, INS, DEL, MNV, INDEL}` (RFC-0003) — encoded by a small action encoder, consumed by a cross-attention predictor. Planning = **combinatorial search over a discrete edit space**; hence CEM-over-factored-categoricals (deferred MCTS variant) over gradient-based optimization (RFC-0008 §4.1). The three operations (scoring, rollout, planning) are all *search/score over discrete edits scored by latent distance after predictor rollout*, with the efficiency thesis being they touch Carbon **once**.

Honest reading: GenoLeWM satisfies the *architectural* definition (predicts next latent state given action, exposes a planner) without yet the *behavioral* one (the demo records execution, not useful planning — stops on patience with non-zero best-distance). Dreamer/TD-MPC earn the title via demonstrated control; GenoLeWM concedes it has earned the title structurally but not empirically. That concession is the paper's integrity.

### 1.5 Comparison table

| Method family | Learns encoder? | Anti-collapse | Action type | Planning | Borrows | Differs |
|---|---|---|---|---|---|---|
| DNA FMs (DNABERT, NT, HyenaDNA, Caduceus, Evo/Carbon) | yes (pretrain) | n/a (gen/MLM) | none | none | Carbon as frozen encoder | adds edit-conditioned latent transition |
| VEP scorers (AlphaMissense, CADD, ESM-1b) | task-tuned | n/a (supervised) | none | none | benchmarks | label-free surprise; not clinical |
| JEPA/LeJEPA (I-JEPA, LeWM) | **yes, jointly** | EMA/**SIGReg** | n/a or control | CEM | latent-prediction objective + Gaussian reg | **freezes** encoder ⇒ collapse-immune but residual-trap-prone |
| Frozen-feature WM (DINO-WM) | no (frozen DINOv2) | n/a | continuous control | CEM/MPC | frozen-encoder strategy | discrete genomic-edit action |
| Latent WMs (Ha-Schmidhuber, Dreamer, TD-MPC) | yes | reconstruction/reward | continuous control | imagination/MPC | world-model framing + CEM-MPC | **action = discrete genomic edit** |

*Distinguishing experiment:* a **rollout-fidelity test against the source-state baseline stratified by edit magnitude** (RFC-0007 §3.2.3) — does the predictor beat `ŝ:=s_t` by a margin that *grows* with true latent displacement? On v0.2 it does not — the empirical heart of the paper.

## 2. DISCUSSION — Why the negative result occurs

Predictable consequence of four interacting causes; the contribution is diagnosing them mechanistically. Low prediction loss is **not** evidence of a learned edit-transition operator.

### 2.1 The latent-residual baseline trap (primary cause)
Training loss `α(1−cos)+β‖ŝ−s‖²/d` (RFC-0005 §3.1). The trivial `ŝ:=s_t`: a single SNV in a ~12 kbp window, mean-pooled over a frozen 1024-d embedding, perturbs the embedding only slightly — so `s_{t+1}≈s_t` and `cos(s_t, s_{t+1})` is near 1. The copy-baseline gets *most* achievable cosine for free; the predictor must capture only the tiny residual `Δ = s_{t+1}−s_t` (small norm, high variance, dominated by the bulk `s_t` direction). Gradient descent is *attracted to the copy solution*. **Evidence:** rollout cosine far below the source-state baseline (phased −0.70897, synthetic −0.689631). A predictor that learned `Δ` would beat copy; one that learned a *distorted* copy sits below it. RFC-0007 §3.2.3 introduced the naive baseline precisely to catch this — the negative result is the design's own canary firing. **Phase-1 collapse-immunity is the cause, not a defense:** the Gaussian regularizer guards *encoder collapse*; nothing in Phase 1 guards against *functional collapse of the predictor onto identity*.

### 2.2 Representation geometry — what an "edit direction" looks like in Carbon's space
For action-conditioning to be learnable, `(s_t, a_v) → Δ` must be *exposed* by Carbon's geometry (similar edits → similar low-dimensional roughly-linear displacements). No guarantee a frozen MLM/causal encoder organizes its space this way — Carbon was trained for likelihood, not edit-linearity. `Δ` may be near-isotropic noise of magnitude comparable to encoder variance (RFC-0009 §3.1: σ_raw inflated in GC-rich/repeat regions regardless of pathogenicity), highly nonlinear, or entangled with nuisance directions. **Show:** linear-vs-MLP probe gap for decoding `Δ`; `Δ`-norm vs encoder regional variance; edit-direction consistency.

### 2.3 Tiny evaluation slices and absence of power
ClinVar N≈16; metrics quantized at 1/16 = 0.0625; bootstrap CIs over 16 examples meaningless. A 1–2 point AUROC "win/loss" is within CI width. Honest statement: not "GenoLeWM trails Carbon" on every row with confidence, but "on slices this size, GenoLeWM does not show a detectable improvement and shows detectable deficits only where deltas exceed the CI."

### 2.4 Frozen encoder may not expose an edit-linear latent (structural cause)
Synthesis: (1) frozen Carbon optimized for the wrong invariance (what the sequence *is*, not how an edit moves it); (2) predictor has no leverage on the encoder (unlike LeWM where the encoder adapts); (3) **the efficiency thesis and the learnability thesis are in tension** — freezing Carbon enables "pay once, query many" but removes the encoder's ability to become edit-predictable. Deepest insight: *the very design choice that delivers efficiency may be the one that costs accuracy*, and the resolution (Phase-2 LoRA + LeJEPA) reintroduces the collapse risk Phase-1 was designed to avoid.

**Drop-in thesis paragraph:** *GenoLeWM's negative result is the joint consequence of a frozen target manifold on which single edits are near-invisible (making the copy-baseline strong), a frozen encoder geometry that was never shaped to expose edit directions (making the residual hard to extract), and evaluation slices too small to distinguish the resulting near-parity from noise. The Phase-1 design eliminated representation collapse by construction and, in doing so, created a residual-prediction problem in which the trivial identity map is a competitive baseline. The result is not that edit-conditioned latent world modeling is impossible, but that it is not learnable against a fully frozen, edit-agnostic encoder on under-powered slices.*

## 3. FUTURE RESEARCH PROGRAM

Twelve directions (hypothesis / smallest falsifying experiment / baseline to beat / failure mode). D1–D4 attack the cause; D5–D8 evaluation/measurement; D9–D12 deployment/scale.

**D1 — Phase-2 LoRA + LeJEPA.** Lightly LoRA-adapt Carbon under the Gaussian regularizer (lr 1e-5) so `Δ` becomes linearly extractable, lifting rollout cosine above the source-state baseline. Baseline: source-state copy + Phase-1. Failure: collapse (kl_reg>10) or still trails copy ⇒ problem is action/objective (→D2).

**D2 — Edit-contrastive / residual-centric objective.** InfoNCE over edits in the same window + predict whitened residual directly to kill the identity attractor. Baseline: shuffled-action predictor. Failure: no true-vs-shuffled separation ⇒ encoder doesn't encode action effects (→D1/D7).

**D3 — Edit-direction probing.** `Δ` is not linearly accessible in frozen Carbon (would explain the result). Train linear+MLP probes for `Δ`; report gap and `Δ`-norm vs regional variance. Failure (for hypothesis): strong linear probe ⇒ signal accessible, predictor/loss at fault (→D2).

**D4 — Curriculum on edit difficulty.** Start on high-displacement edits (nonsense/splice in conserved regions, weak copy-baseline). Baseline: edit-balanced sampler. Failure: final margin unchanged ⇒ geometry ceiling (→D1/D7).

**D5 — Larger eval suites + power analysis.** Compute min-N to detect 2-pt AUROC at 80% power; scale slices; re-report deltas with CIs + achieved power. Failure: even at full power, ties Carbon ⇒ real parity (sharper negative result).

**D6 — Better/adversarial baselines.** Beat (a) linear `s_t→Δ`, (b) kNN-retrieved `Δ` in matched context, (c) Carbon ΔlogLik. Failure: linear/kNN matches predictor ⇒ predictor adds nothing over interpolation.

**D7 — Multi-encoder ablation.** Repeat Phase-1 with ≥2 alternative frozen encoders (HyenaDNA, Caduceus, NT, Evo2); correlate probe linearity (D3) with rollout margin. Failure: all fail identically ⇒ frozen-target paradigm itself (strong evidence for D1).

**D8 — Multi-step rollout cost + VoE controls.** Per-K stratified cosine + surprise on order-scrambled vs correct edit chains (correct order → lower surprise if composition is real). Failure: no order-sensitivity ⇒ predictor treats edits as commutative no-ops (copy again).

**D9 — Realize caching/efficiency thesis honestly.** Implement+benchmark Parquet reference cache; cold/warm latency + per-K AR speedup; close or document the 5× K20 gap. Baseline: cold double-Carbon path. Failure: warm-cache still < 5× ⇒ AR predictor is the bottleneck (→D10).

**D10 — Distillation for latency.** Distill `s_t` + `Δ` prediction into a sub-100M student; measure laptop (MLX/Core ML) warm latency vs <200 ms target. Failure: latency win destroys accuracy ⇒ report accuracy-latency Pareto.

**D11 — Scaling laws for edit-conditioned prediction.** Sweep predictor params, training-edit count, edit-type coverage, encoder scale (Carbon 500M/3B/8B); fit margin-over-copy-baseline vs compute. Failure: flat curve ⇒ frozen paradigm bottleneck (clean scaling-law negative result).

**D12 — Surprise as Bayesian/directional + ensemble with ΔlogLik.** Report AUROC for {raw, calibrated, MC-dropout, surprise⊕ΔlogLik} on a power-adequate ClinVar slice (D5). Failure: ensemble adds nothing over ΔlogLik ⇒ surprise carries no orthogonal signal (retire it honestly).

**Program-level framing:** decision tree rooted at §2's diagnosis. D3/D7 run first (cheap, high-information): is the cause encoder geometry? If yes, D1/D2/D4 are the cure. D5/D6/D8 fix measurement so any cure is provable. D9–D12 earn deployment/scientific claims. The unifying question — *is edit-conditioned latent prediction learnable against a (near-)frozen genomic encoder, and at what compute?* — is the paper's most valuable bequest.

## Citation existence flags (for verification)

- `[VERIFY EXISTENCE]` Carbon-500M/3B/8B (HuggingFaceBio): confirm family, publisher, license, release date.
- `[VERIFY EXISTENCE]` LeWorldModel/LeJEPA (Maes, Le Lidec, Scieur, LeCun, Balestriero, "2026"; arXiv 2603.19312 per skill notes): confirm authorship/title; SIGReg attribution. (NOTE: citations block has a VERIFIED LeJEPA 2025 entry by Balestriero & LeCun, arXiv 2511.08544 — distinguish the LeJEPA paper from the LeWorldModel paper.)
- `[VERIFY EXISTENCE]` Evo 2 (Brixi et al.) and exact Evo (Nguyen et al.) attribution. (NOTE: both VERIFIED in citations block.)
- `[VERIFY EXISTENCE]` GeneJepa (RFC-0001 §4.2 "Oct 2025 masked-gene-token JEPA").
- `[VERIFY EXISTENCE]` TraitGym (Benegas et al.). (NOTE: VERIFIED in citations block.)
- `[VERIFY ATTRIBUTION]` Caduceus, DINO-WM (Zhou et al.), TD-MPC2 (Hansen et al.), DreamerV3, VICReg (Bardes/Ponce/LeCun), I-JEPA, AlphaMissense, gnomAD, ClinVar, BRCA2 SGE (Findlay et al.), HyenaDNA, Nucleotide Transformer, DNABERT/DNABERT-2.
- `[VERIFY EXISTENCE]` PLDM (VICReg-inspired latent world model) before naming as a baseline.

## Key source files (absolute paths)
- `/Users/abdel/dev/me/world-models/GenoLeWM/rfcs/0001-project-scope-and-goals.md`
- `/Users/abdel/dev/me/world-models/GenoLeWM/rfcs/0005-training-objective.md`
- `/Users/abdel/dev/me/world-models/GenoLeWM/rfcs/0007-evaluation-suite.md`
- `/Users/abdel/dev/me/world-models/GenoLeWM/rfcs/0009-surprise-based-pathogenicity-scoring.md`
- `/Users/abdel/dev/me/world-models/GenoLeWM/rfcs/0008-latent-planning.md`
- `/Users/abdel/dev/me/world-models/GenoLeWM/ARCHITECTURE.md`
- `/Users/abdel/dev/me/world-models/GenoLeWM/tools/release/serious_completion_paper.py`
