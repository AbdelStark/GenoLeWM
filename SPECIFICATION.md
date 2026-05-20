# GenoLeWM — Specification

- **Version:** 0.1.0-draft
- **Date:** 2026-05-20
- **Status:** Design phase. No reference implementation yet.
- **Authoritative source:** this document plus the RFCs in [`rfcs/`](rfcs/).

This specification is the synthesized canonical view. The RFCs are the
source of truth for individual subsystems and contain the rationale,
trade-offs, and unresolved questions. If this document and an RFC
disagree, the RFC wins; please open a PR against this document.

---

## 1. Scope

GenoLeWM is an **action-conditioned Joint-Embedding Predictive Architecture
(JEPA)** for DNA, where:

- **State** is a vector embedding of a contiguous DNA window, produced by a
  pre-trained DNA foundation model (default: Carbon-500M).
- **Action** is a structured genomic edit (SNV, indel, MNV, structural
  variant) applied to that window.
- **Prediction target** is the embedding of the post-edit window, produced
  by the same encoder.
- **Predictor** is a small trainable network that maps `(state, action)` to
  predicted next-state in latent space.

The training recipe follows
[LeWorldModel](https://github.com/lucas-maes/le-wm): end-to-end stable
training with only two loss terms (a prediction loss and an isotropic-
Gaussian distributional regularizer on encoder outputs), no EMA, no
auxiliary supervision.

The encoder is initialized from Carbon-500M and is **frozen by default**;
optional LoRA adaptation is permitted in Phase 2 (see RFC-0002).

GenoLeWM is **not** a replacement for Carbon. It is a small head that adds
a planning- and surprise-capable latent dynamics model on top of Carbon's
sequence-likelihood capability.

### 1.1 In scope

- Variant-effect prediction via predictor error / cosine distance.
- Multi-edit haplotype rollout in latent space.
- Planning (model-predictive control) over edit sequences.
- Surprise-based pathogenicity scoring.
- On-device inference on consumer hardware.

### 1.2 Out of scope (for v1)

- De novo DNA generation. Carbon already does this well; GenoLeWM uses
  Carbon embeddings as the substrate and does not decode back to bases.
- Protein structure prediction. AlphaFold / ESM-Fold solve this; GenoLeWM
  operates in nucleotide space.
- Multi-omics fusion (RNA-seq, ATAC-seq, methylation, single-cell).
  These are interesting extensions reserved for v2; see ROADMAP.
- Direct clinical decision support. The output of GenoLeWM is a research
  signal, not a clinical diagnosis. See §11.

---

## 2. Design principles

These principles drive every concrete decision in the RFCs.

1. **Compose with Carbon, don't replace it.** Carbon-500M / 3B / 8B remain
   the encoder. The trainable part of GenoLeWM is small (target: ≤ 30M
   parameters), trainable on a single GPU in hours.

2. **LeWM faithfulness over novelty.** Where a design choice has a
   precedent in LeWM (two-loss objective, no EMA, no teacher, autoregressive
   predictor over action steps), we take it. We deviate only with
   justification.

3. **The action is a first-class object.** Edits are not just deltas on the
   token sequence — they are structured inputs the predictor sees
   explicitly. This is the architectural difference vs. Carbon's
   `logP(alt) − logP(ref)` baseline.

4. **Surprise is the default scoring head.** Predictor error doubles as the
   model's pathogenicity / functional-disruption score. No labels needed
   to compute it.

5. **On-device is a design constraint, not an afterthought.** The
   predictor must fit in ≤ 200 MB at int8 and run with Carbon-500M on a
   16 GB consumer GPU or M-series Mac. This rules out heavy predictor
   architectures.

6. **Verifiability hooks are first-class.** Every inference output exposes
   the hashes of (encoder weights, predictor weights, input window, action
   spec). Downstream STARK attestation is the user's problem, but the
   ingredients are surfaced. See RFC-0011.

7. **Open everything.** Weights, training code, eval scripts, data
   pipelines. Apache-2.0.

---

## 3. System overview

```
       ┌────────────────────────────────────────────────────────┐
       │                       GenoLeWM v0                       │
       └────────────────────────────────────────────────────────┘

  ref window w_ref ──► ┌──────────────┐
                       │ Carbon-500M  │ ──► s_t  ∈ ℝ^d_state
                       │ (frozen)     │
                       └──────────────┘
                                                                 ┌──────────┐
  edit spec a ──────────► ┌──────────────┐ ──► a_emb  ──────────►│Predictor │──► ŝ_{t+1}
   (pos, type, ref, alt)  │ Action Enc.  │                       │   g(·,·) │
                          └──────────────┘                       └──────────┘

  edited window w_alt ─► ┌──────────────┐                              │
                         │ Carbon-500M  │ ──► s_{t+1} ◄────────────────┘
                         │ (frozen)     │                  prediction loss
                         └──────────────┘
```

Two losses:

- **L_pred:** distance between `ŝ_{t+1}` and `s_{t+1}` in latent space
  (cosine + MSE; see RFC-0005).
- **L_reg:** isotropic-Gaussian distributional regularizer on the encoder
  output distribution (LeJEPA-style), applied to the encoder side only.
  Because the encoder is frozen by default, this term is initially
  computed for **monitoring** and only becomes a training loss in Phase 2
  when LoRA is enabled.

For Phase 1 with frozen encoder, the active training loss is **L_pred
alone**; collapse is impossible because the targets `s_{t+1}` are
produced by a frozen, fixed encoder.

---

## 4. State representation

See RFC-0002 for full detail.

- **Encoder:** Carbon-500M by default. Carbon-3B and Carbon-8B are
  supported swaps via config.
- **Window size:** 2,048 6-mer tokens (~12.3 kbp) by default.
- **Layer:** the final transformer layer's hidden states by default; a
  configurable `state_layer` in `[-4, -1]` is supported for ablation.
- **Pooling:** mean-pool over the centered ±N tokens around the edit
  locus (default N=256, i.e., the ~1.5 kbp centered on the edit). The
  pooling window is centered, not the full window, to make the state
  edit-locality-aware.
- **Output dimension:** `d_state` = 1,024 (Carbon-500M hidden size).
- **Caching:** all reference-window embeddings are cached to disk
  (one Parquet shard per chromosome × window-stride) so that training
  does not re-run the encoder. This is essential for throughput.

---

## 5. Action representation

See RFC-0003 for full detail.

An action `a` is a tuple:

```
a = (relative_position, edit_type, ref_bases, alt_bases)
```

- `relative_position` ∈ [0, window_length_bp): integer offset within the
  window. Encoded as a sinusoidal positional embedding (dimension
  `d_pos = 128`).
- `edit_type` ∈ {SNV, INS, DEL, MNV, INV, DUP}: learned embedding
  (`d_type = 64`).
- `ref_bases`, `alt_bases`: variable-length DNA strings. For v1, each is
  capped at 16 bp; longer SVs are handled by a structural-variant adapter
  in v2 (see RFC-0003 §5).
- Each base sequence is tokenized via Carbon's 6-mer tokenizer (padded /
  truncated to length 16), then embedded via a shared 2-layer Transformer
  (`d_seq = 256`).

These four sub-embeddings are concatenated and passed through a
projection MLP to produce `a_emb ∈ ℝ^{d_action}`, with `d_action = 512`
in v1.

For multi-edit (haplotype) inputs, the predictor consumes a *sequence* of
action embeddings, applied autoregressively in latent space (see §6).

---

## 6. Predictor

See RFC-0004 for full detail.

The predictor is a small cross-attention Transformer:

- **Input:** `s_t` (1 token of dim `d_state`) and `a_emb` (1 to K tokens
  of dim `d_action`, projected to `d_state`).
- **Layers:** 4 cross-attention blocks (state attends to action, action
  attends to state, then 2 self-attention blocks on the fused sequence).
- **Hidden dim:** 1,024 (matches `d_state`).
- **Heads:** 8.
- **Output:** the first output token, projected through a 2-layer MLP
  to dim `d_state`. This is `ŝ_{t+1}`.
- **Parameter count target:** ~20M trainable.

For **autoregressive multi-step rollout** (e.g., applying 3 edits in
sequence), the predictor is unrolled K times with KV-caching. Each step's
`ŝ_{t+k+1}` becomes the input state for step `k+1`. This mirrors LeWM's
`ARPredictor`.

---

## 7. Training objective

See RFC-0005 for full detail.

**Phase 1 (frozen encoder):** single loss.

```
L_pred = α · (1 − cos(ŝ, s_{t+1})) + β · ||ŝ − s_{t+1}||²₂ / d_state
```

with `α = 1.0`, `β = 0.1` as starting defaults. The two-component form is
deliberately chosen to combine direction sensitivity (cosine) with
magnitude calibration (MSE).

**Phase 2 (encoder LoRA-adapted):** add the LeJEPA isotropic-Gaussian
regularizer on encoder outputs, weighted at `γ = 0.5`. This is the only
phase in which encoder collapse becomes possible; the LeJEPA term is the
safeguard.

Optimizer: AdamW (β1=0.9, β2=0.95, wd=0.05). Cosine LR with warmup.
Default peak LR 3e-4 for the predictor; 1e-5 for LoRA when active.

Batching: edit-balanced. Each batch has roughly equal counts of SNV / INS
/ DEL / MNV. See RFC-0006 §4.

---

## 8. Data pipeline

See RFC-0006 for full detail.

Training samples are tuples `(w_ref, a, w_alt)` where `w_ref` is a
reference window, `a` is an edit specification, and `w_alt = apply(a, w_ref)`.

**Three sources of edits**, mixed during training:

| Source | Purpose | Mix |
|--------|---------|-----|
| **gnomAD common variants** (AF ≥ 1%) | Realistic edit distribution; biological prior | 40% |
| **Synthetic uniform SNVs** | Uniform action-space coverage | 30% |
| **Synthetic indels** (length ∈ [1, 16]) | Coverage of harder edit types | 20% |
| **Curated pathogenic variants** (ClinVar P/LP) | Hard-negative anchor | 10% |

**Window sampling.** Reference windows are drawn uniformly from
[`HuggingFaceBio/carbon-pretraining-corpus`](https://huggingface.co/datasets/HuggingFaceBio/carbon-pretraining-corpus),
stratified by source (eukaryotic genes, mRNA, splice-enriched mRNA, GTDB
bacterial genomes) in the same proportions as Carbon's Phase-2 mix.

**Encoder caching.** Reference-window embeddings are pre-computed once
and cached. Edited-window embeddings are computed on-the-fly during
training (since the edits change per epoch via sampling), with a small
LRU cache for hot variants.

**Holdout sets.** Three holdouts:
- `holdout-chr`: an entire chromosome held out (chr21 by default).
- `holdout-clinvar`: every ClinVar P/LP variant.
- `holdout-haplotypes`: every multi-edit phased haplotype from gnomAD
  with ≥ 2 simultaneous variants in a 1 kbp window.

---

## 9. Evaluation

See RFC-0007 for full detail.

GenoLeWM is evaluated on three tracks.

### 9.1 Variant-effect prediction (VEP)

Same suite as Carbon's published evaluation:
- **ClinVar coding** (AUROC, AUPRC)
- **ClinVar non-coding** (AUROC, AUPRC)
- **BRCA2** (AUROC, AUPRC, Spearman ρ vs functional scores)
- **TraitGym Mendelian** (AUROC, AUPRC, Spearman ρ)

Scoring head: `score(v) = ||ŝ_{t+1} − s_{t+1}||₂` (surprise; see §10), and
also `score(v) = 1 − cos(ŝ_{t+1}, s_t)` (latent displacement). We report
both.

Baselines: Carbon-500M zero-shot, Carbon-3B zero-shot, GenoLeWM with
random predictor, Evo2-7B published numbers.

### 9.2 Latent rollout fidelity

For held-out multi-edit haplotypes from gnomAD, compute:
- Direct: `s_final = enc(w_haplotype)`.
- Predicted: `ŝ_final = g(g(g(s_0, a_1), a_2), a_3)`.

Metrics: cosine similarity, recall@k against held-out neighbors,
calibration plot of error vs edit count.

### 9.3 Inference efficiency

Wall-clock latency, batched throughput, and memory footprint, measured on:
- a single H100,
- a single consumer RTX 4090,
- an Apple M3 Max.

Targets: see RFC-0010 §3.

---

## 10. Surprise and planning

See RFC-0008 (planning) and RFC-0009 (surprise).

**Surprise** is the predictor's per-variant residual:

```
σ(v) = ||g(s_t, a_v) − enc(apply(v, w_ref))||₂
```

Calibrated against the distribution of σ over benign / common variants,
this yields a percentile-based pathogenicity score with no supervised
training of a classifier.

**Planning** is model-predictive control in latent space:

```
a*_{1:K} = argmin_{a_{1:K}} d(g^K(s_t, a_{1:K}), s_target)
            subject to:
              K ≤ K_max
              cost(a_k) ≤ budget
```

where `g^K` denotes K-step autoregressive predictor rollout.
Search is performed via Cross-Entropy Method (CEM) over the discrete
edit space for Phase 1; tree-search variants (MCTS) are considered for
Phase 2.

---

## 11. Safety, limitations, and intended use

**Intended use.**
- Research on genomic variant effects and latent representations of edits.
- Educational and exploratory tooling for personal genomics.
- A substrate for downstream verifiable-inference experiments.

**NOT intended for.**
- Clinical diagnosis or treatment decisions.
- Selection of embryos, germline edits, or any human reproductive use.
- Direct prediction of disease risk for a named individual without
  appropriate clinical interpretation by a qualified professional.

**Known limitations.**
- The model inherits all biases of Carbon's pretraining mix, which is
  predominantly eukaryotic with limited prokaryotic coverage.
- Variant effects in poorly studied regions (e.g., centromeres, telomeres,
  segmental duplications) are likely unreliable.
- Predictor error correlates with edit type frequency in training, so
  rare edit types (long structural variants) have less reliable surprise
  scores in v1.
- No assessment of population-specific calibration is performed in v1.
  Calibration on under-represented populations is an explicit Phase 2
  workstream.

See RFC-0001 §4 for the full safety framing.

---

## 12. Versioning

GenoLeWM follows semantic versioning **on the inference contract**:

- **MAJOR:** breaking change to the state or action format.
- **MINOR:** new predictor architecture or training recipe, but old
  checkpoints still load with old format.
- **PATCH:** weight updates, eval improvements, no API change.

The encoder version (Carbon-500M / 3B / 8B, and which revision) is
**always part of the model identifier**, e.g.,
`geno-lewm-v0.1.0-carbon-500m-r1`.

---

## 13. Open questions

These are the high-uncertainty design choices that the RFCs flag but do
not yet resolve. Track them in the issue tracker once we are out of
design phase.

1. **State layer selection.** Last layer vs penultimate vs concat-of-last-4.
   See RFC-0002 §6.
2. **Cosine vs MSE weighting.** Whether `(α, β) = (1.0, 0.1)` is correct
   under frozen Carbon, or whether magnitude should be ignored entirely.
   See RFC-0005 §3.
3. **Structural variant action format.** Long SVs do not fit the
   `(ref_bases, alt_bases) ≤ 16 bp` template. See RFC-0003 §5.
4. **Calibration of surprise across genomic context.** Coding vs non-coding,
   GC-rich vs GC-poor regions likely have different baseline surprise
   distributions. See RFC-0009 §4.
5. **LoRA targeting.** Which Carbon layers to LoRA-adapt in Phase 2, and
   at what rank. See RFC-0002 §7.

---

## 14. References

- LeWorldModel (Maes et al., 2026). https://github.com/lucas-maes/le-wm
- Carbon (Hugging Face Bio Research, 2026).
  https://huggingface.co/collections/HuggingFaceBio/carbon
- LeJEPA (Balestriero & LeCun, 2025). Isotropic-Gaussian regularization
  for JEPAs.
- I-JEPA (Assran et al., 2023). Image-based JEPA.
- CodeLeWM (Bakhta, 2026). https://github.com/AbdelStark/CodeLeWM
- ClinVar (Landrum et al., NAR 2018).
- TraitGym (Long et al., 2024).
- Saturation genome editing of BRCA1/BRCA2 (Findlay et al., Nature 2018).
