# GenoLeWM — LOCKED Paper Outline

**Source of truth:** `/Users/abdel/dev/me/world-models/GenoLeWM/paper/EVIDENCE_DOSSIER.md`
**Framing (non-negotiable):** Negative-results + systems/reproducibility paper. GenoLeWM does NOT broadly beat Carbon zero-shot. No clinical / privacy / runtime-assurance / deployment claims. Strongest positive claim: the repo can train / eval / benchmark / replay an action-conditioned genomic-edit latent world model with content-addressed (checksum-receipt) evidence. NEVER inflate negatives into wins.

---

## 0. ONE-SENTENCE CONTRIBUTION

We build, train, and content-addressedly replay an action-conditioned latent world model for genomic edits on top of a frozen DNA foundation model, and report the honest finding that — against a fully frozen, edit-agnostic encoder on tiny benchmark slices — the learned predictor fails to beat both Carbon zero-shot and a trivial "predict-no-change" baseline, diagnosing this as a structural latent-residual / frozen-encoder trap rather than a fixable bug.

---

## 1. TITLE

**Primary (locked):**
> **GenoLeWM: An Action-Conditioned Latent World Model for Genomic Edits — A Reproducible Pipeline and an Honest Negative Result**

**Subtitle / running head:** *Why predicting edit consequences against a frozen DNA encoder is a residual-prediction trap.*

Rationale: keeps "action-conditioned latent world model" and "genomic edits" (the system), foregrounds "reproducible pipeline" (the contribution) and "honest negative result" (the framing). Stronger and more specific than the default "Evidence-Bound Genomic Edit World Models, Benchmarks, and Negative Results" because it names the mechanism. NOTE: the `serious_completion_paper.py` verifier hardcodes the default title string; if the manuscript must pass that verifier verbatim, set the paper's internal `title` field to the default and use the above as display/arXiv title, OR update the verifier's expected title. Flag for the author.

---

## 2. ABSTRACT (5 sentences, Farquhar formula)

1. **(Context/problem)** A DNA foundation model scores a sequence, but it does not predict the *consequence of an explicit edit*: whether a learned latent transition operator can predict the post-edit embedding cheaply enough to support variant scoring, multi-edit rollout, and edit planning is an open question.
2. **(What we did)** We present GenoLeWM, an action-conditioned Joint-Embedding Predictive Architecture that freezes Carbon-500M as a state encoder and trains a small cross-attention predictor `ŝ_{t+1}=g(s_t,a)` to map a reference-window embedding and a discrete edit action to the edited-window embedding, with an unsupervised surprise score, a CEM latent planner, and a fully content-addressed train→eval→benchmark→replay pipeline.
3. **(Most remarkable HONEST number)** On the released v0.2.1-r1 checkpoint, the predictor's multi-edit rollout reaches a cosine similarity of only **0.289** to the true post-edit state — **0.709 below** the trivial "predict-no-change" baseline of ~1.0 — and trails Carbon zero-shot AUROC by **−0.1875** on ClinVar coding and **−0.3125** on ClinVar noncoding.
4. **(Diagnosis)** We show this is not noise but a *structural* latent-residual trap: a single edit barely moves a frozen 1024-dimensional embedding, so the identity map is a strong baseline, the frozen encoder was never shaped to expose edit directions, and the evaluation slices (N=16 for ClinVar) are too small to distinguish near-parity from chance — and the same freezing that delivers the "pay-Carbon-once" efficiency thesis is what removes the encoder's ability to become edit-predictable.
5. **(Contribution boundary)** Our contribution is therefore a reproducible, checksum-receipted systems artifact and a sharp, falsifiable diagnosis — not a model that beats its encoder — and we lay out a twelve-direction program (LoRA unfreezing, edit-contrastive objectives, power-adequate benchmarks) for testing whether edit-conditioned latent world modeling is learnable at all against a near-frozen genomic encoder.

---

## 3. TARGET VENUE / FORMAT

**Primary recommendation: arXiv preprint (cs.LG + q-bio.GN cross-list), comprehensive / not page-limited, then submit to a negative-results or reproducibility venue.**

- **arXiv first** — comprehensive form fits the "single source of truth" framing; no page limit lets the systems/provenance content and full reconcile tables live in the body.
- **Best fit conference track: ICBINB ("I Can't Believe It's Not Better") workshop** (NeurIPS/ICLR) — explicitly for rigorous negative results and "why it didn't work" diagnoses; this paper's mechanism-first negative result is exactly the format.
- **Strong secondary: NeurIPS Datasets & Benchmarks** — the content-addressed benchmark suite, the Carbon-zero-shot baselines, and the reproducibility receipt machinery are a benchmarks/infra contribution; the honest framing is a feature there, not a liability.
- **Do NOT target:** ICML/ICLR/NeurIPS main track or any clinical-genomics venue — the empirical results do not support a main-track "method wins" claim, and any biomedical venue would (correctly) reject the clinical framing.

One-line justification: *the paper's value is reproducibility + an honest, mechanistic negative result; arXiv preserves comprehensiveness and ICBINB/D&B reward exactly that.*

---

## 4. NOTATION CONTRACT (canonical symbols — use consistently everywhere)

| Symbol | Meaning | Canonical value |
|---|---|---|
| `w_ref` | reference DNA window (ACGTN) | 12,288 bp = 2,048 6-mer tokens |
| `w_alt` | edited window = `apply_edit(w_ref, e, preserve_length=True)` | — |
| `enc(·)` | frozen Carbon-500M state encoder (pooled, L2-normalized) | — |
| `s_t` | reference latent state `= enc(w_ref)` | `∈ ℝ^{d_state}` |
| `s_{t+1}` (a.k.a. `s*_{t+1}`) | true post-edit state `= enc(w_alt)` (training target / eval ground truth) | — |
| `ŝ_{t+1}` | predicted post-edit state `= g(s_t, a)` | — |
| `Δ` | true edit residual `= s_{t+1} − s_t` | small norm (core of §6 argument) |
| `a`, `a_emb`, `a_v` | action embedding of edit `v=(chrom,pos,ref,alt)` | `∈ ℝ^{d_action}` |
| `g(·)` | cross-attention Transformer predictor | — |
| `g^K` | K-step autoregressive rollout | — |
| `d_state` | encoder/state dimension | **1024** (released first-experiment config; package default 512 is overridden — flag as caveat) |
| `d_action` | action embedding dimension | **64** (released config; RFC-0003 says 512) |
| `d_hidden` | predictor hidden dimension | 768 |
| `L` | Carbon layer used for pooling (`state_layer`) | **20** (released; RFC default = final/−1) |
| `r` (`pool_radius`) | centered-mean pool radius in tokens | **8 tokens = ±48 bp** (released; RFC = 256 = ±1,536 bp) |
| `K` | rollout / planning horizon (# edits) | benchmarked at K=5, K=20 |
| `α, β` | prediction-loss weights (cosine, MSE) | α=1.0, β=0.1 |
| `γ` | Phase-2 KL regularizer weight (monitored-only in Phase 1) | 0.5 |
| `σ_raw(v)` | raw surprise `= ‖ŝ_{t+1} − s_{t+1}‖_2` | — |
| `σ_cal(v)` | calibrated surprise `= F_bucket(σ_raw)` ∈ [0,1] | — |
| `b(v)` | context bucket `= region|gc|repeat` | 11×3×5 = 165 |
| `score_carbon(v)` | Carbon zero-shot `= logP(w_ref) − logP(w_alt)` | baseline |
| `N` | evaluated variants per slice | 16 / 16 / 32 / 32 / 8 / 8 |

**Loss (canonical form, use verbatim):**
`L_pred = α·(1 − cos(ŝ_{t+1}, s_{t+1})) + β·‖ŝ_{t+1} − s_{t+1}‖²₂ / d_state`, with `ŝ_{t+1} = normalize(s_t + MLP(g(s_t, a)))` (residual + L2, final MLP layer zero-init / identity-at-init).

---

## 5. SECTION PLAN

### 1. Introduction
- The gap: DNA FMs score/embed sequences but do not predict the latent consequence of an *intervention*; frame the open question. [R6 core eqn; §1.1 scaffold]
- Our system in one line: `ŝ_{t+1}=g(s_t,a)`, frozen Carbon encoder, discrete-edit action, three downstream operations (score/rollout/plan). [R6, R7]
- State the honest headline up front: GenoLeWM does NOT beat Carbon broadly; the contribution is a reproducible content-addressed pipeline + a mechanistic negative result. Quote the most remarkable number (rollout cosine 0.289 vs ~1.0 baseline; AUROC delta −0.1875/−0.3125). [Canonical numbers]
- Contributions list (4): (i) the action-conditioned genomic-edit LeWM + frozen-encoder design; (ii) a content-addressed train→eval→benchmark→replay pipeline with checksum receipts and a re-render-to-verify paper toolchain; (iii) a rigorous negative result with a priori naive baselines; (iv) a mechanistic diagnosis (latent-residual trap) + 12-direction program. [R5, R4, R6]

### 2. Related Work
- §2.1 DNA/genomic foundation models — encoders, not edit predictors (DNABERT, Nucleotide Transformer, HyenaDNA, Caduceus, Evo/Evo2, Carbon). Carbon as the frozen encoder. [scaffold §1.1; citations dnabert2021, nucleotidetransformer2025, hyenadna2023, caduceus2024, evo2024, evo2_2025, carbon500m] — MUST mention "DNABERT.*HyenaDNA.*Nucleotide Transformer" (verifier pattern).
- §2.2 Variant-effect prediction + resources, and the precise sense in which GenoLeWM is NOT a clinical predictor (ClinVar, gnomAD, TraitGym, BRCA2 SGE, AlphaMissense). Four axes: label-free, signal-not-decision, compares-to-encoder, does-not-win. [scaffold §1.2; citations clinvar2014, gnomad2020, traitgym2025, alphamissense2023]
- §2.3 JEPA + anti-collapse SSL — the objective and the frozen-encoder collapse-vs-residual pivot (I-JEPA, LeJEPA/LeWM, VICReg, DINO-WM). MUST mention "Joint-Embedding Predictive Architecture" (verifier pattern). [scaffold §1.3; citations jepa2023, lejepa2025]
- §2.4 Latent world models + planning — GenoLeWM as a WM whose action is a discrete genomic edit (Ha-Schmidhuber, Dreamer, TD-MPC, PETS/CEM). [scaffold §1.4; citations worldmodels2018, dreamerv3, pets2018]
- §2.5 Comparison table (Table from scaffold §1.5).

### 3. Method
- §3.1 State encoder — frozen Carbon-500M, 12,288 bp window, `<dna>` wrapping, layer 20, centered-mean pool radius 8 tokens, L2-norm, `d_state=1024`. Content-addressed Parquet+SQLite reference cache. [R1, R7]
- §3.2 Action encoder — `EditSpec`/`RelEdit`, 5 edit types, ≤16 bp, 4 sub-encoders (sinusoidal position, type table, shared SeqMicroEncoder ref/alt), MLP → `d_action=64`. [R1]
- §3.3 Predictor — cross-attention Transformer (state↔action alternating blocks + self-attention), `d_hidden=768`, 8 heads, residual + L2 output, identity-at-init. ARPredictor (KV-cache rollout). [R2]
- §3.4 Training — Phase-1 frozen-encoder loop, loss `L_pred=α(1−cos)+β·MSE/d`, AdamW β₂=0.95, WSD schedule, LeJEPA KL monitored-only, collapse diagnostics every 500 steps, seed 104729, 20,000 steps. [R2, R6]
- §3.5 Surprise scoring — `σ_raw`, context-stratified calibration (gnomAD null model, 165 buckets, backoff, empirical CDF), `σ_cal`. [R4]
- §3.6 Latent planning — CEM-MPC over discrete edits, latent-only inner loop, factored proposal, refit-with-smoothing. [R4]
- §3.7 The efficiency thesis (and its honest scope) — "pay Carbon once," correct for rollout/planning, only a 2× Carbon-call reduction for warm-cache VEP, NOT what the released benchmark measures. [R7]
- **Reconcile box (RFC vs released config)** — explicitly list the load-bearing deltas (state_layer 20 vs −1; pool_radius 8 vs 256; d_action 64 vs 512; n_layers 6+2 vs 4+2; ffn 768 vs 2048; warmup 1000 vs 2000; weight_decay 0.1 vs 0.05; batch 8 vs 256-effective). [R1, R2, R7 reconcile tables]

### 4. Systems & Reproducibility (the real positive contribution)
- §4.1 Content-addressed provenance chain — canonical-JSON SHA-256, `model_id` = hash of manifest, input/output commitments, checksum receipts (integrity NOT attestation). [R5]
- §4.2 Evidence-bound benchmark suite — step kinds, `ok=true` gating, per-step `output_identities`, 28 verified suite outputs (note: template declares 29 step-level outputs; the readiness report counts 28 verified — flag the 28/29 discrepancy honestly). [R5]
- §4.3 Re-render-to-verify paper toolchain — `serious_completion_paper.py`: placeholder rejection, claim-boundary enforcement, **mandatory negative-findings** (verifier structurally REQUIRES negative ClinVar-noncoding AUROC delta, negative BRCA2 Spearman delta, rollout weakness on both splits, K20 < 5.0). This is the integrity mechanism — the paper cannot pass unless the negatives are real. [R5]
- §4.4 Fail-closed network guard, local-first importers, redaction filter — on-device privacy posture (stated as engineering, not a privacy *claim*). [R5]

### 5. Experiments
- §5.1 Setup — released identity (model_id sha256:cddb..., dataset_snapshot, commit d9b0681..., H200), the 6 benchmark slices + N, the two baselines (Carbon zero-shot; source-state no-change). [Canonical numbers, R3]
- §5.2 VEP results (Table 1 + Fig 2) — ClinVar coding/noncoding/BRCA2/TraitGym vs Carbon; mixed-to-negative; one narrow positive (coding accuracy +0.0625 = 1/16, within quantization). [Canonical]
- §5.3 Rollout fidelity (Table 2 + Fig 3) — cosine 0.289/0.302 vs source-state ~1.0; deltas −0.709/−0.690; recall_at_k=1.0 trivially (N=8). [Canonical, R3]
- §5.4 Efficiency (Table 3 + Fig 4) — 115 s single-variant latency (cold subprocess, model load dominated), 0.31 variants/s, 1.83 GiB; decompose the regimes. [R7]
- §5.5 AR rollout speed (Table 4 + Fig 5) — 2.41× K=5 (passes 2× local), 2.47× K=20 (fails 5× target; #42 rescoped); toy-dimension caveat. [R4]
- §5.6 Planning demo — best_distance=23.66, 384 evals, patience-stopped, synthetic proxy evaluator: "does not prove useful planning behavior" (verifier-required phrase). [R4]

### 6. Discussion — Why the Negative Result Occurs
- §6.1 The latent-residual / source-state baseline trap (primary). [scaffold §2.1]
- §6.2 Representation geometry — is `Δ` even edit-linear in Carbon's space? [scaffold §2.2]
- §6.3 Tiny-N / no-power — quantization at 1/16, CIs meaningless. [scaffold §2.3]
- §6.4 Frozen encoder may not expose an edit-linear latent; efficiency-vs-learnability tension. [scaffold §2.4 + drop-in thesis paragraph]

### 7. Limitations
- N=16 ClinVar, N=8 rollout, single-chromosome (chr22 placed-window) training, chr21 holdout not confirmed in v0.2.1 config, calibration not population-validated, planning on a non-learned synthetic proxy, no warm-cache latency measured, AR speedup on toy dims, RFCs all still "Draft." [R3, R4, R6, R7 caveats]

### 8. Negative Findings (explicit, structurally enforced)
- Enumerate the verifier-enforced negatives as first-class results, each tied to its number and the artifact that carries it. [R5, Canonical]

### 9. Reproducibility
- How to replay: model_id, dataset_snapshot, commit, hardware; the content-addressed receipts; the re-render-to-verify path; what a reader can and cannot reproduce without the private data/checkpoint. [R5]

### 10. Future Work
- The 12-direction program (D1–D12) as a decision tree rooted at §6's diagnosis. [scaffold §3]

### 11. Conclusion
- Restate: reproducible artifact + honest mechanistic negative result; the bequeathed question. Reiterate boundaries. [framing]

### Appendices
- A. Full equations & notation (Part I cards). B. Full RFC-vs-config reconcile tables. C. Benchmark suite step graph + output identities. D. Citation verification notes (Part II §B caveats). E. Carbon zero-shot scoring details.

---

## 6. CLAIM → EVIDENCE MAP

| # | Claim (as stated in paper) | Ground-truth number / artifact | Card |
|---|---|---|---|
| C1 | GenoLeWM trails Carbon zero-shot AUROC on ClinVar coding | AUROC 0.734375, delta −0.1875 | R3/R6/Canonical |
| C2 | GenoLeWM trails Carbon zero-shot on ClinVar noncoding (full regression) | AUROC 0.5625 (Δ−0.3125), AP 0.605456 (Δ−0.308967), acc/bal_acc 0.4375 (Δ−0.25) | Canonical |
| C3 | One narrow positive: ClinVar coding accuracy/balanced-accuracy +0.0625 — within the 1/16 quantization step, not significant | acc 0.75 (Δ+0.0625), bal_acc 0.75 (Δ+0.0625); N=16 ⇒ step 0.0625 | R3/R4 |
| C4 | GenoLeWM trails Carbon on BRCA2 saturation | Spearman 0.149194, delta −0.327713 | Canonical |
| C5 | TraitGym is near-zero/noise (nominal +0.056 over a near-zero Carbon) | Spearman −0.0279645, delta +0.055929 | Canonical |
| C6 | Rollout predictor is far worse than predicting no-change (phased) | cosine 0.288861, Δ vs source −0.70897; l2 33.3197, Δ+31.1929 | Canonical |
| C7 | Rollout predictor far worse than no-change (synthetic chains) | cosine 0.301608, Δ −0.689631; l2 28.8029, Δ+25.6371 | Canonical |
| C8 | recall_at_k=1.0 is uninformative at N=8 (Δ vs baseline = 0.0) | recall_at_k 1.0, Δ 0.0; N=8 | R3/R6 |
| C9 | Released single-variant latency is model-load/cold-start dominated, not per-inference compute | 115262.939968 ms (~115 s); benchmark self-doc "includes CLI startup and artifact loading overhead" | R7 |
| C10 | Throughput is ~0.31 variants/s on the released cold path | 0.3095340544239052 variants/s | R7/Canonical |
| C11 | Peak memory ~1.83 GiB | 1966149632 bytes | Canonical |
| C12 | AR KV-cache gives modest speedup; K=20 misses the 5× RFC-0004 target | K5 2.41386 (passes 2.0), K20 2.47322 (fails 5.0); #42 open, rescoped | R4/Canonical |
| C13 | Planning demo executes the model path but does not demonstrate useful planning | best_distance 23.656930390534644, n_evaluations 384, elapsed≈15.34s, stopped_reason patience; synthetic proxy evaluator | R4 |
| C14 | All ClinVar metrics are 1/16-quantized ⇒ N=16 ⇒ statistically underpowered | step 0.0625 = 1/16; HF model card N column | R3/R4/R7 |
| C15 | The full release is content-addressed and replayable | model_id sha256:cddb8f3b..., dataset_snapshot geno-lewm-data-v0.2.1-r1, commit d9b06815..., release_inputs pass, 28 verified suite outputs | R5/Canonical |
| C16 | The paper toolchain structurally enforces the negatives | `serious_completion_paper.py` requires negative noncoding AUROC delta, negative BRCA2 Spearman delta, rollout weakness both splits, K20<5.0, non-empty negative_findings | R5 |
| C17 | Efficiency thesis is correct for rollout/planning, partial for warm-cache VEP, unmeasured in the released regime | ARCHITECTURE §2.1 two-Carbon-call path; warm-cache latency not measured | R7 |
| C18 | Released config diverges from the RFC spec on load-bearing params | state_layer 20 vs −1; pool_radius 8 vs 256; d_action 64 vs 512 | R1/R2/R7 |

### HONEST CLAIM BOUNDARIES (what we explicitly do NOT claim)
- **NOT** that GenoLeWM beats Carbon zero-shot (it does not, broadly). [C1–C7]
- **NOT** any clinical / diagnostic / decision-support utility; outputs are research signals only.
- **NOT** statistical significance for the one positive delta (C3) — within quantization, N=16.
- **NOT** a privacy guarantee, runtime attestation, or correctness-of-execution proof — receipts are integrity/identity only (NOT attestation). [R5]
- **NOT** that the planner does useful planning (C13). [verifier-required phrase]
- **NOT** that the efficiency thesis is demonstrated in the released benchmark (warm-cache/rollout regimes unmeasured). [C9, C17]
- **NOT** that the AR rollout meets its 5× K=20 target (C12).
- **NOT** that the reported latency reflects production serving (single sample, no warmup, cold subprocess). [C9]
- **NOT** a population-general calibration reliability claim.
- **NOT** broad genomic coverage — placed-window training is single-chromosome (chr22).

---

## 7. FIGURE SPECS

### Fig 1 — Architecture / pipeline (TikZ)
- **Shows:** the end-to-end VEP data flow and the "pay Carbon once" thesis. Left: `w_ref` (12,288 bp) → frozen Carbon-500M (layer 20, centered-mean pool r=8) → `s_t` (with a cache hit/miss branch into the Parquet+SQLite reference cache). Edit action `v=(chrom,pos,ref,alt)` → ActionEncoder → `a_emb`. `s_t, a_emb` → cross-attention predictor `g` → `ŝ_{t+1}`. Parallel path: `w_alt = apply_edit(w_ref)` → frozen Carbon (SECOND call, highlighted red) → `s_{t+1}`. Output: `σ_raw = ‖ŝ_{t+1} − s_{t+1}‖₂`. Dashed box "rollout/planning: 0 further Carbon calls" around the `g`-only loop.
- **Annotations:** "frozen" lock icon on both Carbon blocks; "2 Carbon calls (uncached VEP) / 1 (warm) / 1 then 0 (rollout)"; checksum-receipt node at output.
- **Data values:** d_state=1024, d_action=64, d_hidden=768, window 12,288 bp, layer 20, pool_radius 8.

### Fig 2 — VEP delta bar chart (pgfplots, grouped horizontal bars)
- **Shows:** GenoLeWM minus Carbon zero-shot, signed, per metric per slice — the central negative result. Zero line emphasized; bars below zero = worse than Carbon.
- **Axes:** y = {clinvar_coding, clinvar_noncoding} × {accuracy, AUROC, AP, balanced_accuracy}; x = delta vs Carbon.
- **Exact values:** coding: acc +0.0625, AUROC −0.1875, AP −0.098947, bal_acc +0.0625. noncoding: acc −0.25, AUROC −0.3125, AP −0.308967, bal_acc −0.25. Separate small panel for Spearman deltas: BRCA2 −0.327713, TraitGym +0.055929.
- **Caption note:** N=16 (ClinVar), N=32 (BRCA2/TraitGym); ClinVar bars quantized at 1/16.

### Fig 3 — Rollout fidelity vs source-state baseline (pgfplots, paired bars)
- **Shows:** predictor cosine far below the trivial no-change baseline — the latent-residual trap, visually.
- **Axes:** x = {phased_haplotypes, synthetic_edit_chains}; y = cosine similarity [0,1]. Two bars per group: "source-state baseline ≈ 1.0" (baseline = cosine + |delta|, i.e., 0.288861+0.70897 ≈ 0.99783 and 0.301608+0.689631 ≈ 0.99124) and "GenoLeWM predictor".
- **Exact values:** predictor 0.288861 / 0.301608; baseline ≈ 0.99783 / 0.99124; annotate deltas −0.70897 / −0.689631. Secondary axis or inset for L2: predictor 33.3197 / 28.8029, baseline-delta +31.1929 / +25.6371.

### Fig 4 — Efficiency / latency-decomposition (pgfplots, stacked-bar or annotated regimes)
- **Shows:** the 115 s is cold-start + model-load dominated, and that the regime the efficiency thesis describes is unmeasured.
- **Axes:** three labeled regimes on x: "Regime 1: cold subprocess (MEASURED)", "Regime 2: warm-cache VEP (NOT measured)", "Regime 3: rollout/planning (latent-only)". y = latency (log scale, ms).
- **Exact values:** Regime 1 = 115262.94 ms (measured, single point); annotate "model load ≫ 2× Carbon ~160 ms expected"; Regime 2/3 shown as "unmeasured / expected ~80–100 ms warm" with explicit hatching. Throughput annotation 0.31 variants/s; peak memory 1.83 GiB.
- **Honesty note in caption:** subprocess wall-clock includes CLI startup + artifact loading (benchmark self-documents this).

### Fig 5 — AR rollout speedup vs target (pgfplots, bars + target lines)
- **Shows:** measured speedup vs RFC-0004 target; K=20 miss.
- **Axes:** x = {K=5, K=20}; y = speedup (×). Bars = measured; dashed horizontal target lines at 2.0 (K=5) and 5.0 (K=20).
- **Exact values:** K=5 measured 2.41386 (target 2.0 — passes); K=20 measured 2.47322 (target 5.0 — fails). Caption: toy synthetic dims (d_state=64, CPU); #42 open, status rescoped.

---

## 8. TABLE SPECS

### Table 1 — Main VEP results (GenoLeWM vs Carbon zero-shot)
| Slice | N | Metric | GenoLeWM | Δ vs Carbon | Status |
|---|---|---|---|---|---|
| clinvar_coding | 16 | accuracy | 0.75 | +0.0625 | pass |
| clinvar_coding | 16 | AUROC | 0.734375 | −0.1875 | pass |
| clinvar_coding | 16 | average_precision | 0.852976 | −0.098947 | pass |
| clinvar_coding | 16 | balanced_accuracy | 0.75 | +0.0625 | pass |
| clinvar_noncoding | 16 | accuracy | 0.4375 | −0.25 | pass |
| clinvar_noncoding | 16 | AUROC | 0.5625 | −0.3125 | pass |
| clinvar_noncoding | 16 | average_precision | 0.605456 | −0.308967 | pass |
| clinvar_noncoding | 16 | balanced_accuracy | 0.4375 | −0.25 | pass |
| brca2_saturation | 32 | Spearman ρ | 0.149194 | −0.327713 | pass |
| traitgym_mendelian | 32 | Spearman ρ | −0.0279645 | +0.055929 | pass |
- Footnote: "pass" = artifact-coverage/readiness pass, NOT a quality win. ClinVar metrics quantized at 1/16. Issues #53/#55/#56/#197.

### Table 2 — Rollout fidelity (vs source-state no-change baseline)
| Slice | N | cosine_mean | Δ cosine | l2_mean | Δ l2 | recall@k | Δ recall |
|---|---|---|---|---|---|---|---|
| rollout_phased_haplotypes | 8 | 0.288861 | −0.70897 | 33.3197 | +31.1929 | 1.0 | 0.0 |
| rollout_synthetic_edit_chains | 8 | 0.301608 | −0.689631 | 28.8029 | +25.6371 | 1.0 | 0.0 |
- Footnote: baseline = predict `ŝ_{t+1}=s_t` (no change). Negative cosine delta = predictor worse than identity. recall@k uninformative at N=8.

### Table 3 — Inference efficiency (released score path, H200)
| Metric | Value | Note |
|---|---|---|
| single_variant_latency_ms | 115262.939968 (~115 s) | cold subprocess, model load + 2 Carbon calls; one sample, no warmup |
| batched_throughput_variants_per_s | 0.3095340544239052 (~0.31/s) | — |
| peak_memory_bytes | 1966149632 (~1.83 GiB) | RUSAGE_CHILDREN best-effort |
| (v0.1 reference, same H200) | 494 ms / 2.024 v/s / ~1.1 GiB | ~233× faster — likely batch/warmup difference, not regression |

### Table 4 — AR rollout speed
| Horizon | Measured speedup | RFC-0004 target | Result | Status |
|---|---|---|---|---|
| K=5 | 2.41386× | 2.0× | passes | rescoped (#42) |
| K=20 | 2.47322× | 5.0× | fails | rescoped (#42) |
- Footnote: benchmarked on toy synthetic dims (d_state=64, CPU, fp32); not Carbon-scale.

### Table 5 — Artifact identity / reproducibility
| Field | Value |
|---|---|
| model_release | geno-lewm-v0.2.1-r1 |
| model_id | sha256:cddb8f3b9671090201370b9824b9da741b933ff296b651238f022df5f3ed6af4 |
| dataset_snapshot | geno-lewm-data-v0.2.1-r1 |
| commit | d9b06815cf8e64860f51d236b8db6ba55aa4154d |
| hardware | NVIDIA H200 (Linux x86_64, glibc2.35) |
| release_inputs | pass (28 suite outputs verified) |
| receipt provenance kind | checksum_only (integrity/identity, NOT attestation) |
| Carbon encoder revision | 5d31d59b3c845b288a13aedb1358934196852eec |

---

## 9. THE CENTRAL SCIENTIFIC ARGUMENT (Discussion seed — expand in §6)

*GenoLeWM's negative result is over-determined, not accidental. A single genomic edit inside a ~12 kbp window, mean-pooled over a frozen 1024-dimensional Carbon embedding, moves that embedding only slightly, so the true post-edit state `s_{t+1}` is already close to the source state `s_t`; the trivial "predict-no-change" map `ŝ_{t+1}:=s_t` therefore captures most of the achievable cosine for free, and the cosine+MSE loss is gradient-attracted to it — leaving the predictor to learn only the small, high-variance residual `Δ=s_{t+1}−s_t`. This is the latent-residual / source-state baseline trap, and the rollout evidence confirms it: the predictor's cosine (0.289, 0.302) sits ~0.70 BELOW the ~1.0 no-change baseline, i.e., the learned map is a worse-than-identity distortion. The trap is compounded by representation geometry (Carbon was trained for sequence likelihood, never to make edit directions linear or low-dimensional, so `Δ` may be near-isotropic noise) and by power (ClinVar slices of N=16 quantize every metric to multiples of 1/16, so the lone positive — coding accuracy +0.0625 — is exactly one variant and within noise, while the negatives that exceed the CI are the only defensible signed claims). Crucially, the freezing that creates this trap is the same choice that delivers the "pay-Carbon-once" efficiency thesis — yet even that thesis is unobserved in the released numbers, because the 115 s single-variant latency measures a cold subprocess dominated by model loading and two Carbon calls, not the warm-cache rollout/planning regime where Carbon is genuinely amortized away. The honest conclusion is not that edit-conditioned latent world modeling is impossible, but that it is not learnable against a fully frozen, edit-agnostic encoder on under-powered slices — a sharp, falsifiable claim that the future-work program is designed to test.*

---

## 10. RISKS / LANDMINES (and how the paper preempts each)

1. **"N=16 — these results are meaningless."** → Preempt: we say so first, explicitly, in the abstract, §5.1, §6.3, and Limitations; we report N in every table, note the 1/16 quantization, and frame conclusions as "no detectable improvement; detectable deficits only where deltas exceed the CI." We do NOT bootstrap-CI-launder the tiny slices.

2. **"You're comparing to the wrong baseline / why not AlphaMissense?"** → Preempt: §2.2 axis 3 makes the internal-comparison explicit — the scientific question is *does action-conditioning add evidence over the encoder it sits on?*, baseline = Carbon zero-shot + source-state no-change. We never claim to beat SOTA clinical predictors and say so in the boundaries.

3. **"The 115 s latency is absurd / contradicts your efficiency thesis."** → Preempt: §3.7 + §5.4 + Fig 4 decompose the three regimes and state that the benchmark measures cold-subprocess model-loading (self-documented by the harness), NOT the warm-cache/rollout regime the thesis describes; we explicitly mark Regimes 2–3 as unmeasured and refuse to claim efficiency was demonstrated.

4. **"Config doesn't match your RFC spec — which model did you actually run?"** → Preempt: a dedicated Reconcile box (§3) and Appendix B list every load-bearing divergence (state_layer 20 vs −1, pool_radius 8 vs 256, d_action 64 vs 512, n_layers 6+2 vs 4+2, ffn 768 vs 2048, batch 8 vs 256). Released values are canonical; RFC values are flagged as aspirational/spec.

5. **"recall@k = 1.0 looks like a win you're hiding."** → Preempt: Table 2 footnote + §5.3 state recall@k is trivially 1.0 at N=8 (Δ vs baseline = 0.0) and is uninformative; we do not report it as positive.

6. **"Checksum receipts are reproducibility theater — they don't prove the model ran correctly."** → Preempt: §4.1 states receipts are integrity/identity only, NOT attestation (no TEE/SNARK); quote the repo's own line "Checksum receipts prove artifact and output identity; they do not certify runtime behavior." The contribution is replayable provenance, not correctness proof.

7. **"Rollout/AR speedups are on toy dimensions and CPU — not your real model."** → Preempt: Table 4 + Fig 5 footnotes + §5.5 caveat state d_state=64/CPU/fp32 toy dims; we explicitly do not claim the ratios transfer to Carbon-scale, and list this in Limitations.

8. **"This is just a failed project dressed up."** → Preempt: the framing is offensive, not defensive — the contribution is (i) a reproducible content-addressed pipeline whose verifier *structurally requires* the negatives to be real, and (ii) a mechanistic, falsifiable diagnosis with a 12-direction test program. We position at ICBINB/D&B where this is the intended contribution type, and we never overclaim.

9. **(Bonus) "Your single positive (coding accuracy) — isn't that a win?"** → Preempt: §5.2 + C3 state +0.0625 is exactly one of sixteen variants and within the quantization step; we explicitly decline to call it significant.

---

## 11. VERIFIER COMPLIANCE CHECKLIST (so the manuscript passes `serious_completion_paper.py`)

The author must ensure the rendered manuscript contains (the toolchain hard-checks these):
- Required sections incl. Abstract, Introduction, Related Work, Method, Experiments, Citation Metadata, Artifact Inputs, Results, Planning Demo Evidence, Discussion and Learnings, Negative Findings, Limitations, Reproducibility, Conclusions, Artifact Availability, References. [R5]
- Required literal patterns: `Carbon-500M`; `Joint-Embedding Predictive Architecture`; `DNABERT`…`HyenaDNA`…`Nucleotide Transformer` (in order); `K20`…`#42`; `does not prove useful planning behavior`; `negative-results and systems`. [R5]
- 9 benchmark rows present: clinvar_coding, clinvar_noncoding, brca2_saturation, traitgym_mendelian, rollout_phased_haplotypes, rollout_synthetic_edit_chains, inference_efficiency, ar_rollout_speed, release_inputs. [R5]
- Non-empty negative_findings; negative noncoding AUROC delta; negative BRCA2 Spearman delta; rollout weakness both splits; K20 speedup < 5.0; ar_rollout_speed row = "rescoped" with accepted scope decision. [R5]
- No placeholder tokens (tbd/todo/placeholder/coming soon/fake/dummy/lorem ipsum/go here). [R5]
- Title: if passing the verifier verbatim, internal title = the default string `"GenoLeWM: Evidence-Bound Genomic Edit World Models, Benchmarks, and Negative Results"`; display/arXiv title = the improved Section 1 title. [R5 — flagged]
