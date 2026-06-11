# RFC-0007: Evaluation suite

- **Status:** Draft
- **Author(s):** GenoLeWM Project
- **Created:** 2026-05-20
- **Updated:** 2026-06-11
- **Depends on:** RFC-0001, RFC-0002, RFC-0003, RFC-0004, RFC-0005, RFC-0006
- **Supersedes:** —
- **Implementation status:** Partial — artifact-level `geno-lewm-eval`,
  deterministic bootstrap CIs, measured-baseline deltas,
  `geno-lewm-carbon-baseline`, `geno-lewm-eval-all`, generated
  `eval_report.md`, recorded `eval_config.effective.yaml`,
  release-efficiency report generation, v0.2 benchmark-suite
  orchestration, rollout-fidelity metrics, and the v0.2.1 generated
  paper/readiness artifact chain exist. The current public evidence is
  mixed or negative versus Carbon and does not satisfy the broader
  model-quality, K=20 rollout-speed, or planning-quality goals; future
  work is stronger held-out evidence, broader snapshots, and target
  hardware performance.

---

## 1. Summary

This RFC defines GenoLeWM's evaluation suite, organized into three
tracks: variant-effect prediction (VEP), latent-rollout fidelity, and
inference efficiency. Each track has a fixed set of benchmarks, fixed
metrics, fixed baselines, and a reproducible reporting format. The
suite is the contract: a checkpoint is only "released" once it has run
the full suite and reported the numbers.

## 2. Motivation

A research-grade ML project lives or dies by its evaluation. For
GenoLeWM specifically:

- The variant-effect benchmarks are how we demonstrate parity with /
  advantage over Carbon's zero-shot likelihood scoring.
- The rollout-fidelity benchmarks are how we justify the world-model
  framing (if the predictor cannot compose actions, the framing is
  misleading).
- The efficiency benchmarks are how we earn the on-device claim.

The suite is also designed to catch regressions: every release runs
the same numbers, so a 1-point AUROC drop on ClinVar coding is
visible immediately.

## 3. Specification

### 3.1 Variant-effect prediction (VEP)

GenoLeWM's primary product is variant scoring. The eval matches
Carbon's published suite for direct comparability.

#### 3.1.1 Benchmarks

| Benchmark | Source | Variant type | Size |
|-----------|--------|--------------|------|
| **ClinVar coding** | NCBI ClinVar (latest release) | SNV+indel in coding regions | ~50k |
| **ClinVar non-coding** | NCBI ClinVar (latest release) | SNV+indel in non-coding regions | ~30k |
| **BRCA2** | Findlay et al. 2018 saturation genome editing | SNV in BRCA2 exons | ~4k |
| **TraitGym Mendelian** | TraitGym benchmark | curated Mendelian variants | ~10k |

These exactly match Carbon's published numbers; results are directly
comparable to Carbon's model card.

#### 3.1.2 Scoring heads

Two scoring heads are reported per benchmark:

- **Surprise score** (default for VEP):
  ```
  surprise(v) = ||g(s_t, a_v) − enc(apply(v, w_ref))||₂
  ```
  Higher surprise → more pathogenic prediction.

- **Latent displacement**:
  ```
  displacement(v) = 1 − cos(g(s_t, a_v), s_t)
  ```
  Larger displacement → larger predicted functional change.

We report both because they expose different things: surprise measures
"unusual" variants, displacement measures "high-impact" variants.

#### 3.1.3 Metrics

- **AUROC**: area under ROC curve, with positive class = "Pathogenic" or
  "Likely pathogenic", negative class = "Benign" or "Likely benign".
  VUS (variants of uncertain significance) excluded.
- **AUPRC**: area under precision-recall curve. Same labels.
- **Spearman ρ** (BRCA2, TraitGym only): correlation between score and
  the published functional score (BRCA2 has continuous fitness scores).

#### 3.1.4 Baselines

Every report includes:

- **Random** (predictor with random weights at the same architecture).
- **Carbon-500M zero-shot likelihood**: `score(v) = -ΔlogLik(alt, ref)`.
- **Carbon-3B zero-shot likelihood**: same, from Carbon's published model.
- **Published Evo2-7B numbers** (where available).

The first two are run in our pipeline at every release. The last two
are quoted from Carbon's model card with explicit attribution.

#### 3.1.5 Reporting

Each VEP run produces a JSON file:

```json
{
  "model": "geno-lewm-v0.1.0-carbon-500m-r1",
  "benchmark": "clinvar_coding",
  "n_variants": 50127,
  "head": "surprise",
  "auroc": 0.8X,
  "auprc": 0.8X,
  "ci_95_auroc": [0.8X, 0.8X],
  "encoder_id": "HuggingFaceBio/Carbon-500M",
  "encoder_hash": "sha256:...",
  "predictor_hash": "sha256:...",
  "timestamp": "2026-MM-DDTHH:MM:SSZ"
}
```

Confidence intervals are computed via stratified bootstrap (1,000
resamples). All JSON files for a release are aggregated into the
release's `eval_report.md`.

### 3.2 Latent-rollout fidelity

This track tests the world-model claim: do K-step predictor rollouts
match the encoder's representation of the K-edit haplotype?

#### 3.2.1 Benchmarks

- **Phased multi-edit haplotypes from gnomAD**: 1-, 2-, 3-, 5-, and
  8-edit haplotypes from the `holdout-haplotypes` set (RFC-0006 §3.8).
- **Synthetic edit chains**: random sequences of compatible edits in
  held-out windows, K ∈ {1, 2, 3, 5, 8, 13}.

#### 3.2.2 Metrics

For each `(window, edit_list)` instance:

- **Cosine similarity** between predicted final latent `ŝ_{t+K}` and
  encoder ground truth `enc(apply_all(w, edits))`.
- **L2 distance** between the same.
- **Recall@k** of the ground truth among the K-nearest cached reference
  windows in the corpus.

Aggregated across instances: mean ± std, plus per-K stratified
reporting (does fidelity degrade with rollout length?).

#### 3.2.3 Calibration check

A predictor that always outputs `s_t` (i.e., predicts no change) would
score deceptively well on small edits in conserved regions, where the
true `s_{t+1}` is close to `s_t`. To catch this, we report:

- **Naive-baseline cosine**: `cos(s_t, enc(apply(v, w_ref)))`. The
  predictor's cosine should beat this by a meaningful margin.

### 3.3 Inference efficiency

#### 3.3.1 Benchmarks

For each of three reference machines, measure:

- **Single-variant latency** (cold cache, warm cache).
- **Batched throughput** (variants per second, batch sizes 1 / 8 / 64 /
  256).
- **Memory footprint** (peak GPU / unified memory, peak RSS).

#### 3.3.2 Reference machines

| Machine | Spec | Use case |
|---------|------|----------|
| Server | 1× H100 80 GB, 256 GB RAM | training-class inference |
| Workstation | 1× RTX 4090 24 GB, 64 GB RAM | researcher's local GPU |
| Laptop | Apple M3 Max 64 GB | the freedom-tech target |

For the laptop, we use Core ML / MLX backends; for the others, PyTorch
with CUDA. Reports are per-backend.

#### 3.3.3 Targets

| Metric | Server | Workstation | Laptop |
|--------|-------:|------------:|-------:|
| Single-variant latency (warm) | < 5 ms | < 20 ms | < 200 ms |
| Batched throughput (B=256) | > 5,000 v/s | > 1,000 v/s | > 100 v/s |
| Peak memory (predictor only) | < 200 MB | < 200 MB | < 200 MB |
| Peak memory (with Carbon-500M) | < 3 GB | < 3 GB | < 8 GB |

These targets are the public commitment for v1.

### 3.4 Continuous evaluation

Every PR that touches predictor code, training code, or data pipeline
triggers a "smoke eval": a 1k-variant ClinVar coding subset and a
500-window rollout subset. A drop of > 2 AUROC points or > 0.05 cosine
points fails the PR.

Full evals run on every release candidate.

### 3.5 Eval reproducibility

Every eval run records:

- The eval suite's git SHA.
- The model checkpoint's SHA-256.
- The dataset's content-addressed identifier (HF Hub commit SHA for
  ClinVar; date and source for gnomAD).
- The PyTorch / CUDA / Python versions.
- Hardware information.

Three independent re-runs of the same eval on the same model+data are
expected to agree to within ±0.5 AUROC. We log all three and report
the median.

### 3.6 Eval CLI

```
geno-lewm eval --model PATH --benchmark BENCH [--head surprise|displacement]
geno-lewm eval-all --model PATH --output report.md
```

`eval-all` runs every benchmark in §3.1–3.3 and produces the release-
ready Markdown report.

Until that runner is fully wired, the maintainer release helper renders
the same report artifact from measured metrics JSON:

```
python -m tools.release.eval_report --metrics-json metrics.json --output eval_report.md
```

The helper rejects empty metrics and placeholder wording so release
packages cannot pass with handwritten or planned results.

## 4. Rationale and alternatives

### 4.1 Why mirror Carbon's eval suite?

Direct comparability. If our numbers are reported on different
benchmarks, no reader can tell whether GenoLeWM is better, worse, or
comparable to Carbon. Adopting Carbon's published suite removes that
ambiguity entirely.

### 4.2 Why both surprise and displacement heads?

They measure different things and the user is likely to want different
ones for different applications:

- A **clinical-research-style** user cares about pathogenicity
  (surprise: "unusual variants" relative to the model's expectation).
- A **functional-genomics** user cares about effect magnitude
  (displacement: "this variant changes the latent a lot").

Reporting both lets the user pick the right head and prevents us from
quietly optimizing one at the cost of the other.

### 4.3 Why include a "naive baseline" in rollout fidelity?

Without it, a degenerate predictor that outputs `s_t` regardless of the
action could score deceptively well. Reporting both the predictor's
score and the naive baseline's score lets the reader see how much
work the predictor is actually doing.

### 4.4 Why an Apple M3 Max in the reference machines?

The freedom-tech target audience (RFC-0001 §3.3) is overwhelmingly on
Apple Silicon. Treating M3 Max as a first-class target keeps Phase 3
honest. We do not require Apple Silicon for development; it is just a
target.

### 4.5 Why bootstrap CIs rather than reporting just AUROC?

ClinVar's benchmark sizes are large enough that point estimates are
stable, but BRCA2 is small (~4k variants) and its AUROC bootstraps
with a non-trivial spread. Reporting CIs prevents over-interpretation
of small differences.

## 5. Unresolved questions

- Whether to include population-stratified eval (gnomAD has population
  AFs). Probably yes, but only in Phase 2 when we have enough variants
  per population for stable estimates.
- Whether to include functional assays beyond BRCA2 (e.g., MAVE,
  MaveDB). These exist; including them is a question of bandwidth.
- How to handle ClinVar label updates between releases (variants can
  be reclassified). v1 pins to a specific ClinVar release date.

## 6. Future work

- Cross-species evaluation (mouse, fly variant benchmarks).
- Adversarial robustness (deliberately constructed hard variants).
- Calibration evals (reliability diagrams, ECE) for the surprise score.
- A leaderboard, in coordination with the Carbon team if they are
  receptive.

## 7. Changelog

- 2026-06-02 — Updated implementation status for artifact-level eval,
  Carbon baseline scoring, aggregate reporting, and efficiency evidence.
- 2026-05-20 — Initial draft.
