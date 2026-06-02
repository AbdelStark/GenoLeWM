# RFC-0001: Project scope and goals

- **Status:** Draft
- **Author(s):** GenoLeWM Project
- **Created:** 2026-05-20
- **Updated:** 2026-05-20
- **Depends on:** —
- **Supersedes:** —
- **Implementation status:** N/A (scope document)

---

## 1. Summary

GenoLeWM is an action-conditioned Joint-Embedding Predictive Architecture
(JEPA) for DNA. It composes Carbon (a published DNA foundation model)
with a small predictor head trained in the style of LeWorldModel. This
RFC defines what the project is, what it is not, who it is for, and the
safety frame within which all subsequent RFCs operate.

## 2. Motivation

Three independent threads converge on the same project shape.

**(a) Carbon is here, and it is the right encoder.** Carbon-500M / 3B / 8B
were released by HuggingFaceBio in May 2026, are Apache-2.0, and are
small enough to run on consumer hardware. Carbon-500M in particular is
the only DNA foundation model that meets the "fits on a laptop" bar
simultaneously with strong variant-effect benchmark numbers. Building on
Carbon gives us a strong substrate without the cost of pretraining one
from scratch.

**(b) LeWorldModel proved that stable end-to-end JEPAs are now practical.**
LeWM (Maes, Le Lidec, Scieur, LeCun, Balestriero, 2026) shows that a
JEPA can be trained end-to-end from raw inputs with two loss terms
(prediction + Gaussian regularization), no EMA, no teacher network, no
auxiliary supervision. ~15M trainable parameters, single GPU, hours of
wall-clock. The recipe is small enough to port; the architectural
contract is small enough to specify completely; the stability properties
matter most in regimes (like genomics) where reproducibility is hard.

**(c) Personal health AI needs to live on-device.** The framing in
Clement Delangue's Carbon announcement is explicit: open weights, local
inference, transparency. Personal-genome interpretation is the most
sensitive consumer AI use case there is. A small, fast, local model
trained against a strong encoder is the only architecture that meets the
sovereignty constraint.

The intersection of these three is GenoLeWM: a small action-conditioned
JEPA head over Carbon that supports variant scoring, multi-edit rollout,
planning, and local inference.

## 3. Specification

### 3.1 Goals

GenoLeWM aims to:

1. Make **per-edit latent prediction** a first-class operation on top of
   Carbon. The system takes a `(reference window, edit)` pair and
   predicts the post-edit embedding directly, without re-encoding.
2. Match Carbon-500M zero-shot variant-effect-prediction quality on
   ClinVar coding/non-coding while running an order of magnitude faster
   per query (after caching reference embeddings).
3. Support **multi-edit (haplotype) rollout** in latent space, enabling
   compositional reasoning about combinations of variants without
   re-encoding intermediate sequences.
4. Provide a **planning** primitive over discrete edit spaces (RFC-0008).
5. Provide a **surprise-based pathogenicity score** that requires no
   labelled training (RFC-0009).
6. Be **deployable on consumer hardware** (M-series Mac, 16 GB consumer
   GPU) end-to-end, with the predictor head < 200 MB at int8 (RFC-0010).
7. Expose **artifact provenance hooks** in release and demo paths so
   datasets, models, inputs, and outputs can be reproduced and audited
   (RFC-0011).

### 3.2 Non-goals

GenoLeWM explicitly does not:

1. Train a new DNA foundation model from scratch. Carbon is the encoder.
2. Generate DNA sequences. Carbon does this; GenoLeWM operates in
   latent space.
3. Provide clinical decision support. Output is a research signal.
4. Predict protein structure. AlphaFold, ESM-Fold, and others address
   this; GenoLeWM is nucleotide-only.
5. Build a hosted service. On-device first. A hosted variant is welcome
   from the community but not in this project.
6. Replace Carbon's likelihood scoring. We complement it.

### 3.3 Audience

Primary:

- **ML researchers working on DNA foundation models** who want a
  predictor head, planner, and surprise scorer they can attach to
  Carbon (or to other DNA encoders by analogy).
- **Open-source bioinformatics developers** who want to ship
  variant-scoring tools that respect user data sovereignty.
- **Personal-genomics enthusiasts** (the audience Clem's Carbon tweet
  named: Bryan Johnson, Sid Sijbrandij, the "quantified-self with WGS"
  community) who want local-first, transparent tools.

Secondary:

- Reproducible-ML engineers who care about dataset, checkpoint, and
  evaluation artifact integrity.
- Clinical genomics researchers who want a fast first-pass variant
  scorer for hypothesis generation (not for clinical decisions).

### 3.4 Success criteria

The project is considered successful when, by the end of Phase 3
(see ROADMAP):

- A trained checkpoint is published on the Hugging Face Hub.
- AUROC on ClinVar coding meets or exceeds Carbon-500M zero-shot at
  ≥ 10× lower per-variant latency after caching.
- Multi-edit rollout cosine similarity on held-out 3-edit haplotypes
  is ≥ 0.80.
- The on-device app skeleton scores a 100k-variant VCF in < 30 minutes
  on an M3 Max with predictor + Carbon-500M running locally.
- At least one external research group has trained a GenoLeWM-style
  predictor on a different DNA encoder.

### 3.5 Safety framing

Genomic data is permanent, identifying, and family-implicating. The
safety framing is informed by that.

**On data handling.** The reference implementation never sends user
genome data over a network. All inference paths must be runnable
offline. Any cloud-touching component must be optional and clearly
labeled.

**On clinical use.** All outputs are research signals. The README,
documentation, and CLI banners state this. The CLI emits a warning when
ClinVar P/LP variants are detected in user input, pointing to clinical
follow-up rather than relying on the model's output.

**On germline editing and reproductive use.** GenoLeWM is not to be
used to predict the effects of germline edits for reproductive purposes.
This is stated in the license addendum and in the README. We do not
ship features that meaningfully facilitate this use case.

**On population calibration.** The training data is dominated by
populations over-represented in published genomic resources. The
documentation must call this out explicitly and the eval suite must
report per-population performance where possible (Phase 2 workstream).

### 3.6 Relationship to upstream projects

GenoLeWM depends critically on:

- **Carbon** (HuggingFaceBio). If Carbon changes its tokenizer or
  output convention, GenoLeWM bumps its MAJOR version.
- **LeWorldModel** (Maes et al.). We follow LeWM's architectural and
  training recipes; deviations are documented in RFCs.
- **Hugging Face Transformers** as the runtime, **Datasets** as the
  data layer, **safetensors** as the weight format.

GenoLeWM is intended to be a respectful downstream user of Carbon:
attribution is in the README, evaluations are published against
Carbon's own benchmarks, and feature requests upstream go through
HuggingFaceBio's normal channels.

## 4. Rationale and alternatives

### 4.1 Why JEPA over a discriminative scorer?

We considered three options for "what should sit on top of Carbon to do
variant scoring faster":

- **A. A supervised classifier head** (regress variant pathogenicity).
  Rejected because (i) it requires labels, which are scarce and biased;
  (ii) it does not generalize to multi-edit rollout or planning;
  (iii) it would not give us a surprise signal.
- **B. A regression head on `logP(alt) − logP(ref)` deltas**, learned
  to approximate Carbon-3B's scores from Carbon-500M's input. Rejected
  because it is just a distillation; nothing about the resulting model
  supports rollout or planning.
- **C. An action-conditioned JEPA (GenoLeWM).** Selected because it is
  the only option that supports all of: faster scoring (via predictor
  caching), multi-edit rollout (predictor unrolls), planning (CEM over
  predictor), surprise (predictor residual).

### 4.2 Why action-conditioned, not masked-autoencoding?

There is a published `GeneJepa` (Oct 2025) that does masked-gene-token
prediction. That is a representation-learning JEPA without explicit
actions; it learns gene co-expression structure but cannot reason about
interventions. GenoLeWM's distinguishing decision is that **edits are
explicit conditioning inputs**, not just perturbations to a masked
target. This is what enables planning and rollout.

### 4.3 Why Carbon, not Evo2 / Generator-v2 / Nucleotide Transformer?

Carbon is the only DNA foundation model that simultaneously:
- has open weights with a permissive license,
- has a small variant (500M) that runs on consumer hardware,
- matches the larger Evo2-7B on most variant-effect benchmarks,
- has explicit "designed for fine-tuning and continual pretraining"
  framing from the publisher (HuggingFaceBio).

The first three are nice-to-have; the fourth is the deciding factor.
Carbon was released with the expectation that downstream heads would
be built on it. GenoLeWM is exactly that.

### 4.4 Why "world model" framing?

Two reasons.

First, intellectual honesty: the architecture is genuinely a world
model in the LeWM sense — it predicts the next state in latent space
given an action, and it supports planning. Calling it anything else
would obscure what it does.

Second, the framing makes the contract clear to ML researchers: anyone
who has read LeWM knows what the API will look like. This reduces the
cost of community contribution.

## 5. Unresolved questions

- Whether to commit, in v1, to supporting any DNA encoder other than
  Carbon. The architecture is encoder-agnostic in principle. The
  pragmatic answer is "Carbon-only in v1, others via community PRs."
- Whether to provide a hosted API later. Probably not, but the demand
  signal post-launch may force a reassessment.
- Whether to add a regulatory addendum to the license forbidding
  clinical decision use, vs leaning on the README disclaimer alone.

## 6. Future work

This RFC unlocks every other RFC; without a scope, there is nothing to
specify.

Beyond v1: multi-omics integration (RNA-seq, ATAC-seq), cross-species
generalization studies, larger-Carbon-encoder training, and a v2
generative head that can decode latents back to DNA (probably as a
wrapper around Carbon, not a new decoder).

## 7. Changelog

- 2026-05-20 — Initial draft.
