# GenoLeWM

> **An action-conditioned JEPA world model for DNA, built on top of Carbon.**
> Genetic edits become first-class actions. The model learns latent transitions.
> Planning, surprise-based pathogenicity scoring, and on-device personal-genome
> inference fall out naturally.

[![License](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](LICENSE)
[![Status](https://img.shields.io/badge/status-design--phase-orange.svg)](#status)

---

## TL;DR

[Carbon](https://huggingface.co/collections/HuggingFaceBio/carbon) is a powerful
autoregressive DNA foundation model. It scores variants by computing
`logP(alt) − logP(ref)`. That's strong, but two things are missing:

1. The model has no explicit notion of **an edit** as an input it can be
   conditioned on. Every variant scoring call is a full re-encoding of
   `ref` and `alt`.
2. The model has no **predictive latent dynamics**. You cannot roll out
   "what if I apply these three edits in sequence" without re-running the
   model on every intermediate sequence.

[LeWorldModel](https://github.com/lucas-maes/le-wm) gives us a recipe for
action-conditioned predictive coding in latent space: a stable, end-to-end
JEPA with two losses (next-embedding prediction + isotropic-Gaussian
regularizer), no EMA, no teacher network, ~15M trainable parameters.

**GenoLeWM = LeWorldModel × Carbon × genomic edits.**

- **State** `s_t` ∈ ℝ^d : a frozen Carbon embedding of a genomic window.
- **Action** `a_t` : a structured genetic edit (SNV, indel, structural).
- **Predictor** `g(s_t, a_t) → ŝ_{t+1}` : trained to match `enc(edited_window)`.

That single change unlocks:

- **Variant-effect prediction** at a fraction of Carbon's inference cost.
- **Multi-edit haplotype rollout** in latent space (compose actions, never
  decode back to DNA).
- **Planning** over edit sequences via latent MPC (e.g., "minimal edit set
  to restore the reference-like latent neighborhood").
- **Surprise-based pathogenicity scoring** — predictor error as an
  unsupervised pathogenicity signal.
- **On-device personal-genome inference** — Carbon-500M + a ~15M-param
  GenoLeWM head fits on a laptop.

---

## Why this exists

Three convictions, in order of importance:

1. **Personal health AI should not live behind a black-box API.**
   Open weights, open data, on-device. This is the framing
   [Clem Delangue articulated](https://x.com/ClementDelangue/status/2057071550352781771)
   when releasing Carbon, and GenoLeWM is built to inhabit that frame.

2. **Carbon is a likelihood engine. We need a *planning* engine.**
   Variant scoring is a special case of latent rollout. A world model is the
   more general object.

3. **Stable, small, single-GPU JEPAs are now possible.**
   [LeWorldModel](https://github.com/lucas-maes/le-wm) showed that you can
   train a stable end-to-end JEPA without EMA, teachers, or auxiliary
   supervision, with two loss terms. The recipe ports directly.

This project is the genomic sibling of
[CodeLeWM](https://github.com/AbdelStark/CodeLeWM), which applies the same
LeWM recipe to source-code edits.

---

## Repository layout

```
geno-lewm/
├── README.md                      # this file
├── SPECIFICATION.md               # canonical technical specification
├── ARCHITECTURE.md                # high-level architecture diagram + walkthrough
├── ROADMAP.md                     # milestones, owners, dates
├── LICENSE                        # Apache 2.0
├── pyproject.toml                 # package metadata (stub)
├── rfcs/                          # design RFCs (numbered)
│   ├── README.md                  # RFC process and index
│   ├── 0000-template.md           # template for new RFCs
│   ├── 0001-project-scope-and-goals.md
│   ├── 0002-state-encoder-carbon-integration.md
│   ├── 0003-action-representation-genomic-edits.md
│   ├── 0004-predictor-architecture.md
│   ├── 0005-training-objective.md
│   ├── 0006-data-pipeline.md
│   ├── 0007-evaluation-suite.md
│   ├── 0008-latent-planning.md
│   ├── 0009-surprise-based-pathogenicity-scoring.md
│   ├── 0010-on-device-personal-genome-deployment.md
│   └── 0011-verifiable-inference-attestation.md
├── docs/
│   ├── glossary.md                # terminology
│   ├── design-decisions.md        # log of resolved trade-offs
│   └── faq.md                     # frequently-asked questions
├── examples/                      # planned showcase notebooks (placeholders)
│   └── README.md
└── geno_lewm/                     # package source (stubs only at design phase)
    └── __init__.py
```

---

## Status

**Phase 0 — Design.** Spec and RFCs are being written. No model weights yet.

The roadmap targets a usable v0.1 (Phase 1) in 6 weeks: Carbon-500M state
encoder, action encoder for SNVs only, MLP-cross-attention predictor,
training on a 10% slice of `carbon-pretraining-corpus`, evaluation on
ClinVar coding/non-coding.

See [ROADMAP.md](ROADMAP.md).

---

## How to read this repo

If you have 5 minutes: read this README, then skim
[SPECIFICATION.md](SPECIFICATION.md).

If you have 30 minutes: read SPECIFICATION.md end-to-end, then
[ARCHITECTURE.md](ARCHITECTURE.md), then RFC-0001 (scope) and
RFC-0005 (training objective).

If you are going to contribute architecture: read all RFCs in numerical
order; they are mutually consistent and assume each other.

If you are going to contribute experiments: read RFC-0006 (data),
RFC-0007 (eval), and RFC-0009 (surprise scoring).

If you came for the cypherpunk angle: read RFC-0010 (on-device) and
RFC-0011 (verifiable inference). They are the two RFCs that make GenoLeWM
something more than "another bio model."

---

## Acknowledgments

- **Lucas Maes, Quentin Le Lidec, Damien Scieur, Yann LeCun, Randall Balestriero**
  for [LeWorldModel](https://github.com/lucas-maes/le-wm).
- **The Hugging Face Bio team, Zhongguancun Academy, TIGEM / Federico II**
  for [Carbon](https://huggingface.co/collections/HuggingFaceBio/carbon).
- **The CodeLeWM project** for the recipe of porting LeWM to a structured
  symbolic domain.

---

## License

Apache 2.0. See [LICENSE](LICENSE).
