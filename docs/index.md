# GenoLeWM

> **Action-conditioned JEPA world model for DNA, built on top of Carbon.**

[![CI](https://github.com/AbdelStark/GenoLeWM/actions/workflows/ci.yml/badge.svg)](https://github.com/AbdelStark/GenoLeWM/actions/workflows/ci.yml)
[![PyPI](https://img.shields.io/pypi/v/geno-lewm.svg?label=PyPI)](https://pypi.org/project/geno-lewm/)
[![Python](https://img.shields.io/pypi/pyversions/geno-lewm.svg)](https://pypi.org/project/geno-lewm/)
[![License](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](https://github.com/AbdelStark/GenoLeWM/blob/main/LICENSE)
[![Typed](https://img.shields.io/badge/typed-mypy--strict-blue.svg)](https://mypy.readthedocs.io/)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)

GenoLeWM treats **genetic edits as first-class actions**. A frozen DNA
foundation model (Carbon-500M by default) supplies a state vector for a
genomic window; a small trainable predictor head, conditioned on a
structured edit, predicts the post-edit state in the same latent space:

$$
\hat s_{t+1} = g(s_t, a) \qquad s_t = \mathrm{enc}(w_{\text{ref}}) \qquad a = \mathrm{action}(\text{EditSpec})
$$

That single equation unlocks:

- **Variant-effect prediction** at a fraction of Carbon's inference cost.
- **Multi-edit haplotype rollout** in latent space.
- **Planning** over edit sequences via latent MPC.
- **Surprise-based pathogenicity scoring** — predictor error as a signal.
- **On-device personal-genome inference** — Carbon-500M + a ~15M-param head fits on a laptop.

## Where to start

| If you have… | Read |
| --- | --- |
| 5 minutes | this page → [Quickstart](quickstart.md) |
| 30 minutes | [Specification index](spec/index.md) → [Architecture](spec/01-architecture.md) |
| an afternoon | the full [RFC corpus](https://github.com/AbdelStark/GenoLeWM/tree/main/rfcs) |
| a contribution to land | [Contributing](contributing.md) and the [implementation tracker](roadmap/IMPLEMENTATION.md) |

## What ships today

Phase 0 of the project shipped the **production infrastructure layer**:

- **Typed error hierarchy** with a stable code registry (RFC-0012).
- **Structured logging** with privacy redaction by default (RFC-0013).
- **Metrics registry** with Prometheus textfile export.
- **Canonical edit specs** (`EditSpec`, `RelEdit`) and pure-Python apply
  helpers, with property tests (RFC-0003).
- **Synthetic edit samplers** for the data pipeline (RFC-0006 §3.4).
- **Content-addressed attestation primitives** — manifests, receipts,
  input/output commitments, and the `geno-lewm-verify` CLI (RFC-0011).
- **Public-API stability decorators** (`experimental`, `deprecated`)
  with per-call-site deduplication (RFC-0014).
- **CI gates** — AST-level error / event / metric / network / print
  linters, plus a committed public-surface snapshot.
- **Optional PyTorch predictor module** — the base cross-attention
  `Predictor` with identity-at-init and public-surface coverage.
- **Carbon corpus window sampler** — RFC-0006 source-mix sampling,
  deterministic subsetting, and margin/stride window extraction.
- **Edit-balanced training sampler** — RFC-0005 edit-type weights and
  Phase-1 rollout-length mix.
- **Collapse-monitoring diagnostics** — RFC-0005 variance, pairwise
  distance, correlation, and KL-registry metrics with structured alerts.
- **Planning primitives** — RFC-0008 cost functions and factored
  `ActionSampler` for valid window-relative edits.
- **Surprise calibration primitives** — RFC-0009 region / GC / repeat
  labels, deterministic bucket IDs, sparse-bucket back-off, and
  `calibration.parquet` table building from pre-scored reference rows.
- **Deploy runtime contract** — RFC-0010 backend probing and
  fail-closed network guard for offline inference paths.

What's *not* yet shipped: trainer, autoregressive rollout, eval harness,
CEM solver, full surprise scorer, runtime scoring backends, ONNX / Core
ML / GGUF exporters. See the [roadmap](roadmap/IMPLEMENTATION.md).

## Acknowledgments

- **Lucas Maes, Quentin Le Lidec, Damien Scieur, Yann LeCun, Randall
  Balestriero** for [LeWorldModel](https://github.com/lucas-maes/le-wm).
- **The Hugging Face Bio team, Zhongguancun Academy, TIGEM / Federico
  II** for [Carbon](https://huggingface.co/collections/HuggingFaceBio/carbon).
- **The CodeLeWM project** for the recipe of porting LeWM to a
  structured symbolic domain.
