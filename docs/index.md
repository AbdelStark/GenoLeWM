# GenoLeWM

> **Action-conditioned JEPA world model for DNA, built on top of Carbon.**

[![CI](https://github.com/AbdelStark/GenoLeWM/actions/workflows/ci.yml/badge.svg)](https://github.com/AbdelStark/GenoLeWM/actions/workflows/ci.yml)
[![Status](https://img.shields.io/badge/status-alpha%20pre--release-orange.svg)](roadmap/IMPLEMENTATION.md)
[![Python](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://github.com/AbdelStark/GenoLeWM/blob/main/pyproject.toml)
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

The research hypothesis is that this can support:

- **Variant-effect prediction** with fewer Carbon passes once trained.
- **Multi-edit haplotype rollout** in latent space.
- **Planning** over edit sequences via latent MPC.
- **Surprise-based pathogenicity scoring** — predictor error as a signal.
- **Local-first inference** over user-provided variant files once public
  checkpoints and runtime artifacts exist.

## Where to start

| If you have… | Read |
| --- | --- |
| 5 minutes | this page → [Quickstart](quickstart.md) |
| 30 minutes | [Specification index](spec/index.md) → [Architecture](spec/01-architecture.md) |
| an afternoon | the full [RFC corpus](https://github.com/AbdelStark/GenoLeWM/tree/main/rfcs) |
| a contribution to land | [Contributing](contributing.md) and the [implementation tracker](roadmap/IMPLEMENTATION.md) |

## What ships today

The repository currently ships local contracts and release tooling, not
paper results. Install from source through the [Quickstart](quickstart.md)
until the first PyPI tag is cut.

- **Core Python surface:** typed errors, privacy-aware structured logs,
  metrics, canonical edit specs, pure-Python edit application,
  `ActionEncoder`, `Predictor`, `ARPredictor`, surprise scoring, and
  local-only personal-genome importers.
- **Data and training contracts:** Carbon window sampling, gnomAD and
  ClinVar VCF-to-Parquet prep commands, tuple-builder source-mix and
  holdout rules, `GenoLeWMDataset`, fixture smoke training, Carbon
  preflight, and a preflight-gated Carbon-backed trainer launcher.
- **Evaluation and release contracts:** checksum manifests/receipts,
  `geno-lewm-score`, `geno-lewm-verify`, Carbon zero-shot baseline
  scoring, measured metrics aggregation, efficiency-report generation,
  terminal-demo transcript generation, dataset/model/paper package
  verifiers, Hub dry-run/publish helpers, clean-machine replay, and
  final publication-evidence binding.
- **Project guardrails:** public API snapshot tests, duplicate-free
  `__all__` checks, source-language linting for de-scoped trust claims,
  release-blocker issue references, and strict docs rendering.

What is *not* paper-ready yet: no GenoLeWM checkpoint or dataset snapshot
has been published, no Carbon-backed training run has completed, no
paper-grade measured evaluation exists, and no clean-machine terminal
demo has replayed from released public artifacts. See the
[roadmap](roadmap/IMPLEMENTATION.md) and live release blockers
[#101](https://github.com/AbdelStark/GenoLeWM/issues/101),
[#163](https://github.com/AbdelStark/GenoLeWM/issues/163),
[#164](https://github.com/AbdelStark/GenoLeWM/issues/164),
[#165](https://github.com/AbdelStark/GenoLeWM/issues/165),
[#166](https://github.com/AbdelStark/GenoLeWM/issues/166), and
[#167](https://github.com/AbdelStark/GenoLeWM/issues/167).

## Acknowledgments

- **Lucas Maes, Quentin Le Lidec, Damien Scieur, Yann LeCun, Randall
  Balestriero** for [LeWorldModel](https://github.com/lucas-maes/le-wm).
- **The Hugging Face Bio team, Zhongguancun Academy, TIGEM / Federico
  II** for [Carbon](https://huggingface.co/collections/HuggingFaceBio/carbon).
- **The CodeLeWM project** for the recipe of porting LeWM to a
  structured symbolic domain.
