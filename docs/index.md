# GenoLeWM

> **Action-conditioned JEPA world model for DNA, built on top of Carbon.**

[![CI](https://github.com/AbdelStark/GenoLeWM/actions/workflows/ci.yml/badge.svg)](https://github.com/AbdelStark/GenoLeWM/actions/workflows/ci.yml)
[![Status](https://img.shields.io/badge/status-alpha%20v0.2.1%20evidence-blue.svg)](roadmap/IMPLEMENTATION.md)
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
- **Local-first inference** over user-provided variant files using
  released or locally staged model artifacts.

## Where to start

| If you have… | Read |
| --- | --- |
| 5 minutes | this page → [Quickstart](quickstart.md) |
| 30 minutes | [Specification index](spec/index.md) → [Architecture](spec/01-architecture.md) |
| an afternoon | the full [RFC corpus](https://github.com/AbdelStark/GenoLeWM/tree/main/rfcs) |
| a contribution to land | [Contributing](contributing.md) and the [implementation tracker](roadmap/IMPLEMENTATION.md) |

## What ships today

The repository currently ships alpha code plus the public
`geno-lewm-v0.1.0-r1` paper/demo artifact set and the v0.2.1
serious-completion benchmark/planning/paper evidence bundle. Install the
Python package from PyPI after the `v0.2.1` tag workflow publishes, or
from source through the [Quickstart](quickstart.md).

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

The v0.1 publication evidence is public:

- Model: <https://huggingface.co/abdelstark/geno-lewm>
- Dataset: <https://huggingface.co/datasets/abdelstark/geno-lewm-data>
- Demo assets:
  <https://github.com/AbdelStark/GenoLeWM/releases/tag/geno-lewm-v0.1.0-r1>
- Paper:
  <https://github.com/AbdelStark/GenoLeWM/releases/download/geno-lewm-v0.1.0-r1/paper.md>
- Final binder:
  <https://huggingface.co/abdelstark/geno-lewm-runs/resolve/main/geno-lewm-coherent-cd2bfcc/publication/publication_evidence_report.json>

The v0.2.1 serious-completion evidence is also public:

- Artifact tree:
  <https://huggingface.co/abdelstark/geno-lewm-runs/tree/main/geno-lewm-v021-strong-4f36eef-10k-r1>
- Benchmark readiness:
  <https://huggingface.co/abdelstark/geno-lewm-runs/resolve/main/geno-lewm-v021-strong-4f36eef-10k-r1/suite/model/v0.2_benchmark_readiness_report.json>
- Planning demo:
  <https://huggingface.co/abdelstark/geno-lewm-runs/tree/main/geno-lewm-v021-strong-4f36eef-10k-r1/planning-demo>
- Serious-completion paper:
  <https://huggingface.co/abdelstark/geno-lewm-runs/resolve/main/geno-lewm-v021-strong-4f36eef-10k-r1/paper/paper.serious-completion.md>

What is *not* established yet: broad model quality beyond the narrow
chr21 ClinVar v0.1 slice, RFC-0004 rollout-speed closure, useful
multi-edit planning behavior, clinical utility, privacy assurance, or
runtime assurance beyond checksum provenance. See the
[roadmap](roadmap/IMPLEMENTATION.md) and v0.2 epic
[#197](https://github.com/AbdelStark/GenoLeWM/issues/197).

## Acknowledgments

- **Lucas Maes, Quentin Le Lidec, Damien Scieur, Yann LeCun, Randall
  Balestriero** for [LeWorldModel](https://github.com/lucas-maes/le-wm).
- **The Hugging Face Bio team, Zhongguancun Academy, TIGEM / Federico
  II** for [Carbon](https://huggingface.co/collections/HuggingFaceBio/carbon).
- **The CodeLeWM project** for the recipe of porting LeWM to a
  structured symbolic domain.
