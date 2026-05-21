# GenoLeWM

> **An action-conditioned JEPA world model for DNA, built on top of Carbon.**
> Genetic edits become first-class actions. The model learns latent transitions.
> Planning, surprise-based pathogenicity scoring, and on-device personal-genome
> inference fall out naturally.

[![CI](https://github.com/AbdelStark/GenoLeWM/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/AbdelStark/GenoLeWM/actions/workflows/ci.yml)
[![CodeQL](https://github.com/AbdelStark/GenoLeWM/actions/workflows/codeql.yml/badge.svg?branch=main)](https://github.com/AbdelStark/GenoLeWM/actions/workflows/codeql.yml)
[![Docs](https://github.com/AbdelStark/GenoLeWM/actions/workflows/docs.yml/badge.svg?branch=main)](https://abdelstark.github.io/GenoLeWM/)
[![PyPI](https://img.shields.io/pypi/v/geno-lewm.svg?label=PyPI)](https://pypi.org/project/geno-lewm/)
[![Python](https://img.shields.io/pypi/pyversions/geno-lewm.svg)](https://pypi.org/project/geno-lewm/)
[![License](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](LICENSE)
[![Typed: mypy --strict](https://img.shields.io/badge/typed-mypy--strict-blue.svg)](https://mypy.readthedocs.io/)
[![Linted: ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![pre-commit](https://img.shields.io/badge/pre--commit-enabled-brightgreen.svg)](https://github.com/pre-commit/pre-commit)

- 📖 **Docs:** <https://abdelstark.github.io/GenoLeWM/>
- 📘 **Spec:** [SPEC.md](SPEC.md)
- 🗺️ **Roadmap:** [ROADMAP.md](ROADMAP.md)
- 🔒 **Security:** [SECURITY.md](SECURITY.md) • [PRIVACY.md](PRIVACY.md)

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

- **State** `s_t ∈ ℝ^d` : a frozen Carbon embedding of a genomic window.
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

## Status

**Phase 1 — Infrastructure layer is implemented.** The Python package
ships the production substrate (typed errors, structured logging with
privacy redaction, metrics, action specs, attestation primitives, the
verify CLI). The trainer, predictor, and eval surfaces land
incrementally — see [ROADMAP.md](ROADMAP.md) and the
[implementation tracker](docs/roadmap/IMPLEMENTATION.md).

### What ships today

| Module | Status |
| --- | --- |
| `geno_lewm.errors` — typed exception hierarchy + code registry (RFC-0012) | ✅ stable |
| `geno_lewm.observability` — JSONL logger + event registry (RFC-0013) | ✅ stable |
| `geno_lewm._redaction` — privacy redaction filter (RFC-0013 §3.5) | ✅ stable |
| `geno_lewm.metrics` — registered metrics + Prometheus textfile export | ✅ stable |
| `geno_lewm.action` — `EditSpec` / `RelEdit`, `apply_edit`(s), synthetic samplers (RFC-0003 / RFC-0006) | ✅ stable |
| `geno_lewm.attestation` — manifest schema, hashing, commitments, receipts (RFC-0011) | ✅ stable |
| `geno_lewm.cli.verify` — `geno-lewm-verify` checksum-mode receipt verifier | ✅ stable |
| `geno_lewm.api` — `@experimental` / `@deprecated` lifetime decorators (RFC-0014) | ✅ stable |
| Trainer / predictor / scorer / planner / runtime | 🟡 designed, not yet implemented |

The implemented surface is **typed strictly** (mypy `--strict`),
**linted** (ruff with B/C4/UP/N/RUF/SIM/PIE/PTH/PL/PERF/FURB/LOG), and
covered by **417+ unit and property tests**.

---

## Install

GenoLeWM requires **Python 3.10+** and runs without any third-party
runtime dependency for the modules shipped today.

```bash
# uv (recommended)
uv pip install geno-lewm

# pip
pip install geno-lewm

# from source
git clone https://github.com/AbdelStark/GenoLeWM.git
cd GenoLeWM
uv venv && source .venv/bin/activate
uv pip install -e ".[dev]"
```

Optional extras pull in the ML stack: `geno-lewm[train]` (torch +
transformers + datasets), `geno-lewm[eval]` (pysam, cyvcf2),
`geno-lewm[deploy]` (onnx, onnxruntime), `geno-lewm[docs]`
(mkdocs-material + mkdocstrings).

---

## Quickstart

```python
from geno_lewm import EditSpec, apply_edit, RelEdit, EditType

# 1. Build a canonical edit (VCF-style 1-based coordinates).
snv = EditSpec(chrom="chr17", pos=43_091_983, ref="A", alt="T")

# 2. Re-anchor inside an encoder window (0-based inclusive bounds).
rel = snv.relative_to(43_091_900, 43_092_100)
print(rel.rel_pos)                 # 82

# 3. Apply pure-Python to build the s_{t+1} reference for training.
window = "ACGT" * 64
edited = apply_edit(window, RelEdit(0, EditType.SNV, "A", "C"))
```

```bash
# Verify a receipt produced by someone else.
geno-lewm-verify path/to/receipt.json --manifest path/to/manifest.json
```

For the full tour, see the [quickstart guide](https://abdelstark.github.io/GenoLeWM/quickstart/).

---

## Repository layout

```
GenoLeWM/
├── README.md                      # this file
├── SPEC.md                        # top-level specification index
├── SPECIFICATION.md               # synthesized canonical view
├── ARCHITECTURE.md                # high-level walkthrough
├── ROADMAP.md                     # phases, owners, dates
├── LICENSE                        # Apache 2.0
├── CHANGELOG.md                   # Keep a Changelog + SemVer
├── pyproject.toml                 # package metadata, ruff, mypy, pytest
├── mkdocs.yml                     # docs site config
├── .pre-commit-config.yaml        # mirrors every CI gate
├── geno_lewm/                     # package source (typed, strict)
│   ├── action/                    # EditSpec / RelEdit / apply / synthetic samplers
│   ├── attestation/               # manifest, hashing, commitments, receipts
│   ├── cli/                       # geno-lewm-verify
│   ├── api.py                     # @experimental / @deprecated decorators
│   ├── errors.py                  # exception hierarchy + ERROR_CODES
│   ├── observability.py           # JSONL logger + EVENTS registry
│   ├── metrics.py                 # METRICS registry + Prometheus exporter
│   └── _redaction.py              # private redaction filter
├── tests/                         # unit / property / api / lint
├── tools/
│   ├── api/snapshot.py            # public-surface snapshot (CI-gated)
│   └── lint/                      # AST linters (errors / events / no-print / network)
├── docs/                          # mkdocs source tree (rfcs / spec / api / reference)
├── rfcs/                          # 19 design RFCs, numbered
└── examples/                      # notebooks (Phase 0: placeholders)
```

---

## How to read this repo

If you have 5 minutes: read this README, then skim [SPEC.md](SPEC.md).

If you have 30 minutes: read SPEC.md end-to-end, then
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

## Development

```bash
# install dev extras
uv pip install -e ".[dev]"

# install pre-commit (matches CI gates exactly)
pre-commit install

# the full quality gate
ruff format --check geno_lewm tools tests
ruff check geno_lewm tools tests
mypy geno_lewm tools
python -m tools.lint.check_error_codes
python -m tools.lint.check_event_names
python -m tools.lint.check_no_print
python -m tools.lint.check_network_confined
python -m tools.api.snapshot check
pytest -n auto --cov=geno_lewm --cov-branch
```

See [CONTRIBUTING.md](CONTRIBUTING.md) for the RFC process, the test
pyramid, and the review discipline.

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
