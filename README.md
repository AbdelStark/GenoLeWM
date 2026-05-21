<div align="center">

# GenoLeWM

**An action-conditioned JEPA world model for DNA, built on top of [Carbon](https://huggingface.co/collections/HuggingFaceBio/carbon).**

*Genetic edits become first-class actions. The model learns latent transitions.
Variant scoring, multi-edit haplotype rollout, planning, surprise-based
pathogenicity scoring, and on-device personal-genome inference fall out
of one equation.*

```
ŝ_{t+1} = g(s_t, a)        s_t = enc(w_ref)        a = action(EditSpec)
```

[![CI](https://github.com/AbdelStark/GenoLeWM/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/AbdelStark/GenoLeWM/actions/workflows/ci.yml)
[![CodeQL](https://github.com/AbdelStark/GenoLeWM/actions/workflows/codeql.yml/badge.svg?branch=main)](https://github.com/AbdelStark/GenoLeWM/actions/workflows/codeql.yml)
[![Docs](https://github.com/AbdelStark/GenoLeWM/actions/workflows/docs.yml/badge.svg?branch=main)](https://abdelstark.github.io/GenoLeWM/)
[![PyPI](https://img.shields.io/pypi/v/geno-lewm.svg?label=PyPI)](https://pypi.org/project/geno-lewm/)
[![Python](https://img.shields.io/pypi/pyversions/geno-lewm.svg)](https://pypi.org/project/geno-lewm/)
[![License](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](LICENSE)
[![Typed: mypy --strict](https://img.shields.io/badge/typed-mypy--strict-blue.svg)](https://mypy.readthedocs.io/)
[![Linted: ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![Coverage: 95%](https://img.shields.io/badge/coverage-%E2%89%A595%25-brightgreen.svg)](#engineering-discipline)
[![pre-commit](https://img.shields.io/badge/pre--commit-enabled-brightgreen.svg)](https://github.com/pre-commit/pre-commit)

[**Documentation**](https://abdelstark.github.io/GenoLeWM/)
 · [**Specification**](SPEC.md)
 · [**RFCs**](rfcs/)
 · [**Roadmap**](ROADMAP.md)
 · [**Architecture**](ARCHITECTURE.md)
 · [**Privacy**](PRIVACY.md)

</div>

---

## Why GenoLeWM

[Carbon](https://huggingface.co/collections/HuggingFaceBio/carbon) is an
autoregressive DNA foundation model that scores variants via `logP(alt) − logP(ref)`.
That formulation works, but it has two structural limits:

1. **Edits are not inputs.** Every variant scoring call is a full re-encoding of
   `ref` and `alt`. The model has no representation of *an edit*.
2. **No predictive latent dynamics.** You cannot roll out *"what if I apply
   these three edits in sequence"* without re-running the encoder on every
   intermediate sequence.

[LeWorldModel (Maes et al.)](https://github.com/lucas-maes/le-wm) gives a
recipe for action-conditioned predictive coding in latent space: a stable,
end-to-end JEPA with two losses (next-embedding prediction + isotropic-Gaussian
regularizer), no EMA, no teacher network, ~15M trainable parameters.

> **GenoLeWM = LeWorldModel × Carbon × genomic edits.**

| Symbol | Meaning |
|---|---|
| `s_t ∈ ℝ^d` | a frozen Carbon embedding of a genomic window (the **state**) |
| `a_t` | a structured genetic edit (SNV, indel, MNV); the **action** |
| `g(s_t, a_t) → ŝ_{t+1}` | a small trainable **predictor**, target `enc(edited_window)` |

That single change unlocks:

- **Variant-effect prediction** at a fraction of Carbon's per-variant cost.
- **Multi-edit haplotype rollout** in latent space: compose actions without ever decoding back to DNA.
- **Planning** over edit sequences via latent MPC (e.g. *minimal edit set to restore a reference-like neighborhood*).
- **Surprise-based pathogenicity scoring**: predictor residual `‖ŝ_{t+1} − s_{t+1}‖` as an unsupervised signal.
- **On-device personal-genome inference**: Carbon-500M + a ~15M-parameter GenoLeWM head fits on a laptop.
- **Verifiable inference**: content-addressed manifests, input/output commitments, and (Phase 4) STARK-proven forward passes.

---

## Architecture at a glance

```
        ┌──────────────────────────────────────────────────────────┐
        │                     GenoLeWM Runtime                      │
        └──────────────────────────────────────────────────────────┘

  w_ref ──►┌────────────────────────┐                  ┌────────────┐
  (12 kbp) │ Carbon-500M (frozen)   │── s_t ─────────►│            │
           │ layer L, centered mean │                  │ Predictor  │── ŝ_{t+1}
           └────────────────────────┘                  │  g(s, a)   │       │
                                                       │  ~20 M θ   │       │
  EditSpec ─►┌────────────────────┐                    │            │       │
  (chrom,    │ Action encoder     │── a_emb ──────────►│            │       │
   pos,      │ sin-pos + type     │                    └────────────┘       │
   ref,alt)  │ + base emb + MLP   │                                         │
             └────────────────────┘                                         │
                                                                            │
                                       ┌────────────────────────────────────┤
                                       │                                    │
                                       ▼                                    ▼
                              latent rollout                       surprise score
                              (apply next edit)               ‖ŝ_{t+1} − s_{t+1}‖
                              ──► planning (CEM)              ──► VEP, calibration
```

Three properties drop out of the design:

1. **The encoder is the heavy thing.** Carbon dominates compute, memory, and
   energy. Everything else in the diagram is small.
2. **The predictor is the only trainable thing in Phase 1.** Carbon stays frozen;
   the action encoder + predictor together are ~25–30M parameters.
3. **Once you have `ŝ_{t+1}`, every downstream use is cheap.** Surprise is a
   subtraction. Rollout is another predictor call. Planning is CEM over the
   predictor. None of these require a second Carbon pass.

Detailed walkthrough: [`ARCHITECTURE.md`](ARCHITECTURE.md). Module boundaries
and runtime data flow: [`docs/spec/01-architecture.md`](docs/spec/01-architecture.md).

---

## Status

**Phase 1: the production infrastructure layer is implemented.**
The training, predictor, eval, and deployment surfaces land incrementally;
see [ROADMAP.md](ROADMAP.md) and the [implementation tracker](docs/roadmap/IMPLEMENTATION.md).

| Module | RFC | Status |
| --- | --- | --- |
| `geno_lewm.errors`: typed exception hierarchy + code registry | [RFC-0012](rfcs/0012-error-taxonomy.md) | ✅ stable |
| `geno_lewm.observability`: JSONL logger + event registry | [RFC-0013](rfcs/0013-observability.md) | ✅ stable |
| `geno_lewm._redaction`: privacy redaction filter | [RFC-0013 §3.5](rfcs/0013-observability.md) | ✅ stable |
| `geno_lewm.metrics`: registered metrics + Prometheus textfile export | [RFC-0013](rfcs/0013-observability.md) | ✅ stable |
| `geno_lewm.action`: `EditSpec` / `RelEdit`, `apply_edit`(s), synthetic samplers | [RFC-0003](rfcs/0003-action-representation-genomic-edits.md), [RFC-0006](rfcs/0006-data-pipeline.md) | ✅ stable |
| `geno_lewm.attestation`: manifest schema, hashing, commitments, receipts | [RFC-0011](rfcs/0011-verifiable-inference-attestation.md) | ✅ stable |
| `geno_lewm.cli.verify`: `geno-lewm-verify` checksum-mode receipt verifier | [RFC-0011](rfcs/0011-verifiable-inference-attestation.md) | ✅ stable |
| `geno_lewm.api`: `@experimental` / `@deprecated` lifetime decorators | [RFC-0014](rfcs/0014-public-api-and-stability.md) | ✅ stable |
| `encoder/`, `predictor/`, `data/`, `eval/`, `planning/`, `surprise/`, `deploy/` | [RFC-0002](rfcs/0002-state-encoder-carbon-integration.md)–[RFC-0010](rfcs/0010-on-device-personal-genome-deployment.md) | 🟡 designed, landing |

### Phase plan

| Phase | Goal | Headline target | Status |
|---|---|---|---|
| **0. Design** | Lock the spec and 19 RFCs | All RFCs `Accepted` | ✅ shipped |
| **1. Minimum viable predictor** | End-to-end SNV pipeline on Carbon-500M | ≥ 0.80 AUROC, ClinVar coding | 🚧 in progress |
| **2. Full edits + planning** | SNV+INS+DEL+MNV, LoRA, CEM planner, calibrated surprise | ≥ Carbon-500M zero-shot AUROC | ⏳ designed |
| **3. On-device** | ONNX / Core ML / GGUF, int4/int8, desktop app skeleton | < 200 ms / variant on M3 Max | ⏳ designed |
| **4. Verifiable inference** | STARK proof of the predictor forward pass | proof gen < 5 min, verify < 1 s | ⏳ designed |

See [`ROADMAP.md`](ROADMAP.md) for exit criteria, durations, and risks per phase.

---

## Installation

GenoLeWM requires **Python 3.10+**. The implemented surface has *zero* runtime
third-party dependencies; heavier ML stacks are gated behind optional extras.

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

| Extra | Pulls in | When you need it |
|---|---|---|
| `geno-lewm[train]` | `torch`, `transformers`, `datasets`, `accelerate`, … | training the predictor |
| `geno-lewm[eval]`  | `pysam`, `cyvcf2`, `scikit-learn`, `scipy` | running the evaluation suite |
| `geno-lewm[deploy]` | `onnx`, `onnxruntime` | exporting and on-device inference |
| `geno-lewm[docs]`  | `mkdocs-material`, `mkdocstrings` | building the docs site locally |
| `geno-lewm[dev]`   | `pytest`, `hypothesis`, `ruff`, `mypy`, … | contributor workflow |
| `geno-lewm[all]`   | `train` + `eval` + `deploy` | everything except docs and dev |

---

## Quickstart

A five-minute tour of what ships today. The full tour lives at
[**abdelstark.github.io/GenoLeWM/quickstart/**](https://abdelstark.github.io/GenoLeWM/quickstart/).

### 1. Build and apply a canonical edit

```python
from geno_lewm import EditSpec, EditType, RelEdit, apply_edit, apply_edits

# VCF-style: 1-based pos, explicit ref / alt bases. EditType is derived.
snv = EditSpec(chrom="chr17", pos=43_091_983, ref="A", alt="T")
assert snv.edit_type is EditType.SNV

# Re-anchor inside an encoder window (0-based inclusive bounds).
rel = snv.relative_to(window_start_bp=43_091_900, window_end_bp=43_092_100)
print(rel.rel_pos)  # 82

# Pure-Python apply, used to build the s_{t+1} target during training.
window = "ACGT" * 64
edited = apply_edit(window, RelEdit(0, EditType.SNV, "A", "C"))

# Multi-edit composition is order-invariant; edits are sorted right-to-left internally.
edited_haplotype = apply_edits(window, [
    RelEdit(rel_pos=0,  edit_type=EditType.SNV, ref_bases="A", alt_bases="T"),
    RelEdit(rel_pos=4,  edit_type=EditType.SNV, ref_bases="A", alt_bases="C"),
])
```

Validation is strict and typed: bad input raises a subclass of `GenoLeWMError`
with a stable error code (`INPUT.INVALID_EDIT`, `INPUT.OUT_OF_WINDOW`, …) and a
machine-readable `details` payload. See [`docs/spec/04-error-model.md`](docs/spec/04-error-model.md).

### 2. Structured logging with privacy redaction

```python
from geno_lewm import get_logger

log = get_logger("inference", run_id="run-42")
log.info("inference.batch.end", n=10, batch_id="b-1", throughput_per_s=87.2)
# `sample_id` would be denied even with an allowlist hit; strict-mode raises.
```

JSONL on a pipe, pretty on a TTY. Four redaction rules (allowlist / type /
DNA pattern / personal-data deny-list) fire on every payload. Configurable via
`GENO_LEWM_LOG_*` environment variables.

### 3. Verifiable inference primitives

```python
from geno_lewm import (
    EditSpec, PoolingConfig, DtypeConfig,
    compute_input_commitment,
)

edit  = EditSpec(chrom="1", pos=10, ref="A", alt="T")
pool  = PoolingConfig(state_layer=12, pool_type="centered_mean",
                      pool_radius=64, normalize=True)
dtype = DtypeConfig(encoder_dtype="bf16", predictor_dtype="bf16")

window = "ACGT" * 64
print(compute_input_commitment(window, edit, pool, dtype))
# 'sha256:0123…' : byte-stable, content-addressed, reproducible
```

### 4. The verify CLI

```console
$ geno-lewm-verify path/to/receipt.json --manifest path/to/manifest.json
reading receipt:  path/to/receipt.json
  schema_version=1.0.0 attestation.kind=checksum_only
reading manifest: path/to/manifest.json
  model_id ok (sha256:0123456789abcdef0…)
  input_commitment: skipped (no input flags supplied)
  output_commitment ok (sha256:fedcba9876543210…)
ok
```

Exit codes follow [`docs/spec/04-error-model.md`](docs/spec/04-error-model.md):
`0` = verified, `8` = attestation mismatch, etc.

---

## Performance targets

Performance is part of the public contract: these are commitments, not
aspirations. A release that misses any of them is not shippable as v0.1 without
an explicit RFC amendment. Full table: [`docs/spec/08-performance-budget.md`](docs/spec/08-performance-budget.md).

| Operation | H100 | RTX 4090 | M3 Max | CPU-only |
|---|---:|---:|---:|---:|
| Single-variant scoring (warm cache) | < 5 ms | < 20 ms | < 200 ms | < 1.5 s |
| Single-variant scoring (cold; Carbon call) | < 50 ms | < 100 ms | < 800 ms | < 6 s |
| Predictor forward pass (bf16) | < 1 ms | < 3 ms | < 25 ms | < 200 ms |
| 100k-variant VCF scoring | < 1 min | < 5 min | < 30 min | n/a |
| Predictor + Carbon-500M (bf16) memory | < 3 GB | < 3 GB | < 8 GB | n/a |
| Predictor + Carbon-500M (int4) memory | n/a | n/a | < 1 GB | n/a |

---

## Repository layout

```
GenoLeWM/
├── geno_lewm/                  # package source (typed, strict)
│   ├── action/                 # EditSpec / RelEdit / apply / synthetic samplers
│   ├── attestation/            # manifest, hashing, commitments, receipts
│   ├── cli/                    # geno-lewm-verify (more CLIs land in Phase 1–3)
│   ├── api.py                  # @experimental / @deprecated decorators
│   ├── errors.py               # exception hierarchy + ERROR_CODES
│   ├── observability.py        # JSONL logger + EVENTS registry
│   ├── metrics.py              # METRICS registry + Prometheus exporter
│   └── _redaction.py           # privacy redaction filter
├── tests/
│   ├── unit/                   # pure unit tests
│   ├── property/               # Hypothesis property-based tests
│   ├── lint/                   # AST-gate tests
│   └── api/                    # public-surface snapshot test
├── tools/
│   ├── api/snapshot.py         # public-surface snapshot (CI-gated)
│   ├── lint/                   # AST linters: errors / events / no-print / network / licenses
│   └── release/                # PEP 440 version bumper + changelog synthesizer
├── docs/                       # mkdocs source (rfcs / spec / api / reference)
├── rfcs/                       # 19 numbered design RFCs
├── examples/                   # notebooks (placeholders during Phase 0)
├── SPEC.md                     # top-level specification index
├── SPECIFICATION.md            # synthesized canonical view
├── ARCHITECTURE.md             # narrative architecture walkthrough
├── ROADMAP.md                  # phases, owners, dates
├── CHANGELOG.md                # Keep a Changelog 1.1.0 + SemVer 2.0
├── pyproject.toml              # package metadata, ruff, mypy, pytest
├── mkdocs.yml                  # docs site config
├── .pre-commit-config.yaml     # mirrors every CI gate
└── Makefile                    # developer ergonomics (`make help`)
```

---

## How to read this repo

| If you have… | Path |
| --- | --- |
| **5 minutes** | this README, then skim [`SPEC.md`](SPEC.md) |
| **30 minutes** | [`SPEC.md`](SPEC.md) end-to-end → [`ARCHITECTURE.md`](ARCHITECTURE.md) → RFC-0001 (scope), RFC-0005 (training objective) |
| **an afternoon** | the full [RFC corpus](rfcs/) in numerical order; they are mutually consistent and assume each other |
| **contributing experiments** | RFC-0006 (data) → RFC-0007 (eval) → RFC-0009 (surprise) |
| **the on-device + verifiable angle** | RFC-0010 (on-device) → RFC-0011 (verifiable inference) |

---

## Engineering discipline

Every gate below runs on every PR; `make ci` is the single command that rehearses
the full pipeline locally.

| Gate | Tool | Policy |
|---|---|---|
| Formatting | `ruff format --check` | zero diff |
| Linting | `ruff check` | `E, W, F, I, B, C4, UP, N, RUF, SIM, PIE, PTH, TID, ARG, PL, PERF, FURB, LOG, ASYNC`; zero findings |
| Typing | `mypy --strict` | strict mode across `geno_lewm/` and `tools/`; zero errors |
| Tests | `pytest -n auto` | 500+ unit, property, lint, and public-surface tests |
| Coverage | `pytest --cov --cov-branch` | branch coverage ≥ 95% on the implemented surface |
| Public API | `tools/api/snapshot.py` | committed snapshot; any change is a deliberate PR |
| Error codes | `tools/lint/check_error_codes.py` | every raised error has a registered code |
| Log events | `tools/lint/check_event_names.py` | every emitted event is in the registry |
| Network | `tools/lint/check_network_confined.py` | fail-closed: no `urllib` / `requests` / `httpx` outside allowlisted modules |
| Print | `tools/lint/check_no_print.py` | no `print()` in library code; use the logger |
| License | `tools/lint/check_license_headers.py` | SPDX header on every source file |
| Build | `python -m build && twine check` | sdist + wheel build clean |
| Docs | `mkdocs build --strict` | docs fail the build on any warning |
| CI matrix | GitHub Actions | Python 3.10 / 3.11 / 3.12 / 3.13 × Linux / macOS / Windows |
| Security | CodeQL + Dependabot + OSSF Scorecard | weekly + per-PR; trusted-publisher PyPI releases |

```bash
# After cloning:
make install                # editable install with [dev] extras
make hooks                  # mirror CI gates into pre-commit
make ci                     # format-check + lint + types + gates + tests + docs
```

Individual targets (`make test`, `make types`, `make lint-fix`, `make docs-serve`, …)
are documented under `make help`.

---

## Contributing

We welcome contributions, especially RFC reviews during Phase 0–1 and
implementation PRs against the modules listed as designed-not-yet-implemented.

- The PR template, RFC process, and review discipline are in [`CONTRIBUTING.md`](CONTRIBUTING.md).
- The expected behavior in project spaces is in [`CODE_OF_CONDUCT.md`](CODE_OF_CONDUCT.md).
- Security reports go through GitHub Security Advisories; see [`SECURITY.md`](SECURITY.md).
- Privacy guarantees and how they are enforced are in [`PRIVACY.md`](PRIVACY.md).

The implementation tracker (issues, owners, and the current open-question
registry) is at [`docs/roadmap/IMPLEMENTATION.md`](docs/roadmap/IMPLEMENTATION.md).

---

## Citation

If GenoLeWM contributes to academic work, please cite the project alongside
its two intellectual parents.

```bibtex
@software{genolewm2026,
  title  = {{GenoLeWM}: An action-conditioned {JEPA} world model for {DNA}},
  author = {{GenoLeWM Authors}},
  year   = {2026},
  url    = {https://github.com/AbdelStark/GenoLeWM},
  note   = {Apache-2.0},
}
```

---

## Acknowledgments & related work

GenoLeWM stands on two pieces of prior work:

- **[LeWorldModel](https://github.com/lucas-maes/le-wm)** by Lucas Maes, Quentin
  Le Lidec, Damien Scieur, Yann LeCun, and Randall Balestriero, for the stable
  end-to-end JEPA training recipe (LeJEPA) that GenoLeWM specializes to the
  symbolic / genomic domain.
- **[Carbon](https://huggingface.co/collections/HuggingFaceBio/carbon)** by the
  Hugging Face Bio team, Zhongguancun Academy, and TIGEM / Federico II, for the
  autoregressive DNA foundation model that serves as the frozen state encoder.

And on the recipe of porting LeWM to a structured symbolic domain pioneered
by the **CodeLeWM** project, the sibling project for source code.

GenoLeWM is independent of both groups and any errors here are ours.

---

## Safety statement

GenoLeWM is a **research tool**. Its output is a research signal, not a clinical
diagnosis. The runtime is local-first, fails closed on network calls (allow-list
only Hugging Face Hub for first-run downloads), and never logs variant bases.
Clinical decision-making, embryo selection, and any human reproductive use are
**explicitly out of scope**; see [`docs/spec/06-security.md`](docs/spec/06-security.md)
and [`PRIVACY.md`](PRIVACY.md). If a variant in a GenoLeWM scoring report
concerns you, talk to a qualified genetic counselor.

---

## License

GenoLeWM is released under the **Apache License 2.0**. See [`LICENSE`](LICENSE).
