# GenoLeWM — Specification (top-level index)

- **Version:** 0.1.0-draft
- **Date:** 2026-06-01
- **Status:** Alpha implementation. Infrastructure is implemented; the
  first real training run, model release, evaluation report, and
  terminal inference demo remain open.

This file is the entry point into the GenoLeWM specification corpus. The
detailed content lives in two trees:

- [`docs/spec/`](docs/spec/) — eleven canonical sections, normative for v0.1.
- [`rfcs/`](rfcs/) (mirrored at [`docs/rfcs/`](docs/rfcs/)) — per-decision RFCs.

The complementary documents [`SPECIFICATION.md`](SPECIFICATION.md),
[`ARCHITECTURE.md`](ARCHITECTURE.md), and [`ROADMAP.md`](ROADMAP.md) are
preserved at the repository root for the existing readership.

## Executive summary

GenoLeWM is an **action-conditioned Joint-Embedding Predictive Architecture
(JEPA) over DNA**. A pretrained DNA foundation model (Carbon-500M by default,
frozen) supplies a state vector for a contiguous genomic window. A small
trainable predictor head, conditioned on a structured genomic edit, predicts
the post-edit state in the same latent space.

```
ŝ_{t+1} = g(s_t, a)        s_t = enc(w_ref)        a = action(EditSpec)
```

That single equation unlocks: variant-effect prediction at a fraction of
Carbon's cost, multi-edit haplotype rollout, planning via CEM in latent
space, surprise-based pathogenicity scoring, on-device deployment on
consumer hardware, and checksum-based artifact provenance for releases.

## Spec corpus

| # | File | Subject |
|---|------|---------|
| 00 | [overview](docs/spec/00-overview.md) | thesis, goals, non-goals, success criteria |
| 01 | [architecture](docs/spec/01-architecture.md) | module boundaries, runtime flows, invariants |
| 02 | [public-api](docs/spec/02-public-api.md) | Python and CLI surface, stability classes |
| 03 | [data-model](docs/spec/03-data-model.md) | types, schemas, on-disk formats |
| 04 | [error-model](docs/spec/04-error-model.md) | exception hierarchy, failure modes, exit codes |
| 05 | [observability](docs/spec/05-observability.md) | logging, metrics, tracing, redaction |
| 06 | [security](docs/spec/06-security.md) | threat model, trust boundaries, secrets |
| 07 | [testing-strategy](docs/spec/07-testing-strategy.md) | test pyramid, ML-specific tests, CI gates |
| 08 | [performance-budget](docs/spec/08-performance-budget.md) | latency / throughput / memory targets |
| 09 | [release-and-versioning](docs/spec/09-release-and-versioning.md) | semver, deprecation, changelog discipline |
| 10 | [glossary](docs/spec/10-glossary.md) | canonical terms |

## RFC corpus

19 RFCs, indexed at [`rfcs/README.md`](rfcs/README.md). The load-bearing
decisions are:

- RFC-0001 — scope.
- RFC-0002 — state encoder (Carbon).
- RFC-0003 — action encoder (genomic edits).
- RFC-0004 — predictor (cross-attention Transformer).
- RFC-0005 — training objective (cosine + MSE; LeJEPA in Phase 2).
- RFC-0006 — data pipeline (corpus, edit mix, holdouts).
- RFC-0007 — evaluation suite (VEP, rollout, efficiency).
- RFC-0008 — planning (CEM).
- RFC-0009 — surprise scoring (calibrated per context).
- RFC-0010 — deployment (Apple Silicon, int4 / int8).
- RFC-0011 — artifact provenance and checksum receipts.
- RFC-0012 — error taxonomy.
- RFC-0013 — observability and redaction.
- RFC-0014 — API stability policy.
- RFC-0015 — testing and CI gates.
- RFC-0016 — performance budget.
- RFC-0017 — configuration system.
- RFC-0018 — CLI design.
- RFC-0019 — reference desktop app skeleton.

## Conflict resolution

If this index and a section disagree, the section wins.
If a section and an RFC disagree, the RFC wins.
If two RFCs disagree without an explicit `Supersedes` relationship, file
a reconciliation PR.

## What is not specified here

- Project history, contributor list, license text → [`LICENSE`](LICENSE),
  [`README.md`](README.md), [`CONTRIBUTING.md`](CONTRIBUTING.md).
- Implementation roadmap and phase exit criteria → [`ROADMAP.md`](ROADMAP.md).
- Operational security disclosure → [`SECURITY.md`](SECURITY.md).
- Open user-data privacy posture → [`PRIVACY.md`](PRIVACY.md).
- Implementation tracker (issue dashboard) → [`docs/roadmap/IMPLEMENTATION.md`](docs/roadmap/IMPLEMENTATION.md).
