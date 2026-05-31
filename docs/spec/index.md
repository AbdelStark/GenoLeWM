# Specification

- **Version:** 0.1.0.dev0
- **Status:** Design phase. The reference infrastructure layer is
  implemented (errors, observability, attestation primitives, action
  specs, verify CLI), along with the optional-runtime base predictor
  and loss modules, the edit-balanced training sampler, collapse
  diagnostics, and the Carbon corpus window sampler; trainer /
  autoregressive rollout / eval / deploy are upcoming.

This file is the entry point into the GenoLeWM specification corpus.
Detailed content lives in two trees:

- This section (`docs/spec/`) — eleven canonical sections, normative
  for v0.1.
- The per-decision [RFC corpus](https://github.com/AbdelStark/GenoLeWM/tree/main/rfcs).

## Executive summary

GenoLeWM is an **action-conditioned Joint-Embedding Predictive Architecture
(JEPA) over DNA**. A pretrained DNA foundation model (Carbon-500M by
default, frozen) supplies a state vector for a contiguous genomic
window. A small trainable predictor head, conditioned on a structured
genomic edit, predicts the post-edit state in the same latent space.

```
ŝ_{t+1} = g(s_t, a)        s_t = enc(w_ref)        a = action(EditSpec)
```

That single equation unlocks: variant-effect prediction at a fraction
of Carbon's cost, multi-edit haplotype rollout, planning via CEM in
latent space, surprise-based pathogenicity scoring, on-device
deployment on consumer hardware, and a STARK-attested inference path
(Phase 4).

## Spec corpus

| # | File | Subject |
|---|------|---------|
| 00 | [overview](00-overview.md) | thesis, goals, non-goals, success criteria |
| 01 | [architecture](01-architecture.md) | module boundaries, runtime flows, invariants |
| 02 | [public-api](02-public-api.md) | Python and CLI surface, stability classes |
| 03 | [data-model](03-data-model.md) | types, schemas, on-disk formats |
| 04 | [error-model](04-error-model.md) | exception hierarchy, failure modes, exit codes |
| 05 | [observability](05-observability.md) | logging, metrics, tracing, redaction |
| 06 | [security](06-security.md) | threat model, trust boundaries, secrets |
| 07 | [testing-strategy](07-testing-strategy.md) | test pyramid, ML-specific tests, CI gates |
| 08 | [performance-budget](08-performance-budget.md) | latency / throughput / memory targets |
| 09 | [release-and-versioning](09-release-and-versioning.md) | semver, deprecation, changelog discipline |
| 10 | [glossary](10-glossary.md) | canonical terms |

## RFC corpus

19 RFCs covering the load-bearing decisions:

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
- RFC-0011 — attestation (receipts, STARK Phase 4).
- RFC-0012 — error taxonomy.
- RFC-0013 — observability and redaction.
- RFC-0014 — API stability policy.
- RFC-0015 — testing and CI gates.
- RFC-0016 — performance budget.
- RFC-0017 — configuration system.
- RFC-0018 — CLI design.
- RFC-0019 — reference desktop app skeleton.

The full index lives in the repository at
[`rfcs/README.md`](https://github.com/AbdelStark/GenoLeWM/blob/main/rfcs/README.md).

## Conflict resolution

If this index and a section disagree, the section wins.
If a section and an RFC disagree, the RFC wins.
If two RFCs disagree without an explicit `Supersedes` relationship, file
a reconciliation PR.

## What is not specified here

- Project history, contributor list, license text —
  [`LICENSE`](https://github.com/AbdelStark/GenoLeWM/blob/main/LICENSE),
  the [README](../index.md), [`CONTRIBUTING`](../contributing.md).
- Implementation roadmap and phase exit criteria —
  [`ROADMAP.md`](https://github.com/AbdelStark/GenoLeWM/blob/main/ROADMAP.md).
- Operational security disclosure — [`SECURITY`](../security.md).
- Open user-data privacy posture — [`PRIVACY`](../privacy.md).
- Implementation tracker — [`docs/roadmap/IMPLEMENTATION.md`](../roadmap/IMPLEMENTATION.md).
