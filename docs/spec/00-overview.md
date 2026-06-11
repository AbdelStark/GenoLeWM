# 00 — Overview

- Status: Authoritative for v0.1
- Companion RFC: [RFC-0001](../rfcs/0001-project-scope-and-goals.md)
- Last reviewed: 2026-06-06

## Thesis

GenoLeWM is an **action-conditioned Joint-Embedding Predictive Architecture
(JEPA) over DNA**. A pretrained DNA foundation model (Carbon-500M by default,
frozen) supplies a state vector for a contiguous genomic window. A small
trainable predictor head, conditioned on a structured genomic edit, predicts
the post-edit state in the same latent space. Variant scoring, multi-edit
rollout, planning, and unsupervised pathogenicity scoring all reduce to
operations over that predictor.

The architectural contract is one sentence:

> `ŝ_{t+1} = g(s_t, a)` where `s_t = enc(w_ref)`, `a = action(EditSpec)`,
> and the encoder `enc` is frozen.

Every other system commitment is downstream of that single equation.

## Goals (v1)

The public artifact set now includes the v0.2.1 package, model and
dataset artifacts, benchmark/planning/paper tree, and generated paper.
The goals below are roadmap targets unless explicitly tied to a measured
release artifact; v0.1 established a narrow first baseline, and v0.2.1
adds broader but still mixed or negative evidence rather than broad
model-quality proof.

1. **Per-edit latent prediction** at < 10% of Carbon's per-variant inference
   cost after caching reference embeddings.
2. **Variant-effect prediction** matching or exceeding Carbon-500M zero-shot
   AUROC on ClinVar coding and non-coding benchmarks.
3. **Multi-edit haplotype rollout** in latent space with cosine similarity
   ≥ 0.85 against held-out single-edit windows and ≥ 0.80 against held-out
   three-edit haplotypes (Phase 2).
4. **Latent planning** via cross-entropy search over discrete edits, returning
   ordered edit lists for user-specified target states (Phase 2).
5. **Surprise-based pathogenicity scoring** via per-context calibrated
   predictor residuals — no supervised classifier training.
6. **On-device deployment target** on Apple Silicon (M3 Max baseline) with
   measured release gates of single-variant scoring < 200 ms and full-VCF
   scoring of 100k variants < 30 minutes (Phase 3).
7. **Artifact provenance hooks** in release and demo paths —
   content-addressed model identifiers, input commitments, and output
   receipts.

## Non-goals (v1)

1. Pretraining a new DNA foundation model. Carbon is the encoder.
2. Decoding latents back to DNA bases. Carbon does this; GenoLeWM does not.
3. Clinical decision support. Output is a research signal.
4. Protein-structure prediction. Out of scope; nucleotide-only latent.
5. Multi-omics fusion (RNA-seq, ATAC-seq, methylation). Deferred to v2.
6. Hosted inference service. Local-first; community may build hosted
   variants from the open weights.
7. Germline-edit reproductive use. Excluded by safety frame; see
   [§06-security](06-security.md).

## Success criteria

A release is shippable when, jointly:

- The reference checkpoint clears all eval gates in
  [`docs/spec/07-testing-strategy.md`](07-testing-strategy.md) and the
  per-track targets in [`docs/spec/08-performance-budget.md`](08-performance-budget.md).
- A user can install the runtime, score a VCF on a laptop, and produce a
  checksum receipt that can be checked against the released model manifest.
- Every RFC in [`docs/rfcs/`](../rfcs/) is at status `Accepted` or
  `Superseded`.
- Every public surface enumerated in [`02-public-api.md`](02-public-api.md)
  is versioned per the policy in [`09-release-and-versioning.md`](09-release-and-versioning.md).
- Every error in [`04-error-model.md`](04-error-model.md) is raised with the
  documented exception type and observable per [`05-observability.md`](05-observability.md).

## Audience

| Audience | Primary use |
|----------|-------------|
| DNA-foundation-model researchers | predictor head on Carbon (or analogous encoder) |
| Bioinformatics tool builders | local-first variant scoring with reproducible receipts |
| Personal-genomics enthusiasts | desktop app for personal variant exploration |
| Reproducible-ML engineers | dataset, model, evaluation, and demo artifact provenance |
| Clinical-genomics researchers | fast first-pass screening tool (research only) |

The system is explicitly **not** for clinical decision-making, embryo
selection, or any human reproductive use.

## How this corpus is organized

| File | Role |
|------|------|
| [`SPEC.md`](index.md) | top-level index and executive summary |
| [`SPECIFICATION.md`](https://github.com/AbdelStark/GenoLeWM/blob/main/SPECIFICATION.md) | synthesized canonical view (legacy entry point) |
| [`docs/spec/00-overview.md`](00-overview.md) | thesis, goals, non-goals, success criteria (this file) |
| [`docs/spec/01-architecture.md`](01-architecture.md) | system architecture, module boundaries, data flow |
| [`docs/spec/02-public-api.md`](02-public-api.md) | public Python / CLI / runtime surface |
| [`docs/spec/03-data-model.md`](03-data-model.md) | types, schemas, on-disk formats, invariants |
| [`docs/spec/04-error-model.md`](04-error-model.md) | exception hierarchy, failure modes, recovery |
| [`docs/spec/05-observability.md`](05-observability.md) | logging, metrics, tracing, redaction |
| [`docs/spec/06-security.md`](06-security.md) | threat model, trust boundaries, secrets handling |
| [`docs/spec/07-testing-strategy.md`](07-testing-strategy.md) | test pyramid, ML-specific tests, CI gates |
| [`docs/spec/08-performance-budget.md`](08-performance-budget.md) | latency/throughput/memory targets, profiling |
| [`docs/spec/09-release-and-versioning.md`](09-release-and-versioning.md) | semver policy, deprecation, changelog discipline |
| [`docs/spec/10-glossary.md`](10-glossary.md) | canonical terms (also see [`docs/glossary.md`](../glossary.md)) |
| [`docs/rfcs/`](../rfcs/) | per-decision RFCs (also at [`rfcs/`](../rfcs/) — root copy is canonical) |
| [`docs/roadmap/IMPLEMENTATION.md`](../roadmap/IMPLEMENTATION.md) | issue tracker dashboard |

When this document and an RFC disagree, the RFC wins. When two RFCs
disagree, the higher-numbered one wins if it explicitly supersedes;
otherwise file a reconciliation PR.

## Out-of-scope reminders

- No fabricated benchmarks. Numbers ship only after the full eval suite runs.
- Existing public numbers are measured results and negative findings;
  improved-results claims need their own measured evidence.
- No `TBD` shipped: every uncertainty is either decided or promoted to
  `OPEN QUESTION` with an owner.
- No clinical claims. Every UI surface and every CLI banner carries the
  research-tool disclaimer.

## Open questions tied to scope

| ID | Question | Owner | Target |
|----|----------|-------|--------|
| OQ-OVR-1 | Whether to commit to non-Carbon encoders in v1 or only via community PR | core | end of Phase 1 |
| OQ-OVR-2 | Whether to add a license addendum forbidding clinical use vs README-only disclaimer | core | before v0.1.0 tag |
| OQ-OVR-3 | Reassess hosted-API stance only if demand signal post-launch demands | core | end of Phase 3 |
