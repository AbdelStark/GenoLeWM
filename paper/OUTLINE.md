# GenoLeWM Paper Outline — SUPERSEDED

**Status:** Superseded. `paper/main.tex` is the source of truth for the manuscript's title, claims,
structure, figures, tables, and every number. This file is retained only as a pointer and a scope
record; it is a pre-writing planning artifact and is a candidate for deletion.

This document previously carried the outline for *"GenoLeWM: An Action-Conditioned Latent World Model
for Genomic Edits. Post-Release Validity Correction for `v0.2.1-r1`"*, whose thesis was the retraction
itself. **That framing no longer describes the paper.** The manuscript was rewritten as

> **Registered but Not Anticipated: The Edit-Response Geometry of a Frozen Genomic Foundation Model**

a measurement plus a negative result, in which the `v0.2.1-r1` validity correction is a single section
(*"Why the Released Run Failed, and Why a Correct One Would Have Too"*) rather than the thesis. The
retraction's factual record is preserved there and in the historical VEP/rollout/artifact tables.

For the current outline, claims, and scope, read `paper/main.tex` — specifically its abstract, the
contributions list in the Introduction, and the Limitations section. `paper/README.md` summarises it.

## Scope guardrails that still apply

These survive the reframe and are stated in the paper's Introduction ("What we do *not* claim") and
Limitations. They are repeated here only so that this file cannot be read as licensing them:

- No clinical, diagnostic, privacy, deployment, or decision-support claim of any kind.
- No claim to a competitive variant-effect predictor. Every AUROC in the paper is **cohort-dependent**
  — Carbon's own Δlog-likelihood reaches `0.8945` on our ClinVar slice against `0.6`–`0.8` in the
  published literature, so the slice is easy — and none are comparable to a benchmark number.
- No claim that Carbon's informative, interventional edit response generalises to genomic foundation
  models at large; it demonstrably does not (`0.7574` on NT-v2, `0.5460` on HyenaDNA). Every
  interventional statement is scoped to Carbon by name. The predict-no-change trap, by contrast, *is*
  universal.
- No claim about token-level world models, which were not tested; the result is about **pooled** states.
- No claim about encoders larger than 500M or with longer contexts, and no claim to know whether the
  measured spread tracks architecture, objective, or capacity.
- No capability, superiority, or inferiority claim from `v0.2.1-r1`; its L2, surprise/VEP, rollout, and
  planning interpretations remain withdrawn.
- No number from the invalid `bf16` Δlog-likelihood run is cited anywhere; the reported run is fp32.
