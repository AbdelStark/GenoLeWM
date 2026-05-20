# GenoLeWM — Roadmap

This is the public roadmap. Dates are nominal; the public dependencies
(Carbon updates, dataset releases) are out of our control and the
project will adjust without warning when they ship things that change
the picture.

---

## Phase 0 — Design (we are here)

**Goal:** lock down the spec and RFCs so that implementation work is
non-controversial.

**Exit criteria:**
- All 11 RFCs at status `Accepted`.
- Specification at `v0.1.0-draft` → `v0.1.0`.
- A reviewable RFC index in `rfcs/README.md`.

**Estimated duration:** 2 weeks.

---

## Phase 1 — Minimum viable predictor

**Goal:** a single working pipeline end-to-end, narrow scope, defensible
numbers.

**In scope:**
- Carbon-500M encoder (frozen).
- Action encoder for **SNVs only**.
- Predictor: cross-attention, ~20M params.
- Training on a **10% slice** of `carbon-pretraining-corpus`.
- Eval on **ClinVar coding** and **ClinVar non-coding** only.
- Reference checkpoint published on the Hugging Face Hub.

**Out of scope for Phase 1:**
- Indels and structural variants (Phase 2).
- LoRA-adapted encoder (Phase 2).
- BRCA2 / TraitGym evals (Phase 2).
- Planning, surprise calibration as products (Phase 2).
- On-device deployment (Phase 3).

**Exit criteria:**
- Pipeline trains end-to-end on a single H100 in < 24 hours.
- Surprise score reaches ≥ 0.80 AUROC on ClinVar coding.
- Latent rollout cosine similarity ≥ 0.85 on held-out single-edit windows.
- Reference checkpoint, weights, training code, and eval report public
  on the Hub.

**Estimated duration:** 6 weeks.

**Risks:**
- Carbon embedding instability across short windows (e.g., near
  intron-exon boundaries). Mitigation: stratified validation.
- Encoder caching exceeds storage budget. Mitigation: streaming
  re-encoding for low-priority shards.

---

## Phase 2 — Full edit coverage and planning

**Goal:** the full edit vocabulary, the full eval suite, and the
planning loop as a usable product.

**In scope:**
- Action encoder for SNV + INS + DEL + MNV (≤ 16 bp).
- Carbon-500M LoRA adaptation (rank 16, attention layers only).
- Add **L_reg** (LeJEPA isotropic-Gaussian regularizer) to the loss.
- Eval on the full Carbon evaluation suite (BRCA2, TraitGym).
- Multi-edit haplotype rollout from gnomAD phased data.
- **Planning loop** via CEM (RFC-0008), exposed as a CLI and as a
  Python API.
- **Calibrated surprise** with context stratification (RFC-0009).

**Exit criteria:**
- AUROC on ClinVar coding ≥ Carbon-500M zero-shot.
- AUROC on BRCA2 within 2 points of Carbon-3B zero-shot.
- Rollout cosine similarity on 3-edit haplotypes ≥ 0.80.
- Planning loop solves a held-out "minimal-edits-to-target" benchmark
  with ≥ 70% success in ≤ 5 edits.

**Estimated duration:** 10 weeks.

---

## Phase 3 — On-device

**Goal:** a personal-genome interpreter that runs on a laptop.

**In scope:**
- Export pipeline: ONNX, Core ML, GGUF.
- Quantization: int8 predictor, int4 Carbon-500M (with eval on
  quantization-induced drift).
- A reference desktop app skeleton (Tauri, ~not~ a polished product) that
  loads a VCF, runs GenoLeWM + Carbon-500M locally, displays variant
  surprise scores in a table.
- Attestation hooks (RFC-0011) wired into the inference path.

**Exit criteria:**
- Single-variant scoring latency < 200 ms on an M3 Max.
- Whole-VCF scoring of 100k variants in < 30 minutes on the same machine.
- Predictor + action encoder fit in < 200 MB at int8.
- Attestation receipt is reproducible across runs given identical
  weights and inputs.

**Estimated duration:** 8 weeks.

---

## Phase 4 — Verifiable inference

**Goal:** STARK attestation of inference, end-to-end.

**In scope:**
- Commit-and-reveal protocol for `(encoder_hash, predictor_hash,
  input_hash, output_hash)`.
- Reference STARK circuit for the predictor forward pass
  (the action encoder + cross-attention predictor; not Carbon).
- A demo flow: "run GenoLeWM on a personal variant, produce a STARK
  proof that the published GenoLeWM weights produced the reported
  surprise score for this input."

**Exit criteria:**
- A working proof-and-verify cycle on a real ClinVar variant.
- Proof generation time < 5 minutes on a workstation.
- Verification time < 1 second.

**Estimated duration:** 12 weeks. Probably longer in practice — STARK
proving over Transformer forward passes is research-grade.

**Note:** Phase 4 is the project's North Star. It is the bridge to
StarkWare's Integrity Thesis and the freedom-tech narrative around
personal health AI. It is also the most uncertain phase.

---

## Cross-phase workstreams

These run in parallel with the phases above.

### Documentation
- Glossary kept in sync with RFCs (`docs/glossary.md`).
- FAQ updated after each major version.
- A tutorial notebook per phase, published on the Hub.

### Community
- Public RFC PRs for any change to spec or RFCs.
- Discussion forum for evaluation methodology disputes.

### Benchmarks
- Continuous evaluation harness; no regression > 1 AUROC point on any
  benchmark between published checkpoints without an explicit RFC change.

### Encoder upgrades
- Track Carbon releases. Re-train against Carbon-3B and Carbon-8B at
  end of Phase 2; benchmark and decide whether to make 3B the default
  for Phase 3.

---

## Anti-roadmap (things we will not do)

- We will not add a chat / assistant layer on top of GenoLeWM in v1. The
  model is a tool, not a product.
- We will not enter the clinical decision-support market. The output is
  a research signal; clinical use requires regulatory work that is out
  of scope.
- We will not train a separate from-scratch DNA encoder. Carbon is the
  encoder. The whole project leverages Carbon's quality.
- We will not build a web service. On-device first; if a hosted variant
  is needed later, the community can build it from the open weights.
