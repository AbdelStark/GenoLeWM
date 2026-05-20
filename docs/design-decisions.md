# Design decisions log

A record of resolved design trade-offs across GenoLeWM. The RFCs
contain the *rationale* for each decision; this document is the
*index* — a single place to look up "what did we decide about X and
where is the justification?"

When an RFC ships a new resolved decision, add an entry here.
When a decision is amended, append a new entry (do not edit the old
one); the history is part of the value.

---

## Architecture

### State encoder is Carbon-500M, frozen in Phase 1

- **Decided:** 2026-05-20
- **RFC:** [0002 §3.1, §4.2](../rfcs/0002-state-encoder-carbon-integration.md)
- **Rationale (short):** Carbon-500M is the smallest model that meets
  the published quality bar, fits the consumer-hardware deployment
  target, and unlocks single-GPU training when frozen.

### State vectors are L2-normalized at the encoder

- **Decided:** 2026-05-20
- **RFC:** [0002 §3.5, §4.5](../rfcs/0002-state-encoder-carbon-integration.md)
- **Rationale (short):** Stable cosine + MSE loss combination; stable
  distance-based surprise calculation.

### Window length is 12,288 bp (2,048 6-mer tokens)

- **Decided:** 2026-05-20
- **RFC:** [0002 §3.2, §4.4](../rfcs/0002-state-encoder-carbon-integration.md)
- **Rationale (short):** Middle ground between exon coverage and
  encoding cost.

### Pooling is centered-mean over ± 256 tokens

- **Decided:** 2026-05-20
- **RFC:** [0002 §3.4, §4.1](../rfcs/0002-state-encoder-carbon-integration.md)
- **Rationale (short):** Edit-local; no extra parameters; outperforms
  global mean on edit-sensitive tasks (to be confirmed by ablation).

### Action encoder uses four sub-encoders (position, type, ref, alt)

- **Decided:** 2026-05-20
- **RFC:** [0003 §3.4, §4.1](../rfcs/0003-action-representation-genomic-edits.md)
- **Rationale (short):** Inductive bias matching structure of an edit;
  shared ref/alt SeqMicroEncoder enforces compositional generalization.

### v1 caps `len(ref)` and `len(alt)` at 16 bp

- **Decided:** 2026-05-20
- **RFC:** [0003 §3.1, §3.5, §4.3](../rfcs/0003-action-representation-genomic-edits.md)
- **Rationale (short):** Covers > 95% of clinically relevant short
  variants; SVs require separate adapter (v2 RFC).

### Predictor is cross-attention Transformer (4 cross + 2 self blocks)

- **Decided:** 2026-05-20
- **RFC:** [0004 §3.1, §4.1](../rfcs/0004-predictor-architecture.md)
- **Rationale (short):** Variable-length action sequences without
  arch change; cross-attention exposes structured action sub-embeddings
  to state.

### Predictor output MLP final layer is zero-initialized

- **Decided:** 2026-05-20
- **RFC:** [0004 §3.4, §4.3](../rfcs/0004-predictor-architecture.md)
- **Rationale (short):** Identity-at-init; predictor starts by
  outputting `s_t`, making early training stable.

---

## Training

### Loss is `α · (1 − cos) + β · MSE / d_state`

- **Decided:** 2026-05-20
- **RFC:** [0005 §3.1, §4.1](../rfcs/0005-training-objective.md)
- **Rationale (short):** Cosine for direction, MSE for magnitude
  calibration; matches LeWM recipe ported to L2-normalized embeddings.

### LeJEPA regularizer is monitored-only in Phase 1

- **Decided:** 2026-05-20
- **RFC:** [0005 §3.2, §3.3, §4.4](../rfcs/0005-training-objective.md)
- **Rationale (short):** Frozen encoder → collapse impossible →
  regularizer not needed as training term; computed for monitoring to
  catch unexpected drift.

### Optimizer is AdamW with `β₂ = 0.95`

- **Decided:** 2026-05-20
- **RFC:** [0005 §3.4](../rfcs/0005-training-objective.md)
- **Rationale (short):** Stability with small batches over
  high-dimensional latents; standard for JEPA training.

### LR schedule is WSD (warmup-stable-decay)

- **Decided:** 2026-05-20
- **RFC:** [0005 §3.5, §4.3](../rfcs/0005-training-objective.md)
- **Rationale (short):** Phase-transition friendly; checkpoint at the
  end of stable phase, continue training with fresh decay schedule when
  LoRA is enabled.

### Batch size 256, edit-balanced sampling

- **Decided:** 2026-05-20
- **RFC:** [0005 §3.7, §4.5, §4.6](../rfcs/0005-training-objective.md)
- **Rationale (short):** Matches LeWM; supports stable covariance
  estimation in Phase 2; per-type balance gives indels enough training
  signal.

---

## Data

### Reference corpus is `HuggingFaceBio/carbon-pretraining-corpus`

- **Decided:** 2026-05-20
- **RFC:** [0006 §3.1, §4.1](../rfcs/0006-data-pipeline.md)
- **Rationale (short):** In-distribution for Carbon → most reliable
  encoder outputs; pre-processed and tokenization-validated; public.

### Edit-source mix is 40 gnomAD / 30 synthetic SNV / 20 synthetic indel / 10 ClinVar

- **Decided:** 2026-05-20
- **RFC:** [0006 §3.3, §4.2](../rfcs/0006-data-pipeline.md)
- **Rationale (short):** Balance of realism (gnomAD), action coverage
  (synthetic), and hard signal (ClinVar).

### Windows overlap at 67% (stride 8,192 bp)

- **Decided:** 2026-05-20
- **RFC:** [0006 §3.2, §4.3](../rfcs/0006-data-pipeline.md)
- **Rationale (short):** Each position covered by ~3 windows; gives
  predictor multiple contexts per variant.

### Three holdouts: `holdout-chr` (chr21), `holdout-clinvar`, `holdout-haplotypes`

- **Decided:** 2026-05-20
- **RFC:** [0006 §3.8, §4.5](../rfcs/0006-data-pipeline.md)
- **Rationale (short):** Clean spatial generalization (entire
  chromosome); clean known-pathogenic generalization (ClinVar P/LP);
  clean multi-edit generalization (gnomAD haplotypes).

---

## Evaluation

### VEP benchmarks mirror Carbon's published suite

- **Decided:** 2026-05-20
- **RFC:** [0007 §3.1, §4.1](../rfcs/0007-evaluation-suite.md)
- **Rationale (short):** Direct comparability with Carbon's model card
  numbers.

### Two scoring heads reported: surprise and displacement

- **Decided:** 2026-05-20
- **RFC:** [0007 §3.1.2, §4.2](../rfcs/0007-evaluation-suite.md)
- **Rationale (short):** Different uses → different signals; prevents
  optimizing one at the cost of the other.

### Rollout fidelity reported per-K with a naive baseline

- **Decided:** 2026-05-20
- **RFC:** [0007 §3.2, §4.3](../rfcs/0007-evaluation-suite.md)
- **Rationale (short):** Catches degenerate predictors that output
  `s_t` regardless of action.

### Efficiency benchmarks include Apple M3 Max

- **Decided:** 2026-05-20
- **RFC:** [0007 §3.3, §4.4](../rfcs/0007-evaluation-suite.md)
- **Rationale (short):** Freedom-tech / personal-genome target audience
  skews Mac; first-class target for Phase 3 honesty.

---

## Planning

### Default solver is CEM

- **Decided:** 2026-05-20
- **RFC:** [0008 §3.4, §4.1](../rfcs/0008-latent-planning.md)
- **Rationale (short):** Discrete edit space; no per-task training;
  fast enough on H100 to amortize per query.

### Planning never calls Carbon during search

- **Decided:** 2026-05-20
- **RFC:** [0008 §2](../rfcs/0008-latent-planning.md)
- **Rationale (short):** Efficiency thesis of the world-model framing;
  pay for Carbon once, run thousands of CEM rollouts at predictor cost.

---

## Surprise

### Calibrated surprise is the published score; raw residual is also exposed

- **Decided:** 2026-05-20
- **RFC:** [0009 §3.5, §3.7, §4.4](../rfcs/0009-surprise-based-pathogenicity-scoring.md)
- **Rationale (short):** Context-aware percentile is interpretable;
  raw exposed for debugging and recalibration.

### Calibration distribution is gnomAD common variants (AF ≥ 1%)

- **Decided:** 2026-05-20
- **RFC:** [0009 §3.4, §4.2](../rfcs/0009-surprise-based-pathogenicity-scoring.md)
- **Rationale (short):** Biology's tolerated background; appropriate
  null model.

### Calibration buckets by `(region_class, gc_bin, repeat_class)` with back-off

- **Decided:** 2026-05-20
- **RFC:** [0009 §3.3, §4.3](../rfcs/0009-surprise-based-pathogenicity-scoring.md)
- **Rationale (short):** Standard pattern (matches CADD); back-off
  handles sparse buckets gracefully.

---

## Deployment

### Primary on-device target is Apple Silicon

- **Decided:** 2026-05-20
- **RFC:** [0010 §3.1, §4.1](../rfcs/0010-on-device-personal-genome-deployment.md)
- **Rationale (short):** User overlap; hardware quality for this size
  range; signed-binary distribution maturity.

### Carbon weights are not bundled; pulled from Hugging Face Hub on first run

- **Decided:** 2026-05-20
- **RFC:** [0010 §3.2, §4.2](../rfcs/0010-on-device-personal-genome-deployment.md)
- **Rationale (short):** Artifact size; canonical-source provenance;
  attestation surface.

### Automatic updates disabled

- **Decided:** 2026-05-20
- **RFC:** [0010 §3.8, §4.5](../rfcs/0010-on-device-personal-genome-deployment.md)
- **Rationale (short):** Reproducibility of published results requires
  pinned model versions.

### Runtime fails closed on network calls

- **Decided:** 2026-05-20
- **RFC:** [0010 §3.7](../rfcs/0010-on-device-personal-genome-deployment.md)
- **Rationale (short):** Privacy contract; silent online fallback
  unacceptable for personal-genome data.

---

## Verifiable inference

### Receipts are checksum-only in v1, STARK-proven in Phase 4

- **Decided:** 2026-05-20
- **RFC:** [0011 §3.3, §4.3](../rfcs/0011-verifiable-inference-attestation.md)
- **Rationale (short):** Lightweight ingredients are usable today;
  STARK proving of Transformer inference is still research-grade.

### STARK target is predictor + action encoder, not Carbon

- **Decided:** 2026-05-20
- **RFC:** [0011 §3.5, §4.3](../rfcs/0011-verifiable-inference-attestation.md)
- **Rationale (short):** 500M-param STARK proving infeasible; 25M-param
  is on the research frontier and tractable; Carbon committed as public
  input.

### Optional TEE attestation as v1.1 intermediate

- **Decided:** 2026-05-20
- **RFC:** [0011 §3.3, §4.4](../rfcs/0011-verifiable-inference-attestation.md)
- **Rationale (short):** Production-mature today; weaker than STARK but
  meaningfully better than checksum-only.
