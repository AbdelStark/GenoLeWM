# GenoLeWM — Architecture

This document is a narrative walk-through of the system. It complements
[SPECIFICATION.md](SPECIFICATION.md) (which is normative) by explaining
*how* the parts fit together at runtime.

---

## 1. The picture

```
                     ┌─────────────────────────────────────────────┐
                     │              GenoLeWM Runtime                │
                     └─────────────────────────────────────────────┘

   reference window  ────► [ Carbon-500M encoder (frozen)  ]  ───► s_t
   w_ref (≈12 kbp)         layer L, mean-pool ±256 around edit

                                                                       ┌──────────┐
   edit a (pos, type, ref, alt)  ────► [ Action Encoder ]  ──► a_emb ─►│Predictor │
                                       (sin pos + type +              │  g(·,·)  │
                                        seq emb + MLP)                └─────┬────┘
                                                                            │
                                                                            ▼
                                                                         ŝ_{t+1}
                                                                            │
                                                                            ├─► surprise score = ||ŝ − s_{t+1}||
                                                                            │   (RFC-0009)
                                                                            │
                                                                            ├─► latent rollout (apply next edit)
                                                                            │   (RFC-0008)
                                                                            │
                                                                            └─► planning (CEM/MPC over edits)
                                                                                (RFC-0008)
```

Three observations.

1. **The encoder is the heavy thing.** Carbon-500M is ~500M parameters
   and dominates compute, memory, and energy. Everything else in the
   diagram is small.
2. **The predictor is the trainable thing.** Carbon stays frozen in
   Phase 1. The action encoder + predictor together are ~25–30M
   parameters.
3. **Once you have `ŝ_{t+1}`, every downstream use is cheap.** Surprise
   is a subtraction. Rollout is another predictor call. Planning is
   CEM over the predictor. None of these require Carbon.

This is the whole point: by paying for Carbon **once per reference
window**, we get a predictor that supports thousands of cheap downstream
queries.

---

## 2. Three running examples

To ground the rest of this document, here are three concrete tasks the
system must handle.

### 2.1 Score a single variant

```
input:  (chromosome=17, position=43,071,077, ref="C", alt="T")
        # BRCA1 c.5096G>A variant
output: surprise_score, latent_displacement, confidence_interval
```

Runtime:
1. Look up the reference window centered on position 43,071,077.
2. Run Carbon-500M on that window once → `s_t`. (Or look it up in cache.)
3. Encode the action `(0, SNV, "C", "T")` → `a_emb`.
4. Predictor: `ŝ_{t+1} = g(s_t, a_emb)`.
5. Run Carbon-500M on the *edited* window → `s_{t+1}`.
6. Return `||ŝ_{t+1} − s_{t+1}||₂`.

Total Carbon calls: 2. With reference caching, **1**.

### 2.2 Roll out a haplotype

```
input:  reference window, list of edits [a1, a2, a3]
output: ŝ_final  (predicted embedding of the haplotype window)
```

Runtime:
1. Look up reference embedding `s_0`.
2. Predictor step 1: `ŝ_1 = g(s_0, a_emb_1)`.
3. Predictor step 2: `ŝ_2 = g(ŝ_1, a_emb_2)`.
4. Predictor step 3: `ŝ_3 = g(ŝ_2, a_emb_3)`.

Total Carbon calls: 1. **Three edits, one Carbon call.** This is the
efficiency thesis.

### 2.3 Plan an edit set

```
input:  current latent s_t, target latent s_target
        edit budget K = 5
output: ordered list of edits [a1, ..., aK]
```

Runtime: Cross-Entropy Method over K-edit sequences, scoring each
candidate by latent distance after K-step predictor rollout. See
RFC-0008.

Total Carbon calls: 1 (for the initial encoding). The search itself
never touches Carbon.

---

## 3. Module map

This map describes the current repository layout. Planned public
contracts that do not yet have implementation modules are called out in
the roadmap instead of being shown as present code here.

```
geno_lewm/
├── encoder/
│   ├── carbon.py            # Carbon model + tokenizer wrapper
│   ├── windowing.py         # window extraction, padding, multiple-of-6 alignment
│   ├── pooling.py           # mean / attn / last-token / centered-pool
│   └── cache.py             # disk cache (Parquet) for reference embeddings
│
├── action/
│   ├── encoder.py           # the action encoder module (RFC-0003)
│   ├── spec.py              # EditSpec dataclass and validation
│   ├── apply.py             # apply_edit(window, edit) → edited window
│   └── synthetic.py         # synthetic edit samplers
│
├── predictor/
│   ├── model.py             # cross-attention predictor (RFC-0004)
│   ├── ar.py                # autoregressive rollout wrapper
│   └── losses.py            # L_pred (cosine + MSE), L_reg (LeJEPA)
│
├── data/
│   ├── corpus.py            # HF dataset wrapper for carbon-pretraining-corpus
│   ├── gnomad.py            # gnomAD common variants loader
│   ├── clinvar.py           # ClinVar loader, P/LP/B/LB labels
│   ├── builder.py           # tuple (w_ref, a, w_alt) builder and holdout policy
│   └── _vcf.py              # shared VCF parsing helpers
│
├── evaluation.py            # artifact-level binary metrics and report payloads
├── carbon_zero_shot.py      # Carbon baseline score artifact generation
│
├── planning/
│   ├── costs.py             # edit-sequence cost functions
│   └── sampling.py          # factored action sampler
│
├── surprise/
│   ├── context.py           # scored context payloads
│   ├── score.py             # ||ŝ - s||, cos(ŝ, s_t), bayesian variants
│   └── calibration.py       # context-aware calibration of surprise distributions
│
├── deploy/
│   ├── runtime.py           # local runtime facade and fail-closed network guard
│   └── import_/             # local personal-genome import helpers
│
├── provenance/
│   ├── commitment.py        # input/output commitments
│   ├── hashing.py           # canonical JSON and SHA-256 helpers
│   ├── manifest.py          # model artifact manifests
│   └── receipt.py           # checksum receipt schema and IO
│
├── cli/
│   ├── train.py             # entry point: train predictor
│   ├── score.py             # entry point: score a VCF / single variant
│   ├── rollout.py           # entry point: haplotype rollout
│   ├── plan.py              # entry point: planning
│   ├── eval.py              # entry point: evaluate score/label artifacts
│   ├── eval_all.py          # entry point: aggregate eval report artifacts
│   ├── carbon_baseline.py   # entry point: Carbon zero-shot baseline
│   ├── prepare_gnomad.py    # entry point: prepare gnomAD shards
│   ├── prepare_clinvar.py   # entry point: prepare ClinVar shards
│   ├── update.py            # entry point: user-approved model update checks
│   ├── export.py            # entry point: export scaffold
│   └── verify.py            # entry point: verify manifests and receipts
│
├── training/
│   ├── fixture.py           # deterministic fixture smoke training
│   ├── preflight.py         # real-run preflight report
│   ├── real.py              # Carbon-backed training launcher
│   ├── trainer.py           # torch trainer primitives
│   ├── collapse.py          # collapse metrics and alerts
│   └── sampling.py          # rollout/edit source sampling
│
├── config/
│   ├── loader.py            # typed config loading and overrides
│   ├── schema.py            # closed config schema
│   └── defaults/            # train/score/eval/plan defaults
│
└── __init__.py
```

The module map mirrors the RFCs one-to-one, which is deliberate: every
RFC has a home, and every module has an RFC that defines its contract.

---

## 4. Data flow during training

```
[ corpus.py ] ── windows ──┐
                            ├── [ builder.py ] ── (w_ref, a, w_alt) ──┐
[ gnomad.py / synthetic ] ──┘                                         │
                                                                       ▼
                                                       [ encoder/carbon.py ]
                                                                       │
                                                                       │ s_t, s_{t+1}
                                                                       ▼
                                                       [ action/encoder.py ]
                                                                       │ a_emb
                                                                       ▼
                                                       [ predictor/model.py ]
                                                                       │ ŝ_{t+1}
                                                                       ▼
                                                       [ predictor/losses.py ]
                                                                       │ L_pred
                                                                       ▼
                                                            backward + optimizer step
                                                            (only predictor params)
```

Reference-window encodings are cached, so the first epoch is encoder-bound
and subsequent epochs are predictor-bound. See RFC-0006 §6 on the cache
hit-rate budget.

---

## 5. Data flow during inference

For VEP scoring:

```
input variant ──► [ window lookup ] ──► [ encoder cache hit? ]
                                              │
                                ┌─────────────┴────────────┐
                                │ yes                       │ no
                                ▼                           ▼
                          [ s_t from disk ]      [ Carbon-500M ]
                                │                           │
                                └───────────┬───────────────┘
                                            ▼
                                  [ action encoder + predictor ]
                                            │
                                            ▼ ŝ_{t+1}
                                  [ Carbon-500M on edited window ]
                                            │
                                            ▼ s_{t+1}
                                  [ surprise = ||ŝ - s|| ]
```

For haplotype rollout, the second Carbon call is omitted — we use only
`ŝ_final` as the predicted latent.

For planning, we never decode and never call Carbon during search; we
operate entirely in latent space.

---

## 6. What's in the trained checkpoint

A GenoLeWM checkpoint is a directory:

```
geno-lewm-v0.1.0-carbon-500m-r1/
├── config.json              # all hyperparameters, encoder identifier
├── action_encoder.safetensors
├── predictor.safetensors
├── lora/                    # optional, only present in Phase 2+
│   └── carbon_lora.safetensors
├── tokenizer/               # symlink or copy of Carbon tokenizer
├── encoder_hash.txt         # SHA-256 of the frozen Carbon weights
├── train_config.yaml        # full training recipe used
└── eval_report.md           # eval numbers at release time
```

The checkpoint does **not** ship Carbon weights — they are pulled from
the Hugging Face Hub at load time. This is intentional: it keeps the
GenoLeWM artifact small (~120 MB at fp16) and forces explicit
configuration of the encoder version.

---

## 7. What's *not* in the architecture

To prevent scope creep, here are things explicitly absent from v1:

- **No discriminator network.** Surprise comes directly from predictor
  residual; we do not train a separate classifier on `(s_t, ŝ_{t+1},
  s_{t+1})` triples. v2 can add one if useful.
- **No language modeling head.** GenoLeWM does not predict tokens or
  decode back to DNA. Carbon does that.
- **No protein head.** The latent is DNA-only. Protein embeddings can
  be a downstream concatenation in v2.
- **No EMA target encoder.** LeWM's stability result removes the need;
  we will measure collapse explicitly in monitoring (see RFC-0005 §6).
- **No mixed-precision training of the encoder.** Carbon stays in
  bf16 / fp16 as released. We do not re-quantize during training.
