# 01 — Architecture

- Status: Authoritative for v0.1
- Companion RFCs: [RFC-0002](../../rfcs/0002-state-encoder-carbon-integration.md),
  [RFC-0003](../../rfcs/0003-action-representation-genomic-edits.md),
  [RFC-0004](../../rfcs/0004-predictor-architecture.md),
  [RFC-0008](../../rfcs/0008-latent-planning.md),
  [RFC-0009](../../rfcs/0009-surprise-based-pathogenicity-scoring.md),
  [RFC-0010](../../rfcs/0010-on-device-personal-genome-deployment.md)
- Narrative companion: [`ARCHITECTURE.md`](../../ARCHITECTURE.md)

## Module boundaries

The repository is organized so that every RFC has exactly one home module
and every module has exactly one defining RFC.

```
geno_lewm/
├── encoder/           # RFC-0002 — Carbon state encoder + window cache
│   ├── carbon.py
│   ├── windowing.py
│   ├── pooling.py
│   └── cache.py
├── action/            # RFC-0003 — EditSpec + action encoder
│   ├── encoder.py
│   ├── spec.py
│   ├── apply.py
│   └── synthetic.py
├── predictor/         # RFC-0004 — cross-attention predictor + losses
│   ├── model.py
│   ├── ar.py
│   └── losses.py
├── data/              # RFC-0006 — corpus, loaders, holdouts, tuple builder
│   ├── corpus.py
│   ├── gnomad.py
│   ├── clinvar.py
│   ├── builder.py
│   └── holdouts.py
├── eval/              # RFC-0007 — VEP, rollout, efficiency
│   ├── vep.py
│   ├── rollout.py
│   ├── efficiency.py
│   └── calibration.py
├── planning/          # RFC-0008 — CEM, MCTS (Phase 2), cost functions
│   ├── cem.py
│   ├── mcts.py
│   └── cost.py
├── surprise/          # RFC-0009 — raw + calibrated surprise
│   ├── score.py
│   └── calibration.py
├── deploy/            # RFC-0010 + RFC-0019 — export, quantization, runtime
│   ├── onnx.py
│   ├── coreml.py
│   ├── ggml.py
│   ├── runtime.py
│   └── attestation.py
├── attestation/       # RFC-0011 — manifest, receipts, verifier
│   ├── manifest.py
│   ├── receipt.py
│   ├── verify.py
│   └── hashing.py
├── cli/               # RFC-0018 — command-line surface
│   ├── train.py
│   ├── score.py
│   ├── rollout.py
│   ├── plan.py
│   ├── eval.py
│   ├── export.py
│   ├── prepare.py
│   └── verify.py
├── config/            # RFC-0017 — Hydra config schema
│   └── defaults/
├── errors.py          # RFC-0012 — exception hierarchy (single file by design)
├── observability.py   # RFC-0013 — logger factory, metric names, redaction
└── __init__.py
```

Tests mirror the layout under `tests/`, with `tests/unit/` for unit tests,
`tests/integration/` for cross-module tests, `tests/property/` for
property-based tests, and `tests/ml/` for ML-specific tests (collapse
checks, rollout fidelity smoke tests). See
[`07-testing-strategy.md`](07-testing-strategy.md).

## Runtime data flow

### Training

```
corpus.py + synthetic samplers ──► builder.py
                                       │
                                       │ (w_ref, a, w_alt)
                                       ▼
                          encoder/carbon.py (frozen)
                                       │
                                       │ s_t, s_{t+1}
                                       ▼
                          action/encoder.py ──► a_emb
                                       │
                                       ▼
                          predictor/model.py ──► ŝ_{t+1}
                                       │
                                       ▼
                          predictor/losses.py ──► L_pred
                                       │
                                       ▼
                              optimizer step
                              (predictor params only in Phase 1)
```

Cache hits dominate from epoch 2 onward; see RFC-0006 §6.

### Inference — single-variant scoring

```
variant ──► window lookup ──► [cache hit?]
                                  │
                       ┌──────────┴──────────┐
                       │ yes                  │ no
                       ▼                      ▼
                  s_t from disk         Carbon-500M call
                       │                      │
                       └─────────┬────────────┘
                                 ▼
                      action encoder + predictor ──► ŝ_{t+1}
                                 │
                                 ▼
                      Carbon-500M on edited window ──► s_{t+1}
                                 │
                                 ▼
                      surprise.score ──► σ_raw, σ_calibrated
                                 │
                                 ▼
                      receipt.write
```

For haplotype rollout, the second Carbon call is omitted; the predictor's
output is the predicted final latent.

For planning, neither Carbon call occurs inside the search loop. The
search amortizes one Carbon call across thousands of predictor rollouts.

## Invariants

These hold across all subsystems. Violations are bugs, not features.

| ID | Invariant | Enforced by |
|----|-----------|-------------|
| INV-ARCH-1 | The encoder is frozen in Phase 1 (no gradients touch Carbon weights) | `requires_grad_(False)` at construction; checked at trainer start |
| INV-ARCH-2 | State and predictor outputs live in the same `d_state`-dim L2-normalized space | encoder L2-norms outputs; predictor output projection normalizes |
| INV-ARCH-3 | Action embeddings are projected to the predictor's hidden dim by exactly one learned linear layer | predictor input-projection module is a single `nn.Linear` |
| INV-ARCH-4 | Multi-edit application is right-to-left by descending `rel_pos` | `action/apply.py::apply_edits` |
| INV-ARCH-5 | Cache lookups are content-addressed by `(window_hash, encoder_hash, state_layer, pool_type, pool_radius)` | `encoder/cache.py` |
| INV-ARCH-6 | Every inference call produces a receipt; receipts are forward-compatible | `attestation/receipt.py::write` |
| INV-ARCH-7 | No inference path performs a network call after first-run setup | runtime fail-closed guard in `deploy/runtime.py` |
| INV-ARCH-8 | Every public function with a side effect on disk takes an explicit path argument | linted (custom AST check in CI) |
| INV-ARCH-9 | bf16 is the default predictor dtype; up-cast to fp32 occurs only at the loss boundary | trainer config defaults |
| INV-ARCH-10 | Reference embeddings cached on disk are never overwritten in place; new configs produce new shards | `encoder/cache.py::write_shard` |

## Concurrency model

- **Training** uses single-process single-GPU by default. Multi-GPU support
  is via `accelerate` and is opt-in. No multi-process data sharding inside
  the trainer; the `DataLoader` uses N worker processes for IO and
  on-the-fly encoding.
- **Inference (server)** uses async batching: requests queue into a
  predictor batch with a 5 ms gather window, then issue one forward pass.
- **Inference (laptop)** is single-threaded by default. Batched VCF scoring
  uses a producer-consumer pipeline with one Carbon worker and one
  predictor worker.
- **Planning** is single-process. Parallelizable to multi-GPU in Phase 2 if
  CEM sample counts exceed single-GPU memory.

There is no asynchronous training, no parameter server, no distributed
optimizer state. Multi-node training is out of scope for v1.

## Determinism

| Path | Determinism guarantee |
|------|-----------------------|
| Training with `--deterministic` and pinned seeds | bit-exact within same hardware + driver |
| Training without `--deterministic` | reproducible loss curves to within sampling noise |
| Inference (CUDA bf16) | bit-exact within same GPU + driver version |
| Inference (Core ML) | bit-exact within same OS minor version |
| Inference (ONNX Runtime CPU) | bit-exact across machines |
| Verifier re-run | bit-exact when backend matches receipt's `runtime.backend` |

The receipt format (RFC-0011 §3.3) explicitly records the runtime backend
so that verifiers know which determinism class to apply.

## Failure semantics

Subsystems fail fast and fail loud. The contract:

- **Validation errors** (malformed input) raise a typed exception from
  `geno_lewm.errors` (see [`04-error-model.md`](04-error-model.md)) with
  a structured payload.
- **Capacity errors** (out-of-memory, disk full) are reported with the
  resource that exhausted and surface a remediation hint.
- **Cache misses** are not errors; they trigger a fall-through to live
  encoding with a `cache_miss` metric increment.
- **Network failures during first-run setup** raise `RuntimeSetupError`
  with a clear retry strategy.
- **Backend mismatches** during verification raise `AttestationMismatchError`
  with the expected and observed backends.

No subsystem swallows exceptions. No subsystem returns `None` on failure.

## What is not in the architecture

To keep scope honest, the following are intentionally absent in v1:

- Discriminator networks. Surprise is the predictor residual; no separate
  classifier is trained on `(s_t, ŝ_{t+1}, s_{t+1})`.
- Language modeling heads. Decoding to DNA is Carbon's job.
- Protein heads.
- EMA target encoders (LeWM's stability result removes the need).
- Mixed-precision training of Carbon — encoder dtype is fixed at load time.
- Online model updates (auto-update is disabled by design; see RFC-0010 §3.8).
- A web service. No HTTP API in v1.
