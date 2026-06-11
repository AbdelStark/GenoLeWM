# GenoLeWM Architecture

GenoLeWM is an action-conditioned latent world model for genomic edits.
It uses a frozen DNA foundation model as a state encoder and trains a
small action-conditioned predictor over that latent state space.

The current public release is an alpha research system. The architecture
is useful for reproducing the published experiments and inspecting the
model path; it is not clinical software and does not establish broad
model-quality superiority over Carbon.

## Runtime Flow

```text
reference window --Carbon encoder--> s_t
edit spec --------action encoder----> a_t
(s_t, a_t) -------predictor---------> s_hat_{t+1}
edited window ----Carbon encoder----> s_{t+1}
loss = distance(s_hat_{t+1}, s_{t+1}) + collapse regularization
```

The frozen Carbon encoder is the expensive part. The trainable GenoLeWM
components are the action encoder and the predictor. Once a reference
state is available, rollout and planning can search through the smaller
predictor path instead of repeatedly running Carbon during the search.

## Main Components

### State Encoder

`geno_lewm.encoder` wraps Carbon model/tokenizer loading, sequence
window normalization, pooling, and optional local state caching.
The public scoring path still needs Carbon available at runtime because
the score compares the predicted post-edit state with the Carbon-encoded
edited state.

### Edit And Action Encoding

`geno_lewm.action` defines canonical genomic edit objects:

- `EditSpec` for chromosome-position-reference-alternate edits;
- `RelEdit` for window-relative edits;
- edit validation and edit application helpers;
- synthetic edit samplers used by tests and training data builders.

The action encoder maps a validated edit into a fixed-width embedding
that the predictor can attend to.

### Predictor And Rollout

`geno_lewm.predictor` contains the cross-attention predictor and the
autoregressive rollout wrapper. Single-edit prediction estimates the
post-edit latent state. Multi-edit rollout repeatedly feeds predicted
states back through the predictor.

The current benchmark evidence shows that rollout fidelity remains weak
against a source-state baseline, and K=20 rollout speed remains below
the original 5x target.

### Surprise Scoring

`geno_lewm.surprise` computes a raw latent residual and a calibrated
surprise value:

- `sigma_raw`: uncalibrated distance between predicted and encoded
  post-edit states;
- `sigma_calibrated`: percentile-like value from the released
  calibration table;
- `bucket_id`, `confidence`, and `low_confidence`: calibration-context
  metadata.

These values are research signals. They are not clinical classifications
or validated risk probabilities.

### Planning

`geno_lewm.planning` provides cost functions, action sampling, and a CEM
planner. The released planning demo proves that the manifest-backed
planning path executes against public artifacts. It does not prove that
the selected edits are biologically useful.

### Runtime And Provenance

`geno_lewm.deploy` loads model directories, resolves optional runtime
dependencies, applies the local network boundary, and exposes
single-variant / VCF scoring.

`geno_lewm.provenance` implements manifests, canonical hashing,
input/output commitments, and checksum receipts. The main files are
`manifest.py`, `commitment.py`, and `receipt.py`. Receipts bind artifact
and output identity. They do not certify runtime behavior, privacy, or
scientific correctness.

## Package Map

```text
geno_lewm/
├── action/       edit specs, edit application, action encoder
├── cli/          train, score, eval, rollout, plan, verify, export commands
├── config/       typed config loading and closed config schema
├── data/         VCF parsing, ClinVar/gnomAD prep, tuple streaming
├── deploy/       runtime facade, scoring, import/export helpers
├── encoder/      Carbon wrapper, windowing, pooling, cache helpers
├── planning/     CEM planning, cost functions, action sampling
├── predictor/    cross-attention predictor, losses, AR rollout
├── provenance/   manifests, commitments, receipts, hashes
├── surprise/     raw and calibrated surprise scoring
└── training/     fixture trainer, Carbon preflight, real trainer launcher
```

Evaluation entry points live in `geno_lewm/evaluation.py` for metric
orchestration and `geno_lewm/carbon_zero_shot.py` for Carbon baseline
scoring.

## Training Data Flow

```text
reference windows ─┐
                   ├── tuple builder ── (w_ref, edit, w_alt)
edit sources ──────┘                         │
                                             ├── Carbon(w_ref) -> s_t
                                             ├── action_encoder(edit) -> a_t
                                             ├── predictor(s_t, a_t) -> s_hat
                                             └── Carbon(w_alt) -> s_{t+1}
```

Training packages record the resolved config, data snapshot identity,
preflight report, training metrics, checkpoint files, manifest/card, and
checksum inventory.

## Inference Data Flow

Single-variant scoring validates that the `REF` allele matches the
supplied reference window before model inference.

```text
variant + reference window
        │
        ├── validate coordinate / REF match
        ├── Carbon(reference window) -> s_t
        ├── action_encoder(edit) -> a_t
        ├── predictor(s_t, a_t) -> s_hat_{t+1}
        ├── Carbon(edited window) -> s_{t+1}
        └── surprise score + optional checksum receipt
```

VCF scoring repeats the same path over locally supplied VCF and FASTA
inputs. The project does not provide a hosted scoring service.

## Published Evidence Boundary

The public v0.2.1 run tree contains benchmark, rollout, planning, and
paper evidence. The current result is best described as systems evidence
with negative or mixed model-quality findings:

- GenoLeWM does not broadly beat Carbon.
- K=20 autoregressive rollout speed remains below target.
- The planning demo exercises the released model path but does not prove
  useful planning behavior.
- Fixture smoke outputs are CI evidence, not model results.

Use the generated paper and Hugging Face model card for artifact-bound
claims.
