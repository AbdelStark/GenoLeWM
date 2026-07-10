# What does this edit do?

*A post-release correction to the GenoLeWM experiment and what must be tested
next.*

GenoLeWM asks a world-model question about genomic edits:

> Given a DNA window and a structured edit, can a small model predict the
> edited window's representation without rerunning the full DNA encoder?

The architecture is straightforward. Carbon-500M encodes a reference window,
an action encoder represents the edit, and a predictor estimates Carbon's
post-edit state.

```text
s_t       = Carbon(reference window)
a_t       = action_encoder(edit)
s_hat     = predictor(s_t, a_t)
s_target  = Carbon(edited window)
```

If the state contract is consistent, that setup could support cheap edit
scoring, multi-edit rollout, and planning over structured genomic actions.

The published v0.1 and v0.2.1 experiments did not test that intended method.
A post-release audit found a state-contract defect that invalidates their
scientific interpretation.

## The correction

The release configs declared `encoder.normalize: true`, but the Carbon wrapper
did not apply the requested L2 normalization after pooling. Source and target
states therefore remained raw vectors. The predictor, however, returned a
unit-normalized vector.

The audit found a second contract failure. Training encoded reference states
without an edit locus, triggering global-mean pooling, while targets were
pooled around the edit. Rollout then switched reference states to edit-centered
pooling, and historical distractors could be centered at different loci. Cache
v1 did not include that center in its identity.

Even the historical pools labeled "edit-centered" were not centered on the
intended DNA token. Carbon emits a leading `<dna>` control token before its
six-mer tokens, but the old conversion used `edit_locus // 6` directly as a
hidden-state index. The correct index starts one token later. Every intended
center was shifted one hidden token left, and early-window loci could center a
pool on the control token. The run therefore mixed state scale, coordinate
frame, and an off-by-one token layout.

The audit also found a reproducibility failure below the state contract.
Carbon's pinned `tokenizer.py` delegated to
`Qwen/Qwen3-4B-Base` without pinning a revision or requiring local files.
Hashing the local custom-tokenizer file therefore did not identify the
transitive tokenizer actually loaded and did not make execution self-contained.

The released rollout audit makes the mismatch visible:

| Slice | Source norm | Target norm | Prediction norm |
| --- | ---: | ---: | ---: |
| Phased haplotypes | 33.982 | 33.595 | 1.000 |
| Synthetic edit chains | 29.253 | 29.089 | 1.000 |

Those are not small numerical differences. They are different state spaces.
An L2 distance between a raw vector with norm near 30 and a unit vector is
dominated by the scale mismatch.

The compatibility name for the released behavior is `legacy_raw_v1`. The
corrected contract is `l2_normalized_v2`, which applies L2 normalization after
Carbon pooling and requires source, target, prediction, cache, rollout, and
planning paths to agree on one normalization view and one committed pooling
center.

No published checkpoint was trained under `l2_normalized_v2`.

The repaired source path implements Carbon's pure-DNA tokenizer from the local
runtime package, validates the `<dna>`/`</dna>` and six-mer layout, and derives
the pooling center from the observed DNA-content start. Those changes establish
code contracts only. They do not rescue an old checkpoint or provide fresh
model-quality evidence.

## What is invalid

The mismatch invalidates every released interpretation that depends on L2
distance between predicted and encoded states:

- `sigma_raw` residual magnitude;
- the calibration table fitted to that residual;
- variant-effect rankings derived from those values;
- rollout L2 distances;
- the planning demo's `best_distance` objective;
- mechanistic stories inferred from those distances.

The recorded JSON and tables remain reproducible outputs of the legacy
implementation. Reproducibility does not make the scientific quantity valid.

This means the release cannot support either a positive or a negative claim
about the intended normalized GenoLeWM method. The earlier statement that the
model failed against Carbon is withdrawn for the same reason a success claim
would be withdrawn: the evaluated score did not implement the declared method.

## What about cosine similarity?

Cosine similarity is invariant to vector magnitude, so the raw-versus-unit
scale mismatch does not algebraically invalidate a single cosine calculation.
That does not rescue the released rollout experiment.

The predictor was trained from raw global-mean sources toward raw targets whose
nominal edit centers were one hidden token left under the invalid contract.
Autoregressive rollout then started from raw sources using that same shifted
centering rule and fed unit-normalized predictions back into the
model, creating additional coordinate and scale shifts. The
released cosine rows are therefore historical and confounded.

They should not be described as evidence that the model learned useful
dynamics, failed to learn dynamics, beat a source-state baseline, or lost to
one. A corrected K>1 experiment must retrain under `l2_normalized_v2`, share a
pooling center across compared states, and keep the same contract at every step.

## Phase 2 did not adapt the encoder

The v0.2.1 run was described as Phase 2, but it did not train LoRA or another
Carbon adapter. Carbon remained frozen.

The reported KL term was computed only from frozen target states. It changed
the scalar value printed as total loss, but it was constant with respect to the
action encoder and predictor. It supplied no gradient to any optimized
parameter.

That is an implementation fact, not an ablation result. The run cannot be used
to argue that the regularizer helped, hurt, or failed to prevent collapse. A
future training path must fail closed when a configured regularizer has no
trainable gradient path.

## What remains true

The audit narrows the release claims; it does not erase every artifact.

The following systems statements remain directly inspectable:

- model, dataset, run, and report artifacts have content-addressed identities;
- manifests and checksum receipts bind the artifact and output bytes they
  include, but the historical encoder identity did not bind the unpinned
  transitive Qwen tokenizer;
- the training, packaging, evaluation, planning, and publication paths ran;
- the released planning command executed against its manifest-backed
  checkpoint, although its objective value is invalid;
- the autoregressive speed artifact measured predictor execution on synthetic
  tensors with its recorded H200/CUDA shape and settings;
- fixture outputs remain CI and integration evidence.

These are infrastructure claims. They do not establish representation quality,
variant-effect prediction, calibrated surprise, planning capability, clinical
utility, privacy assurance, or deployment readiness.

The speed artifact also has a narrow scope. It measures the predictor-only
autoregressive path, not exact checkpoint quality and not end-to-end Carbon
scoring. It can support an engineering optimization claim only within that
recorded benchmark shape.

## The experiment that should run next

The next useful result is not a larger benchmark table. It is a contract
validation run that makes a scientific benchmark possible.

### 1. Validate the state geometry

Before training, encode paired reference and edited windows under
`l2_normalized_v2` and record:

```text
||s_t||_2
||s_target||_2
||s_target - s_t||_2
cos(s_t, s_target)
```

All source and target norms should be finite and near one. The edit delta
distribution should be stratified by edit type, location, and sequence
context. This is a measurement of the corrected target geometry, not yet a
model result.

### 2. Prove that training uses the same contract

Cache entries should remain raw and versioned, with normalization applied at
the consumer boundary. Training checkpoints must record the effective state
contract and reject resume attempts across incompatible contracts.

The predictor input, target, and output should be checked in the training loop.
Any loss term advertised as active must have a verified gradient path to an
optimized parameter.

### 3. Run a small K=1 sanity experiment

Train a small, fresh `l2_normalized_v2` lineage first. Compare it with simple
baselines using preregistered metrics. Inspect loss components and state norms,
and stop if the contract drifts.

This run should answer only whether the corrected implementation can learn a
one-step mapping. It should not be promoted as variant-effect prediction or
planning evidence.

### 4. Test K>1 without changing state spaces

Only after K=1 passes should autoregressive evaluation feed predicted states
back under the exact contract used in training. Report both cosine and L2,
source-state baselines, uncertainty, and per-step norm diagnostics.

Acceptance criteria must be written before the run. Historical v0.2.1 cosine
rows are not thresholds for the corrected method.

### 5. Rebuild scoring and calibration from scratch

A corrected checkpoint needs a new calibration table. Legacy `sigma_raw` and
`sigma_calibrated` values cannot be transformed after the fact because the
weights themselves were optimized under the wrong contract.

Variant-effect evaluation should use pinned data identities, explicit labels,
appropriate baselines, confidence intervals, and enough rows for the stated
claim. Until then, the scoring surface is an implementation API, not a
scientific instrument.

### 6. Reopen planning only after the objective is valid

Planning requires a target state and candidate states in the same validated
space. A synthetic demo can establish execution, but capability requires a
preregistered task where objective improvement corresponds to a meaningful
outcome and survives held-out evaluation.

The historical `best_distance` value cannot be reused as a benchmark target.

## Why this correction matters

Artifact-bound research needs two kinds of integrity.

The first is byte integrity: which code, checkpoint, config, and report produced
an output? GenoLeWM's manifests and receipts address included bytes, but the
historical tokenizer path shows that a file hash is not a complete runtime
identity when that file performs an unpinned transitive load.

The second is semantic integrity: did the output quantity implement the method
that the documentation claimed? The released state contract and pooling
coordinate did not.

Both are necessary. A perfectly hashed invalid metric is still invalid.

The useful outcome of this audit is therefore not a new story about why a
model succeeded or failed. It is a tighter experimental boundary:

> GenoLeWM published a reproducible legacy pipeline, but it has not yet
> published a valid evaluation of its intended normalized world-model method.

More precisely, the archived outputs and their included artifacts are
reproducible records; the historical runtime was not fully self-contained
because of its transitive tokenizer dependency.

That is the claim the current artifacts support. The next result must come from
fresh `l2_normalized_v2` training, not reinterpretation of old weights.

*GenoLeWM is alpha research software. It is not a clinical or diagnostic tool,
medical evidence, or a deployment, privacy, or planning-capability claim.*

[1]: https://github.com/AbdelStark/GenoLeWM "GenoLeWM source repository"
[2]: https://huggingface.co/HuggingFaceBio/Carbon-500M "Carbon-500M model repository"
[3]: https://huggingface.co/abdelstark/geno-lewm-runs "GenoLeWM historical run artifacts"
