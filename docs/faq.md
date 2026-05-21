# Frequently asked questions

Answers to questions we expect from three audiences: ML researchers,
genomicists, and the personal-health-AI community.

If you have a question that should be here, open a PR adding it.

---

## For ML researchers

### Why JEPA over a discriminative variant scorer?

A discriminative scorer requires labels (which are scarce and biased),
does not support multi-edit rollout, and does not give a surprise
signal. A JEPA gets all of those out of the box. The details are in
[RFC-0001 §4.1](rfcs/0001-project-scope-and-goals.md).

### Why action-conditioned JEPA instead of a masked-prediction JEPA?

Published bio-JEPAs (GeneJepa, CGM-JEPA, ECG-JEPA) are
representation-learning models that do masked prediction. They produce
embeddings but cannot reason about interventions. The whole point of
the action-conditioned formulation is to make edits a first-class
input the model is conditioned on, which is what unlocks rollout,
planning, and surprise scoring. See [RFC-0001 §4.2](rfcs/0001-project-scope-and-goals.md).

### Why is the encoder frozen?

Compute and stability. With a frozen Carbon-500M, GenoLeWM trains on
a single H100 in hours, and representation collapse is mechanically
impossible (the targets are fixed by a frozen encoder). Phase 2
optionally enables LoRA, at which point the LeJEPA regularizer becomes
a live training term. See [RFC-0002 §4.2](rfcs/0002-state-encoder-carbon-integration.md)
and [RFC-0005 §3.3](rfcs/0005-training-objective.md).

### What's the trainable parameter count?

Predictor: ~40M in the default config (`d_hidden = 1024`), or ~22M in
the smaller variant (`d_hidden = 768`) that becomes the on-device
default. Action encoder: ~2.5M. LoRA adapters (Phase 2): ~5M. So the
trainable footprint is ~25–47M depending on configuration, well within
single-GPU range.

### How does this differ from CodeLeWM?

The recipe is the same. The encoder is different (Carbon vs a code
embedding model), and the action encoder is different (genomic edits
vs code edits). The predictor architecture and training recipe port
directly. CodeLeWM is the sibling project for source code.

### What's the relationship to Carbon?

GenoLeWM **uses** Carbon. We do not modify, retrain, or distill it.
Carbon-500M (frozen) is the state encoder; the GenoLeWM trainable parts
(predictor + action encoder + optional LoRA) sit on top. Carbon's
likelihood scoring (`ΔlogLik(alt, ref)`) remains the right tool for
its purposes; GenoLeWM's surprise score is a different signal that
complements rather than replaces it.

### Why STARK proofs rather than SNARKs for verifiable inference?

No trusted setup, post-quantum, arithmetic-circuit shape matches
Transformer inference. The full argument is in
[RFC-0011 §4.2](rfcs/0011-verifiable-inference-attestation.md).

### Can I swap Carbon for another DNA encoder?

The architecture is encoder-agnostic in principle. v1 supports Carbon
only. A community PR adding Evo2, Generator-v2, or Nucleotide
Transformer as an alternative encoder would be welcome; the contract
is laid out in `encoder/carbon.py`.

---

## For genomicists

### Is this a clinical tool?

No. It is a research tool. Output is a research signal, not a clinical
diagnosis. If a variant in a GenoLeWM scoring report concerns you,
talk to a qualified genetic counselor. The README and the desktop app
both surface this prominently.

### What benchmarks does GenoLeWM target?

The same suite Carbon uses: ClinVar coding, ClinVar non-coding, BRCA2
(Findlay et al. saturation editing), TraitGym Mendelian. We report on
the same metrics (AUROC, AUPRC, Spearman ρ where applicable). Direct
side-by-side with Carbon is the point. See
[RFC-0007](rfcs/0007-evaluation-suite.md).

### How does it handle structural variants?

It doesn't, in v1. Edits are capped at 16 bp for both ref and alt. A
separate SV adapter is planned for v2 ([RFC-0003 §3.5](rfcs/0003-action-representation-genomic-edits.md)).
This covers > 95% of clinically relevant short variants.

### What populations is the calibration valid for?

The current calibration is built on gnomAD common variants without
explicit population stratification. This means the calibration is most
reliable for variants from populations well-represented in gnomAD, and
less reliable for others. Per-population calibration is a Phase 2
workstream; documented as an explicit limitation in the meantime.

### Can I score a whole VCF?

Yes. `geno-lewm-score --vcf my.vcf.gz --output scores.parquet`. With
Carbon-500M on an M3 Max, expect ~30 minutes for 100k variants. See
[RFC-0010 §3.5](rfcs/0010-on-device-personal-genome-deployment.md).

### How does the surprise score compare to CADD / AlphaMissense / Carbon-likelihood?

Different signal. CADD aggregates many features; AlphaMissense is
trained on protein-structure context; Carbon's likelihood is the
log-probability difference of `alt` vs `ref` under an autoregressive
DNA model. Surprise is the predictor's residual: how much the predicted
post-edit latent differs from the actual post-edit latent. We expect
these signals to correlate but not be redundant; Phase 2 will report
correlations and explore ensembling.

### Does the model know about non-coding variants?

Yes. Carbon's pretraining corpus includes substantial non-coding
sequence, and the training mix for GenoLeWM (RFC-0006) draws from the
full corpus. The eval reports ClinVar non-coding separately from
ClinVar coding so non-coding performance is visible.

### Why centered-mean pooling around the edit?

To make the state vector edit-local. A globally-pooled state would
dilute the edit's effect across 12,288 bp; the centered mean
concentrates the state around the ~3 kbp where edits actually act.
See [RFC-0002 §3.4](rfcs/0002-state-encoder-carbon-integration.md).

### Is GenoLeWM applicable to non-human genomes?

Carbon's pretraining is predominantly eukaryotic with some prokaryotic
content; the architecture is species-agnostic. We have not specifically
evaluated GenoLeWM on, e.g., model organisms, but the eval suite can be
extended to mouse, fly, yeast benchmarks (planned for v2).

---

## For the personal-health-AI community

### Does the model see my data?

No. The inference runs entirely on your device. The runtime fails
closed on network calls: if any inference path attempts to phone home,
it raises an error rather than silently degrading to online mode. See
[RFC-0010 §3.7](rfcs/0010-on-device-personal-genome-deployment.md).

### Does Anthropic / Hugging Face / anyone see what I score?

No. There is no telemetry. We do not collect usage data. Crash logs are
sanitized to exclude variant data. The only network calls are during
first-run setup (model download from Hugging Face Hub) and explicit
user-initiated updates.

### Can I run this on a laptop?

Yes. The primary on-device target is Apple Silicon (M3 Max or better
recommended). With int4 quantization of Carbon-500M and int8 of the
predictor, the model fits in ~600 MB of memory and scores single
variants in < 200 ms. See [RFC-0010 §3.5](rfcs/0010-on-device-personal-genome-deployment.md).

### Can I run it on Windows / Linux?

Yes. CUDA workstation (RTX 4090-class) is a first-class target.
CPU-only is supported as an accessibility fallback (slower).

### What if a new GenoLeWM version gives different scores?

By design, you can roll back. Updates are explicit; previous model
versions are preserved as side-by-side installs. Published results
referencing `geno-lewm-v0.1.0-carbon-500m-r1` will keep producing the
same scores months later. See [RFC-0010 §3.8](rfcs/0010-on-device-personal-genome-deployment.md).

### Can I prove to someone that my score came from the real model?

In v1: yes, via a checksum-only receipt. Anyone with the same model
weights and the same input can re-run the inference and verify a
bit-match (on supported backends). In Phase 4, the receipt will
include a STARK proof that no re-run is needed for verification.
See [RFC-0011](rfcs/0011-verifiable-inference-attestation.md).

### Will there be an iOS app?

Plausibly feasible technically; not in v1. Apple Silicon laptop is
the v1 target. iOS is a v2 consideration.

### Will there be a hosted version?

Not from us. The whole architecture is local-first. If the community
builds a hosted variant from the open weights, that's their choice;
GenoLeWM the project will remain on-device-first.

### How do I import my 23andMe / AncestryDNA / MyHeritage data?

The runtime accepts those formats directly and converts them locally to
VCF for scoring. Conversion never leaves your machine. See
[RFC-0010 §3.9](rfcs/0010-on-device-personal-genome-deployment.md).

### Should I act on a high surprise score?

Talk to a genetic counselor. GenoLeWM is a research tool; its surprise
score is a single signal among many that clinical genetics integrates.
A high score warrants follow-up by a qualified professional, not direct
action.

---

## For contributors

### How do I contribute?

The project is in design phase (Phase 0). The most valuable
contributions right now are reviews of the RFCs: open a PR with
inline comments or proposed edits. Implementation contributions will
become valuable starting with Phase 1.

### Can I add a new encoder?

Yes, via a community PR. The encoder contract is `CarbonStateEncoder`
in `geno_lewm/encoder/carbon.py`. Any encoder that conforms to the
same interface (`encode`, `encode_batch`, `encoder_hash`, `d_state`)
should work.

### How do I propose a new feature?

Write an RFC. The template is `rfcs/0000-template.md`. Open a PR
adding it under the next free number. Discussion happens in the PR
review.

### What's the project's governance?

Currently informal: core contributors and PR-based review. As the
project matures, we will adopt a more formal governance model
(probably similar to LeWorldModel's, since the projects are siblings).

### What's the license?

Apache 2.0. See [LICENSE](https://github.com/AbdelStark/GenoLeWM/blob/main/LICENSE).
