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

### What does the receipt verify?

The receipt verifies schema validity, model manifest identity, optional
input commitment, and output commitment. It is reproducibility and
tamper-detection plumbing, not a model-quality or runtime-assurance
guarantee.

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

The v0.1 release reports a narrow chr21 ClinVar slice: 3,000 variants,
AUROC `0.5191596847727398`, AP `0.1651739690365932`, and balanced
accuracy `0.5`. The June 8 v0.2 readiness run adds measured
GenoLeWM-vs-Carbon evidence for ClinVar coding, ClinVar non-coding,
BRCA2 (Findlay et al. saturation editing), and TraitGym Mendelian, with
exact evaluated variant identities and negative findings. The June 9
#203 rerun applied that suite to the #202 checkpoint lineage and still
supports a negative-results/systems framing rather than an
improved-results claim. Those results are benchmark evidence, not
clinical utility claims or a public v0.2 model release. The generated
#205 serious-completion paper package records that framing in
[`paper.serious-completion.md`](https://huggingface.co/abdelstark/geno-lewm-runs/resolve/main/geno-lewm-v021-strong-4f36eef-10k-r1/paper/paper.serious-completion.md),
with an `ok=true`
[`paper_package_report.json`](https://huggingface.co/abdelstark/geno-lewm-runs/resolve/main/geno-lewm-v021-strong-4f36eef-10k-r1/paper/paper_package_report.json).
See
[RFC-0007](rfcs/0007-evaluation-suite.md).

### How does it handle structural variants?

It doesn't, in v1. Edits are capped at 16 bp for both ref and alt. A
separate SV adapter is planned for v2 ([RFC-0003 §3.5](rfcs/0003-action-representation-genomic-edits.md)).
The >95% short-variant coverage note is a design rationale, not a
measured GenoLeWM dataset result; the first dataset snapshot must report
the observed retained/excluded variant counts.

### What populations is the calibration valid for?

The v0.1 model package includes a public `calibration.parquet`, but the
release does not establish population-stratified calibration validity.
Treat the calibration as part of the released scorer artifact, not as a
population-general reliability claim. Per-population calibration is a
v0.2 workstream and must remain an explicit limitation until measured.

### Can I score a whole VCF?

Yes, with local model artifacts. The v0.1 terminal demo scored a public
32-row chr21 VCF and recorded score/receipt JSONL hashes. The local-only
form is
`geno-lewm-score --model-dir ./model --vcf my.vcf --fasta GRCh38.fa --output scores.jsonl`.
Add `--receipt receipts.jsonl` to write one checksum receipt per scored
alternate.
See [RFC-0010 §3.5](rfcs/0010-on-device-personal-genome-deployment.md).

### How does the surprise score compare to CADD / AlphaMissense / Carbon-likelihood?

Different signal. CADD aggregates many features; AlphaMissense is
trained on protein-structure context; Carbon's likelihood is the
log-probability difference of `alt` vs `ref` under an autoregressive DNA
model. Surprise is the predictor's residual: how much the predicted
post-edit latent differs from the actual post-edit latent. The v0.1
release reports near-chance first-release metrics, not a robust
comparison against those scorers. v0.2 should report correlations,
baselines, and negative findings before making comparison claims.

### Does the model know about non-coding variants?

The architecture is designed to use Carbon's non-coding sequence
representations, and the GenoLeWM training mix (RFC-0006) draws from the
full corpus. The v0.1 release does not establish separate non-coding
performance; the June 8 and June 9 v0.2 readiness runs report coding
and non-coding ClinVar splits separately so non-coding performance is
visible, with the non-coding row preserved as a negative finding.

### Why centered-mean pooling around the edit?

To make the state vector edit-local. A globally-pooled state would
dilute the edit's effect across 12,288 bp; the centered mean
concentrates the state around the ~3 kbp where edits actually act.
See [RFC-0002 §3.4](rfcs/0002-state-encoder-carbon-integration.md).

### Is GenoLeWM applicable to non-human genomes?

Carbon's pretraining is predominantly eukaryotic with some prokaryotic
content; the architecture is species-agnostic. GenoLeWM has not been
evaluated on model organisms. Extending the eval suite to mouse, fly, or
yeast benchmarks is planned for v2.

---

## For the personal-health-AI community

### Does the model see my data?

The intended release path is local-only: scoring should run on your
device and fail closed if an inference path attempts a network call.
The v0.1 terminal demo records runtime-preflight and network-guard
evidence for the released demo path. That is artifact replay evidence,
not a general privacy assurance for every future runtime. See
[RFC-0010 §3.7](rfcs/0010-on-device-personal-genome-deployment.md).

### Does Anthropic / Hugging Face / anyone see what I score?

The project does not provide a hosted scoring service or telemetry
pipeline. The intended release contract keeps scoring local; network use
is limited to explicit setup/update actions such as downloading model
artifacts. The v0.1 demo records this behavior for the released demo
path; new runtime paths must preserve the same boundary before docs can
extend the claim.

### Can I run this on a laptop?

That remains a target, not a broad published benchmark result. The v0.1
release includes an `efficiency_report.json` for the first artifact set,
but the Apple Silicon and quantized local runtime targets still need
dedicated measurement before users should treat RFC-0010 budget numbers
as achieved. See
[RFC-0010 §3.5](rfcs/0010-on-device-personal-genome-deployment.md).

### Can I run it on Windows / Linux?

The planned v1 runtime targets Apple Silicon and CUDA workstations
(RTX 4090-class). CPU-only is an accessibility fallback target. Public
support should be judged from the v0.1 release demo and measured
efficiency report for the published artifact set, plus future v0.2
platform-specific benchmarks.

### What if a new GenoLeWM version gives different scores?

By design, users should be able to roll back. Updates are explicit, and
previous model versions are intended to remain side-by-side installs.
The public v0.1 checkpoint has a manifest identity and checksum receipt
stream for the terminal demo. Reproducible replay beyond the released
demo inputs should be validated with the same manifest and receipt
contracts. See
[RFC-0010 §3.8](rfcs/0010-on-device-personal-genome-deployment.md).

### Can someone check that my score matches the released artifacts?

Yes, via a checksum-only receipt. Anyone with the same model artifacts
and the same input can check the manifest identity and commitments, and
can re-run inference when they have the required local runtime and input
artifacts.
See [RFC-0011](rfcs/0011-artifact-provenance-receipts.md).

### Will there be an iOS app?

Plausibly feasible technically; not in v1. Apple Silicon laptop is
the v1 target. iOS is a v2 consideration.

### Will there be a hosted version?

Not from us. The whole architecture is local-first. If the community
builds a hosted variant from the open weights, that's their choice;
GenoLeWM the project will remain on-device-first.

### How do I import my 23andMe / AncestryDNA / MyHeritage data?

The release design accepts those formats and converts them locally to
VCF for scoring. The v0.1 terminal demo proves public VCF/FASTA scoring
from released artifacts, not every personal-genome import workflow. See
[RFC-0010 §3.9](rfcs/0010-on-device-personal-genome-deployment.md).

### Should I act on a high surprise score?

Talk to a genetic counselor. GenoLeWM is a research tool; its surprise
score is a single signal among many that clinical genetics integrates.
A high score warrants follow-up by a qualified professional, not direct
action.

---

## For contributors

### How do I contribute?

The project is in alpha implementation. The most valuable contributions
now are narrow PRs that move v0.2 benchmark and rollout readiness
forward: broader held-out data/eval builders, Carbon-baseline deltas,
rollout speed gates, planning APIs/CLI, first PyPI tag work, and docs
that keep claims tied to measured evidence.

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
