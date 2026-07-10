# Frequently Asked Questions

## Is GenoLeWM a clinical tool?

No. GenoLeWM is alpha research software. No corrected scientific score is
currently published, and no output is a clinical diagnosis, clinical risk
probability, or medical advice.

## What does the model do?

GenoLeWM treats a genomic edit as an action. Carbon encodes a reference
DNA window into a latent state, the action encoder embeds the edit, and
the predictor estimates the post-edit latent state.

## What is Carbon's role?

Carbon-500M is the frozen state encoder in the released path. GenoLeWM
does not retrain Carbon. The trainable components are the action encoder
and predictor. The released wrapper executed Carbon's custom tokenizer, whose
pinned source made an unpinned, network-capable
`Qwen/Qwen3-4B-Base` lookup. The corrected source path uses a self-contained
pure-DNA tokenizer built from the mounted Carbon files instead. That repair is
not a corrected model result.

## What does `sigma_raw` mean?

`sigma_raw` is the uncalibrated distance between the predicted post-edit state
and the Carbon-encoded edited state. That definition is meaningful only when
both states share a declared, validated contract. Published checkpoints use
`legacy_raw_v1`, which mixed raw targets with unit-normalized predictions;
training sources and targets also used global versus edit-centered pooling,
and every historical centered pool was shifted one hidden token left by a
missing `<dna>` control-token offset.
Their `sigma_raw` values are invalid as edit-effect or surprise scores. They
are retained only for historical compatibility and are not probabilities of
pathogenicity.

## What does `sigma_calibrated` mean?

`sigma_calibrated` maps `sigma_raw` through a calibration table. The released
table was fitted to invalid mixed-contract residuals, so published calibrated
values are historical outputs, not contextual scientific scores. A future
`l2_normalized_v2` lineage needs a new calibration table and fresh validation.

## What does a checksum receipt verify?

A receipt binds model manifest identity, optional input commitment, and
output commitment. It supports reproducible artifact inspection and
tamper detection. It does not prove model quality, clinical validity,
privacy, or runtime behavior. In particular, hashing the historical Carbon
`tokenizer.py` did not bind the mutable tokenizer it fetched transitively from
`Qwen/Qwen3-4B-Base`.

## Is Carbon tokenization self-contained now?

The current source path used for new `l2_normalized_v2` work implements the
pure-DNA branch locally from Carbon's pinned DNA and tokenizer configuration.
It does not execute the upstream network-capable tokenizer wrapper. It also
validates the `<dna>`/`</dna>` layout and resolves the edit center after the
leading control token. These are fail-closed runtime invariants; no fresh
checkpoint has yet established model quality under them.

## Can I score a VCF?

Yes, with local model artifacts and a local FASTA reference:

```bash
geno-lewm-score \
  --model-dir /path/to/model \
  --backend auto \
  --vcf variants.vcf \
  --fasta reference.fa.gz \
  --output scores.jsonl \
  --receipt receipts.jsonl
```

The scorer requires the reference alleles in the VCF to match the FASTA
windows it extracts.

## Can I run it in the browser?

The Hugging Face Space is a public artifact console:
<https://huggingface.co/spaces/abdelstark/geno-lewm>. It can inspect
artifacts and checkpoint metadata. Legacy scientific scoring is disabled.
Do not use it for private genome data.

## Does GenoLeWM beat Carbon?

The published v0.2.1 comparison cannot answer this question. That run used
`legacy_raw_v1`; its residual-based VEP values are invalid, and its cosine
values are confounded by mixed-scale, mixed-coordinate training and a changed
rollout source representation. The exact rows and
signed deltas remain available for artifact audit, but they support neither a
superiority nor an inferiority claim about `l2_normalized_v2`. No corrected
comparison is currently published.

## What does the planning demo prove?

It proves that the released manifest-backed legacy path executed against
public artifacts. Its L2 objective compared incompatible state scales, so the
operands also lacked a consistently committed pooling center. The reported
objective is invalid and the demo does not prove useful edit
selection or biological design capability.

## Does the model support structural variants?

The released scoring and training surfaces focus on short variants.
Large structural variant support is not established by the public
release.

## Does the model keep my data private?

The current source runtime is local-first: local VCF/FASTA inputs, local model
artifacts, local output paths, no telemetry by default, and redacted logging.
The published lineage predated the self-contained tokenizer repair and could
make the unpinned Qwen tokenizer lookup described above. Local-first is a
design and implementation boundary, not a general privacy certification.

## Where are the public artifacts?

- Package: <https://pypi.org/project/geno-lewm/0.2.1/>
- Source/wheel release: <https://github.com/AbdelStark/GenoLeWM/releases/tag/v0.2.1>
- Model package: <https://huggingface.co/abdelstark/geno-lewm>
- Dataset package: <https://huggingface.co/datasets/abdelstark/geno-lewm-data>
- Run tree and generated paper: <https://huggingface.co/abdelstark/geno-lewm-runs/tree/main/geno-lewm-v021-strong-4f36eef-10k-r1>

## How do I contribute?

Open a focused issue or pull request with tests and validation. Public
docs should stay tied to measured artifacts, explicit state contracts, and
current validity boundaries.
