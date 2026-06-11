# Frequently Asked Questions

## Is GenoLeWM a clinical tool?

No. GenoLeWM is alpha research software. Scores are research signals,
not clinical diagnoses, clinical risk probabilities, or medical advice.

## What does the model do?

GenoLeWM treats a genomic edit as an action. Carbon encodes a reference
DNA window into a latent state, the action encoder embeds the edit, and
the predictor estimates the post-edit latent state.

## What is Carbon's role?

Carbon-500M is the frozen state encoder in the released path. GenoLeWM
does not retrain Carbon. The trainable components are the action encoder
and predictor.

## What does `sigma_raw` mean?

`sigma_raw` is the uncalibrated latent residual between the predicted
post-edit state and the Carbon-encoded edited state. Larger values mean
the edit was more surprising to the learned predictor in that context.
It is not a probability of pathogenicity.

## What does `sigma_calibrated` mean?

`sigma_calibrated` maps `sigma_raw` through the released calibration
table. It is bounded between 0 and 1 and should be treated as a
contextual research score. Low-confidence calibration rows are marked in
the output.

## What does a checksum receipt verify?

A receipt binds model manifest identity, optional input commitment, and
output commitment. It supports reproducible artifact inspection and
tamper detection. It does not prove model quality, clinical validity,
privacy, or runtime behavior.

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

The Hugging Face Space is a public artifact console and research demo:
<https://huggingface.co/spaces/abdelstark/geno-lewm>. It can inspect
artifacts and attempt a compatible single-variant score. Do not use it
for private genome data.

## Does GenoLeWM beat Carbon?

No broad superiority claim is supported. Current v0.2.1 evidence is
mixed or negative versus Carbon on most benchmark rows. The generated
paper records this as a negative-results and systems-evidence story.
The public run tree includes exact evaluated identities for ClinVar
coding, ClinVar non-coding, BRCA2 saturation-editing, and TraitGym
Mendelian rows; treat those as benchmark evidence, not clinical utility
claims.

## What does the planning demo prove?

It proves that the released manifest-backed planning path can execute
against public artifacts. It does not prove useful edit selection or
biological design capability.

## Does the model support structural variants?

The released scoring and training surfaces focus on short variants.
Large structural variant support is not established by the public
release.

## Does the model keep my data private?

The intended runtime is local-first: local VCF/FASTA inputs, local model
artifacts, local output paths, no telemetry by default, and redacted
logging. That is a design and implementation boundary, not a general
privacy certification.

## Where are the public artifacts?

- Package: <https://pypi.org/project/geno-lewm/0.2.1/>
- Source/wheel release: <https://github.com/AbdelStark/GenoLeWM/releases/tag/v0.2.1>
- Model package: <https://huggingface.co/abdelstark/geno-lewm>
- Dataset package: <https://huggingface.co/datasets/abdelstark/geno-lewm-data>
- Run tree and generated paper: <https://huggingface.co/abdelstark/geno-lewm-runs/tree/main/geno-lewm-v021-strong-4f36eef-10k-r1>

## How do I contribute?

Open a focused issue or pull request with tests and validation. Public
docs should stay tied to measured artifacts and should preserve the
current limitations.
