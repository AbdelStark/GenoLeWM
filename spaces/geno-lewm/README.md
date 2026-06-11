---
title: GenoLeWM
colorFrom: green
colorTo: gray
sdk: gradio
sdk_version: 5.49.1
app_file: app.py
python_version: "3.10"
suggested_hardware: t4-small
license: apache-2.0
models:
  - abdelstark/geno-lewm
  - abdelstark/geno-lewm-runs
datasets:
  - abdelstark/geno-lewm-data
tags:
  - genomics
  - world-model
  - variant-effect-prediction
  - bioinformatics
  - gradio
---

# GenoLeWM

Action-conditioned latent world models for genomic edits.

This Space is the public project console for GenoLeWM. It shows the
released artifacts, measured benchmark results, model limitations, and a
checkpoint-backed scoring path.

## What This Space Does

- Shows the current model, dataset, benchmark, planning, and paper
  artifacts.
- Summarizes the v0.2.1 benchmark evidence without hiding negative
  findings.
- Downloads the released trained checkpoint files on demand.
- Verifies that the action encoder and predictor weights can be loaded.
- Attempts single-variant scoring with the checkpoint when the Space
  runtime can also resolve the pinned Carbon-500M encoder.
- Uses a synthetic, sequence-consistent default scoring example. For real
  variants, the supplied `REF` allele must match the pasted FASTA reference
  window at the relative locus; the app checks this before loading the model.

## How To Read The Model-Run Screen

The **Checkpoint** tab is the model-run screen. It first materializes the
released checkpoint artifacts, then the scorer validates a
`CHROM:POS:REF:ALT` edit against the supplied reference window. If the
reference allele does not match the sequence at the implied offset, the app
stops before model inference.

Successful runs return a JSON payload:

- `sigma_raw` is the uncalibrated latent residual between the predicted
  post-edit state and the Carbon-encoded edited state. Treat it as a
  research/debug ranking signal; it is not a probability of pathogenicity.
- `sigma_calibrated` maps that residual through the released calibration
  table. Higher means more surprising relative to that calibration background;
  it is not a clinical risk score.
- `bucket_id`, `confidence`, and `low_confidence` describe the calibration
  context. A low-confidence result should be treated as especially tentative.
- `input_preflight` records the parsed variant, relative offset, observed
  reference base, and window length used for the strict input check.
- `runtime_note` explains whether Carbon-500M was remapped from the Hub, and
  `receipt_path` points to the checksum receipt for artifact/output identity.

## Boundaries

- No clinical utility claim.
- No broad claim that GenoLeWM beats Carbon.
- Fixture or demo outputs are not model-quality evidence.
- Checksum provenance covers artifact identity; it is not a runtime
  assurance system.

Full package documentation: <https://abdelstark.github.io/GenoLeWM/>
