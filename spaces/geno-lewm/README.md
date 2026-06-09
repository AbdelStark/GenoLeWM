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

## Boundaries

- No clinical utility claim.
- No broad claim that GenoLeWM beats Carbon.
- Fixture or demo outputs are not model-quality evidence.
- Checksum provenance covers artifact identity; it is not a runtime
  assurance system.

Full package documentation: <https://abdelstark.github.io/GenoLeWM/>
