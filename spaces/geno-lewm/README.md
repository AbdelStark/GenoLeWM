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

This Space is the public project console for GenoLeWM. It shows released
artifacts, historical implementation outputs, and model limitations. Legacy
scientific scoring is disabled after the 2026-07-10 state-contract audit.

Every published checkpoint uses `legacy_raw_v1`: raw pooled Carbon
source/target states were combined with unit-normalized predictions. That
mismatch invalidates released L2 residuals, VEP/calibration scores, and the
planning objective. Training sources were also globally pooled while targets
were edit-centered; every historical centered pool was shifted one hidden token
left because the leading `<dna>` token was omitted, rollout/candidate centers
differed, and cache v1 omitted that identity. The pinned Carbon tokenizer also
made an unpinned, network-capable `Qwen/Qwen3-4B-Base` lookup, so the historical
runtime was not self-contained. Released cosine values remain reproducible but are
confounded by invalid training. The v0.2.1 Phase 2 KL also had no gradient
path to any trainable parameter. Corrected results require a fresh
`l2_normalized_v2` lineage.

The source-level pure-DNA tokenizer and token-layout-aware centering repairs do
not provide a corrected checkpoint or model-quality evidence.

## What This Space Does

- Shows the current model, dataset, benchmark, planning, and paper
  artifacts.
- Displays v0.2.1 rows as historical outputs without reusing their withdrawn
  positive or negative interpretation.
- Downloads the released trained checkpoint files on demand.
- Verifies that the action encoder and predictor weights can be loaded.
- Displays the legacy checkpoint identity and declared encoder configuration
  for provenance inspection.
- Does not run single-variant scientific scoring with published checkpoints.

## How To Read The Checkpoint Screen

The **Checkpoint** tab materializes the released artifacts and can verify that
the action encoder and predictor weights load. This is an integrity and
compatibility check, not model-quality or scientific-score evidence.

Historical output fields may still appear in downloaded reports:

- `sigma_raw` is the uncalibrated latent residual between the predicted
  post-edit state and the Carbon-encoded edited state. Published values mix
  incompatible state scales and are invalid as edit-effect rankings.
- `sigma_calibrated` maps that residual through the released calibration
  table. Because the underlying residual is invalid, the legacy calibrated
  value is not a scientific or clinical score.
- `bucket_id`, `confidence`, and `low_confidence` describe the calibration
  context of the historical output; they do not restore validity.

## Boundaries

- No clinical utility claim.
- No superiority or inferiority claim from published legacy metrics.
- No valid scientific scoring with a published checkpoint.
- No valid L2 rollout, VEP/calibration, or planning-objective claim from
  `legacy_raw_v1` artifacts.
- Fixture or demo outputs are not model-quality evidence.
- Checksum provenance covers artifact identity; it is not a runtime
  assurance system.

Full package documentation: <https://abdelstark.github.io/GenoLeWM/>
