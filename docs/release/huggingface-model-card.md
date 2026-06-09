---
license: apache-2.0
language:
  - en
library_name: geno-lewm
base_model:
  - HuggingFaceBio/Carbon-500M
datasets:
  - abdelstark/geno-lewm-data
tags:
  - genomics
  - bioinformatics
  - variant-effect-prediction
  - world-model
  - carbon-500m
  - research
---

# GenoLeWM model package

<p>
  <a href="https://huggingface.co/spaces/abdelstark/geno-lewm"><img alt="Space" src="https://img.shields.io/badge/Space-GenoLeWM-FFD21E?style=for-the-badge&logo=huggingface&logoColor=000000"></a>
  <a href="https://huggingface.co/abdelstark/geno-lewm"><img alt="Model" src="https://img.shields.io/badge/Checkpoint-abdelstark%2Fgeno--lewm-FFD21E?style=for-the-badge&logo=huggingface&logoColor=000000"></a>
  <a href="https://huggingface.co/abdelstark/geno-lewm-runs/tree/main/geno-lewm-v021-strong-4f36eef-10k-r1"><img alt="Run tree" src="https://img.shields.io/badge/Run%20Tree-v0.2.1-0B7285?style=for-the-badge&logo=huggingface&logoColor=ffffff"></a>
  <a href="https://github.com/AbdelStark/GenoLeWM"><img alt="GitHub" src="https://img.shields.io/badge/GitHub-GenoLeWM-181717?style=for-the-badge&logo=github&logoColor=ffffff"></a>
</p>

GenoLeWM is an alpha research project for action-conditioned latent world
models over genomic edits. This repository contains the public v0.1 model
package: the trainable GenoLeWM predictor/action-encoder artifacts, calibration
file, training evidence, evaluation evidence, and checksums.

This is not a standard `transformers.AutoModel.from_pretrained()` package. The
checkpoint is loaded by the `geno-lewm` runtime. Carbon-500M is a frozen state
encoder dependency and is not bundled in this repository.

## Claim Boundary

Use this checkpoint as a research artifact for reproducible local scoring,
artifact inspection, and method development. Do not use it for clinical
diagnosis, clinical decision support, deployment readiness claims, privacy
claims, or broad claims that GenoLeWM outperforms Carbon. The measured results
below are narrow artifact-level evaluations.

## Published Artifacts

| Artifact | Location | Notes |
| --- | --- | --- |
| v0.1 release checkpoint | this repository | Stable public package `geno-lewm-v0.1.0-r1` |
| Generated package model card | [`model_card.md`](https://huggingface.co/abdelstark/geno-lewm/blob/main/model_card.md) | Checksum-bound output from `tools.release.model_package` |
| Training evidence | [`training_run_manifest.json`](https://huggingface.co/abdelstark/geno-lewm/blob/main/training_run_manifest.json), [`training_run_card.md`](https://huggingface.co/abdelstark/geno-lewm/blob/main/training_run_card.md), [`training_run_SHA256SUMS`](https://huggingface.co/abdelstark/geno-lewm/blob/main/training_run_SHA256SUMS) | Carbon-backed training run evidence |
| Evaluation evidence | [`eval_metrics.json`](https://huggingface.co/abdelstark/geno-lewm/blob/main/eval_metrics.json), [`eval_report.md`](https://huggingface.co/abdelstark/geno-lewm/blob/main/eval_report.md), [`eval_config.effective.yaml`](https://huggingface.co/abdelstark/geno-lewm/blob/main/eval_config.effective.yaml) | Held-out chr21 ClinVar evaluation |
| Efficiency evidence | [`efficiency_report.json`](https://huggingface.co/abdelstark/geno-lewm/blob/main/efficiency_report.json) | Release efficiency measurement |
| Integrity manifest | [`SHA256SUMS`](https://huggingface.co/abdelstark/geno-lewm/blob/main/SHA256SUMS) | Package file hashes |
| Interactive Space | [`abdelstark/geno-lewm`](https://huggingface.co/spaces/abdelstark/geno-lewm) | Artifact browser and checkpoint-backed scoring UI |
| Dataset package | [`abdelstark/geno-lewm-data`](https://huggingface.co/datasets/abdelstark/geno-lewm-data) | Public data snapshot and data card |
| v0.2.1 run tree | [`abdelstark/geno-lewm-runs`](https://huggingface.co/abdelstark/geno-lewm-runs/tree/main/geno-lewm-v021-strong-4f36eef-10k-r1) | Newer benchmark/demo checkpoint and result artifacts |

The generated `model_card.md` in this repository is intentionally terse because
it is part of the checksum-bound release package. This top-level card is the
human-facing Hugging Face model documentation.

## Model Identity

| Field | Value |
| --- | --- |
| Release id | `geno-lewm-v0.1.0-r1` |
| Model version | `0.1.0` |
| Manifest id | `sha256:861ec142cc87f3fac01751ef538553356dfba439e6da99064b4adb121e75c215` |
| Predictor artifact | `predictor.safetensors` |
| Predictor hash | `sha256:6642c604a1352727969c86664f291fd6d2193c1c65bc6f9baf9b716469c52731` |
| Action encoder hash | `sha256:8b2311d768855ab440b26dbbef5ddbda252cc8bb2c69509d28fa4bcf8eff025a` |
| Calibration hash | `sha256:d4cf4778ac8e5557d363aca43cd13723b0ed9983b83215ab164d2b642b886201` |
| Frozen encoder | Carbon-500M, mounted as `/carbon` in release jobs |
| Encoder revision | `5d31d59b3c845b288a13aedb1358934196852eec` |
| Dataset snapshot | `geno-lewm-data-v0.1.0-r1` |

The newer Space default checkpoint is separate:
`geno-lewm-v0.2.1-r1` in
[`geno-lewm-v021-strong-4f36eef-10k-r1/suite/model`](https://huggingface.co/abdelstark/geno-lewm-runs/tree/main/geno-lewm-v021-strong-4f36eef-10k-r1/suite/model).
It is published as run-tree evidence, not as a replacement for this stable v0.1
model package.

## Training Summary

The v0.1 checkpoint was trained as a JEPA-style predictor over frozen
Carbon-500M latent states.

| Field | Value |
| --- | --- |
| Run id | `first-snv-carbon-500m-r1` |
| Config | `training_config.effective.yaml` |
| Commit | `cd2bfccb33ec5a2df3c4707e8be8443f4682dad3` |
| Samples | 160,000 |
| Steps | 20,000 |
| Final training loss | 0.36124 |
| Status | completed |

## v0.1 Evaluation

Held-out ClinVar GRCh38 chr21, binary P/LP versus B/LB labels. Scores use
`sigma_raw`; intervals are deterministic stratified bootstrap confidence
intervals from `eval_metrics.json`.

| Split | N | Positives | Negatives | Metric | Value | 95% CI |
| --- | ---: | ---: | ---: | --- | ---: | --- |
| `eval_clinvar_chr21` | 3,000 | 494 | 2,506 | AUROC | 0.519160 | 0.491366 to 0.546846 |
| `eval_clinvar_chr21` | 3,000 | 494 | 2,506 | Average precision | 0.165174 | 0.155331 to 0.177035 |
| `eval_clinvar_chr21` | 3,000 | 494 | 2,506 | Balanced accuracy at 0.5 | 0.500000 | 0.500000 to 0.500000 |
| `eval_clinvar_chr21` | 3,000 | 494 | 2,506 | Accuracy at 0.5 | 0.164667 | 0.164667 to 0.164667 |

Negative finding: this v0.1 slice does not establish useful clinical
performance, non-coding performance, multi-edit behavior, or superiority over
Carbon.

## v0.1 Efficiency

Measured by `tools.release.efficiency_report` on `cuda:NVIDIA H200`.

| Measurement | Value |
| --- | ---: |
| Single-variant latency | 494.056 ms |
| Batched throughput | 2.024 variants/s |
| Peak memory | 1,152,656,384 bytes |

## v0.2.1 Run-Tree Benchmark Evidence

The Space also exposes the newer `geno-lewm-v0.2.1-r1` checkpoint from the run
tree. Its benchmark suite is broader than v0.1 and includes Carbon zero-shot
comparisons, but the results are mixed and mostly negative relative to Carbon on
the measured slices.

| Slice | N | Metric | GenoLeWM | Baseline | Delta |
| --- | ---: | --- | ---: | ---: | ---: |
| ClinVar coding | 16 | AUROC | 0.734375 | 0.921875 | -0.187500 |
| ClinVar coding | 16 | Average precision | 0.852976 | 0.951923 | -0.098947 |
| ClinVar coding | 16 | Balanced accuracy | 0.750000 | 0.687500 | +0.062500 |
| ClinVar non-coding | 16 | AUROC | 0.562500 | 0.875000 | -0.312500 |
| ClinVar non-coding | 16 | Average precision | 0.605456 | 0.914423 | -0.308967 |
| ClinVar non-coding | 16 | Balanced accuracy | 0.437500 | 0.687500 | -0.250000 |
| BRCA2 saturation | 32 | Spearman rho | 0.149194 | 0.476906 | -0.327713 |
| TraitGym Mendelian | 32 | Spearman rho | -0.027965 | -0.083894 | +0.055929 |
| Phased-haplotype rollout | 8 | Cosine mean | 0.288861 | 0.997831 | -0.708970 |
| Synthetic edit-chain rollout | 8 | Cosine mean | 0.301608 | 0.991240 | -0.689631 |

The v0.2.1 readiness report is `ok=true` for artifact coverage and provenance.
That is not a model-quality success claim. The rollout speed report is
`ok=false`: k=5 measured 2.41x speedup against a 2x target, while k=20 measured
2.47x against a 5x target and missed the target.

The v0.2.1 efficiency report measured one sample with no warmup on
`cuda:NVIDIA H200`: 115,262.94 ms single-variant latency, 0.3095 variants/s
throughput, and 1,966,149,632 bytes peak memory. Treat that as run evidence, not
a production serving benchmark.

## Loading Artifacts

Install the package:

```bash
python -m pip install "geno-lewm[train,eval]==0.2.1"
```

Download the v0.1 model package:

```python
from huggingface_hub import snapshot_download

model_dir = snapshot_download("abdelstark/geno-lewm")
```

Download the v0.2.1 run-tree model artifacts:

```python
from huggingface_hub import snapshot_download

run_dir = snapshot_download(
    "abdelstark/geno-lewm-runs",
    allow_patterns="geno-lewm-v021-strong-4f36eef-10k-r1/suite/model/*",
)
```

For scoring, Carbon-500M must also be available. The release manifests record
the encoder as `/carbon` because training, evaluation, and demo jobs mounted
`HuggingFaceBio/Carbon-500M` there at revision
`5d31d59b3c845b288a13aedb1358934196852eec`. The Space can resolve and remap
that encoder from the Hub before scoring.

Example single-variant invocation once the model directory and Carbon encoder
are available:

```bash
geno-lewm-score \
  --model-dir "$MODEL_DIR" \
  --backend auto \
  --variant chrSynthetic:3073:A:T \
  --window ACGTACGTACGTACGT \
  --window-start-bp 3064 \
  --receipt receipt.json
```

The `REF` allele in `--variant` must match the supplied reference window at the
variant locus. If it does not, scoring fails before model inference.

## Limitations

- Alpha research checkpoint; not a clinical, diagnostic, or deployment model.
- v0.1 evaluation is narrow: held-out chr21 ClinVar P/LP versus B/LB labels.
- v0.2.1 benchmark evidence is broader but mixed, with multiple negative deltas
  versus Carbon zero-shot and source-state rollout baselines.
- Carbon-500M is required at runtime and is resolved separately from this model
  package.
- Calibration is proof-scale and should be interpreted only within the reported
  artifact context.
- Fixture outputs and UI demos are not model-quality evidence.

## License

Apache-2.0.
