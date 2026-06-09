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

# GenoLeWM checkpoint and evidence bundle

<p>
  <a href="https://huggingface.co/spaces/abdelstark/geno-lewm"><img alt="Space" src="https://img.shields.io/badge/Space-GenoLeWM-FFD21E?style=for-the-badge&logo=huggingface&logoColor=000000"></a>
  <a href="https://huggingface.co/abdelstark/geno-lewm"><img alt="Checkpoint" src="https://img.shields.io/badge/Checkpoint-abdelstark%2Fgeno--lewm-FFD21E?style=for-the-badge&logo=huggingface&logoColor=000000"></a>
  <a href="https://huggingface.co/datasets/abdelstark/geno-lewm-data"><img alt="Dataset" src="https://img.shields.io/badge/Dataset-abdelstark%2Fgeno--lewm--data-0B7285?style=for-the-badge&logo=huggingface&logoColor=ffffff"></a>
  <a href="https://github.com/AbdelStark/GenoLeWM"><img alt="GitHub" src="https://img.shields.io/badge/GitHub-GenoLeWM-181717?style=for-the-badge&logo=github&logoColor=ffffff"></a>
</p>

GenoLeWM is an alpha research system for action-conditioned latent world models
over genomic edits. This model repository publishes the stable v0.1 checkpoint
package: trainable GenoLeWM predictor and action-encoder weights, calibration
data, manifest-bound hashes, training evidence, evaluation evidence, and release
checksums.

This is not a standard `transformers.AutoModel.from_pretrained()` checkpoint.
Load it through the `geno-lewm` runtime. Carbon-500M is a frozen state encoder
dependency and is resolved separately from this repository.

## Claim Boundary

Use this checkpoint for reproducible research, local artifact inspection,
method development, and demonstration scoring. Do not use it as a diagnostic
model, clinical decision-support system, deployment-readiness claim, privacy
claim, or evidence of broad superiority over Carbon. The measured results below
are narrow artifact-level evaluations, and the broader v0.2.1 run-tree results
are mixed or negative against the measured baselines.

No clinical utility claim is made.

## At A Glance

| Item | Value |
| --- | --- |
| Stable package in this repo | `geno-lewm-v0.1.0-r1` |
| Newer run-tree checkpoint used by the Space | `geno-lewm-v0.2.1-r1` under `abdelstark/geno-lewm-runs` |
| Runtime | `geno-lewm` package, manifest verification, local Carbon-500M encoder |
| Main task | Single-SNV surprise scoring over reference windows |
| Strongest honest conclusion | Reproducible systems artifact with measured negative and mixed model-quality findings |
| Not supported | Clinical utility, deployment readiness, privacy assurance, runtime assurance, broad model superiority |

## Which Checkpoint Should I Use?

| Target | Location | Use when |
| --- | --- | --- |
| Stable package | [`abdelstark/geno-lewm`](https://huggingface.co/abdelstark/geno-lewm) | You need the checksum-bound v0.1 release package with a stable manifest and package-level evidence. |
| Space/default demo checkpoint | [`geno-lewm-v021-strong-4f36eef-10k-r1/suite/model`](https://huggingface.co/abdelstark/geno-lewm-runs/tree/main/geno-lewm-v021-strong-4f36eef-10k-r1/suite/model) | You want the newer v0.2.1 checkpoint used by the Hugging Face Space and the broader benchmark/planning evidence. |
| Interactive demo | [`abdelstark/geno-lewm` Space](https://huggingface.co/spaces/abdelstark/geno-lewm) | You want to inspect artifacts or score a compatible single variant through the hosted UI. |

## Published Artifacts

| Artifact | Location | Notes |
| --- | --- | --- |
| Stable model package | this repository | Public package `geno-lewm-v0.1.0-r1`. |
| Generated package card | [`model_card.md`](https://huggingface.co/abdelstark/geno-lewm/blob/main/model_card.md) | Checksum-bound output from `tools.release.model_package`; kept terse so it can be re-rendered exactly. |
| Manifest | [`manifest.json`](https://huggingface.co/abdelstark/geno-lewm/blob/main/manifest.json) | Canonical identity for weights, calibration, config, encoder, and eval report. |
| Package checksums | [`SHA256SUMS`](https://huggingface.co/abdelstark/geno-lewm/blob/main/SHA256SUMS) | Hashes for the v0.1 release package files. |
| Training evidence | [`training_run_manifest.json`](https://huggingface.co/abdelstark/geno-lewm/blob/main/training_run_manifest.json), [`training_run_card.md`](https://huggingface.co/abdelstark/geno-lewm/blob/main/training_run_card.md), [`training_run_SHA256SUMS`](https://huggingface.co/abdelstark/geno-lewm/blob/main/training_run_SHA256SUMS) | Carbon-backed training-run metadata and hashes. |
| Evaluation evidence | [`eval_metrics.json`](https://huggingface.co/abdelstark/geno-lewm/blob/main/eval_metrics.json), [`eval_report.md`](https://huggingface.co/abdelstark/geno-lewm/blob/main/eval_report.md), [`eval_config.effective.yaml`](https://huggingface.co/abdelstark/geno-lewm/blob/main/eval_config.effective.yaml) | Held-out chr21 ClinVar evaluation for v0.1. |
| Efficiency evidence | [`efficiency_report.json`](https://huggingface.co/abdelstark/geno-lewm/blob/main/efficiency_report.json) | Release efficiency measurement for v0.1. |
| Dataset package | [`abdelstark/geno-lewm-data`](https://huggingface.co/datasets/abdelstark/geno-lewm-data) | Public data snapshot and data card. |
| v0.2.1 run tree | [`abdelstark/geno-lewm-runs`](https://huggingface.co/abdelstark/geno-lewm-runs/tree/main/geno-lewm-v021-strong-4f36eef-10k-r1) | Newer checkpoint, broader benchmark suite, planning demo, and generated paper. |
| Generated paper | [`paper.serious-completion.md`](https://huggingface.co/abdelstark/geno-lewm-runs/resolve/main/geno-lewm-v021-strong-4f36eef-10k-r1/paper/paper.serious-completion.md) | Manuscript-style synthesis of experiments, learnings, negative findings, and benchmark limits. |

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

The newer Space default checkpoint has model id
`sha256:cddb8f3b9671090201370b9824b9da741b933ff296b651238f022df5f3ed6af4`
and lives in the v0.2.1 run tree. It is run evidence, not a replacement for
the stable v0.1 package in this repository.

## Method

GenoLeWM treats an edit as an action over a frozen genomic state embedding.
For a reference window, Carbon-500M encodes the pre-edit state. GenoLeWM's
trainable action encoder represents the edit, and the predictor estimates the
post-edit latent state. Surprise scores are derived from the relationship
between predicted and observed post-edit states, then calibrated by the
packaged calibration artifact.

The package intentionally separates:

- frozen state encoder: Carbon-500M, loaded separately;
- trainable GenoLeWM weights: `predictor.safetensors` and
  `action_encoder.safetensors`;
- release identity: `manifest.json` and `SHA256SUMS`;
- evidence: training, evaluation, efficiency, and run-tree reports.

## Install And Load

Install the package:

```bash
python -m pip install "geno-lewm[train,eval]==0.2.1"
```

Download the stable v0.1 model package:

```python
from huggingface_hub import snapshot_download

model_dir = snapshot_download("abdelstark/geno-lewm")
```

Download the newer v0.2.1 run-tree model used by the Space:

```python
from pathlib import Path
from huggingface_hub import snapshot_download

prefix = "geno-lewm-v021-strong-4f36eef-10k-r1/suite/model"
run_root = snapshot_download(
    "abdelstark/geno-lewm-runs",
    allow_patterns=f"{prefix}/*",
)
model_dir = Path(run_root) / prefix
```

Carbon-500M must also be available. The release manifests record the encoder as
`/carbon` because training, evaluation, and demo jobs mounted
`HuggingFaceBio/Carbon-500M` there at revision
`5d31d59b3c845b288a13aedb1358934196852eec`. The Space resolves and remaps that
encoder from the Hub before scoring.

## Scoring Contract

The scorer is strict about reference consistency. The `REF` allele in
`chrom:pos:ref:alt` must match the supplied reference window at the one-based
locus implied by `pos` and `--window-start-bp`. If it does not match, scoring
raises `WindowMismatchError` before model inference.

Example single-variant score with a synthetic reference-matched window:

```bash
geno-lewm-score \
  --model-dir "$MODEL_DIR" \
  --backend auto \
  --variant chrSynthetic:3073:A:T \
  --window "$(python - <<'PY'
print("ACGT" * 3072)
PY
)" \
  --window-start-bp 0 \
  --receipt receipt.json
```

For real variants, use a FASTA-backed VCF path so the runtime builds windows
from the same reference assembly as the variant coordinates.

## v0.1 Training Summary

The stable v0.1 checkpoint was trained as a JEPA-style predictor over frozen
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

Interpretation: this v0.1 slice does not establish useful clinical
performance, non-coding performance, multi-edit behavior, or superiority over
Carbon.

## v0.1 Efficiency

Measured by `tools.release.efficiency_report` on `cuda:NVIDIA H200`.

| Measurement | Value |
| --- | ---: |
| Single-variant latency | 494.056 ms |
| Batched throughput | 2.024 variants/s |
| Peak memory | 1,152,656,384 bytes |

## v0.2.1 Benchmark Evidence

The v0.2.1 run tree includes broader benchmark coverage and Carbon zero-shot
comparisons. Results remain mixed or negative relative to the measured
baselines.

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
That is not a model-quality success claim. The autoregressive rollout speed
report is `ok=false`: K=5 measured 2.41x speedup against a 2x target, while
K=20 measured 2.47x against a 5x target and missed the target.

The v0.2.1 efficiency report measured one sample with no warmup on
`cuda:NVIDIA H200`: 115,262.94 ms single-variant latency, 0.3095 variants/s
throughput, and 1,966,149,632 bytes peak memory. Treat that as run evidence,
not a production serving benchmark.

## Planning Demo Evidence

The v0.2.1 run tree includes a deterministic synthetic multi-SNV planning demo.
It ran with manifest-backed runtime evaluation and recorded 384 evaluations,
`best_distance=23.656930390534644`, and `stopped_reason=patience`.

This is not useful-planning evidence. It shows that the planner, checkpoint,
window artifacts, and receipt path run end to end; it does not show biological
or clinical planning utility.

## Troubleshooting

| Symptom | Likely cause | Fix |
| --- | --- | --- |
| `WindowMismatchError: window bases do not match edit.ref_bases at locus` | The variant `REF` allele does not match the supplied window at the requested coordinate. | Use a FASTA-backed VCF path or correct `--window`, `--window-start-bp`, and the variant `REF` allele so they describe the same reference sequence. |
| Space says scoring did not complete and mentions Carbon-500M | The runtime could not resolve or mount the frozen Carbon encoder, or optional ML dependencies failed to load. | Inspect the artifact panel, reload the Space after dependency startup, or run locally with Carbon-500M available. |
| `AutoModel.from_pretrained("abdelstark/geno-lewm")` fails | This repo is a GenoLeWM package, not a native Transformers architecture repo. | Use `snapshot_download` plus `geno-lewm-score` or the `geno_lewm.deploy` runtime. |

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

## Citation And Reports

- Paper: [`paper.serious-completion.md`](https://huggingface.co/abdelstark/geno-lewm-runs/resolve/main/geno-lewm-v021-strong-4f36eef-10k-r1/paper/paper.serious-completion.md)
- Paper verifier: [`paper_package_report.json`](https://huggingface.co/abdelstark/geno-lewm-runs/resolve/main/geno-lewm-v021-strong-4f36eef-10k-r1/paper/paper_package_report.json)
- Project docs: <https://abdelstark.github.io/GenoLeWM/>
- Source repository: <https://github.com/AbdelStark/GenoLeWM>

## License

Apache-2.0 for GenoLeWM source and metadata. Upstream Carbon-500M and dataset
terms apply to their respective artifacts.
