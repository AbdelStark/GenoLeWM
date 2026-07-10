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

> **Checkpoint validity notice (2026-07-10):** Every checkpoint documented on
> this page, including the stable v0.1 package and the v0.2.1 run-tree model,
> uses the `legacy_raw_v1` state contract. Although the configs declared
> `encoder.normalize: true`, the released implementation trained and evaluated
> on raw pooled Carbon states. Training sources used global pooling while
> targets were edit-centered, but every historical centered pool was shifted one
> hidden token left because the leading `<dna>` control token was omitted from
> the coordinate conversion; rollout/candidate centers differed and cache v1
> omitted that identity. The pinned Carbon `tokenizer.py` also performed an
> unpinned, network-capable `Qwen/Qwen3-4B-Base` lookup, so its hash did not
> establish a self-contained runtime identity. The v0.2.1 Phase 2 KL was computed from frozen
> target states; it changed the reported scalar loss but was constant with
> respect to the trainable predictor and action encoder, so it supplied no
> gradient. The metrics below are retained only as historical implementation
> outputs. These defects invalidate the checkpoints and metrics as evidence
> for the intended normalized method. Corrected results require fresh training
> and evaluation in a new `l2_normalized_v2` checkpoint lineage.
> The repaired source loader uses local pure-DNA tokenization and validated token
> layout, but no checkpoint has yet supplied model-quality evidence under those
> repairs.

GenoLeWM is an alpha research system for action-conditioned latent world models
over genomic edits. This model repository publishes the stable v0.1 checkpoint
package: trainable GenoLeWM predictor and action-encoder weights, calibration
data, manifest-bound hashes, training evidence, evaluation evidence, and release
checksums.

This is not a standard `transformers.AutoModel.from_pretrained()` checkpoint.
Load it through the `geno-lewm` runtime. Carbon-500M is a frozen state encoder
dependency and is resolved separately from this repository.

## Claim Boundary

Use this checkpoint for historical replay, local artifact inspection, and
method development. Do not use it for scientific scoring or as a diagnostic
model, clinical decision-support system, deployment-readiness claim, privacy
claim, or evidence of broad superiority over Carbon. The measured results below
are historical evaluations of the `legacy_raw_v1` implementation. They are not
valid evidence about the intended normalized GenoLeWM method.

No clinical utility claim is made.

## At A Glance

| Item | Value |
| --- | --- |
| Stable package in this repo | `geno-lewm-v0.1.0-r1` |
| Newer run-tree checkpoint used by the Space | `geno-lewm-v0.2.1-r1` under `abdelstark/geno-lewm-runs` |
| State contract for both published checkpoints | `legacy_raw_v1` |
| Runtime | Historical `geno-lewm` + Carbon path; not self-contained because its pinned tokenizer made an unpinned transitive Qwen lookup |
| Historical task | Single-SNV residual output over reference windows |
| Scientific status | Historical implementation outputs; normalized-method results not yet published |
| Strongest honest conclusion | Reproducible legacy systems artifact, not a valid evaluation of the intended normalized method |
| Not supported | Clinical utility, deployment readiness, privacy assurance, runtime assurance, broad model superiority |

## Which Checkpoint Should I Use?

| Target | Location | Use when |
| --- | --- | --- |
| Stable package | [`abdelstark/geno-lewm`](https://huggingface.co/abdelstark/geno-lewm) | You need checksum-bound replay of the historical v0.1 `legacy_raw_v1` artifact. Do not use it to evaluate the normalized method. |
| Space/default demo checkpoint | [`geno-lewm-v021-strong-4f36eef-10k-r1/suite/model`](https://huggingface.co/abdelstark/geno-lewm-runs/tree/main/geno-lewm-v021-strong-4f36eef-10k-r1/suite/model) | You need historical v0.2.1 artifact replay or UI compatibility. Its benchmark and planning outputs do not evaluate the normalized method. |
| Artifact console | [`abdelstark/geno-lewm` Space](https://huggingface.co/spaces/abdelstark/geno-lewm) | You want to inspect historical artifacts and checkpoint metadata. Legacy scientific scoring is disabled. |

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
| Historical generated paper | [`paper.serious-completion.md`](https://huggingface.co/abdelstark/geno-lewm-runs/resolve/main/geno-lewm-v021-strong-4f36eef-10k-r1/paper/paper.serious-completion.md) | Superseded manuscript containing the withdrawn negative-result interpretation; retained only for provenance. |

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
| Effective state contract | `legacy_raw_v1` |
| Dataset snapshot | `geno-lewm-data-v0.1.0-r1` |

The newer Space default checkpoint has model id
`sha256:cddb8f3b9671090201370b9824b9da741b933ff296b651238f022df5f3ed6af4`
and lives in the v0.2.1 run tree. It also uses `legacy_raw_v1`; it is historical
run evidence, not a corrected replacement for the stable v0.1 package.

## Method

GenoLeWM treats an edit as an action over a frozen genomic state embedding.
For a reference window, Carbon-500M encodes the pre-edit state. GenoLeWM's
trainable action encoder represents the edit, and the predictor estimates the
post-edit latent state. Under a valid same-contract lineage, candidate surprise
scores can be derived from the relationship between predicted and observed
post-edit states and calibrated with a matching artifact. The published legacy
residuals and calibration do not satisfy that contract.

The package intentionally separates:

- frozen state encoder: Carbon-500M, loaded separately;
- trainable GenoLeWM weights: `predictor.safetensors` and
  `action_encoder.safetensors`;
- release identity: `manifest.json` and `SHA256SUMS`;
- evidence: training, evaluation, efficiency, and run-tree reports.

The intended method applies L2 normalization to Carbon state vectors before
training, scoring, rollout evaluation, and planning. Published checkpoints did
not apply that transformation. Compatibility therefore preserves them as
`legacy_raw_v1`; corrected runs must declare `l2_normalized_v2` and use a new
checkpoint identity rather than reinterpret old weights or metrics.

The same boundary applies to tokenization and coordinates. The released custom
Carbon tokenizer could resolve `Qwen/Qwen3-4B-Base` over the network without a
pinned revision, and historical centered pooling ignored the leading `<dna>`
hidden token. Current source repairs both paths with a local pure-DNA tokenizer
and token-ID-derived centers. They are implementation invariants, not evidence
that a corrected model has learned a valid transition.

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
`5d31d59b3c845b288a13aedb1358934196852eec`. The Space no longer resolves this
encoder for scoring; its published-checkpoint surface is inspection-only.

## Scoring Contract

The runtime preserves this input-validation contract for compatibility, but
running a published `legacy_raw_v1` checkpoint reproduces invalid historical
residual output rather than a scientific score. The hosted Space therefore
disables scoring. The `REF` allele in
`chrom:pos:ref:alt` must match the supplied reference window at the one-based
locus implied by `pos` and `--window-start-bp`. If it does not match, scoring
raises `WindowMismatchError` before model inference.

Historical replay example with a synthetic reference-matched window:

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

Do not interpret the output as edit effect, surprise, pathogenicity, or model
quality. Corrected scientific scoring requires a newly trained
`l2_normalized_v2` checkpoint and matching calibration table.

## Historical v0.1 Training Summary

The stable v0.1 checkpoint was trained as a JEPA-style predictor over frozen
Carbon-500M raw pooled latent states under `legacy_raw_v1`.

| Field | Value |
| --- | --- |
| Run id | `first-snv-carbon-500m-r1` |
| Config | `training_config.effective.yaml` |
| Commit | `cd2bfccb33ec5a2df3c4707e8be8443f4682dad3` |
| Samples | 160,000 |
| Steps | 20,000 |
| Final training loss | 0.36124 |
| Status | completed |

## Historical v0.1 Evaluation

Held-out ClinVar GRCh38 chr21, binary P/LP versus B/LB labels. Scores use
`sigma_raw`; intervals are deterministic stratified bootstrap confidence
intervals from `eval_metrics.json`. These values measure the released legacy
implementation; they do not evaluate the intended normalized method.

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

The table is retained as historical `legacy_raw_v1` output. The v0.2.1 run tree
includes broader benchmark coverage and Carbon zero-shot comparisons, but the
results do not evaluate the intended normalized method. In addition, its
Phase 2 KL was constant with respect to every optimized parameter and supplied
no gradient, so the claimed active regularization was absent from optimization.

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

The v0.2.1 readiness report is `ok=true` for artifact coverage and provenance,
not for scientific validity of the normalized method. The autoregressive
rollout speed report is `ok=false`: K=5 measured 2.41x speedup against a 2x target, while
K=20 measured 2.47x against a 5x target and missed the target.

The v0.2.1 efficiency report measured one sample with no warmup on
`cuda:NVIDIA H200`: 115,262.94 ms single-variant latency, 0.3095 variants/s
throughput, and 1,966,149,632 bytes peak memory. Treat that as run evidence,
not a production serving benchmark.

## Planning Demo Evidence

The v0.2.1 run tree includes a deterministic synthetic multi-SNV planning demo.
It ran with manifest-backed runtime evaluation and recorded 384 evaluations,
`best_distance=23.656930390534644`, and `stopped_reason=patience`.

This is historical `legacy_raw_v1` execution evidence, not useful-planning or
normalized-method evidence. It shows only that the planner, legacy checkpoint,
window artifacts, and receipt path ran end to end.

## Troubleshooting

| Symptom | Likely cause | Fix |
| --- | --- | --- |
| `WindowMismatchError: window bases do not match edit.ref_bases at locus` | The variant `REF` allele does not match the supplied window at the requested coordinate. | Use a FASTA-backed VCF path or correct `--window`, `--window-start-bp`, and the variant `REF` allele so they describe the same reference sequence. |
| Space has no scoring controls | Legacy scientific scoring is intentionally disabled after the state-contract audit. | Use the Space for artifact inspection; wait for a fresh `l2_normalized_v2` checkpoint before scientific scoring. |
| `AutoModel.from_pretrained("abdelstark/geno-lewm")` fails | This repo is a GenoLeWM package, not a native Transformers architecture repo. | Use `snapshot_download` plus `geno-lewm-score` or the `geno_lewm.deploy` runtime. |

## Limitations

- Alpha research checkpoint; not a clinical, diagnostic, or deployment model.
- All published checkpoints use `legacy_raw_v1`; no corrected
  `l2_normalized_v2` result is published.
- The v0.2.1 Phase 2 KL supplied no gradient to trainable parameters.
- v0.1 evaluation is narrow: held-out chr21 ClinVar P/LP versus B/LB labels.
- v0.1 and v0.2.1 metrics are retained only as historical implementation
  outputs and must not be cited as evaluation of the intended normalized method.
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
