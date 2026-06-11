# GenoLeWM

**Action-conditioned latent world models for genomic edits.**

[![CI](https://img.shields.io/github/actions/workflow/status/AbdelStark/GenoLeWM/ci.yml?branch=main&label=ci&style=for-the-badge)](https://github.com/AbdelStark/GenoLeWM/actions/workflows/ci.yml) [![CodeQL](https://img.shields.io/github/actions/workflow/status/AbdelStark/GenoLeWM/codeql.yml?branch=main&label=codeql&style=for-the-badge)](https://github.com/AbdelStark/GenoLeWM/actions/workflows/codeql.yml) [![Docs](https://img.shields.io/github/actions/workflow/status/AbdelStark/GenoLeWM/docs.yml?branch=main&label=docs&style=for-the-badge)](https://abdelstark.github.io/GenoLeWM/)
[![Hugging Face Space](https://img.shields.io/badge/hugging%20face-space-FFD21E?style=for-the-badge&logo=huggingface&logoColor=black)](https://huggingface.co/spaces/abdelstark/geno-lewm) [![Checkpoint](https://img.shields.io/badge/checkpoint-published-FF9D00?style=for-the-badge&logo=huggingface&logoColor=black)](https://huggingface.co/abdelstark/geno-lewm)
[![Python](https://img.shields.io/badge/python-3.10%2B-3776AB?style=for-the-badge&logo=python&logoColor=white)](pyproject.toml) [![License](https://img.shields.io/badge/license-Apache--2.0-2f6f9f?style=for-the-badge)](LICENSE) [![Typed](https://img.shields.io/badge/typed-mypy--strict-2f6f9f?style=for-the-badge)](https://mypy.readthedocs.io/) [![Ruff](https://img.shields.io/badge/lint-ruff-D7FF64?style=for-the-badge&logo=ruff&logoColor=black)](https://github.com/astral-sh/ruff)

GenoLeWM models a genomic edit as an action in a latent state space.
A frozen DNA encoder maps a reference sequence window to `s_t`; a
trainable action encoder and predictor estimate the post-edit state
`s_{t+1}`.

```text
reference window --Carbon encoder--> s_t
edit spec --------action encoder----> a_t
(s_t, a_t) -------predictor---------> s_hat_{t+1}
edited window ----Carbon encoder----> s_{t+1}
loss = distance(s_hat_{t+1}, s_{t+1}) + collapse regularization
```

The package is an alpha research system. It is not a diagnostic device,
not a clinical decision system, and not evidence of privacy or runtime
assurance beyond local execution contracts and checksum provenance.

## Install

```bash
python -m pip install geno-lewm
```

Use extras for heavyweight paths:
```bash
python -m pip install "geno-lewm[eval]"    # FASTA/VCF evaluation utilities
python -m pip install "geno-lewm[train]"   # torch/transformers training paths
python -m pip install "geno-lewm[dev]"     # tests, lint, typing, packaging
```

Source development:
```bash
git clone https://github.com/AbdelStark/GenoLeWM.git
cd GenoLeWM
uv pip install -e ".[dev,docs]"
```

## What Ships

| Surface | Status |
| --- | --- |
| Python package | `geno-lewm==0.2.1` on PyPI |
| Core data model | `EditSpec`, `RelEdit`, edit application, typed errors, metrics, redaction-safe logs |
| Model code | action encoder, cross-attention predictor, AR rollout wrapper, CEM planner |
| Data pipeline | local gnomAD/ClinVar VCF-to-Parquet builders, tuple builder, holdout filtering, dataset packaging |
| Training | fixture smoke trainer, Carbon preflight, single-process Carbon trainer, packaged run evidence |
| Evaluation | scorer, Carbon baseline scorer, binary metrics, Spearman metrics, rollout-fidelity metrics, eval aggregation |
| Demo | terminal scoring demo, checksum receipts, runtime preflight, batch receipt report, planning demo |
| Release tooling | dataset/model/training/paper package validators, sdist inventory gate, clean-machine replay helpers |

## Public Artifacts

| Artifact | Link |
| --- | --- |
| PyPI package | <https://pypi.org/project/geno-lewm/0.2.1/> |
| Source/wheel release | <https://github.com/AbdelStark/GenoLeWM/releases/tag/v0.2.1> |
| Model package | <https://huggingface.co/abdelstark/geno-lewm> |
| Dataset package | <https://huggingface.co/datasets/abdelstark/geno-lewm-data> |
| Planning demo artifacts | <https://huggingface.co/abdelstark/geno-lewm-runs/tree/main/geno-lewm-v021-strong-4f36eef-10k-r1/planning-demo> |
| v0.2.1 benchmark/planning/paper tree | <https://huggingface.co/abdelstark/geno-lewm-runs/tree/main/geno-lewm-v021-strong-4f36eef-10k-r1> |
| v0.2.1 benchmark readiness report | <https://huggingface.co/abdelstark/geno-lewm-runs/resolve/main/geno-lewm-v021-strong-4f36eef-10k-r1/suite/model/v0.2_benchmark_readiness_report.json> |
| v0.2.1 generated paper | <https://huggingface.co/abdelstark/geno-lewm-runs/resolve/main/geno-lewm-v021-strong-4f36eef-10k-r1/paper/paper.serious-completion.md> |

## Results

The current evidence is useful systems evidence with mixed or negative
model-quality results. Do not cite it as broad superiority over Carbon.

| Track | GenoLeWM result | Baseline comparison |
| --- | ---: | --- |
| ClinVar coding | AUROC `0.734375`, AP `0.8529761904761904`, balanced accuracy `0.75` | vs Carbon: AUROC `-0.1875`, AP `-0.09894688644688643`, balanced accuracy `+0.0625` |
| ClinVar non-coding | AUROC `0.5625`, AP `0.6054563492063492`, balanced accuracy `0.4375` | vs Carbon: AUROC `-0.3125`, AP `-0.30896672771672784`, balanced accuracy `-0.25` |
| BRCA2 saturation | Spearman rho `0.14919354838709678` | vs Carbon: `-0.32771260997067453` |
| TraitGym Mendelian | Spearman rho `-0.02796450759873114` | vs Carbon: `+0.05592901519746229` |
| Phased-haplotype rollout | cosine mean `0.28886058350550603`, L2 mean `33.319687258878126`, Recall@4 `1.0` | weak vs source-state baseline: cosine `-0.7089701215468133`, L2 `+31.19289051130368` |
| Synthetic edit-chain rollout | cosine mean `0.30160847029349436`, L2 mean `28.802888778495763`, Recall@4 `1.0` | weak vs source-state baseline: cosine `-0.6896310938123016`, L2 `+25.637059814259455` |
| AR rollout speed | K=5 speedup `2.413859489667916`; K=20 speedup `2.4732225135799566` | K=5 passes its local target; K=20 remains below the original `5x` target |
| Planning demo | `best_distance=23.656930390534644`, `384` evaluations, patience stop | released-artifact execution evidence, not useful-planning evidence |

The generated paper and readiness report preserve these negative
findings. The strongest current claim is that the repository can train,
package, evaluate, benchmark, and replay a genomic-edit world-model
pipeline with content-addressed evidence.

The GenoLeWM-FX pivot is documented as a killed feasibility experiment:
public TraitGym labels are available, but no reproducible 10k-50k
teacher-delta cache is locked under the experiment contract.
The only follow-up trajectory is a narrow precomputed-Borzoi overlap
audit; it is not a model-quality claim.

## Quickstart

```python
from geno_lewm import EditSpec, EditType, RelEdit, apply_edits

edit = EditSpec(chrom="chr17", pos=43_091_983, ref="A", alt="T")
assert edit.edit_type is EditType.SNV

rel = edit.relative_to(window_start_bp=43_091_900, window_end_bp=43_092_100)
print(rel.rel_pos)

window = "ACGT" * 64
edited = apply_edits(
    window,
    [
        RelEdit(rel_pos=0, edit_type=EditType.SNV, ref_bases="A", alt_bases="T"),
        RelEdit(rel_pos=4, edit_type=EditType.SNV, ref_bases="A", alt_bases="C"),
    ],
)
print(edited[:12])
```

Verify a checksum receipt:

```bash
geno-lewm-verify examples/data/verify_receipt/receipt.json \
  --manifest examples/data/verify_receipt/manifest.json
```

Run a local fixture trainer smoke test:

```bash
geno-lewm-train --fixture-smoke --run-dir /tmp/geno-lewm-smoke --steps 50
```

Score a local VCF with a local model package:

```bash
geno-lewm-score \
  --model-dir /path/to/model \
  --backend auto \
  --vcf variants.vcf \
  --fasta reference.fa.gz \
  --output scores.jsonl \
  --receipt receipts.jsonl \
  --batch-size 64 \
  --no-progress
```

Run manifest-backed planning:

```bash
geno-lewm-plan \
  --model-dir /path/to/model \
  --window-fasta window.fa \
  --target-fasta target.fa \
  --output plan.json \
  --horizon 5 \
  --iterations 5 \
  --samples 1024 \
  --elite 64
```

## Training Pipeline

The real training path is explicit about data identity, Carbon runtime
readiness, config identity, and packaged run artifacts.

1. Validate the checked dataset rebuild spec.

   ```bash
   python -m tools.release.dataset_snapshot \
     --spec-json configs/first_experiment/dataset-snapshot-snv.json \
     --check-spec
   ```

2. Check staged upstream inputs before building a dataset snapshot.

   ```bash
   python -m tools.release.dataset_snapshot \
     --spec-json configs/first_experiment/dataset-snapshot-snv.json \
     --check-inputs
   ```

3. Run Carbon preflight for the packaged dataset and training config.

   ```bash
   geno-lewm-train --carbon-preflight \
     --dataset-dir /path/to/dataset-package \
     --carbon-model-dir /path/to/carbon-model \
     --training-config configs/first_experiment/train-carbon-500m-snv.yaml \
     --run-dir /path/to/run \
     --preflight-output /path/to/run/training_preflight_report.json
   ```

4. Launch training and package the run evidence.

   ```bash
   geno-lewm-train --carbon-train --package-release-run \
     --dataset-dir /path/to/dataset-package \
     --carbon-model-dir /path/to/carbon-model \
     --training-config configs/first_experiment/train-carbon-500m-snv.yaml \
     --run-dir /path/to/run
   ```

The run package contains the checkpoint, metrics, logs, resolved config,
preflight report, manifest/card, and checksum inventory. Resume checks
validate run id, dataset snapshot, seed split, and config identity.

## Evaluation And Benchmarks

Compute measured binary or Spearman metrics:

```bash
geno-lewm-eval \
  --scores-jsonl scores.jsonl \
  --labels-jsonl labels.jsonl \
  --baseline-scores-jsonl carbon_scores.jsonl \
  --baseline-score-field carbon_zero_shot_score \
  --baseline-name carbon_zero_shot \
  --output-metrics eval_metrics.json \
  --split clinvar_coding
```

Aggregate metrics and render the release report:

```bash
geno-lewm-eval-all \
  --metrics-json eval_metrics.json \
  --output-metrics aggregate/eval_metrics.json \
  --output-report aggregate/eval_report.md \
  --require-v02-vep-metrics \
  --require-v02-rollout-metrics
```

Run the benchmark-suite command graph from a manifest:

```bash
python -m tools.release.v02_benchmark_suite \
  --manifest configs/first_experiment/v0.2_benchmark_suite.template.json \
  --output-report /path/to/v0.2_benchmark_suite_report.json \
  --execute
```

Measure release efficiency:

```bash
python -m bench.inference --release-efficiency \
  --model-dir /path/to/model \
  --variant chr17:43091983:A:T \
  --window /path/to/window.txt \
  --output /path/to/efficiency_report.json
```

Pure solver timing for planning:

```bash
python -m bench.planning --output /path/to/planning.performance.json
```

## Demo Pipeline

Generate a terminal demo transcript from released model/data artifacts:

```bash
python tools/demo/terminal_inference.py \
  --model-dir /path/to/model \
  --vcf demo.vcf \
  --fasta reference.fa.gz \
  --output-dir /path/to/demo \
  --backend auto \
  --batch-size 64
```

The demo writes `terminal-demo-transcript.md`,
`terminal_demo_manifest.json`, `runtime_preflight_report.json`,
`batch_receipt_report.json`, `scores.jsonl`, and `receipts.jsonl`.
These files are release evidence only when generated from non-fixture
artifacts and checked by the package validators.

## Repository Layout

```text
geno_lewm/      package code
bench/          inference, rollout, training, and planning benchmarks
configs/        checked dataset, training, eval, and benchmark configs
docs/           public documentation and generated API pages
examples/       small public examples and receipt fixtures
paper/          manuscript source and paper evidence dossier
tests/          unit, integration, eval, ML smoke, API, and lint gates
tools/          release, CI, dataset, demo, and documentation tooling
```

## Quality Gates

Run the full local gate before publishing code:

```bash
uv lock --check
uv run ruff check .
uv run ruff format --check .
uv run mypy geno_lewm tools
uv run python tools/api/snapshot.py check
uv run python -m tools.lint.check_scope_language
uv run pytest
uv run mkdocs build --strict
```

Package gate:

```bash
rm -rf dist
uv run python -m build --outdir dist
uv run twine check --strict dist/*
uv run python -m tools.release.check_sdist_assets dist/*.tar.gz
```

CI runs the same core contracts plus fixture-backed eval and ML smoke
jobs. Fixture outputs are test evidence, not model results.

## Limitations

- Current measured benchmarks are mixed or negative relative to Carbon.
- The K=20 AR rollout speed target remains open.
- The released planning demo exercises the model path but does not show
  useful planning behavior.
- No GenoLeWM-FX model or demo ships; the FX pivot is stopped at the
  feasibility gate.
- The precomputed-Borzoi rescue is complete as a no-positive-claim
  result: the residual lift is small and non-significant, no training or
  benchmark claim is open, and exact fipip overlap is not claimed.
- No clinical utility claim; the public model evidence is not clinical
  evidence.
- Personal-genome workflows are local-first, but local execution is not
  the same thing as a privacy assurance.
- Checksum receipts prove artifact and output identity; they do not
  certify runtime behavior.
