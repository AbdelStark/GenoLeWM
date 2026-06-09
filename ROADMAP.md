# GenoLeWM Roadmap

GenoLeWM is usable as an alpha Python ML research package. The package,
model artifacts, dataset artifacts, benchmark bundle, planning demo, and
generated paper are public. The remaining roadmap is about improving the
science and performance evidence, not finishing release plumbing.

## Current Release State

| Surface | State |
| --- | --- |
| Python package | `geno-lewm==0.2.1` on PyPI |
| Source/wheel assets | <https://github.com/AbdelStark/GenoLeWM/releases/tag/v0.2.1> |
| Model package | <https://huggingface.co/abdelstark/geno-lewm> |
| Dataset package | <https://huggingface.co/datasets/abdelstark/geno-lewm-data> |
| Benchmark/planning/paper bundle | <https://huggingface.co/abdelstark/geno-lewm-runs/tree/main/geno-lewm-v021-strong-4f36eef-10k-r1> |

## Active Research Work

| Track | Current evidence | Next exit signal |
| --- | --- | --- |
| Rollout speed | K=5 speedup passes local target; K=20 remains below RFC-0004 `5x` target | measured K=20 pass from the real predictor path, with report and hardware identity |
| Planning | CEM API/CLI/demo exist; released demo stopped on patience with weak distance | useful planning behavior on released artifacts or a clear negative benchmark |
| Model quality | Current downstream evidence is mixed or negative versus Carbon | stronger held-out results with exact splits, baselines, confidence intervals, and negative findings |
| Data coverage | published dataset package and benchmark inputs are public | broader reproducible snapshots with refreshed split-integrity and leakage evidence |
| Training reproducibility | packaged Carbon-backed runs and fixture ML smoke exist | deterministic rerun evidence, collapse diagnostics, and regression thresholds for non-fixture training |
| Release infrastructure | package is public; trusted publishing needs account-side PyPI mapping | tag workflow publishes through OIDC without maintainer-token fallback |

## Pipeline Contracts

Dataset snapshot:

```bash
python -m tools.release.dataset_snapshot \
  --spec-json configs/first_experiment/dataset-snapshot-snv.json \
  --check-spec
```

Training preflight and run:

```bash
geno-lewm-train --carbon-preflight \
  --dataset-dir /path/to/dataset-package \
  --carbon-model-dir /path/to/carbon-model \
  --training-config configs/first_experiment/train-carbon-500m-snv.yaml \
  --run-dir /path/to/run

geno-lewm-train --carbon-train --package-release-run \
  --dataset-dir /path/to/dataset-package \
  --carbon-model-dir /path/to/carbon-model \
  --training-config configs/first_experiment/train-carbon-500m-snv.yaml \
  --run-dir /path/to/run
```

Evaluation:

```bash
geno-lewm-eval \
  --scores-jsonl scores.jsonl \
  --labels-jsonl labels.jsonl \
  --baseline-scores-jsonl carbon_scores.jsonl \
  --baseline-score-field carbon_zero_shot_score \
  --baseline-name carbon_zero_shot \
  --output-metrics eval_metrics.json

geno-lewm-eval-all \
  --metrics-json eval_metrics.json \
  --output-metrics aggregate/eval_metrics.json \
  --output-report aggregate/eval_report.md \
  --require-v02-vep-metrics \
  --require-v02-rollout-metrics
```

Benchmark suite:

```bash
python -m tools.release.v02_benchmark_suite \
  --manifest configs/first_experiment/v0.2_benchmark_suite.template.json \
  --output-report /path/to/v0.2_benchmark_suite_report.json \
  --execute
```

Terminal demo:

```bash
python tools/demo/terminal_inference.py \
  --model-dir /path/to/model \
  --vcf demo.vcf \
  --fasta reference.fa.gz \
  --output-dir /path/to/demo
```

## Claim Boundaries

- No clinical utility claims.
- No broad model-quality claim until measured evidence supports it.
- No privacy or runtime assurance claim beyond local execution contracts
  and checksum provenance.
- Fixture outputs are useful tests, not benchmark results.
- The current generated paper is negative-results/systems evidence.

## Maintainer Gates

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
