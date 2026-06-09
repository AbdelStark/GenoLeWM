# Implementation Tracker

This page records the current engineering surface. GitHub issues remain
the source of truth for task state.

## Implemented

- Core package: typed errors, public API snapshot, edit specs, edit
  application, provenance helpers, metrics, redaction-safe logs.
- Model code: optional Carbon encoder wrapper, action encoder,
  predictor, AR rollout wrapper, CEM planner.
- Data: VCF-to-Parquet builders, tuple-builder source mix, holdout
  filtering, `GenoLeWMDataset`.
- Training: fixture smoke path, Carbon preflight, Carbon trainer launch,
  packaged run manifests/cards/checksums.
- Evaluation: scorer, receipts, Carbon baseline scorer, binary metrics,
  Spearman metrics, rollout-fidelity metrics, aggregate reports.
- Release evidence: dataset/model/training/demo/paper/package
  validators, source-distribution inventory gate, clean-machine replay
  helpers.
- CI: lint, format, type check, API snapshot, docs, unit/integration
  tests, fixture-backed eval smoke, fixture-backed ML smoke, package
  build.
- Public API guard: current public module-map docs,
  `tests/api/public_surface.json`, duplicate-free `__all__`, and
  planning solver types not described as stable top-level exports.

## Public Evidence

| Evidence | Link |
| --- | --- |
| PyPI package | <https://pypi.org/project/geno-lewm/0.2.1/> |
| Source/wheel release | <https://github.com/AbdelStark/GenoLeWM/releases/tag/v0.2.1> |
| Model package | <https://huggingface.co/abdelstark/geno-lewm> |
| Dataset package | <https://huggingface.co/datasets/abdelstark/geno-lewm-data> |
| Benchmark/planning/paper tree | <https://huggingface.co/abdelstark/geno-lewm-runs/tree/main/geno-lewm-v021-strong-4f36eef-10k-r1> |
| Generated paper | <https://huggingface.co/abdelstark/geno-lewm-runs/resolve/main/geno-lewm-v021-strong-4f36eef-10k-r1/paper/paper.serious-completion.md> |

## Open Engineering Gaps

| Gap | Required evidence |
| --- | --- |
| K=20 rollout speed | target-passing benchmark report from the real predictor path |
| Planning quality | released-artifact planning benchmark with useful behavior, or a clear negative result |
| Broader model quality | held-out splits, Carbon baselines, confidence intervals, exact evaluated variant identities |
| Data scale | reproducible snapshots with split integrity, leakage checks, and data cards |
| Training reproducibility | repeated non-fixture runs with matching config/dataset/checkpoint identities |
| PyPI trusted publishing | successful tag workflow through OIDC without token fallback |

## Do Not Overclaim

- Current benchmark evidence is mixed or negative versus Carbon.
- The planning demo is execution evidence, not useful-planning proof.
- Fixture smoke runs are CI checks, not model results.
- checksum provenance covers artifact/output identity, not runtime
  certification.
- GenoLeWM is not for diagnosis or clinical decision support.

## Standard Local Gate

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
