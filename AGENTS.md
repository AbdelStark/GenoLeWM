# Agent Context

Treat the live checkout, tests, docs, release artifacts, and GitHub
issues as authoritative. Do not rely on old plans or release notes when
choosing work.

## Current Project State

GenoLeWM is an alpha Python ML research package for action-conditioned
latent world models over genomic edits. The package is published as
`geno-lewm==0.2.1`.

Public artifacts:

- PyPI package: <https://pypi.org/project/geno-lewm/0.2.1/>
- GitHub source/wheel release:
  <https://github.com/AbdelStark/GenoLeWM/releases/tag/v0.2.1>
- Model package: <https://huggingface.co/abdelstark/geno-lewm>
- Dataset package:
  <https://huggingface.co/datasets/abdelstark/geno-lewm-data>
- v0.2.1 benchmark/planning/paper tree:
  <https://huggingface.co/abdelstark/geno-lewm-runs/tree/main/geno-lewm-v021-strong-4f36eef-10k-r1>
- v0.2.1 generated paper:
  <https://huggingface.co/abdelstark/geno-lewm-runs/resolve/main/geno-lewm-v021-strong-4f36eef-10k-r1/paper/paper.serious-completion.md>

## Implemented Surface

- Typed error taxonomy and stable error-code registry.
- Privacy-aware structured logging, redaction, metrics, and event
  registry.
- Canonical edit specs, relative edits, edit application, and synthetic
  edit samplers.
- Optional-runtime Carbon state encoder wrapper.
- Action encoder, cross-attention predictor, AR rollout wrapper, and CEM
  planning API/CLI.
- gnomAD and ClinVar VCF-to-Parquet shard preparation commands.
- RFC-0006 tuple-builder contracts for source mix, ClinVar fallback,
  absolute variant providers, and holdout filtering.
- `GenoLeWMDataset` deterministic tuple streaming.
- Fixture smoke trainer and preflight-gated Carbon trainer launcher.
- Manifest-backed scoring, checksum receipts, VCF score/receipt JSONL,
  runtime preflight, terminal-demo manifest, and batch receipt reports.
- Binary ClinVar metrics, Spearman metrics, Carbon baseline scoring,
  rollout-fidelity metrics, eval aggregation, efficiency reports, and
  benchmark-suite orchestration.
- Public API guards: current public module-map docs,
  `tests/api/public_surface.json`, duplicate-free `__all__`, and
  planning solver types not described as stable top-level exports.
- Dataset, model, training-run, demo, paper, release-candidate,
  publication, and source-distribution verification tooling.

## Current Evidence Boundary

The current benchmark evidence is mixed or negative:

- GenoLeWM does not broadly beat Carbon.
- K=20 AR rollout speed remains below the RFC-0004 target.
- The released planning demo exercises the manifest-backed model path but
  does not prove useful planning behavior.
- Fixture smoke outputs are CI evidence, not model results.

Do not add clinical, deployment-readiness, privacy-assurance, or runtime
assurance claims. The checksum provenance contract covers
artifact/output identity only; checksum receipts do not certify runtime
behavior.

## High-Value Work

Prefer bounded changes that improve one of these:

- K=20 rollout speed implementation and benchmark evidence.
- Planning quality or target-hardware planner performance.
- Broader held-out data snapshots and split-integrity evidence.
- Training reproducibility, collapse diagnostics, and benchmark gates.
- Public docs that preserve the measured limitations while making the
  package easy to install, train, evaluate, and demo.

## Validation

Use focused tests while editing. Before claiming a public-facing slice is
ready, run the strongest relevant subset:

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

For package/release edits also run:

```bash
rm -rf dist
uv run python -m build --outdir dist
uv run twine check --strict dist/*
uv run python -m tools.release.check_sdist_assets dist/*.tar.gz
```

Keep generated local junk out of commits unless it is an intentional
release artifact.
