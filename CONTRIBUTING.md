# Contributing to GenoLeWM

GenoLeWM is an alpha Python ML research project. The v0.1/v0.2.1 model,
dataset, demo, paper, and run artifacts are public, but their earlier model
interpretation was withdrawn after the 2026-07-10 state-contract audit. The
checkpoints are `legacy_raw_v1` artifacts, not evidence for the intended
normalized method.

The most valuable contributions now are narrow, tested changes that
improve benchmark evidence, runtime reliability, reproducibility, and
public documentation.

## High-Value Work

- Broader held-out benchmark builders with pinned upstream revisions and
  deterministic smoke fixtures.
- Trainer and evaluation paths that enforce `l2_normalized_v2`, reject
  no-gradient regularizers, and emit auditable fresh-lineage artifacts.
- AR rollout speed work and benchmark gates for the K=20 target.
- Planning API/CLI work backed by corrected same-contract predictor and
  evaluation evidence.
- Release automation for future evidence packages.
- Documentation that keeps public claims aligned with measured behavior.

## Code of Conduct

This project follows the contributor [Code of Conduct](CODE_OF_CONDUCT.md).
By participating, you agree to abide by its terms.

## Before Opening a PR

1. Check the relevant GitHub issue and linked docs.
2. Keep the PR to one shippable unit.
3. Add or update tests for changed behavior.
4. Update docs and changelog when public behavior changes.
5. Run the strongest relevant validation you can run locally.

If no issue exists, open one first for non-trivial work.

## Local Setup

```bash
git clone https://github.com/AbdelStark/GenoLeWM.git
cd GenoLeWM

uv venv
source .venv/bin/activate
uv pip install -e ".[dev]"

pre-commit install
pytest
```

Install heavier extras only when needed:

```bash
uv pip install -e ".[train,eval,deploy,dev]"
```

## Validation

Use focused checks while developing, then broaden before review.

```bash
ruff format --check .
ruff check .
mypy geno_lewm tools
pytest
python tools/api/snapshot.py check
mkdocs build --strict
```

The public API snapshot is a contract. Public additions or removals must
update `tests/api/public_surface.json` and explain the compatibility
impact.

## Documentation Discipline

Docs must separate:

- implemented behavior;
- measured results;
- planned work;
- fixture-only examples.

Published `legacy_raw_v1` values may be quoted only as historical
implementation outputs. Do not interpret their L2 residuals, VEP/calibration,
or planning objectives scientifically. Do not describe their cosine values as
positive or negative model results: they are confounded by invalid training.
New scientific evidence must declare `l2_normalized_v2`, use a fresh
checkpoint/calibration lineage, and verify that every claimed regularizer has
a gradient path to an optimized parameter.

Do not add benchmark or model-quality claims unless the code and
artifacts needed to reproduce them are committed or linked from the
release.

## Data and Privacy

Personal-data reproducers are forbidden. Use synthetic FASTA/VCF files
or public benchmark data.

Data-related PRs must document:

- upstream dataset and revision;
- preprocessing steps;
- split rules and leakage checks;
- generated artifact hashes;
- licensing and use restrictions.

## Style

- Formatter: `ruff format`.
- Linter: `ruff check`.
- Type checker: `mypy --strict` configuration in `pyproject.toml`.
- Commits: imperative mood, short summary, reference the issue in the
  PR body.
- Source files: Apache-2.0 SPDX header.

## Design Notes

Keep design discussion close to the code, tests, and public docs it
changes. Large architecture changes should update `ARCHITECTURE.md`, the
affected doc page, and the relevant tests in the same PR.

## Review Expectations

Authors should:

- explain the problem, solution, validation, and caveats;
- keep generated files and snapshots intentional;
- respond to review with either a change or a concrete reason.

Reviewers should:

- check scope against the linked issue;
- focus on correctness, reproducibility, privacy, and API stability;
- cite files, tests, docs, or artifacts in feedback.

## Communication

- Bugs and feature work: GitHub issues.
- Security: GitHub Security Advisories; see [SECURITY.md](SECURITY.md).
- Design: GitHub issues, pull requests, or Discussions.

There is no required chat channel. The written record is the source of
truth.
