# Contributing to GenoLeWM

GenoLeWM is an alpha Python ML research project. The infrastructure is
partly implemented, but the first real training run, released model, and
terminal inference demo are still open work.

The most valuable contributions now are narrow, tested changes that move
the repository toward a paper-ready first experiment.

## High-Value Work

- Carbon encoder integration that runs on a clean machine.
- Dataset builders with pinned upstream revisions and deterministic
  smoke fixtures.
- Trainer and evaluation paths that emit publishable artifacts.
- Model and dataset release automation.
- Terminal demos that run real inference rather than fixtures.
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

## RFCs and Specs

Write or amend an RFC when a change locks a load-bearing design decision
that affects multiple subsystems. Routine bug fixes and narrow
implementation work do not need new RFCs.

When implementation diverges from an old RFC, update the RFC or mark it
retired instead of letting stale design text survive.

## Review Expectations

Authors should:

- explain the problem, solution, validation, and caveats;
- keep generated files and snapshots intentional;
- respond to review with either a change or a concrete reason.

Reviewers should:

- check scope against the linked issue;
- focus on correctness, reproducibility, privacy, and API stability;
- cite files, tests, specs, or RFCs in feedback.

## Communication

- Bugs and feature work: GitHub issues.
- Security: GitHub Security Advisories; see [SECURITY.md](SECURITY.md).
- Design: RFC PRs or GitHub Discussions.

There is no required chat channel. The written record is the source of
truth.
