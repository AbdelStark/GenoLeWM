# Contributing to GenoLeWM

GenoLeWM is in **Phase 0 — Design**. The reference implementation is
not yet written. The most valuable contributions today are:

- **RFC reviews.** Read the [RFCs](rfcs/) and open PRs with inline
  comments or proposed edits.
- **Spec reviews.** Read the [`docs/spec/`](docs/spec/) corpus and
  propose corrections, missing invariants, or under-specified contracts.
- **Open-question resolution.** Each RFC and each spec section lists
  open questions tagged `OQ-<area>-<n>`. Pick one, write a follow-up
  RFC or a PR.

When implementation begins (Phase 1), the
[implementation tracker](docs/roadmap/IMPLEMENTATION.md) is the primary
queue.

## Code of conduct

This project follows the contributor [Code of Conduct](CODE_OF_CONDUCT.md).
By participating, you agree to abide by its terms.

## How to propose changes

### Documentation / spec / RFC changes

- Open a PR titled with the affected area: `docs: …`, `spec: …`, `rfc: …`.
- For changes to an RFC, bump the `Updated:` date and document the diff
  in a trailing `## 7. Changelog` entry.
- For new RFCs, copy `rfcs/0000-template.md` and use the next free
  number.

### Code changes (Phase 1+)

- One PR = one shippable unit. If you need to land more than one logical
  change, file separate PRs.
- Every PR must reference a tracker issue. If none exists, file one
  first.
- Every PR must:
  - Pass all CI gates documented in [RFC-0015](rfcs/0015-testing-strategy.md).
  - Add tests for the new behavior (see the test pyramid in
    [`docs/spec/07-testing-strategy.md`](docs/spec/07-testing-strategy.md)).
  - Update CHANGELOG.md when the change is user-visible.
  - Update relevant RFCs / spec sections when the change locks a
    decision or fixes a documented invariant.

## Local setup

```bash
# clone
git clone https://github.com/AbdelStark/GenoLeWM.git
cd GenoLeWM

# Python env (pyproject + uv recommended)
uv venv
source .venv/bin/activate
uv pip install -e ".[dev,train,eval,deploy]"

# pre-commit
pre-commit install

# Tests (no model weights required for unit/property)
pytest tests/unit tests/property tests/ml
```

## Shell completion

Every Typer-based console script (everything in `[project.scripts]`
*except* `geno-lewm-verify`, which predates the dispatcher) ships
shell completion. Pick the command you reach for most often — the
completion script registers all 11 of them at once because they share
the same Click backend.

```bash
# bash (auto-detected from $SHELL)
geno-lewm-train --install-completion

# Or write the script to disk and source it manually:
geno-lewm-train --show-completion > ~/.geno-lewm.bash
echo 'source ~/.geno-lewm.bash' >> ~/.bashrc
```

zsh and fish are detected the same way; pass `--install-completion`
under the target shell to install. RFC-0018 §3.6 documents this as
the only completion mechanism in v1.

## Style and tooling

- Formatter: `ruff format` (configured in `pyproject.toml`).
- Linter: `ruff check` (configured in `pyproject.toml`).
- Type checker: `mypy --strict`.
- Custom AST checks live in `tools/lint/` and run in CI.
- Commits: imperative mood, ≤ 72 chars in the summary line; reference
  the tracker issue.
- License header (Apache-2.0 SPDX) at the top of every source file.

## RFC discipline

If your change locks a new design decision, write an RFC. The bar:

- The decision is load-bearing (multiple subsystems depend on it).
- A reasonable engineer might pick differently.
- Implementation diverges across files unless the decision is written
  down.

Lightweight bug fixes do not need RFCs; in doubt, ask in the PR.

## Issue triage

Triage labels:

- `triage:needs-info` — author has not provided enough information.
- `triage:reproduction` — reproducer welcome.
- `triage:reviewed` — accepted into the backlog.
- `triage:wontfix` — closed with reason.

Labels for type / area / priority / effort follow the schema in
[`docs/roadmap/IMPLEMENTATION.md`](docs/roadmap/IMPLEMENTATION.md).

## Tests, not screenshots

Bug reports that include reproduction steps and a synthetic reproducer
get triaged quickly. Bug reports with screenshots and no reproducer
become `triage:needs-info`.

Personal-data reproducers are explicitly forbidden. Synthetic VCFs and
FASTAs only.

## Governance

The project is informally maintained by a small core team. As the
project matures, we will adopt a more formal governance model (probably
similar to LeWorldModel's). Until then:

- Decisions on RFCs require at least one core reviewer approval.
- Decisions on implementation PRs require one approving review.
- A second reviewer is required for changes to the privacy or security
  posture, the public API, or the manifest / receipt formats.

The current core team is listed in [`docs/maintainers.md`](docs/maintainers.md).

## Licensing

GenoLeWM is Apache-2.0. By contributing, you agree your contributions
are licensed under Apache-2.0 and (where applicable) the project's
license addendum on clinical / reproductive use.

Contributions copied from other projects must be license-compatible;
include the upstream license header and a NOTICE entry if required.

## Communication

- Bug reports: GitHub issues with the appropriate template.
- Security issues: see [`SECURITY.md`](SECURITY.md).
- Design discussion: GitHub Discussions or RFC PR threads.

There is no Discord, Slack, or chat in v0.1. The asynchronous,
written-record channels are intentional.

## Code review

- Reviewer expectations:
  - Verify the PR matches the linked issue's scope.
  - Run the CI gates locally if anything looks off.
  - Be specific in feedback: cite spec sections, RFCs, or test files.
- Author expectations:
  - Respond to feedback within a week (or close the PR with a reason).
  - Squash trivial fix-up commits before merge.

We optimize for honest, direct review. No emojis in code-review comments;
no marketing tone. Disagreement is welcome; resolve in the spec or in an
RFC, not in PR ping-pong.
