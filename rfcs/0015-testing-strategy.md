# RFC-0015: Testing strategy and CI gates

- **Status:** Draft
- **Author(s):** GenoLeWM Project
- **Created:** 2026-05-20
- **Updated:** 2026-05-20
- **Depends on:** RFC-0007, RFC-0012, RFC-0013
- **Supersedes:** —
- **Implementation status:** Not started

---

## 1. Summary

This RFC specifies the five-layer test pyramid (unit, property, ML,
integration, eval), the per-PR and per-release CI gates, the fixture
discipline, and the reproducibility requirements. The eval suite itself
is specified in [RFC-0007](0007-evaluation-suite.md); this RFC defines
the engineering harness around it.

## 2. Motivation

ML projects have a recurring failure mode: lots of model-quality numbers,
zero engineering tests. GenoLeWM commits to both. The testing strategy
is the structure that makes "no release without all gates green" a
mechanical claim rather than a slogan.

## 3. Specification

The full contract is in [`docs/spec/07-testing-strategy.md`](../docs/spec/07-testing-strategy.md).
Load-bearing decisions:

### 3.1 Layers

1. **Unit** (`tests/unit/`): isolated behavior, ≥ 90% line coverage on
   touched modules per PR.
2. **Property** (`tests/property/`): every `INV-*` invariant in the
   spec corpus.
3. **ML** (`tests/ml/`): identity-at-init, loss-decreases-on-fixture,
   no-NaN/Inf, collapse heuristics, receipt determinism.
4. **Integration** (`tests/integration/`): train→eval smoke, score
   VCF, export round-trip, verifier accept/reject.
5. **Eval** (`tests/eval/`): smoke eval (1k variants) and full eval
   (RFC-0007).

### 3.2 Per-PR CI gates

In CI order:

1. `ruff check .`
2. `ruff format --check .`
3. `mypy --strict geno_lewm/`
4. Custom AST checks (no `print`, network imports confined, error
   discipline).
5. `pytest tests/unit`
6. `pytest tests/property --hypothesis-seed=<commit-hash>`
7. `pytest tests/ml`
8. `pytest tests/integration -k 'not slow'`
9. Coverage gate (changed files ≥ 90%).
10. Smoke eval (1k ClinVar coding + 500-window rollout) if PR touches
    relevant paths.
11. License headers present on every `geno_lewm/*.py`.

### 3.3 Per-release gates

All per-PR gates plus:

- Full eval suite (RFC-0007).
- Performance benchmarks
  ([`docs/spec/08-performance-budget.md`](../docs/spec/08-performance-budget.md)).
- Reproducibility: build twice, compare artifact hashes.
- Receipt re-verification across machines on supported backends.
- Privacy property test: 10 k random payloads, zero leaks.
- Manual checklist (clinical banner, SECURITY contacts, CHANGELOG).

### 3.4 Custom AST checks

Implemented in `tools/lint/`. Each is a tree walk that fails CI on
violation:

| Check | Rule |
|-------|------|
| `no_print` | `print(...)` not allowed in `geno_lewm/` |
| `network_confined` | `urllib`, `httpx`, `requests`, `aiohttp` imports only in `geno_lewm/deploy/runtime.py` and `geno_lewm/cli/update.py` |
| `raise_geno_lewm_error` | every `raise X(...)` has `X` a subclass of `GenoLeWMError` |
| `registered_error_code` | every `GenoLeWMError(code=...)` uses a literal in `ERROR_CODES` |
| `registered_event_name` | every `logger.info(event=...)` uses a literal in `EVENTS` |
| `registered_metric_name` | every `counter.inc("name")` / `histogram.observe("name")` uses a literal in `METRICS` |

The checks live alongside the code they enforce; they have unit tests
of their own.

### 3.5 Fixture discipline

Fixtures live under `tests/fixtures/`. None contain real user data.
Small fixtures (< 1 MB) are committed; larger ones are generated from
seeded random by `tests/conftest.py`. Each fixture's docstring documents
seed and provenance.

### 3.6 Reproducibility

- `PYTEST_RANDOM_SEED` env (default: commit hash mod 2^32).
- ML tests pin `torch.manual_seed` and `numpy.random.seed`.
- Deterministic backends required for verifier tests.
- Tests do not write outside `tmp_path` provided by pytest.

### 3.7 Coverage tool

`pytest-cov` with branch coverage enabled. The coverage gate is
**changed-files** coverage, not whole-project coverage, to avoid the
ratchet pathology of new code lowering global numbers.

### 3.8 Nightly / cron jobs

- Larger smoke eval (5 k variants).
- Memory regression check.
- Cross-platform smoke (macOS, Linux, Windows on CPU-only).
- Dependency audit (`pip-audit`).

Failures open a tracking issue automatically; they do not block PRs.

## 4. Rationale and alternatives

### 4.1 Why a separate ML test layer rather than mixing into unit / integration?

ML-specific failures (collapse, NaN drift) have a different shape from
either unit or integration failures. Keeping the layer separate lets us
budget runtime explicitly and gate on the right kinds of regression.

### 4.2 Why changed-files coverage instead of project-wide?

A project-wide coverage gate punishes new contributions that add
correctly-tested code into a file whose siblings happen to have low
coverage. Changed-files coverage is fair and equally tight.

### 4.3 Why hypothesis-seed from the commit hash?

Reproducibility. A property test failure on commit `abc1234` must be
reproducible on commit `abc1234` regardless of when the run happened.
Pinning the seed to the commit hash gives that for free.

### 4.4 Why AST checks rather than runtime asserts?

AST checks fail at PR review time. Runtime asserts fail in production.
Same correctness contract, drastically lower cost when violated.

### 4.5 Why is the smoke eval gated on path-touched paths only?

A docs-only PR should not run a 5-minute eval. A predictor-touching PR
should. Path-touched gating is the simplest right answer.

## 5. Unresolved questions

- Whether to add `tests/benchmark/` with `pytest-benchmark` historical
  comparisons. Tracked as OQ-TEST-1.
- Whether the smoke eval should include a non-coding ClinVar subset by
  default. Tracked as OQ-TEST-2.
- When to enable `python -X dev` in CI. Tracked as OQ-TEST-3.
- Whether to add mutation testing (`mutmut`) as a gate; currently
  reserved for post-v1.

## 6. Future work

- An ML-eval cache that memoizes evals by `model_id` so repeated CI runs
  don't re-eval unchanged checkpoints.
- A `geno-lewm-bench` CLI exposing the benchmark harness publicly
  (related to OQ-PERF-2).
- Cross-version checkpoint compatibility tests run nightly against the
  prior MAJOR.

## 7. Changelog

- 2026-05-20 — Initial draft.
