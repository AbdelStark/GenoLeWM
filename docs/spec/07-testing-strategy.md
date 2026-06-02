# 07 — Testing strategy

- Status: Authoritative for v0.1
- Companion RFC: [RFC-0015](../rfcs/0015-testing-strategy.md)

GenoLeWM uses a five-layer test pyramid that covers correctness,
property invariants, ML-specific failure modes, integration paths, and
end-to-end inference. CI runs every layer except the slow ML eval on
every PR; the ML eval and full integration runs are gated on release
candidates. No release ships without all gates green.

## Layers

### 1. Unit tests (`tests/unit/`)

- **Target:** every public function and class with isolated behavior.
- **Style:** pytest, hypothesis where applicable.
- **Coverage gate:** ≥ 90% line coverage on touched modules per PR.
- **Runtime budget:** entire suite ≤ 60 s on a laptop.

### 2. Property tests (`tests/property/`)

- **Target:** invariants from each spec section (the `INV-*` table).
- **Tool:** Hypothesis with seeded strategies (so failures are reproducible).
- **Examples:**
  - `apply_edit` round-trip: `apply_edit(window, e)` of length `len(window) - len(e.ref) + len(e.alt)`.
  - `EditSpec` validation rejects every non-ACGT base.
  - Cache writes followed by reads return the same vector bit-exact.
  - Canonical JSON of a manifest yields a stable SHA-256 across runs.
  - The redaction filter drops any DNA string ≥ 20 bp regardless of where
    it appears in `data`.
- **Runtime budget:** ≤ 120 s on a laptop.

### 3. ML tests (`tests/ml/`)

These are fast smoke tests of model-specific properties that the hosted
CI gate should catch before the full paper/eval path runs. They are
fixture-backed and must not require private model or data files.

Current hosted coverage:

- **Fixture training health**: the dependency-light `geno-lewm-train
  --fixture-smoke` path emits finite loss, `nan_loss_count=0`,
  collapse-health metrics, claim-boundary text, and fixture-only dataset
  identity.
- **Deterministic resume identity**: a resumed fixture run reproduces the
  uninterrupted checkpoint identity for the same seed and target step.
- **Collapse heuristics**: controlled healthy and degenerate synthetic
  batches produce the expected collapse-monitor alert behavior.
- **Optional torch predictor smoke**: when torch is installed, a tiny
  predictor preserves its identity-at-init contract and reduces loss on a
  fixed CPU minibatch; when torch is unavailable, the test skips
  explicitly.
- **Runtime budget:** each hosted `tests/ml` test should complete in
  ≤ 10 s on a laptop CPU, excluding optional dependency installation.

Future coverage should add deterministic receipt replay from a tiny
public scorer fixture once that artifact exists.

### 4. Integration tests (`tests/integration/`)

End-to-end paths across multiple modules, using small fixture data.

- **Train → eval smoke**: 50-step training on a 100-window fixture, then a
  100-variant ClinVar fixture eval. Pass if AUROC > 0.55 (much weaker than
  release; this catches plumbing breakage, not quality regressions).
- **Score VCF**: score a 50-variant fixture VCF; verify receipts are well-
  formed and `score_vcf` honors `batch_size`.
- **Export → import**: train a tiny predictor, export to ONNX / Core ML /
  GGUF, reload, verify numerical agreement to within tolerance.
- **Cache → reuse**: build a cache, run training with source `s_t` cache
  hits, verify the training is bit-exact equivalent to a no-cache run
  on supported backends, and confirm edited `s_{t+1}` targets are still
  encoded live.
- **Verifier**: produce a receipt, run the verifier without re-running
  inference, verify it accepts; tamper with a single byte of weights and
  verify it rejects with `ManifestHashMismatchError`.
- **Runtime budget:** ≤ 10 minutes on CPU; ≤ 3 minutes on GPU.

### 5. ML eval (`tests/eval/` and release eval gates)

The hosted eval smoke gate runs on generated public fixture artifacts.
The full real-data eval suite (RFC-0007) runs only on release candidates
and on documented release hardware.

- **Smoke eval (PRs):** `python -m tools.ci.eval_smoke_gate` generates
  score/label JSONL fixtures, runs `geno-lewm-eval` and
  `geno-lewm-eval-all`, writes `eval_smoke_summary.json`, and fails when
  AUROC, average precision, balanced accuracy, or AUROC delta versus the
  generated Carbon-baseline fixture crosses the configured threshold.
  The summary records `real_model_path.status=not_attempted` because the
  hosted gate does not use private data, released checkpoints, rollout
  artifacts, or paper benchmark inputs.
- **Full eval (release):** the full benchmark suite from
  [`08-performance-budget.md`](08-performance-budget.md). Run on a
  documented reference machine. Numbers persisted in
  `eval_report.md` for the release.

## Test categories by subsystem

| Subsystem | Unit | Property | ML | Integration | Eval |
|-----------|------|----------|----|-------------|------|
| `encoder/*` | ✓ | ✓ | — | ✓ | — |
| `action/*` | ✓ | ✓ | — | ✓ | — |
| `predictor/*` | ✓ | ✓ | ✓ | ✓ | — |
| `data/*` | ✓ | ✓ | — | ✓ | — |
| `eval/*` | ✓ | — | — | ✓ | ✓ |
| `planning/*` | ✓ | ✓ | ✓ | ✓ | — |
| `surprise/*` | ✓ | ✓ | ✓ | ✓ | — |
| `deploy/*` | ✓ | — | — | ✓ | — |
| `provenance/*` | ✓ | ✓ | — | ✓ | — |
| `cli/*` | ✓ | — | — | ✓ | — |
| `errors.py` | ✓ | ✓ | — | ✓ | — |
| `observability.py` | ✓ | ✓ | — | ✓ | — |

## CI gates

### Per-PR (mandatory)

1. **Lint**: `ruff check .` exits zero.
2. **Format**: `ruff format --check .` exits zero.
3. **Type check**: `mypy --strict geno_lewm/` exits zero.
4. **Custom AST checks**: no `print` in `geno_lewm/`, no `urllib`/`requests`
   imports outside `deploy/runtime.py` and `cli/update.py`, every raised
   exception is a `GenoLeWMError` subclass, every raised error has a
   registered `code`.
5. **Unit suite**: `pytest tests/unit -q` passes.
6. **Property suite**: `pytest tests/property -q --hypothesis-seed=<commit-hash>`
   passes.
7. **ML smoke**: `pytest tests/ml -q --tb=long --durations=10` passes
   in the dedicated `ml-smoke` CI job.
8. **Eval smoke**:
   `python -m tools.ci.eval_smoke_gate --work-dir .eval-smoke --summary-json .eval-smoke/eval_smoke_summary.json`
   passes in the dedicated `eval-smoke` CI job.
9. **Integration suite**: `pytest tests/integration -q -k 'not slow'` passes.
10. **Coverage gate**: changed-files coverage ≥ 90%.
11. **License headers**: every source file under `geno_lewm/` has the
    Apache-2.0 SPDX header.

### Per-release (mandatory)

1. All per-PR gates.
2. Full eval suite (RFC-0007).
3. Performance benchmarks against the targets in
   [`08-performance-budget.md`](08-performance-budget.md).
4. Reproducibility check: build twice from the lockfile; compare artifact
   hashes.
5. Receipt verifier: score a fixed variant set; re-run on a different host
   on supported backends; bit-match check.
6. Privacy audit: run the redaction property test against 10k random
   payloads; zero leaks.
7. Manual checklist signed off: clinical-banner present, SECURITY.md
   contact valid, CHANGELOG updated.

### Nightly (best effort)

- Larger smoke eval (5k variants).
- Memory regression check.
- Cross-platform smoke (macOS, Linux, Windows).

## Fixtures and corpora

Fixtures live in `tests/fixtures/`. None contain real personal data.

| Fixture | Purpose |
|---------|---------|
| `chr22_100kbp.fa` | tiny reference FASTA snippet |
| `variants_50.vcf` | 50-variant synthetic VCF |
| `clinvar_smoke.parquet` | 1,000-row ClinVar subset |
| `gnomad_smoke.parquet` | 1,000-row gnomAD subset |
| `corpus_smoke/` | 100 sequences for windowing tests |
| `tiny_checkpoint/` | a smallest-possible model checkpoint |
| `receipt_valid.json` and `receipt_tampered.json` | verifier tests |

Fixtures are committed only when small (< 1 MB each). Larger fixtures are
generated by `tests/conftest.py` from seeded synthetic data.

## Reproducibility

- Test runs honor a `PYTEST_RANDOM_SEED` env var; defaults to the commit
  hash modulo 2^32.
- ML smoke tests pin deterministic fixture seeds and, where torch is
  available, `torch.manual_seed`.
- Deterministic backends are required for verifier tests.

## Mutation and fuzz testing (post-v1)

- `mutmut` over the typed APIs, gated on a 70%+ kill rate.
- `python-afl` against the VCF parser and FASTA loader paths.

## Test data privacy

- Real user data is never committed.
- Synthetic fixtures use deterministic seeded random and document the
  seed in the fixture's docstring.
- Any new fixture under `tests/fixtures/` requires a reviewer to confirm
  it carries no personal data.

## Invariants

| ID | Invariant | Enforced by |
|----|-----------|-------------|
| INV-TEST-1 | Every public API symbol has at least one unit test | API-coverage script in CI |
| INV-TEST-2 | Every `INV-*` invariant in the corpus has a corresponding test in `tests/property/` | catalog test |
| INV-TEST-3 | Every error code in `ERROR_CODES` is raised by at least one test | catalog test |
| INV-TEST-4 | Every event name in `EVENTS` is emitted by at least one test | catalog test |
| INV-TEST-5 | CI gates run in the same order as documented | workflow lint |

## Open questions

| ID | Question | Owner | Target |
|----|----------|-------|--------|
| OQ-TEST-1 | Whether to add a `tests/benchmark/` track with `pytest-benchmark` and historical comparisons | core | v0.2 |
| OQ-TEST-2 | Whether the smoke eval should also include a non-coding subset by default | core | end of Phase 1 |
| OQ-TEST-3 | When to enable `python -X dev` in CI (catches more ResourceWarning issues; slightly slower) | core | once feature surface stabilizes |
