# RFC-0014: Public Python API surface and stability policy

- **Status:** Draft
- **Author(s):** GenoLeWM Project
- **Created:** 2026-05-20
- **Updated:** 2026-05-20
- **Depends on:** RFC-0001, RFC-0002, RFC-0003, RFC-0004, RFC-0008, RFC-0009, RFC-0010, RFC-0011
- **Supersedes:** —
- **Implementation status:** Not started

---

## 1. Summary

This RFC pins the public Python surface of GenoLeWM at v0.1 and the
policy that governs how that surface evolves. The surface is enumerated
in [`docs/spec/02-public-api.md`](../docs/spec/02-public-api.md); the
present RFC fixes the stability contract, the deprecation lifecycle, the
typing contract, and the enforcement mechanism.

## 2. Motivation

Without a pinned public surface, internal refactors leak through and
downstream code breaks unpredictably. With one, contributors know which
symbols cost a MAJOR to change and which are safe to refactor. The
distinction is load-bearing for v1 → v2 evolution and for downstream
research code that pins to GenoLeWM.

## 3. Specification

### 3.1 Stability classes

Three classes:

- **Stable.** Re-exported from `geno_lewm.__init__` or a subpackage's
  `__init__`. Breaking changes require MAJOR + deprecation period +
  CHANGELOG entry.
- **Experimental.** Documented but not in `__init__` re-exports.
  Marked with `@experimental`. May change in any MINOR.
- **Internal.** Underscore-prefixed or under any `geno_lewm/internal/`
  submodule. No stability guarantee.

### 3.2 The stable surface (v0.1)

Enumerated section-by-section in
[`docs/spec/02-public-api.md`](../docs/spec/02-public-api.md). Summary:

- Top-level: `__version__`, `GenoLeWMRuntime`, `EditSpec`, `EditType`,
  `SurpriseResult`, `PlanningResult`, `errors` submodule.
- `geno_lewm.encoder`: `CarbonStateEncoder`.
- `geno_lewm.action`: `EditSpec`, `RelEdit`, `EditType`, `ActionEncoder`,
  `apply_edit`, `apply_edits`.
- `geno_lewm.predictor`: `Predictor`, `ARPredictor`.
- `geno_lewm.surprise`: `SurpriseResult`, `score_variant`, `score_vcf`.
- `geno_lewm.planning`: `PlanningConfig`, `PlanningResult`, `plan`.
- `geno_lewm.deploy`: `GenoLeWMRuntime`.
- `geno_lewm.attestation`: `Receipt`, `write_receipt`, `read_receipt`,
  `verify_receipt`.

### 3.3 The experimental surface (v0.1)

- `planning.mcts.*`
- `deploy.tee_attestation.*`
- `attestation.stark.*`
- `encoder.lora.*`
- `surprise.bayesian.*`
- `surprise.directional.*`

Each lives behind `@experimental`, which emits a single `FutureWarning`
per process on first import.

### 3.4 Typing contract

- Every public symbol has explicit type annotations.
- Package ships `py.typed` per PEP 561.
- `mypy --strict` passes against the public surface in CI.
- `tests/typecheck/test_public_api.py` pins the contract with
  `reveal_type` assertions per stable symbol.

### 3.5 Breaking-change definition

See [`docs/spec/09-release-and-versioning.md`](../docs/spec/09-release-and-versioning.md#what-constitutes-a-breaking-change)
for the canonical list. Summary: signature changes, dtype/shape changes,
default-value changes that affect outputs, tightened validation, renamed
events / codes / metrics, and CLI exit-code changes.

### 3.6 Deprecation lifecycle

1. Add `@deprecated("reason")` decorator (or doc block) and a CHANGELOG
   entry under `Deprecated`.
2. The deprecated symbol works for ≥ 1 MINOR release.
3. Emits `DeprecationWarning` once per process and per call site.
4. Removed in the next MAJOR; CHANGELOG entry under `Removed`.
5. Migration guide section in the release notes.

### 3.7 API snapshot for CI

`tests/api/test_public_surface.py` snapshots the public API:

```python
PUBLIC_SURFACE = freeze(geno_lewm)   # symbol → signature
```

A snapshot file is committed at `tests/api/public_surface.json`. CI
diff:

- New entry → MINOR-or-MAJOR (auto-detected). MINOR is fine; MAJOR
  must be justified in the PR description.
- Removed entry → MAJOR required.
- Changed signature → MAJOR required.

Reviewers approve API changes by approving the snapshot diff.

### 3.8 ABI

GenoLeWM exposes no native ABI. There is no C extension surface to pin
in v1; PyTorch and HF Transformers are dependencies, not re-exported.

## 4. Rationale and alternatives

### 4.1 Why pin the surface at all in v0.x?

The Carbon ecosystem and the personal-genomics community will start
pinning to releases as soon as the first weights ship. We want the
surface to be a contract from v0.1, not from v1.0.

### 4.2 Why three classes rather than two?

A two-class stable/internal split forces every novel feature into either
"frozen forever" or "no docs". The experimental class lets us ship and
document features whose interface is still settling (planner variants,
attestation backends) without locking us in.

### 4.3 Why snapshot tests rather than only mypy?

mypy catches signature drift but not new public symbols sneaking in via
`from .submodule import *`. A snapshot of the resolved surface catches
both.

### 4.4 Why not use `__all__` only?

`__all__` is advisory; tooling (IDE completion, mypy) does not enforce
it. The snapshot file is enforced in CI.

## 5. Unresolved questions

- Whether to expose `Receipt` as a Pydantic v2 model for downstream
  JSON-Schema generation. Tracked as OQ-API-1.
- Whether `EditSpec.relative_to` should return a sum type rather than
  raise on out-of-window inputs. Tracked as OQ-API-2.
- Whether to publish a `geno_lewm.bench` namespace for downstream
  benchmark harnesses. Tracked as OQ-API-3.

## 6. Future work

- Auto-generated API reference into `docs/api/` from the resolved
  surface.
- Snapshot diff renderer that summarizes the change in CHANGELOG language
  for release engineers.
- A "compat tax" badge on `@experimental` symbols that ages and warns
  more loudly the longer the symbol stays experimental.

## 7. Changelog

- 2026-05-20 — Initial draft.
