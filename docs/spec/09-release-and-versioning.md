# 09 — Release and versioning

- Status: Authoritative for v0.1
- Companion RFC: [RFC-0017](../rfcs/0017-configuration-system.md),
  [RFC-0014](../rfcs/0014-public-api-and-stability.md)

Versioning is a contract with users. A score reproduced from
`geno-lewm-v0.1.0-carbon-500m-r1` six months from now must agree with
the original. Breaking this contract destroys the project's trust budget.
This document defines what versioning means here, how releases are cut,
and how deprecations land.

## Versioning surfaces

GenoLeWM has three versioned surfaces, each with its own semver number.

1. **The Python package** (`geno_lewm`).
2. **The model checkpoints** (e.g., `geno-lewm-v0.1.0-carbon-500m-r1`).
3. **On-disk schemas** (window cache, gnomAD/ClinVar shards, calibration
   table, receipt, manifest).

A single GitHub release line ties these together via the CHANGELOG.

## Package semver

The Python package follows PEP 440 / SemVer 2.0.

- **MAJOR (`X.0.0`)**: breaking change to the [public API](02-public-api.md)
  or to any field of the [data model](03-data-model.md) where the change
  affects existing artifacts.
- **MINOR (`0.X.0`)**: additive change — new public symbols, new optional
  fields on data classes / schemas, new CLI commands, new opt-in features.
- **PATCH (`0.0.X`)**: backwards-compatible fixes; no API change; no
  numerical change in outputs.

Pre-1.0 caveat (current): per PEP 440, MINOR bumps may introduce minor
breaking changes during the `0.x` series. We will *not* exercise this
license: until 1.0, MINOR bumps remain strictly additive. This is the
public commitment.

### What constitutes a breaking change

- Renaming, removing, or changing the signature of a stable public
  symbol (`02-public-api.md`).
- Changing the dtype, shape, or numerical contract of a documented input
  or output.
- Tightening validation (narrowing accepted enum values, stricter
  invariants).
- Changing default values that affect outputs numerically.
- Changing CLI exit codes.
- Reformatting structured logs in a way that breaks downstream parsers.
- Renaming an error `code`, event name, or metric name.

### Pre-releases

- Alpha: `0.1.0a1`, `0.1.0a2`, ... — internal use; weights not published.
- Beta: `0.1.0b1` — public preview; weights pinned but expected to be
  superseded.
- Release candidate: `0.1.0rc1` — only fixes after this; weights candidate.

## Model checkpoint versioning

Checkpoints are identified by a release id:

```
geno-lewm-v<MAJOR>.<MINOR>.<PATCH>-<encoder-id>-r<revision>
```

Examples: `geno-lewm-v0.1.0-carbon-500m-r1`, `geno-lewm-v0.2.0-carbon-3b-r1`.

Rules:

- The `<MAJOR>.<MINOR>.<PATCH>` tracks the package semver under which the
  checkpoint was trained.
- The `<encoder-id>` identifies the frozen encoder (`carbon-500m`,
  `carbon-3b`, etc.) and is part of the trust anchor.
- The `r<revision>` is bumped when re-trained with the same package
  version and the same encoder but different data, seed, or hyperparams.
- A new revision invalidates downstream caches keyed on `encoder_hash`.

The full identifier is paired with a `model_id` (SHA-256 of the manifest;
see [RFC-0011 §3.7](../rfcs/0011-verifiable-inference-attestation.md#37-manifest-schema))
that is the cryptographic source of truth.

## Schema versioning

Every on-disk artifact carries a `schema_version` (semver string).

| Schema | Current | Owner |
|--------|---------|-------|
| Manifest | `1.0.0` | RFC-0011 |
| Receipt | `1.0.0` | RFC-0011 |
| Window-embedding shard | `1.0.0` | RFC-0002, RFC-0006 |
| gnomAD shard | `1.0.0` | RFC-0006 |
| ClinVar shard | `1.0.0` | RFC-0006 |
| Calibration table | `1.0.0` | RFC-0009 |

Loader contract: accept any schema with the same MAJOR; ignore unknown
optional fields; reject unknown required fields with `SchemaCompatError`.

## Deprecation policy

1. A function, field, or behavior is marked deprecated by adding the
   `@deprecated("reason")` decorator (or a `Deprecated:` block in the
   field's docstring) and updating the CHANGELOG.
2. The deprecated symbol must continue to work for at least one MINOR
   release.
3. The deprecation emits a `DeprecationWarning` once per process and
   once per call site.
4. The symbol is removed in the next MAJOR.
5. The removal entry in CHANGELOG names the deprecating release.

CLI flag deprecations follow the same lifetime. New CLI flags must not
collide with existing short forms unless they are aliases.

## Backwards compatibility for trained checkpoints

The runtime supports loading **all minor versions back to the most recent
MAJOR boundary**.

- A `geno-lewm` v0.3.0 runtime loads checkpoints from v0.1.x and v0.2.x.
- A v1.0.0 runtime loads only v0.X if a migration utility shipped; otherwise
  refuses with `SchemaCompatError`.
- Migrations are scripts under `geno_lewm/migrations/` and are tested in
  `tests/integration/test_migrations.py`.

## Release cadence

- **PATCH:** as needed, typically within a week of a regression report.
- **MINOR:** every 6–12 weeks during the v0.x series.
- **MAJOR:** rare; only when accumulated breaking changes justify it,
  paired with a migration guide.

No automatic background updates; user-initiated via `geno-lewm-update`
(see [RFC-0010 §3.8](../rfcs/0010-on-device-personal-genome-deployment.md#38-update-mechanism)).
The update command consumes a JSON release index from the Hugging Face
Hub with `model_version`, `release_id`, `manifest_url`, and optional
`artifact_base_url` fields per release entry. Artifact bytes are still
trusted only after their hashes match the fetched manifest.

## Release process

1. Open `release/v<X>.<Y>.<Z>` branch from `main`.
2. Update `__version__` in `geno_lewm/__init__.py`.
3. Update `pyproject.toml` version.
4. Generate CHANGELOG section: `git log v<prev>..HEAD --no-merges`
   transformed by the changelog tool; manually curated into Added /
   Changed / Deprecated / Removed / Fixed / Security sections.
5. Run release-gate CI (full eval suite, performance benchmarks,
   reproducibility check, redaction property test, signed-artifact build).
6. Open a release PR. Require sign-off from a non-author maintainer.
7. Merge to `main`. Tag `v<X>.<Y>.<Z>`.
8. CI builds the PyPI artifacts from `uv.lock`, publishes them through
   `.github/workflows/release-pypi.yml` via trusted publishing, and
   emits GitHub/Sigstore artifact attestations.
9. Publish CHANGELOG to the release notes.
10. For model releases: upload `geno-lewm-v<X>.<Y>.<Z>-<encoder>-r1` to
    the HuggingFace Hub with full manifest and signed `eval_report.md`.
11. Post-release: open a tracking issue for v<X>.<Y>.<Z+1> placeholder
    fixes if any.

## CHANGELOG discipline

- Located at `CHANGELOG.md` at the repo root.
- Format: [Keep a Changelog](https://keepachangelog.com/en/1.1.0/) plus
  semver headers, no project-specific deviations.
- Sections: Added, Changed, Deprecated, Removed, Fixed, Security.
- A release without a corresponding CHANGELOG entry is not a release.
- Pre-release entries roll into the GA section at tag time.

## Long-term support

- Active line: the latest MAJOR. PATCHes flow here freely.
- Prior MAJOR: receives security PATCHes for 6 months after the next
  MAJOR's release.
- Older MAJORs: best-effort community maintenance.

## Yanking

A release with a serious bug may be yanked from PyPI (PEP 592) and have
its checkpoint flagged on the Hub. The CHANGELOG records the yank, the
reason, and the upgrade path.

A receipt produced by a yanked model_id is still verifiable as authentic,
but the verifier emits a `model_yanked` warning when checking against a
revocation list.

Revocation-list mechanism is an [open question](#open-questions).

## Distribution

| Channel | Status | Use |
|---------|--------|-----|
| PyPI | Stable | `pip install geno-lewm` |
| HuggingFace Hub | Stable | model checkpoints |
| GitHub releases | Stable | desktop app binaries |
| Homebrew | Planned | post v1 |
| conda-forge | Future | post v1 |

## Invariants

| ID | Invariant | Enforced by |
|----|-----------|-------------|
| INV-REL-1 | A MINOR bump does not change any field of the public API in a breaking way | release-gate check vs API snapshot |
| INV-REL-2 | A PATCH does not change scoring outputs on a fixed input | release-gate `bit_match_baseline` |
| INV-REL-3 | Every release has a CHANGELOG section | release-gate check |
| INV-REL-4 | Every released model has a published `model_id` and `eval_report.md` | release-gate check |
| INV-REL-5 | Deprecated symbols emit warnings for at least one MINOR before removal | per-version test |

## Open questions

| ID | Question | Owner | Target |
|----|----------|-------|--------|
| OQ-REL-1 | Mechanism for revoking a yanked `model_id`; possibly a published, signed revocation list | core | Phase 4 start |
| OQ-REL-2 | Whether to adopt CalVer for the model checkpoint id (e.g., `2026.05-r1`) instead of semver | core | v0.2 |
| OQ-REL-3 | Whether to publish a stable Python ABI commitment (likely no until 1.0) | core | 1.0 |
