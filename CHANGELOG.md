# Changelog

All notable changes to GenoLeWM are documented here.

The project follows Semantic Versioning before 1.0: public API removals
or incompatible command changes require an explicit compatibility note.

## [Unreleased]

### Scientific Validity Correction

- Added explicit encoder state contracts. Published v0.1 and v0.2.1
  checkpoints are classified as `legacy_raw_v1`; new corrected runs must use
  a distinct `l2_normalized_v2` checkpoint lineage.
- Corrected the encoder path so `normalize: true` applies L2 normalization
  consistently to live states and raw pooled cache hits.
- Replaced the ambiguous cache v1 key with cache schema `2.0.0`. Centered
  pooling now commits `center_token`; legacy Parquet shards are rejected and
  legacy SQLite indexes are invalidated because they can collide across loci.
- Corrected the Carbon token-coordinate mapping. Historical centered pooling
  used `edit_locus // 6` directly against hidden states even though the first
  hidden token is `<dna>`, shifting every intended center one hidden token left
  and sometimes centering a pool on the control token. The repaired path
  derives the DNA-content start from validated token IDs.
- Corrected training to pool source and target states at the same edit locus,
  persist the complete encoder representation in resume checkpoints, and
  reject cross-identity resumes.
- Corrected encoder provenance. `l2_normalized_v2` manifests and cache keys
  commit Carbon weights plus runtime-critical config and tokenizer files;
  manifest-backed loading verifies that local runtime before inference.
- Removed the unpinned transitive tokenizer load from the Carbon runtime path.
  The pinned upstream `tokenizer.py` delegated to a network-capable
  `Qwen/Qwen3-4B-Base` lookup without a revision, so hashing the local file did
  not make execution self-contained. The repaired loader implements Carbon's
  pure-DNA tokenization from local configuration and validates its control-token
  layout. This is a code-path correction; no corrected model-quality result is
  published.
- Changed new programmatic config defaults to `l2_normalized_v2`. Loading a
  schema-`1.0.0` config without an explicit state contract migrates it to
  `legacy_raw_v1`; incoherent normalized configs now fail during loading.
- Disclosed that the v0.2.1 Phase 2 KL was computed from frozen target states.
  It changed the reported scalar loss but was constant with respect to the
  trainable parameters and supplied no gradient.
- Reclassified every published model-quality metric as a historical output of
  the legacy implementation. Those values do not evaluate the intended
  normalized method; corrected results require fresh training and evaluation.

### Compatibility

- `WindowCacheKey` and `WindowCacheRecord` now require `center_token`, and
  cache schema `1.0.0` is intentionally not reusable.
- Rollout state specs/examples move to schema `1.2.0` and bind cache schema,
  raw-storage semantics, materialized state contract, encoder identity,
  pooling locus, and state width. Older ambiguous rows must be regenerated.

### Added

- Added fixture-backed scoring tutorial notebooks for a single
  ClinVar-like SNV and a one-row VCF, including checksum receipt
  validation and notebook execution tests. These examples are scoped as
  fixture smoke coverage, not model-quality evidence.
- Added a fixture-scale BRCA2 saturation mechanics notebook with tests,
  keeping published-data Spearman evidence explicitly open.
- Added release and CI reports for paper PDF/TeX builds, coverage-gate
  JSON output, tuple-throughput persistence, training reproducibility,
  dataset-integrity region checks, shard-prep artifact identities, and
  desktop signing preflight.
- Added a required CI paper job that builds the checked TeX paper PDF,
  writes `paper_tex_build_report.json`, and uploads both artifacts.
- Added desktop scaffold contracts for a persistent research-use safety
  banner and local VCF/FASTA file picker controls.

### Changed

- Removed obsolete public planning/specification scaffolding from the
  source tree and documentation site. Public docs now center on
  installation, architecture, API, artifacts, model-card evidence,
  privacy/security boundaries, and paper-related release material.
- Updated the source-distribution asset gate to require public release
  assets, paper sources, Space source, examples, configs, and release
  tooling rather than agent context, roadmap, or design-process files.
- Added pure-Python data/evaluation coverage for shard preparation and
  evaluation edge cases, raising global `fail_under` and the changed-file
  coverage gate from 83% to 84%.
- Recorded deterministic trace and artifact identities across planning,
  rollout, cache maintenance, data preparation, Carbon windows, fixture
  training, Carbon-backed training, real-training throughput, and model
  export paths.
- Clarified example-notebook blockers, FAQ release evidence, and planning
  package status without expanding the measured evidence boundary.
- Made unsupported export targets fail closed until a backed runtime
  implementation exists.

## [0.2.1] - 2026-06-09

### Added

- Published `geno-lewm==0.2.1` to PyPI.
- Published the GitHub source/wheel release for `v0.2.1`.
- Published the Hugging Face model package, dataset package, run tree,
  benchmark artifacts, planning-demo artifacts, and generated paper.
- Added public Hugging Face model-card documentation for choosing the
  stable checkpoint package versus the newer v0.2.1 run-tree checkpoint.
- Added the Hugging Face Space as an artifact console and compatible
  single-variant research demo.
- Added package validators for model, dataset, training-run, demo,
  paper, release-candidate, publication, and clean-machine replay
  artifacts.

### Changed

- Reframed public documentation around the measured result: useful
  systems evidence with mixed or negative model-quality findings.
- Narrowed checksum receipts to artifact/output identity. Receipts do
  not certify runtime behavior, privacy, clinical validity, or model
  quality.
- Moved public provenance helpers under `geno_lewm.provenance` and kept
  the public API snapshot guarded by `tests/api/public_surface.json`.
- Removed the healthcare-industry package classifier to keep package
  metadata aligned with the research-only boundary.

### Evidence Boundary

- GenoLeWM does not broadly beat Carbon.
- K=20 autoregressive rollout speed remains below the original target.
- The released planning demo exercises the manifest-backed model path
  but does not prove useful planning behavior.
- Fixture smoke outputs are CI evidence, not model results.

### Known Scientific Invalidation (documented 2026-07-10)

- The v0.1 and v0.2.1 configs declared `encoder.normalize: true`, but the
  released runtime ignored it. All published checkpoints therefore use raw
  pooled Carbon states and are now identified as `legacy_raw_v1`.
- The v0.2.1 Phase 2 KL was evaluated only on frozen target states. It was a
  constant, no-gradient addition to the reported loss rather than an active
  regularizer for the predictor or action encoder.
- The published metrics remain reproducible historical implementation outputs,
  but they do not evaluate the intended normalized method. Corrected evidence
  requires a newly trained and evaluated `l2_normalized_v2` lineage.

### Security

- PyPI release workflow configured for Trusted Publishing; `0.2.1` was
  published through a maintainer-token fallback after PyPI rejected the
  trusted-publisher claim.
- CodeQL and Scorecard checks run in hosted CI.

## [0.1.0-draft] - 2026-05-20

### Added

- Initial package scaffold for action-conditioned genomic edit world
  modeling.
- Core error taxonomy, redaction helpers, observability primitives,
  metric registry, edit specs, edit application, provenance receipts,
  verify CLI, and public API decorators.
- Initial open-source process files: license, security policy, privacy
  policy, contributing guide, and code of conduct.

### Security

- Added local-first privacy posture, redaction-by-default logging, and a
  network-boundary linter for runtime code.
