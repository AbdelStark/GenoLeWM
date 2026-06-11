# Changelog

All notable changes to GenoLeWM are documented here.

The project follows Semantic Versioning before 1.0: public API removals
or incompatible command changes require an explicit compatibility note.

## [Unreleased]

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
