# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog 1.1.0](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning 2.0.0](https://semver.org/spec/v2.0.0.html).

For pre-1.0 versioning policy (no breaking changes in MINOR until 1.0),
see [`docs/spec/09-release-and-versioning.md`](docs/spec/09-release-and-versioning.md).

## [Unreleased]

### Added

- Top-level [`SPEC.md`](SPEC.md) as the canonical entry point into the
  specification corpus.
- Eleven-section spec corpus at [`docs/spec/`](docs/spec/) covering
  overview, architecture, public API, data model, error model,
  observability, security, testing strategy, performance budget, release
  and versioning, and glossary.
- Cross-cutting RFCs 0012–0019 covering error taxonomy, observability,
  public API stability, testing strategy, performance budget,
  configuration system, CLI design, and the reference desktop app.
- Open-source process documents: [`SECURITY.md`](SECURITY.md),
  [`PRIVACY.md`](PRIVACY.md), [`CONTRIBUTING.md`](CONTRIBUTING.md),
  [`CODE_OF_CONDUCT.md`](CODE_OF_CONDUCT.md), and `CHANGELOG.md` (this
  file).
- Implementation tracker at
  [`docs/roadmap/IMPLEMENTATION.md`](docs/roadmap/IMPLEMENTATION.md).
- `docs/rfcs/` symlink to `rfcs/` so both canonical paths resolve.

### Changed

- [`SPECIFICATION.md`](SPECIFICATION.md) header now points readers to
  [`SPEC.md`](SPEC.md) and the per-section corpus as the authoritative
  entry points; the file remains as the synthesized canonical view.
- RFC index ([`rfcs/README.md`](rfcs/README.md)) updated with the eight
  new cross-cutting RFCs and a subsystem column.

### Security

- Network fail-closed contract documented in
  [`docs/spec/06-security.md`](docs/spec/06-security.md) and enforced by
  CI AST checks specified in [RFC-0015](rfcs/0015-testing-strategy.md).
- Redaction-by-default observability filter specified in
  [RFC-0013](rfcs/0013-observability.md); `GENO_LEWM_REDACTION_STRICT=1`
  is the documented default.

### Deprecated

_None._

### Removed

_None._

### Fixed

_None._

## [0.1.0-draft] — 2026-05-20

### Added

- Initial repository scaffold.
- 11 design RFCs (0001–0011) covering scope, encoder, action,
  predictor, training, data, eval, planning, surprise, deployment,
  attestation.
- `SPECIFICATION.md` synthesized canonical view.
- `ARCHITECTURE.md` narrative walk-through.
- `ROADMAP.md` phase plan.
- Glossary, FAQ, design-decision log under [`docs/`](docs/).
- Apache-2.0 license.
- `pyproject.toml` package stub.
