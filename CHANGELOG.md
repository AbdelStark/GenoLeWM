# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog 1.1.0](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning 2.0.0](https://semver.org/spec/v2.0.0.html).

For pre-1.0 versioning policy (no breaking changes in MINOR until 1.0),
see [`docs/spec/09-release-and-versioning.md`](docs/spec/09-release-and-versioning.md).

## [Unreleased]

### Added

- **CLI dispatcher and stub command surface** (issues #30, #31;
  RFC-0018).
  - `geno_lewm/cli/_dispatch.py` — shared Typer dispatch helpers:
    `SharedOptions` dataclass, `shared_option_decls`, `finalize_shared`,
    `print_banner`, `run_app`, `not_yet_implemented`. Catches
    `GenoLeWMError` at exactly one place and maps each subclass to the
    exit code documented in `docs/spec/04-error-model.md` (2 / 3 / 4 /
    5 / 6 / 7 / 8 / 9; 130 for `KeyboardInterrupt`).
  - Eleven new Typer stub commands: `train`, `score`, `rollout`, `plan`,
    `eval`, `eval-all`, `export`, `cache-windows`, `prepare-gnomad`,
    `prepare-clinvar`, `update`. Each accepts the full shared flag set
    from RFC-0018 §3.2, prints the non-dismissible safety banner
    (RFC-0018 §3.7, suppressible only with both `--quiet` and
    `--no-banner`), and exits with code 9 advertising the GitHub
    tracking issue for the eventual implementation.
  - `[project.scripts]` registers all 12 console scripts (the 11 new
    stubs + the existing `geno-lewm-verify`).
  - Shell completion (issue #31) via Typer's built-in
    `--install-completion` / `--show-completion`; install steps
    documented in CONTRIBUTING.md.
  - `typer>=0.12` added to the base runtime dependency set; previously
    the package shipped with zero runtime dependencies.
  - `tests/unit/test_cli_dispatcher.py` — 84 tests covering the banner
    contract, the shared-flag validator, exit-code mapping for every
    error family, `--version` for every stub, the
    `pyproject.toml` ↔ module-layout invariant, and the `--help`
    smoke test for every console script.

- **Test pyramid scaffold and shared fixtures** (issue #85; RFC-0015
  §3.6).
  - `tests/conftest.py` resolves `PYTEST_RANDOM_SEED` (env override, or
    `HEAD` SHA mod 2**32 by default) and surfaces it in the pytest
    header for reproducible failures. Exposes the resolved value via
    the `random_seed` fixture; `seeded_random` returns a per-test
    `random.Random` so randomness never leaks across tests.
  - New synthetic fixtures usable by every test layer:
    `synthetic_window` (4 kB `ACGT`), `synthetic_edit_spec`,
    `synthetic_pooling_config`, `synthetic_dtype_config`,
    `synthetic_receipt_output`, `fixtures_dir`, `utc_now`,
    `stable_isoformat`.
  - New test directories created (per RFC-0015 §3.1):
    `tests/ml/`, `tests/integration/`, `tests/eval/`,
    `tests/typecheck/`, `tests/fixtures/`. Each ships an `__init__.py`
    that documents the layer's purpose. `tests/typecheck/` already
    holds runtime checks for the `py.typed` marker and the
    sortedness / completeness of `geno_lewm.__all__`.
  - `tests/fixtures/sample_window.fa` and
    `tests/fixtures/sample_receipt.json` — small canned data files;
    `tests/integration/test_fixtures_load.py` smokes them through the
    public `read_receipt` loader.

- **Performance harness, microbench suite, and regression detector**
  (issues #90, #91, #92; RFC-0016).
  - `bench/_harness.py` — stdlib-only timing library. `time_callable`
    returns a `BenchResult` with samples / median / IQR (P25, P75) and
    a metadata block (commit, machine, Python, platform, dtype).
    `write_result` persists JSON at
    `bench/results/<machine>/<benchmark>.json`. Machine slug honours
    `GENO_LEWM_BENCH_MACHINE` so CI runners write to distinct trees.
  - `bench/inference.py`, `bench/training.py`, `bench/planning.py`,
    `bench/profile.py` — per-target benchmark scripts and profiler
    invocations. Planning emits placeholder JSON until the CEM solver
    lands (#59 / #60 / #61).
  - `tests/benchmark/test_microbench.py` — `pytest-benchmark` suite
    over the hot paths (canonical-JSON hashing, sha256 file/bytes,
    receipt commitments, `EditSpec` validation, `apply_edit` /
    `apply_edits` batches). Marked `bench` and deselected from the
    default `pytest` run; the nightly job opts in with
    `pytest -m bench --benchmark-only --benchmark-json=...`.
  - `tools/ci/perf_regression.py` — diffs current results against the
    committed baseline at `bench/results/baseline/`. Handles both the
    bench-harness JSON shape and pytest-benchmark JSON; fails when any
    benchmark's median exceeds the baseline by more than the
    configured threshold (default 5 %, RFC-0016 §3.7). Treats missing
    baselines as warm-up and never gates on new benchmarks.
  - `.github/workflows/perf-nightly.yml` — daily cron that runs the
    harness, the pytest microbench suite, and the regression detector,
    uploading the result tree as a workflow artifact.
  - `pytest-benchmark>=4` added to the `[dev]` optional extras.

- **Changed-files coverage gate** (issue #88; `tools/ci/coverage_gate.py`).
  - Cobertura XML + `git diff origin/<base>...HEAD` → per-file coverage
    on the lines a PR adds or modifies; fails if any touched Python
    file under `geno_lewm/` falls below the configured threshold
    (default 90 %). Avoids the project-wide ratchet pathology called
    out in RFC-0015 §4.2.
  - Wired into `.github/workflows/ci.yml` as a step on the canonical
    matrix combo (Ubuntu × Python 3.12), gated to `pull_request`
    events. The `actions/checkout@v6` step now uses `fetch-depth: 0`
    so the gate can resolve the base ref locally.
  - Inputs are explicit (`--coverage-xml`, `--base`, `--threshold`,
    `--prefix`, `--diff-file`) so the gate is unit-testable without a
    real git repo.

- **Release tooling** (issue #102; `tools/release/`).
  - `tools/release/bump.py` rewrites the canonical `__version__`
    assignment in `geno_lewm/__init__.py` after validating the new
    string against the project's PEP 440 subset (release,
    `aN`/`bN`/`rcN`, `.postN`, `.devN`) and enforcing strict-monotone
    ordering. `--dry-run` emits the unified diff without touching the
    tree; `--show` prints the current version.
  - `tools/release/changelog.py` synthesises a Keep-a-Changelog 1.1.0
    section from `git log <since>..<until>`, mapping conventional /
    area-prefixed commits to `Added` / `Changed` / `Deprecated` /
    `Removed` / `Fixed` / `Security` buckets and flagging breaking
    (`feat!:` / `fix!:`) commits. Default `--dry-run` mode prints the
    section to stdout; `--write` lifts the existing `[Unreleased]`
    block in `CHANGELOG.md` into a dated `[X.Y.Z]` heading and
    re-opens an empty placeholder.
  - Both helpers are pure stdlib and run as `python -m
    tools.release.{bump,changelog}` so the release runner does not
    need optional dependencies installed.

- **Distribution & packaging.**
  - PEP 440-compliant version (`0.1.0.dev0`) sourced dynamically by
    Hatch from `geno_lewm/__init__.py` so package metadata and the
    runtime `__version__` cannot drift.
  - `py.typed` marker so downstream type checkers honour the
    package's mypy-strict signatures.
  - Curated top-level `geno_lewm.__all__` re-exporting the
    implemented surface (errors, observability, attestation, action
    specs, decorators).
  - `tools/__init__.py` so `python -m tools.*` runs as documented.
  - Optional dependency groups split into `train` / `eval` / `deploy` /
    `dev` / `docs` / `all`.

- **Modern quality tooling.**
  - Ruff lint+format with the full B, C4, UP, N, RUF, SIM, PIE, PTH,
    PL, PERF, FURB, LOG, ASYNC rule set; zero remaining findings.
  - Mypy `--strict` clean across `geno_lewm/` and `tools/` (25 source
    files, 0 errors).
  - `[tool.pytest.ini_options]` with strict markers / strict config /
    `filterwarnings = ["error"]`.
  - Branch coverage at a 95 % gate.
  - Pre-commit configuration mirroring every CI gate
    (`.pre-commit-config.yaml`).
  - `.editorconfig` and `.gitattributes` for cross-editor / cross-OS
    consistency.

- **CI/CD pipeline.**
  - `.github/workflows/ci.yml` — matrix tests on Python 3.10 / 3.11 /
    3.12 / 3.13 across Linux / macOS / Windows, ruff lint+format,
    mypy --strict, the five contract gates (errors / events / surface
    / no-print / network), `mkdocs --strict` build, sdist+wheel build
    with import-sanity smoke test, codecov upload, single
    required-check fan-in.
  - `.github/workflows/release.yml` — tag-driven PyPI publish via OIDC
    Trusted Publishing, TestPyPI dry-run on manual dispatch, GitHub
    release with extracted changelog notes.
  - `.github/workflows/codeql.yml` — weekly + per-PR static analysis
    (security-extended queries).
  - `.github/workflows/docs.yml` — GitHub Pages deploy.

- **Documentation site (mkdocs-material).**
  - `https://abdelstark.github.io/GenoLeWM/` with material theme,
    mkdocstrings, dark/light palette, search, and code annotations.
  - Auto-generated API reference, error-code table, log-event table.
  - RFC corpus rendered into the docs tree at build time with
    rewritten cross-links.
  - `docs/quickstart.md` walking through every shipped module.

- **Open-source hygiene.**
  - `.github/CODEOWNERS` mapping spec / RFC / privacy / security paths
    to project lead review.
  - `.github/dependabot.yml` for weekly minor/patch updates + security
    advisories on pip and GitHub Actions.
  - `.github/FUNDING.yml`.
  - README badges (CI, CodeQL, docs, PyPI, Python, license, mypy
    strict, ruff, pre-commit).

### Changed

- `tools/api/snapshot.py` now emits a Python-version-stable signature
  for enums (`enum[IntEnum](SNV=0, INS=1, …)` instead of the
  synthesized `__init__` signature that drifted between 3.10, 3.11,
  3.12, 3.13). The committed snapshot at
  `tests/api/public_surface.json` was regenerated.
- `geno_lewm.observability.logged_run` drops two unused locals and
  uses `contextlib.suppress` for the on-crash flush path.
- `geno_lewm.metrics.Histogram.snapshot` returns a typed
  `HistogramSnapshot` `TypedDict` rather than `dict[str, object]`.
- `geno_lewm.cli.verify.verify` accepts `stream: IO[str] | None`
  (was untyped `object | None`).

### Removed

- Legacy `.github/workflows/lint-errors.yml` (subsumed by the new
  multi-job `ci.yml`).
- `docs/rfcs` filesystem symlink (replaced by a docs-build-time
  generator that emits a docs-tree mirror with rewritten links).

### Security

- PyPI Trusted Publishing (OIDC) on the release workflow — no
  long-lived API tokens are stored in repository secrets.
- CodeQL Python analysis on every PR + weekly schedule.

## [0.1.0-draft] — 2026-05-20

### Added

- Initial repository scaffold.
- 19 design RFCs (0001–0019) covering scope, encoder, action,
  predictor, training, data, eval, planning, surprise, deployment,
  attestation, error taxonomy, observability, API stability, testing,
  performance budget, configuration, CLI, and the desktop app.
- `SPECIFICATION.md` synthesized canonical view.
- `SPEC.md` top-level index of the specification corpus.
- `ARCHITECTURE.md` narrative walk-through.
- `ROADMAP.md` phase plan.
- Eleven-section spec corpus at [`docs/spec/`](docs/spec/) covering
  overview, architecture, public API, data model, error model,
  observability, security, testing strategy, performance budget,
  release and versioning, and glossary.
- Open-source process documents: [`SECURITY.md`](SECURITY.md),
  [`PRIVACY.md`](PRIVACY.md), [`CONTRIBUTING.md`](CONTRIBUTING.md),
  [`CODE_OF_CONDUCT.md`](CODE_OF_CONDUCT.md).
- Implementation tracker at
  [`docs/roadmap/IMPLEMENTATION.md`](docs/roadmap/IMPLEMENTATION.md).
- Glossary, FAQ, design-decision log under [`docs/`](docs/).
- Apache-2.0 license.
- `pyproject.toml` package stub.
- Phase 1 infrastructure modules implemented and tested
  (`errors`, `observability`, `_redaction`, `metrics`, `action`,
  `attestation`, `cli.verify`, `api`).

### Security

- Network fail-closed contract documented in
  [`docs/spec/06-security.md`](docs/spec/06-security.md) and enforced
  by the `check_network_confined` AST linter.
- Redaction-by-default observability filter; the
  `GENO_LEWM_REDACTION_STRICT=1` strict mode is the documented
  default.
