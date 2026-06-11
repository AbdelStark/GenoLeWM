# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog 1.1.0](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning 2.0.0](https://semver.org/spec/v2.0.0.html).

For pre-1.0 versioning policy (no breaking changes in MINOR until 1.0),
see [`docs/spec/09-release-and-versioning.md`](docs/spec/09-release-and-versioning.md).

## [Unreleased]

### Added

- Added a training reproducibility report gate that compares two
  deterministic training-run archives for matching run identity,
  bit-exact checkpoint/config/dataset artifact hashes, and
  deterministic-throughput drop against a baseline run.

## [0.2.1] - 2026-06-09

### Release Summary

- First Python package release for GenoLeWM's alpha public surface.
- Publishes the v0.2.1 serious-completion artifact chain: stronger
  post-v0.2 checkpoint lineage, broader benchmark-suite evidence,
  released-artifact planning demo, and generated paper package.
- Preserves the measured negative-results/systems framing: GenoLeWM does
  not broadly beat Carbon, K20 rollout speed remains below the RFC-0004
  target, and the planning demo does not prove useful planning behavior.

### Changed

- Narrowed RFC-0011 and the public roadmap to artifact provenance and
  checksum receipts. The receipt schema now accepts only
  `provenance.kind="checksum_only"`; unsupported future runtime
  assurance modes are rejected instead of treated as forward-compatible
  placeholders.
- The scope-language guard now checks notebook source and text outputs
  so public examples cannot reintroduce unsupported trust claims.
- Made `geno_lewm.provenance` the active public namespace for manifest,
  checksum, commitment, and receipt helpers. The old legacy import
  package has been removed from the active public API.
- Pruned CLI scaffold factory helpers from command-module public
  surfaces. Stub command modules now expose `app` and `cli_main`; the
  reusable `build_stub_app` / `make_cli_main` helpers remain internal to
  `geno_lewm.cli._stub_main`.
- Removed the healthcare-industry package classifier so future package
  metadata matches the documented research-only, non-clinical safety
  boundary.
- Rewrote the README, roadmap, implementation tracker, contributor
  guide, and `AGENTS.md` agent context around the current alpha status,
  first experiment gaps, dataset/model release plan, and terminal demo
  target.

### Added

- **Terminal demo transcript generator.**
  - `tools/demo/terminal_inference.py` runs `geno-lewm-score` with
    explicit score and receipt outputs, records stdout/stderr and
    manifest identity into a Markdown transcript, and rejects
    fixture/test manifests by default so release demos do not publish
    scaffold output as model behavior.
  - Demo transcript generation now fails unless `scores.jsonl` and
    `receipts.jsonl` exist, are non-empty JSONL, and have their hashes,
    row counts, and JSONL field names recorded in the transcript.
  - `terminal_demo_manifest.json` records JSONL field names and a compact
    `score_receipt_batch` summary with record count, checked score
    fields, receipt stream, model id, calibration hash, and runtime
    identity.
  - The paper/demo package verifier rejects stale terminal-demo JSONL
    field lists and stale `score_receipt_batch` summaries that no longer
    match the generated score, receipt, and batch-report artifacts.
  - `tools/release/batch_receipt_report.py` generates
    `batch_receipt_report.json` from the terminal-demo score and receipt
    JSONL streams, checking row counts, model id, calibration hash,
    runtime identity, row ordering, output commitments, and score-output
    equality.
  - `tools/release/paper_draft.py` generates the first-experiment paper
    draft from model, dataset, eval-report, terminal transcript, and
    batch receipt report artifacts; it records dataset package metadata,
    model package metadata, source metrics, eval report, efficiency
    report, and demo evidence in Artifact Availability, and rejects
    placeholder wording by default.

- **Paper/demo release package verifier.**
  - `configs/first_experiment/train-carbon-500m-snv.yaml` and
    `configs/first_experiment/eval-clinvar-snv.yaml` provide checked
    first paper-run config files that load through the closed GenoLeWM
    config schema.
  - Carbon training preflight now validates `--training-config` through
    the same closed schema and records the resolved config payload in
    `training_preflight_report.json`.
  - `tools/release/paper_package.py` validates the model directory,
    dataset snapshot, generated terminal transcript, and optional
    artifact-grounded paper draft before a first paper/demo release is
    linked publicly.
  - `tools/release/dataset_package.py` builds `data_card.md`,
    normalized `dataset_package.json`, `dataset_manifest.json`,
    `split_integrity.json`, and `SHA256SUMS` from release metadata and
    already-produced dataset shard files.
  - `tools/release/dataset_snapshot.py` now writes
    `dataset_snapshot_report.json` with the checked snapshot-spec hash,
    upstream source file hashes, staged file identities, and
    public-safe source references, and includes the report in
    `SHA256SUMS`.
  - The paper/demo verifier rejects stale dataset cards or manifests
    that no longer match the packaged `dataset_package.json` metadata.
  - The paper/demo verifier now requires `dataset_snapshot_report.json`
    and rejects missing reports, private absolute source paths, and
    stale staged-file identities.
  - The paper/demo verifier re-renders generated paper drafts from the
    current release artifacts and rejects stale paper Markdown, including
    drafts that no longer name `model_package.json`.
  - `tools/release/dataset_integrity.py` recomputes dataset file
    identities, split record counts, and train/eval comparable-key
    leakage checks from `dataset_manifest.json`.
  - Split-integrity checks now inspect Parquet row counts and variant
    keys when shard files expose `chrom`, `pos`, `ref`, and `alt`.
  - `tools/release/training_run.py` builds `training_run_manifest.json`,
    `training_run_card.md`, and `training_run_SHA256SUMS` from run
    metadata, metrics, logs, config, dataset manifest, checkpoint files,
    seeds, hardware/runtime notes, monitoring flags, and optional
    `training_preflight_report.json` evidence.
  - Release training-run verification now requires
    `training_preflight_report.json`, checks its schema, `ok=true`
    status, dataset snapshot, resolved config identity, checksums, and
    public-relative path references before a paper/demo package can pass.
  - `geno-lewm-train --carbon-train` now preflights the exact
    `training_config.effective.yaml` used by the launch, mirrors
    `training_preflight_report.json` into the run directory, and supports
    `--package-release-run` to write `training_run_manifest.json`,
    `training_run_card.md`, and `training_run_SHA256SUMS` immediately
    after a successful Carbon-backed run.
  - `tools/release/model_package.py` builds normalized
    `model_package.json`, `model_card.md`, and `SHA256SUMS` from
    `manifest.json`, manifest-backed checkpoint artifacts,
    model-release metadata, packaged `eval_metrics.json`, and
    `efficiency_report.json`.
  - Model and paper package verification now reject eval/efficiency
    evidence whose model release id, dataset snapshot, commit, or
    model-result identity is mixed across otherwise valid artifacts.
  - `tools/release/hub_release.py` dry-runs the Hugging Face Hub upload
    plan after package verification and records model files,
    checksum-covered training-run artifacts, model and dataset
    `SHA256SUMS`, terminal-demo files with portable package-local
    paths, hashes, release links, commit SHA, and upload commands for
    the model, dataset, and GitHub release artifact set when the URLs
    identify supported targets.
  - Hub dry-runs now require a public `paper_url` whenever a paper
    artifact is part of the candidate, and
    `.github/workflows/release-hub-dry-run.yml` runs the package,
    Hub-plan, and release-candidate gates without publishing weights or
    requiring Hub credentials.
  - `tools/release/hub_publish.py` and
    `.github/workflows/release-hub-publish.yml` add the credentialed
    publication path for verified model, dataset, and terminal-demo
    artifacts. The workflow requires `HF_TOKEN`, GitHub release
    credentials, protected `release` environment approval, supported
    Hugging Face/GitHub target URLs, and regenerates the final
    `release_candidate_report.json` after upload from public links and
    fetched public artifact bytes, then runs the clean-machine terminal
    replay from that report before uploading publication evidence.
    Replay fetches pass the Hugging Face token only to Hub artifact
    downloads and the GitHub token only to the release asset listing.
  - `tools/release/clean_machine_demo.py` downloads the published model
    files, dataset snapshot files, and GitHub release demo assets from a
    ready release-candidate report, verifies their SHA-256 values
    against the Hub upload plan, re-runs the release-package verifier on
    the downloaded model/dataset/demo package, reruns the terminal demo
    from those public bytes, and writes `clean_machine_demo_report.json`
    with the release-candidate report identity, downloaded artifact
    identities, package-verification result, replay transcript and
    manifest identities, and replay score, receipt, runtime-preflight,
    and batch-report artifact hashes, without serializing fetch tokens.
  - `tools/release/publication_report.py` writes
    `publication_evidence_report.json` after credentialed Hub publication
    and clean-machine replay, binding the Hub release plan,
    release-candidate report, publish report, and replay report by file
    identity and failing on candidate, readiness, download, or
    replay-artifact mismatches.
  - The verifier checks manifest artifact hashes, `SHA256SUMS`, model
    card/data card sections, stale model-card metadata, generated
    eval-report sections, dataset manifest file hashes, dataset
    split-integrity evidence,
    training-run evidence, fixture-manifest rejection, transcript
    markers from the real score command, and demo score/receipt JSONL
    hashes, row counts, and batch receipt report.
  - Demo batch receipt reports are rejected unless their model id and
    calibration hash match the packaged model manifest.
  - `tools/release/release_candidate.py` now records manifest-backed
    predictor, action-encoder, calibration, training-config,
    `model_package.json`, `dataset_snapshot_report.json`, and
    `training_preflight_report.json` plus `training_run_SHA256SUMS`
    artifact identities, Hub upload inventories, provider-backed public
    model/dataset/demo artifact hash checks, and a machine-readable
    `readiness` checklist in the final publication decision report.
  - `--skip-public-link-check` is now accepted only for explicit
    fixture rehearsals that also pass `--allow-fixture-manifest`; a
    non-fixture release candidate remains `ready=false` when public
    link or artifact hash checks are skipped.
  - `tools/release/eval_report.py` renders `eval_report.md` from
    measured metrics JSON and rejects empty metrics or placeholder
    wording before the package verifier accepts the artifact.
  - `geno-lewm-eval --scores-jsonl ... --labels-jsonl ... --output-metrics ...`
    now computes artifact-level ClinVar P/LP versus B/LB metrics and
    writes the measured metrics JSON consumed by the eval-report
    generator.
  - Eval metrics include deterministic stratified bootstrap confidence
    intervals by default (`--bootstrap-resamples`, `--bootstrap-seed`,
    `--ci-level`) and record an explicit omission note when resampling
    is disabled.
  - `geno-lewm-eval` can attach a matched measured baseline score
    artifact with `--baseline-scores-jsonl ... --baseline-name ...`;
    report metrics then include the baseline value and delta while
    preserving the baseline score artifact path.
  - `geno-lewm-carbon-baseline --vcf ... --fasta ... --carbon-model-dir ... --output-scores ...`
    now writes Carbon zero-shot baseline score JSONL with
    `carbon_zero_shot_score = -(logLik_alt - logLik_ref)` and optional
    sequence log-likelihood cache rows for reuse across real eval runs.
  - `geno-lewm-eval-all --metrics-json ... --output-metrics ... --output-report ...`
    now validates and aggregates measured metrics JSON artifacts into
    aggregate metrics JSON plus `eval_report.md`, writes
    `eval_config.effective.yaml`, records it as an `eval_config`
    artifact, and rejects mixed model, dataset, commit, hardware, or
    core artifact identities.

- **RFC-0006 training tuple-builder contract.**
  - `geno_lewm.data.builder` adds `WindowContext`, `TrainingTuple`,
    `EditSourceCount`, `HoldoutInterval`, and `HoldoutPolicy` so the
    future trainer receives a checked `(window_id, action, target_window)`
    stream instead of ad hoc tuples.
  - `build_training_tuples()` enforces the RFC-0006 3/3/1/1 source
    allocation, supports explicit ClinVar-to-synthetic-SNV fallback,
    filters chromosome/interval/edit-key/record holdouts, and validates
    provider output before target windows are materialized.
  - Synthetic SNV/indel providers and an absolute `EditSpec` provider
    are available for fixture tests and for wiring prepared gnomAD and
    ClinVar shards later.

- **Local gnomAD and ClinVar shard preparation.**
  - `geno-lewm-prepare-gnomad --input-vcf ... --output ...` converts a
    local gnomAD VCF/VCF.gz into the documented Parquet shard, keeping
    PASS variants above the global AF threshold and splitting
    multi-allelic rows per alternate.
  - `geno-lewm-prepare-clinvar --input-vcf ... --release ... --output ...`
    converts a local ClinVar VCF/VCF.gz into the documented Parquet
    shard, retaining VUS/OTHER rows while `label_set()` excludes them
    from labelled eval sets.

- **Deterministic fixture training smoke path.**
  - `geno-lewm-train --fixture-smoke --run-dir ... --steps 50` now
    writes a resolved config, metrics JSON, log, fixture checkpoint,
    fixture dataset manifest, and `training_run.json` metadata.
  - Fixture smoke runs can resume from the fixture checkpoint and are
    covered by bit-equal continuation tests. The output is release
    plumbing evidence only, not a Carbon-backed model result.

- **PyPI release workflow hardening** (issue #100).
  - `.github/workflows/release-pypi.yml` is the intended
    trusted-publisher workflow path for tagged releases.
  - Release artifacts build from the committed `uv.lock`, run the
    package and source-distribution gates, emit GitHub/Sigstore build
    provenance, and attach `SHA256SUMS` to the GitHub release.
  - The `0.2.1` upload exposed an account-side PyPI trusted-publisher
    configuration gap (`invalid-publisher`), so the validated
    distributions were published with the maintainer token available to
    the release runner and the fallback was recorded in #201.

- **Receipt-verification tutorial notebook** (issue #99).
  - `examples/07_verify_receipt.ipynb` verifies a committed
    checksum-only fixture receipt against its manifest, recomputes the
    input commitment, and validates the output commitment through the
    public `geno-lewm-verify` path.

- **CLI discovery flags wired to the config layer** (issue #29;
  RFC-0017 §3.8 + RFC-0018 §3.2).
  - `--print-config` renders the resolved config of the invoked
    command as canonical YAML on stdout.
  - `--print-config-tree` prepends a `# resolved from: <path>`
    provenance comment (single source today; Hydra-style multi-source
    composition lands with future work).
  - `--explain encoder.dtype` (or any dotted-key) renders the schema
    docstring + type + default — implemented via
    `geno_lewm.config.describe_field`.
  - Each Typer stub now passes `default_config_name` through to the
    dispatcher (`train`/`score`/`eval`/`plan`) so the right YAML
    template loads when the discovery flags fire.
  - `tests/unit/test_cli_dispatcher.py` extended with parametrised
    coverage of each flag against every stub (34 new cases) plus a
    targeted `--explain unknown.key` → exit code 3 test.

- **Configuration schema and YAML defaults** (issue #28; RFC-0017).
  - `geno_lewm/config/schema.py` — frozen dataclasses for every
    subsystem (`EncoderConfig`, `PredictorConfig`,
    `ActionEncoderConfig`, `OptimizerConfig`, `DataConfig`,
    `EvalConfig`, `ObservabilityConfig`, `RuntimeConfig`) plus the
    top-level `GenoLeWMConfig`. RFC-0017 §3.2 left the choice of
    Pydantic vs dataclasses open; dataclasses keep the base runtime
    dep footprint minimal (only `pyyaml` was added).
  - `geno_lewm/config/loader.py` — YAML loader + typed validator.
    Coerces YAML payloads through the schema, rejects unknown top-
    level keys with the new `UnknownTopLevelKeyError`
    (`CONFIG.UNKNOWN_TOP_LEVEL_KEY`), and rejects unknown sub-fields
    with `ConfigError`. Type coercion handles `Literal` enums,
    `tuple[str, ...]` (from YAML lists), `bool`/`int`/`float`/`str`,
    and `X | None` unions.
  - `geno_lewm/config/defaults/{train,score,eval,plan}.yaml` —
    canonical Phase 1 defaults mirroring every field in the schema.
  - `write_resolved_config()` emits the resolved config as canonical
    YAML so `${run_id}/config.resolved.yaml` is byte-stable
    (RFC-0017 §3.5).
  - `describe_field()` returns `{name, type, default, doc}` for any
    dotted-key in the schema; consumed by the `--explain` CLI flag
    (PR #29 wiring still to come).
  - `pyyaml>=6` and `types-PyYAML>=6` added to base / dev deps.

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
    implemented surface (errors, observability, provenance, action
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

- PyPI release workflow configured for Trusted Publishing (OIDC);
  `0.2.1` was published through a maintainer-token fallback after PyPI
  rejected the trusted-publisher claim.
- CodeQL Python analysis on every PR + weekly schedule.

## [0.1.0-draft] — 2026-05-20

### Added

- Initial repository scaffold.
- 19 design RFCs (0001–0019) covering scope, encoder, action,
  predictor, training, data, eval, planning, surprise, deployment,
  artifact provenance, error taxonomy, observability, API stability, testing,
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
  `provenance`, `cli.verify`, `api`).

### Security

- Network fail-closed contract documented in
  [`docs/spec/06-security.md`](docs/spec/06-security.md) and enforced
  by the `check_network_confined` AST linter.
- Redaction-by-default observability filter; the
  `GENO_LEWM_REDACTION_STRICT=1` strict mode is the documented
  default.
