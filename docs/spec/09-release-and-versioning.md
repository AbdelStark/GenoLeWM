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
see [RFC-0011 §3](../rfcs/0011-artifact-provenance-receipts.md#3-manifest))
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
   emits GitHub/Sigstore build provenance. The source distribution must
   include the release-critical repo assets used by documented release
   gates: `bench/`, `configs/first_experiment/`, `tools/`, `docs/`,
   `rfcs/`, `examples/`, README, ROADMAP, and AGENTS.
9. Publish CHANGELOG to the release notes.
10. For paper/demo releases: validate the checked first-experiment
    dataset snapshot spec with
    `python -m tools.release.dataset_snapshot --spec-json configs/first_experiment/dataset-snapshot-snv.json --check-spec`,
    preflight staged local upstream inputs with the same spec and
    `--check-inputs`, then build the snapshot from explicit local upstream inputs with the
    same spec and `--dataset-dir ... --overwrite`.
    The command writes `dataset_input_check_report.json` and
    `dataset_snapshot_report.json` with the checked
    spec hash and upstream source file identities using public-safe
    source references, plus path/hash/size identities for generated
    input-check,
    metadata, manifest, data-card, split-integrity, and nested package-file
    artifacts. The report is
    included in `SHA256SUMS`, and the package verifier rejects missing,
    stale, duplicate-file, or private-path snapshot reports, plus invalid or duplicate
    checksum paths.
11. For lower-level dataset checks or custom snapshot assembly: generate
    the normalized dataset metadata, data card, manifest,
    split-integrity report, and checksums with
    `python -m tools.release.dataset_package --dataset-dir ... --metadata-json ...`.
    Generated dataset package metadata must carry
    `generated_by=tools.release.dataset_package`.
    The paper/demo package verifier must reject stale `data_card.md` or
    `dataset_manifest.json` output that no longer matches
    `dataset_package.json`, and it must reject `split_integrity.json`
    whose `generated_by` header is not
    `tools.release.dataset_integrity` or whose train/eval leakage
    evidence lacks comparable keys. `split_integrity.json` must record
    observed label/class balance, and `data_card.md` must render that
    generated split-level class-balance summary.
12. For model/demo releases: preflight the real training inputs with
    `geno-lewm-train --carbon-preflight --dataset-dir ... --carbon-model-dir ... --training-config ... --run-dir ...`
    before launching the Carbon-backed trainer. The first-experiment
    training config is checked in at
    `configs/first_experiment/train-carbon-500m-snv.yaml` and requests
    `runtime.device: cuda`; preflight
    validates the packaged dataset release evidence set, including
    `dataset_package.json`, `dataset_input_check_report.json`,
    `dataset_snapshot_report.json`, and `SHA256SUMS`; rejects stale
    input-check evidence; validates the effective config against the
    closed GenoLeWM config schema; verifies that a CUDA accelerator is
    available with at least 40 GiB of device memory by default; and records
    the config hash, resolved payload, accelerator probe, Carbon model
    file identities, dependency probes, and dataset evidence in
    `training_preflight_report.json`. After completion,
    package the run with
    `python -m tools.release.training_run --run-dir ... --metadata-json ...`
    or run `geno-lewm-train --carbon-train --package-release-run` so
    the training command packages the completed run immediately;
    generated `training_run_manifest.json` must carry
    `generated_by=tools.release.training_run`;
    release training metadata must name
    `training_preflight_report.json` so the run manifest and
    `training_run_SHA256SUMS` bind the preflight evidence.
    Release-mode verification must require that preflight report to
    include dataset core-file evidence for `dataset_package.json`,
    `dataset_input_check_report.json`, `dataset_snapshot_report.json`,
    and `SHA256SUMS`. Carbon resume launches with
    `--resume-from predictor_checkpoint.pt` must reject checkpoints
    whose run id, dataset snapshot, seed split, or config identity do
    not match the target run, and must record the resumed step in
    metrics, logs, and `training_run.json`. The final
    package verifier must reject stale `training_run_card.md` content
    that no longer matches `training_run_manifest.json`.
13. For model/demo releases: generate the matched Carbon zero-shot
    baseline score artifact with
    `geno-lewm-carbon-baseline --vcf ... --fasta ... --carbon-model-dir ... --output-scores ... --logp-cache-jsonl ...`,
    whose rows carry `generated_by=geno-lewm-carbon-baseline`.
    Optional sequence log-likelihood cache rows must be scoped to the
    Carbon model and revision before reuse, with unique sequence
    SHA-256 keys within that scope.
    Pass the generated score artifact to `geno-lewm-eval` with
    `--baseline-scores-jsonl ... --baseline-score-field carbon_zero_shot_score --baseline-name carbon_zero_shot`.
    Primary model score rows passed to `geno-lewm-eval` must carry
    `generated_by=geno-lewm-score`. The command must record checkpoint,
    config, dataset-manifest, eval-config, efficiency, score, label, and
    baseline-score artifacts as package-relative paths under
    `--artifact-root`, defaulting to the metrics output directory, write
    `eval_config.effective.yaml` beside the metrics JSON, and reject
    absolute paths outside that root.
14. For model/demo releases: stage the aggregate measured metrics as
    `<model-dir>/eval_metrics.json` and generate `eval_report.md` with
    `geno-lewm-eval-all --metrics-json ... --output-metrics ... --output-report ...`.
    For v0.2 benchmark-readiness candidates, pass
    `--require-v02-vep-metrics --require-v02-rollout-metrics` so
    aggregation fails before writing `eval_metrics.json` or
    `eval_report.md` when coding/non-coding ClinVar, BRCA2 saturation,
    TraitGym Mendelian VEP rows, or rollout-fidelity rows are missing
    required coverage. VEP rows require Carbon-baseline deltas,
    confidence intervals, and evaluated variant-key identities; rollout
    rows require phased-haplotype and synthetic edit-chain cosine, L2,
    and Recall@k metric coverage.
    The metrics payload must carry `generated_by=geno-lewm-eval` or
    `generated_by=geno-lewm-eval-all`; the report renderer must reject
    other generator labels and conclusions that do not explicitly
    reference every measured metric name, split, measured value, and
    baseline delta when present. `negative_findings` must be a
    non-empty list rendered as `## Negative Findings`. Baseline delta rows must supply
    `baseline`, `baseline_value`, and `delta_vs_baseline` together and carry
    matching `evaluated_variant_keys_sha256` and
    `baseline_evaluated_variant_keys_sha256` values.
    The command must also write `eval_config.effective.yaml` beside
    `eval_metrics.json` and record it, plus metrics input JSON files, as
    package-relative artifacts under the aggregate metrics directory so
    the report is tied to the committed eval config plus explicit CLI
    overrides without private absolute paths. The lower-level
    `python -m tools.release.eval_report --metrics-json ... --output ...`
    renderer remains available for already-aggregated metrics payloads.
    A v0.2 candidate may use
    `python -m tools.release.v02_benchmark_suite --manifest ... --output-report ...`
    to plan or execute the benchmark command graph for scoring,
    Carbon-baseline scoring, per-benchmark eval, rollout metrics,
    aggregate report generation, and readiness validation. Plan-only
    suite reports must keep `ok=false`; execute-mode suite reports
    clear each step's declared output files, then require each command
    to exit successfully and write those output files again. Passed
    execute-mode steps MUST record output identities by package-local path
    plus SHA-256 and size, but do not replace the generated metric,
    efficiency, rollout-speed, package, or readiness validators. The
    suite report records the manifest by package-local path plus SHA-256
    and size identity, not by build-machine absolute path. The final
    release-input readiness command that consumes `--suite-report` MUST
    run after the executed suite report exists; a first suite execution
    MUST NOT treat the suite report it is still writing as validated
    input evidence. A second-pass suite manifest MAY express this final
    readiness command with `readiness.suite_report`.
    Rollout benchmark entries may include a state-generation step that
    first runs `python -m tools.release.rollout_state_examples --spec-jsonl ... --cache-dir ...`
    when the manifest supplies cache-keyed example specs, then runs
    `python -m tools.release.rollout_state_rows --examples-jsonl ... --model-dir ...`
    before `geno-lewm-rollout`. The examples helper must resolve
    package-local cache keys for measured source/target/candidate latent
    states and record the cache index plus spec identities. The rows
    helper must consume those measured latent examples, load the
    manifest-backed action encoder and predictor, write
    `geno-lewm-rollout-states` JSONL, and record package-relative
    provenance. The rows helper MUST reject example rows that are missing
    the supported `schema_version=1.0.0` or the
    `tools.release.rollout_state_examples` generator marker. These
    helpers do not run Carbon encoding or construct held-out haplotypes,
    so those cache-backed specs remain required benchmark inputs. When
    rollout metrics are used for release-input readiness,
    `geno-lewm-rollout` must record package-relative
    `rollout_state_examples_report` and `rollout_state_rows_report`
    artifacts.
    Readiness input artifact identities plus readiness, efficiency,
    suite, and nested rollout-speed command path arguments must use
    public-safe paths plus SHA-256 and size where applicable so
    caller-supplied absolute CLI paths do not enter release reports.
    Direct `bench.rollout` speed reports consumed by readiness MUST carry
    a claim boundary that preserves the distinction between rollout-speed
    timing evidence and model-quality, clinical, privacy, or
    release-readiness evidence.
    Release-input readiness MUST consume an executed passing suite report
    generated by `tools.release.v02_benchmark_suite`; the suite report
    MUST carry passed-step output identities that match declared step
    outputs by package-local path, and those output identities MUST
    include every metrics JSON artifact consumed by readiness. The suite
    report MUST also preserve negative findings and a claim boundary that
    keep measured model-quality claims dependent on downstream artifact
    validators. The `release_inputs` row MUST record checked metrics
    artifact paths, efficiency input identities, and suite output
    identities from the validated payloads. The readiness report MUST
    carry measured efficiency latency, throughput, peak memory,
    runtime/sample context, limitations, sanitized efficiency command
    provenance, and row-derived `readiness` plus `blockers` entries with
    issue refs.
    Metric conclusions MUST include split/track context, measured values,
    baseline deltas, confidence intervals, and evaluated variant-key
    identities when those fields are available. Non-passing metric
    conclusions MUST include missing metrics, missing confidence
    intervals, baseline gaps, failed targets, or release-input findings
    when those fields are present.
    If the RFC-0004 AR rollout speed target is not met and the project
    explicitly accepts a re-scope in #42/#197, generate
    `python -m tools.release.rollout_speed_scope --rollout-speed-report ... --output ...`
    with UTC generated and accepted timestamps, HTTP(S) accepted decision
    URL, rationale, replacement target, and GitHub issue refs including
    #42 and #197. The generated scope report MUST record the failing
    `bench.rollout` input identity plus scope and nested rollout command
    path arguments with public-safe path text, MUST reject source
    `bench.rollout` reports missing or weakening the direct
    rollout-speed claim boundary, and MUST preserve negative findings
    plus a claim boundary stating that the failed target remains
    not passing rollout-speed evidence. The readiness report's
    `scope_decisions` summary MUST preserve the accepted scope report
    identity, accepter, rationale, replacement target, generated/accepted
    timestamps, decision URL, and issue refs. Re-scoped metric
    conclusions MUST include failed-target details plus the accepted
    decision URL, rationale, replacement target, and issue refs.
    `tools.release.v02_benchmark_readiness` may then consume
    `--rollout-speed-scope-report ...`, but only when that report binds
    the exact failing `bench.rollout` path/SHA-256/size identity and
    failed K targets, has valid GitHub issue refs including #42 and
    #197, when the scope command plus copied `bench.rollout` summary
    command are public-safe and current, and when the scope negative
    findings plus claim boundary preserve those limits.
    The readiness row is recorded as `rescoped`, not as passing
    rollout-speed evidence.
15. For model/demo releases: normalize measured efficiency evidence as
    `<model-dir>/efficiency_report.json` with
    `python -m bench.inference --release-efficiency --model-dir ... --vcf ... --fasta ... --variant ... --window ... --output-json ...`;
    the report must include single-variant latency, batched throughput,
    peak memory, benchmark command, hardware/runtime notes, input
    identities as package-relative paths or explicit inline labels,
    samples, warm-up, limitations, and
    `generated_by=tools.release.efficiency_report`.
16. For model/demo releases: generate the model card and checksums with
    `python -m tools.release.model_package --model-dir ... --metadata-json ...`;
    the model-package command requires `eval_metrics.json` and
    `efficiency_report.json`, writes normalized `model_package.json`,
    requires `generated_by=tools.release.model_package`,
    verifies `eval_report.md` is the exact render of the metrics JSON,
    checks eval/efficiency release id, dataset snapshot, commit, and
    model-result identity against the manifest-backed package, renders
    `model_card.md` from `model_package.json` plus `manifest.json`, and
    requires model metadata to include the training preflight report,
    training run manifest/card, and training-run checksums as
    `extra_files`. It includes all generated source artifacts,
    model-local eval artifact references from `eval_metrics.json`, and
    `training_preflight_report.json` in `SHA256SUMS`; package and Hub
    release verifiers reject invalid checksum digests and duplicate
    checksum paths, and the package verifier rejects training-run
    dataset snapshot, training config path/hash, or commit identity that
    does not match the manifest plus eval/efficiency evidence before
    publication planning. The model-package,
    dataset-package, and training-run package command success reports
    must serialize package-local output names rather than private
    workstation paths.
17. For terminal demo releases: generate the transcript and
    machine-readable demo manifest from the actual score command with
    `python -m tools.demo.terminal_inference --model-dir ... --vcf ... --fasta ... --output-dir ...`.
    The demo runner must clear owned score, receipt, batch-report, and
    demo-manifest outputs before invoking the score command so stale
    JSONL rows cannot satisfy a later run.
    This also emits runtime preflight and batch receipt reports for the
    same score/receipt artifacts, records the VCF input summary and
    JSONL field names in the transcript and manifest, and stores a
    compact score/receipt batch summary in `terminal_demo_manifest.json`.
    The transcript must include generated time, exit code, model
    release/version/id, artifact-input paths, and an explicit statement
    that model-quality claims require the published evaluation report.
    The transcript and manifest must render the score command, inputs,
    and output artifacts with package-local paths rather than private
    workstation roots. Before writing `terminal_demo_manifest.json`, the
    demo runner must re-open `runtime_preflight_report.json` and reject
    stale or mutated model, input, command, backend, runtime-requirement,
    or model-artifact evidence from a different run. The paper/demo package
    verifier must reject runtime preflight reports generated with
    fixture/test manifest allowance enabled, stale VCF input summaries,
    runtime-preflight command drift from the terminal-demo manifest command,
    stale terminal-demo manifest `runtime_preflight` summaries, stale
    transcript claim-boundary or artifact-input markers, stale
    manifest field lists, or `score_receipt_batch` summaries that no longer match the
    generated artifacts.
18. For paper/demo releases: generate the paper draft with
    `python -m tools.release.paper_draft --model-dir ... --dataset-dir ... --demo-dir ... --output ...`.
    Draft generation must reject stale `eval_report.md` output that no
    longer matches the packaged `eval_metrics.json`, and stale
    terminal-demo VCF summaries that no longer match the packaged demo VCF.
    It must use a UTC `Generated: ...Z` timestamp.
    It must render that scored-input summary in Demo Evidence.
    It must include Citation Metadata derived from the same release
    artifact identities.
    It must copy Negative Findings from the generated eval report.
    The generated Artifact Availability section must name
    `model_package.json`, `dataset_package.json`,
    `dataset_input_check_report.json`,
    `dataset_snapshot_report.json`, `eval_metrics.json`,
    `eval_config.effective.yaml`, `eval_report.md`,
    `efficiency_report.json`, and the terminal demo evidence, using
    package-local artifact names rather than build-machine root paths.
19. For model/demo releases: run
    `python -m tools.release.paper_package --model-dir ... --dataset-dir ... --demo-dir ... --paper-path ...`.
    The verifier re-renders the paper draft from the current model,
    dataset, and demo artifacts, re-renders `model_card.md` from
    `model_package.json` plus `manifest.json`, and rejects stale
    Markdown or drafts missing Citation Metadata or Negative Findings. It also cross-checks eval/efficiency release id, dataset
    snapshot, commit, and model-result identity, requires the training
    run's checksum-covered preflight report, resolves eval artifact
    paths inside the package, validates primary/baseline score JSONL
    `generated_by` markers, and rejects non-ok, stale,
    dataset/config-mismatched, or private-path preflight evidence.
20. Dry-run the Hub upload plan with
    `python -m tools.release.hub_release --model-dir ... --dataset-dir ... --demo-dir ... --repo-id ... --dataset-url ... --demo-url ... --paper-url ... --commit-sha ...`.
    If `--paper-path` is provided, `--paper-url` is required. The
    dry-run plan must record model upload files from both `SHA256SUMS`
    and `training_run_SHA256SUMS`, checksum-covered dataset upload
    files including dataset `SHA256SUMS`, and portable terminal-demo
    upload files with unique GitHub release asset names. When a paper
    URL is present, it must also record the verified public-safe paper
    source name/path, SHA-256, and size before emitting exact-file publication commands for
    recognized Hugging Face and GitHub URLs. Hugging Face commands must
    upload each verified model/dataset file to its planned destination
    rather than syncing whole package directories. The emitted
    `hub_release_plan.json`
    is a versioned generated artifact with `schema_version=1.0.0` and
    `generated_by=tools.release.hub_release`.
    The `.github/workflows/release-hub-dry-run.yml` workflow runs this
    dry-run planner plus the package verifier and release-candidate
    report without uploading weights or requiring Hub credentials.
21. Publish the verified public artifacts with
    `python -m tools.release.hub_publish --model-dir ... --dataset-dir ... --demo-dir ... --repo-id ... --dataset-url ... --demo-url ... --paper-url ... --commit-sha ... --plan-output ... --candidate-output ... --publish-output ...`
    or `.github/workflows/release-hub-publish.yml`. This path requires
    `HF_TOKEN`, GitHub release credentials, a supported Hugging Face
    dataset URL, a supported GitHub release-tag demo URL, and protected
    `release` environment approval in CI. It does not allow fixture
    manifests or skipped public checks. It uploads only the model,
    dataset, terminal-demo, and matching paper artifacts named by the
    verified Hub plan. Paper artifacts require a direct GitHub release
    download URL whose final asset name matches the verified paper file,
    because the final release-candidate report hashes the public paper
    URL bytes. The helper then regenerates the final release-candidate
    report from the public links and fetched public artifact bytes. The
    protected workflow must then run
    `python -m tools.release.clean_machine_demo` from that report with
    native runtime checks enabled before running
    `python -m tools.release.publication_assets` to bind the GitHub
    release target, upload command, and publication-evidence asset
    identities and then uploading those assets to the demo release tag.
    Workflow credentials may be used only for scoped Hugging Face and
    GitHub fetches and must not be serialized into release reports.
22. The final publication decision report from
    `python -m tools.release.release_candidate --model-dir ... --dataset-dir ... --demo-dir ... --paper-path ... --paper-url ... --repo-id ... --dataset-url ... --demo-url ... --commit-sha ... --output ...`;
    it must emit `ready=true`, bind `dataset_package.json`,
    `dataset_input_check_report.json`,
    `dataset_snapshot_report.json`, `model_package.json`,
    `eval_metrics.json`, `eval_report.md`, and `efficiency_report.json`,
    manifest-backed predictor/action/calibration/config artifacts, and
    `training_preflight_report.json` plus `training_run_SHA256SUMS`,
    serialize Hub model/dataset/demo upload inventories and a
    machine-readable `readiness` checklist with public-safe artifact paths,
    include `issue_refs` that map readiness rows and blockers to the live
    release issues (#163, #164, #165, #166, #167, and #101),
    and record successful public-link reachability checks for model,
    dataset, demo, and paper plus public artifact exact file-set, hash,
    and size checks for model, dataset, demo, and the paper bytes before public release
    notes link the artifacts. Those checks reject missing or unexpected
    public files, fetch public artifacts, and compare them with the
    SHA-256 and size values from the verified upload inventories.
    `--skip-public-link-check` is valid only together with
    `--allow-fixture-manifest` for offline fixture rehearsals; a
    non-fixture candidate remains `ready=false` when public checks are
    skipped.
23. Replay the terminal demo from public bytes with
    `python -m tools.release.clean_machine_demo --release-candidate-report ... --output-dir ...`.
    The replay command must consume a generated ready release-candidate
    report, reject hand-authored reports or stale embedded Hub plans by
    source header and model/repo/URL identity, reject candidates missing
    generated readiness rows, candidates with non-empty blockers, and
    candidates with skipped, missing, incomplete, or failed public
    link/artifact checks, then download published model files, dataset
    snapshot files, and GitHub release demo assets after rejecting
    unsafe embedded Hub-plan destinations or malformed expected hashes,
    verify their SHA-256 values against the Hub upload plan, re-run the
    release-package verifier on the downloaded model/dataset/demo
    package, rerun the terminal demo from the downloaded artifacts, and
    reject replayed `terminal_demo_manifest.json` files whose source
    header, status, model id, downloaded `model/manifest.json`
    hash/size identity, VCF/FASTA input identities,
    `runtime_preflight` summary, `score_receipt_batch` summary, or replay
    artifact hash/size identities do not match before writing
    `clean_machine_demo_report.json`. The report
    must include the
    release-candidate report filename plus hash/size identity,
    output-directory-relative downloaded artifact identities,
    package-verification result, replay transcript and manifest
    identities, and replay score, receipt, runtime-preflight, and
    batch-report artifact hashes without serializing private absolute
    workstation paths. Optional fetch credentials must not change the
    public URL/hash evidence model.
24. Bind final publication evidence with
    `python -m tools.release.publication_report --plan ... --release-candidate ... --publish-report ... --clean-machine-demo-report ... --output ...`.
    The report must bind the Hub release plan, release-candidate report,
    credentialed publish report, and clean-machine replay report by
    public-safe filename plus hash/size identity, including the
    clean-machine replay's recorded
    release-candidate report filename/path, hash, and size identity plus the
    verified paper file source name, URL, hash, and size identity plus
    the full paper-critical release-candidate artifact table for model,
    dataset, eval, demo, and paper identities plus public-safe
    release-candidate readiness rows plus public link and public artifact
    check summaries, every uploaded release-candidate artifact identity
    in that table checked against the Hub plan and downloaded public artifact,
    plus the replayed terminal-demo manifest's model id, downloaded
    `manifest.json` identity, VCF/FASTA input identities,
    `runtime_preflight` summary, and replayed runtime-preflight
    model/input identities without private absolute paths. It must
    require the candidate-embedded Hub plan to
    match `hub_release_plan.json`, require the generated readiness
    checklist with all expected rows marked `ok=true`, require empty
    candidate blockers and current readiness `issue_refs`, require
    generated `public_links` and `public_artifacts` sections with
    required checks present and passing, and fail if the final candidate,
    exact Hub-plan download set, public source URLs, hashes, or replay
    artifact set disagree.
    Its `issues` entries must carry `issue_refs` so final publication
    failures route back to #163, #164, #165, #166, #167, and #101.
    The protected workflow uploads `hub_release_plan.json`,
    `release_candidate_report.json`, `hub_publish_report.json`,
    `clean_machine_demo_report.json`, and
    `publication_evidence_report.json`,
    `publication_evidence_assets.json`, plus the clean-machine replay
    transcript, replay manifest, score/receipt JSONL streams, runtime
    preflight report, and batch receipt report as public GitHub release
    assets after this binder passes.
25. Post-release: open a tracking issue for v<X>.<Y>.<Z+1> placeholder
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

A receipt produced by a yanked model_id is still checkable against its manifest,
but the verifier emits a `model_yanked` warning when checking against a
revocation list.

Revocation-list mechanism is an [open question](#open-questions).

## Distribution

| Channel | Status | Use |
|---------|--------|-----|
| PyPI | Planned first tag | package wheels and sdists after trusted publishing is configured |
| HuggingFace Hub | v0.1 published | model checkpoints and dataset artifacts |
| GitHub releases | v0.1 published | terminal-demo, paper, and publication-evidence assets |
| Homebrew | Planned | post v1 |
| conda-forge | Future | post v1 |

## Invariants

| ID | Invariant | Enforced by |
|----|-----------|-------------|
| INV-REL-1 | A MINOR bump does not change any field of the public API in a breaking way | release-gate check vs API snapshot |
| INV-REL-2 | A PATCH does not change scoring outputs on a fixed input | release-gate `bit_match_baseline` |
| INV-REL-3 | Every release has a CHANGELOG section | release-gate check |
| INV-REL-4 | Every released model has a published `model_id`, normalized `model_package.json`, model card that re-renders from metadata and manifest, source `eval_metrics.json`, generated `eval_report.md` that exactly matches that metrics JSON, `efficiency_report.json`, dataset card, terminal demo transcript, runtime preflight report, batch receipt report, public artifact links that pass reachability checks and exact file-set/hash/size checks including the paper bytes, ready release-candidate report, and clean-machine replay report that materializes the public model, dataset, and demo artifacts | `tools.release.paper_package`; `tools.release.release_candidate`; `tools.release.clean_machine_demo` |
| INV-REL-5 | Deprecated symbols emit warnings for at least one MINOR before removal | per-version test |
| INV-REL-6 | The sdist includes release tooling, benchmark harnesses, first-experiment configs, the checked dataset snapshot spec, docs, RFCs, examples, README, ROADMAP, and AGENTS | `tools.release.check_sdist_assets`; `tests/lint/test_docs_release_blocker_contract.py` |

## Open questions

| ID | Question | Owner | Target |
|----|----------|-------|--------|
| OQ-REL-1 | Mechanism for revoking a yanked `model_id`; possibly a published, signed revocation list | core | post-first-demo release hardening |
| OQ-REL-2 | Whether to adopt CalVer for the model checkpoint id (e.g., `2026.05-r1`) instead of semver | core | v0.2 |
| OQ-REL-3 | Whether to publish a stable Python ABI commitment (likely no until 1.0) | core | 1.0 |
