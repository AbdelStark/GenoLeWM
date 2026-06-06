# Agent Context

This file is the working context for coding agents operating in this
repository. Treat the current worktree, tests, docs, and GitHub issues
as authoritative; do not rely on stale plans or old project positioning.

## Current Direction

GenoLeWM is an alpha Python ML research project for action-conditioned
world models over genomic edits. The first paper/demo publication
(`geno-lewm-v0.1.0-r1`) is complete. It includes:

- a reproducible dataset snapshot and data card;
- a trained SNV predictor checkpoint with model card and manifest;
- a generated evaluation report with measured results and conclusions;
- an end-to-end terminal demo that runs real local model inference from
  released artifacts;
- a clean-machine replay and final `publication_evidence_report.json`
  with `ok=true`.

The near-term target is now v0.2 benchmark and rollout readiness:
stronger held-out evaluation, measured GenoLeWM-vs-Carbon deltas,
autoregressive rollout speed evidence, and planning-ready interfaces
without expanding claim boundaries.

Checksum manifests, input/output commitments, and receipts are in scope
as artifact provenance. Do not add claims about unsupported runtime
assurance modes. `python -m tools.lint.check_scope_language`
guards Markdown, Python, config/JSON, and notebook source/output text
for de-scoped trust-language regressions.

## Implemented Surface

As of June 6, 2026, the implemented and tested surface includes:

- typed error taxonomy and stable error-code registry;
- privacy-aware logging, redaction, metrics, and event registry;
- canonical edit specs, relative edits, edit application, and synthetic
  edit samplers;
- optional-runtime action encoder, base cross-attention predictor, and
  `ARPredictor` rollout wrapper;
- dependency-free RFC-0006 tuple-builder contracts for source mix,
  ClinVar fallback, absolute variant providers, and holdout filtering;
- local gnomAD and ClinVar VCF-to-Parquet shard preparation commands
  with schema-checked loaders;
- a lazy Carbon state encoder wrapper for optional local Transformers
  runtimes;
- predictor losses and pure-Python training stability helpers;
- deterministic `geno-lewm-train --fixture-smoke` run-artifact path for
  release plumbing tests;
- `geno_lewm.data.GenoLeWMDataset` for deterministic tuple streaming
  from checked windows and edit-source providers without importing torch
  in core environments;
- `geno_lewm.training.encode_training_batch` looks up untargeted source
  `s_t` states in the documented `GENO_LEWM_CACHE` index when present,
  falls back to live Carbon encoding on cache miss, and keeps edited
  `s_{t+1}` target states live-encoded;
- torch trainer core for Carbon-encoded minibatches, AdamW parameter
  groups, WSD learning-rate scheduling, gradient clipping, and distinct
  data/predictor/LoRA seed records;
- preflight-gated `geno-lewm-train --carbon-train` launch plumbing that
  wires the packaged dataset, Carbon encoder, dataset iterator,
  predictor/action encoder, optimizer, checkpoint, metrics, logs, and
  training-run metadata surfaces, with resume checkpoints validated
  against run id, dataset snapshot, seed split, and config identity;
- surprise scoring for single variants and FASTA-backed VCF rows;
- `geno-lewm-score` for manifest-backed local scorer components;
- optional native runtime loading from manifest-backed local artifacts
  when `torch`, `transformers`, and `safetensors` are installed;
- checksum receipts for single variants and per-row VCF JSONL sidecars;
- generated `runtime_preflight_report.json` for terminal demo model,
  input, dependency, backend, and network-guard readiness evidence;
- generated `batch_receipt_report.json` for terminal demo score/receipt
  JSONL streams, checked score fields, model id, calibration hash, and
  runtime identity;
- generated `terminal_demo_manifest.json` for terminal demo command,
  input, transcript, score/receipt schema, batch receipt summary, and
  report artifact identities;
- `geno-lewm-eval` artifact-level ClinVar-style metrics with
  deterministic bootstrap confidence intervals from score and label
  JSONL files, plus optional matched measured-baseline comparisons;
- `geno-lewm-carbon-baseline` generation of Carbon zero-shot baseline
  score JSONL from a local Carbon LM, held-out VCF, FASTA, and optional
  log-likelihood cache;
- `geno-lewm-eval-all` aggregation of validated metrics JSON into
  packaged source `eval_metrics.json` plus generated `eval_report.md`,
  with `eval_config.effective.yaml` recorded as a required report
  artifact;
- `bench.inference --release-efficiency` generation of measured latency,
  throughput, peak memory, hardware/runtime notes, command, and input
  identities as validated `efficiency_report.json`;
- dedicated fixture-backed `tests/ml` smoke coverage for finite fixture
  training loss, collapse-health signals, deterministic resume identity,
  and optional torch predictor initialization/learning when torch is
  installed; `.github/workflows/ci.yml` runs this as the separate
  `ml-smoke` job;
- hosted fixture-backed eval smoke regression checking with
  `python -m tools.ci.eval_smoke_gate`, which generates public
  score/label JSONL fixtures, runs `geno-lewm-eval` and
  `geno-lewm-eval-all`, enforces AUROC/AP/balanced-accuracy/baseline-delta
  thresholds, and records why real checkpoint/dataset evaluation is not
  attempted by this CI gate;
- source-distribution inventory checking with
  `python -m tools.release.check_sdist_assets dist/*.tar.gz`, wired into
  CI and the PyPI release workflow after package metadata validation,
  and covering the first-publication release toolchain for dataset,
  model, eval, efficiency, terminal demo, clean-machine replay, and
  final publication evidence;
- public v0.1 publication evidence: dataset package/data card, real
  Carbon-backed training run, checkpoint/model card, measured chr21
  ClinVar evaluation, efficiency report, terminal demo transcript and
  receipt streams, paper artifact, clean-machine replay, and final
  publication binder;
- manifest verification, update/runtime scaffolds, desktop scaffolds,
  release tooling, current public module-map docs, the
  `tests/api/public_surface.json` public API snapshot, and
  duplicate-free `__all__` checks.

## Not Done Yet

Do not present these as complete:

- no broad v0.2 benchmark report has established model quality beyond
  the narrow v0.1 chr21 ClinVar release slice;
- no attention KV-cache speedup benchmark closes the RFC-0004
  autoregressive rollout target;
- no multi-edit planning demo proves useful planning behavior from
  released artifacts;
- no first PyPI package tag has been cut;
- no clinical, deployment-readiness, privacy-assurance, or runtime
  assurance claims are supported beyond the recorded release evidence
  and checksum provenance contracts.

Fixture outputs are useful for tests, but they are not model results. The
v0.1 measured values are first-release evidence and negative findings,
not broad model-quality claims.

## Release Evidence Rules

Treat local release tooling as contracts, not as a substitute for public
artifact evidence. v0.1 exercised these gates; future v0.2 releases must
exercise them again with stronger data, eval, and rollout evidence.
When reporting status or choosing the next task, keep this mapping
explicit:

| Evidence gate | Local contract that exists | v0.1 status and future boundary |
| --- | --- | --- |
| Dataset snapshot ([#163](https://github.com/AbdelStark/GenoLeWM/issues/163)) | `python -m tools.release.dataset_snapshot` plus `python -m tools.release.dataset_package` produce package metadata, data card, manifest, split-integrity, snapshot report, and checksums | Completed and published for v0.1; v0.2 needs broader benchmark snapshots and refreshed split evidence |
| Carbon-backed training ([#164](https://github.com/AbdelStark/GenoLeWM/issues/164)) | `geno-lewm-train --carbon-preflight` and `geno-lewm-train --carbon-train --package-release-run` bind dataset, config, Carbon model, metrics, logs, checkpoint, and training-run checksums | Completed and published for v0.1; v0.2 training should wait for stronger data/eval gates |
| Evaluation and efficiency ([#165](https://github.com/AbdelStark/GenoLeWM/issues/165)) | `geno-lewm-eval`, `geno-lewm-carbon-baseline`, `geno-lewm-eval-all`, and `python -m bench.inference --release-efficiency` generate metrics, report, effective eval config, and efficiency evidence | Completed for the narrow v0.1 release; broader GenoLeWM-vs-Carbon deltas, rollout fidelity, and benchmark gates remain open |
| Source distribution | `python -m build`, `twine check`, and `python -m tools.release.check_sdist_assets dist/*.tar.gz` verify package metadata plus the full first-publication toolchain, benchmark harnesses, first-experiment configs, docs, RFCs, examples, README, ROADMAP, and AGENTS in the sdist | A tagged package release has been built by the protected release workflow from the checked tree |
| Terminal demo ([#166](https://github.com/AbdelStark/GenoLeWM/issues/166)) | `python tools/demo/terminal_inference.py` writes transcript, score/receipt JSONL, runtime preflight, batch receipt report, and terminal demo manifest | Completed for v0.1; v0.2 should demonstrate benchmark/planning behavior without clinical claims |
| Paper and publication ([#167](https://github.com/AbdelStark/GenoLeWM/issues/167), [#101](https://github.com/AbdelStark/GenoLeWM/issues/101)) | `python -m tools.release.paper_draft`, `paper_package`, `release_candidate`, `clean_machine_demo`, and `publication_report` bind paper text, package verification, public links, clean replay, and final evidence | Completed for v0.1 with a public final binder; future releases must preserve the same evidence contract |

## High-Priority Work

Prefer changes that directly improve v0.2 evidence after the v0.1
paper/demo release:

- audit stale v0.1 issue blockers and keep README, roadmap, docs, and
  issue comments aligned with the public release evidence;
- harden dataset builders, tuple-builder wiring, split enforcement,
  real upstream gnomAD/ClinVar release validation, manifests, and data
  cards; rebuild release snapshots with
  `python -m tools.release.dataset_snapshot --spec-json configs/first_experiment/dataset-snapshot-snv.json --check-spec`
  for public spec validation, then the same spec with
  `--check-inputs` for staged upstream source-file hash/size preflight,
  then the same spec with
  `--dataset-dir ... --overwrite` and explicit local upstream files so
  normalized `dataset_package.json`, `dataset_manifest.json`,
  `data_card.md`, `split_integrity.json`,
  `dataset_input_check_report.json`, `dataset_snapshot_report.json`,
  and `SHA256SUMS` are all generated together; the snapshot report must
  record the checked spec hash and upstream source file identities
  without private absolute input paths and bind input-check evidence plus generated metadata,
  manifest, data-card, split-integrity, and nested package-file artifact
  identities; the verifier must require that report in `SHA256SUMS`,
  reject duplicate snapshot file entries, reject stale report file
  identities, reject stale generated package identities, reject
  stale data cards or manifests that no longer match
  `dataset_package.json`, require
  `generated_by=tools.release.dataset_package` on generated dataset
  package metadata, reject invalid or duplicate `SHA256SUMS` paths, and
  `split_integrity.json` must record row counts, observed label/class
  balance, Parquet variant-key counts, comparable-key leakage checks,
  and the `tools.release.dataset_integrity` source header, failing when
  no train/eval comparable-key comparison can be made; `data_card.md`
  must render the same class-balance summary;
- extend fixture smoke training into the real Carbon-backed trainer so
  reproducible checkpoints, logs, and collapse diagnostics are produced;
  use the checked first-experiment configs in
  `configs/first_experiment/`; use `geno-lewm-train --carbon-preflight`
  first to verify the packaged dataset, local Carbon model directory,
  generated dataset release evidence (`dataset_package.json`,
  `dataset_input_check_report.json`, `dataset_snapshot_report.json`, and
  `SHA256SUMS`), closed-schema effective training config plus resolved
  config payload, run directory, and optional ML dependencies; preflight
  must reject stale dataset input-check evidence before launch; run
  `geno-lewm-train --carbon-train --package-release-run` on a supported
  `geno-lewm[train]` environment so real Carbon-encoded batches produce
  checkpoints, metrics, logs, and packaged training-run evidence; use
  `--resume-from predictor_checkpoint.pt` only with a compatible
  Carbon checkpoint whose run id, dataset snapshot, seed split, and
  config identity match the target run;
  package completed runs with
  `python -m tools.release.training_run`, whose manifest must carry
  `generated_by=tools.release.training_run`; release-mode verification
  must require the preflight report's dataset core-file evidence for
  `dataset_package.json`, `dataset_input_check_report.json`,
  `dataset_snapshot_report.json`, and `SHA256SUMS`; the final package
  verifier must reject stale `training_run_card.md` content that no
  longer matches `training_run_manifest.json`;
- package checkpoints with `python -m tools.release.model_package` so
  normalized `model_package.json`, `model_card.md`, and `SHA256SUMS` are
  generated from the manifest, release metadata, packaged
  `eval_metrics.json`, and `efficiency_report.json`; generated
  `model_package.json` must carry
  `generated_by=tools.release.model_package`; model metadata must list
  the training preflight report, training run manifest/card, and
  training-run checksums as `extra_files`; the package verifier
  requires model-local eval artifact references from `eval_metrics.json`
  to appear in `SHA256SUMS`, rejects invalid or duplicate checksum
  paths, and
  rejects stale model cards that do not re-render from
  `model_package.json` plus `manifest.json`, rejects training-run
  dataset/config/commit evidence that does not match the manifest plus
  eval/efficiency evidence, and rejects eval/efficiency evidence whose
  release id, dataset snapshot, commit, or model-result identity is mixed
  across artifacts;
- make evaluation write measured metrics JSON, confidence intervals, and
  measured-baseline deltas with `geno-lewm-eval`; produce the Carbon
  zero-shot baseline score artifact with `geno-lewm-carbon-baseline` and
  pass it through
  `--baseline-score-field carbon_zero_shot_score --baseline-name carbon_zero_shot`;
  primary score rows must carry `generated_by=geno-lewm-score`, and
  Carbon baseline rows must carry
  `generated_by=geno-lewm-carbon-baseline`; optional baseline
  log-likelihood cache rows must be scoped to the Carbon model and
  revision before reuse; `geno-lewm-eval` must record
  checkpoint/config/dataset/eval-config/efficiency/score artifact paths
  relative to `--artifact-root`, write `eval_config.effective.yaml`
  beside `eval_metrics.json`, and reject absolute paths outside that root;
  then aggregate and render `eval_report.md` with `geno-lewm-eval-all`;
  the aggregate must record `eval_config.effective.yaml` and metrics
  inputs as package-relative artifact paths under the aggregate metrics
  directory, and `tools.release.eval_report` must reject metrics payloads
  missing the required `eval_config` artifact; any baseline rows must
  supply `baseline`, `baseline_value`, and `delta_vs_baseline` together,
  carry a baseline score artifact in the metrics payload, plus matching
  `evaluated_variant_keys_sha256` and `baseline_evaluated_variant_keys_sha256`
  metric fields, and
  `tools.release.paper_package` must resolve those eval artifact paths
  inside the package and validate score JSONL `generated_by` markers;
  `tools.release.eval_report` must reject `eval_metrics.json` whose
  `generated_by` is not `geno-lewm-eval` or `geno-lewm-eval-all`;
  metric conclusions must explicitly reference every measured metric
  name, split, measured value, and baseline delta when present from
  `eval_metrics.json`, and `negative_findings` must be a
  non-empty list rendered as `## Negative Findings`;
  `tools.release.paper_package` must reject reports missing generated
  Summary/Artifacts evidence markers or reports that no longer match the
  packaged `eval_metrics.json`;
- generate measured latency, throughput, memory, command,
  hardware/runtime notes, and input identities with
  `python -m bench.inference --release-efficiency`; the release package
  must validate `efficiency_report.json`, require the
  `tools.release.efficiency_report` source header, require input
  identities to use package-relative paths or explicit inline labels,
  and cross-check it against the packaged eval metrics before public
  claims use those values;
- make the terminal demo consume only released model/data artifacts and
  record a reproducible transcript plus score/receipt JSONL hashes, row
  counts, JSONL field names, generated time, exit code, model
  release/version/id, artifact-input paths, a claim-boundary sentence, a
  generated `terminal_demo_manifest.json`
  with VCF input summary and `score_receipt_batch`, a generated
  `runtime_preflight_report.json`, and a generated
  `batch_receipt_report.json`; the demo runner must clear owned score,
  receipt, batch-report, and demo-manifest outputs before invoking the
  score command so stale JSONL rows cannot satisfy a later run; release
  preflight evidence must require
  native runtime dependencies and must not allow fixture/test manifests;
  the demo runner must re-open `runtime_preflight_report.json` before
  writing `terminal_demo_manifest.json` and reject stale or mutated
  model, input, command, backend, runtime-requirement, or model-artifact
  evidence from a different run;
  the verifier must reject stale or
  package-external demo VCF/FASTA identities, stale VCF input summaries,
  non-canonical command or output artifact paths, runtime-preflight
  command drift from the terminal-demo manifest command, stale
  terminal-demo manifest `runtime_preflight` summaries, stale transcript
  claim-boundary or artifact-input markers, stale JSONL
  field lists, stale `score_receipt_batch` summaries, and score/receipt
  batches whose model id or calibration hash do not match the packaged
  model manifest;
- generate the first experiment paper draft from release artifacts with
  `python -m tools.release.paper_draft` so conclusions stay tied to the
  eval report, efficiency report, and artifact identities; draft
  generation must reject stale `eval_report.md` output that no longer
  matches `eval_metrics.json` and stale terminal-demo VCF summaries,
  must require a UTC `Generated: ...Z` timestamp, must render the
  scored-input summary in Demo Evidence, and must include
  generated Citation Metadata and Negative Findings; the draft must name
  `model_package.json`, `dataset_package.json`,
  `dataset_input_check_report.json`, `dataset_snapshot_report.json`, `eval_metrics.json`,
  `eval_config.effective.yaml`, `eval_report.md`,
  `efficiency_report.json`, and demo evidence paths using
  package-local artifact names rather than build-machine root paths, and
  `tools.release.paper_package` must reject stale paper Markdown or
  drafts missing Citation Metadata or Negative Findings that no longer
  re-render from the current artifacts;
- make the paper/demo release package pass
  `python -m tools.release.paper_package`;
- dry-run Hub publication with `python -m tools.release.hub_release`
  before any model checkpoint is uploaded; paper candidates must pass a
  public `--paper-url`; the generated, versioned
  `hub_release_plan.json` must include model upload inventories covering
  both `SHA256SUMS` and `training_run_SHA256SUMS`, dataset inventories
  that include `SHA256SUMS`, and portable terminal-demo upload
  inventories whose GitHub release asset names are unique. When a paper
  URL is present, it must also bind the verified public-safe paper
  source name/path, SHA-256, and size, with
  Hugging Face commands uploading exact verified files instead of whole
  package directories, and
  `.github/workflows/release-hub-dry-run.yml` runs the package,
  Hub-plan, and release-candidate gates without publishing weights or
  requiring Hub credentials;
- publish the verified model, dataset, terminal-demo, and matching
  paper artifacts with
  `python -m tools.release.hub_publish` or
  `.github/workflows/release-hub-publish.yml` only after the dry-run is
  clean; the publish path requires `HF_TOKEN`, GitHub release
  credentials, supported Hugging Face dataset URLs, supported GitHub
  release-tag demo URLs, direct GitHub release download paper URLs whose
  final asset name matches the verified paper file, and no
  fixture-manifest override, then uploads only files named by the
  verified Hub plan before regenerating the final release-candidate
  report from public links and fetched public artifact bytes; the
  protected workflow must then run
  `python -m tools.release.clean_machine_demo` from that report with
  native runtime checks enabled before running
  `python -m tools.release.publication_assets` to bind the GitHub
  release target, upload command, and evidence-asset hashes, then upload
  those public publication evidence assets to the demo release tag,
  passing release credentials only for scoped Hub/GitHub fetches and not
  into serialized reports;
- generate `release_candidate_report.json` with
  `python -m tools.release.release_candidate` so the package verifier,
  Hub dry-run plan, public URL reachability checks, commit SHA, model
  id, dataset snapshot, model package metadata, dataset package metadata,
  dataset input check report, dataset snapshot report, source metrics JSON, effective eval config,
  generated eval report, efficiency report,
  manifest-backed predictor/action/calibration/config artifacts,
  `training_preflight_report.json`, `training_run_SHA256SUMS`,
  Hub model/dataset/demo upload inventories, provider-backed public
  artifact exact file-set, hash, and size checks plus direct paper byte
  hash/size checks, public-safe artifact
  paths, key artifact hashes, and the
  release-candidate `readiness` checklist are bound in one
  publication decision; readiness rows and blockers must preserve
  `issue_refs` back to #163, #164, #165, #166, #167, and #101; public
  link and artifact hash/size checks may be
  skipped only for explicit fixture rehearsals that pass
  `--allow-fixture-manifest`;
- keep dataset/model/training-run/paper-draft command reports and the
  terminal-demo transcript/manifest public-safe: success JSON and
  transcript text should use package-local artifact names rather than
  private absolute workstation paths;
- run
  `python -m tools.release.clean_machine_demo --release-candidate-report ... --output-dir ...`
  after a generated ready public release-candidate report exists so
  hand-authored reports, missing or failed readiness rows, non-empty
  candidate blockers, skipped or failed public link checks, skipped,
  missing, incomplete, or failed public artifact checks, stale embedded
  Hub plans, unsafe Hub-plan destinations, and malformed expected hashes
  are rejected before the published model, dataset, and terminal-demo
  assets are downloaded,
  hash-checked, verified again with `tools.release.paper_package`,
  replayed through `geno-lewm-score`, and checked before report writing
  so the replayed terminal demo manifest has a valid source header,
  passing status, matching model id, downloaded `model/manifest.json`
  hash/size identity, matching VCF/FASTA input identities, matching
  `runtime_preflight` summary, matching `score_receipt_batch` summary,
  and matching replay artifact
  hashes/sizes; the final publication binder also checks the replay
  manifest's VCF/FASTA input identities against the downloaded demo
  artifacts and checks the replay manifest's artifact table against the
  clean-machine replay report for the transcript, scores, receipts,
  runtime preflight, and batch report.
  Before scoring, the replay helper checks downloaded demo VCF/FASTA
  hashes and sizes against the downloaded demo manifest; after scoring,
  it rejects replay manifests whose VCF/FASTA identities do not match
  those downloaded inputs. The
  replay is recorded in
  `clean_machine_demo_report.json` with the release-candidate report
  filename plus hash/size identity, output-directory-relative downloaded
  artifact identities, package-verification result,
  replay transcript/manifest identities, and replay score/receipt/report
  artifact hashes without fetch tokens or private absolute workstation
  paths;
- run `python -m tools.release.publication_report --plan ... --release-candidate ... --publish-report ... --clean-machine-demo-report ... --output ...`
  after credentialed Hub publication and clean-machine replay so
  `publication_evidence_report.json` binds the release plan,
  release-candidate report, publish report, and replay report by
  public-safe filename plus hash/size identity, including the
  clean-machine replay's recorded
  release-candidate report filename/path, hash, and size identity, the
  verified paper file source name, URL, hash, and size identity, the full
  paper-critical release-candidate artifact table for model, dataset,
  eval, demo, and paper identities, public-safe release-candidate
  readiness rows plus public link and public artifact check summaries,
  with every uploaded release-candidate artifact identity in that table
  checked against the Hub plan and downloaded public artifact, the
  replayed terminal-demo manifest's model id, downloaded
  `manifest.json` identity, VCF/FASTA input identities,
  `runtime_preflight` summary, replayed runtime-preflight model/input
  identities, generated
  source-report headers, and the
  candidate-embedded Hub plan, then verifies the generated
  release-candidate readiness checklist, empty candidate blockers,
  readiness `issue_refs`, required/passing candidate `public_links` and
  `public_artifacts` checks, exact Hub-plan download set, public source
  URLs, and hashes; its `issues` entries must carry `issue_refs` so
  final publication failures route back to #163, #164, #165, #166,
  #167, and #101 before
  `hub_release_plan.json`, `release_candidate_report.json`,
  `hub_publish_report.json`, `clean_machine_demo_report.json`, and
  `publication_evidence_report.json`,
  `publication_evidence_assets.json`, plus the clean-machine replay
  transcript, manifest, score/receipt JSONL streams, runtime preflight
  report, and batch receipt report are uploaded as public evidence
  assets without private absolute workstation paths;
- keep README, roadmap, docs, and GitHub issues synchronized with actual
  implementation status.

Avoid broad refactors unless they remove a concrete blocker for those
deliverables.

## Claim Boundaries

Public docs and demos must distinguish:

- implemented behavior;
- measured results;
- fixture-only examples;
- planned work.

Do not add benchmark, model-quality, clinical, privacy, or runtime-assurance
claims unless the code and artifacts needed to reproduce them are
committed or linked from a release.

## Validation

Use focused tests while editing. Before claiming a project-direction
slice is ready, run the strongest relevant set:

```bash
uv run ruff check .
uv run ruff format --check .
uv run mypy geno_lewm tools
uv run python tools/api/snapshot.py check
uv run mkdocs build --strict
uv run pytest
```

For docs/scope cleanup, also search for stale unsupported-assurance
claims:

```bash
uv run python -m tools.lint.check_scope_language
```

Only de-scoping or historical-boundary mentions should remain.

For public-contract cleanup, also keep the coordination docs aligned:
module maps should match the current package layout rather than planned
paths, and RFC-0014/spec text should point to
`tests/api/public_surface.json` as the exhaustive enforced symbol list.
Upcoming planning solver types are not described as stable top-level exports.

## Issue Anchors

Current paper/demo blockers are tracked in GitHub issues:

- #163 dataset snapshot and data card;
- #164 first SNV Carbon-backed training run;
- #165 paper-ready results report;
- #166 terminal real-inference showcase;
- #167 first experiment paper package;
- #101 model checkpoint Hub release.

Keep issue comments concrete: state what changed, what was validated,
and which blockers remain.
