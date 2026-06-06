# GenoLeWM Roadmap

Last updated: 2026-06-06

This roadmap reflects the current project direction after the
`geno-lewm-v0.1.0-r1` paper/demo publication. The first release proved
the artifact chain with public model, dataset, demo, paper, and final
publication evidence. The next phase is v0.2 benchmark and rollout
readiness: stronger held-out evaluation, measured baseline deltas,
autoregressive rollout speed evidence, and planning-ready surfaces.
Track the post-release epic in
[#197](https://github.com/AbdelStark/GenoLeWM/issues/197).
Checksum manifests and receipts remain in scope as artifact provenance.
Future runtime assurance mechanisms beyond checksum provenance are out
of scope for the current roadmap.

---

## Current Position

Implemented:

- typed errors and stable error-code registry;
- structured logging, privacy redaction, event registry, metrics;
- edit representation, edit application, and synthetic edit samplers;
- lazy Carbon state encoder wrapper for optional local Transformers
  runtimes;
- base cross-attention `Predictor`, `ARPredictor` rollout wrapper,
  predictor loss contracts, and several pure-Python training utilities;
- deterministic `geno-lewm-train --fixture-smoke` path that writes
  resolved config, metrics, logs, checkpoint, dataset manifest, and
  training-run metadata for release plumbing tests;
- preflight-gated `geno-lewm-train --carbon-train` path that records
  CUDA/VRAM readiness, places the predictor/action encoder on the
  configured device, and can resume from compatible Carbon checkpoints
  after validating run id, dataset snapshot, seed split, and config
  identity;
- RFC-0006 training tuple builder with per-window source mix,
  ClinVar-to-synthetic fallback, absolute variant providers, and
  holdout filtering;
- source-state cache lookup in `geno_lewm.training.encode_training_batch`
  using the documented `GENO_LEWM_CACHE` index, with live Carbon
  fallback on cache miss and edited targets encoded on the fly;
- local gnomAD and ClinVar VCF-to-Parquet shard builders exposed through
  `geno-lewm-prepare-gnomad` and `geno-lewm-prepare-clinvar`;
- `geno-lewm-eval` artifact-level metrics and deterministic bootstrap
  confidence intervals from score JSONL and held-out ClinVar-style
  label JSONL;
- `geno-lewm-carbon-baseline` Carbon zero-shot baseline score artifact
  generation from an explicit local Carbon LM, VCF, FASTA, and optional
  log-likelihood cache;
- `geno-lewm-eval-all` aggregation of validated measured metrics JSON
  into packaged source `eval_metrics.json` plus generated
  `eval_report.md`, with `eval_config.effective.yaml` recorded as an
  required eval artifact;
- `bench.inference --release-efficiency` generation of validated
  single-variant latency, batched throughput, peak memory,
  hardware/runtime notes, command, and input identities;
- dedicated fixture-backed `tests/ml` smoke coverage for finite fixture
  training loss, collapse-health signals, deterministic resume identity,
  and optional torch predictor initialization/learning when torch is
  installed; CI runs it as the separate `ml-smoke` job;
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
- manifest, hash, input/output commitment, and checksum receipt helpers;
- single-variant score receipt emission for manifest-verified local
  scorer components;
- per-row VCF receipt sidecars as JSONL, with one v1 checksum receipt
  per scored alternate;
- optional native runtime component loading from manifest-backed local
  artifacts when the ML stack is installed;
- API snapshot tests, release tooling, docs scaffolding, PyPI workflow;
- runtime/update/desktop scaffolds;
- v0.1 public publication evidence: dataset package/data card, real
  Carbon-backed training run, checkpoint/model card, measured chr21
  ClinVar evaluation, efficiency evidence, terminal transcript and
  receipt streams, paper artifact, clean-machine replay, and final
  `publication_evidence_report.json` with `ok=true`.

Not implemented end-to-end yet:

- attention KV-cache speedups for RFC-0004 autoregressive rollout;
- broader held-out benchmark coverage beyond the narrow v0.1 chr21
  ClinVar release slice;
- measured v0.2 GenoLeWM-vs-Carbon baseline deltas over coding and
  non-coding splits with exact evaluated variant identities;
- rollout-fidelity state-row generation plus performance regression gates
  beyond the implemented `geno-lewm-rollout` metrics aggregator;
- planning-ready API/CLI demos backed by measured predictor evidence;
- first PyPI package tag.

## Completed v0.1 Publication Evidence

| Gate | Issue | v0.1 evidence |
| --- | --- | --- |
| Dataset snapshot and data card | [#163](https://github.com/AbdelStark/GenoLeWM/issues/163) | Public dataset package and data card: <https://huggingface.co/datasets/abdelstark/geno-lewm-data> |
| First Carbon-backed run | [#164](https://github.com/AbdelStark/GenoLeWM/issues/164) | Published `geno-lewm-coherent-cd2bfcc` run evidence, 20,000 steps / 160,000 samples |
| Paper-ready results report | [#165](https://github.com/AbdelStark/GenoLeWM/issues/165) | Published `eval_metrics.json`, `eval_report.md`, `eval_config.effective.yaml`, and `efficiency_report.json` |
| Terminal real-inference showcase | [#166](https://github.com/AbdelStark/GenoLeWM/issues/166) | Public terminal transcript and score/receipt evidence: <https://github.com/AbdelStark/GenoLeWM/releases/tag/geno-lewm-v0.1.0-r1> |
| First experiment paper package | [#167](https://github.com/AbdelStark/GenoLeWM/issues/167) | Public paper artifact: <https://github.com/AbdelStark/GenoLeWM/releases/download/geno-lewm-v0.1.0-r1/paper.md> |
| Model checkpoint Hub release | [#101](https://github.com/AbdelStark/GenoLeWM/issues/101) | Public model package, model card, checkpoint files, manifest, checksums, eval report, and demo links: <https://huggingface.co/abdelstark/geno-lewm> |

## Release Evidence Gates

Local tools define the reusable contract for paper/demo releases. The
v0.1 release crossed the first-publication line by exercising those
contracts on public artifacts and clean-machine replay evidence; v0.2
must reuse them with stronger data, evaluation, and rollout evidence.

| Gate | Local contract | v0.1 status and v0.2 boundary |
| --- | --- | --- |
| Dataset package | `python -m tools.release.dataset_snapshot --spec-json configs/first_experiment/dataset-snapshot-snv.json --check-spec` validates the checked snapshot spec; `--check-inputs` records staged upstream input hashes/sizes; `python -m tools.release.dataset_snapshot` and `python -m tools.release.dataset_package` generate package metadata, data card, manifest, split-integrity, input-check report, snapshot report, and checksums once local upstream files are staged | Completed and published for v0.1 ([#163](https://github.com/AbdelStark/GenoLeWM/issues/163)); v0.2 needs broader benchmark snapshots and refreshed split evidence |
| Carbon-backed training | `geno-lewm-train --carbon-preflight` and `geno-lewm-train --carbon-train --package-release-run` bind config, dataset, CUDA/VRAM readiness, Carbon model, metrics, logs, checkpoint, and training-run checksums | Completed and published for v0.1 ([#164](https://github.com/AbdelStark/GenoLeWM/issues/164)); v0.2 training should wait for stronger data/eval gates |
| Evaluation and efficiency | `geno-lewm-eval`, `geno-lewm-carbon-baseline`, `geno-lewm-eval-all --require-v02-vep-metrics --require-v02-rollout-metrics`, `geno-lewm-rollout`, `python -m tools.release.rollout_state_examples`, `python -m tools.release.rollout_state_rows`, `python -m tools.release.v02_benchmark_suite`, and `python -m bench.inference --release-efficiency` generate or orchestrate metrics, report, effective eval config, aggregate VEP/rollout metric coverage checks, cache-keyed latent examples, rollout-state rows, and efficiency evidence | Completed for the narrow v0.1 release ([#165](https://github.com/AbdelStark/GenoLeWM/issues/165)); broader GenoLeWM-vs-Carbon deltas, real held-out latent rollout specs/states, and benchmark gates remain open |
| v0.2 benchmark readiness | `python -m tools.release.v02_benchmark_readiness --metrics-json ... --rollout-speed-report ... --rollout-speed-scope-report ... --efficiency-report ... --output ... --require-ok` reconciles measured eval values/deltas, efficiency, AR rollout speed, optional accepted rollout-speed scope decisions, confidence-interval coverage, rollout generation report artifacts, and release-input provenance into `v0.2_benchmark_readiness_report.json` | New v0.2 gate for [#197](https://github.com/AbdelStark/GenoLeWM/issues/197); expected `ok=false` until broader benchmark rows, CI-bearing VEP metrics, non-fixture release inputs, and the [#42](https://github.com/AbdelStark/GenoLeWM/issues/42) rollout speed target either passes or is explicitly re-scoped through `python -m tools.release.rollout_speed_scope` |
| Source distribution | `python -m build`, `twine check`, and `python -m tools.release.check_sdist_assets dist/*.tar.gz` verify package metadata and the release-critical repo assets needed by the dataset, model, eval, efficiency, terminal-demo, clean-replay, and publication-evidence gates | No tagged package release has been built by the protected release workflow yet |
| Terminal demo | `python tools/demo/terminal_inference.py` emits transcript, score/receipt JSONL, runtime preflight, batch receipt report, and terminal demo manifest | Completed for v0.1 ([#166](https://github.com/AbdelStark/GenoLeWM/issues/166)); v0.2 should demonstrate benchmark/planning behavior without clinical claims |
| Paper and publication | `python -m tools.release.paper_draft`, `python -m tools.release.paper_package`, `python -m tools.release.release_candidate`, `python -m tools.release.clean_machine_demo`, and `python -m tools.release.publication_report` bind paper text, package verification, public links, clean replay, and final evidence | Completed for v0.1 through [#167](https://github.com/AbdelStark/GenoLeWM/issues/167) and [#101](https://github.com/AbdelStark/GenoLeWM/issues/101); final binder has `ok=true` and zero issues |

---

## Milestone 0 - Direction Cleanup

**Goal:** make public claims match the project state.

**Exit criteria:**

- README, roadmap, agent context, and implementation tracker state the
  real current status.
- Unsupported runtime-assurance work beyond checksum provenance is
  removed from active docs and issue planning.
- GitHub issues track remaining v0.2 benchmark, rollout, and planning gaps.
- The scope-language guard covers docs, code, config, JSON, and notebooks
  so unsupported trust claims cannot reappear in public examples.
- Tests pass after the receipt schema rejects unsupported runtime
  assurance modes and after notebook scope-language coverage is enforced.

**Status:** completed for v0.1; ongoing as part of post-release hygiene.

---

## Milestone 1 - First Real Inference Slice

**Goal:** run one real genomic edit through the intended model path from
terminal input to a scored output.

**Scope:**

- load a pinned Carbon checkpoint or deterministic local substitute in
  development mode;
- produce a state vector for a reference window;
- apply an SNV and produce the alternate-window target state;
- run the GenoLeWM action encoder + predictor path;
- emit a structured score object and optional checksum receipt;
- expose the slice through a terminal command with fixture and real
  input modes.

**Exit criteria:**

- `geno-lewm-score` or equivalent terminal command runs on a clean clone;
- command output includes input summary, model identity, score fields,
  latency, and artifact paths;
- transcript records generated score/receipt JSONL hashes and row
  counts plus JSONL field names, and the release verifier checks those
  files exist;
- demo fixture is small enough for CI;
- docs include a copy-paste demo transcript generated from the command
  by `tools/demo/terminal_inference.py`.

**Status:** completed for v0.1 through the public terminal demo release
and clean-machine replay.

---

## Milestone 2 - Dataset Snapshot

**Goal:** make the first experiment data fully reproducible.

**Scope:**

- dataset builder for the selected Carbon corpus slice;
- local release-file builders for gnomAD common variants and ClinVar
  labels;
- training tuple builder wired to prepared edit sources, with source
  mix and holdout enforcement;
- ClinVar coding/non-coding evaluation snapshot;
- synthetic smoke fixtures for CI;
- data cards covering source, license, preprocessing, splits, and known
  leakage risks;
- manifest hashes for all published files.

**Exit criteria:**

- `python -m tools.release.dataset_snapshot --spec-json configs/first_experiment/dataset-snapshot-snv.json --check-spec`
  validates the checked first-experiment snapshot spec without requiring
  local upstream files; the same spec with
  `--check-inputs` verifies the staged local Carbon, gnomAD, and
  ClinVar source files and records their SHA-256/size identities before
  conversion; the same spec with
  `--dataset-dir ... --overwrite` rebuilds the local release snapshot
  from explicit local upstream Carbon, gnomAD, and ClinVar files, and writes
  `dataset_input_check_report.json` plus
  `dataset_snapshot_report.json` with the checked spec hash and upstream
  source file identities plus generated input-check metadata/manifest/data-card/integrity
  artifact identities; the release verifier requires that report, checks
  its file, nested package-file, and generated-package identities,
  rejects private absolute source paths and duplicate snapshot file
  entries, and requires it in `SHA256SUMS`;
- `geno-lewm-prepare-gnomad --input-vcf ... --output ...` writes
  `${GENO_LEWM_DATA}/gnomad/{release}/variants.parquet` with PASS
  variants above the global AF threshold;
- `geno-lewm-prepare-clinvar --input-vcf ... --release ... --output ...`
  writes `${GENO_LEWM_DATA}/clinvar/{release}/variants.parquet`, loads
  VUS rows, and excludes VUS/OTHER from labelled eval sets;
- tuple-builder tests validate the 3/3/1/1 source allocation, ClinVar
  fallback behavior, variant coordinate filtering, deterministic
  synthetic providers, and holdout filtering;
- `python -m tools.release.dataset_package --dataset-dir ... --metadata-json ...`
  writes normalized `dataset_package.json`, `data_card.md`,
  `dataset_manifest.json`, `dataset_input_check_report.json`,
  `split_integrity.json`, and `SHA256SUMS`
  from the rebuilt shard files and release metadata; the release
  verifier requires `generated_by=tools.release.dataset_package` and
  rejects stale data cards or manifests that no longer match
  `dataset_package.json`, plus invalid or duplicate checksum paths;
- train/eval split integrity tests pass and the generated
  `split_integrity.json` reports observed record counts, label/class
  balance, Parquet variant-key counts, comparable-key leakage checks,
  and the `tools.release.dataset_integrity` source header, failing when
  no train/eval comparable-key comparison can be made; `data_card.md`
  renders the same class-balance summary;
- dataset card and manifest are published with the release artifacts.

**Status:** completed for v0.1; v0.2 should broaden the benchmark data
and refresh split evidence rather than reopen the first snapshot gate.

---

## Milestone 3 - First Training Run

**Goal:** train the minimum viable SNV predictor and publish a checkpoint.

**Scope:**

- frozen Carbon encoder;
- SNV-only action encoder;
- predictor head with documented parameter count;
- train config committed under version control;
  `configs/first_experiment/train-carbon-500m-snv.yaml` is the checked
  first paper-run training config and
  `configs/first_experiment/eval-clinvar-snv.yaml` records the matching
  eval configuration;
- deterministic fixture smoke mode for CLI/release plumbing;
- clean-machine `geno-lewm-train --carbon-preflight` readiness report
  for the packaged dataset, local Carbon model directory, training
  config, run directory, and optional ML dependencies; the CLI records
  `training_config.effective.yaml` so preflight and release packaging
  bind the same effective config;
- `geno_lewm.data.GenoLeWMDataset` iterator over checked source windows
  and edit-source providers;
- torch trainer core for Carbon-encoded batches, AdamW parameter groups,
  WSD learning-rate scheduling, gradient clipping, and distinct
  data/predictor/LoRA seed records;
- preflight-gated `geno-lewm-train --carbon-train` launcher that connects
  the packaged dataset, local Carbon encoder, dataset iterator, action
  encoder, predictor, optimizer, checkpoint, metrics, logs, and
  `training_run.json` metadata path, and can resume from compatible
  Carbon checkpoints whose run id, dataset snapshot, seed split, and
  config identity match the target run;
- collapse monitoring and deterministic seed plumbing;
- checkpoint export with model card and manifest.

**Exit criteria:**

- fixture smoke training runs end-to-end from a clean environment;
- first-experiment train/eval config files load through the closed
  GenoLeWM config schema;
- `geno-lewm-train --carbon-preflight --dataset-dir ... --carbon-model-dir ... --training-config ... --run-dir ...`
  writes `training_preflight_report.json` with `ok=true`, the training
  config hash, the resolved config payload, and fresh packaged dataset
  evidence including `dataset_package.json`,
  `dataset_input_check_report.json`, `dataset_snapshot_report.json`, and
  `SHA256SUMS`;
- trainer core consumes Carbon-encoded source/target states and
  relative edits through `geno_lewm.training.TorchTrainer`, including
  source `s_t` cache lookup when a compatible cache index is present;
- dataset iterator streams deterministic training tuples with source
  windows and enforces source mix plus holdout policy before encoding;
- `geno-lewm-train --carbon-train --dataset-dir ... --carbon-model-dir ... --training-config ... --run-dir ...`
  refuses to launch unless Carbon preflight succeeds and can add
  `--package-release-run` to emit `training_run_manifest.json`,
  `training_run_card.md`, and `training_run_SHA256SUMS` immediately
  after a successful run; `--resume-from predictor_checkpoint.pt`
  continues only from compatible checkpoints and records the resumed
  step in metrics, logs, and `training_run.json`;
- Carbon-backed training runs end-to-end from a clean environment;
- checkpoint, config, logs, and metrics are archived;
- `python -m tools.release.training_run --run-dir ... --metadata-json ...`
  writes `training_run_manifest.json`, `training_run_card.md`, and
  `training_run_SHA256SUMS` for the completed run, including the
  checksum-covered `training_preflight_report.json` for release
  Carbon-backed runs and `generated_by=tools.release.training_run`.
  Release-mode verification requires the preflight report's dataset
  core-file evidence for `dataset_package.json`,
  `dataset_input_check_report.json`, `dataset_snapshot_report.json`, and
  `SHA256SUMS`;
  the paper/demo package verifier rejects stale `training_run_card.md`
  content that no longer matches `training_run_manifest.json`;
- model card states data, hardware, runtime, failure modes, and intended
  research-only use;
- `python -m tools.release.model_package --model-dir ... --metadata-json ...`
  writes normalized `model_package.json`, the model card, and
  `SHA256SUMS` from manifest-backed checkpoint artifacts, packaged
  `eval_metrics.json`, packaged `efficiency_report.json`, model-local
  eval artifact references from the metrics payload, and release metadata;
  it requires `generated_by=tools.release.model_package` and rejects
  model metadata that omits required training-run evidence files from
  `extra_files`; it rejects eval/efficiency evidence whose release id,
  dataset snapshot, commit, or model-result identity does not match the
  manifest-backed package.
  `tools.release.paper_package` rejects stale model cards that no longer
  match `model_package.json` plus `manifest.json` and rejects missing
  checksum entries, invalid checksum digests, and duplicate checksum
  paths for model-local eval artifacts; it also binds training-run
  dataset snapshot, training config path/hash, and commit identity to the
  manifest plus eval/efficiency evidence;
- negative results are accepted if the run is reproducible and analyzed.

**Status:** completed for v0.1; the published run is a first evidence
baseline, not a reason to publish stronger model-quality claims.

---

## Milestone 4 - Evaluation Report

**Goal:** produce the first paper-grade results table.

**Scope:**

- ClinVar coding and non-coding metrics;
- Carbon zero-shot baseline;
- rollout cosine-similarity/L2/Recall@k metrics from measured state rows;
- throughput and memory measurements;
- confidence intervals where sample size permits;
- ablations for action encoding and predictor loss where feasible.

**Exit criteria:**

- `eval_metrics.json` is packaged as the source metrics artifact and
  `eval_report.md` is generated by code, not handwritten; the metrics
  payload must carry `generated_by=geno-lewm-eval` or
  `generated_by=geno-lewm-eval-all`, and measured baseline delta rows
  must carry matching evaluated variant-key hashes;
- `eval_config.effective.yaml` records the committed eval config plus
  explicit CLI overrides used to generate or aggregate the report;
  `geno-lewm-eval` writes it beside the metrics JSON, and
  `geno-lewm-eval-all --require-v02-vep-metrics --require-v02-rollout-metrics`
  refreshes it while recording metrics inputs as package-relative
  artifact paths under the aggregate metrics directory and failing
  incomplete v0.2 VEP or rollout-fidelity coverage;
  accepted metrics payloads must include this file as the `eval_config`
  artifact;
- `python -m tools.release.v02_benchmark_suite --manifest ... --output-report ...`
  writes a package-relative command plan for scoring, Carbon-baseline,
  eval, rollout, aggregate, and readiness commands; without `--execute`
  the report keeps `ok=false`, and with `--execute` it clears each
  step's declared output files, then requires the command to exit
  successfully and write those outputs again, but does not replace the
  downstream metrics/readiness validators;
- `python -m tools.release.rollout_state_examples --spec-jsonl ... --cache-dir ...`
  resolves explicit cache keys for measured source, target, and
  candidate latent states into the examples JSONL consumed by
  `tools.release.rollout_state_rows`;
- `python -m tools.release.rollout_state_rows --examples-jsonl ... --model-dir ...`
  generates `geno-lewm-rollout-states` JSONL from those measured
  source/target/candidate latent examples and the manifest-backed action
  encoder/predictor; both helpers record package-relative provenance but
  do not run Carbon encoding or construct held-out haplotypes. Release
  readiness requires rollout metrics to carry both generation reports as
  package-relative artifacts;
- `python -m tools.release.rollout_speed_scope --rollout-speed-report ... --output ...`
  records an accepted #42/#197 decision to re-scope a failed RFC-0004
  speed target; readiness accepts this only when the scope report binds
  the exact failing `bench.rollout` hash/size, failed K targets,
  accepted decision URL, rationale, and replacement target, and it still
  records the AR-speed row as `rescoped` rather than passing speed
  evidence;
- `efficiency_report.json` is generated with
  `python -m bench.inference --release-efficiency ... --output-json ...`
  and records measured single-variant latency, batched throughput, peak
  memory, benchmark command, hardware/runtime notes, package-relative
  or inline input identities, samples, warm-up, limitations, and the
  `tools.release.efficiency_report` source header;
- `geno-lewm-eval --scores-jsonl ... --labels-jsonl ... --output-metrics ...`
  emits the measured metrics payload for held-out ClinVar-style labels,
  including deterministic stratified bootstrap confidence intervals
  unless explicitly disabled, and can attach a matched measured baseline
  score artifact via `--baseline-scores-jsonl ... --baseline-name ...`;
  primary score rows are rejected unless
  `generated_by=geno-lewm-score`; report artifact paths are recorded
  package-relative under `--artifact-root`, defaulting to the metrics
  output directory, including the generated `eval_config.effective.yaml`,
  and absolute paths outside that root are rejected;
- `geno-lewm-carbon-baseline --vcf ... --fasta ... --carbon-model-dir ... --output-scores ...`
  writes `carbon_zero_shot_scores.jsonl` with
  `carbon_zero_shot_score = -(logLik_alt - logLik_ref)` and optional
  sequence log-likelihood cache rows scoped to the Carbon model and
  revision before reuse; Carbon baseline rows are rejected
  unless `generated_by=geno-lewm-carbon-baseline`;
- baseline comparison rows in `eval_report.md` are accepted only when
  `baseline`, `baseline_value`, and `delta_vs_baseline` are supplied
  together and the metrics payload records a baseline score artifact;
- `python -m tools.release.eval_report --metrics-json ... --output ...`
  accepts the measured metrics payload, rejects placeholder content, and
  rejects metrics payloads not generated by the eval CLIs or missing the
  required `eval_config` artifact; conclusions
  must explicitly reference every measured metric name, split, measured
  value, and baseline delta when present from `eval_metrics.json`;
  `negative_findings` must be a non-empty list and
  is rendered as `## Negative Findings`;
- `python -m tools.release.paper_package` verifies the generated
  eval-report Summary, Artifacts, Results, model/dataset identity lines,
  checkpoint/config/dataset-manifest/eval-config/efficiency-report rows,
  and baseline score artifact rows when baselines are reported; it
  resolves eval artifact paths inside the package and validates
  primary/baseline score JSONL `generated_by` markers; it re-renders
  `eval_report.md` from packaged `eval_metrics.json`, validates
  `efficiency_report.json`, rejects missing or mismatched efficiency
  source headers, rejects eval/efficiency identity mismatches against
  the manifest release id and training dataset snapshot, and rejects
  stale Markdown;
- `geno-lewm-eval-all --metrics-json ... --output-metrics ... --output-report ... --require-v02-vep-metrics --require-v02-rollout-metrics`
  aggregates validated measured metrics into the paper-report artifact
  without rerunning benchmarks, rejects incomplete v0.2 VEP and
  rollout-fidelity metric coverage, and records required
  `eval_config.effective.yaml`;
- report distinguishes measured values from planned targets;
- all metrics link to config, commit, dataset manifest, and checkpoint;
- conclusions list what worked, what failed, and what experiment comes next.

**Status:** completed for the narrow v0.1 release; v0.2 needs broader
ClinVar/Carbon baseline, real rollout-fidelity state-row generation, and
performance benchmarks.

---

## Milestone 5 - Paper/Demo Release

**Goal:** make the first public showcase reproducible end to end.

**Scope:**

- terminal demo with real model inference;
- published dataset snapshot;
- published model checkpoint;
- README quickstart using the released artifacts;
- generated paper draft with experiment, measured results, limitations,
  conclusions, and artifact availability.

**Exit criteria:**

- a new user can install, download the released artifacts, and run the
  terminal demo without private files;
- demo package includes `terminal-demo-transcript.md`, `scores.jsonl`,
  `receipts.jsonl`, `terminal_demo_manifest.json`,
  `runtime_preflight_report.json`, and
  `batch_receipt_report.json` with matching hashes, row counts, JSONL
  field names, model id, calibration hash, runtime identity, checked
  score fields, native-runtime dependency evidence, backend probes,
  package-local demo input identities, VCF input summary,
  generated time, exit code, model release/version, artifact-input
  transcript markers, claim-boundary wording,
  fixture-manifest allowance set
  to false, canonical command paths, canonical artifact paths, and
  per-row score/receipt outputs; the demo runner clears owned score,
  receipt, batch-report, and demo-manifest outputs before invoking the
  score command so stale JSONL rows cannot satisfy a later run; the
  demo runner re-opens `runtime_preflight_report.json` before writing
  `terminal_demo_manifest.json` and rejects stale or mutated model,
  input, command, backend, runtime-requirement, or model-artifact
  evidence from a different run; the
  package verifier rejects stale
  VCF input summaries, stale transcript claim-boundary or artifact-input
  markers, stale terminal-demo manifest `runtime_preflight` summaries,
  stale manifest field lists, stale
  `score_receipt_batch` summaries, and score/receipt
  batches whose model id or calibration hash do not match the packaged
  model manifest;
- `python -m tools.release.paper_package` passes for the model,
  dataset, demo, and paper artifacts;
- `python -m tools.release.paper_draft --model-dir ... --dataset-dir ... --demo-dir ... --output ...`
  generated the paper draft from release artifacts before verification,
  rejecting stale `eval_report.md` output that no longer matches
  `eval_metrics.json` and stale terminal-demo VCF summaries, rendering
  the scored-input summary in Demo Evidence, requiring a UTC
  `Generated: ...Z` timestamp, and including generated
  Citation Metadata and Negative Findings, with Artifact Availability entries for
  `model_package.json`,
  `dataset_package.json`, `dataset_input_check_report.json`,
  `dataset_snapshot_report.json`,
  `eval_metrics.json`, `eval_config.effective.yaml`, `eval_report.md`,
  `efficiency_report.json`, and the terminal demo evidence, using
  package-local artifact names rather than build-machine root paths; the package
  verifier rejects stale paper drafts or drafts missing Citation
  Metadata or Negative Findings that no longer match the current
  artifact set;
- `python -m tools.release.hub_release --model-dir ... --dataset-dir ... --demo-dir ...`
  emits a verified dry-run upload plan before the Hub release is pushed,
  with `--paper-url` required whenever a paper artifact is part of the
  candidate and the verified public-safe paper source name/path plus
  hash/size recorded in the plan; the versioned plan now inventories model files including
  training-run checksum artifacts, checksum-covered dataset files
  including `SHA256SUMS`, and portable terminal-demo files with unique
  GitHub release asset names, and emits exact-file model/dataset upload
  commands plus demo and matching paper upload commands when the public
  URLs identify supported targets;
- `.github/workflows/release-hub-dry-run.yml` can run the package
  verifier, Hub dry-run planner, and release-candidate report without
  publishing weights or requiring Hub credentials;
- `python -m tools.release.hub_publish --model-dir ... --dataset-dir ... --demo-dir ...`
  and `.github/workflows/release-hub-publish.yml` can publish the
  verified model, dataset, terminal-demo, and matching paper artifacts
  after the dry-run is clean, requiring `HF_TOKEN`, GitHub release
  credentials, supported Hugging Face/GitHub target URLs, and protected
  `release` environment approval, with the workflow syncing the locked
  `dev`, `train`, `eval`, and `deploy` extras for the native
  clean-machine replay; the publish helper uploads only files named by
  the verified Hub plan and requires paper artifacts to use a direct
  GitHub release download URL whose final asset name matches the
  verified paper file. It regenerates the final
  `release_candidate_report.json` after upload from public links and
  fetched public artifact bytes. The protected workflow runs the
  clean-machine replay from that report before
  running `python -m tools.release.publication_assets` to bind the
  GitHub release target, upload command, and publication-evidence asset
  identities before uploading those assets to the demo release tag, using
  release credentials only for scoped artifact fetches;
- `python -m tools.release.release_candidate --model-dir ... --dataset-dir ... --demo-dir ... --paper-path ... --paper-url ... --repo-id ... --dataset-url ... --demo-url ... --commit-sha ... --output ...`
  emits `release_candidate_report.json` with `ready=true`, including
  successful reachability checks for the public model, dataset, demo,
  and paper links plus hashes for `dataset_package.json`,
  `dataset_input_check_report.json`, `dataset_snapshot_report.json`, `model_package.json`,
  `eval_metrics.json`, `eval_config.effective.yaml`, `eval_report.md`,
  `efficiency_report.json`,
  manifest-backed predictor/action/calibration/config artifacts, and
  `training_preflight_report.json` plus `training_run_SHA256SUMS`, with
  Hub model/dataset/demo upload inventories, provider-backed public
  artifact exact file-set, hash, and size checks plus direct paper byte
  hash/size checks, public-safe artifact
  paths, and a machine-readable
  `readiness` checklist serialized in the report, with `issue_refs`
  routing readiness rows and blockers to #163, #164, #165, #166, #167,
  and #101; public checks can be skipped only for
  fixture rehearsals that explicitly allow fixture manifests;
- dataset/model/training-run/paper-draft package command reports and
  terminal-demo transcript/manifest output serialize package-local
  artifact names rather than private absolute workstation paths;
- `python -m tools.release.clean_machine_demo --release-candidate-report ... --output-dir ...`
  downloads the published model, dataset, and terminal-demo assets from
  the generated ready release-candidate report, rejecting hand-authored
  reports, missing or failed readiness rows, non-empty candidate blockers,
  skipped or failed public link checks, skipped, missing, incomplete, or
  failed public artifact checks, stale embedded Hub plans, unsafe
  Hub-plan destinations, and malformed expected hashes before network
  fetches, verifies downloaded SHA-256 values, re-runs the release-package verifier on
  the downloaded artifacts, reruns the terminal demo from those public
  bytes, then rejects replayed `terminal_demo_manifest.json` files with
  invalid source headers, non-passing status, model id mismatch,
  downloaded `model/manifest.json` hash/size mismatch, stale
  VCF/FASTA input identities, stale `runtime_preflight` summaries,
  stale `score_receipt_batch` summaries, or replay artifact hash/size
  drift before writing the clean-machine report. The final publication binder also checks the replay manifest's VCF/FASTA
  input identities against the downloaded demo artifacts and checks the
  replay manifest's artifact table against the clean-machine replay
  report for the transcript, scores, receipts, runtime preflight, and
  batch report.
  Before scoring, the replay helper checks the downloaded demo
  VCF/FASTA hashes and sizes against the downloaded demo manifest; after
  scoring, it rejects replay manifests whose VCF/FASTA identities do not
  match those downloaded inputs. The replay helper writes
  `clean_machine_demo_report.json` with the
  release-candidate report filename plus hash/size identity,
  output-directory-relative downloaded artifact identities,
  package-verification result, and replay
  transcript/manifest/score/receipt/report artifact hashes; optional
  Hub/GitHub tokens are used only for downloads and are not
  recorded in the report, and private absolute workstation paths are not
  serialized;
- `python -m tools.release.publication_report --plan ... --release-candidate ... --publish-report ... --clean-machine-demo-report ... --output ...`
  emits `publication_evidence_report.json` after the credentialed Hub
  publish and clean-machine replay, binding the four top-level reports
  by public-safe filename plus SHA-256 identity plus the
  clean-machine replay's recorded release-candidate report
  filename/path, hash, and size identity, the verified paper file source
  name, URL, hash, and size identity, the full paper-critical
  release-candidate artifact table for model, dataset, eval, demo, and
  paper identities, public-safe release-candidate readiness rows plus
  public link and public artifact check summaries, every uploaded
  release-candidate artifact identity in that table checked against the
  Hub plan plus the downloaded public artifact, and the
  replayed terminal-demo manifest's model id, downloaded
  `manifest.json` identity, VCF/FASTA input identities,
  `runtime_preflight` summary, and replayed runtime-preflight
  model/input identities, and failing
  if the release candidate's
  embedded Hub plan, final candidate, generated readiness checklist,
  empty candidate blockers, current readiness `issue_refs`, generated
  `public_links` and `public_artifacts` sections, exact Hub-plan
  download set, public source URLs, hashes, or replay artifact set
  disagree without serializing private absolute workstation paths; its
  `issues` entries carry `issue_refs` so final publication failures
  route back to #163, #164, #165, #166, #167, and #101; the
  protected workflow uploads the Hub plan, release-candidate report,
  publish report, clean-machine replay report, final publication
  evidence report, publication evidence asset manifest, replay
  transcript, replay manifest, score/receipt JSONL streams, runtime
  preflight report, and batch receipt report as public GitHub release
  assets;
- GitHub release links to dataset, model, eval report, and demo transcript;
- paper draft has enough evidence for a first workshop/preprint submission.

**Status:** completed for v0.1 with public model, dataset, demo, paper,
clean-machine replay, and final publication evidence.

---

## Later Work

After the first paper/demo release, the v0.2 workstreams are:

- indels and MNVs;
- multi-edit rollout;
- calibrated surprise-score validation beyond the v0.1 artifact;
- planning with CEM;
- ONNX/Core ML export and local desktop workflow;
- larger Carbon checkpoints and LoRA adaptation.

These are now the active roadmap. Each workstream needs measured
evidence and negative findings before it can support public claims.

---

## Anti-Roadmap

GenoLeWM will not:

- present fixture outputs as model results;
- publish a clinical-use claim;
- require users to upload personal genome data to a hosted service;
- train a separate DNA encoder from scratch for the first paper;
- add a chat layer before the model/eval path is real.
