# Implementation Tracker

Last updated: 2026-06-06

This file is a human-maintained snapshot of the current execution plan.
GitHub issues remain the source of truth for issue state.

## Current Status

The repository has moved beyond the original spec-bootstrap phase. The
implemented surface now includes:

- error taxonomy and error-code gates;
- observability, redaction, metrics, and WandB sink integration;
- action representation, edit application, and synthetic samplers;
- RFC-0006 tuple-builder contracts for source mix, ClinVar fallback,
  absolute variant providers, and holdout filtering;
- local gnomAD and ClinVar VCF-to-Parquet shard builders with
  schema-checked Parquet loaders;
- lazy Carbon state encoder wrapper for optional local Transformers
  runtimes;
- base cross-attention `Predictor`, `ARPredictor` rollout wrapper,
  predictor losses, and training stability helpers;
- deterministic fixture smoke trainer that writes config, metrics, log,
  checkpoint, dataset manifest, and training-run metadata;
- preflight-gated Carbon trainer launcher with compatible checkpoint
  resume validation for run id, dataset snapshot, seed split, and config
  identity;
- surprise score library, FASTA-backed VCF scoring, and the score CLI
  path for manifest-verified injected scorer components;
- optional native runtime component loading from manifest-backed local
  artifacts when the ML stack is installed;
- artifact manifests, checksum receipts, single-variant score receipt
  emission, per-row VCF receipt JSONL sidecars, and the verify notebook;
- `geno-lewm-eval` artifact-level ClinVar-style metrics with
  deterministic bootstrap confidence intervals from score and label
  JSONL files, plus optional matched measured-baseline comparisons that
  require a recorded baseline score artifact;
- `geno-lewm-carbon-baseline` generation of Carbon zero-shot baseline
  score JSONL from a local Carbon LM, held-out VCF, FASTA, and optional
  sequence log-likelihood cache;
- `geno-lewm-eval-all` aggregation of validated metrics JSON into
  packaged source `eval_metrics.json` plus generated `eval_report.md`,
  with `eval_config.effective.yaml` recorded as a required report
  artifact;
- `bench.inference --release-efficiency` generation of measured latency,
  throughput, peak memory, command, hardware/runtime notes, and input
  identities as validated `efficiency_report.json`;
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
  CI and the PyPI release workflow after package metadata validation;
- `tools.release.paper_package` validation of generated eval-report
  Summary/Artifacts markers, required artifact rows, model/dataset
  identity lines, baseline score artifact rows, exact `eval_report.md`
  rendering from packaged `eval_metrics.json`, and valid
  `efficiency_report.json`;
- runtime/update/desktop scaffolds;
- release tooling, API-snapshot tooling, duplicate-free `__all__`
  checks, and current public module-map docs.

The v0.1 paper/demo release is complete. Remaining high-priority work is
post-release evidence: broader held-out benchmarks, GenoLeWM-vs-Carbon
baseline deltas, RFC-0004 attention KV-cache speedups, rollout-fidelity
state-row generation beyond the implemented metrics aggregator,
planning-ready APIs/CLI, and the first PyPI package tag.

## Active Milestones

| Milestone | Purpose | Exit Signal |
| --- | --- | --- |
| Direction cleanup | remove stale claims and align issues/docs | completed for v0.1; ongoing for v0.2 |
| Real inference slice | terminal command runs one true score path | completed by the v0.1 public terminal transcript |
| Dataset snapshot | reproducible first experiment data | completed by the v0.1 public dataset package |
| First training run | SNV predictor checkpoint | completed by the published `geno-lewm-coherent-cd2bfcc` run |
| Evaluation report | first paper-grade results | completed for the narrow v0.1 chr21 ClinVar slice |
| Paper/demo release | public showcase | completed by public model, dataset, demo, paper, and final binder |
| v0.2 benchmark readiness | stronger held-out science evidence | measured benchmark report with exact variants and negative findings |
| v0.2 rollout/planning readiness | scalable rollout and planning substrate | AR rollout speed gate plus planning demo backed by measured evidence |

## Release Evidence Ledger

Use this ledger as the implementation tracker's source of truth for what
the local release contracts proved in v0.1 and what future releases must
re-run. Do not use local fixture/tooling evidence alone for v0.2 claims.

| Issue | Local contract | v0.1 status and future boundary |
| --- | --- | --- |
| #163 dataset snapshot | `python -m tools.release.dataset_snapshot`; `python -m tools.release.dataset_package` | Completed for v0.1; v0.2 needs broader benchmark snapshots and refreshed split evidence |
| #164 first Carbon-backed run | `geno-lewm-train --carbon-preflight`; `geno-lewm-train --carbon-train --package-release-run` | Completed for v0.1; v0.2 training should wait for stronger data/eval gates |
| #165 results report | `geno-lewm-eval`; `geno-lewm-carbon-baseline`; `geno-lewm-eval-all --require-v02-vep-metrics --require-v02-rollout-metrics`; `geno-lewm-rollout`; `python -m tools.release.rollout_state_examples`; `python -m tools.release.rollout_state_rows`; `python -m tools.release.v02_benchmark_suite`; `python -m bench.inference --release-efficiency` | Completed for the narrow v0.1 release; broader benchmark and real held-out latent rollout specs/states remain open |
| #197 v0.2 benchmark readiness | `python -m tools.release.v02_benchmark_readiness --metrics-json ... --rollout-speed-report ... --rollout-speed-scope-report ... --efficiency-report ... --output ... --require-ok` | New coverage/provenance contract; `--require-ok` records measured values/deltas, measured efficiency latency/throughput/memory, efficiency command provenance, row-derived readiness/blocker entries with issue refs, and metric conclusions with split/track, confidence-interval, evaluated variant-key, missing-metric, baseline-gap, failed-target, and release-input context, and also requires CI-bearing VEP metrics, rollout generation reports, and non-fixture, package-relative release inputs; readiness input identities and readiness, efficiency, or nested rollout-speed command path arguments use public-safe paths plus SHA-256 and size where applicable, while the `release_inputs` row records checked metrics artifact paths and efficiency input identities; expected `ok=false` until broader benchmark rows pass and #42 rollout speed either passes from measured artifacts or is explicitly re-scoped with `tools.release.rollout_speed_scope` |
| #20 release packaging | `python -m build`; `twine check`; `python -m tools.release.check_sdist_assets dist/*.tar.gz` over the full first-publication toolchain | Tagged package release built by the protected workflow from the checked tree |
| #166 terminal showcase | `python tools/demo/terminal_inference.py`; `python -m tools.release.clean_machine_demo` | Completed for v0.1; future demos should demonstrate stronger benchmark/planning behavior without clinical claims |
| #167/#101 paper and publication | `python -m tools.release.paper_draft`; `python -m tools.release.paper_package`; `python -m tools.release.hub_release`; `python -m tools.release.hub_publish`; `python -m tools.release.publication_report` | Completed for v0.1 with public model, dataset, demo, paper, and final binder links |

## Remaining High-Priority Gaps

| Gap | Existing Issue(s) | Notes |
| --- | --- | --- |
| Carbon encoder and cache scale | #32, #36 | v0.1 proved the released artifact path; v0.2 still needs broader Carbon validation and cache-build throughput evidence |
| Trainer reproducibility and regression gates | #44, #47 | v0.1 run evidence exists; v0.2 needs deterministic repeatability and benchmark gates beyond the first run |
| Dataset builders and split enforcement | #49, #50, #51, #52 | v0.1 dataset publication exists; audit remaining issue deltas against the actual pipeline, then narrow v0.2 work around larger shards, holdouts, and warm-cache throughput |
| ClinVar, baseline, and rollout evaluation | #53, #55, #56, #57 | v0.1 measured release evidence exists; v0.2 needs broader coding/non-coding benchmarks, Carbon baseline deltas, real rollout state-row artifacts, and exact evaluated variant identities |
| Score CLI and terminal demo | #62, #65 | v0.1 clean-machine scoring transcript exists; close or re-scope remaining work to reusable examples, quickstart polish, and future benchmark/planning demos |
| Model checkpoint Hub release | #101 | v0.1 model release is complete; future work is PyPI/source-package publication and v0.2 model package evidence |
| Hosted ML smoke gate | #89 | Dedicated `tests/ml` fixture smoke coverage and CI `ml-smoke` job exist; this remains separate from #54's hosted eval smoke-regression gate |
| Hosted eval smoke gate | #54 | Dedicated `tools.ci.eval_smoke_gate`, `tests/eval`, and CI `eval-smoke` job exist; this remains separate from real ClinVar/rollout benchmark execution |
| Paper-grade docs and tutorials | #94, #95, #96, #97, #98 | Should wait for real artifacts where possible |
| Public provenance API naming | #162 | `geno_lewm.provenance` is now the active namespace; the legacy import package and receipt JSON field have been removed |

## De-Scoped Work

The active roadmap no longer includes runtime assurance mechanisms
beyond checksum provenance. The package accepts only checksum receipts today.
Closed historical issues that referenced the previous direction should
stay closed and should not block paper/demo work.

## Paper-Ready Definition

The first paper/demo release is not ready until:

- dataset snapshot and preprocessing scripts are public;
- dataset package artifacts are generated from a checked snapshot spec
  validated by
  `python -m tools.release.dataset_snapshot --spec-json configs/first_experiment/dataset-snapshot-snv.json --check-spec`,
  staged-input identities checked with the same spec and
  `--check-inputs`, then built from staged local upstream files with the same spec and
  `--dataset-dir ... --overwrite`, including
  normalized `dataset_package.json`, `dataset_manifest.json`,
  `data_card.md`, `split_integrity.json`,
  `dataset_input_check_report.json`,
  `dataset_snapshot_report.json`, and `SHA256SUMS`; the snapshot report
  records the checked spec hash and upstream source file identities
  without private absolute input paths and binds input-check evidence plus generated metadata,
  manifest, data-card, split-integrity, and nested package-file
  artifacts by path/hash/size; the release verifier requires
  that report in `SHA256SUMS`, rejects stale report file identities,
  rejects duplicate snapshot file entries, rejects stale generated package identities, and
  rejects stale card or manifest output that no longer matches
  `dataset_package.json`; generated dataset package metadata must carry
  `generated_by=tools.release.dataset_package`; checksum inventories
  must reject invalid digests and duplicate paths; `split_integrity.json`
  also records observed label/class balance plus the
  `tools.release.dataset_integrity` source header and fails when no
  train/eval comparable-key comparison can be made; `data_card.md`
  renders the same class-balance summary;
- train/eval configs are committed under `configs/first_experiment/`,
  and the Carbon preflight records the effective training config hash
  plus resolved closed-schema config payload and CUDA/VRAM accelerator
  readiness;
- real training inputs are preflighted with
  `geno-lewm-train --carbon-preflight`;
  preflight requires the generated dataset package evidence set,
  including `dataset_package.json`, `dataset_input_check_report.json`,
  `dataset_snapshot_report.json`, and `SHA256SUMS`, requires
  `runtime.device: cuda` for the first-experiment config, checks the
  default 40 GiB CUDA memory threshold, and rejects stale input-check
  evidence before launch;
- Carbon-encoded minibatches can be trained through
  `geno_lewm.training.TorchTrainer` with AdamW parameter groups, WSD LR
  scheduling, gradient clipping, and distinct data/predictor/LoRA seed
  records; the real launcher places the Carbon encoder, predictor,
  action encoder, and encoded minibatches on the configured device;
- completed training run evidence is generated with
  `python -m tools.release.training_run` or
  `geno-lewm-train --carbon-train --package-release-run`, including
  checksum-covered `training_preflight_report.json` for release
  Carbon-backed runs and `generated_by=tools.release.training_run`;
  release-mode verification requires the preflight report's dataset
  core-file evidence for `dataset_package.json`,
  `dataset_input_check_report.json`, `dataset_snapshot_report.json`, and
  `SHA256SUMS`;
  the final package verifier rejects stale `training_run_card.md`
  content that no longer matches `training_run_manifest.json`;
- checkpoint and model card are published;
- checkpoint package artifacts are generated with
  `python -m tools.release.model_package`, including normalized
  `model_package.json`, rendered `model_card.md`, packaged
  `eval_metrics.json`, `efficiency_report.json`, and model-local eval
  artifact references from the metrics payload in the checksum set;
  generated `model_package.json` must carry
  `generated_by=tools.release.model_package`, model metadata must include
  the training preflight report, training run manifest/card, and
  training-run checksums as `extra_files`, and the package verifier
  rejects stale model cards that do not re-render from
  `model_package.json` plus `manifest.json`, rejects invalid or
  duplicate checksum paths, rejects training-run dataset/config/commit
  evidence that does not match the manifest plus eval/efficiency
  evidence, and rejects mixed eval/efficiency release id, dataset
  snapshot, commit, or model-result identity across artifacts;
- evaluation metrics JSON and confidence intervals are generated with
  `geno-lewm-eval`, including matched baseline score artifacts when a
  measured baseline is reported, and accepted metrics payloads carry
  `generated_by=geno-lewm-eval` or `generated_by=geno-lewm-eval-all`;
  `geno-lewm-eval` records its report artifact table as package-relative
  paths under `--artifact-root`, defaulting to the metrics output
  directory, writes `eval_config.effective.yaml` beside `eval_metrics.json`,
  and rejects absolute paths outside that root;
- Carbon zero-shot baseline scores are generated with
  `geno-lewm-carbon-baseline --vcf ... --fasta ... --carbon-model-dir ... --output-scores ...`
  and consumed by `geno-lewm-eval` with
  `--baseline-score-field carbon_zero_shot_score`; optional
  log-likelihood cache rows are scoped to the Carbon model and revision
  before reuse;
  `geno-lewm-eval` requires primary score rows from `geno-lewm-score`
  and Carbon baseline rows from `geno-lewm-carbon-baseline`;
- evaluation report is generated from packaged measured metrics JSON
  with
  `geno-lewm-eval-all --require-v02-vep-metrics --require-v02-rollout-metrics`,
  which refreshes and records `eval_config.effective.yaml` plus metrics
  inputs as package-relative artifact paths under the aggregate metrics
  directory and fails incomplete v0.2 VEP or rollout-fidelity coverage;
  `geno-lewm-rollout` can add rollout-fidelity metric rows from measured
  latent-state JSONL with per-K stratification; the eval-report parser
  rejects metrics payloads missing the required `eval_config` artifact;
  baseline comparisons must supply `baseline`, `baseline_value`, and
  `delta_vs_baseline` together; conclusions must explicitly reference
  every measured metric name, split, measured value, and baseline delta
  when present from `eval_metrics.json`;
- `python -m tools.release.v02_benchmark_suite --manifest ... --output-report ...`
  can plan or execute the v0.2 benchmark commands from a JSON manifest;
  plan-only reports keep `ok=false`, while execute-mode clears each
  step's declared output files, then requires the command to exit
  successfully and write those outputs again, with measured claims still
  deferred to generated artifact validators; the suite report records
  the manifest by package-local path plus SHA-256 and size identity;
- `python -m tools.release.rollout_state_examples --spec-jsonl ... --cache-dir ...`
  resolves explicit cache keys for measured source, target, and
  candidate latent states into examples JSONL;
- `python -m tools.release.rollout_state_rows --examples-jsonl ... --model-dir ...`
  generates rollout-state JSONL from those measured latent examples and
  the manifest-backed action encoder/predictor. These helpers bridge
  precomputed source/target/candidate states to `geno-lewm-rollout`; they
  are not Carbon encoder runs or held-out haplotype generators. Release
  readiness requires rollout metrics to carry both generation reports as
  package-relative artifacts;
- `python -m tools.release.rollout_speed_scope --rollout-speed-report ... --output ...`
  records an accepted #42/#197 re-scope for failed RFC-0004 AR rollout
  speed targets. Readiness consumes it only when the report binds the
  exact failing `bench.rollout` path/SHA-256/size identity, failed K
  targets, valid GitHub issue refs including #42 and #197, UTC generated
  and accepted timestamps, HTTP(S) decision URL, and public-safe scope
  plus nested rollout command path identities. Scope negative findings
  and claim boundaries must preserve that the failed target is not
  passing rollout-speed evidence. The readiness row remains `rescoped`
  rather than passing speed evidence, and `scope_decisions` preserves the
  accepted report identity, accepter, rationale, replacement target,
  timestamps, decision URL, and issue refs. Metric conclusions must also
  include failed-target details and accepted decision context for
  re-scoped rows;
- eval-report `negative_findings` must be non-empty and render as
  `## Negative Findings`;
  baseline delta rows must carry matching evaluated variant-key hashes;
  `tools.release.paper_package` resolves eval artifact paths inside the
  package and validates score JSONL `generated_by` markers;
- efficiency evidence is generated with
  `python -m bench.inference --release-efficiency` and records measured
  single-variant latency, batched throughput, peak memory, command,
  hardware/runtime notes, package-relative or inline input identities,
  samples, warm-up, limitations, and the
  `tools.release.efficiency_report` source header;
- terminal demo runs real model inference from released artifacts;
- demo transcript is generated by `tools/demo/terminal_inference.py`
  from the actual `geno-lewm-score` command, including generated time,
  exit code, model release/version/id, artifact-input paths, and an
  explicit claim-boundary sentence;
- demo command, model/input identities, VCF input summary, transcript
  hash, score/receipt hashes, JSONL field names, generated report hashes, and compact
  score/receipt batch metadata are summarized by generated
  `terminal_demo_manifest.json`;
- demo runtime readiness is summarized by generated
  `runtime_preflight_report.json`, which must require native runtime
  dependencies and must record fixture/test manifest allowance as false;
- demo score and receipt JSONL streams are summarized by generated
  `batch_receipt_report.json`, including checked score fields, model id,
  calibration hash, runtime identity, receipt stream, and record count;
  the demo runner clears owned score, receipt, batch-report, and
  demo-manifest outputs before invoking the score command so stale JSONL
  rows cannot satisfy a later run;
  the demo runner re-opens `runtime_preflight_report.json` before
  writing `terminal_demo_manifest.json` and rejects stale or mutated
  model, input, command, backend, runtime-requirement, or model-artifact
  evidence from a different run;
  the package verifier rejects stale transcript claim-boundary or
  artifact-input markers and stale terminal-demo manifest
  `runtime_preflight` summaries;
  the package verifier rejects stale manifest JSONL field lists, stale
  `score_receipt_batch` summaries, and score/receipt batches whose
  model id or calibration hash do not match the packaged model manifest;
- first experiment paper draft is generated with
  `python -m tools.release.paper_draft`, rejecting stale
  `eval_report.md` output that no longer matches `eval_metrics.json`,
  rejecting stale terminal-demo VCF summaries,
  requiring a UTC `Generated: ...Z` timestamp,
  rendering the scored-input summary in Demo Evidence,
  including generated Citation Metadata,
  including Negative Findings copied from the generated eval report,
  and naming
  `model_package.json`, `dataset_package.json`,
  `dataset_input_check_report.json`, `dataset_snapshot_report.json`, `eval_metrics.json`,
  `eval_config.effective.yaml`, `eval_report.md`,
  `efficiency_report.json`, and demo evidence paths using
  package-local artifact names rather than build-machine root paths;
  the package verifier rejects paper drafts or drafts missing Citation
  Metadata or Negative Findings that no longer match the current
  artifact set;
- `python -m tools.release.paper_package` passes for the model,
  dataset, demo, and paper artifacts;
- `python -m tools.release.hub_release` emits a versioned dry-run Hub
  upload plan for the verified release candidate, requiring a public
  paper URL when a paper artifact is included, and records model upload
  inventories from both `SHA256SUMS` and `training_run_SHA256SUMS`,
  dataset upload inventories including `SHA256SUMS`, and portable
  terminal-demo upload inventories with unique GitHub release asset
  names. When a paper URL is present, it also binds the verified paper
  file path/hash/size before emitting publication commands for recognized Hub/GitHub
  targets; Hugging Face commands upload each
  verified model/dataset file to its planned destination instead of
  syncing whole package directories;
- `.github/workflows/release-hub-dry-run.yml` runs the package verifier,
  Hub dry-run planner, and release-candidate report without publishing
  weights or requiring Hub credentials;
- `python -m tools.release.hub_publish` and
  `.github/workflows/release-hub-publish.yml` publish the verified
  model, dataset, terminal-demo, and matching paper artifacts after a
  clean dry-run, requiring `HF_TOKEN`, GitHub release credentials,
  supported Hugging Face/GitHub target URLs, direct GitHub release
  download paper URLs whose final asset name matches the verified paper
  file, and protected `release` environment approval, with the workflow
  syncing the locked `dev`, `train`, `eval`, and `deploy` extras for the
  native clean-machine replay, then uploading only files named by the
  verified Hub plan before regenerating the final release-candidate
  report from the public links and fetched public artifact bytes; the
  protected workflow
  runs the clean-machine terminal replay from that report with native
  runtime checks enabled before running
  `python -m tools.release.publication_assets` to bind the GitHub
  release target, upload command, and publication-evidence asset
  identities, then uploading those assets to the demo release tag, using
  release credentials only for scoped artifact fetches;
- `python -m tools.release.release_candidate` emits
  `release_candidate_report.json` with `ready=true` for the same model,
  dataset, demo, paper, public URL reachability checks, commit SHA, Hub
  repo id, model package metadata, dataset package metadata, dataset
  snapshot report, source metrics JSON, effective eval config,
  generated eval report, efficiency report,
  manifest-backed predictor/action/calibration and training-config
  artifacts, `training_preflight_report.json`,
  `training_run_SHA256SUMS`, and Hub model/dataset/demo upload
  inventories, provider-backed public artifact exact file-set, hash, and
  size checks, direct paper byte hash/size checks, plus a
  `readiness` checklist that records which publication requirements are
  satisfied or blocked; public checks can be skipped only for explicit
  fixture rehearsals that allow fixture manifests, otherwise the report
  remains `ready=false`;
- `python -m tools.release.clean_machine_demo` consumes the generated
  ready release-candidate report, rejects hand-authored reports and
  stale embedded Hub plans by source header and model/repo/URL identity,
  rejects missing or failed readiness rows, non-empty candidate blockers,
  skipped or failed public link checks, skipped, missing, incomplete, or
  failed public artifact checks, unsafe embedded Hub-plan destinations,
  or malformed expected hashes, downloads the published model files,
  dataset snapshot files,
  and GitHub release demo assets, verifies their SHA-256 values against the
  Hub upload plan, re-runs the release-package verifier on the
  downloaded model/dataset/demo package, reruns the terminal demo from
  those public bytes, and rejects replayed terminal demo manifests with
  invalid source headers, non-passing status, model id mismatch,
  downloaded `model/manifest.json` hash/size mismatch, stale
  VCF/FASTA input identities, stale `runtime_preflight` summaries,
  stale `score_receipt_batch` summaries, or replay artifact hash/size
  drift before writing the clean-machine report. The final publication binder also checks the replay
  manifest's VCF/FASTA input identities against the downloaded demo
  artifacts and checks the replay manifest's artifact table against the
  clean-machine replay report for the transcript, scores, receipts,
  runtime preflight, and batch report.
  Before scoring, the replay helper checks the downloaded demo
  VCF/FASTA hashes and sizes against the downloaded demo manifest; after
  scoring, it rejects replay manifests whose VCF/FASTA identities do not
  match those downloaded inputs. The replay helper writes
  `clean_machine_demo_report.json` with
  the release-candidate report filename plus hash/size identity,
  output-directory-relative downloaded artifact identities,
  package-verification result, replay transcript and manifest
  identities, and replay score, receipt, runtime-preflight, and
  batch-report artifact hashes, without serializing fetch tokens or
  private absolute workstation paths;
- `python -m tools.release.publication_report` runs after credentialed
  Hub publication and clean-machine replay, writes
  `publication_evidence_report.json`, and binds the Hub release plan,
  release-candidate report, publish report, and clean-machine replay
  report by public-safe filename plus hash/size identity, including the
  clean-machine replay's recorded release-candidate report
  filename/path, hash, and size identity plus the verified paper file
  source name, URL, hash, and size identity plus the full
  paper-critical release-candidate artifact table for model, dataset,
  eval, demo, and paper identities, public-safe release-candidate
  readiness rows plus public link and public artifact check summaries,
  with every uploaded release-candidate artifact identity in that table
  checked against the Hub plan and downloaded public artifact, plus
  the replayed terminal-demo manifest's model id, downloaded
  `manifest.json` identity, VCF/FASTA input identities,
  `runtime_preflight` summary, and replayed runtime-preflight
  model/input identities without
  private absolute paths, while failing on a candidate embedded-plan
  mismatch, a missing generated readiness
  checklist, non-empty candidate blockers, stale readiness `issue_refs`,
  missing or failed candidate `public_links` or `public_artifacts`
  checks, exact download-set, public source URL, hash, or
  replay-artifact mismatches; the
  protected workflow uploads the Hub plan, release-candidate report,
  publish report, clean-machine replay report, final publication
  evidence report, publication evidence asset manifest, replay
  transcript, replay manifest, score/receipt JSONL streams, runtime
  preflight report, and batch receipt report as public GitHub release
  assets;
- receipt semantics cover the published demo mode without implying
  unsupported trust guarantees;
- README and docs show measured values only where they are measured;
- privacy and safety docs match the demo behavior.

## GitHub Issue State

- #15 now tracks artifact provenance and checksum receipts, not external
  runtime assurance mechanisms beyond checksum provenance.
- #62 and #65 have local implementation coverage plus v0.1
  clean-machine artifact-backed scoring evidence. Their stale
  `status:blocked` labels should be removed; close them if no
  non-v0.1 acceptance remains, or re-scope any quickstart/tutorial
  follow-up under #197.
- #101 tracked the v0.1 model Hub release with `model_card.md`,
  `eval_metrics.json`, `eval_config.effective.yaml`, `eval_report.md`,
  `efficiency_report.json`,
  `manifest.json`, training config, checksum files, and links to the
  dataset snapshot and terminal demo transcript. That release is now
  public and closed through PR #196; future model packages should preserve
  the same `python -m tools.release.paper_package` contract.
- #162 now has a local `geno_lewm.provenance` public namespace. The
  legacy import package has been removed from the active public surface,
  and receipt JSON now serializes the field as `provenance`.
- #49 and #50 now have local release-file prep commands for gnomAD and
  ClinVar, and the v0.1 dataset publication used public dataset
  artifacts. Audit the remaining issue bodies against that pipeline and
  keep only narrower v0.2 shard/data-quality deltas.
- #51 now has a local tuple-builder contract plus
  `geno_lewm.data.GenoLeWMDataset`, which deterministically streams
  source windows and training tuples without importing torch in core
  environments. `geno_lewm.training.encode_training_batch` now looks up
  untargeted source `s_t` states in the documented cache index when
  present and falls back to live encoding on misses. Remaining work
  should focus on v0.2 prepared-shard scale, holdout membership deltas,
  and measured warm-cache throughput validation.
- #44 and #47 now have deterministic fixture smoke coverage through
  `geno-lewm-train --fixture-smoke` plus a torch trainer core for
  Carbon-encoded minibatches, AdamW groups, WSD scheduling, gradient
  clipping, distinct seed records, and a preflight-gated
  `geno-lewm-train --carbon-train` launcher. v0.1 supplied a real
  Carbon-backed run; remaining work is deterministic repeatability on
  supported backends and benchmark gates beyond the first run.
- #163 through #167 tracked the v0.1 paper/demo chain: dataset snapshot,
  first training run, generated evaluation report, terminal showcase, and
  paper package. They are closed with public evidence through PR #196.
  Remaining data/eval/demo work should be routed through #197 or narrower
  v0.2 issues, not reopened as missing v0.1 publication evidence.

## Validation Expectations

For project-direction changes:

- `rg` finds no active docs promising unsupported runtime assurance beyond checksum provenance;
- receipt tests confirm unsupported runtime assurance modes are rejected;
- score CLI/runtime tests cover single-variant receipt emission and
  per-row VCF receipt JSONL sidecars;
- public module-map docs match the current package layout and do not
  list removed or absent paths such as `eval/`, `holdouts.py`,
  `deploy/provenance.py`, or export modules that have not landed;
- public API docs/RFC-0014 point to `tests/api/public_surface.json` as
  the exhaustive enforced symbol list, and upcoming planning solver
  types are not described as stable top-level exports;
- CLI scaffold factory helpers remain private to the shared stub factory
  and do not leak into command-module public surfaces;
- README and roadmap state current implementation gaps honestly;
- docs build and focused tests pass.
