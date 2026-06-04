# Implementation Tracker

Last updated: 2026-06-02

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

Remaining blockers are concentrated in the ML path: validating the
native runtime loader against actual Carbon/predictor artifacts, proving
RFC-0004 attention KV-cache speedups, dataset snapshots, Carbon-backed
trainer/evaluator integration, model release, and a real terminal
inference demo.

## Active Milestones

| Milestone | Purpose | Exit Signal |
| --- | --- | --- |
| Direction cleanup | remove stale claims and align issues/docs | README, roadmap, context, and issues match current scope |
| Real inference slice | terminal command runs one true score path | demo transcript generated from real command output |
| Dataset snapshot | reproducible first experiment data | dataset card, manifest, split checks, rebuild command |
| First training run | SNV predictor checkpoint | checkpoint, config, logs, model package metadata, model card, manifest |
| Evaluation report | first paper-grade results | generated report with baselines and conclusions |
| Paper/demo release | public showcase | release links dataset, model, report, demo transcript |

## Release Evidence Ledger

Use this ledger as the implementation tracker's source of truth for what
is locally contracted versus still missing for the first paper/demo
release. Do not close #163 through #167 or #101 from local
fixture/tooling evidence alone.

| Issue | Local contract | Remaining release evidence |
| --- | --- | --- |
| #163 dataset snapshot | `python -m tools.release.dataset_snapshot`; `python -m tools.release.dataset_package` | Actual pinned Carbon, gnomAD, and ClinVar inputs, public snapshot package, and data-card links |
| #164 first Carbon-backed run | `geno-lewm-train --carbon-preflight`; `geno-lewm-train --carbon-train --package-release-run` | Clean supported-environment run on public dataset artifacts with logs, metrics, and checkpoint metadata |
| #165 results report | `geno-lewm-eval`; `geno-lewm-carbon-baseline`; `geno-lewm-eval-all`; `python -m bench.inference --release-efficiency` | Real score, label, baseline, benchmark, and conclusion artifacts from the first experiment |
| #20 release packaging | `python -m build`; `twine check`; `python -m tools.release.check_sdist_assets dist/*.tar.gz` over the full first-publication toolchain | Tagged package release built by the protected workflow from the checked tree |
| #166 terminal showcase | `python tools/demo/terminal_inference.py`; `python -m tools.release.clean_machine_demo` | Clean-machine replay from public model, dataset, and demo artifacts with transcript and manifest hashes |
| #167/#101 paper and publication | `python -m tools.release.paper_draft`; `python -m tools.release.paper_package`; `python -m tools.release.hub_release`; `python -m tools.release.hub_publish`; `python -m tools.release.publication_report` | Public model, dataset, demo, paper, Hub, and protected publish workflow evidence links |

## Remaining High-Priority Gaps

| Gap | Existing Issue(s) | Notes |
| --- | --- | --- |
| Carbon encoder on a clean machine | #32, #36 | Needed before real inference or training can be credible |
| Trainer scaffold and deterministic run config | #44, #47 | Fixture smoke trainer emits reproducible logs/checkpoints; Carbon-backed launcher and resume validation exist; clean supported-runtime Carbon run and bit-exact deterministic evidence remain |
| Dataset builders and split enforcement | #49, #50, #51, #52 | Local VCF-to-Parquet prep, tuple-builder contracts, source-state cache lookup, and split-integrity evidence exist; real upstream release runs, published shards, holdout data, and warm-cache throughput validation remain |
| ClinVar and baseline evaluation | #53, #55, #56 | Artifact-level ClinVar-style score/label metrics with deterministic bootstrap CIs and matched measured-baseline deltas exist via `geno-lewm-eval`; `geno-lewm-carbon-baseline` writes Carbon zero-shot baseline score JSONL for `--baseline-score-field carbon_zero_shot_score`; `geno-lewm-eval-all` aggregates validated metric artifacts into `eval_report.md`; `bench.inference --release-efficiency` generates the validated latency/throughput/memory artifact; hosted generated-fixture eval smoke regression is enforced by `tools.ci.eval_smoke_gate`; full benchmark runner and real Carbon baseline run remain |
| Score CLI and terminal demo | #62, #65 | Local scorer/CLI path can auto-load native artifacts, emit per-row VCF receipt JSONL, and generate runtime preflight evidence; still needs real artifacts and Carbon validation |
| Model checkpoint Hub release | #101 | Requires model package metadata, model card, eval report, manifest, training config, checksum files, and demo links |
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
  with `geno-lewm-eval-all`, which refreshes and records
  `eval_config.effective.yaml` plus metrics inputs as package-relative
  artifact paths under the aggregate metrics directory; the eval-report
  parser rejects metrics payloads missing the required `eval_config`
  artifact; baseline comparisons must supply `baseline`,
  `baseline_value`, and `delta_vs_baseline` together; conclusions must
  explicitly reference every measured metric name, split, measured value,
  and baseline delta when present from `eval_metrics.json`;
  `negative_findings` must be non-empty and render as
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
- #62 and #65 have local implementation coverage for surprise scoring,
  FASTA-backed VCF scoring, native component loading, and receipt
  emission. They should remain open until clean-machine artifact-backed
  integration proves the public command path.
- #101 tracks model Hub release with `model_card.md`,
  `eval_metrics.json`, `eval_config.effective.yaml`, `eval_report.md`,
  `efficiency_report.json`,
  `manifest.json`, training config, checksum files, and links to the
  dataset snapshot and terminal demo transcript. The final package should pass
  `python -m tools.release.paper_package`. #101 now has a local
  model-package generator contract, Hub dry-run planner, and credentialed
  Hub publish workflow; the actual trained checkpoint and first
  credentialed upload run remain open.
- #162 now has a local `geno_lewm.provenance` public namespace. The
  legacy import package has been removed from the active public surface,
  and receipt JSON now serializes the field as `provenance`.
- #49 and #50 now have local release-file prep commands for gnomAD and
  ClinVar; they remain open until real upstream release files are
  processed, sized, and published with the dataset snapshot.
- #51 now has a local tuple-builder contract plus
  `geno_lewm.data.GenoLeWMDataset`, which deterministically streams
  source windows and training tuples without importing torch in core
  environments. `geno_lewm.training.encode_training_batch` now looks up
  untargeted source `s_t` states in the documented cache index when
  present and falls back to live encoding on misses. It remains open
  until prepared real shards, holdout membership inputs, and measured
  warm-cache throughput validation land.
- #44 and #47 now have deterministic fixture smoke coverage through
  `geno-lewm-train --fixture-smoke` plus a torch trainer core for
  Carbon-encoded minibatches, AdamW groups, WSD scheduling, gradient
  clipping, distinct seed records, and a preflight-gated
  `geno-lewm-train --carbon-train` launcher. They remain open until a
  clean-machine Carbon-backed run emits real checkpoints/logs/metrics,
  deterministic torch runs are confirmed on a supported backend, and
  benchmark gates land.
- #163 through #167 track the paper/demo chain: dataset snapshot, first
  training run, generated evaluation report, terminal showcase, and
  paper package. #163 now has a local dataset-package generator contract;
  #51 has a local tuple-builder contract; #166 now has a generated
  terminal demo manifest contract with package-local demo input checks,
  runtime-preflight command parity, and canonical command/artifact path
  checks plus a clean-machine replay helper for published
  model/dataset/demo artifacts; the actual
  upstream snapshot, real split inputs, released checkpoint,
  clean-machine transcript from public bytes, and published edit shards
  remain open.

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
