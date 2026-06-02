# GenoLeWM

**Action-conditioned JEPA world models for genomic edits, built on top
of Carbon.**

[![CI](https://github.com/AbdelStark/GenoLeWM/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/AbdelStark/GenoLeWM/actions/workflows/ci.yml)
[![CodeQL](https://github.com/AbdelStark/GenoLeWM/actions/workflows/codeql.yml/badge.svg?branch=main)](https://github.com/AbdelStark/GenoLeWM/actions/workflows/codeql.yml)
[![Docs](https://github.com/AbdelStark/GenoLeWM/actions/workflows/docs.yml/badge.svg?branch=main)](https://abdelstark.github.io/GenoLeWM/)
[![Status](https://img.shields.io/badge/status-alpha%20pre--release-orange.svg)](ROADMAP.md)
[![Python](https://img.shields.io/badge/python-3.10%2B-blue.svg)](pyproject.toml)
[![License](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](LICENSE)
[![Typed: mypy --strict](https://img.shields.io/badge/typed-mypy--strict-blue.svg)](https://mypy.readthedocs.io/)
[![Linted: ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)

[Documentation](https://abdelstark.github.io/GenoLeWM/) |
[Specification](SPEC.md) |
[Roadmap](ROADMAP.md) |
[Architecture](ARCHITECTURE.md) |
[Privacy](PRIVACY.md)

---

## Status

GenoLeWM is an alpha research codebase. The core infrastructure is in
place and tested; the first real training run, published checkpoint,
public dataset snapshot, and terminal inference demo are the next
project milestones.

As of June 2, 2026:

| Area | Current state |
| --- | --- |
| Edit/action representation | Implemented: `EditSpec`, `RelEdit`, edit application, synthetic edit samplers, and optional-runtime `ActionEncoder` |
| Privacy-safe infrastructure | Implemented: typed errors, structured logging, redaction, metrics |
| Artifact provenance | Implemented: content-addressed manifests, input/output commitments, checksum receipt verification |
| CLI surface | Implemented scaffolds plus working `geno-lewm-verify`, `geno-lewm-update`, data prep, score, and fixture train paths |
| Desktop/runtime scaffolds | Present but not a complete product |
| Carbon encoder integration | Lazy `CarbonStateEncoder` wrapper is implemented; clean-machine checkpoint-backed inference remains a gap |
| Data/training stream | Carbon window sampler, tuple-builder contract, `GenoLeWMDataset` iterator, source-state cache lookup in the trainer batch encoder, and local gnomAD/ClinVar VCF-to-Parquet prep commands exist; real shard publication and warm-cache throughput validation remain gaps |
| Predictor/training | Base cross-attention `Predictor`, `ARPredictor` rollout wrapper, losses, collapse checks, deterministic fixture smoke training, torch trainer core, WSD scheduling, optimizer grouping, `geno-lewm-train --carbon-preflight`, and preflight-gated `geno-lewm-train --carbon-train` launch plumbing exist; true attention KV-cache speedups and the first Carbon-backed experiment remain open |
| Evaluation | `geno-lewm-carbon-baseline` writes Carbon zero-shot score JSONL from a local Carbon LM and held-out VCF/FASTA; `geno-lewm-eval` computes measured ClinVar-style binary metrics, deterministic bootstrap CIs, optional measured-baseline deltas from matched score/label JSONL artifacts with identical evaluated variant-key hashes, and an effective eval config artifact; `geno-lewm-eval-all` aggregates validated metric JSON into source `eval_metrics.json` plus generated `eval_report.md`; `bench.inference --release-efficiency` writes validated latency, throughput, memory, hardware/runtime, and input identity evidence; first full benchmark table is not published |
| Package/model release | PyPI release workflow and package metadata exist; first PyPI tag and model checkpoint release to the Hub remain open |

No GenoLeWM model checkpoint is released yet. Results in this repository
are fixtures or design targets unless explicitly marked as measured.

---

## Reader Map

| If you want to... | Start here |
| --- | --- |
| Understand what is implemented today | [Status](#status) and [What You Can Run Today](#what-you-can-run-today) |
| Try the stable Python surface | [Install](#install) and [Quickstart](#quickstart) |
| Audit the first-paper plan | [First Experiment Target](#first-experiment-target) and [Paper-Ready Checklist](#paper-ready-checklist) |
| Contribute code | [Repository Layout](#repository-layout), [Development](#development), and [Contributing](#contributing) |
| Check safety and data-handling boundaries | [Safety](#safety), [PRIVACY.md](PRIVACY.md), and [SECURITY.md](SECURITY.md) |

---

## Why This Exists

Current DNA foundation models usually score a variant by comparing two
full sequence likelihoods: one for the reference allele and one for the
alternate allele. GenoLeWM instead makes the edit itself an action in a
latent world model:

```text
s_t = enc(window_ref)
a_t = action(edit)
s_hat_{t+1} = g(s_t, a_t)
loss = distance(s_hat_{t+1}, enc(window_alt)) + representation regularization
```

The goal is to learn a small action-conditioned predictor on top of a
frozen DNA encoder. If this works, the same model can support:

- single-variant effect scoring;
- multi-edit latent rollout;
- planning over edit sequences;
- surprise scores based on prediction residuals;
- local-first inference on personal genome files.

The project deliberately optimizes for a publishable, reproducible ML
system: explicit data snapshots, model cards, evaluation reports,
calibration artifacts, and terminal demos are first-class deliverables.

---

## Architecture

```text
reference window
    |
    v
Carbon encoder (frozen) -------------------> state s_t
                                                |
genomic edit -> action encoder -> action a_t    |
                                                v
                                      predictor g(s_t, a_t)
                                                |
                                                v
                                      predicted next state
                                                |
                                                v
                               surprise / rollout / planning
```

The intended training target is `enc(edited_window)`. Carbon remains the
heavy frozen state encoder; GenoLeWM trains the action encoder and
predictor. The deployed package keeps heavyweight ML dependencies behind
extras so the pure-Python utilities stay lightweight.

Detailed design:

- [ARCHITECTURE.md](ARCHITECTURE.md) - narrative architecture walkthrough
- [docs/spec/01-architecture.md](docs/spec/01-architecture.md) - module boundaries
- [docs/spec/03-data-model.md](docs/spec/03-data-model.md) - dataset and checkpoint layouts
- [ROADMAP.md](ROADMAP.md) - current execution plan

---

## Install

Python 3.10 or newer is required. The first PyPI release has not been
cut yet, so install from source for now:

```bash
git clone https://github.com/AbdelStark/GenoLeWM.git
cd GenoLeWM
uv venv
source .venv/bin/activate
uv pip install -e "."
```

For development extras:

```bash
git clone https://github.com/AbdelStark/GenoLeWM.git
cd GenoLeWM
uv venv
source .venv/bin/activate
uv pip install -e ".[dev,docs]"
```

Optional extras:

| Extra | Use |
| --- | --- |
| `geno-lewm[train]` | PyTorch, Transformers, datasets, training utilities |
| `geno-lewm[eval]` | VCF/FASTA parsing and evaluation dependencies |
| `geno-lewm[deploy]` | ONNX export/runtime dependencies |
| `geno-lewm[docs]` | MkDocs documentation build |
| `geno-lewm[dev]` | Tests, linting, typing, packaging checks |
| `geno-lewm[all]` | Train, eval, and deploy extras |

---

## What You Can Run Today

These commands exercise local contracts. They are useful for development
and release hardening, but they do not prove model quality because the
public dataset snapshot, checkpoint, measured evaluation, and terminal
demo release are still open.

| Task | Command | What it proves |
| --- | --- | --- |
| Verify a checksum receipt fixture | `geno-lewm-verify examples/data/verify_receipt/receipt.json --manifest examples/data/verify_receipt/manifest.json` | Receipt schema, manifest identity, and output commitment plumbing work locally |
| Run fixture training smoke | `geno-lewm-train --fixture-smoke --run-dir /tmp/geno-lewm-smoke --steps 50` | Trainer packaging path can emit deterministic fixture artifacts without optional Carbon weights |
| Validate the first-experiment dataset spec | `python -m tools.release.dataset_snapshot --spec-json configs/first_experiment/dataset-snapshot-snv.json --check-spec` | Dataset rebuild metadata, source layout, split coverage, and staged paths are internally consistent without local upstream files |
| Check public API drift | `uv run python tools/api/snapshot.py check` | The exported Python surface matches `tests/api/public_surface.json` |
| Check retired-scope language | `uv run python -m tools.lint.check_scope_language` | Public docs/code do not reintroduce unsupported runtime-assurance claims |
| Build docs strictly | `uv run mkdocs build --strict` | MkDocs renders the public documentation with strict link/page checks |

---

## Quickstart

### Canonical edits

```python
from geno_lewm import EditSpec, EditType, RelEdit, apply_edit, apply_edits

edit = EditSpec(chrom="chr17", pos=43_091_983, ref="A", alt="T")
assert edit.edit_type is EditType.SNV

relative = edit.relative_to(window_start_bp=43_091_900, window_end_bp=43_092_100)
print(relative.rel_pos)

window = "ACGT" * 64
edited = apply_edit(window, RelEdit(0, EditType.SNV, "A", "C"))

haplotype = apply_edits(
    window,
    [
        RelEdit(rel_pos=0, edit_type=EditType.SNV, ref_bases="A", alt_bases="T"),
        RelEdit(rel_pos=4, edit_type=EditType.SNV, ref_bases="A", alt_bases="C"),
    ],
)
```

All validation failures use typed `GenoLeWMError` subclasses with stable
machine-readable codes.

### Privacy-safe logging

```python
from geno_lewm import get_logger

log = get_logger("inference", run_id="run-42")
log.info("inference.batch.end", n=10, batch_id="b-1", throughput_per_s=87.2)
```

The logging layer is deny-list and allow-list based. It rejects long DNA
strings and personal-data fields before events leave the process.

### Checksum provenance

```python
from geno_lewm import DtypeConfig, EditSpec, PoolingConfig, compute_input_commitment

edit = EditSpec(chrom="1", pos=10, ref="A", alt="T")
pool = PoolingConfig(state_layer=12, pool_type="centered_mean", pool_radius=64, normalize=True)
dtype = DtypeConfig(encoder_dtype="bf16", predictor_dtype="bf16")

window = "ACGT" * 64
print(compute_input_commitment(window, edit, pool, dtype))
```

`geno-lewm-verify` checks receipt schema validity, manifest identity,
optional input commitments, and output commitments:

```console
$ geno-lewm-verify examples/data/verify_receipt/receipt.json \
    --manifest examples/data/verify_receipt/manifest.json
reading receipt:  examples/data/verify_receipt/receipt.json
  schema_version=1.0.0 provenance.kind=checksum_only
reading manifest: examples/data/verify_receipt/manifest.json
  model_id ok (sha256:3bcf3c87e5dd99...)
  input_commitment: skipped (no input flags supplied)
  output_commitment ok (sha256:982aee9fc1786...)
ok
```

This is reproducibility and tamper-detection plumbing. It is not a
model-quality or runtime-assurance guarantee.

`geno-lewm-score --variant ... --receipt path/to/receipt.json` writes
one canonical receipt. `geno-lewm-score --vcf ... --receipt
path/to/receipts.jsonl` writes one canonical receipt per scored ALT as a
JSONL sidecar. Both paths require manifest-verified local scorer
components. The runtime can now attempt local native component loading
when `torch`, `transformers`, and `safetensors` are installed; a
clean-machine demo still needs published model artifacts and an actual
Carbon checkpoint validation run.

---

## First Experiment Target

The first paper-ready experiment should be intentionally narrow:

| Component | Target |
| --- | --- |
| Encoder | Frozen Carbon-500M state vectors |
| Edits | SNVs only |
| Data | Versioned Carbon corpus slice plus prepared gnomAD/ClinVar shards and held-out ClinVar coding/non-coding variants |
| Model | Action encoder + predictor head |
| Metrics | rollout cosine similarity, residual distribution, AUROC/AUPRC against ClinVar labels, throughput |
| Release artifacts | dataset package metadata, dataset input check report, dataset card, model package metadata, model card, checkpoint, manifest, source metrics JSON, effective eval config, eval report, efficiency report, terminal demo transcript, terminal demo manifest, runtime preflight report, batch receipt report |

The first conclusions should be honest even if the result is negative:
whether latent action prediction learns anything beyond Carbon
zero-shot scoring, where it fails, what error modes dominate, and which
next experiment is justified.

**Live Release Blockers**

| Gate | Issue | Current blocker |
| --- | --- | --- |
| Dataset snapshot and data card | [#163](https://github.com/AbdelStark/GenoLeWM/issues/163) | Real upstream Carbon, gnomAD, and ClinVar release inputs must be processed, packaged, and published |
| First Carbon-backed run | [#164](https://github.com/AbdelStark/GenoLeWM/issues/164) | Clean-machine training must emit real checkpoints, metrics, logs, and training-run metadata |
| Paper-ready results report | [#165](https://github.com/AbdelStark/GenoLeWM/issues/165) | Measured ClinVar metrics, Carbon baseline deltas, efficiency evidence, conclusions, and negative findings must be generated from real artifacts |
| Terminal real-inference showcase | [#166](https://github.com/AbdelStark/GenoLeWM/issues/166) | The demo must replay from released public model, dataset, and demo artifacts, not fixtures |
| First experiment paper package | [#167](https://github.com/AbdelStark/GenoLeWM/issues/167) | Draft must bind the public dataset, checkpoint, eval, efficiency, terminal demo, artifact availability, conclusions, and negative findings |
| Model checkpoint Hub release | [#101](https://github.com/AbdelStark/GenoLeWM/issues/101) | Hub model card, checkpoint files, manifest, checksums, eval report, and demo links must be published |

### Release Evidence Matrix

Use this table to separate local release contracts from paper-ready
evidence. Green local tooling is necessary, but it is not a substitute
for real artifacts from the first experiment.

| Evidence artifact | Local contract | Paper-release status |
| --- | --- | --- |
| Dataset package | `python -m tools.release.dataset_snapshot --spec-json configs/first_experiment/dataset-snapshot-snv.json --check-spec` validates the checked rebuild spec; `--check-inputs` hashes staged upstream files; the same spec with `--dataset-dir ... --overwrite` writes `dataset_input_check_report.json`, `dataset_snapshot_report.json`, `dataset_package.json`, `dataset_manifest.json`, `data_card.md`, `split_integrity.json`, and `SHA256SUMS` | Blocked on running the command against the actual pinned Carbon, gnomAD, and ClinVar inputs, then publishing the resulting files ([#163](https://github.com/AbdelStark/GenoLeWM/issues/163)) |
| Training run | `geno-lewm-train --carbon-preflight ...` and `geno-lewm-train --carbon-train --package-release-run ...` bind config, dataset, Carbon model, checkpoint, logs, metrics, and `training_run_SHA256SUMS` | Blocked on a completed clean-machine Carbon-backed run over the published dataset snapshot ([#164](https://github.com/AbdelStark/GenoLeWM/issues/164)) |
| Evaluation and efficiency | `geno-lewm-eval`, `geno-lewm-carbon-baseline`, `geno-lewm-eval-all`, and `python -m bench.inference --release-efficiency` generate `eval_metrics.json`, `eval_config.effective.yaml`, `eval_report.md`, and `efficiency_report.json` | Blocked on real GenoLeWM scores, Carbon zero-shot baseline scores, labels, and benchmark outputs from the released checkpoint/dataset pair ([#165](https://github.com/AbdelStark/GenoLeWM/issues/165)) |
| Terminal demo | `python tools/demo/terminal_inference.py ...` records `terminal-demo-transcript.md`, `terminal_demo_manifest.json`, `runtime_preflight_report.json`, `scores.jsonl`, `receipts.jsonl`, and `batch_receipt_report.json` | Blocked on public model, dataset, and demo artifacts plus a clean-machine replay from those artifacts ([#166](https://github.com/AbdelStark/GenoLeWM/issues/166)) |
| Paper and publication evidence | `python -m tools.release.paper_draft`, `python -m tools.release.paper_package`, `python -m tools.release.release_candidate`, `python -m tools.release.clean_machine_demo`, and `python -m tools.release.publication_report` bind the paper, Hub plan, public links, replay, and final evidence report | Blocked on the real artifact set and a protected publish workflow run with reachable public links ([#167](https://github.com/AbdelStark/GenoLeWM/issues/167), [#101](https://github.com/AbdelStark/GenoLeWM/issues/101)) |

---

## Paper-Ready Checklist

The project is not paper-ready until all of these are true:

- Dataset snapshot is reproducible from scripts and pinned revisions,
  starting from a checked snapshot spec and explicit local upstream
  files with
  `python -m tools.release.dataset_snapshot --spec-json configs/first_experiment/dataset-snapshot-snv.json --check-spec`
  for public spec validation, then
  `python -m tools.release.dataset_snapshot --spec-json configs/first_experiment/dataset-snapshot-snv.json --check-inputs`
  to record SHA-256 and byte-size identities for staged upstream inputs,
  then
  `python -m tools.release.dataset_snapshot --spec-json configs/first_experiment/dataset-snapshot-snv.json --dataset-dir ... --overwrite`
  once the upstream files are staged under
  `configs/first_experiment/inputs/`.
  That command stages Carbon source-mix files, builds gnomAD and
  ClinVar Parquet shards from local VCF/VCF.gz inputs, writes
  `dataset_package.json`, runs
  `python -m tools.release.dataset_package --dataset-dir ... --metadata-json ...`,
  and emits `dataset_input_check_report.json`,
  `dataset_snapshot_report.json`, `dataset_manifest.json`,
  `data_card.md`, `split_integrity.json`, and `SHA256SUMS`. The snapshot
  report records the checked spec hash plus upstream source file hashes
  without embedding private absolute input paths, binds the input-check
  report, generated
  dataset package metadata, manifest, data card, and split-integrity
  artifacts by path/hash/size, and keeps the nested package file table
  aligned with the top-level staged file identities,
  is included in `SHA256SUMS`, and is validated by the release verifier. The release
  verifier checks that generated dataset package metadata carries
  `generated_by=tools.release.dataset_package` and that the data card
  and manifest still match `dataset_package.json`; it also rejects
  invalid or duplicate `SHA256SUMS` paths;
  the split-integrity report covers record counts, file identities,
  observed label/class balance, Parquet variant-key extraction,
  train/eval leakage checks, and the
  `tools.release.dataset_integrity` source header; leakage evidence
  fails closed when train/eval comparable keys are missing, and the data
  card renders the same class-balance summary from `split_integrity.json`.
- Training tuples are built through `geno_lewm.data.build_training_tuples`
  or streamed through `geno_lewm.data.GenoLeWMDataset` so source mix,
  ClinVar fallback, and holdout exclusions are enforced before the
  trainer sees a batch.
- The real trainer core uses `geno_lewm.training.encode_training_batch`
  and `geno_lewm.training.TorchTrainer` to turn Carbon-encoded source
  and target windows plus relative edits into predictor steps with
  AdamW parameter groups, WSD learning-rate scheduling, gradient
  clipping, and distinct data/predictor/LoRA seed records. Source
  `s_t` states use the documented window cache when a compatible
  `$GENO_LEWM_CACHE/embeddings/index.sqlite` is present; cache misses
  fall through to live untargeted Carbon encoding, while edited
  `s_{t+1}` targets are still encoded on the fly.
- Train/eval configs are committed and can be run from a clean machine;
  the first-experiment checked configs live under
  `configs/first_experiment/`, and Carbon training preflight validates
  the effective training config against the closed GenoLeWM schema
  before launch;
  fixture smoke training is available via
  `geno-lewm-train --fixture-smoke --run-dir ... --steps 50`;
  real training inputs are preflighted with
  `geno-lewm-train --carbon-preflight --dataset-dir ... --carbon-model-dir ... --training-config ... --run-dir ...`;
  that preflight now requires the packaged dataset release evidence set:
  `dataset_package.json`, `dataset_manifest.json`, `data_card.md`,
  `split_integrity.json`, `dataset_input_check_report.json`,
  `dataset_snapshot_report.json`, and `SHA256SUMS`, and it rejects stale
  input-check evidence before the trainer can launch;
  the single-process launcher is
  `geno-lewm-train --carbon-train --dataset-dir ... --carbon-model-dir ... --training-config ... --run-dir ...`;
  the CLI writes `training_config.effective.yaml`, preflights that exact
  effective config, mirrors `training_preflight_report.json` into the
  run directory, and `--package-release-run` builds
  `training_run_manifest.json`, `training_run_card.md`, and
  `training_run_SHA256SUMS` immediately after a successful Carbon-backed
  run; `--resume-from predictor_checkpoint.pt` is available for Carbon
  runs but only accepts checkpoints whose run id, dataset snapshot, seed
  split, and config identity match the target run, and the resumed step
  is recorded in metrics, logs, and `training_run.json`;
  the paper run still requires a completed clean-machine Carbon-backed
  execution;
  completed training evidence is packaged with
  `python -m tools.release.training_run --run-dir ... --metadata-json ...`.
  Release training-run packages include checksum-covered
  `training_preflight_report.json`, require
  `generated_by=tools.release.training_run`, and release-mode
  verification requires the preflight report's dataset core-file
  evidence for `dataset_package.json`, `dataset_input_check_report.json`,
  `dataset_snapshot_report.json`, and `SHA256SUMS`. The paper/demo
  verifier rejects missing, stale, incomplete, or private-path preflight
  evidence plus `training_run_card.md` drift from
  `training_run_manifest.json` before model publication can pass.
- Checkpoint is packaged with
  `python -m tools.release.model_package --model-dir ... --metadata-json ...`
  before publication; the model-package command writes normalized
  `model_package.json`, renders `model_card.md` from that metadata plus
  `manifest.json`, requires
  `generated_by=tools.release.model_package`, requires packaged
  `eval_metrics.json` plus `efficiency_report.json`, verifies
  `eval_report.md` is rendered from the metrics source, requires the
  `tools.release.efficiency_report` source header, cross-checks
  eval/efficiency release id, dataset snapshot, commit, and model-result
  identity, requires model metadata to list
  `training_preflight_report.json`, `training_run_manifest.json`,
  `training_run_card.md`, and `training_run_SHA256SUMS` as release
  evidence, and includes all generated source artifacts plus model-local
  eval artifact references from `eval_metrics.json` in `SHA256SUMS`.
  The paper/package verifier
  re-renders the model card, rejects invalid or duplicate checksum
  paths, binds training-run dataset snapshot, training config path/hash,
  and commit identity to the manifest plus eval/efficiency evidence, and
  rejects stale model metadata before Hub dry-runs or release-candidate
  reports pass.
- Evaluation metrics are first generated from real score/label artifacts
  with `geno-lewm-eval --scores-jsonl ... --labels-jsonl ... --efficiency-report ... --output-metrics ...`;
  primary score rows must carry `generated_by=geno-lewm-score`;
  `geno-lewm-eval` records checkpoint, config, dataset-manifest,
  effective eval config, efficiency, score, label, and baseline-score artifacts as
  package-relative paths under `--artifact-root` (defaulting to the
  metrics output directory), writes `eval_config.effective.yaml` beside
  `eval_metrics.json`, and prevents absolute private workstation paths
  from entering release metrics JSON;
  accepted metrics payloads must carry `generated_by=geno-lewm-eval`
  or `generated_by=geno-lewm-eval-all`, so paper reports cannot be
  rendered from hand-labelled metrics JSON;
  Carbon zero-shot baseline scores are generated separately with
  `geno-lewm-carbon-baseline --vcf ... --fasta ... --carbon-model-dir ... --output-scores ... --logp-cache-jsonl ...`
  and each baseline row carries
  `generated_by=geno-lewm-carbon-baseline`; optional sequence
  log-likelihood cache rows are scoped to the Carbon model and revision
  before reuse. Baseline scores are attached with
  `--baseline-scores-jsonl ... --baseline-score-field carbon_zero_shot_score --baseline-name carbon_zero_shot`;
  generated reports that include baseline comparisons are rejected unless
  `baseline`, `baseline_value`, and `delta_vs_baseline` are supplied
  together and the metrics payload also records a baseline score artifact;
  this emits
  deterministic stratified bootstrap confidence intervals by default and
  records an omission reason when bootstrap resampling is disabled;
  multiple metrics artifacts are then aggregated and rendered with
  `geno-lewm-eval-all --metrics-json ... --output-metrics ... --output-report ...`.
  That command refreshes `eval_config.effective.yaml` next to
  `eval_metrics.json`; the eval-report parser requires each accepted
  metrics payload to record it as a package-relative `eval_config`
  artifact, and generated reports must include the same artifact row. Metrics
  inputs must also live under the
  aggregate metrics directory so the report is tied to the committed
  eval config plus explicit CLI overrides without private absolute paths.
  Metric conclusions in `eval_metrics.json` must explicitly reference
  every measured metric name, split, measured value, and baseline delta
  when a baseline is present; `negative_findings` must be a non-empty
  list rendered as `## Negative Findings`, so generic result summaries
  cannot be packaged as paper conclusions.
  Inference efficiency evidence is generated separately with
  `python -m bench.inference --release-efficiency --model-dir ... --vcf ... --fasta ... --variant ... --window ... --output-json ...`
  so single-variant latency, batched throughput, peak memory,
  hardware/runtime notes, command, and package-relative or inline input
  identities are machine-readable release artifacts rather than prose
  claims.
  The lower-level report renderer remains available as
  `python -m tools.release.eval_report --metrics-json ... --output ...`
  and includes baselines, confidence intervals, hardware, wall-clock
  cost, and known failure modes, but it rejects metrics payloads whose
  generator is not one of the eval CLIs. The paper/package verifier requires
  generated report markers, the Summary/Artifacts/Results sections,
  model and dataset identity lines, checkpoint/config/dataset-manifest
  plus efficiency-report artifact rows, and baseline score artifacts
  whenever baseline rows are reported; it resolves eval artifact paths
  inside the package and validates primary/baseline score JSONL
  `generated_by` markers; model-local eval artifact references must also
  be listed in model `SHA256SUMS`; it also re-renders
  `eval_report.md` from the packaged `eval_metrics.json`, validates
  `efficiency_report.json`, checks that eval and efficiency evidence
  agree with the manifest release id and training dataset snapshot, and
  rejects stale Markdown.
- Terminal demo runs real model inference, not fixtures.
- Demo transcript is generated by `tools/demo/terminal_inference.py`
  from the actual `geno-lewm-score` command and records generated time,
  exit code, model release/version/id, score/receipt JSONL hashes, row
  counts, JSONL field names, artifact-input paths, and an explicit
  claim-boundary sentence; the same run emits
  `terminal_demo_manifest.json` to bind the command, model id, input
  identities, VCF input summary, transcript hash, score/receipt hashes,
  generated report hashes, and a compact `score_receipt_batch` summary
  with record count, checked score fields, receipt stream, model id,
  calibration hash, and runtime identity as machine-readable release evidence. The demo runner
  clears owned score, receipt, batch-report, and demo-manifest outputs
  before invoking the score command so stale JSONL rows cannot satisfy a
  later run. The package
  verifier rejects stale input identities, stale VCF input summaries, or
  VCF/FASTA demo inputs that are not shipped inside the demo package, and it requires recorded
  commands plus artifact labels to resolve to the canonical package
  files; it also rejects runtime-preflight command drift from the
  terminal-demo manifest command, stale terminal-demo manifest
  `runtime_preflight` summaries that no longer match
  `runtime_preflight_report.json`, stale transcript claim-boundary or
  artifact-input markers, stale manifest JSONL field lists, or
  `score_receipt_batch` summaries that no longer match the packaged
  score, receipt, and batch-report artifacts. The same run also emits
  `runtime_preflight_report.json` to record model/input hashes, native
  runtime dependency availability, backend probes, and the fail-closed
  network guard; release verification rejects reports generated with
  fixture/test manifest allowance enabled. Before writing
  `terminal_demo_manifest.json`, the demo runner re-opens that preflight
  report and rejects stale or mutated evidence whose model id, release
  id, VCF/FASTA identities, command argv, requested backend, runtime
  requirement flags, or model artifact checks no longer match the same
  run. The same run also emits
  `batch_receipt_report.json` so the score rows, receipt rows, model
  id, calibration hash, runtime identity, and per-row output
  commitments are checked as one batch artifact. The
  release-package verifier rejects score/receipt batches whose model id
  or calibration hash do not match the packaged model manifest.
- Paper draft is generated from the release artifacts with
  `python -m tools.release.paper_draft --model-dir ... --dataset-dir ... --demo-dir ... --output ...`
  so Citation Metadata, Results, Conclusions, Negative Findings,
  Limitations, and Artifact Availability are grounded in the generated
  eval report, efficiency report, manifest, dataset package, and demo
  evidence. Draft generation rejects stale
  `eval_report.md` output that no longer matches `eval_metrics.json`
  and stale terminal-demo VCF summaries that no longer match the
  packaged demo VCF, requires a UTC `Generated: ...Z` timestamp, then
  renders that scored-input summary in Demo Evidence.
  The draft names
  `model_package.json`, `dataset_package.json`,
  `dataset_input_check_report.json`,
  `dataset_snapshot_report.json`, `eval_metrics.json`,
  `eval_config.effective.yaml`, `eval_report.md`,
  `efficiency_report.json`, and demo evidence paths, using
  package-local artifact names rather than build-machine root paths;
  the package verifier re-renders the draft from the current artifact
  set and rejects stale Markdown or drafts missing Citation Metadata or
  Negative Findings.
- Release package passes `python -m tools.release.paper_package` across
  the model, dataset, demo, and paper artifacts.
- Hub publication dry-run passes
  `python -m tools.release.hub_release --model-dir ... --dataset-dir ... --demo-dir ...`
  before any checkpoint upload; paper candidates require `--paper-url`.
  The versioned `hub_release_plan.json` records model files from `SHA256SUMS` plus
  `training_run_SHA256SUMS`, dataset files plus dataset `SHA256SUMS`,
  and demo files from portable `terminal_demo_manifest.json` with unique
  GitHub release asset names. When a paper artifact is included, it also
  records the verified public-safe paper source name/path, SHA-256, and
  size next to the public paper URL. For a direct GitHub
  `.../releases/download/<tag>/<paper-file>` URL whose asset name
  matches the verified paper file, the plan also emits the exact paper
  upload command. Private files beside the package are never published
  by a directory sync.
  The non-publishing `.github/workflows/release-hub-dry-run.yml`
  workflow runs the package verifier, Hub dry-run planner, and release
  candidate report without requiring Hub credentials.
- Credentialed publication runs
  `python -m tools.release.hub_publish --model-dir ... --dataset-dir ... --demo-dir ...`
  through `.github/workflows/release-hub-publish.yml` after the dry-run
  is clean. The workflow requires the protected `release` environment,
  `HF_TOKEN`, and GitHub release permissions; it syncs the locked
  `dev`, `train`, `eval`, and `deploy` extras so the clean-machine
  replay has the native runtime stack available; it uploads only the
  model, dataset, demo, and matching paper files named by the verified
  Hub plan. Paper publication requires a direct GitHub release download
  URL whose final asset name matches the verified paper file, because
  the final release-candidate check hashes the public paper URL bytes.
  The helper then regenerates `release_candidate_report.json` from the
  public links and fetched public artifact bytes. The protected workflow then runs the
  clean-machine terminal replay from that ready report with native
  runtime checks enabled. It passes the release `HF_TOKEN` only to
  Hugging Face artifact fetches and the GitHub token only to the release
  asset listing. After the final binder passes, the workflow uploads
  `hub_release_plan.json`, `release_candidate_report.json`,
  `hub_publish_report.json`, `clean_machine_demo_report.json`, and
  `publication_evidence_report.json`, then runs
  `python -m tools.release.publication_assets` to write
  `publication_evidence_assets.json` with the GitHub release target and
  evidence-asset hashes and upload command. It uploads that manifest plus the
  clean-machine replay transcript, manifest, score/receipt JSONL
  streams, runtime preflight report, and batch receipt report to the
  public demo release tag, and keeps the replay directory as a workflow
  artifact for debugging.
- A generated release-candidate report from
  `python -m tools.release.release_candidate --model-dir ... --dataset-dir ... --demo-dir ... --paper-path ... --paper-url ... --repo-id ... --dataset-url ... --demo-url ... --commit-sha ... --output ...`
  binds the package verifier, Hub publication plan, public-link reachability
  checks, commit, model id, dataset snapshot, dataset package metadata,
  dataset snapshot report, source metrics JSON, effective eval config,
  generated eval report, efficiency report,
  manifest-backed checkpoint/config/calibration artifacts,
  training-run checksums, Hub model/dataset/demo upload
  inventories, and key artifact hashes using package-role artifact paths
  rather than private absolute workstation paths. It also emits a `readiness`
  checklist covering package verification, model artifacts, dataset
  artifacts, terminal-demo evidence, paper artifact, public links,
  provider-backed public artifact exact file-set, hash, and size checks
  plus direct paper byte hash/size checks, and
  upload-plan completeness; readiness rows and blockers carry `issue_refs`
  pointing to the live release issues that own each failure. `ready=true`
  requires the model, dataset, demo, and
  paper URLs to be reachable and, for recognized Hugging Face/GitHub
  targets, requires the remote listings to contain exactly the expected
  model, dataset, and terminal-demo files, and requires the public paper
  URL bytes to match the verified paper file hash and size. Fetched
  public bytes must match the upload-inventory SHA-256 and size values unless the command
  is explicitly run in offline fixture mode with both
  `--allow-fixture-manifest` and `--skip-public-link-check`; skipping
  public checks without fixture mode keeps `ready=false`.
- Dataset, model, training-run, paper-draft, and terminal-demo command
  reports use package-local artifact names in their success JSON output;
  the terminal transcript uses the same portable names for the score
  command, output artifacts, and input references. These artifacts must
  not serialize private workstation roots.
- Clean-machine terminal replay from
  `python -m tools.release.clean_machine_demo --release-candidate-report ... --output-dir ...`
  downloads the published model files, dataset snapshot files, and
  GitHub release demo assets named by the generated ready
  release-candidate report. It rejects hand-authored reports, candidates
  missing generated readiness rows, candidates with non-empty blockers,
  skipped or failed public link checks, and skipped, missing, incomplete,
  or failed public artifact checks before any replay download. It also
  rejects embedded Hub plans whose source headers or model/repo/URL
  identities do not match, rejects unsafe Hub-plan destinations or
  malformed expected hashes before network fetches, verifies downloaded SHA-256
  values against the Hub plan,
  re-runs `tools.release.paper_package` on the downloaded model,
  dataset, and demo package, reruns `geno-lewm-score` from those
  downloaded bytes, then rejects replayed `terminal_demo_manifest.json`
  files with invalid source headers, non-passing status, model id
  mismatch, downloaded `model/manifest.json` hash/size mismatch, stale
  VCF/FASTA input identities, stale `runtime_preflight` summaries,
  stale `score_receipt_batch` summaries, or replay artifact hash/size drift
  before writing the clean-machine report. The final publication binder also checks the replay manifest's VCF/FASTA
  input identities against the downloaded demo artifacts and checks the
  replay manifest's artifact table against the clean-machine replay
  report for the transcript, scores, receipts, runtime preflight, and
  batch report.
  Before scoring, the replay helper checks the downloaded demo
  VCF/FASTA hashes and sizes against the downloaded demo manifest; after
  scoring, it rejects replay manifests whose VCF/FASTA identities do not
  match those downloaded inputs. The replay tool writes
  `clean_machine_demo_report.json` with
  the release-candidate report filename plus hash/size identity,
  output-directory-relative downloaded artifact identities,
  package-verification result, replay transcript and manifest identities,
  and replay score, receipt, runtime-preflight, and batch-report artifact
  hashes without serializing private absolute workstation paths. Optional
  `HF_TOKEN`, `HUGGINGFACE_HUB_TOKEN`,
  `GH_TOKEN`, or `GITHUB_TOKEN` environment values are used only for
  authenticated fetches and are never serialized into the report.
- Final publication evidence from
  `python -m tools.release.publication_report --plan ... --release-candidate ... --publish-report ... --clean-machine-demo-report ... --output ...`
  writes `publication_evidence_report.json`, which binds the Hub release
  plan, release-candidate report, credentialed publish report, and
  clean-machine replay report by public-safe filename plus hash/size
  identity, including the
  clean-machine replay's recorded release-candidate report
  filename/path, hash, and size identity, the verified paper file source
  name, URL, hash, and size identity, the full paper-critical
  `release_candidate_artifacts` table for model, dataset, eval, demo,
  and paper identities, public-safe release-candidate readiness rows
  plus public link and public artifact check summaries, every uploaded
  release-candidate artifact identity in that table checked against the
  Hub plan plus the downloaded public artifact, and the replayed terminal-demo
  manifest's model id, downloaded `manifest.json` identity, VCF/FASTA
  input identities, `runtime_preflight` summary, and replayed
  runtime-preflight model/input identities without private absolute paths. It also rejects a release
  candidate whose embedded Hub plan differs from
  `hub_release_plan.json`, requires the generated readiness checklist
  with all expected rows marked `ok=true`, empty candidate blockers, and
  current `issue_refs`, requires generated `public_links` and
  `public_artifacts` sections with required checks present and passing
  for the model, dataset, demo, and paper/public artifact targets, and
  fails the release gate if the published
  candidate, final readiness check, exact Hub-plan download set, public
  source URLs, hashes, or replay artifacts disagree. Its `issues`
  entries carry `issue_refs` so final publication failures route back to
  #163, #164, #165, #166, #167, and #101. The protected publish workflow
  uploads the resulting evidence JSON files and asset manifest as
  GitHub release assets, so paper/demo release notes can link durable
  public evidence rather than a retention-scoped workflow artifact.
- README and docs distinguish measured results from targets.
- Privacy statement and safety boundaries are consistent with the demo.

Current gaps are tracked in [ROADMAP.md](ROADMAP.md),
[docs/roadmap/IMPLEMENTATION.md](docs/roadmap/IMPLEMENTATION.md), and
GitHub issues.

---

## Repository Layout

```text
GenoLeWM/
├── geno_lewm/
│   ├── action/          # edit specs, relative edits, edit application, samplers
│   ├── provenance/      # preferred manifest, hashing, commitment, receipt API
│   ├── cli/             # console entry points
│   ├── deploy/          # runtime/update/export scaffolds
│   ├── encoder/         # Carbon windowing/cache scaffolds
│   ├── evaluation.py    # measured metrics and eval report payloads
│   ├── carbon_zero_shot.py # Carbon baseline score artifacts
│   ├── planning/        # latent planning contracts
│   ├── predictor/       # predictor, rollout, and loss contracts
│   ├── surprise/        # surprise scoring/calibration contracts
│   ├── training/        # fixture/Carbon training and preflight helpers
│   ├── errors.py        # typed exception hierarchy
│   ├── observability.py # structured logs and event registry
│   └── metrics.py       # metrics registry/export
├── bench/               # local benchmark and release-efficiency harnesses
├── configs/             # checked first-experiment training/eval configs
├── tests/               # unit, property, lint, API snapshot, benchmark tests
├── tools/               # API snapshot, lint gates, release tooling
├── docs/                # MkDocs source
├── rfcs/                # design records
├── examples/            # executable notebooks and fixture data
├── desktop/             # reference desktop scaffold
└── pyproject.toml
```

---

## Development

```bash
make install
make hooks
make ci
```

Important gates:

| Gate | Command |
| --- | --- |
| Lockfile | `uv lock --check` |
| Format | `ruff format --check .` |
| Lint | `ruff check .` |
| Types | `mypy geno_lewm tools` |
| Tests | `pytest` |
| ML smoke | `pytest tests/ml -q --tb=long --durations=10` |
| Eval smoke | `python -m tools.ci.eval_smoke_gate --work-dir .eval-smoke --summary-json .eval-smoke/eval_smoke_summary.json` |
| Public API | `python tools/api/snapshot.py check` |
| Scope language | `python -m tools.lint.check_scope_language` |
| Dataset spec | `python -m tools.release.dataset_snapshot --spec-json configs/first_experiment/dataset-snapshot-snv.json --check-spec` |
| Release docs contract | `pytest tests/lint/test_docs_release_blocker_contract.py -q` |
| Docs | `mkdocs build --strict` |
| Package build | `python -m build && twine check --strict dist/* && python -m tools.release.check_sdist_assets dist/*.tar.gz` |

The public API snapshot is intentional. If you change a public symbol,
update the snapshot in the same PR and explain the compatibility impact.

---

## Contributing

Start with [CONTRIBUTING.md](CONTRIBUTING.md). The most useful
contributions now are implementation work that moves the project toward
the first real experiment:

- Carbon encoder integration that works on a clean machine.
- Dataset builders with pinned revisions, tuple-builder wiring, holdout
  enforcement, and small deterministic smoke fixtures.
- Trainer/evaluator paths that produce publishable artifacts.
- A terminal demo that runs a released checkpoint on a real variant.
- Documentation that keeps claims aligned with measured behavior.

Personal-genome reproducers are not accepted. Use synthetic data or
public benchmark files.

---

## Safety

GenoLeWM is a research tool. It is not a diagnostic device, clinical
decision-support system, or medical product. Do not use it for embryo
selection, reproductive decision-making, or clinical care.

The runtime is designed to be local-first. Variant data should remain on
the user's machine unless the user explicitly exports it. See
[PRIVACY.md](PRIVACY.md) and [SECURITY.md](SECURITY.md).

---

## Citation

```bibtex
@software{genolewm2026,
  title  = {{GenoLeWM}: Action-conditioned {JEPA} world models for genomic edits},
  author = {{GenoLeWM Authors}},
  year   = {2026},
  url    = {https://github.com/AbdelStark/GenoLeWM},
  note   = {Apache-2.0},
}
```

---

## Acknowledgments

GenoLeWM builds on the LeWorldModel/LeJEPA idea of action-conditioned
latent prediction and on Carbon as the frozen DNA foundation model. The
project is independent; any errors in implementation or interpretation
are ours.
