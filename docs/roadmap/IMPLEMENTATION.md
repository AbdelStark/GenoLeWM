# Implementation Tracker — 2026-05-20

Generated from the spec corpus committed on branch
`spec/bootstrap-2026-05-20` (PR #1) and the issue set filed via the
spec-bootstrap turn. Every implementable unit of work in the spec corpus
is an issue below. Each issue is independently shippable; cross-tracker
dependencies are noted under "Cross-cutting dependencies."

The full label taxonomy, milestones, and CI gates are documented in
[`../spec/09-release-and-versioning.md`](../spec/09-release-and-versioning.md)
and [RFC-0015](../../rfcs/0015-testing-strategy.md).

## Milestones

| Milestone | Phase | Description |
|-----------|-------|-------------|
| `v0.1` | Phase 1 (MVP) | Carbon-500M frozen, SNVs only, ClinVar coding/non-coding eval, reference checkpoint |
| `v0.2` | Phase 2 | Indels + MNVs, LoRA encoder, LeJEPA, planning, calibrated surprise, full eval |
| `v0.3` | Phase 3 | Export pipeline, quantization, desktop app skeleton, TEE attestation |
| `v0.4` | Phase 4 | STARK proving of predictor forward pass |
| `Backlog` | — | Unscheduled work, post-v1 |

## Tracking issues


| Subsystem | RFC | Tracking | Children |
|-----------|-----|---------:|---------:|
| Error taxonomy | 0012 | [#2](https://github.com/AbdelStark/GenoLeWM/issues/2) | 2 |
| Observability | 0013 | [#3](https://github.com/AbdelStark/GenoLeWM/issues/3) | 5 |
| Configuration system | 0017 | [#4](https://github.com/AbdelStark/GenoLeWM/issues/4) | 2 |
| CLI design | 0018 | [#5](https://github.com/AbdelStark/GenoLeWM/issues/5) | 2 |
| State encoder (Carbon) | 0002 | [#6](https://github.com/AbdelStark/GenoLeWM/issues/6) | 5 |
| Action representation and encoder | 0003 | [#7](https://github.com/AbdelStark/GenoLeWM/issues/7) | 4 |
| Predictor architecture | 0004 | [#8](https://github.com/AbdelStark/GenoLeWM/issues/8) | 3 |
| Training loop and objective | 0005 | [#9](https://github.com/AbdelStark/GenoLeWM/issues/9) | 4 |
| Data pipeline | 0006 | [#10](https://github.com/AbdelStark/GenoLeWM/issues/10) | 5 |
| Evaluation suite | 0007 | [#11](https://github.com/AbdelStark/GenoLeWM/issues/11) | 6 |
| Latent planning (CEM) | 0008 | [#12](https://github.com/AbdelStark/GenoLeWM/issues/12) | 3 |
| Surprise scoring and calibration | 0009 | [#13](https://github.com/AbdelStark/GenoLeWM/issues/13) | 4 |
| Deployment and runtime | 0010 | [#14](https://github.com/AbdelStark/GenoLeWM/issues/14) | 8 |
| Attestation and receipts | 0011 | [#15](https://github.com/AbdelStark/GenoLeWM/issues/15) | 6 |
| Reference desktop app | 0019 | [#16](https://github.com/AbdelStark/GenoLeWM/issues/16) | 3 |
| Public API stability | 0014 | [#17](https://github.com/AbdelStark/GenoLeWM/issues/17) | 2 |
| Testing strategy and CI gates | 0015 | [#18](https://github.com/AbdelStark/GenoLeWM/issues/18) | 5 |
| Performance budget | 0016 | [#19](https://github.com/AbdelStark/GenoLeWM/issues/19) | 3 |
| Documentation, packaging, release | — | [#20](https://github.com/AbdelStark/GenoLeWM/issues/20) | 10 |

## Milestone: v0.1

Total issues: **62**

| # | Title | Area | Type | Priority | Effort | RFC |
|---:|------|------|------|---------:|-------:|-----|
| [#21](https://github.com/AbdelStark/GenoLeWM/issues/21) | errors: implement GenoLeWMError hierarchy and ERROR_CODES registry | `errors` | `type:feature` | `p0` | `m` | RFC-0012 |
| [#22](https://github.com/AbdelStark/GenoLeWM/issues/22) | errors: AST linter `raise_geno_lewm_error` and `registered_error_code` | `errors` | `type:ci` | `p0` | `s` | RFC-0012, RFC-0015 |
| [#23](https://github.com/AbdelStark/GenoLeWM/issues/23) | observability: implement structured logger and EVENTS registry | `observability` | `type:feature` | `p0` | `m` | RFC-0013 |
| [#24](https://github.com/AbdelStark/GenoLeWM/issues/24) | observability: implement redaction filter (default strict) | `observability` | `type:security` | `p0` | `m` | RFC-0013 |
| [#25](https://github.com/AbdelStark/GenoLeWM/issues/25) | observability: implement metrics registry + Prometheus textfile exporter | `observability` | `type:feature` | `p1` | `m` | RFC-0013 |
| [#26](https://github.com/AbdelStark/GenoLeWM/issues/26) | observability: opt-in wandb sink for training metrics | `observability` | `type:feature` | `p2` | `s` | RFC-0013 |
| [#27](https://github.com/AbdelStark/GenoLeWM/issues/27) | observability: AST linter `registered_event_name` and `registered_metric_name` | `observability` | `type:ci` | `p1` | `s` | RFC-0013, RFC-0015 |
| [#28](https://github.com/AbdelStark/GenoLeWM/issues/28) | config: implement Hydra defaults tree and Pydantic schema | `config` | `type:feature` | `p0` | `m` | RFC-0017 |
| [#29](https://github.com/AbdelStark/GenoLeWM/issues/29) | config: implement `--print-config`, `--print-config-tree`, `--explain` | `config` | `type:feature` | `p1` | `s` | RFC-0017, RFC-0018 |
| [#30](https://github.com/AbdelStark/GenoLeWM/issues/30) | cli: implement dispatcher (`_dispatch.py`) and shared flags | `cli` | `type:feature` | `p0` | `m` | RFC-0018, RFC-0012 |
| [#31](https://github.com/AbdelStark/GenoLeWM/issues/31) | cli: implement shell completion (bash, zsh, fish) | `cli` | `type:feature` | `p2` | `s` | RFC-0018 |
| [#32](https://github.com/AbdelStark/GenoLeWM/issues/32) | encoder: implement CarbonStateEncoder (frozen Carbon-500M) | `encoder` | `type:feature` | `p0` | `m` | RFC-0002 |
| [#33](https://github.com/AbdelStark/GenoLeWM/issues/33) | encoder: implement windowing and tokenizer wrapping | `encoder` | `type:feature` | `p0` | `s` | RFC-0002 |
| [#34](https://github.com/AbdelStark/GenoLeWM/issues/34) | encoder: implement pooling strategies (centered_mean default) | `encoder` | `type:feature` | `p0` | `s` | RFC-0002 |
| [#35](https://github.com/AbdelStark/GenoLeWM/issues/35) | encoder: implement Parquet shard cache + SQLite index | `encoder` | `type:feature` | `p0` | `m` | RFC-0002, RFC-0006 |
| [#36](https://github.com/AbdelStark/GenoLeWM/issues/36) | encoder: implement `geno-lewm-cache-windows` CLI | `encoder` | `type:feature` | `p1` | `m` | RFC-0002, RFC-0006, RFC-0018 |
| [#37](https://github.com/AbdelStark/GenoLeWM/issues/37) | action: implement EditSpec / RelEdit / EditType with validation | `action` | `type:feature` | `p0` | `s` | RFC-0003 |
| [#38](https://github.com/AbdelStark/GenoLeWM/issues/38) | action: implement apply_edit / apply_edits (right-to-left) | `action` | `type:feature` | `p0` | `s` | RFC-0003 |
| [#39](https://github.com/AbdelStark/GenoLeWM/issues/39) | action: implement ActionEncoder module (4 sub-encoders + projection MLP) | `action` | `type:feature` | `p0` | `m` | RFC-0003 |
| [#40](https://github.com/AbdelStark/GenoLeWM/issues/40) | action: implement synthetic edit samplers (SNV / indel / MNV) | `action` | `type:feature` | `p0` | `s` | RFC-0003, RFC-0006 |
| [#41](https://github.com/AbdelStark/GenoLeWM/issues/41) | predictor: implement cross-attention Predictor module | `predictor` | `type:feature` | `p0` | `l` | RFC-0004 |
| [#42](https://github.com/AbdelStark/GenoLeWM/issues/42) | predictor: implement ARPredictor rollout with KV cache | `predictor` | `type:feature` | `p0` | `m` | RFC-0004 |
| [#43](https://github.com/AbdelStark/GenoLeWM/issues/43) | predictor: implement losses (cosine + MSE; LeJEPA monitoring) | `predictor` | `type:feature` | `p0` | `s` | RFC-0005 |
| [#44](https://github.com/AbdelStark/GenoLeWM/issues/44) | training: implement trainer scaffold (AdamW + WSD schedule) | `training` | `type:feature` | `p0` | `m` | RFC-0005, RFC-0017 |
| [#45](https://github.com/AbdelStark/GenoLeWM/issues/45) | training: implement edit-balanced sampler | `training` | `type:feature` | `p1` | `s` | RFC-0005, RFC-0006 |
| [#46](https://github.com/AbdelStark/GenoLeWM/issues/46) | training: implement collapse monitoring (var, pairwise dist, kl_reg) | `training` | `type:feature` | `p1` | `s` | RFC-0005 |
| [#47](https://github.com/AbdelStark/GenoLeWM/issues/47) | training: reproducibility plumbing (seeds, deterministic mode) | `training` | `type:feature` | `p1` | `s` | RFC-0005 |
| [#48](https://github.com/AbdelStark/GenoLeWM/issues/48) | data: implement Carbon-pretraining-corpus loader and window sampler | `data` | `type:feature` | `p0` | `m` | RFC-0006 |
| [#49](https://github.com/AbdelStark/GenoLeWM/issues/49) | data: implement gnomAD loader + geno-lewm-prepare-gnomad CLI | `data` | `type:feature` | `p0` | `m` | RFC-0006, RFC-0018 |
| [#50](https://github.com/AbdelStark/GenoLeWM/issues/50) | data: implement ClinVar loader + geno-lewm-prepare-clinvar CLI | `data` | `type:feature` | `p0` | `m` | RFC-0006, RFC-0018 |
| [#51](https://github.com/AbdelStark/GenoLeWM/issues/51) | data: implement tuple builder and IterableDataset | `data` | `type:feature` | `p0` | `m` | RFC-0006 |
| [#52](https://github.com/AbdelStark/GenoLeWM/issues/52) | data: implement holdout enforcement (chr21, ClinVar, haplotypes) | `data` | `type:feature` | `p0` | `s` | RFC-0006 |
| [#53](https://github.com/AbdelStark/GenoLeWM/issues/53) | eval: implement ClinVar coding/non-coding VEP harness | `eval` | `type:feature` | `p0` | `m` | RFC-0007 |
| [#54](https://github.com/AbdelStark/GenoLeWM/issues/54) | eval: implement smoke eval gate (1k variants + 500 rollout) | `eval` | `type:ci` | `p0` | `s` | RFC-0007, RFC-0015 |
| [#55](https://github.com/AbdelStark/GenoLeWM/issues/55) | eval: implement Carbon zero-shot baseline runner | `eval` | `type:feature` | `p1` | `s` | RFC-0007 |
| [#56](https://github.com/AbdelStark/GenoLeWM/issues/56) | eval: implement geno-lewm-eval and geno-lewm-eval-all CLIs | `eval` | `type:feature` | `p1` | `s` | RFC-0007, RFC-0018 |
| [#58](https://github.com/AbdelStark/GenoLeWM/issues/58) | eval: implement efficiency benchmark harness | `eval` | `type:perf` | `p1` | `m` | RFC-0007, RFC-0016 |
| [#62](https://github.com/AbdelStark/GenoLeWM/issues/62) | surprise: implement raw surprise score and score_variant/score_vcf | `surprise` | `type:feature` | `p0` | `m` | RFC-0009 |
| [#63](https://github.com/AbdelStark/GenoLeWM/issues/63) | surprise: implement context stratification labels (region, GC, repeat) | `surprise` | `type:feature` | `p1` | `m` | RFC-0009 |
| [#64](https://github.com/AbdelStark/GenoLeWM/issues/64) | surprise: implement calibration table builder + on-disk format | `surprise` | `type:feature` | `p1` | `m` | RFC-0009 |
| [#65](https://github.com/AbdelStark/GenoLeWM/issues/65) | surprise: implement geno-lewm-score CLI | `surprise` | `type:feature` | `p1` | `s` | RFC-0009, RFC-0018 |
| [#74](https://github.com/AbdelStark/GenoLeWM/issues/74) | attestation: implement content-addressed model IDs + manifest schema | `attestation` | `type:feature` | `p0` | `m` | RFC-0011 |
| [#75](https://github.com/AbdelStark/GenoLeWM/issues/75) | attestation: implement input commitment | `attestation` | `type:feature` | `p0` | `s` | RFC-0011 |
| [#76](https://github.com/AbdelStark/GenoLeWM/issues/76) | attestation: implement receipt writer/reader | `attestation` | `type:feature` | `p0` | `s` | RFC-0011 |
| [#77](https://github.com/AbdelStark/GenoLeWM/issues/77) | attestation: implement geno-lewm-verify CLI (checksum mode) | `attestation` | `type:feature` | `p0` | `s` | RFC-0011, RFC-0018 |
| [#83](https://github.com/AbdelStark/GenoLeWM/issues/83) | api: implement public API snapshot test | `docs` | `type:test` | `p1` | `s` | RFC-0014 |
| [#84](https://github.com/AbdelStark/GenoLeWM/issues/84) | api: implement @experimental and @deprecated decorators | `docs` | `type:feature` | `p2` | `s` | RFC-0014 |
| [#85](https://github.com/AbdelStark/GenoLeWM/issues/85) | testing: scaffold test pyramid and fixture data | `ci` | `type:test` | `p0` | `m` | RFC-0015 |
| [#86](https://github.com/AbdelStark/GenoLeWM/issues/86) | testing: implement GitHub Actions per-PR CI workflow | `ci` | `type:ci` | `p0` | `m` | RFC-0015 |
| [#87](https://github.com/AbdelStark/GenoLeWM/issues/87) | testing: implement AST linters (no_print, network_confined) | `ci` | `type:ci` | `p0` | `s` | RFC-0015 |
| [#88](https://github.com/AbdelStark/GenoLeWM/issues/88) | testing: implement coverage gate (changed-files ≥ 90%) | `ci` | `type:ci` | `p1` | `s` | RFC-0015 |
| [#89](https://github.com/AbdelStark/GenoLeWM/issues/89) | testing: implement ML smoke suite (identity-at-init, loss-decreases) | `ci` | `type:test` | `p0` | `s` | RFC-0015 |
| [#90](https://github.com/AbdelStark/GenoLeWM/issues/90) | perf: implement benchmark harness (bench/) and persisted results | `ci` | `type:perf` | `p1` | `m` | RFC-0016 |
| [#91](https://github.com/AbdelStark/GenoLeWM/issues/91) | perf: implement pytest-benchmark microbench suite for hot paths | `ci` | `type:perf` | `p2` | `s` | RFC-0016 |
| [#92](https://github.com/AbdelStark/GenoLeWM/issues/92) | perf: implement nightly perf regression detector | `ci` | `type:perf` | `p2` | `s` | RFC-0016 |
| [#93](https://github.com/AbdelStark/GenoLeWM/issues/93) | docs: set up mkdocs/material site and auto-generated API reference | `docs` | `type:docs` | `p2` | `m` | — |
| [#94](https://github.com/AbdelStark/GenoLeWM/issues/94) | docs: tutorial notebook `01_score_single_variant.ipynb` | `docs` | `type:docs` | `p2` | `s` | RFC-0009, RFC-0011 |
| [#95](https://github.com/AbdelStark/GenoLeWM/issues/95) | docs: tutorial notebook `02_score_brca2_saturation.ipynb` | `docs` | `type:docs` | `p2` | `s` | RFC-0007, RFC-0009 |
| [#96](https://github.com/AbdelStark/GenoLeWM/issues/96) | docs: tutorial notebook `03_score_vcf.ipynb` | `docs` | `type:docs` | `p2` | `s` | RFC-0009, RFC-0010 |
| [#100](https://github.com/AbdelStark/GenoLeWM/issues/100) | release: implement PyPI release workflow | `packaging` | `type:ci` | `p1` | `m` | — |
| [#101](https://github.com/AbdelStark/GenoLeWM/issues/101) | release: implement HuggingFace Hub upload workflow for model checkpoints | `packaging` | `type:ci` | `p1` | `m` | RFC-0011 |
| [#102](https://github.com/AbdelStark/GenoLeWM/issues/102) | release: implement CHANGELOG + version-bump tooling | `packaging` | `type:ci` | `p2` | `s` | — |

## Milestone: v0.2

Total issues: **6**

| # | Title | Area | Type | Priority | Effort | RFC |
|---:|------|------|------|---------:|-------:|-----|
| [#57](https://github.com/AbdelStark/GenoLeWM/issues/57) | eval: implement rollout-fidelity harness | `eval` | `type:feature` | `p2` | `s` | RFC-0007 |
| [#59](https://github.com/AbdelStark/GenoLeWM/issues/59) | planning: implement CEM solver and `plan()` API | `planning` | `type:feature` | `p1` | `m` | RFC-0008 |
| [#60](https://github.com/AbdelStark/GenoLeWM/issues/60) | planning: implement cost functions and ActionSampler | `planning` | `type:feature` | `p1` | `s` | RFC-0008 |
| [#61](https://github.com/AbdelStark/GenoLeWM/issues/61) | planning: implement geno-lewm-plan CLI | `planning` | `type:feature` | `p2` | `s` | RFC-0008, RFC-0018 |
| [#97](https://github.com/AbdelStark/GenoLeWM/issues/97) | docs: tutorial notebook `04_multi_edit_rollout.ipynb` | `docs` | `type:docs` | `p2` | `s` | RFC-0004, RFC-0007 |
| [#98](https://github.com/AbdelStark/GenoLeWM/issues/98) | docs: tutorial notebook `05_planning_minimal_edits.ipynb` | `docs` | `type:docs` | `p2` | `s` | RFC-0008 |

## Milestone: v0.3

Total issues: **12**

| # | Title | Area | Type | Priority | Effort | RFC |
|---:|------|------|------|---------:|-------:|-----|
| [#66](https://github.com/AbdelStark/GenoLeWM/issues/66) | deploy: implement GenoLeWMRuntime runtime contract | `deploy` | `type:feature` | `p1` | `m` | RFC-0010 |
| [#67](https://github.com/AbdelStark/GenoLeWM/issues/67) | deploy: implement ONNX export | `deploy` | `type:feature` | `p1` | `m` | RFC-0010 |
| [#68](https://github.com/AbdelStark/GenoLeWM/issues/68) | deploy: implement Core ML export | `deploy` | `type:feature` | `p1` | `m` | RFC-0010 |
| [#69](https://github.com/AbdelStark/GenoLeWM/issues/69) | deploy: implement GGUF export for CPU-only runners | `deploy` | `type:feature` | `p2` | `m` | RFC-0010 |
| [#70](https://github.com/AbdelStark/GenoLeWM/issues/70) | deploy: implement int8 predictor and int4 Carbon quantization | `deploy` | `type:feature` | `p1` | `m` | RFC-0010, RFC-0016 |
| [#71](https://github.com/AbdelStark/GenoLeWM/issues/71) | deploy: implement geno-lewm-export CLI | `deploy` | `type:feature` | `p2` | `s` | RFC-0010, RFC-0018 |
| [#72](https://github.com/AbdelStark/GenoLeWM/issues/72) | deploy: implement personal-genome format converters (23andMe, Ancestry, MyHeritage, Sequencing.com) | `deploy` | `type:feature` | `p2` | `m` | RFC-0010 |
| [#73](https://github.com/AbdelStark/GenoLeWM/issues/73) | deploy: implement geno-lewm-update CLI (explicit, user-initiated) | `deploy` | `type:feature` | `p2` | `s` | RFC-0010, RFC-0018 |
| [#78](https://github.com/AbdelStark/GenoLeWM/issues/78) | attestation: implement TEE-signed receipts (v1.1) | `attestation` | `type:feature` | `p2` | `m` | RFC-0011 |
| [#80](https://github.com/AbdelStark/GenoLeWM/issues/80) | desktop: scaffold Tauri 2 application | `desktop` | `type:feature` | `p1` | `m` | RFC-0019 |
| [#81](https://github.com/AbdelStark/GenoLeWM/issues/81) | desktop: implement file-drop, FASTA picker, scoring UI | `desktop` | `type:feature` | `p2` | `m` | RFC-0019 |
| [#82](https://github.com/AbdelStark/GenoLeWM/issues/82) | desktop: implement signing + notarization (macOS) and signed releases | `desktop` | `type:security` | `p2` | `m` | RFC-0019 |

## Milestone: v0.4

Total issues: **2**

| # | Title | Area | Type | Priority | Effort | RFC |
|---:|------|------|------|---------:|-------:|-----|
| [#79](https://github.com/AbdelStark/GenoLeWM/issues/79) | attestation: STARK circuit prototype for predictor forward pass | `attestation` | `type:feature` | `p2` | `l` | RFC-0011 |
| [#99](https://github.com/AbdelStark/GenoLeWM/issues/99) | docs: tutorial notebook `07_verify_receipt.ipynb` | `docs` | `type:docs` | `p2` | `s` | RFC-0011 |

## Cross-cutting dependencies

| Blocker | Blocks | Reason |
|---------|--------|--------|
| RFC-0012 error taxonomy → `geno_lewm/errors.py` (#21) | every subsystem | typed raises depend on the registry |
| RFC-0013 observability core (#23) | every subsystem | log/metric registries gate AST checks |
| RFC-0017 configuration → `geno_lewm/config/` (#28) | every CLI command, trainer | configs are required at process start |
| RFC-0014 API snapshot test (#81) | every public-symbol PR | snapshot gate |
| `data/prepare-gnomad` (#48) + `data/prepare-clinvar` (#49) | training, eval, calibration | data shards |
| `encoder/cache-windows` (#43) | trainer, scorer | reference embeddings |
| `eval/smoke` (#54) | per-PR CI | regression detection |
| RFC-0010 export pipeline (#67–#71) | desktop app, attestation receipts | quantized artifacts |
| RFC-0011 manifest + receipt (#73–#76) | deploy, surprise scorer, verifier | provenance contract |

## Open questions

The aggregate open-question list lives across the RFCs and the spec
corpus. Each `OQ-*` identifier is tracked there; resolution lands as
either an RFC amendment or a follow-up implementation issue.

Sections:

- [`docs/spec/00-overview.md` open questions](../spec/00-overview.md#open-questions-tied-to-scope)
- [`docs/spec/02-public-api.md` open questions](../spec/02-public-api.md#open-questions)
- [`docs/spec/03-data-model.md` open questions](../spec/03-data-model.md#open-questions)
- [`docs/spec/04-error-model.md` open questions](../spec/04-error-model.md#open-questions)
- [`docs/spec/05-observability.md` open questions](../spec/05-observability.md#open-questions)
- [`docs/spec/06-security.md` open questions](../spec/06-security.md#open-questions)
- [`docs/spec/07-testing-strategy.md` open questions](../spec/07-testing-strategy.md#open-questions)
- [`docs/spec/08-performance-budget.md` open questions](../spec/08-performance-budget.md#open-questions)
- [`docs/spec/09-release-and-versioning.md` open questions](../spec/09-release-and-versioning.md#open-questions)

## Regeneration

This file is mechanically regenerated from `/tmp/issues_manifest.json`
by the spec-bootstrap turn. Future regeneration after issues land in
new states should run the same pipeline and commit the diff. The
canonical state of each issue is on GitHub; this file is a snapshot of
the corpus → issue mapping at filing time.

