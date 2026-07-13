# GenoLeWM Architecture

GenoLeWM is an action-conditioned latent world model for genomic edits.
It uses a frozen DNA foundation model as a state encoder and trains a
small action-conditioned predictor over that latent state space.

The current public release is an alpha research system. Every published
checkpoint uses the `legacy_raw_v1` state contract: the release configs
declared normalization, but the implementation supplied raw pooled Carbon
states to training and evaluation. Those checkpoints remain useful for
artifact replay and model-path inspection, but their metrics do not evaluate
the intended normalized method. The system is not clinical software.

## Runtime Flow

```text
reference window --Carbon encoder--> s_t
edit spec --------action encoder----> a_t
(s_t, a_t) -------predictor---------> s_hat_{t+1}
edited window ----Carbon encoder----> s_{t+1}
loss = distance(s_hat_{t+1}, s_{t+1}) + collapse regularization
```

The frozen Carbon encoder is the expensive part. The trainable GenoLeWM
components are the action encoder and the predictor. Once a reference
state is available, rollout and planning can search through the smaller
predictor path instead of repeatedly running Carbon during the search.

## Main Components

### State Encoder

`geno_lewm.encoder` wraps Carbon model/tokenizer loading, sequence
window normalization, pooling, explicit latent-state normalization, and
optional local state caching. `l2_normalized_v2` applies L2 normalization
after pooling. `legacy_raw_v1` preserves the historical raw pooled state
contract for compatibility only.

Cache schema `3.0.0` stores raw post-pooling, pre-normalization vectors. Its
identity includes the encoder runtime hash, layer, pooling mode/radius,
`center_token`, and logical compute dtype. Pooling and normalization explicitly
round their final outputs to canonical IEEE-754 FP32 before any downstream
consumer. The physical Parquet representation is recorded separately as
fixed-size FP32 and therefore preserves those canonical state bits for BF16,
FP16, and FP32 compute modes without mislabeling FP16 bytes. New shard paths
include the schema generation, encoder hash, logical dtype, physical dtype,
layer, and pooling identity; encoder ID and contig components are fixed ASCII
SHA-256 digests, avoiding filesystem, Unicode, case, and punctuation aliases.
Schema-2 shards remain readable and reindexable under an explicit legacy replay
policy but are never written by the schema-3 writer. Corrected training and
schema-3 rollout evidence fail closed unless v3 provenance is selected. Their
historical `dtype` column did not always describe the FP16 physical payload,
so compatibility reads support replay only; dtype-faithful evidence requires
regenerating the shard as schema 3.

Cross-process shard publication is serialized across path and key reservation.
Writes use no-follow directory traversal where supported, durably create each
namespace ancestor, stage and reopen a strict non-nullable Arrow schema, compare
canonical serialized row bits, and hard-link the final name without clobbering
an existing winner. A losing writer reopens and bit-verifies that winner. One
direct `BEGIN IMMEDIATE` SQLite transaction then inserts the batch with FULL
durability; whole-index repair/reindex alone uses a private database and atomic
replacement after `integrity_check`. The index stores cache schema and physical
encoding, allowing v2/v3 coexistence without ambiguous selection. Batched lookup
groups keys by shard and reads each required Parquet row group once. Cache v1
omitted the pooling center and remains deliberately invalidated. This foundation
does not claim completion of a full cache build or corrected training/evaluation.
Race-resistant cache I/O requires the directory-descriptor and no-follow
primitives provided by Linux and macOS; it fails closed on Windows and on any
runtime without those primitives. There is no path-based publication fallback.
A durable single-publication intent repairs a crash between final-shard linking
and index commit without scanning existing shards, while first-index creation is
private and atomically published so readers never observe an empty bootstrap.

The corrected Carbon path does not execute the upstream custom tokenizer. The
pinned upstream `tokenizer.py` delegated to an unpinned, network-capable
`Qwen/Qwen3-4B-Base` tokenizer lookup, so hashing that file alone could not make
the runtime self-contained. GenoLeWM now implements only the pure-DNA branch
from the local Carbon `dna_config.json` and `tokenizer_config.json`, validates
the control-token and six-mer layout, and loads model weights locally. This is
a runtime-contract repair, not evidence that a corrected checkpoint performs
well.

Centered pooling now derives its index from the validated token IDs. Carbon
places a leading `<dna>` control token before the first six-mer, so the DNA
token for a base-pair locus is at `dna_content_start + edit_locus // 6`.
Historical code used only `edit_locus // 6`; every intended center was one
hidden token too far left and some pools included the control token as their
center. Those historical rows remain invalid even when compared with a
coordinate-matched source or candidate label.

Training and evaluation use the same edit locus for source, target, and
candidate pooling. Resume checkpoints bind the encoder runtime identity,
revision, dtype, layer, pooling mode/radius, and effective normalization, so a
lineage cannot resume across latent representations.
The public scoring path still needs Carbon available at runtime because
the score compares the predicted post-edit state with the Carbon-encoded
edited state.

### Edit And Action Encoding

`geno_lewm.action` defines canonical genomic edit objects:

- `EditSpec` for chromosome-position-reference-alternate edits;
- `RelEdit` for window-relative edits;
- edit validation and edit application helpers;
- synthetic edit samplers used by tests and training data builders.

The action encoder maps a validated edit into a fixed-width embedding
that the predictor can attend to.

### Predictor And Rollout

`geno_lewm.predictor` contains the cross-attention predictor and the
autoregressive rollout wrapper. Single-edit prediction estimates the
post-edit latent state. Multi-edit rollout repeatedly feeds predicted
states back through the predictor.

The published rollout rows are historical implementation outputs. Their L2
distances compare raw source/target states with unit-normalized predictions
and are invalid. Their cosine values are scale-invariant, but remain
confounded by training under that mismatched contract and by autoregressive
train/rollout distribution shift. No corrected K>1 fidelity result is yet
published. The separate K=20 predictor timing artifact remains below its
original 5x engineering target.

### Surprise Scoring

`geno_lewm.surprise` computes a latent residual and a calibrated value:

- `sigma_raw`: uncalibrated distance between predicted and encoded
  post-edit states;
- `sigma_calibrated`: percentile-like value from the released
  calibration table;
- `bucket_id`, `confidence`, and `low_confidence`: calibration-context
  metadata.

For a newly trained `l2_normalized_v2` lineage, these fields are candidate
research outputs that still require validation. Values from published
`legacy_raw_v1` checkpoints mix raw targets with unit-normalized predictions;
their residuals and calibrations are invalid as scientific scores and are
retained only for compatibility and artifact inspection. They are not
clinical classifications or validated risk probabilities.

### Planning

`geno_lewm.planning` provides cost functions, action sampling, and a CEM
planner. The released demo proves only that the manifest-backed legacy path
executed against public artifacts. Its L2 objective used the mismatched
`legacy_raw_v1` state spaces, so `best_distance` is not a valid planning
objective value and the demo does not establish edit-selection capability.

### Runtime And Provenance

`geno_lewm.deploy` loads model directories, resolves optional runtime
dependencies, applies the local network boundary, and exposes
single-variant / VCF scoring.

`geno_lewm.provenance` implements manifests, canonical hashing,
input/output commitments, and checksum receipts. The main files are
`manifest.py`, `commitment.py`, and `receipt.py`. Receipts bind artifact
and output identity. They do not certify runtime behavior, privacy, or
scientific correctness.

For `l2_normalized_v2`, the manifest encoder hash covers Carbon weights and
the runtime-critical model config, DNA/tokenizer config, tokenizer vocabulary,
and custom tokenizer code. The native runtime verifies that local package and
uses the self-contained pure-DNA tokenizer before loading it. This closes the
unpinned transitive tokenizer dependency; the earlier claim that hashing
`tokenizer.py` alone established full local runtime identity was false.
Historical `legacy_raw_v1` manifests retain their weight-only hash semantics
for byte-compatible replay.

## Package Map

```text
geno_lewm/
├── action/       edit specs, edit application, action encoder
├── cli/          train, score, eval, rollout, plan, verify, export commands
├── config/       typed config loading and closed config schema
├── data/         VCF prep, tuple streaming, membership manifests/indexes
├── deploy/       runtime facade, scoring, import/export helpers
├── encoder/      Carbon wrapper, windowing, pooling, cache helpers
├── planning/     CEM planning, cost functions, action sampling
├── predictor/    cross-attention predictor, losses, AR rollout
├── provenance/   manifests, commitments, receipts, hashes
├── surprise/     raw and calibrated surprise scoring
└── training/     fixture trainer, Carbon preflight, real trainer launcher
```

Evaluation entry points live in `geno_lewm/evaluation.py` for metric
orchestration and `geno_lewm/carbon_zero_shot.py` for Carbon baseline
scoring.

### Command Packaging Boundary

Installed console-script targets must resolve without the repository's
source-only top-level `tools` package. Calibration lives in
`geno_lewm.cli.calibrate`; evaluation-report parsing/rendering and the v0.2
metric-requirement checks live in dependency-closed private package modules.
Training-run manifest/card/checksum assembly used after Carbon training is
likewise package-owned.
The corresponding `tools.release` modules are compatibility wrappers that
delegate to and re-export the installed implementations, so operator workflows
and existing imports do not fork from wheel behavior.

The build gate installs the wheel, changes to directories outside the checkout,
loads every stable console-script entry point, runs each `--help`, and executes
functional calibration-error, evaluation-aggregation, and training-release
packaging smokes.

## Training Data Flow

```text
reference windows ─┐
                   ├── tuple builder ── (w_ref, edit, w_alt)
edit sources ──────┘                         │
                                             ├── Carbon(w_ref) -> s_t
                                             ├── action_encoder(edit) -> a_t
                                             ├── predictor(s_t, a_t) -> s_hat
                                             └── Carbon(w_alt) -> s_{t+1}
```

Training packages record the resolved config, data snapshot identity,
preflight report, training metrics, checkpoint files, manifest/card, and
checksum inventory. New lineages also record the state-contract version.
The v0.2.1 Phase 2 KL term was computed only from frozen target states, so it
had no gradient with respect to the predictor or action encoder; it must not
be described as active regularization.

### v0.3 staging lineage boundary

The v0.3 data path separates remote staging verification, offline lineage
assembly, and future membership construction:

```text
generation-pinned gnomAD source lock
        │
        ├── per-autosome staging receipt ─┐
        ├── exact-revision postflight ────┼── offline lineage assembler
        │                                 │          │
corrected ClinVar release audit ──────────┤          ├── content-addressed lineage
ClinVar exact-revision postflight ────────┘          └── membership_status=not_created
```

`tools/data/v03_gnomad_lock.py remote-postflight` verifies each complete
gnomAD namespace at an immutable Hub revision and reruns a full Parquet audit.
`tools/data/v03_snapshot_lineage.py` then consumes only local evidence. It
cross-binds the repository, revision, namespace, source commit, chromosome,
namespace file inventory, receipt identity, Parquet identity, and fresh audit
for all 22 autosomes. It also binds the corrected ClinVar release audit to the
four-file exact-revision postflight, then reconciles its repository, revision,
namespace, producer commit, release, source/output identities, trusted
nine-field schema, full-scan counts, and original claim boundary. The existing
audit validation remains mandatory; postflight evidence augments it rather
than replacing it.

Lineage assembly remains an operator tool, while its read-only capture and
semantic verifier live in dependency-closed package code. The tool delegates
to and re-exports that single verifier implementation. Consequently, an
installed wheel can verify the exact bundled lineage during the default
membership-store open path without resolving the repository's `tools` tree.
The package verifier still performs one read, duplicate-key rejection,
type-strict semantic checks, content-ID recomputation, and deep freezing of the
captured value.

The input and output formats are closed Draft 2020-12 schemas with stable IDs
under `configs/data_v03/`. The output includes source-specific data-use terms
and the exact fields materialized from gnomAD and ClinVar. These records are
provenance and policy metadata, not a transfer of upstream rights.

The lineage assembler deliberately has no membership writer or network client.
A lineage candidate cannot be described as a dataset snapshot until a separate
downstream step constructs, audits, and commits memberships and leakage controls.
See the [operator guide](docs/data-v03-snapshot-lineage.md) for required
evidence and failure behavior.

### Dataset membership boundary

Fixture tests and small callers can continue to use the in-memory
`MembershipArtifact` contract. Production-scale v0.3 memberships use a
separate versioned store:

```text
snapshot-lineage.json + exact local staged Parquet shards
                  │
                  ├── private single-descriptor capture + exact verification
                  ├── derive canonical MembershipRow values
                  ├── SQLite-backed ordering, R-tree, dedup, leakage checks
                  ├── independent temp-store verification + fsync
                  └── atomic five-file store publication
```

The five files are `manifest.json`, `memberships.parquet`, `lookup.sqlite`,
the exact bundled `snapshot-lineage.json`, and `build-receipt.json`. The
Parquet file is the portable row artifact. The SQLite file is a derived,
manifest-bound point/R-tree range index used by the tuple builder and validation
stream without loading all variant keys into Python. A format-independent
digest over length-framed canonical rows defines semantic identity; the
manifest's separate physical identity commits the exact Parquet, SQLite,
lineage, and receipt bytes. Full verification rejects layout drift and scans
both row encodings independently with bounded memory.

PyArrow is loaded only by build and verification adapters. Manifest parsing
and read-only SQLite lookups stay on core dependencies. The current repository
ships this contract and synthetic tests only: it does not contain or claim a
real v0.3 membership store, split result, or phased-haplotype membership.
ClinVar membership derivation admits normalized B/LB/LP/P rows as labeled
memberships: benign-family rows are negative benchmark labels, while P/LP rows
on training chromosomes remain eligible training anchors. The gnomAD variant
membership path is not presented as the RFC's phased-haplotype set.

The stable `geno_lewm.data.membership_store` module is a small public facade.
Its private implementation has one-way dependencies: the closed contract is
the base; lineage, receipt, and storage depend on that contract; the verifier
depends on those three; the writer depends on the verifier; and the read-only
runtime depends on the verifier and storage. The membership lineage adapter
depends on the installable snapshot-lineage verifier, never on the operational
tool package. This keeps source ingestion, physical encoding, independent
verification, and online lookup separately testable without import cycles.

## Inference Data Flow

Single-variant scoring validates that the `REF` allele matches the
supplied reference window before model inference.

```text
variant + reference window
        │
        ├── validate coordinate / REF match
        ├── Carbon(reference window) -> s_t
        ├── action_encoder(edit) -> a_t
        ├── predictor(s_t, a_t) -> s_hat_{t+1}
        ├── Carbon(edited window) -> s_{t+1}
        └── surprise score + optional checksum receipt
```

VCF scoring repeats the same path over locally supplied VCF and FASTA
inputs. The project does not provide a hosted scoring service.

## Published Evidence Boundary

The public v0.2.1 run tree contains benchmark, rollout, planning, and paper
artifacts. The 2026-07-10 post-release audit supersedes their earlier model
interpretation:

- v0.1 and v0.2.1 checkpoints are `legacy_raw_v1`, not evaluations of the
  intended normalized method;
- L2 residual, VEP/calibration, and planning-objective values are invalid;
- cosine values remain reproducible historical outputs but are confounded and
  support neither superiority nor inferiority claims;
- the v0.2.1 Phase 2 KL supplied no gradient to trainable parameters;
- artifact hashes, manifests, pipeline execution, and predictor-only timing
  remain systems evidence within their stated scope;
- fixture smoke outputs are CI evidence, not model results.

Use the corrected paper and Hugging Face model card for current claim
boundaries. Use the old run tree only to reproduce the historical artifacts.
