# 03 — Data model

- Status: Authoritative for v0.1
- Companion RFCs: [RFC-0002](../rfcs/0002-state-encoder-carbon-integration.md),
  [RFC-0003](../rfcs/0003-action-representation-genomic-edits.md),
  [RFC-0006](../rfcs/0006-data-pipeline.md),
  [RFC-0009](../rfcs/0009-surprise-based-pathogenicity-scoring.md),
  [RFC-0011](../rfcs/0011-artifact-provenance-receipts.md)

Every persistent type, schema, and on-disk format that crosses a process
boundary is specified here. Schema versions are part of every artifact;
changes follow the policy in [`09-release-and-versioning.md`](09-release-and-versioning.md).

## In-memory types

### `EditSpec`

```python
@dataclass(frozen=True, slots=True)
class EditSpec:
    chrom: str           # non-empty, e.g., "chr17" or "17"
    pos: int             # 1-based, VCF convention; >= 1
    ref: str             # uppercase ACGT only, len in [1, V1_MAX_LEN]
    alt: str             # uppercase ACGT only, len in [1, V1_MAX_LEN]
    edit_type: EditType  # derived from (len(ref), len(alt))
```

Constants:

- `V1_MAX_LEN = 16` (bp). Edits longer than this raise `UnsupportedEditError`
  with an `edit_type == SV` payload.

Invariants:

- `ref != alt` (else `InvalidEditError`).
- `set(ref) ⊆ {A, C, G, T}`, same for `alt` (else `InvalidEditError`).
- `chrom` is the contig name as it appears in the reference FASTA.
- `pos` is 1-based, matching VCF semantics.

### `RelEdit`

```python
@dataclass(frozen=True, slots=True)
class RelEdit:
    rel_pos: int        # 0-based offset within window, in bp
    edit_type: EditType
    ref_bases: str
    alt_bases: str
```

Invariants:

- `0 <= rel_pos < window_length_bp`.
- `window[rel_pos : rel_pos + len(ref_bases)] == ref_bases.upper()` at
  apply time (else `WindowMismatchError`).

### `TrainingTuple`

```python
@dataclass
class TrainingTuple:
    window_id: str                   # SHA-256 (hex) of the reference window
    rel_edit: RelEdit                # the action
    target_window: str               # the edited window string
    edit_source: Literal["gnomad", "synthetic_snv",
                         "synthetic_indel", "clinvar"]
```

Used internally by the data pipeline; not persisted as a tuple but
emitted in batches via `IterableDataset`.

### `SurpriseResult`

See [`02-public-api.md`](02-public-api.md) for the field list; consumers
treat it as immutable.

## On-disk: window-embedding cache

- **Format:** Parquet shards.
- **Path:** `${GENO_LEWM_CACHE}/embeddings/{encoder_id}/{state_layer}/{pool_type}_{pool_radius}/chr{contig}_{stride_block}.parquet`
- **Compression:** Zstandard, level 9.
- **One row per cached window.**

### Schema (Parquet)

| column          | type             | nullable | description |
|-----------------|------------------|----------|-------------|
| `chrom`         | string           | no       | chromosome / contig |
| `start_bp`      | int64            | no       | inclusive |
| `end_bp`        | int64            | no       | exclusive (`end_bp - start_bp == window_bp`) |
| `window_hash`   | binary(32)       | no       | SHA-256 of the uppercased ACGT string |
| `encoder_hash`  | binary(32)       | no       | SHA-256 of the encoder weights file |
| `state_layer`   | int8             | no       | layer index used |
| `pool_type`     | string           | no       | one of `centered_mean`, `global_mean`, `attention` |
| `pool_radius`   | int32            | no       | pool radius in tokens |
| `dtype`         | string           | no       | one of `bf16`, `fp16`, `fp32` |
| `embedding`     | list<float16>    | no       | the state vector; length == d_state |
| `untargeted`    | bool             | no       | true iff no edit locus was specified |
| `created_at`    | int64            | no       | UTC unix nanoseconds |
| `schema_version`| string           | no       | always `1.0.0` for v0.1 |

The cache is **content-addressed** by
`(window_hash, encoder_hash, state_layer, pool_type, pool_radius, dtype)`.
Changing any of these fields invalidates the cached entry; the cache
loader treats absence under a new key as a cache miss, not an error.

The cache writer never overwrites existing rows; a write that would
duplicate a key is a no-op (post-hash equality check).

### Cache index (SQLite)

A companion SQLite database `${GENO_LEWM_CACHE}/embeddings/index.sqlite`
maps `window_hash` (hex, 64 chars) → (Parquet shard path, row offset).

```sql
CREATE TABLE window_index (
    window_hash TEXT NOT NULL,
    encoder_hash TEXT NOT NULL,
    state_layer INTEGER NOT NULL,
    pool_type TEXT NOT NULL,
    pool_radius INTEGER NOT NULL,
    dtype TEXT NOT NULL,
    shard_path TEXT NOT NULL,
    row_offset INTEGER NOT NULL,
    created_at INTEGER NOT NULL,
    PRIMARY KEY (window_hash, encoder_hash, state_layer, pool_type, pool_radius, dtype)
);
CREATE INDEX idx_shard_path ON window_index(shard_path);
```

The SQLite file is rebuildable from the Parquet shards at any time via
`geno-lewm-cache-windows --reindex`.

## On-disk: gnomAD shard

- **Path:** `${GENO_LEWM_DATA}/gnomad/{release}/variants.parquet`
- **Default release:** `v4.1`.
- **Builder:** `geno-lewm-prepare-gnomad --input-vcf <local.vcf[.gz]> --output ${GENO_LEWM_DATA}`.

| column          | type     | description |
|-----------------|----------|-------------|
| `chrom`         | string   | contig |
| `pos`           | int64    | 1-based |
| `ref`           | string   | uppercase ACGT |
| `alt`           | string   | uppercase ACGT |
| `af_global`     | float32  | global allele frequency |
| `af_afr`        | float32  | African |
| `af_ami`        | float32  | Amish |
| `af_amr`        | float32  | Admixed American |
| `af_asj`        | float32  | Ashkenazi Jewish |
| `af_eas`        | float32  | East Asian |
| `af_fin`        | float32  | Finnish |
| `af_nfe`        | float32  | Non-Finnish European |
| `af_oth`        | float32  | Other |
| `af_sas`        | float32  | South Asian |
| `filter`        | string   | gnomAD VCF FILTER field |
| `schema_version`| string   | always `1.0.0` for v0.1 |

Only variants with `af_global >= 0.01` and `filter == 'PASS'` are included.

## On-disk: ClinVar shard

- **Path:** `${GENO_LEWM_DATA}/clinvar/{release}/variants.parquet`
- **Release** is a date string like `2026-04-15` matching NCBI's release.
- **Builder:** `geno-lewm-prepare-clinvar --input-vcf <local.vcf[.gz]> --release {release} --output ${GENO_LEWM_DATA}`.

| column                   | type    | description |
|--------------------------|---------|-------------|
| `chrom`                  | string  | contig |
| `pos`                    | int64   | 1-based |
| `ref`                    | string  | uppercase ACGT |
| `alt`                    | string  | uppercase ACGT |
| `clinical_significance`  | string  | enum: `P`, `LP`, `LB`, `B`, `VUS`, `OTHER` |
| `review_status`          | string  | ClinVar review status |
| `gene_symbol`            | string  | nullable |
| `clinvar_id`             | int64   | ClinVar variation ID |
| `schema_version`         | string  | always `1.0.0` for v0.1 |

VUS is included for completeness but excluded from eval label sets.

## On-disk: dataset release package

The first paper/demo dataset snapshot is published as one directory:

```
geno-lewm-data-v0.1.0-r1/
├── data_card.md
├── dataset_package.json
├── dataset_manifest.json
├── split_integrity.json
├── SHA256SUMS
├── carbon/
│   └── ...
└── clinvar/
    └── ...
```

`python -m tools.release.dataset_package --dataset-dir DIR --metadata-json dataset_package.json`
generates the normalized metadata file, card, manifest, split-integrity
report, and checksum file from already-built shards. The metadata JSON
records `snapshot_id`, upstream source
revisions, preprocessing commands, split policy, leakage checks,
intended use, limitations, and the relative paths of included files.
`tools.release.paper_package` rejects a dataset package when
`data_card.md` or `dataset_manifest.json` no longer matches
`dataset_package.json`.

The generated `dataset_manifest.json` is the machine-readable source of
truth for release packaging. It contains:

- `schema_version`, fixed at `1.0.0`;
- `snapshot_id`, for example `geno-lewm-data-v0.1.0-r1`;
- `sources`, including upstream names and pinned revisions;
- `splits`, including per-split record counts;
- `files`, each with a relative path, `sha256:<hex>` digest,
  `size_bytes`, and optional split/description metadata.

`split_integrity.json` is generated from the manifest. It recomputes
file hashes/sizes, observes row counts for JSONL, VCF, text, CSV/TSV,
and Parquet files, records observed label/class balance from
`clinical_significance`, `label`, or VCF `CLNSIG` fields, and extracts
comparable keys from JSONL `locus_key`, `variant_key`,
variant-coordinate, or `record_id` rows, from VCF alternate rows, and
from Parquet rows that expose `chrom`, `pos`, `ref`, and `alt` columns.
It also counts genomic regions from JSONL or Parquet `chrom` with
`start_bp`/`end_bp` or `window_start_bp`/`window_end_bp`, plus variant
spans from VCF and variant-coordinate rows, and rejects train/holdout
region intersections when explicit holdout splits are packaged. It
fails when a train/eval pair lacks comparable-key evidence and is not
an explicit train/holdout pair with genomic-region evidence. It
also records `generated_by: tools.release.dataset_integrity`; the
paper/demo verifier rejects reports with a missing or mismatched source
header. The generated `data_card.md` renders the same split-level class
balance from `split_integrity.json`.

## On-disk: calibration table

- **Path inside the checkpoint:** `calibration.parquet`
- **Built by:** `geno-lewm-cache-windows --build-calibration`
- **Schema** ([RFC-0009 §3.4](../rfcs/0009-surprise-based-pathogenicity-scoring.md#34-calibration-distribution)):

| column          | type           | description |
|-----------------|----------------|-------------|
| `bucket_id`     | string         | `{region_class}\|{gc_bin}\|{repeat_class}` |
| `n_calibration` | int64          | number of gnomAD variants in this bucket |
| `cdf`           | list<float32>  | 1001 points: F(σ_raw) at σ-grid quantiles |
| `sigma_grid`    | list<float32>  | the σ_raw grid the CDF is evaluated on |
| `back_off_to`   | string         | parent bucket id if this bucket is sparse; nullable |
| `schema_version`| string         | always `1.0.0` for v0.1 |

Bucket IDs are ASCII pipe-joined labels. Full buckets use
`{region_class}|{gc_bin}|{repeat_class}`. Parent buckets omit the
rightmost factors (`{region_class}|{gc_bin}`, then `{region_class}`),
and the final catch-all bucket is `*`.

The builder consumes pre-scored reference rows (`bucket_id`, `sigma_raw`)
and writes full, parent, and catch-all bucket CDFs. `confidence` and
`low_confidence` are derived at scoring time from the selected bucket's
`n_calibration`; they are not stored as separate Parquet columns.

## On-disk: checkpoint directory

```
geno-lewm-v0.1.0-carbon-500m-r1/
├── manifest.json
├── predictor.safetensors
├── action_encoder.safetensors
├── calibration.parquet
├── train_config.yaml
├── model_package.json
├── eval_metrics.json
├── eval_report.md
├── efficiency_report.json
├── model_card.md
├── SHA256SUMS
├── encoder_hash.txt
├── tokenizer/              # symlink or copy of Carbon's tokenizer
└── lora/                   # Phase 2+ only
    └── carbon_lora.safetensors
```

The `manifest.json` schema is normative and frozen at v0.1; see
[RFC-0011 §3](../rfcs/0011-artifact-provenance-receipts.md#3-manifest).

All weight files use `safetensors`. Canonical serialization for hashing
sorts the state dict by key (UTF-8 lexicographic) before encoding.

Dataset release candidates are prepared by
`python -m tools.release.dataset_snapshot --spec-json ... --dataset-dir ... --overwrite`.
The snapshot spec names the pinned upstream Carbon files plus local
gnomAD and ClinVar VCF/VCF.gz release files. The command stages Carbon
files, builds gnomAD and ClinVar Parquet shards, writes
`dataset_package.json`, and runs the dataset-package generator so the
manifest, data card, split-integrity report, and checksums are generated
from the same local inputs. It also writes
`dataset_snapshot_report.json`, which records the checked spec hash and
upstream source file hashes using public-safe source references rather
than private absolute local paths. `dataset_snapshot_report.json` is
included in `SHA256SUMS`; `tools.release.paper_package` validates its
schema, package paths, generated metadata/manifest/data-card/integrity
hashes/sizes, staged file hashes/sizes/splits, and source references
before accepting a release dataset.

The checked first-experiment training/eval configs live under
`configs/first_experiment/`. `geno-lewm-train --carbon-preflight`
validates the training config with the closed GenoLeWM config schema and
records both the config hash and resolved payload in
`training_preflight_report.json` before a Carbon-backed run can launch.
The report uses public-relative path references plus hashes and sizes so
it can be published with a training run without leaking local filesystem
roots. Release training-run packages include that report in
`training_run_SHA256SUMS`; the paper/demo verifier rejects missing,
stale, non-ok, dataset-mismatched, config-mismatched, or private-path
preflight evidence.

Checkpoint release candidates are prepared by
`python -m tools.release.eval_report`,
`python -m bench.inference --release-efficiency`, and then
`python -m tools.release.model_package`. The eval-report helper renders
`eval_report.md` from packaged measured metrics JSON
(`eval_metrics.json`); the inference benchmark writes measured
`efficiency_report.json` with single-variant latency, batched
throughput, peak memory, command, hardware/runtime notes, samples,
warm-up, limitations, and input identities; metrics conclusions must
explicitly reference every measured metric name before those conclusions
can feed a paper/report draft; the model-package helper
writes normalized `model_package.json`, renders `model_card.md`, and
writes `SHA256SUMS` from `manifest.json` plus release metadata, with
`eval_metrics.json` and `efficiency_report.json` included as packaged
source files. The model and paper/package verifiers cross-check
eval/efficiency release id, dataset snapshot, commit, and model-result
identity against the manifest-backed package. They also re-render the
model card from `model_package.json` plus `manifest.json` and reject
stale output before release-candidate reports pass.
`python -m tools.release.paper_draft` renders the paper/report draft
from the same artifact set, including Artifact Availability entries for
`model_package.json`, `dataset_package.json`,
`dataset_snapshot_report.json`, `eval_metrics.json`,
`eval_config.effective.yaml`, `eval_report.md`,
`efficiency_report.json`, and the terminal demo evidence. The final
paper/demo package verifier checks that normalized
model-package metadata, the model card, generated eval report, packaged
metrics JSON, efficiency report, manifest artifact hashes, checksum file, data card,
dataset package metadata, dataset manifest, paper package, terminal demo
transcript,
`runtime_preflight_report.json`, and `batch_receipt_report.json` are
present and internally consistent; it re-renders the model card and
paper draft from the current artifacts and rejects stale Markdown.
`python -m tools.release.release_candidate`
then emits `release_candidate_report.json`, binding that package result
to the Hub dry-run plan, public artifact URL reachability checks,
commit SHA, model id, dataset snapshot, source metrics JSON, generated
eval report, efficiency report, model package metadata, dataset package
metadata, dataset snapshot report, manifest-backed
predictor/action/calibration/config artifact identities,
`training_preflight_report.json`, `training_run_SHA256SUMS`, Hub
model/dataset/demo upload inventories, provider-backed checks that the
public model, dataset, and demo listings expose the expected files, the
`readiness` checklist, and key artifact hashes before the artifacts are
linked from a public release.

## On-disk: receipt

```
--receipt PATH
```

Canonical JSON: keys sorted lexicographically, no whitespace, UTF-8.
Schema is normative at version `1.0.0`; see
[RFC-0011 §5](../rfcs/0011-artifact-provenance-receipts.md#5-output-receipt).
The v1 receipt commits one score output. Single-variant scoring writes
this file when the caller passes `--receipt PATH`. VCF scoring writes a
JSONL sidecar at `--receipt PATH` with one canonical v1 receipt per
scored alternate. Release demos first write
`runtime_preflight_report.json` from `tools.release.runtime_preflight`;
that report records the model/input artifact identities, required native
runtime dependency probes, backend probes, and the fail-closed network
guard for the terminal command. Release verification rejects preflight
reports generated with fixture/test manifest allowance enabled. Release
demos also write
`batch_receipt_report.json` from `tools.release.batch_receipt_report`;
this aggregate report does not replace per-row receipts, but verifies
the score/receipt row count, artifact hashes, model id, calibration
hash, runtime identity, row ordering, and score-output equality for the
batch. `tools.release.paper_package` also compares the batch report's
model id and calibration hash with the packaged model manifest before a
demo package is accepted.

## Wire formats

- **VCF / VCF.gz** consumed at the CLI boundary. `cyvcf2` is the parser
  (pinned ≥ 0.30 for indexed-VCF iterators).
- **FASTA** consumed for reference genome assemblies. We require the
  index (`.fai`) to be present; if missing, the CLI builds it via `pysam`.
- **23andMe / AncestryDNA / MyHeritage raw data** consumed by the
  desktop runtime; conversion is a local-only step that produces a VCF in
  a tmpdir. These array formats do not include VCF `REF` alleles; the
  converter requires a local reference-allele map keyed by `(chrom, pos)`
  and fails with `VcfParseError` when the reference allele is absent.
  The conversion is documented and tested per format.
- **Sequencing.com WGS JSON** consumed where available; conversion
  supports VCF-equivalent variant rows with explicit reference and
  alternate alleles.

## Schema versioning

- Every on-disk artifact carries a top-level `schema_version` field.
- Schema bumps follow semver; the contract is documented in
  [`09-release-and-versioning.md`](09-release-and-versioning.md).
- Loaders accept any schema with the same MAJOR and ignore unknown
  optional fields; unknown required fields raise `SchemaCompatError`.

## Invariants

| ID | Invariant | Enforced by |
|----|-----------|-------------|
| INV-DATA-1 | EditSpec validates ACGT-only bases at construction | `EditSpec.__post_init__` |
| INV-DATA-2 | Window content is uppercased before hashing for the cache | `encoder/windowing.py::canonicalize` |
| INV-DATA-3 | Cache rows are immutable; no in-place updates | `encoder/cache.py::write_shard` |
| INV-DATA-4 | Manifest hashes are computed over canonical JSON (sorted keys, no whitespace) | `provenance/hashing.py::canonical_json_sha256` |
| INV-DATA-5 | Calibration buckets back off in a fixed order: (region, gc, repeat) → (region, gc) → (region) → (*) | `surprise/context.py::backoff_chain` |
| INV-DATA-5A | Calibration table files match the documented Parquet schema exactly | `surprise/calibration.py::read_calibration_table` |
| INV-DATA-6 | gnomAD variants with `filter != "PASS"` are never used for calibration or training | `data/gnomad.py::filter_passing` |
| INV-DATA-7 | ClinVar VUS rows are loaded but excluded from labelled eval | `data/clinvar.py::label_set` |
| INV-DATA-8 | All datetimes on disk are UTC ISO-8601 with second resolution; durations are integer nanoseconds | linter rule |
| INV-DATA-9 | Receipt JSON is canonical-JSON (sorted keys, no whitespace, UTF-8) | `provenance/receipt.py::write` |
| INV-DATA-10 | Cache reads never write back; cache writes never overwrite | both `encoder/cache.py` paths |

## Open questions

| ID | Question | Owner | Target |
|----|----------|-------|--------|
| OQ-DATA-1 | Whether to add a `phase` field to TrainingTuple for haplotype tuples vs single-edit tuples | core | v0.2 |
| OQ-DATA-2 | Whether calibration tables should also store per-bucket bootstrap std for confidence-aware downstream use | core | v0.2 |
| OQ-DATA-3 | Whether gnomAD/ClinVar Parquet shards should be split by chromosome for selective loading | core | when corpus exceeds 50 GB |
