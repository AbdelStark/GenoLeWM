# 03 — Data model

- Status: Authoritative for v0.1
- Companion RFCs: [RFC-0002](../rfcs/0002-state-encoder-carbon-integration.md),
  [RFC-0003](../rfcs/0003-action-representation-genomic-edits.md),
  [RFC-0006](../rfcs/0006-data-pipeline.md),
  [RFC-0009](../rfcs/0009-surprise-based-pathogenicity-scoring.md),
  [RFC-0011](../rfcs/0011-verifiable-inference-attestation.md)

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
├── eval_report.md
├── encoder_hash.txt
├── tokenizer/              # symlink or copy of Carbon's tokenizer
└── lora/                   # Phase 2+ only
    └── carbon_lora.safetensors
```

The `manifest.json` schema is normative and frozen at v0.1; see
[RFC-0011 §3.7](../rfcs/0011-verifiable-inference-attestation.md#37-manifest-schema).

All weight files use `safetensors`. Canonical serialization for hashing
sorts the state dict by key (UTF-8 lexicographic) before encoding.

## On-disk: receipt

```
{output_path}.receipt.json
```

Canonical JSON: keys sorted lexicographically, no whitespace, UTF-8.
Schema is normative at version `1.0.0`; see
[RFC-0011 §3.3](../rfcs/0011-verifiable-inference-attestation.md#33-output-receipt).

## Wire formats

- **VCF / VCF.gz** consumed at the CLI boundary. `cyvcf2` is the parser
  (pinned ≥ 0.30 for indexed-VCF iterators).
- **FASTA** consumed for reference genome assemblies. We require the
  index (`.fai`) to be present; if missing, the CLI builds it via `pysam`.
- **23andMe / AncestryDNA / MyHeritage raw data** consumed by the
  desktop runtime; conversion is a local-only step that produces a VCF in
  a tmpdir. The conversion is documented and tested per format.
- **Sequencing.com WGS JSON** consumed where available; conversion to
  VCF follows the format's public schema.

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
| INV-DATA-4 | Manifest hashes are computed over canonical JSON (sorted keys, no whitespace) | `attestation/hashing.py::canonical_json_sha256` |
| INV-DATA-5 | Calibration buckets back off in a fixed order: (region, gc, repeat) → (region, gc) → (region) → (*) | `surprise/context.py::backoff_chain` |
| INV-DATA-5A | Calibration table files match the documented Parquet schema exactly | `surprise/calibration.py::read_calibration_table` |
| INV-DATA-6 | gnomAD variants with `filter != "PASS"` are never used for calibration or training | `data/gnomad.py::filter_passing` |
| INV-DATA-7 | ClinVar VUS rows are loaded but excluded from labelled eval | `data/clinvar.py::label_set` |
| INV-DATA-8 | All datetimes on disk are UTC ISO-8601 with second resolution; durations are integer nanoseconds | linter rule |
| INV-DATA-9 | Receipt JSON is canonical-JSON (sorted keys, no whitespace, UTF-8) | `attestation/receipt.py::write` |
| INV-DATA-10 | Cache reads never write back; cache writes never overwrite | both `encoder/cache.py` paths |

## Open questions

| ID | Question | Owner | Target |
|----|----------|-------|--------|
| OQ-DATA-1 | Whether to add a `phase` field to TrainingTuple for haplotype tuples vs single-edit tuples | core | v0.2 |
| OQ-DATA-2 | Whether calibration tables should also store per-bucket bootstrap std for confidence-aware downstream use | core | v0.2 |
| OQ-DATA-3 | Whether gnomAD/ClinVar Parquet shards should be split by chromosome for selective loading | core | when corpus exceeds 50 GB |
