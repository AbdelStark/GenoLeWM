# v0.3 membership-store contract

The scalable membership store is the production-size counterpart to the
fixture-oriented in-memory `MembershipArtifact`. It is designed for the full
autosomal gnomAD membership plus the pinned ClinVar source without constructing
a Python tuple or set containing every row or variant key.

This repository currently ships the contract and synthetic tests. It does not
ship a real v0.3 membership store, prove a real split, or define the phased
haplotype holdout described by historical RFC-0006. A proximity window over
unphased variants is not a substitute for phased haplotype membership.

## Artifact layout

One immutable store directory contains exactly:

| File | Purpose |
| --- | --- |
| `manifest.json` | Closed semantic contract, exact source lineage, counts, row digest, and file identities |
| `memberships.parquet` | Portable canonically ordered membership rows |
| `lookup.sqlite` | Read-only point index, integer R-tree interval index, and role iterator |
| `snapshot-lineage.json` | Exact lineage bytes consumed by the builder; self-contained source provenance |
| `build-receipt.json` | Closed builder commit, digest-pinned container, package, runtime, and creation record |

The manifest schema is
`configs/data_v03/membership-store.schema.json`. Build inputs use
`configs/data_v03/membership-build-spec.schema.json`; the physical build receipt
uses `configs/data_v03/membership-build-receipt.schema.json`. All three reject
unknown fields.

## Build flow

The builder accepts no arbitrary row stream. Its source specification must name
all 22 local gnomAD Parquet shards and the local ClinVar Parquet shard. Each
file is matched to the immutable repository, revision, namespace, artifact
path, SHA-256, size, row count, schema, and remote-postflight evidence already
recorded in `snapshot-lineage.json`.

```text
exact snapshot lineage
        +
23 local staged Parquet files
        │
        ├── single-descriptor private source capture + exact byte/schema/count checks
        ├── source-specific canonical row derivation
        ├── disk-backed duplicate and cross-role rejection
        ├── numeric-chromosome canonical ordering
        ├── independent verification of the complete temporary store
        └── file/directory fsync + atomic publication
```

Only normalized ClinVar P/LP rows become known-pathogenic holdout memberships;
other classes and rows outside primary autosomes are deterministically counted
as filtered. The v0.3 chromosome-role contract covers autosomes only. gnomAD
shards must cover chromosomes 1 through 22 exactly, but those variant rows are
source/split memberships, not a phased-haplotype claim. The row reason bits are
`1` for gnomAD source membership and `2` for ClinVar P/LP membership. Every
output role must be non-empty, every row role must match its chromosome
assignment, and one canonical variant may not appear in multiple roles.

An abbreviated build spec looks like this; a real spec must contain all 23
entries and no ellipsis:

```json
{
  "$schema": "./membership-build-spec.schema.json",
  "schema_version": "geno-lewm.membership-build-spec.v1",
  "artifact_id": "geno-lewm-data-v0.3.0-membership-r1",
  "snapshot_lineage": "snapshot-lineage.json",
  "snapshot_lineage_sha256": "sha256:0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef",
  "builder": {
    "git_commit": "0123456789abcdef0123456789abcdef01234567",
    "container_image": "ghcr.io/abdelstark/geno-lewm@sha256:0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef"
  },
  "sources": [
    {
      "kind": "gnomad",
      "chromosome": "1",
      "path": "sources/gnomad-chr1.parquet"
    },
    {
      "kind": "clinvar",
      "path": "sources/clinvar.parquet"
    }
  ]
}
```

Build and independently verify it with:

```bash
uv run python -m tools.data.v03_membership_store build \
  --spec-json /path/to/membership-build.json \
  --output-dir /path/to/geno-lewm-data-v0.3.0-membership-r1

uv run python -m tools.data.v03_membership_store verify \
  --store-dir /path/to/geno-lewm-data-v0.3.0-membership-r1
```

Build and full verification require PyArrow from the `dev` or `train` extra.
Runtime lookup uses only SQLite from the Python standard library.

## Identity and verification

The semantic `content_identity` covers:

- assembly and chromosome roles;
- exact snapshot-lineage bytes and lineage identity;
- every source revision, namespace, artifact and verification binding;
- source, split, row, distinct-variant, and filtered counts;
- a SHA-256 over length-framed canonical rows in numeric chromosome order.

The semantic identity deliberately excludes container-format and build-runtime
bytes so the same row contract is not redefined by a PyArrow or SQLite encoding
detail. The separate `physical_identity` covers the semantic identity plus the
SHA-256 and size of Parquet, SQLite, bundled lineage, and build-receipt bytes.
The verifier rejects extra files, directories, symlinks, and duplicate file
bindings; validates bundled lineage and receipt provenance; scans Parquet and
SQLite independently; reconciles every R-tree entry; and requires both semantic
scans to match the manifest.

## Consumption

`MembershipStore.open(path)` verifies once and opens `lookup.sqlite` read-only.
Point queries use a composite SQLite index. Interval queries constrain the
integer R-tree first, then join membership rows by integer primary key while
applying role/source filters. `iter_role("validation")` and
`iter_role("evaluation")` stream rows in bounded batches. The
`MembershipStoreHoldoutPolicy` adapter can be passed where the existing tuple
builder accepts `HoldoutPolicy`; it excludes validation/evaluation chromosomes
and training-chromosome windows intersecting ClinVar P/LP membership before
tuples reach training. Source-filtered interval and edit lookups remain indexed
and never construct a million-key Python set. Ordinary gnomAD variant
membership does not trigger a haplotype exclusion. Unplaced windows and X, Y,
or MT windows fail closed because they are outside the autosomal role contract.

Store and holdout-policy equality/hash are based on `content_identity`. Pickle
contains no live SQLite connection; each process and thread lazily opens its own
immutable read-only handle, and a post-fork PID guard discards inherited handle
state before lookup.

Opening with `verify=False` skips the expensive full file scan and checks the
exact layout and manifest immediately, then SQLite schema/metadata on the first lookup. Use that mode only after one trusted
preflight has retained the successful full-verification result for the exact
`content_identity`.

## Validation

```bash
uv run pytest tests/unit/test_data_membership_store.py -q
uv run ruff check geno_lewm/data/_membership_store_*.py \
  geno_lewm/data/membership_store.py \
  tools/data/v03_membership_store.py tests/unit/test_data_membership_store.py
uv run mypy --strict geno_lewm/data/_membership_store_*.py \
  geno_lewm/data/membership_store.py \
  tools/data/v03_membership_store.py
uv run python tools/api/snapshot.py check
uv run --extra docs mkdocs build --strict
```
