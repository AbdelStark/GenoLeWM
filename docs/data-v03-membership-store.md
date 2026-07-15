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

The manifest binds a closed lineage evidence profile. Real artifacts report
`official`; reduced test-only artifacts report `synthetic_fixture`. Both remain
fully verifiable, but verification summaries expose the profile so downstream
automation cannot mistake fixture mechanics for production evidence.

## Build flow

The builder accepts no arbitrary row stream. Its source specification must name
all 22 local gnomAD Parquet shards and the local ClinVar Parquet shard. Each
file is matched to the immutable repository, revision, namespace, artifact
path, SHA-256, size, row count, schema, and remote-postflight evidence already
recorded in `snapshot-lineage.json`. For a non-fixture build, the official
snapshot-lineage verifier captures that file once, verifies its closed semantic
contract (including every gnomAD and ClinVar postflight Parquet audit), and
returns the exact captured payload and identity with deeply immutable parsed
semantics. Those postflight summaries include normalized identities for every
verified remote file and reconcile the receipt/audit and Parquet identities.
The membership builder derives source bindings only from that verified return
value and persists those same captured bytes; it never reopens the lineage
path. The reduced synthetic-fixture path exists only for contract tests and is
not publication evidence.

The read-only lineage capture and semantic verifier are part of the installed
package and have no dependency on the repository's operational `tools` tree.
The lineage assembly CLI delegates to the same implementation. Default
`MembershipStore.open(..., verify=True)` behavior therefore remains available
from a wheel install while preserving duplicate-key rejection, single-read byte
identity, deep immutability, and the complete non-fixture semantic contract.

```text
exact snapshot lineage
        +
23 local staged Parquet files
        │
        ├── single-descriptor private source capture + exact byte/schema/count checks
        ├── source-specific canonical row derivation
        ├── disk-backed duplicate and cross-role rejection
        ├── numeric-chromosome canonical ordering
        ├── independent single-read private-snapshot verification
        └── file/directory fsync + atomic no-clobber publication
```

Normalized ClinVar B/LB/LP/P rows are labeled memberships. They follow exactly
the same chromosome split as gnomAD: train is chromosomes 1–19 and 22,
validation is chromosome 20, and evaluation is chromosome 21. In particular,
train-chromosome P/LP rows remain usable training anchors; being a ClinVar
membership does not make a train row a holdout. VUS/OTHER, non-primary contigs,
X, Y, and MT are excluded from labeled membership while their raw artifact
identity and filtered counts remain bound in source provenance.

Raw ClinVar assertion rows remain in Parquet and SQLite. A canonical variant
with both benign-family and pathogenic-family assertions is rejected. Duplicate
same-target assertions are retained for provenance but collapse to one benchmark
variant, deterministically preferring P over LP and B over LB, then the smallest
source-row identity. `clinvar_class_role_counts` therefore reports unique
labeled variants; `source_role_counts` reports each exact source ID by role, and
`source_kind_role_counts["clinvar"]` reports the aggregate raw retained
assertions. The row reason bits are `1` for gnomAD source membership and `2` for
ClinVar labeled membership. Every output role is non-empty and every row role
must match its chromosome assignment.

These are variant memberships, not phased haplotypes. The pinned inputs contain
no phased haplotype membership, so the store records that capability as
unavailable rather than synthesizing proximity-based or pseudo-phased labels.

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
export GENO_LEWM_VERIFIED_BUILD_CONTAINER_IMAGE='ghcr.io/abdelstark/geno-lewm@sha256:<64-hex-digest>'

uv run python -m tools.data.v03_membership_store build \
  --spec-json /path/to/membership-build.json \
  --output-dir /path/to/geno-lewm-data-v0.3.0-membership-r1

uv run python -m tools.data.v03_membership_store verify \
  --store-dir /path/to/geno-lewm-data-v0.3.0-membership-r1
```

The declared builder commit must equal `git rev-parse HEAD`, and the checkout
must remain clean at both provenance gates. The container value
must equal `GENO_LEWM_VERIFIED_BUILD_CONTAINER_IMAGE`, injected by the trusted
container launcher; an installed checkout without verifiable Git state or a
missing/mismatched container binding fails before ingestion. The receipt records
both verification mechanisms. This is an invocation binding, not a claim that
an environment variable is a cryptographic container attestation.

Build and full verification require PyArrow from the `dev` or `train` extra.
Runtime lookup uses only SQLite from the Python standard library.

## Identity and verification

The semantic `content_identity` covers:

- assembly and chromosome roles;
- exact snapshot-lineage bytes and lineage identity;
- every source revision, namespace, artifact and verification binding;
- source, split, row, distinct-variant, filtered, source×role, and unique
  ClinVar-class×role counts;
- a SHA-256 over length-framed canonical rows in numeric chromosome order.

The semantic identity deliberately excludes container-format and build-runtime
bytes so the same row contract is not redefined by a PyArrow or SQLite encoding
detail. The separate `physical_identity` covers the semantic identity plus the
SHA-256 and size of Parquet, SQLite, bundled lineage, and build-receipt bytes.
The verifier opens each published file once, copies it through that checked
descriptor into private read-only storage, and hashes and parses only that
snapshot. It rejects extra files, directories, symlinks, and duplicate file
bindings; validates bundled lineage and receipt provenance; scans Parquet and
SQLite independently; reconciles every R-tree entry and both cross-tabs; and
requires both semantic scans to match the manifest. Scalar SQLite tables are
`STRICT` with closed type, range, role, source, and ClinVar-label checks.

## Consumption

`MembershipStore.open(path)` verifies once and opens `lookup.sqlite` read-only.
Point queries use a composite SQLite index. Interval queries constrain the
integer R-tree first, then join membership rows by integer primary key while
applying role/source filters. `iter_role(...)` streams raw membership assertions;
`iter_labeled_clinvar(...)` streams the deterministic unique labeled-variant
benchmark unit in bounded batches. The
`MembershipStoreHoldoutPolicy` adapter can be passed where the existing tuple
builder accepts `HoldoutPolicy`; it excludes validation/evaluation chromosomes
only. Train-role ClinVar B/LB/LP/P memberships remain data memberships and never
become holdouts. Source-filtered lookups remain indexed and never construct a
million-key Python set. Unplaced windows and X, Y, or MT windows fail closed
because they are outside the autosomal role contract.

Store and holdout-policy equality/hash are based on `content_identity`. Pickle
contains no live SQLite connection; each process and thread lazily opens its own
immutable read-only handle, and a post-fork PID guard discards inherited handle
state before lookup. Connections owned by short-lived threads are weakly
reclaimed when those thread objects expire; `close()` deterministically closes
all remaining live-thread connections. A verified open retains a private
content-bound SQLite snapshot, so replacement of the published path after open
cannot change later thread/process queries.

Opening with `verify=False` skips the expensive private capture and full file
scan, checks the exact layout and manifest immediately, then checks SQLite
schema/metadata on first lookup. It does not defend against later path
replacement. Use that mode only inside an externally enforced immutable-storage
boundary after retaining a trusted verification result for the exact identity.

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
