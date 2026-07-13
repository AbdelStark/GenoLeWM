# v0.3 snapshot-lineage assembly

The v0.3 snapshot-lineage assembler turns already-produced staging evidence
into one immutable, content-addressed lineage record. It is an offline evidence
reconciler. It does not upload data, contact Hugging Face, create dataset
memberships, or claim that a publishable train/validation/evaluation snapshot
exists.

The public contracts are:

- `configs/data_v03/snapshot-lineage-spec.schema.json` for the local evidence
  bundle specification;
- `configs/data_v03/snapshot-lineage.schema.json` for the assembled lineage;
- `tools/data/v03_snapshot_lineage.py` for fail-closed assembly and content-ID
  verification.

Both schemas use JSON Schema Draft 2020-12 and reject unknown fields. The
lineage schema fixes `membership_status` to `not_created` and has no membership
property. Both schemas require exactly one shard for each autosome and bind
chromosomes 20 and 21 to validation and evaluation, respectively; all other
autosomes are training shards. Schema validation alone does **not** recompute
or verify `lineage_id`. Consumers must run the verifier described below.

## Prerequisites

Assembly requires evidence that is not committed to this repository:

- exactly 22 gnomAD v4.1 staging receipts, one for each autosome;
- exactly 22 gnomAD remote-postflight reports, each produced against an
  immutable 40-character Hugging Face dataset revision;
- one corrected ClinVar `2026-04-15` GRCh38 audit;
- one ClinVar remote-postflight report for that corrected shard, produced
  against its immutable 40-character Hugging Face dataset revision;
- the checked gnomAD source lock in
  `configs/data_v03/gnomad-v4.1-exomes-autosomes.source-lock.json`.

The repository therefore provides the contract and fixture tests, not a
ready-to-assemble production lineage. Do not substitute fixture evidence for
live staging evidence.

## Produce the remote postflights

### gnomAD autosomes

After one gnomAD namespace has been published by the staging workflow, verify
the complete namespace at its immutable Hub revision:

```bash
uv run python -m tools.data.v03_gnomad_lock remote-postflight \
  --repo-id abdelstark/geno-lewm-data \
  --revision "$HUB_REVISION" \
  --namespace "$STAGING_NAMESPACE" \
  --expected-source-commit "$SOURCE_COMMIT" \
  --expected-chromosome "$CHROMOSOME" \
  --output-json "evidence/postflights/chr${CHROMOSOME}.json"
```

Repeat this for chromosomes 1 through 22. The verifier resolves the exact Hub
revision, downloads the complete expected namespace, recomputes every file
identity, checks the receipt and source-lock evidence, and performs a fresh
full Parquet scan. Every JSON artifact is captured once, and the Parquet hash,
identity reconciliation, and scan all use one private byte snapshot opened as
a non-symlink regular file. A successful report is still evidence about one staging
namespace; it is not a snapshot-membership record.

### ClinVar corrected shard

Verify the corrected ClinVar namespace at its exact Hub revision. The producer
commit must be present in the local Git object database because the verifier
derives the trusted schema and transform contract from those immutable bytes:

```bash
git fetch origin "$CLINVAR_SOURCE_COMMIT"
uv run --extra dev --with huggingface-hub \
  python -m tools.data.v03_clinvar_postflight \
  --repo-id abdelstark/geno-lewm-data \
  --revision "$CLINVAR_HUB_REVISION" \
  --namespace "$CLINVAR_STAGING_NAMESPACE" \
  --expected-source-commit "$CLINVAR_SOURCE_COMMIT" \
  --expected-release 2026-04-15 \
  --output-json evidence/clinvar-remote-postflight.json
```

The expected namespace is
`staging/clinvar-2026-04-15-archive-${CLINVAR_SOURCE_COMMIT:0:12}-r1`.
The verifier requires exactly the release Parquet, audit, prepare report, and
runtime report; reconciles them; and independently full-scans the Parquet.
See the [ClinVar staging-verification guide](data-v03-staging-verification.md)
for the source-byte and scientific claim boundaries.

## Author the assembly spec

Each gnomAD shard entry must bind both local evidence files by relative path
and prefixed SHA-256. This is an abbreviated structural example; the literal
spec must contain all 22 entries and cannot contain the ellipsis:

```json
{
  "$schema": "./snapshot-lineage-spec.schema.json",
  "schema_version": "geno-lewm.v03-snapshot-lineage-spec.v1",
  "candidate_snapshot_id": "geno-lewm-data-v0.3.0-r1",
  "reference_genome": "GRCh38",
  "gnomad": {
    "repo": "abdelstark/geno-lewm-data",
    "repo_type": "dataset",
    "shards": [
      {
        "chromosome": "1",
        "split_role": "train",
        "revision": "<immutable 40-character Hub commit>",
        "namespace": "<exact staging namespace>",
        "receipt_file": "receipts/chr1.json",
        "receipt_sha256": "sha256:<64 lowercase hex characters>",
        "postflight_file": "postflights/chr1.json",
        "postflight_sha256": "sha256:<64 lowercase hex characters>"
      }
    ]
  },
  "clinvar": {
    "repo": "abdelstark/geno-lewm-data",
    "repo_type": "dataset",
    "revision": "<immutable 40-character Hub commit>",
    "namespace": "staging/clinvar-2026-04-15-archive-<source commit prefix>-r1",
    "audit_file": "clinvar-audit.json",
    "audit_sha256": "sha256:<64 lowercase hex characters>",
    "postflight_file": "clinvar-remote-postflight.json",
    "postflight_sha256": "sha256:<64 lowercase hex characters>"
  }
}
```

Paths must stay inside the spec directory. Revisions must be full, non-zero,
lowercase commit hashes. Compute evidence hashes from the final bytes; changing
whitespace changes the hash.

## Assemble

```bash
uv run python -m tools.data.v03_snapshot_lineage assemble \
  --spec-json /path/to/evidence/lineage-spec.json \
  --gnomad-source-lock-json \
    configs/data_v03/gnomad-v4.1-exomes-autosomes.source-lock.json \
  --output-json /path/to/evidence/snapshot-lineage.json
```

Success writes the lineage once and prints a compact JSON summary containing
`lineage_id`, `candidate_snapshot_id`, and
`"membership_status":"not_created"`. Re-running with identical bytes is
idempotent. The assembler refuses to replace an existing output with different
bytes. Publication uses a same-directory unique staging file, file and
directory `fsync`, and an atomic no-clobber link. Competing writers can only
observe the winning bytes; a symlink or other non-regular output is rejected.

For every gnomAD shard, assembly cross-checks the postflight's exact repository,
revision, namespace, source commit, and chromosome; its exact nine-file
namespace inventory; the local receipt SHA-256 and size; the receipt's Parquet
SHA-256 and size; and type-strict equality between the verifier's fresh Parquet
audit and the local receipt audit. Any mismatch aborts before output is written.
The fresh full-scan audit is persisted verbatim under each shard's
`remote_postflight.parquet_audit`, so the lineage retains the independently
recomputed scientific evidence rather than only a report digest.

For ClinVar, assembly first performs the existing closed audit validation, then
type-strictly reconciles the postflight's exact 16-key surface: repository and
type, Hub revision, namespace, producer commit, release, claim boundary, four
verified files, nine completed checks, and every nested identity. The audit
file identity must equal the local audit bytes. The Parquet identity, records,
and class balance must equal the audit output. Metadata and scanned row counts,
chromosome totals, schema-version balance, nine-field schema, null counts, and
positive value ranges are checked again before the recomputed audit and report
identity are persisted under `clinvar.remote_postflight`.

The ClinVar source archive is not one of the four remote namespace files. Its
release, SHA-256, and size are receipt-reconciled; its MD5 and URL remain
receipt-only fields and are not represented as independently recomputed source
bytes. The lineage preserves this limitation verbatim.

## Verify an existing lineage

Run the semantic verifier before consuming or promoting a lineage candidate:

```bash
uv run python -m tools.data.v03_snapshot_lineage verify \
  --lineage-json /path/to/evidence/snapshot-lineage.json
```

The verifier captures the JSON bytes once, rejects duplicate keys and unknown
fields throughout the closed artifact, removes `lineage_id`, recomputes the
canonical SHA-256 commitment, and compares it type-strictly with the recorded
ID. It also requires `membership_status: not_created`, the exact conservative
claim boundary, exactly one shard per autosome, the fixed chromosome-to-role
policy in canonical numeric order, common source/transform/execution bindings,
the exact archived ClinVar URL, exact artifact paths,
gnomAD record and byte totals, and the complete gnomAD and ClinVar postflight
audit invariants. The persisted remote file-identity maps are cross-checked
against each gnomAD receipt and Parquet output and the ClinVar audit and Parquet
output. A recomputed ID cannot legitimize internally inconsistent records,
balances, totals, or evidence identities. Success prints a machine-readable
JSON summary and does not modify the lineage.

Python consumers that need to bind or persist the exact verified file must use
the single-capture API and must not reopen the path after verification:

```python
from pathlib import Path

from tools.data.v03_snapshot_lineage import capture_verified_snapshot_lineage

verified = capture_verified_snapshot_lineage(Path("snapshot-lineage.json"))
lineage = verified.lineage
exact_bytes = verified.payload
exact_sha256 = verified.payload_sha256
exact_size_bytes = verified.size_bytes
```

The returned payload, SHA-256, size, and deeply immutable parsed mapping all
come from the same read. Nested objects are read-only mappings and nested arrays
are tuples. This prevents a verifier/consumer path-replacement race, prevents a
consumer from mutating verified semantics away from the committed bytes, and
avoids reconstructing bytes from a parsed JSON value.

## Upstream data-use boundary

The assembled lineage records the following source-specific `data_use`
objects. The check date is part of the versioned lineage contract; operators
must review the current upstream pages before redistribution or a materially
new use.

| Source bound by this lineage | License or reuse status | Attribution and restrictions |
| --- | --- | --- |
| gnomAD v4.1 exomes primary data | [gnomAD policies](https://gnomad.broadinstitute.org/policies) place primary data under CC0-1.0. The current transform materializes variant coordinates/alleles, PASS status, global and population allele frequencies, and the GenoLeWM schema version. | gnomAD requests citation of its [flagship paper](https://doi.org/10.1038/s41586-023-06045-0) and a browser link. Do not attempt participant reidentification. Third-party annotations can have separate licenses and require a new review before inclusion or redistribution. |
| ClinVar GRCh38 archived VCF, release 2026-04-15 | [ClinVar's data-use page](https://www.ncbi.nlm.nih.gov/clinvar/docs/maintenance_use/) says public data are freely available for use. [NCBI policy](https://www.ncbi.nlm.nih.gov/home/about/policies/) places no restrictions on molecular-data use or distribution, but NCBI does not receive or transfer rights that a submitter or source country may claim; the lineage therefore records SPDX `NOASSERTION`. | Attribute ClinVar and cite a current ClinVar publication. ClinVar is not intended for direct diagnosis or medical decision-making without genetics-professional review, and NCBI does not independently verify submitted assertions. |

GenoLeWM's Apache-2.0 license covers project source and project-authored
metadata. It does not replace upstream data terms or grant rights that the
project does not hold.

## Validation for contract changes

```bash
uv run --extra dev pytest tests/unit/test_data_v03_snapshot_lineage.py -q
uv run --extra dev ruff check tools/data/v03_snapshot_lineage.py \
  tests/unit/test_data_v03_snapshot_lineage.py
uv run --extra dev mypy --strict tools/data/v03_snapshot_lineage.py
uv run --extra docs mkdocs build --strict
```

Changes to either public schema ID, the required namespace inventory, the
data-use binding, or membership semantics require matching tests,
architecture documentation, and a changelog entry.
