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
- `tools/data/v03_snapshot_lineage.py` for fail-closed assembly.

Both schemas use JSON Schema Draft 2020-12 and reject unknown fields. The
lineage schema fixes `membership_status` to `not_created` and has no membership
property.

## Prerequisites

Assembly requires evidence that is not committed to this repository:

- exactly 22 gnomAD v4.1 staging receipts, one for each autosome;
- exactly 22 gnomAD remote-postflight reports, each produced against an
  immutable 40-character Hugging Face dataset revision;
- one corrected ClinVar `2026-04-15` GRCh38 audit;
- the checked gnomAD source lock in
  `configs/data_v03/gnomad-v4.1-exomes-autosomes.source-lock.json`.

The repository therefore provides the contract and fixture tests, not a
ready-to-assemble production lineage. Do not substitute fixture evidence for
live staging evidence.

## Produce each remote postflight

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
full Parquet scan. A successful report is still evidence about one staging
namespace; it is not a snapshot-membership record.

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
    "namespace": "staging/clinvar-2026-04-15-corrected-r1",
    "audit_file": "clinvar-audit.json",
    "audit_sha256": "sha256:<64 lowercase hex characters>"
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
bytes.

For every gnomAD shard, assembly cross-checks the postflight's exact repository,
revision, namespace, source commit, and chromosome; its exact nine-file
namespace inventory; the local receipt SHA-256 and size; the receipt's Parquet
SHA-256 and size; and type-strict equality between the verifier's fresh Parquet
audit and the local receipt audit. Any mismatch aborts before output is written.

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
uv run pytest tests/unit/test_data_v03_snapshot_lineage.py -q
uv run ruff check tools/data/v03_snapshot_lineage.py \
  tests/unit/test_data_v03_snapshot_lineage.py
uv run mypy --strict tools/data/v03_snapshot_lineage.py
uv run --extra docs mkdocs build --strict
```

Changes to either public schema ID, the required namespace inventory, the
data-use binding, or membership semantics require matching tests,
architecture documentation, and a changelog entry.
