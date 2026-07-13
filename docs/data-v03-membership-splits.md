# v0.3 membership split evidence

`tools/data/v03_membership_splits.py` exports the deterministic ClinVar
validation and evaluation streams from one verified membership store and audits
one exact placed-window training artifact against those held roles. The result
is an atomic, checksum-closed evidence directory; it is not a dataset package
or a training-runtime integration.

## Published membership prerequisite

The first real v0.3 variant-membership candidate is published in
`abdelstark/geno-lewm-data` at exact Hub commit
[`96e97a7ffe1e9ad8f9a98f690b220a32ac75ddc2`](https://huggingface.co/datasets/abdelstark/geno-lewm-data/tree/96e97a7ffe1e9ad8f9a98f690b220a32ac75ddc2/candidates/v0.3/geno-lewm-data-v0.3.0-r1/membership/geno-lewm-v03-membership-fd7f4bbde476-r1/success).
The successful Hugging Face Job built from GenoLeWM commit
`fd7f4bbde476a1608b6dec236e65be9ea6568342`, then downloaded and reverified
the exact returned Hub revision. A separate independent download verified the
15-file inventory, all `SHA256SUMS` entries, and the full store semantics.

| Binding | Exact value |
| --- | --- |
| Membership artifact | `geno-lewm-data-v0.3.0-membership-r1` |
| Membership Hub path | `candidates/v0.3/geno-lewm-data-v0.3.0-r1/membership/geno-lewm-v03-membership-fd7f4bbde476-r1/success` |
| Content identity | `sha256:7fa661eefacf70258b8392aff88a6faea2749c812680d4a2bfc41376d061ff7a` |
| Physical identity | `sha256:d7ea2c4b8413768c9128c70a299a11f4adf35140102778a71cf56e69fb4db536` |
| Canonical rowset | `sha256:d268f2e2b67cce56c5d5099ec1ddcbd810fbb5973e6c96a929fd2c99fbd25f68` |
| Membership rows | `2,335,042` |
| Distinct variants | `2,259,268` |
| Roles | train `2,251,087`; validation `53,002`; evaluation `30,953` |
| Lineage profile | `official` |

This is a real deterministic **variant-membership candidate**. It is not a
released v0.3 snapshot, phased-haplotype membership, evidence of dataset
representativeness, a model or benchmark result, or a clinical-validity claim.

## Exact placed-window input

The current train-window prerequisite is the GRCh38 chr22 artifact at the same
immutable dataset revision:

| Binding | Exact value |
| --- | --- |
| Repository | `abdelstark/geno-lewm-data` |
| Revision | `96e97a7ffe1e9ad8f9a98f690b220a32ac75ddc2` |
| Artifact path | `placed/gnomad-common-windows.jsonl` |
| SHA-256 | `sha256:ec76046771a163fbc22f326df26e2a332767eaa045dd919718c1cf86c4fbe0ac` |
| Size | `4,186,560` bytes |
| Records | `976` |
| Dataset manifest | `dataset_manifest.json`; `sha256:c3aa8f22b79e76fa5b6e3a43e02675cfc02d56dc7dc9fa36128c81874537016c`; `5,051` bytes |
| Dataset snapshot ID | `geno-lewm-data-v0.1.0-r1` |
| Manifest split | `train_placed_gnomad_common` |
| Assembly and role | GRCh38, chromosome 22, train |

Each JSONL object must contain exactly these eight fields:

```text
record_id, source, variant_source, chrom,
start_bp, end_bp, sequence, variant_count
```

The placed windows come from the older released `geno-lewm-data-v0.1.0-r1`
package; consuming them here does not relabel that package as v0.3. The tool
captures its exact dataset manifest and requires the declared snapshot ID. It
then cross-checks the window path, SHA-256, byte size, record count, and
`train_placed_gnomad_common` split against that manifest. The file is documented
as GRCh38 by the published package; the tool records that assembly but does not
infer it from sequence bytes. The loader rejects duplicate record IDs,
duplicate intervals, non-canonical DNA, non-positive variant counts, and rows
where `end_bp - start_bp != len(sequence)`. Every row must be placed on a
GRCh38 train chromosome. Unplaced or unassigned rows, chromosomes 20 and 21,
and identity drift fail before publication. Carbon corpus windows without
absolute coordinates cannot substitute for this artifact.

## Exported streams

`MembershipStore.iter_labeled_clinvar(...)` produces canonically ordered,
deduplicated labeled variants. Each JSONL row uses the existing evaluator
contract:

```json
{
  "alt": "G",
  "chrom": "20",
  "clinical_significance": "P",
  "pos": 12345,
  "ref": "A"
}
```

The matching VCF contains exactly the same ordered unique
`(chrom, pos, ref, alt)` keys. The report records each file's path, SHA-256, and
byte size; the stream's unique-key digest; and its B/LB/LP/P and binary counts.
The exporter reconciles these counts with the verified store manifest.

| Role | Chromosome | Total | Positive P/LP | Negative B/LB |
| --- | ---: | ---: | ---: | ---: |
| Validation | 20 | 34,657 | 5,051 | 29,606 |
| Evaluation | 21 | 20,653 | 3,407 | 17,246 |

Those are pinned expectations derived from the exact membership candidate.
Different totals are an error, not a new result to publish silently.

## Nonintersection audits

The authoritative audit examines every verified placed window twice:

1. `MembershipStoreHoldoutPolicy` must accept the window as placed train data.
2. `MembershipStore.overlaps_interval(...)` must find zero validation or
   evaluation memberships in the window interval.

The report records the number of windows scanned, policy exclusions, indexed
overlaps, and a pass status. A successful report has zero exclusions and zero
overlaps across all 976 current windows.

A deterministic sample is a redundant cross-check, not a replacement for the
exhaustive scan. Algorithm `sha256-priority-v1` computes each priority as
SHA-256 over the decimal seed, one null byte, and canonical JSON containing
`record_id`, `chrom`, `start_bp`, `end_bp`, and the window-sequence SHA-256. It
selects the lexicographically smallest requested priorities and commits their
ordered identities in `sample_digest`. The report also records the seed,
requested and observed sizes, policy exclusions, and indexed overlaps. Output
bytes are deterministic only when the exact producer source, schema bytes,
captured inputs, and every CLI binding—including artifact ID, origin fields,
seed, and sample size—are also identical.

## Build the evidence directory

Force-download both inputs at the exact revision. Never resolve `main` for an
evidence run:

```bash
REVISION=96e97a7ffe1e9ad8f9a98f690b220a32ac75ddc2
MEMBERSHIP_PATH=candidates/v0.3/geno-lewm-data-v0.3.0-r1/membership/geno-lewm-v03-membership-fd7f4bbde476-r1/success
INPUTS=/tmp/geno-lewm-v03-split-inputs

hf download abdelstark/geno-lewm-data \
  --repo-type dataset \
  --revision "$REVISION" \
  --include "$MEMBERSHIP_PATH/**" \
  --force-download \
  --local-dir "$INPUTS"

hf download abdelstark/geno-lewm-data \
  dataset_manifest.json placed/gnomad-common-windows.jsonl \
  --repo-type dataset \
  --revision "$REVISION" \
  --force-download \
  --local-dir "$INPUTS"
```

Then generate the evidence from a checkout containing the intended exact tool
source:

```bash
uv sync --frozen --extra dev
PRODUCER_SHA="<exact-40-character-GenoLeWM-commit>"
CONTAINER_IMAGE=ghcr.io/astral-sh/uv@sha256:35b0aa516fbcf6f18624919cfc38fa02ab3458e0ffcd3c03e932051b37f315db
export GENO_LEWM_VERIFIED_SPLIT_CONTAINER_IMAGE="$CONTAINER_IMAGE"

test "$(git rev-parse HEAD)" = "$PRODUCER_SHA"
test -z "$(git status --porcelain=v1 --untracked-files=all)"
test "$(git remote get-url origin)" = "https://github.com/AbdelStark/GenoLeWM.git"

uv run --no-sync python -m tools.data.v03_membership_splits \
  --store-dir "$INPUTS/$MEMBERSHIP_PATH/store" \
  --placed-windows-jsonl "$INPUTS/placed/gnomad-common-windows.jsonl" \
  --dataset-manifest-json "$INPUTS/dataset_manifest.json" \
  --output-dir /tmp/geno-lewm-v03-membership-splits-r1 \
  --artifact-id geno-lewm-v03-membership-splits-r1 \
  --membership-repository abdelstark/geno-lewm-data \
  --membership-revision "$REVISION" \
  --membership-artifact-path "$MEMBERSHIP_PATH" \
  --training-windows-repository abdelstark/geno-lewm-data \
  --training-windows-revision "$REVISION" \
  --training-windows-artifact-path placed/gnomad-common-windows.jsonl \
  --expected-store-content-identity \
    sha256:7fa661eefacf70258b8392aff88a6faea2749c812680d4a2bfc41376d061ff7a \
  --expected-store-physical-identity \
    sha256:d7ea2c4b8413768c9128c70a299a11f4adf35140102778a71cf56e69fb4db536 \
  --expected-store-rowset-sha256 \
    sha256:d268f2e2b67cce56c5d5099ec1ddcbd810fbb5973e6c96a929fd2c99fbd25f68 \
  --expected-dataset-manifest-sha256 \
    sha256:c3aa8f22b79e76fa5b6e3a43e02675cfc02d56dc7dc9fa36128c81874537016c \
  --expected-dataset-snapshot-id geno-lewm-data-v0.1.0-r1 \
  --expected-placed-windows-sha256 \
    sha256:ec76046771a163fbc22f326df26e2a332767eaa045dd919718c1cf86c4fbe0ac \
  --expected-placed-windows-size-bytes 4186560 \
  --expected-placed-windows-record-count 976 \
  --producer-git-commit "$PRODUCER_SHA" \
  --container-image "$CONTAINER_IMAGE" \
  --sample-seed 20260713 \
  --sample-size 128 \
  --report-schema-path \
    configs/data_v03/membership-split-evidence.schema.json
```

Production runs require the store's `official` lineage profile, an exact clean
producer checkout, and a digest-pinned container value supplied through the
`GENO_LEWM_VERIFIED_SPLIT_CONTAINER_IMAGE` launcher binding. The invocation
gate also requires the canonical HTTPS origin and verifies that the tool and
schema are tracked at the declared producer commit. The report records these
bindings and sets `publication_eligible` only when the official invocation gate
passed. A hidden fixture override exists only for reduced contract tests; it
records `invocation_verified: false` and `publication_eligible: false`.
Environment equality is an invocation binding, not a cryptographic attestation
of the running container.

## Artifact and failure contract

A successful output directory contains exactly:

```text
SHA256SUMS
contract/membership-split-evidence.schema.json
evidence/membership-split-evidence.json
splits/validation/clinvar-chr20.labels.jsonl
splits/validation/clinvar-chr20.vcf
splits/evaluation/clinvar-chr21.labels.jsonl
splits/evaluation/clinvar-chr21.vcf
```

The Draft 2020-12 report schema rejects unknown fields and structurally closes:

- operator-asserted membership repository, revision, and artifact path, plus
  verified local content, physical, rowset, lineage-profile, and
  chromosome-role identities;
- operator-asserted placed-window repository and revision, plus manifest-bound
  path, SHA-256, byte size, record count, dataset snapshot, upstream split,
  assembly, semantic role, chromosomes, and exact row fields;
- both stream identities, counts, binary classes, and keyset digests;
- exhaustive and deterministic-sample audit results; and
- the producer commit, digest-pinned container, invocation result, and explicit
  variant-level, unphased publication boundary.

The producer stages every file privately, validates the report against the
copied schema, requires the exact output inventory, writes `SHA256SUMS`, fsyncs
the tree, and renames it into place only after every check succeeds. Existing
outputs are never replaced. Input-identity, count, overlap, schema, or
publication failure leaves no partial public directory.

JSON Schema cannot express every arithmetic relationship in this report. The
producer separately reconciles stream totals and B/LB/LP/P counts against the
verified manifest, derives the exhaustive count from the captured window file,
and bounds the deterministic sample by that same universe. Independent review
must rerun those semantic checks against the exact store and window bytes;
schema validation alone is not a semantic verifier.

## Dataset and training binding

Dataset-package schema `1.1.0` closes the earlier double-counting ambiguity by
requiring one role on every listed artifact:

| Role | Package semantics |
| --- | --- |
| `split_data` | Declares a split and record count; it is eligible for the matching training or evaluation loader and contributes to split totals. |
| `split_companion` | Declares the same split and record count as the exact `split_data` path named by `companion_of`; it does not contribute a second copy of those records. |
| `evidence` | Declares no split, record count, or companion; it is verified provenance and is not training input. |

The optional `membership_and_split_evidence` object is closed to exactly two
bindings. `membership_store` records its package-relative path, artifact ID,
semantic content identity, physical identity, and rowset digest. `report`
records the report path, copied schema path, artifact ID, and schema version.
Package verification requires all store and report files to have the
`evidence` role, opens and fully verifies the store, pins the tracked report
schema digest and version, validates publication eligibility, and reconciles
the report's streams and placed training windows with their declared
`split_data` and `split_companion` artifacts.

The same binding is checked at each implemented consumer boundary:

```text
schema-1.1 dataset package
        ├── Carbon preflight: roles, companions, store + report
        ├── training runtime: one verified store + membership holdout policy
        ├── training-run verifier: dataset store/report + full metrics/policy binding
        └── paper verifier: dataset package == input check == snapshot report
```

The runtime serializes the complete store/report binding,
`MembershipStoreHoldoutPolicy.to_dict()`, and its canonical SHA-256 identity
into metrics, checkpoints, and bound training-run metadata. Resume and release
verification fail closed on any drift. Release verification hashes checkpoint
files but does not deserialize them.

Legacy dataset schema `1.0.0` remains unchanged and forbids artifact roles,
companions, and membership binding. Legacy training-run schema `1.0.0` remains
unbound and omits the new manifest and card surface; bound training runs use
schema `1.1.0`.

These contracts and verifiers do not complete the canonical schema-`1.1.0`
dataset assembler, publish a released v0.3 snapshot, or produce corrected
model-quality evidence. The split evidence remains variant-level and unphased;
it must not be described as phased-haplotype membership or clinical evidence.

## Local contract checks

```bash
uv run --extra dev pytest tests/unit/test_data_v03_membership_splits.py -q
uv run --extra dev ruff check tools/data/v03_membership_splits.py \
  tests/unit/test_data_v03_membership_splits.py
uv run --extra dev mypy --strict tools/data/v03_membership_splits.py
uv run --extra docs mkdocs build --strict
```
