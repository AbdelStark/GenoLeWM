# ClinVar v0.3 staging verification

The corrected ClinVar staging shard can be checked independently at one
immutable Hugging Face dataset revision. The verifier rejects mutable refs,
requires the exact four-file namespace, downloads every file at the same
40-character revision, and performs a full Parquet scan.

Writing the immutable postflight report requires anchored `dir_fd` filesystem
operations. The tested Linux and macOS paths provide that capability; Windows
and any other unsupported platform fail before creating an output. Run the
publication step on a supported platform rather than weakening it to a
pathname-only write.

The known corrected artifact is verified with:

```bash
git fetch origin 304128e4d4f3a7985a3687089cb6947d41396c2f
uv run --extra dev --with huggingface-hub \
  python -m tools.data.v03_clinvar_postflight \
  --repo-id abdelstark/geno-lewm-data \
  --revision 9e1a2b279681177a7ca00b30b9eb8048b511d1cb \
  --namespace staging/clinvar-2026-04-15-archive-304128e4d4f3-r1 \
  --expected-source-commit 304128e4d4f3a7985a3687089cb6947d41396c2f \
  --expected-release 2026-04-15 \
  --output-json clinvar-remote-postflight.json
```

The source commit must be present in the local Git object database. The tool
reads the producer's ClinVar data module, shared VCF module, CLI, and report
helper directly with `git cat-file`; it does not trust the current checkout.
It derives the schema, label vocabulary, nullable fields, allele alphabet,
command name, default allele-length limit, output path, and receipt fields
from those immutable source bytes.

The resulting report records:

- the exact Hub repository, revision, namespace, and four-file identity set;
- the exact source commit and hashes of the four source-contract blobs;
- reconciliation of audit, prepare, runtime, source, command, release, and
  output identities;
- recomputed Parquet SHA-256, size, schema, metadata and scanned row counts,
  class balance, chromosome balance, schema-version balance, null counts, and
  value ranges; and
- an explicit list of completed checks and the original scientific claim
  boundary.

The closed Draft 2020-12 report schema is
[`configs/data_v03/clinvar-remote-postflight.schema.json`](https://github.com/AbdelStark/GenoLeWM/blob/main/configs/data_v03/clinvar-remote-postflight.schema.json).
The v0.3 snapshot-lineage assembler requires the report's relative path and
prefixed SHA-256 in its input spec, verifies the exact report bytes, and
type-strictly reconciles the Hub revision, namespace, source commit, release,
audit and Parquet identities, row counts, schema, classes, chromosomes, null
counts, and ranges. The lineage preserves the report SHA-256, size, verified
files, checks, and recomputed Parquet audit under `clinvar.remote_postflight`.
Copying report fields into another manifest without binding the report bytes is
not enough.

## Verification boundary

The source archive is not one of the four files in the staging namespace.
Consequently, this postflight reconciles its release, SHA-256, and size across
the immutable audit and prepare receipts, but it does not independently
re-download the archive. The audit's MD5 and URL are preserved in the report
with that limitation stated explicitly. Operators who need a fresh source-byte
check must use the pinned URL and all three recorded identities before treating
the source download as reproduced.

The report proves artifact integrity and producer-contract consistency only.
It does not establish a leakage-safe split, label correctness,
representativeness, clinical utility, or model quality.

## Source terms and use restrictions

GenoLeWM source code is Apache-2.0; that license does not replace the terms of
the upstream data. Review the official
[ClinVar disclaimer and data-use policy](https://www.ncbi.nlm.nih.gov/clinvar/docs/maintenance_use/)
before downloading, copying, or redistributing ClinVar data. NCBI asks for
ClinVar attribution when its data are copied or distributed, states that the
information is not intended for direct diagnostic or medical decision-making
use without professional review, and notes that NIH does not independently
verify submitted information. This staging verifier grants no clinical-use or
redistribution permission.
