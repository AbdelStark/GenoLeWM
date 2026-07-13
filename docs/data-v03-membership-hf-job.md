# v0.3 membership-store Hugging Face Job

`tools/jobs/v03_build_membership_store.sh` is the release-grade hosted runner
for constructing the real v0.3 variant-membership store. Run it only from an
immutable GenoLeWM commit that contains both the scalable membership builder
and this job contract.

## Locked inputs

The runner refuses overrides for the scientific inputs. It force-downloads and
verifies these exact dataset commits:

| Input | Immutable Hub revision |
| --- | --- |
| Snapshot lineage candidate | `4e5c641d3720a28f28d0d3efb3c5969678e84fe3` |
| 22 gnomAD v4.1 autosomal Parquets | `f3676763b3f7f71d0d0d098588e9bf377faa0c5c` |
| ClinVar 2026-04-15 Parquet | `9e1a2b279681177a7ca00b30b9eb8048b511d1cb` |

The lineage file path, its SHA-256, and its byte size are also pinned. The job
derives all 23 artifact paths from those lineage bytes and rejects any
repository, revision, namespace, path, digest, size, chromosome set, or source
cardinality drift before invoking the builder.

## Launch contract

Choose a new positive `RUN_ATTEMPT` for each intentional retry. The source SHA
must be a full 40-character lowercase commit, the checkout must be completely
clean (including untracked files), and the container must use an immutable
`@sha256:` digest. The script header contains the copyable `hf jobs run`
command; its essential shape is:

```bash
SHA="<merged-commit-containing-the-membership-builder-and-runner>"
RUN_ATTEMPT=1
IMAGE="ghcr.io/astral-sh/uv@sha256:35b0aa516fbcf6f18624919cfc38fa02ab3458e0ffcd3c03e932051b37f315db"

hf jobs run \
  --flavor cpu-upgrade \
  --secrets HF_TOKEN \
  --env COMMIT_SHA="$SHA" \
  --env RUN_ATTEMPT="$RUN_ATTEMPT" \
  --env CONTAINER_IMAGE="$IMAGE" \
  --timeout 4h \
  --detach \
  -- "$IMAGE" \
  bash -lc 'set -euo pipefail; git clone https://github.com/AbdelStark/GenoLeWM.git /workspace/GenoLeWM; cd /workspace/GenoLeWM; git checkout --detach "$COMMIT_SHA"; test "$(git rev-parse HEAD)" = "$COMMIT_SHA"; test -z "$(git status --porcelain=v1 --untracked-files=all)"; uv sync --frozen --extra train; exec uv run --no-sync bash tools/jobs/v03_build_membership_store.sh'
```

`HF_TOKEN` needs write access to the `abdelstark/geno-lewm-data` dataset. The
namespace includes the source-SHA prefix and `RUN_ATTEMPT`; existing namespaces
fail closed instead of being replaced.

## Success and failure semantics

The job builds the closed 23-source specification, constructs the store, and
runs the membership verifier's independent full scan. Only then does it create
a checksum-closed bundle and submit that complete directory in one
parent-conditional Hub commit. It force-downloads the exact Hub revision
returned by that commit, compares the complete file set, verifies every
checksum, and reruns the membership verifier on the downloaded store.

A successful log ends with:

```text
GENO_LEWM_V03_MEMBERSHIP_OK <40-character-hub-revision> <success-namespace>
```

No failure path uploads. A failed run leaves diagnostic files only in the
ephemeral job workspace, and it never creates a partial or candidate-negative
remote namespace.

This artifact records deterministic variant memberships, not phased haplotypes.
It does not release the v0.3 snapshot, establish dataset representativeness,
measure model quality or benchmark performance, or support clinical validity.

## Local contract checks

These checks exercise the lineage-to-spec and orchestration boundaries without
claiming that the real hosted build ran:

```bash
uv run pytest tests/unit/test_data_v03_membership_job.py \
  tests/unit/test_jobs_v03_membership_contract.py -q
bash -n tools/jobs/v03_build_membership_store.sh
```
