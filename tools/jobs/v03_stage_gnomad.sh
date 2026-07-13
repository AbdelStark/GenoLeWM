#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
#
# Stage one generation-pinned gnomAD v4.1 exome autosome for the v0.3 data
# pipeline. A successful receipt proves source-object and transform integrity
# for one chromosome only. It does not establish snapshot membership or split leakage control.
# It does not establish dataset representativeness, model quality, benchmark performance,
# or clinical validity.
#
# Exact HF Jobs submission recipe (repeat with CHROMOSOME=1..22 from the exact merge SHA):
#   SHA="$(git rev-parse HEAD)"
#   CHROMOSOME=22
#   IMAGE="ghcr.io/astral-sh/uv@sha256:35b0aa516fbcf6f18624919cfc38fa02ab3458e0ffcd3c03e932051b37f315db"
#   hf jobs run \
#     --flavor cpu-upgrade \
#     --secrets HF_TOKEN \
#     --env COMMIT_SHA="$SHA" \
#     --env CHROMOSOME="$CHROMOSOME" \
#     --env CONTAINER_IMAGE="$IMAGE" \
#     --label project=geno-lewm-v03 \
#     --label task=gnomad-stage \
#     --timeout 8h \
#     --detach \
#     -- "$IMAGE" \
#     bash -lc 'set -euo pipefail; git clone https://github.com/AbdelStark/GenoLeWM.git /workspace/GenoLeWM; cd /workspace/GenoLeWM; git checkout --detach "$COMMIT_SHA"; test "$(git rev-parse HEAD)" = "$COMMIT_SHA"; uv sync --frozen --extra train; exec uv run --no-sync bash tools/jobs/v03_stage_gnomad.sh'
set -euo pipefail

WORK="${WORK:-/tmp/geno-lewm-v03-stage-gnomad}"
COMMIT_SHA="${COMMIT_SHA:?COMMIT_SHA is required}"
CHROMOSOME="${CHROMOSOME:?CHROMOSOME is required}"
CONTAINER_IMAGE="${CONTAINER_IMAGE:?CONTAINER_IMAGE is required}"
HF_TOKEN="${HF_TOKEN:?HF_TOKEN is required}"
SOURCE_LOCK="${SOURCE_LOCK:-configs/data_v03/gnomad-v4.1-exomes-autosomes.source-lock.json}"
export HF_TOKEN

PREFLIGHT_DIR="$(mktemp -d)"
PREFLIGHT_SELECTION="$PREFLIGHT_DIR/selection.json"

cleanup_preflight() {
  rm -rf "$PREFLIGHT_DIR"
}
trap cleanup_preflight EXIT

log() { echo "=== $* ==="; }
fatal() { echo "FATAL: $*" >&2; exit 2; }

json_field() {
  local json_path=$1
  local field_path=$2
  uv run python -c '
import json
import sys
from pathlib import Path

value = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
for part in sys.argv[2].split("."):
    value = value[part]
if isinstance(value, bool):
    print("true" if value else "false")
elif isinstance(value, (str, int, float)):
    print(value)
else:
    raise SystemExit(f"FATAL: {sys.argv[2]} is not a scalar JSON field")
' "$json_path" "$field_path"
}

log "verify exact repository revision and clean tracked checkout"
OBSERVED_COMMIT_SHA="$(git rev-parse HEAD)"
[ "$OBSERVED_COMMIT_SHA" = "$COMMIT_SHA" ] \
  || fatal "commit drift: expected $COMMIT_SHA, observed $OBSERVED_COMMIT_SHA"
git diff --quiet -- . || fatal "tracked worktree differs from $COMMIT_SHA"
git diff --cached --quiet -- . || fatal "index differs from $COMMIT_SHA"
git cat-file -e "$COMMIT_SHA:$SOURCE_LOCK" \
  || fatal "source lock is not tracked at $COMMIT_SHA: $SOURCE_LOCK"
EXPECTED_LOCK_BLOB="$(git rev-parse "$COMMIT_SHA:$SOURCE_LOCK")"
OBSERVED_LOCK_BLOB="$(git hash-object "$SOURCE_LOCK")"
[ "$OBSERVED_LOCK_BLOB" = "$EXPECTED_LOCK_BLOB" ] \
  || fatal "source lock bytes drifted from $COMMIT_SHA"

log "select chromosome $CHROMOSOME from the checked source lock"
uv run python -m tools.data.v03_gnomad_lock select \
  --lock-json "$SOURCE_LOCK" \
  --chromosome "$CHROMOSOME" \
  --commit-sha "$COMMIT_SHA" \
  --container-image "$CONTAINER_IMAGE" \
  --output-json "$PREFLIGHT_SELECTION"

EXPECTED_REPOSITORY="$(json_field "$PREFLIGHT_SELECTION" execution.repository)"
OBSERVED_REPOSITORY="$(git remote get-url origin)"
[ "$OBSERVED_REPOSITORY" = "$EXPECTED_REPOSITORY" ] \
  || fatal "repository drift: expected $EXPECTED_REPOSITORY, observed $OBSERVED_REPOSITORY"

UPLOAD_REPO="$(json_field "$PREFLIGHT_SELECTION" publication.repo)"
UPLOAD_REPO_TYPE="$(json_field "$PREFLIGHT_SELECTION" publication.repo_type)"
REMOTE_NAMESPACE="$(json_field "$PREFLIGHT_SELECTION" publication.namespace)"
METADATA_URL="$(json_field "$PREFLIGHT_SELECTION" source.metadata_url)"
MEDIA_URL="$(json_field "$PREFLIGHT_SELECTION" source.media_url)"
RELEASE="$(json_field "$PREFLIGHT_SELECTION" release)"
MIN_AF="$(json_field "$PREFLIGHT_SELECTION" transform.min_af)"
MAX_ALLELE_LEN="$(json_field "$PREFLIGHT_SELECTION" transform.max_allele_len)"
SOURCE_LOCK_SCHEMA="$(json_field "$PREFLIGHT_SELECTION" source_lock.schema.path)"
git cat-file -e "$COMMIT_SHA:$SOURCE_LOCK_SCHEMA" \
  || fatal "source lock schema is not tracked at $COMMIT_SHA: $SOURCE_LOCK_SCHEMA"
EXPECTED_SCHEMA_BLOB="$(git rev-parse "$COMMIT_SHA:$SOURCE_LOCK_SCHEMA")"
OBSERVED_SCHEMA_BLOB="$(git hash-object "$SOURCE_LOCK_SCHEMA")"
[ "$OBSERVED_SCHEMA_BLOB" = "$EXPECTED_SCHEMA_BLOB" ] \
  || fatal "source lock schema bytes drifted from $COMMIT_SHA"

log "prove immutable namespace is unused before doing expensive work"
uv run python -m tools.data.v03_gnomad_lock probe-namespace \
  --repo-id "$UPLOAD_REPO" \
  --repo-type "$UPLOAD_REPO_TYPE" \
  --namespace "$REMOTE_NAMESPACE" \
  >/dev/null

rm -rf "$WORK"
PUBLISH_DIR="$WORK/publish"
EVIDENCE_DIR="$PUBLISH_DIR/evidence"
DATASET_ROOT="$PUBLISH_DIR/data"
INPUT_DIR="$WORK/input"
INPUT_VCF="$INPUT_DIR/gnomad.exomes.${RELEASE}.sites.chr${CHROMOSOME}.vcf.bgz"
SELECTION_JSON="$EVIDENCE_DIR/selection.json"
GCS_METADATA_JSON="$EVIDENCE_DIR/gcs-object-metadata.json"
METADATA_VERIFICATION_JSON="$EVIDENCE_DIR/gcs-metadata-verification.json"
SOURCE_IDENTITY_JSON="$EVIDENCE_DIR/source-stream-identity.json"
PREPARE_REPORT_JSON="$EVIDENCE_DIR/prepare-report.json"
RECEIPT_JSON="$EVIDENCE_DIR/receipt.json"
OUTPUT_PARQUET="$DATASET_ROOT/gnomad/$RELEASE/variants.parquet"
mkdir -p "$EVIDENCE_DIR" "$INPUT_DIR"
cp "$PREFLIGHT_SELECTION" "$SELECTION_JSON"
cp "$SOURCE_LOCK" "$EVIDENCE_DIR/source-lock.json"
cp "$SOURCE_LOCK_SCHEMA" "$EVIDENCE_DIR/source-lock.schema.json"

log "fetch and verify generation-pinned GCS metadata"
curl "$METADATA_URL" \
  --fail \
  --silent \
  --show-error \
  --location \
  --retry 5 \
  --retry-all-errors \
  --output "$GCS_METADATA_JSON"
uv run python -m tools.data.v03_gnomad_lock verify-metadata \
  --selection-json "$SELECTION_JSON" \
  --metadata-json "$GCS_METADATA_JSON" \
  --output-json "$METADATA_VERIFICATION_JSON"

log "download the locked source object"
curl "$MEDIA_URL" \
  --fail \
  --silent \
  --show-error \
  --location \
  --retry 5 \
  --retry-all-errors \
  --output "$INPUT_VCF"

log "verify source size and upstream MD5; record streamed SHA-256"
uv run python -m tools.data.v03_gnomad_lock hash-source \
  --selection-json "$SELECTION_JSON" \
  --input-vcf "$INPUT_VCF" \
  --output-json "$SOURCE_IDENTITY_JSON"

log "run the existing streaming gnomAD preparer"
uv run geno-lewm-prepare-gnomad \
  --quiet \
  --no-banner \
  --input-vcf "$INPUT_VCF" \
  --output "$DATASET_ROOT" \
  --release "$RELEASE" \
  --min-af "$MIN_AF" \
  --max-allele-len "$MAX_ALLELE_LEN" \
  > "$PREPARE_REPORT_JSON"

log "reconcile source, argv, runtime, peak RSS, counts, filters, and output identity"
uv run python -m tools.data.v03_gnomad_lock author-receipt \
  --selection-json "$SELECTION_JSON" \
  --metadata-verification-json "$METADATA_VERIFICATION_JSON" \
  --source-identity-json "$SOURCE_IDENTITY_JSON" \
  --prepare-report-json "$PREPARE_REPORT_JSON" \
  --input-vcf "$INPUT_VCF" \
  --dataset-root "$DATASET_ROOT" \
  --output-parquet "$OUTPUT_PARQUET" \
  --output-json "$RECEIPT_JSON"

[ "$(json_field "$RECEIPT_JSON" ok)" = "true" ] \
  || fatal "staging receipt did not validate"
[ "$(json_field "$RECEIPT_JSON" publication.namespace)" = "$REMOTE_NAMESPACE" ] \
  || fatal "receipt namespace drifted after transform"

log "re-prove immutable namespace and publish with bounded stale-parent retries"
uv run python -m tools.data.v03_gnomad_lock publish \
  --repo-id "$UPLOAD_REPO" \
  --repo-type "$UPLOAD_REPO_TYPE" \
  --namespace "$REMOTE_NAMESPACE" \
  --publish-dir "$PUBLISH_DIR" \
  --commit-message "stage gnomAD $RELEASE chr$CHROMOSOME at $COMMIT_SHA"

log "completed immutable gnomAD staging namespace: $REMOTE_NAMESPACE"
