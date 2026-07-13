#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
#
# Stage one generation-pinned gnomAD v4.1 exome autosome for the v0.3 data
# pipeline. A successful receipt proves source-object and transform integrity
# for one chromosome only. It does not establish snapshot membership or split leakage control.
# It does not establish dataset representativeness, model quality, benchmark performance,
# or clinical validity.
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

check_remote_namespace_absent() {
  uv run python - "$UPLOAD_REPO" "$UPLOAD_REPO_TYPE" "$REMOTE_NAMESPACE" <<'PY'
import json
import os
import sys
import urllib.error
import urllib.parse
import urllib.request

repo_id, repo_type, namespace = sys.argv[1:]
if repo_type != "dataset":
    raise SystemExit(f"FATAL: unsupported locked repo type: {repo_type}")
repo_path = urllib.parse.quote(repo_id, safe="/")
namespace_path = urllib.parse.quote(namespace.strip("/"), safe="/")
repo_url = f"https://huggingface.co/api/datasets/{repo_path}/revision/main"
tree_url = (
    f"https://huggingface.co/api/datasets/{repo_path}/tree/main/{namespace_path}"
    "?recursive=false&expand=false"
)
repo_request = urllib.request.Request(
    repo_url,
    headers={"Authorization": f"Bearer {os.environ['HF_TOKEN']}"},
)
try:
    with urllib.request.urlopen(repo_request, timeout=30) as response:
        repo_info = json.load(response)
except urllib.error.HTTPError as exc:
    raise SystemExit(
        f"FATAL: cannot resolve remote parent commit: HTTP {exc.code}"
    ) from exc
except (OSError, ValueError) as exc:
    raise SystemExit(f"FATAL: cannot resolve remote parent commit: {exc}") from exc
repo_sha = repo_info.get("sha")
if not isinstance(repo_sha, str) or len(repo_sha) != 40:
    raise SystemExit("FATAL: remote repository did not report a full parent commit")

tree_request = urllib.request.Request(
    tree_url,
    headers={"Authorization": f"Bearer {os.environ['HF_TOKEN']}"},
)
try:
    with urllib.request.urlopen(tree_request, timeout=30) as response:
        json.load(response)
except urllib.error.HTTPError as exc:
    if exc.code == 404:
        print(repo_sha)
        raise SystemExit(0) from None
    raise SystemExit(
        f"FATAL: cannot prove remote namespace absence: HTTP {exc.code}"
    ) from exc
except (OSError, ValueError) as exc:
    raise SystemExit(f"FATAL: cannot prove remote namespace absence: {exc}") from exc
raise SystemExit(f"FATAL: immutable namespace already exists: {namespace}")
PY
}

log "prove immutable namespace is unused before doing expensive work"
check_remote_namespace_absent >/dev/null

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

log "recheck the immutable namespace immediately before publication"
REMOTE_PARENT_COMMIT="$(check_remote_namespace_absent)"

log "upload completed evidence and shard to $UPLOAD_REPO/$REMOTE_NAMESPACE"
uv run python - \
  "$UPLOAD_REPO" \
  "$UPLOAD_REPO_TYPE" \
  "$REMOTE_NAMESPACE" \
  "$PUBLISH_DIR" \
  "$REMOTE_PARENT_COMMIT" \
  "$COMMIT_SHA" \
  "$RELEASE" \
  "$CHROMOSOME" <<'PY'
import os
import sys

from huggingface_hub import HfApi

(
    repo_id,
    repo_type,
    namespace,
    publish_dir,
    parent_commit,
    commit_sha,
    release,
    chromosome,
) = sys.argv[1:]
api = HfApi(token=os.environ["HF_TOKEN"])
commit = api.upload_folder(
    repo_id=repo_id,
    repo_type=repo_type,
    folder_path=publish_dir,
    path_in_repo=namespace,
    parent_commit=parent_commit,
    commit_message=f"stage gnomAD {release} chr{chromosome} at {commit_sha}",
)
print(f"uploaded commit: {commit.oid}")
PY

log "completed immutable gnomAD staging namespace: $REMOTE_NAMESPACE"
