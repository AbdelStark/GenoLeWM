#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
#
# Exact-revision schema-1.1 v0.3 dataset snapshot publication for HF Jobs.
#
# The resulting candidate binds deterministic unphased variant memberships. It
# does not establish phased haplotypes, population representativeness, model
# quality, benchmark performance, or clinical validity.
#
#   SHA="<40-character merged GenoLeWM commit>"
#   RUN_ATTEMPT=1
#   GENERATED_AT="<explicit UTC timestamp, for example 2026-07-13T18:00:00Z>"
#   IMAGE="ghcr.io/astral-sh/uv@sha256:35b0aa516fbcf6f18624919cfc38fa02ab3458e0ffcd3c03e932051b37f315db"
#   hf jobs run \
#     --flavor cpu-upgrade \
#     --secrets HF_TOKEN \
#     --env COMMIT_SHA="$SHA" \
#     --env RUN_ATTEMPT="$RUN_ATTEMPT" \
#     --env GENERATED_AT="$GENERATED_AT" \
#     --env CONTAINER_IMAGE="$IMAGE" \
#     --timeout 6h \
#     --detach \
#     -- "$IMAGE" \
#     bash -lc 'set -euo pipefail; git clone https://github.com/AbdelStark/GenoLeWM.git /workspace/GenoLeWM; cd /workspace/GenoLeWM; git checkout --detach "$COMMIT_SHA"; test "$(git rev-parse HEAD)" = "$COMMIT_SHA"; test -z "$(git status --porcelain=v1 --untracked-files=all)"; uv sync --frozen --extra evidence; exec uv run --no-sync bash tools/jobs/v03_publish_dataset_snapshot.sh'

set -euo pipefail

WORK="/tmp/geno-lewm-v03-dataset-snapshot"
COMMIT_SHA="${COMMIT_SHA:?COMMIT_SHA is required}"
CONTAINER_IMAGE="${CONTAINER_IMAGE:?CONTAINER_IMAGE is required}"
RUN_ATTEMPT="${RUN_ATTEMPT:?RUN_ATTEMPT is required}"
GENERATED_AT="${GENERATED_AT:?GENERATED_AT is required}"
HF_TOKEN="${HF_TOKEN:?HF_TOKEN is required}"

DATA_REPO="abdelstark/geno-lewm-data"
UPLOAD_REPO="abdelstark/geno-lewm-data"
MEMBERSHIP_REVISION="96e97a7ffe1e9ad8f9a98f690b220a32ac75ddc2"
MEMBERSHIP_CANDIDATE_ROOT="candidates/v0.3/geno-lewm-data-v0.3.0-r1/membership/geno-lewm-v03-membership-fd7f4bbde476-r1"
MEMBERSHIP_PATH="${MEMBERSHIP_CANDIDATE_ROOT}/success"
SPLIT_REVISION="6d2ec7dd68af636ba8c594774c3c55a236c0995f"
SPLIT_PATH="${MEMBERSHIP_CANDIDATE_ROOT}/splits/geno-lewm-v03-membership-splits-bb24f6344274-r2/success"
GNOMAD_REVISION="f3676763b3f7f71d0d0d098588e9bf377faa0c5c"
CLINVAR_REVISION="9e1a2b279681177a7ca00b30b9eb8048b511d1cb"
TRAINING_WINDOWS_PATH="placed/gnomad-common-windows.jsonl"
SNAPSHOT_ID="geno-lewm-data-v0.3.0-r1"
RUN_NAME="${RUN_NAME:-geno-lewm-v03-dataset-snapshot-${COMMIT_SHA:0:12}-r${RUN_ATTEMPT}}"
PUBLISH_NAMESPACE="${MEMBERSHIP_CANDIDATE_ROOT}/snapshots/${RUN_NAME}/success"

log() { echo "=== $* ==="; }
fatal() { echo "FATAL: $*" >&2; exit 2; }

log "validate immutable source, container, timestamp, attempt, and clean checkout"
[[ "$COMMIT_SHA" =~ ^[0-9a-f]{40}$ ]] \
  || fatal "COMMIT_SHA must be a full lowercase 40-character Git SHA"
[[ "$CONTAINER_IMAGE" =~ ^[^@[:space:]]+@sha256:[0-9a-f]{64}$ ]] \
  || fatal "CONTAINER_IMAGE must be digest-pinned"
[[ "$RUN_ATTEMPT" =~ ^[1-9][0-9]*$ ]] \
  || fatal "RUN_ATTEMPT must be a positive canonical integer"
[[ "$GENERATED_AT" =~ ^[0-9]{4}-[0-9]{2}-[0-9]{2}T[0-9]{2}:[0-9]{2}:[0-9]{2}(\.[0-9]+)?Z$ ]] \
  || fatal "GENERATED_AT must be an explicit UTC ISO-8601 timestamp ending in Z"
[[ "$RUN_NAME" =~ ^[A-Za-z0-9][A-Za-z0-9._-]*$ ]] \
  || fatal "RUN_NAME is not a safe immutable namespace component"

OBSERVED_COMMIT_SHA="$(git rev-parse HEAD)"
[ "$OBSERVED_COMMIT_SHA" = "$COMMIT_SHA" ] \
  || fatal "commit drift: expected $COMMIT_SHA, observed $OBSERVED_COMMIT_SHA"
REPOSITORY_ROOT="$(git rev-parse --show-toplevel)"
[ "$(pwd -P)" = "$(cd "$REPOSITORY_ROOT" && pwd -P)" ] \
  || fatal "snapshot job must run from the repository root"
case "$REPOSITORY_ROOT/" in
  "$WORK/"*) fatal "fixed snapshot workspace must not contain the checkout" ;;
esac
case "$WORK/" in
  "$REPOSITORY_ROOT/"*) fatal "fixed snapshot workspace must remain outside the checkout" ;;
esac
git diff --quiet -- . || fatal "tracked worktree differs from $COMMIT_SHA"
git diff --cached --quiet -- . || fatal "index differs from $COMMIT_SHA"
[ -z "$(git status --porcelain=v1 --untracked-files=all)" ] \
  || fatal "snapshot checkout contains untracked or modified inputs"
[ "$(git remote get-url origin)" = "https://github.com/AbdelStark/GenoLeWM.git" ] \
  || fatal "origin is not the canonical GenoLeWM repository"
for tracked_path in \
  tools/jobs/v03_publish_dataset_snapshot.sh \
  tools/release/v03_dataset_snapshot.py \
  tools/release/dataset_package.py \
  tools/release/dataset_integrity.py \
  tools/data/v03_gnomad_lock.py \
  configs/data_v03/membership-split-evidence.schema.json
do
  git cat-file -e "$COMMIT_SHA:$tracked_path" \
    || fatal "required snapshot input is not tracked at $COMMIT_SHA: $tracked_path"
done
command -v hf >/dev/null 2>&1 || fatal "the evidence environment lacks the HF CLI"
python - <<'PY'
import huggingface_hub
import jsonschema
import pyarrow
PY

export HF_TOKEN
rm -rf "$WORK"
INPUT_ROOT="$WORK/input"
MEMBERSHIP_DOWNLOAD="$INPUT_ROOT/membership"
SPLIT_DOWNLOAD="$INPUT_ROOT/splits"
GNOMAD_ROOT="$INPUT_ROOT/gnomad"
CLINVAR_ROOT="$INPUT_ROOT/clinvar"
WINDOWS_ROOT="$INPUT_ROOT/windows"
PUBLIC_DIR="$WORK/public"
REMOTE_ROOT="$WORK/verified-remote"
mkdir -p "$INPUT_ROOT"

log "force-download the exact membership success bundle"
hf download "$DATA_REPO" \
  --repo-type dataset \
  --revision "$MEMBERSHIP_REVISION" \
  --include "$MEMBERSHIP_PATH/**" \
  --force-download \
  --local-dir "$MEMBERSHIP_DOWNLOAD"
MEMBERSHIP_BUNDLE="$MEMBERSHIP_DOWNLOAD/$MEMBERSHIP_PATH"

log "force-download the exact membership-split success bundle"
hf download "$DATA_REPO" \
  --repo-type dataset \
  --revision "$SPLIT_REVISION" \
  --include "$SPLIT_PATH/**" \
  --force-download \
  --local-dir "$SPLIT_DOWNLOAD"
SPLIT_BUNDLE="$SPLIT_DOWNLOAD/$SPLIT_PATH"

log "derive the closed 22+1 prepared-source download list from published evidence"
SOURCE_IDENTITIES="$MEMBERSHIP_BUNDLE/evidence/source-download-identities.json"
mapfile -t GNOMAD_PATHS < <(
  python - "$SOURCE_IDENTITIES" "$GNOMAD_REVISION" <<'PY'
import json
import sys
from pathlib import Path

payload = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
revision = sys.argv[2]
entries = [entry for entry in payload["files"] if entry["kind"] == "gnomad"]
if len(entries) != 22 or any(entry["revision"] != revision for entry in entries):
    raise SystemExit("FATAL: gnomAD source identity cardinality or revision drifted")
for entry in entries:
    print(entry["artifact_path"])
PY
)
[ "${#GNOMAD_PATHS[@]}" -eq 22 ] || fatal "expected exactly 22 gnomAD source paths"
CLINVAR_PATH="$(
  python - "$SOURCE_IDENTITIES" "$CLINVAR_REVISION" <<'PY'
import json
import sys
from pathlib import Path

payload = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
revision = sys.argv[2]
entries = [entry for entry in payload["files"] if entry["kind"] == "clinvar"]
if len(entries) != 1 or entries[0]["revision"] != revision:
    raise SystemExit("FATAL: ClinVar source identity cardinality or revision drifted")
print(entries[0]["artifact_path"])
PY
)"

log "force-download all exact prepared gnomAD and ClinVar sources"
hf download "$DATA_REPO" "${GNOMAD_PATHS[@]}" \
  --repo-type dataset \
  --revision "$GNOMAD_REVISION" \
  --force-download \
  --local-dir "$GNOMAD_ROOT"
hf download "$DATA_REPO" "$CLINVAR_PATH" \
  --repo-type dataset \
  --revision "$CLINVAR_REVISION" \
  --force-download \
  --local-dir "$CLINVAR_ROOT"

log "force-download the exact placed-window training artifact"
hf download "$DATA_REPO" "$TRAINING_WINDOWS_PATH" \
  --repo-type dataset \
  --revision "$MEMBERSHIP_REVISION" \
  --force-download \
  --local-dir "$WINDOWS_ROOT"
TRAINING_WINDOWS="$WINDOWS_ROOT/$TRAINING_WINDOWS_PATH"

log "assemble and independently verify the local schema-1.1 snapshot"
python -m tools.release.v03_dataset_snapshot assemble \
  --membership-bundle-dir "$MEMBERSHIP_BUNDLE" \
  --split-bundle-dir "$SPLIT_BUNDLE" \
  --gnomad-root "$GNOMAD_ROOT" \
  --clinvar-root "$CLINVAR_ROOT" \
  --training-windows "$TRAINING_WINDOWS" \
  --dataset-dir "$PUBLIC_DIR" \
  --split-repository "$DATA_REPO" \
  --split-revision "$SPLIT_REVISION" \
  --split-artifact-path "$SPLIT_PATH" \
  --snapshot-id "$SNAPSHOT_ID" \
  --generated-at "$GENERATED_AT" \
  --producer-git-commit "$COMMIT_SHA" \
  --container-image "$CONTAINER_IMAGE" \
  > "$WORK/assembly-report.json"
python -m tools.release.v03_dataset_snapshot verify \
  --dataset-dir "$PUBLIC_DIR" \
  > "$WORK/local-verification-report.json"
cmp "$WORK/assembly-report.json" "$WORK/local-verification-report.json" \
  || fatal "assembly report differs from independent local verification"
(
  cd "$PUBLIC_DIR"
  sha256sum -c SHA256SUMS
)

log "publish the complete verified snapshot in one conflict-safe Hub commit"
PUBLISH_REPORT="$(
  python -m tools.data.v03_gnomad_lock publish \
    --repo-id "$UPLOAD_REPO" \
    --repo-type dataset \
    --namespace "$PUBLISH_NAMESPACE" \
    --publish-dir "$PUBLIC_DIR" \
    --commit-message "publish verified schema-1.1 v0.3 dataset snapshot from $COMMIT_SHA"
)"
HUB_REVISION="${PUBLISH_REPORT#uploaded commit: }"
[[ "$HUB_REVISION" =~ ^[0-9a-f]{40}$ ]] \
  || fatal "Hub publication did not return an immutable 40-character revision"

log "force-download and reverify the exact resulting Hub commit"
test ! -e "$REMOTE_ROOT" || fatal "remote verification directory already exists"
hf download "$UPLOAD_REPO" \
  --repo-type dataset \
  --revision "$HUB_REVISION" \
  --include "$PUBLISH_NAMESPACE/**" \
  --force-download \
  --local-dir "$REMOTE_ROOT"
REMOTE_BUNDLE="$REMOTE_ROOT/$PUBLISH_NAMESPACE"
cmp "$PUBLIC_DIR/SHA256SUMS" "$REMOTE_BUNDLE/SHA256SUMS" \
  || fatal "remote checksum manifest differs from the local verified snapshot"
python -m tools.release.v03_dataset_snapshot verify \
  --dataset-dir "$REMOTE_BUNDLE" \
  > "$WORK/remote-verification-report.json"
cmp "$WORK/local-verification-report.json" "$WORK/remote-verification-report.json" \
  || fatal "remote verification report differs from local verification"
(
  cd "$REMOTE_BUNDLE"
  sha256sum -c SHA256SUMS
)

printf '%s\n' "$PUBLISH_REPORT"
echo "GENO_LEWM_V03_SNAPSHOT_OK $HUB_REVISION $PUBLISH_NAMESPACE"

