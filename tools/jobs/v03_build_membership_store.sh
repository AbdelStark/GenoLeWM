#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
#
# Exact-revision v0.3 membership-store build for Hugging Face Jobs.
#
# This job creates variant memberships, not phased haplotypes. A successful
# bundle is not a released v0.3 snapshot and does not establish dataset
# representativeness, model quality, benchmark performance, or clinical
# validity.
#
# Submit only after the membership implementation and this runner are merged at
# the same immutable, clean commit:
#
#   SHA="<40-character GenoLeWM commit>"
#   RUN_ATTEMPT=1
#   IMAGE="ghcr.io/astral-sh/uv@sha256:35b0aa516fbcf6f18624919cfc38fa02ab3458e0ffcd3c03e932051b37f315db"
#   hf jobs run \
#     --flavor cpu-upgrade \
#     --secrets HF_TOKEN \
#     --env COMMIT_SHA="$SHA" \
#     --env RUN_ATTEMPT="$RUN_ATTEMPT" \
#     --env CONTAINER_IMAGE="$IMAGE" \
#     --timeout 4h \
#     --detach \
#     -- "$IMAGE" \
#     bash -lc 'set -euo pipefail; git clone https://github.com/AbdelStark/GenoLeWM.git /workspace/GenoLeWM; cd /workspace/GenoLeWM; git checkout --detach "$COMMIT_SHA"; test "$(git rev-parse HEAD)" = "$COMMIT_SHA"; test -z "$(git status --porcelain=v1 --untracked-files=all)"; uv sync --frozen --extra train; exec uv run --no-sync bash tools/jobs/v03_build_membership_store.sh'

set -euo pipefail

WORK="${WORK:-/tmp/geno-lewm-v03-membership}"
COMMIT_SHA="${COMMIT_SHA:?COMMIT_SHA is required}"
CONTAINER_IMAGE="${CONTAINER_IMAGE:?CONTAINER_IMAGE is required}"
RUN_ATTEMPT="${RUN_ATTEMPT:?RUN_ATTEMPT is required}"
HF_TOKEN="${HF_TOKEN:?HF_TOKEN is required}"

LINEAGE_REPO="abdelstark/geno-lewm-data"
LINEAGE_REVISION="4e5c641d3720a28f28d0d3efb3c5969678e84fe3"
LINEAGE_PATH="candidates/v0.3/geno-lewm-data-v0.3.0-r1/lineage/snapshot-lineage.json"
LINEAGE_SHA256="sha256:dcc7031bb1b409e55112c1f6576a878b9566b954d32ea75056a04b9ba1e95bea"
LINEAGE_SIZE_BYTES="195040"
DATA_REPO="abdelstark/geno-lewm-data"
GNOMAD_REVISION="f3676763b3f7f71d0d0d098588e9bf377faa0c5c"
CLINVAR_REVISION="9e1a2b279681177a7ca00b30b9eb8048b511d1cb"
UPLOAD_REPO="abdelstark/geno-lewm-data"
CANDIDATE_ID="geno-lewm-data-v0.3.0-r1"
ARTIFACT_ID="geno-lewm-data-v0.3.0-membership-r1"
RUN_NAME="${RUN_NAME:-geno-lewm-v03-membership-${COMMIT_SHA:0:12}-r${RUN_ATTEMPT}}"
PUBLISH_NAMESPACE="candidates/v0.3/${CANDIDATE_ID}/membership/${RUN_NAME}/success"

log() { echo "=== $* ==="; }
fatal() { echo "FATAL: $*" >&2; exit 2; }

log "validate immutable source, container, attempt, and clean checkout before writes"
[[ "$COMMIT_SHA" =~ ^[0-9a-f]{40}$ ]] \
  || fatal "COMMIT_SHA must be a full lowercase 40-character Git SHA"
[[ "$CONTAINER_IMAGE" =~ ^[^@[:space:]]+@sha256:[0-9a-f]{64}$ ]] \
  || fatal "CONTAINER_IMAGE must be digest-pinned"
[[ "$RUN_ATTEMPT" =~ ^[1-9][0-9]*$ ]] \
  || fatal "RUN_ATTEMPT must be a positive canonical integer"
[[ "$RUN_NAME" =~ ^[A-Za-z0-9][A-Za-z0-9._-]*$ ]] \
  || fatal "RUN_NAME is not a safe immutable namespace component"

OBSERVED_COMMIT_SHA="$(git rev-parse HEAD)"
[ "$OBSERVED_COMMIT_SHA" = "$COMMIT_SHA" ] \
  || fatal "commit drift: expected $COMMIT_SHA, observed $OBSERVED_COMMIT_SHA"
git diff --quiet -- . || fatal "tracked worktree differs from $COMMIT_SHA"
git diff --cached --quiet -- . || fatal "index differs from $COMMIT_SHA"
[ -z "$(git status --porcelain=v1 --untracked-files=all)" ] \
  || fatal "membership builder checkout contains untracked or modified inputs"
[ "$(git remote get-url origin)" = "https://github.com/AbdelStark/GenoLeWM.git" ] \
  || fatal "origin is not the canonical GenoLeWM repository"

for tracked_path in \
  tools/jobs/v03_build_membership_store.sh \
  tools/data/v03_membership_job.py \
  tools/data/v03_membership_store.py \
  tools/data/v03_gnomad_lock.py \
  configs/data_v03/membership-build-spec.schema.json \
  configs/data_v03/membership-store.schema.json \
  configs/data_v03/membership-build-receipt.schema.json
do
  git cat-file -e "$COMMIT_SHA:$tracked_path" \
    || fatal "required build input is not tracked at $COMMIT_SHA: $tracked_path"
done

export HF_TOKEN
export GENO_LEWM_VERIFIED_BUILD_CONTAINER_IMAGE="$CONTAINER_IMAGE"

rm -rf "$WORK"
INPUT_ROOT="$WORK/input"
EVIDENCE_DIR="$WORK/evidence"
STORE_DIR="$WORK/store/$ARTIFACT_ID"
PUBLIC_DIR="$WORK/public"
LINEAGE_ROOT="$INPUT_ROOT/lineage"
GNOMAD_ROOT="$INPUT_ROOT/gnomad"
CLINVAR_ROOT="$INPUT_ROOT/clinvar"
LINEAGE_JSON="$LINEAGE_ROOT/$LINEAGE_PATH"
DOWNLOAD_PLAN="$EVIDENCE_DIR/download-plan.json"
BUILD_SPEC="$INPUT_ROOT/membership-build.json"
SOURCE_IDENTITIES="$EVIDENCE_DIR/source-download-identities.json"
BUILD_REPORT="$EVIDENCE_DIR/membership-build-report.json"
VERIFY_REPORT="$EVIDENCE_DIR/membership-verify-report.json"
mkdir -p "$INPUT_ROOT" "$EVIDENCE_DIR" "$(dirname "$STORE_DIR")"

log "force-download the exact lineage candidate"
hf download "$LINEAGE_REPO" "$LINEAGE_PATH" \
  --repo-type dataset \
  --revision "$LINEAGE_REVISION" \
  --force-download \
  --local-dir "$LINEAGE_ROOT"

log "validate lineage bytes and close the exact 23-file download plan"
python -m tools.data.v03_membership_job author-download-plan \
  --lineage-json "$LINEAGE_JSON" \
  --expected-lineage-sha256 "$LINEAGE_SHA256" \
  --expected-lineage-size-bytes "$LINEAGE_SIZE_BYTES" \
  --output-json "$DOWNLOAD_PLAN"

mapfile -t GNOMAD_PATHS < <(
  python - "$DOWNLOAD_PLAN" "$GNOMAD_REVISION" <<'PY'
import json
import sys
from pathlib import Path

plan = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
revision = sys.argv[2]
entries = [entry for entry in plan["downloads"] if entry["kind"] == "gnomad"]
if len(entries) != 22 or any(entry["revision"] != revision for entry in entries):
    raise SystemExit("FATAL: gnomAD download plan revision or cardinality mismatch")
for entry in entries:
    print(entry["artifact_path"])
PY
)
test "${#GNOMAD_PATHS[@]}" -eq 22 \
  || fatal "validated gnomAD download plan did not contain 22 paths"
CLINVAR_PATH="$({
  python - "$DOWNLOAD_PLAN" "$CLINVAR_REVISION" <<'PY'
import json
import sys
from pathlib import Path

plan = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
revision = sys.argv[2]
entries = [entry for entry in plan["downloads"] if entry["kind"] == "clinvar"]
if len(entries) != 1 or entries[0]["revision"] != revision:
    raise SystemExit("FATAL: ClinVar download plan revision or cardinality mismatch")
print(entries[0]["artifact_path"])
PY
})"

log "force-download all 22 lineage-bound gnomAD Parquets"
hf download "$DATA_REPO" "${GNOMAD_PATHS[@]}" \
  --repo-type dataset \
  --revision "$GNOMAD_REVISION" \
  --force-download \
  --local-dir "$GNOMAD_ROOT"

log "force-download the lineage-bound ClinVar Parquet"
hf download "$DATA_REPO" "$CLINVAR_PATH" \
  --repo-type dataset \
  --revision "$CLINVAR_REVISION" \
  --force-download \
  --local-dir "$CLINVAR_ROOT"

log "verify every downloaded path, SHA-256, and size before authoring the build spec"
cp configs/data_v03/membership-build-spec.schema.json \
  "$INPUT_ROOT/membership-build-spec.schema.json"
python -m tools.data.v03_membership_job author-spec \
  --lineage-json "$LINEAGE_JSON" \
  --expected-lineage-sha256 "$LINEAGE_SHA256" \
  --expected-lineage-size-bytes "$LINEAGE_SIZE_BYTES" \
  --gnomad-download-root "$GNOMAD_ROOT" \
  --clinvar-download-root "$CLINVAR_ROOT" \
  --artifact-id "$ARTIFACT_ID" \
  --builder-git-commit "$COMMIT_SHA" \
  --container-image "$CONTAINER_IMAGE" \
  --output-json "$BUILD_SPEC" \
  --identity-report-json "$SOURCE_IDENTITIES"

log "build the closed 23-source membership store"
test ! -e "$STORE_DIR" || fatal "immutable membership output already exists: $STORE_DIR"
python -m tools.data.v03_membership_store build \
  --spec-json "$BUILD_SPEC" \
  --output-dir "$STORE_DIR" \
  > "$BUILD_REPORT"

log "independently full-scan the completed membership store"
python -m tools.data.v03_membership_store verify \
  --store-dir "$STORE_DIR" \
  > "$VERIFY_REPORT"
python - "$BUILD_REPORT" "$VERIFY_REPORT" "$ARTIFACT_ID" <<'PY'
import json
import sys
from pathlib import Path

build = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
verify = json.loads(Path(sys.argv[2]).read_text(encoding="utf-8"))
artifact_id = sys.argv[3]
if build.get("ok") is not True or verify.get("ok") is not True:
    raise SystemExit("FATAL: membership build or independent verification was not successful")
if build.get("artifact_id") != artifact_id or verify.get("artifact_id") != artifact_id:
    raise SystemExit("FATAL: membership artifact_id drifted")
for field in ("content_identity", "physical_identity", "row_count"):
    if build.get(field) != verify.get(field):
        raise SystemExit(f"FATAL: membership {field} drifted between build and verification")
if len(verify.get("source_role_counts", {})) != 23:
    raise SystemExit("FATAL: verified membership store is not bound to exactly 23 sources")
PY

log "assemble the already-verified success bundle"
mkdir -p "$PUBLIC_DIR/store" "$PUBLIC_DIR/evidence" "$PUBLIC_DIR/contract"
cp -a "$STORE_DIR/." "$PUBLIC_DIR/store/"
cp "$DOWNLOAD_PLAN" "$PUBLIC_DIR/evidence/"
cp "$SOURCE_IDENTITIES" "$PUBLIC_DIR/evidence/"
cp "$BUILD_REPORT" "$PUBLIC_DIR/evidence/"
cp "$VERIFY_REPORT" "$PUBLIC_DIR/evidence/"
cp "$BUILD_SPEC" "$PUBLIC_DIR/contract/"
cp configs/data_v03/membership-build-spec.schema.json "$PUBLIC_DIR/contract/"
cp configs/data_v03/membership-store.schema.json "$PUBLIC_DIR/contract/"
cp configs/data_v03/membership-build-receipt.schema.json "$PUBLIC_DIR/contract/"
python - \
  "$PUBLIC_DIR/evidence/job-summary.json" \
  "$COMMIT_SHA" \
  "$CONTAINER_IMAGE" \
  "$RUN_NAME" \
  "$PUBLISH_NAMESPACE" \
  "$LINEAGE_REVISION" \
  "$GNOMAD_REVISION" \
  "$CLINVAR_REVISION" \
  "$ARTIFACT_ID" <<'PY'
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

(
    output,
    source_commit,
    container_image,
    run_name,
    namespace,
    lineage_revision,
    gnomad_revision,
    clinvar_revision,
    artifact_id,
) = sys.argv[1:]
payload = {
    "schema_version": "geno-lewm.v03-membership-job-summary.v1",
    "generated_by": "tools.jobs.v03_build_membership_store",
    "generated_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
    "ok": True,
    "source_commit_sha": source_commit,
    "container_image": container_image,
    "run_name": run_name,
    "namespace": namespace,
    "artifact_id": artifact_id,
    "inputs": {
        "lineage_revision": lineage_revision,
        "gnomad_revision": gnomad_revision,
        "clinvar_revision": clinvar_revision,
    },
    "claim_boundary": (
        "This success bundle contains verified variant memberships, not phased haplotypes. "
        "It is not a released v0.3 snapshot and does not establish dataset representativeness, "
        "model quality, benchmark performance, or clinical validity."
    ),
}
Path(output).write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
PY
test -z "$(find "$PUBLIC_DIR" -type l -print -quit)" \
  || fatal "success bundle must not contain symbolic links"
(
  cd "$PUBLIC_DIR"
  find store evidence contract -type f -print0 \
    | sort -z \
    | xargs -0 sha256sum > SHA256SUMS
  sha256sum -c SHA256SUMS
)

log "publish the complete verified bundle in one conflict-safe Hub commit"
PUBLISH_REPORT="$(
  python -m tools.data.v03_gnomad_lock publish \
    --repo-id "$UPLOAD_REPO" \
    --repo-type dataset \
    --namespace "$PUBLISH_NAMESPACE" \
    --publish-dir "$PUBLIC_DIR" \
    --commit-message "publish verified v0.3 membership store from $COMMIT_SHA"
)"
HUB_REVISION="${PUBLISH_REPORT#uploaded commit: }"
[[ "$HUB_REVISION" =~ ^[0-9a-f]{40}$ ]] \
  || fatal "Hub publication did not return an immutable 40-character revision"

log "force-download the exact resulting Hub commit and reverify bytes and store semantics"
test ! -e "$WORK/verified-remote" \
  || fatal "remote verification directory already exists"
hf download "$UPLOAD_REPO" \
  --repo-type dataset \
  --revision "$HUB_REVISION" \
  --include "$PUBLISH_NAMESPACE/**" \
  --force-download \
  --local-dir "$WORK/verified-remote"
REMOTE_BUNDLE="$WORK/verified-remote/$PUBLISH_NAMESPACE"
python - "$PUBLIC_DIR" "$REMOTE_BUNDLE" <<'PY'
import sys
from pathlib import Path

local = Path(sys.argv[1])
remote = Path(sys.argv[2])
if not remote.is_dir():
    raise SystemExit("FATAL: exact Hub commit did not contain the success namespace")
local_files = {
    path.relative_to(local).as_posix() for path in local.rglob("*") if path.is_file()
}
remote_files = {
    path.relative_to(remote).as_posix() for path in remote.rglob("*") if path.is_file()
}
if local_files != remote_files:
    raise SystemExit("FATAL: exact Hub namespace file set differs from the local bundle")
PY
cmp "$PUBLIC_DIR/SHA256SUMS" "$REMOTE_BUNDLE/SHA256SUMS" \
  || fatal "remote checksum manifest differs from the locally verified manifest"
(
  cd "$REMOTE_BUNDLE"
  sha256sum -c SHA256SUMS
)
REMOTE_VERIFY_REPORT="$WORK/remote-membership-verify-report.json"
python -m tools.data.v03_membership_store verify \
  --store-dir "$REMOTE_BUNDLE/store" \
  > "$REMOTE_VERIFY_REPORT"
cmp "$VERIFY_REPORT" "$REMOTE_VERIFY_REPORT" \
  || fatal "remote membership verification report differs from the local report"

printf '%s\n' "$PUBLISH_REPORT"
echo "GENO_LEWM_V03_MEMBERSHIP_OK $HUB_REVISION $PUBLISH_NAMESPACE"
