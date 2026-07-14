#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
#
# Exact-trace H200 interruption/resume proof for the v0.3 Carbon window cache.
#
# A successful bundle proves that one exact public production-training trace
# was encoded on one measured H200, interrupted after durable cache shards,
# resumed without re-encoding those shards, checksum-closed, published once,
# and replayed from the exact resulting Hub revision. It does not establish the
# RFC-0006 10%-corpus/24-hour target, training quality, biological validity, or
# clinical validity.
#
# Exact HF Jobs submission recipe (run only from the exact merged commit).
# The raw `hf jobs run --volume` grammar in huggingface-hub 1.8.0 cannot carry
# a repository revision. Use the PEP 723 launcher below: it requests the model
# through HfApi with an exact revision-bearing, read-only Volume and rejects a
# returned JobInfo unless the full public launch contract is preserved.
#   SHA="$(git rev-parse HEAD)"
#   uv run --script tools/research/v03_cache_h200_launch.py \
#     --source-commit "$SHA" \
#     --run-attempt 1

set -euo pipefail

WORK_ROOT="/work/geno-lewm-v03-cache-h200-proof"
CARBON_DIR="/carbon"
RUNTIME_IDENTITY="configs/data_v03/carbon-500m-l2-runtime-identity.json"
PROOF_SCHEMA="configs/data_v03/cache-h200-proof.schema.json"
TRAINING_CONFIG="configs/data_v03/train-carbon-500m-snv-l2-epoch-r1.yaml"
CREATED_AT_NS="1783987200000000000"
BATCH_SIZE="8"
ROWS_PER_SHARD="256"
MIN_DURABLE_SHARDS="2"
SHARD_WAIT_TIMEOUT_SECONDS="14400"
EXPECTED_TRACE_REVISION="da0d86cde7bf88de2015ab7c516f356e9ae89469"
EXPECTED_TRACE_ARTIFACT_PATH="training-traces/v0.3/geno-lewm-v03-training-trace-48b5bf71397f-712d612d85ea-job-6a55f38e85d9643ce16d29e7-r1/success"
EXPECTED_CARBON_REPOSITORY="HuggingFaceBio/Carbon-500M"
EXPECTED_CARBON_REVISION="5d31d59b3c845b288a13aedb1358934196852eec"

COMMIT_SHA="${COMMIT_SHA:?COMMIT_SHA is required}"
CONTAINER_IMAGE="${CONTAINER_IMAGE:?CONTAINER_IMAGE is required}"
CARBON_REPOSITORY="${CARBON_REPOSITORY:?CARBON_REPOSITORY is required}"
CARBON_REVISION="${CARBON_REVISION:?CARBON_REVISION is required}"
TRACE_REPOSITORY="${TRACE_REPOSITORY:?TRACE_REPOSITORY is required}"
TRACE_REVISION="${TRACE_REVISION:?TRACE_REVISION is required}"
TRACE_ARTIFACT_PATH="${TRACE_ARTIFACT_PATH:?TRACE_ARTIFACT_PATH is required}"
RUN_ATTEMPT="${RUN_ATTEMPT:?RUN_ATTEMPT is required}"
HF_TOKEN="${HF_TOKEN:?HF_TOKEN is required}"
export GENO_LEWM_CACHE_H200_PROOF_DECLARED_CONTAINER_IMAGE="$CONTAINER_IMAGE"

UPLOAD_REPOSITORY="abdelstark/geno-lewm-data"
RUN_NAME="geno-lewm-v03-cache-h200-proof-${COMMIT_SHA:0:12}-r${RUN_ATTEMPT}"
PUBLISH_NAMESPACE="candidates/v0.3/geno-lewm-data-v0.3.0-r1/cache-h200-proofs/${RUN_NAME}/success"

DOWNLOAD_ROOT="$WORK_ROOT/download"
BUNDLE_DIR="$WORK_ROOT/bundle"
LOG_ROOT="$WORK_ROOT/logs"
RUNTIME_ROOT="$WORK_ROOT/runtime"
REMOTE_DOWNLOAD_ROOT="$WORK_ROOT/remote"
HARDWARE_JSON="$RUNTIME_ROOT/hardware.json"
CLI_REPORT="$RUNTIME_ROOT/cache-build-cli-report.json"
CACHE_BUILD_STATE="$BUNDLE_DIR/evidence/cache_build_state.json"

ATTEMPT1_RUN_ID="${RUN_NAME}-attempt1"
RESUME_RUN_ID="${RUN_NAME}-resume"
ATTEMPT1_LOG="$LOG_ROOT/${ATTEMPT1_RUN_ID}.jsonl"
RESUME_LOG="$LOG_ROOT/${RESUME_RUN_ID}.jsonl"

log() { echo "=== $* ==="; }
fatal() { echo "FATAL: $*" >&2; exit 2; }

ensure_physical_work_directory() {
  local directory=$1
  if [ -L "$directory" ]; then
    fatal "work directory must not be a symbolic link: $directory"
  fi
  if [ -e "$directory" ] && [ ! -d "$directory" ]; then
    fatal "work path exists but is not a directory: $directory"
  fi
  if [ ! -e "$directory" ]; then
    mkdir -- "$directory"
  fi
  test "$(cd "$directory" && pwd -P)" = "$directory" \
    || fatal "work directory is not the expected physical path: $directory"
  test -w "$directory" || fatal "work directory is not writable: $directory"
}

validate_trace_artifact_path() {
  local artifact_path=$1
  [[ "$artifact_path" =~ ^training-traces/v0\.3/geno-lewm-v03-training-trace-[A-Za-z0-9._-]+/success$ ]] \
    || fatal "TRACE_ARTIFACT_PATH must be an immutable v0.3 training-trace success namespace"
  case "/$artifact_path/" in
    *//* | */./* | */../*) fatal "TRACE_ARTIFACT_PATH contains an unsafe path component" ;;
  esac
}

completed_shards() {
  local state_path=$1
  python - "$state_path" <<'PY'
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
payload = json.loads(path.read_text(encoding="utf-8"))
completed = payload.get("completed_shards")
if not isinstance(completed, list):
    raise SystemExit("FATAL: cache build state lacks a completed_shards list")
print(len(completed))
PY
}

wait_until_stopped() {
  local process_id=$1
  local process_state
  local attempt
  for attempt in $(seq 1 100); do
    kill -0 "$process_id" 2>/dev/null || fatal "cache process exited before SIGSTOP settled"
    process_state="$(ps -o stat= -p "$process_id" | tr -d '[:space:]')"
    case "$process_state" in
      *T*) return 0 ;;
    esac
    sleep 0.1
  done
  fatal "cache process did not enter a stopped state"
}

CACHE_PID=""
cleanup_background_process() {
  local exit_code=$?
  trap - EXIT
  if [ -n "$CACHE_PID" ] && kill -0 "$CACHE_PID" 2>/dev/null; then
    kill -TERM "$CACHE_PID" 2>/dev/null || true
    kill -CONT "$CACHE_PID" 2>/dev/null || true
    wait "$CACHE_PID" 2>/dev/null || true
  fi
  exit "$exit_code"
}
trap cleanup_background_process EXIT

log "validate exact public source, immutable inputs, and clean canonical checkout"
[[ "$COMMIT_SHA" =~ ^[0-9a-f]{40}$ ]] \
  || fatal "COMMIT_SHA must be a full lowercase 40-character Git SHA"
[[ "$TRACE_REVISION" =~ ^[0-9a-f]{40}$ ]] \
  || fatal "TRACE_REVISION must be a full lowercase 40-character Hub revision"
test "$TRACE_REVISION" = "$EXPECTED_TRACE_REVISION" \
  || fatal "TRACE_REVISION differs from the exact corrected public training trace"
[[ "$CARBON_REVISION" =~ ^[0-9a-f]{40}$ ]] \
  || fatal "CARBON_REVISION must be a full lowercase 40-character Hub revision"
test "$CARBON_REVISION" = "$EXPECTED_CARBON_REVISION" \
  || fatal "CARBON_REVISION differs from the corrected Carbon runtime"
[[ "$CONTAINER_IMAGE" =~ ^[^@[:space:]]+@sha256:[0-9a-f]{64}$ ]] \
  || fatal "CONTAINER_IMAGE must be digest-pinned"
[[ "$RUN_ATTEMPT" =~ ^[1-9][0-9]*$ ]] \
  || fatal "RUN_ATTEMPT must be a positive canonical integer"
[[ "$TRACE_REPOSITORY" =~ ^[A-Za-z0-9._-]+/[A-Za-z0-9._-]+$ ]] \
  || fatal "TRACE_REPOSITORY must be a safe owner/name repository id"
test "$TRACE_REPOSITORY" = "abdelstark/geno-lewm-data" \
  || fatal "TRACE_REPOSITORY must be the canonical public training-trace dataset"
[[ "$CARBON_REPOSITORY" =~ ^[A-Za-z0-9._-]+/[A-Za-z0-9._-]+$ ]] \
  || fatal "CARBON_REPOSITORY must be a safe owner/name repository id"
test "$CARBON_REPOSITORY" = "$EXPECTED_CARBON_REPOSITORY" \
  || fatal "CARBON_REPOSITORY must be the canonical Carbon model repository"
validate_trace_artifact_path "$TRACE_ARTIFACT_PATH"
test "$TRACE_ARTIFACT_PATH" = "$EXPECTED_TRACE_ARTIFACT_PATH" \
  || fatal "TRACE_ARTIFACT_PATH differs from the exact corrected public training trace"

REPOSITORY_ROOT="$(git rev-parse --show-toplevel)"
test "$(pwd -P)" = "$(cd "$REPOSITORY_ROOT" && pwd -P)" \
  || fatal "H200 proof job must run from the repository root"
test "$(git rev-parse HEAD)" = "$COMMIT_SHA" \
  || fatal "checkout commit differs from COMMIT_SHA"
git diff --quiet -- . || fatal "tracked worktree differs from COMMIT_SHA"
git diff --cached --quiet -- . || fatal "index differs from COMMIT_SHA"
test -z "$(git status --porcelain=v1 --untracked-files=all)" \
  || fatal "checkout contains modified or untracked inputs"
test "$(git remote get-url origin)" = "https://github.com/AbdelStark/GenoLeWM.git" \
  || fatal "origin is not the canonical GenoLeWM repository"

for tracked_path in \
  tools/jobs/v03_cache_h200_proof.sh \
  tools/research/v03_cache_h200_launch.py \
  tools/research/v03_cache_h200_proof.py \
  tools/data/v03_gnomad_lock.py \
  "$RUNTIME_IDENTITY" \
  "$PROOF_SCHEMA" \
  "$TRAINING_CONFIG" \
  configs/data_v03/training-trace.schema.json
do
  git cat-file -e "$COMMIT_SHA:$tracked_path" \
    || fatal "required proof input is not tracked at COMMIT_SHA: $tracked_path"
  test "$(git hash-object "$tracked_path")" = "$(git rev-parse "$COMMIT_SHA:$tracked_path")" \
    || fatal "required proof input bytes differ from COMMIT_SHA: $tracked_path"
done

PUBLIC_SOURCE_COMMIT="$(
  curl \
    --fail \
    --silent \
    --show-error \
    --location \
    --retry 5 \
    --retry-all-errors \
    "https://api.github.com/repos/AbdelStark/GenoLeWM/commits/$COMMIT_SHA" \
  | python -c '
import json
import sys

payload = json.load(sys.stdin)
sha = payload.get("sha")
if not isinstance(sha, str):
    raise SystemExit("FATAL: canonical GitHub response lacks a commit SHA")
print(sha)
'
)"
test "$PUBLIC_SOURCE_COMMIT" = "$COMMIT_SHA" \
  || fatal "COMMIT_SHA is not publicly resolvable at the canonical GitHub repository"

command -v hf >/dev/null 2>&1 || fatal "the job environment lacks the HF CLI"
command -v nvidia-smi >/dev/null 2>&1 || fatal "the job environment lacks nvidia-smi"
test -d "$CARBON_DIR" || fatal "Carbon-500M is not mounted read-only at $CARBON_DIR"
test ! -L "$CARBON_DIR" || fatal "Carbon-500M mount must not be a symbolic link"
test "$(cd "$CARBON_DIR" && pwd -P)" = "$CARBON_DIR" \
  || fatal "Carbon-500M mount is not the expected physical path"
export HF_TOKEN

log "prove immutable proof namespace absence before any /work write"
uv run --no-sync python -m tools.data.v03_gnomad_lock probe-namespace \
  "--repo-id" "$UPLOAD_REPOSITORY" \
  "--repo-type" "dataset" \
  "--namespace" "$PUBLISH_NAMESPACE" \
  >/dev/null

ensure_physical_work_directory "/work"
test ! -e "$WORK_ROOT" || fatal "fixed proof workspace already exists: $WORK_ROOT"
mkdir -p \
  "$DOWNLOAD_ROOT" \
  "$BUNDLE_DIR/cache" \
  "$BUNDLE_DIR/evidence" \
  "$BUNDLE_DIR/trace" \
  "$BUNDLE_DIR/proof" \
  "$LOG_ROOT" \
  "$RUNTIME_ROOT"

log "force-download the exact public production training trace"
hf download "$TRACE_REPOSITORY" \
  "--repo-type" "dataset" \
  "--revision" "$TRACE_REVISION" \
  "--include" "$TRACE_ARTIFACT_PATH/**" \
  "--force-download" \
  "--local-dir" "$DOWNLOAD_ROOT"
DOWNLOADED_TRACE="$DOWNLOAD_ROOT/$TRACE_ARTIFACT_PATH"
test -d "$DOWNLOADED_TRACE" || fatal "exact trace namespace was not downloaded"
test ! -L "$DOWNLOADED_TRACE" || fatal "downloaded trace namespace is a symbolic link"
cp -a "$DOWNLOADED_TRACE/." "$BUNDLE_DIR/trace/"

log "verify trace checksum closure, shape, lineage, and empty cache destinations"
uv run --no-sync python -m tools.research.v03_cache_h200_proof preflight \
  "--bundle-dir" "$BUNDLE_DIR" \
  "--trace-dir" "$BUNDLE_DIR/trace" \
  "--trace-repository" "$TRACE_REPOSITORY" \
  "--trace-revision" "$TRACE_REVISION" \
  "--trace-artifact-path" "$TRACE_ARTIFACT_PATH" \
  "--runtime-identity" "$RUNTIME_IDENTITY" \
  "--source-commit" "$COMMIT_SHA" \
  "--container-image" "$CONTAINER_IMAGE"

log "measure and bind the mounted H200, CUDA, driver, Python, and Carbon runtime"
NVIDIA_SMI_QUERY_RAW="$(
  nvidia-smi --query-gpu=index,name,memory.total,compute_cap,driver_version --format=csv,noheader,nounits
)"
test -n "$NVIDIA_SMI_QUERY_RAW" || fatal "nvidia-smi returned an empty hardware query"
python - \
  "$HARDWARE_JSON" \
  "$COMMIT_SHA" \
  "$CONTAINER_IMAGE" \
  "$CARBON_DIR" \
  "$RUNTIME_IDENTITY" \
  "$NVIDIA_SMI_QUERY_RAW" <<'PY'
import csv
import json
import platform
import sys
from io import StringIO
from pathlib import Path

import torch

from geno_lewm.encoder._identity import encoder_runtime_hash

output_path = Path(sys.argv[1])
source_commit = sys.argv[2]
container_image = sys.argv[3]
carbon_dir = Path(sys.argv[4])
runtime_identity_path = Path(sys.argv[5])
raw_query = sys.argv[6].strip()

runtime_identity = json.loads(runtime_identity_path.read_text(encoding="utf-8"))
expected_runtime_hash = runtime_identity.get("runtime_hash")
observed_runtime_hash = encoder_runtime_hash(carbon_dir)
if observed_runtime_hash != expected_runtime_hash:
    raise SystemExit(
        "FATAL: mounted Carbon runtime hash drifted: "
        f"expected {expected_runtime_hash}, observed {observed_runtime_hash}"
    )
if not torch.cuda.is_available():
    raise SystemExit("FATAL: CUDA is required for the H200 cache proof")
if torch.cuda.device_count() != 1:
    raise SystemExit(
        f"FATAL: H200 cache proof requires exactly one CUDA device, observed {torch.cuda.device_count()}"
    )
properties = torch.cuda.get_device_properties(0)
if "H200" not in properties.name:
    raise SystemExit(f"FATAL: H200 cache proof requires H200, observed {properties.name}")
if properties.total_memory <= 0:
    raise SystemExit("FATAL: CUDA reported non-positive H200 memory")

rows = list(csv.reader(StringIO(raw_query)))
if len(rows) != 1 or len(rows[0]) != 5:
    raise SystemExit("FATAL: nvidia-smi must report exactly one five-field GPU row")
index, query_name, _memory_mib, query_capability, driver_version = (
    field.strip() for field in rows[0]
)
compute_capability = f"{properties.major}.{properties.minor}"
if index != "0" or query_name != properties.name:
    raise SystemExit("FATAL: torch and nvidia-smi GPU identities differ")
if query_capability != compute_capability:
    raise SystemExit("FATAL: torch and nvidia-smi compute capabilities differ")
if not driver_version or not torch.version.cuda:
    raise SystemExit("FATAL: CUDA or NVIDIA driver version is unavailable")

payload = {
    "schema_version": "geno-lewm.v03-h200-hardware.v1",
    "generated_by": "tools.jobs.v03_cache_h200_proof",
    "source_commit_sha": source_commit,
    "container_image": container_image,
    "nvidia_smi_query_raw": raw_query,
    "device": {
        "type": "cuda",
        "index": 0,
        "name": properties.name,
        "total_memory_bytes": properties.total_memory,
        "compute_capability": compute_capability,
    },
    "runtime": {
        "python_version": platform.python_version(),
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "driver_version": driver_version,
    },
}
output_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
PY
HARDWARE="$(
  python - "$HARDWARE_JSON" <<'PY'
import json
import sys
from pathlib import Path

payload = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
device = payload["device"]
runtime = payload["runtime"]
print(
    f"{device['name']}; {device['total_memory_bytes']} bytes; "
    f"CUDA {runtime['cuda_version']}; driver {runtime['driver_version']}; single GPU"
)
PY
)"

CACHE_ARGV=(
  geno-lewm-cache-windows
  "--quiet" "--no-banner"
  "--cache-dir" "$BUNDLE_DIR/cache"
  "--requests-jsonl" "$BUNDLE_DIR/trace/cache_build_requests.jsonl"
  "--evidence-dir" "$BUNDLE_DIR/evidence"
  "--encoder-runtime-identity" "$RUNTIME_IDENTITY"
  "--carbon-model-dir" "$CARBON_DIR"
  "--config" "$TRAINING_CONFIG"
  "--created-at-ns" "$CREATED_AT_NS"
  "--batch-size" "$BATCH_SIZE"
  "--rows-per-shard" "$ROWS_PER_SHARD"
  "--device" "cuda"
  "--hardware" "$HARDWARE"
  "--log-dir" "$LOG_ROOT"
  "--json-report" "$CLI_REPORT"
)

log "launch attempt 1 and wait for two durable completed shards"
GENO_LEWM_RUN_ID="$ATTEMPT1_RUN_ID" "${CACHE_ARGV[@]}" \
  >"$RUNTIME_ROOT/attempt1.stdout" \
  2>"$RUNTIME_ROOT/attempt1.stderr" &
CACHE_PID=$!
WAIT_DEADLINE=$((SECONDS + SHARD_WAIT_TIMEOUT_SECONDS))
while :; do
  if [ -f "$CACHE_BUILD_STATE" ]; then
    DURABLE_SHARDS="$(completed_shards "$CACHE_BUILD_STATE")"
    if [ "$DURABLE_SHARDS" -ge "$MIN_DURABLE_SHARDS" ]; then
      break
    fi
  fi
  if ! kill -0 "$CACHE_PID" 2>/dev/null; then
    set +e
    wait "$CACHE_PID"
    ATTEMPT1_EARLY_RC=$?
    set -e
    CACHE_PID=""
    fatal "attempt 1 exited before interruption point with status $ATTEMPT1_EARLY_RC"
  fi
  if [ "$SECONDS" -ge "$WAIT_DEADLINE" ]; then
    fatal "timed out waiting for two durable cache shards"
  fi
  sleep 1
done

log "freeze attempt 1 and capture its incomplete durable contract"
kill -STOP "$CACHE_PID"
wait_until_stopped "$CACHE_PID"
test -s "$ATTEMPT1_LOG" || fatal "attempt 1 JSONL log is missing"
uv run --no-sync python -m tools.research.v03_cache_h200_proof capture-partial \
  "--bundle-dir" "$BUNDLE_DIR" \
  "--attempt-log" "$ATTEMPT1_LOG" \
  "--stopped-pid" "$CACHE_PID"

log "terminate the stopped process and require the POSIX SIGTERM status"
kill -TERM "$CACHE_PID"
kill -CONT "$CACHE_PID"
set +e
wait "$CACHE_PID"
ATTEMPT1_RC=$?
set -e
CACHE_PID=""
[ "$ATTEMPT1_RC" -eq 143 ] \
  || fatal "interrupted cache process returned $ATTEMPT1_RC instead of 143"
uv run --no-sync python -m tools.research.v03_cache_h200_proof finalize-interruption \
  "--bundle-dir" "$BUNDLE_DIR" \
  "--attempt-log" "$ATTEMPT1_LOG" \
  "--attempt-exit-code" "$ATTEMPT1_RC"

log "resume with byte-identical argv and a distinct external JSONL run identity"
GENO_LEWM_RUN_ID="$RESUME_RUN_ID" "${CACHE_ARGV[@]}" \
  >"$RUNTIME_ROOT/resume.stdout" \
  2>"$RUNTIME_ROOT/resume.stderr"
test -s "$RESUME_LOG" || fatal "resume JSONL log is missing"

log "retire the unheld runtime-only cache publication lock"
uv run --no-sync python -m tools.research.v03_cache_h200_proof retire-runtime-lock \
  "--bundle-dir" "$BUNDLE_DIR"

log "author and independently replay the closed outer proof bundle"
uv run --no-sync python -m tools.research.v03_cache_h200_proof author \
  "--bundle-dir" "$BUNDLE_DIR" \
  "--trace-repository" "$TRACE_REPOSITORY" \
  "--trace-revision" "$TRACE_REVISION" \
  "--trace-artifact-path" "$TRACE_ARTIFACT_PATH" \
  "--runtime-identity" "$RUNTIME_IDENTITY" \
  "--source-commit" "$COMMIT_SHA" \
  "--container-image" "$CONTAINER_IMAGE" \
  "--hardware-json" "$HARDWARE_JSON" \
  "--resume-log" "$RESUME_LOG"
uv run --no-sync python -m tools.research.v03_cache_h200_proof verify-existing \
  "--bundle-dir" "$BUNDLE_DIR"
LOCAL_CHECKSUMS_SHA256="$(sha256sum "$BUNDLE_DIR/SHA256SUMS" | cut -d' ' -f1)"

log "re-prove namespace absence and publish the entire closed outer bundle"
uv run --no-sync python -m tools.data.v03_gnomad_lock probe-namespace \
  "--repo-id" "$UPLOAD_REPOSITORY" \
  "--repo-type" "dataset" \
  "--namespace" "$PUBLISH_NAMESPACE" \
  >/dev/null
PUBLISH_REPORT="$(
  uv run --no-sync python -m tools.data.v03_gnomad_lock publish \
    "--repo-id" "$UPLOAD_REPOSITORY" \
    "--repo-type" "dataset" \
    "--namespace" "$PUBLISH_NAMESPACE" \
    "--publish-dir" "$BUNDLE_DIR" \
    "--commit-message" "publish v0.3 H200 cache interruption proof from $COMMIT_SHA"
)"
HUB_REVISION="${PUBLISH_REPORT#uploaded commit: }"
[[ "$HUB_REVISION" =~ ^[0-9a-f]{40}$ ]] \
  || fatal "proof publication did not return an immutable Hub revision"
test "$(sha256sum "$BUNDLE_DIR/SHA256SUMS" | cut -d' ' -f1)" = "$LOCAL_CHECKSUMS_SHA256" \
  || fatal "local proof bundle changed after checksum closure"

log "force-download and replay the exact published Hub revision"
test ! -e "$REMOTE_DOWNLOAD_ROOT" || fatal "remote replay directory already exists"
mkdir -p "$REMOTE_DOWNLOAD_ROOT"
hf download "$UPLOAD_REPOSITORY" \
  "--repo-type" "dataset" \
  "--revision" "$HUB_REVISION" \
  "--include" "$PUBLISH_NAMESPACE/**" \
  "--force-download" \
  "--local-dir" "$REMOTE_DOWNLOAD_ROOT"
REMOTE_BUNDLE="$REMOTE_DOWNLOAD_ROOT/$PUBLISH_NAMESPACE"
test -d "$REMOTE_BUNDLE" || fatal "exact published proof namespace was not downloaded"
cmp "$BUNDLE_DIR/SHA256SUMS" "$REMOTE_BUNDLE/SHA256SUMS" \
  || fatal "published proof checksum manifest differs from the local bundle"
uv run --no-sync python -m tools.research.v03_cache_h200_proof verify-existing \
  "--bundle-dir" "$REMOTE_BUNDLE"

printf '%s\n' "$PUBLISH_REPORT"
echo "GENO_LEWM_V03_CACHE_H200_PROOF_OK $HUB_REVISION $PUBLISH_NAMESPACE"
