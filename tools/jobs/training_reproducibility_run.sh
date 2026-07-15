#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
#
# Exact-SHA H200 N-D-D-N training reproducibility suite for issue #47.
#
# The four 500-step arms measure only bit-exact replay and the throughput cost
# of deterministic torch controls on one supported backend. They do not provide
# model-quality, benchmark-quality, biological, or clinical evidence.
#
# Exact HF Jobs submission recipe (run from the exact clean commit to test):
#   SHA="$(git rev-parse HEAD)"
#   RUN_ATTEMPT=1
#   IMAGE="ghcr.io/astral-sh/uv@sha256:35b0aa516fbcf6f18624919cfc38fa02ab3458e0ffcd3c03e932051b37f315db"
#   hf jobs run \
#     --flavor h200 \
#     --volume hf://models/HuggingFaceBio/Carbon-500M:/carbon:ro \
#     --secrets HF_TOKEN \
#     --env COMMIT_SHA="$SHA" \
#     --env RUN_ATTEMPT="$RUN_ATTEMPT" \
#     --timeout 4h \
#     --detach \
#     -- "$IMAGE" \
#     bash -lc 'set -euo pipefail; git clone https://github.com/AbdelStark/GenoLeWM.git /workspace/GenoLeWM; cd /workspace/GenoLeWM; git checkout --detach "$COMMIT_SHA"; test "$(git rev-parse HEAD)" = "$COMMIT_SHA"; uv sync --frozen --extra train; exec uv run --no-sync bash tools/jobs/training_reproducibility_run.sh'
set -euo pipefail

WORK="${WORK:-/tmp/geno-training-reproducibility}"
CARBON_DIR="${CARBON_DIR:-/carbon}"
MIN_CUDA_VRAM_GB="120"
EXPECTED_CARBON_RUNTIME_HASH="sha256:a1fd1dd20756c7248b7f9ca95c59c821f0329530fd49c6fea253a8df9a6a6311"
COMMIT_SHA="${COMMIT_SHA:?COMMIT_SHA is required}"
CONTAINER_IMAGE="${CONTAINER_IMAGE:-ghcr.io/astral-sh/uv@sha256:35b0aa516fbcf6f18624919cfc38fa02ab3458e0ffcd3c03e932051b37f315db}"

DETERMINISTIC_CONFIG="${DETERMINISTIC_CONFIG:-configs/reproducibility/train-carbon-500m-snv-deterministic-500.yaml}"
BASELINE_CONFIG="${BASELINE_CONFIG:-configs/reproducibility/train-carbon-500m-snv-baseline-500.yaml}"
DATASET_REFERENCE="${DATASET_REFERENCE:-configs/reproducibility/dataset-reference-r2.json}"
DATASET_REPO="${DATASET_REPO:-abdelstark/geno-lewm-runs}"
DATASET_REVISION="${DATASET_REVISION:-1200467a6b940cb5b1230d9a7db0be74e51bd50d}"
DATASET_PATH="${DATASET_PATH:-geno-lewm-l2-p1-smoke-304128e4d4f3-50-r2/dataset}"

STEPS="${STEPS:-500}"
EXPECTED_SAMPLE_COUNT="${EXPECTED_SAMPLE_COUNT:-4000}"
MAX_THROUGHPUT_DROP="${MAX_THROUGHPUT_DROP:-0.15}"
MAX_REPEAT_SPREAD="${MAX_REPEAT_SPREAD:-0.05}"
RUN_ATTEMPT="${RUN_ATTEMPT:-1}"
RUN_NAME="${RUN_NAME:-geno-lewm-repro-h200-${COMMIT_SHA:0:12}-500-r${RUN_ATTEMPT}}"
UPLOAD_REPO="${UPLOAD_REPO:-abdelstark/geno-lewm-runs}"

CURRENT_STAGE="bootstrap"
RUN_PROTECTED=0
CANDIDATE_UPLOAD_ALLOWED=0

log() { echo "=== $* ==="; }

copy_available_tree() {
  local source=$1
  local destination=$2
  if [ -d "$source" ]; then
    mkdir -p "$destination"
    cp -a "$source/." "$destination/"
  fi
}

write_failure_evidence() {
  local exit_code=$1
  local failed_stage=$2
  local output=$3
  python - "$exit_code" "$failed_stage" "$output" "$COMMIT_SHA" "$RUN_NAME" <<'PY'
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

exit_code, stage, output, commit_sha, run_name = sys.argv[1:]
payload = {
    "schema_version": "1.0.0",
    "generated_by": "tools.jobs.training_reproducibility_run",
    "generated_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
    "ok": False,
    "exit_code": int(exit_code),
    "stage": stage,
    "commit_sha": commit_sha,
    "run_name": run_name,
    "claim_boundary": (
        "The N-D-D-N reproducibility contract did not complete successfully. "
        "This candidate-negative bundle is diagnostic evidence, never a success marker."
    ),
}
Path(output).write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
PY
}

upload_candidate_on_failure() {
  local exit_code=$?
  local failed_stage=$CURRENT_STAGE
  local candidate_bundle_rc=0
  local upload_rc=0
  local candidate_nonce
  local candidate_root="$RUN_NAME/candidate-negative"
  local candidate_namespace
  trap - EXIT
  if [ "$exit_code" -eq 0 ] || [ "$RUN_PROTECTED" -eq 1 ]; then
    return "$exit_code"
  fi

  set +e
  candidate_nonce="$(python -c 'import uuid; print(uuid.uuid4().hex)')"
  if [ -z "$candidate_nonce" ]; then
    candidate_nonce="${BASHPID}-${RANDOM}"
  fi
  candidate_namespace="$candidate_root/$candidate_nonce"
  log "assemble fail-closed candidate evidence for $failed_stage (exit $exit_code)"
  rm -rf "$WORK/public"
  mkdir -p "$WORK/public/evidence" "$WORK/public/contract"
  copy_available_tree "$WORK/evidence" "$WORK/public/evidence"
  copy_available_tree "$WORK/runs" "$WORK/public/runs"
  copy_available_tree "$WORK/dataset" "$WORK/public/dataset"
  copy_available_tree "$WORK/download" "$WORK/public/download"
  find "$WORK/public" -type d -name .cache -prune -exec rm -rf {} +
  [ ! -f "$DETERMINISTIC_CONFIG" ] || cp "$DETERMINISTIC_CONFIG" "$WORK/public/contract/"
  [ ! -f "$BASELINE_CONFIG" ] || cp "$BASELINE_CONFIG" "$WORK/public/contract/"
  [ ! -f "$DATASET_REFERENCE" ] || cp "$DATASET_REFERENCE" "$WORK/public/contract/"
  write_failure_evidence \
    "$exit_code" \
    "$failed_stage" \
    "$WORK/public/evidence/failure.json"
  (
    cd "$WORK/public" || exit 1
    find . -type f ! -path ./SHA256SUMS -print0 \
      | sort -z \
      | xargs -0 sha256sum > SHA256SUMS
    sha256sum -c SHA256SUMS
  )
  candidate_bundle_rc=$?

  if [ "$candidate_bundle_rc" -ne 0 ]; then
    echo "FATAL: candidate-negative checksum closure failed with exit $candidate_bundle_rc" >&2
  elif [ "$CANDIDATE_UPLOAD_ALLOWED" -eq 1 ] && [ -n "${HF_TOKEN:-}" ]; then
    hf upload \
      "$UPLOAD_REPO" \
      "$WORK/public" \
      "$candidate_namespace" \
      --repo-type model
    upload_rc=$?
    if [ "$upload_rc" -eq 0 ]; then
      RUN_PROTECTED=1
      log "published candidate-negative evidence for $failed_stage"
    else
      echo "FATAL: candidate-negative upload failed with exit $upload_rc" >&2
    fi
  else
    echo "FATAL: candidate upload unavailable; evidence remains at $WORK/public" >&2
  fi
  echo "GENO_LEWM_TRAINING_REPRODUCIBILITY_NOT_PROVEN $RUN_NAME" >&2
  return "$exit_code"
}

trap upload_candidate_on_failure EXIT

CURRENT_STAGE="contract_preflight"
log "validate immutable N-D-D-N launch before writes"
test -n "${HF_TOKEN:-}" || { echo "FATAL: HF_TOKEN is required" >&2; exit 1; }
set +e
JOB_PREFLIGHT_REPORT="$(
  python -m tools.research.training_reproducibility_preflight \
    --repo-root . \
    --deterministic-config "$DETERMINISTIC_CONFIG" \
    --baseline-config "$BASELINE_CONFIG" \
    --dataset-reference "$DATASET_REFERENCE" \
    --expected-commit-sha "$COMMIT_SHA" \
    --run-name "$RUN_NAME" \
    --run-attempt "$RUN_ATTEMPT" \
    --steps "$STEPS" \
    --expected-sample-count "$EXPECTED_SAMPLE_COUNT" \
    --dataset-repo "$DATASET_REPO" \
    --dataset-revision "$DATASET_REVISION" \
    --dataset-path "$DATASET_PATH" \
    --carbon-model-dir "$CARBON_DIR" \
    --expected-carbon-runtime-hash "$EXPECTED_CARBON_RUNTIME_HASH" \
    --upload-repo "$UPLOAD_REPO" \
    --container-image "$CONTAINER_IMAGE" \
    --min-cuda-vram-gb "$MIN_CUDA_VRAM_GB" \
    --max-throughput-drop "$MAX_THROUGHPUT_DROP" \
    --max-repeat-spread "$MAX_REPEAT_SPREAD"
)"
preflight_rc=$?
set -e
if [ "$preflight_rc" -ne 0 ]; then
  printf '%s\n' "$JOB_PREFLIGHT_REPORT" >&2
  exit "$preflight_rc"
fi

CURRENT_STAGE="remote_namespace_preflight"
REMOTE_STATUS="$(
  curl -sS -L --retry 3 \
    -H "Authorization: Bearer $HF_TOKEN" \
    -o /dev/null \
    -w '%{http_code}' \
    "https://huggingface.co/api/models/$UPLOAD_REPO/tree/main/$RUN_NAME?recursive=false"
)"
case "$REMOTE_STATUS" in
  404) CANDIDATE_UPLOAD_ALLOWED=1 ;;
  200)
    echo "FATAL: immutable reproducibility namespace already exists: $RUN_NAME" >&2
    exit 1
    ;;
  *)
    echo "FATAL: could not establish remote namespace availability (HTTP $REMOTE_STATUS)" >&2
    exit 1
    ;;
esac

CURRENT_STAGE="workspace_initialization"
mkdir -p "$WORK/download" "$WORK/evidence" "$WORK/runs"
printf '%s\n' "$JOB_PREFLIGHT_REPORT" > "$WORK/evidence/job_contract_preflight.json"

CURRENT_STAGE="accelerator_preflight"
test -d "$CARBON_DIR" || { echo "FATAL: Carbon checkpoint is not mounted at $CARBON_DIR" >&2; exit 1; }
python - \
  "$MIN_CUDA_VRAM_GB" \
  "$CARBON_DIR" \
  "$EXPECTED_CARBON_RUNTIME_HASH" \
  "$COMMIT_SHA" \
  "$WORK/evidence/runtime_preflight.json" \
  "$RUN_NAME" <<'PY'
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import torch

from geno_lewm.encoder._identity import encoder_runtime_hash

minimum_gb = float(sys.argv[1])
carbon_dir = Path(sys.argv[2])
expected_runtime_hash = sys.argv[3]
commit_sha = sys.argv[4]
output_path = Path(sys.argv[5])
run_name = sys.argv[6]
observed_runtime_hash = encoder_runtime_hash(carbon_dir)
if observed_runtime_hash != expected_runtime_hash:
    raise SystemExit(
        "FATAL: Carbon runtime hash mismatch: "
        f"expected {expected_runtime_hash}, observed {observed_runtime_hash}"
    )
if not torch.cuda.is_available():
    raise SystemExit("FATAL: CUDA is required for the reproducibility suite")
properties = torch.cuda.get_device_properties(0)
observed_gb = properties.total_memory / (1024**3)
print(
    "torch",
    torch.__version__,
    "cuda_device",
    properties.name,
    f"{observed_gb:.1f} GiB",
    "carbon_runtime",
    observed_runtime_hash,
)
if "H200" not in properties.name:
    raise SystemExit(f"FATAL: reproducibility suite requires H200, observed {properties.name}")
if observed_gb < minimum_gb:
    raise SystemExit(
        f"FATAL: CUDA device has {observed_gb:.1f} GiB; need at least {minimum_gb:.1f} GiB"
    )
payload = {
    "schema_version": "1.0.0",
    "generated_by": "tools.jobs.training_reproducibility_run",
    "generated_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
    "ok": True,
    "source_commit_sha": commit_sha,
    "run_name": run_name,
    "carbon_runtime_hash": observed_runtime_hash,
    "expected_carbon_runtime_hash": expected_runtime_hash,
    "accelerator": {
        "requested_device": "cuda",
        "available": True,
        "device_count": torch.cuda.device_count(),
        "device_name": properties.name,
        "total_memory_bytes": properties.total_memory,
        "minimum_memory_gb": minimum_gb,
    },
    "torch_version": torch.__version__,
}
output_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
PY
nvidia-smi

CURRENT_STAGE="dataset_download"
log "download immutable r2 dataset package"
hf download "$DATASET_REPO" \
  --repo-type model \
  --revision "$DATASET_REVISION" \
  --include "$DATASET_PATH/**" \
  --local-dir "$WORK/download"
cp -a "$WORK/download/$DATASET_PATH" "$WORK/dataset"

CURRENT_STAGE="dataset_integrity"
python - "$DATASET_REFERENCE" "$WORK/dataset" <<'PY'
import hashlib
import json
import sys
from pathlib import Path

reference = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
dataset = Path(sys.argv[2])
expected = reference.get("expected_files")
if not isinstance(expected, dict) or not expected:
    raise SystemExit("FATAL: dataset reference expected_files is missing")
for relative, wanted in sorted(expected.items()):
    path = dataset / relative
    if not path.is_file():
        raise SystemExit(f"FATAL: immutable dataset file is missing: {relative}")
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    observed = f"sha256:{digest}"
    if observed != wanted:
        raise SystemExit(
            f"FATAL: immutable dataset hash mismatch for {relative}: "
            f"expected {wanted}, observed {observed}"
        )
manifest = json.loads((dataset / "dataset_manifest.json").read_text(encoding="utf-8"))
if manifest.get("snapshot_id") != reference.get("snapshot_id"):
    raise SystemExit("FATAL: immutable dataset snapshot_id mismatch")
print("immutable dataset verified", reference["revision"], reference["path"])
PY
(
  cd "$WORK/dataset"
  sha256sum -c SHA256SUMS
)

run_arm() {
  local label=$1
  local config=$2
  local run_dir="$WORK/runs/$label"
  test ! -e "$run_dir" || { echo "FATAL: arm directory already exists: $run_dir" >&2; exit 1; }

  CURRENT_STAGE="arm.$label.preflight"
  log "preflight fresh arm $label"
  env -u CUBLAS_WORKSPACE_CONFIG \
    GENO_LEWM_CACHE="$WORK/cache/$label" \
    geno-lewm-train --carbon-preflight \
    --run-dir "$run_dir" \
    --dataset-dir "$WORK/dataset" \
    --carbon-model-dir "$CARBON_DIR" \
    --training-config "$config" \
    --min-cuda-vram-gb "$MIN_CUDA_VRAM_GB" \
    --no-banner --quiet

  CURRENT_STAGE="arm.$label.train"
  log "train fresh arm $label ($STEPS steps)"
  env -u CUBLAS_WORKSPACE_CONFIG \
    GENO_LEWM_CACHE="$WORK/cache/$label" \
    geno-lewm-train --carbon-train \
    --run-dir "$run_dir" \
    --dataset-dir "$WORK/dataset" \
    --carbon-model-dir "$CARBON_DIR" \
    --training-config "$config" \
    --steps "$STEPS" \
    --min-cuda-vram-gb "$MIN_CUDA_VRAM_GB" \
    --package-release-run \
    --no-banner --quiet

  CURRENT_STAGE="arm.$label.checksum"
  (
    cd "$run_dir"
    sha256sum -c training_run_SHA256SUMS
  )
}

# Counterbalance order effects with two independent repeats of each mode.
# Every call launches fresh preflight and training processes, with isolated
# caches and output directories.
run_arm "baseline-a" "$BASELINE_CONFIG"
run_arm "deterministic-a" "$DETERMINISTIC_CONFIG"
run_arm "deterministic-b" "$DETERMINISTIC_CONFIG"
run_arm "baseline-b" "$BASELINE_CONFIG"

CURRENT_STAGE="reproducibility_verifier"
log "verify bit-exact replay and counterbalanced deterministic throughput contract"
python -m tools.release.training_reproducibility \
  --baseline-run-a "$WORK/runs/baseline-a" \
  --deterministic-run-a "$WORK/runs/deterministic-a" \
  --deterministic-run-b "$WORK/runs/deterministic-b" \
  --baseline-run-b "$WORK/runs/baseline-b" \
  --output "$WORK/evidence/training_reproducibility_report.json" \
  --max-throughput-drop "$MAX_THROUGHPUT_DROP" \
  --max-repeat-spread "$MAX_REPEAT_SPREAD" \
  --require-preflight

CURRENT_STAGE="success_bundle_assembly"
rm -rf "$WORK/public"
mkdir -p "$WORK/public/evidence" "$WORK/public/runs" "$WORK/public/contract"
cp "$WORK/evidence/job_contract_preflight.json" "$WORK/public/evidence/"
cp "$WORK/evidence/runtime_preflight.json" "$WORK/public/evidence/"
cp "$WORK/evidence/training_reproducibility_report.json" "$WORK/public/evidence/"
cp -a "$WORK/runs/." "$WORK/public/runs/"
cp "$DETERMINISTIC_CONFIG" "$WORK/public/contract/"
cp "$BASELINE_CONFIG" "$WORK/public/contract/"
cp "$DATASET_REFERENCE" "$WORK/public/contract/"
(
  cd "$WORK/public"
  find evidence runs contract -type f -print0 \
    | sort -z \
    | xargs -0 sha256sum > SHA256SUMS
)

CURRENT_STAGE="atomic_success_publish_and_verify"
log "conditionally publish one complete success namespace and verify its immutable commit"
test ! -e "$WORK/verified-success" || {
  echo "FATAL: immutable verification directory already exists: $WORK/verified-success" >&2
  exit 1
}
ATOMIC_PUBLISH_REPORT="$(
  python -m tools.release.atomic_hub_publish \
    --bundle-dir "$WORK/public" \
    --repo-id "$UPLOAD_REPO" \
    --repo-type model \
    --run-name "$RUN_NAME" \
    --source-commit-sha "$COMMIT_SHA" \
    --verification-dir "$WORK/verified-success" \
    --max-attempts 3
)"
printf '%s\n' "$ATOMIC_PUBLISH_REPORT" > "$WORK/evidence/atomic_success_publish_report.json"
RUN_PROTECTED=1

printf '%s\n' "$ATOMIC_PUBLISH_REPORT"
echo "GENO_LEWM_TRAINING_REPRODUCIBILITY_OK $RUN_NAME"
