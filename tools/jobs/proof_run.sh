#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
#
# Exact-SHA correction-control smoke run for Hugging Face Jobs.
#
# A standalone run proves only that the corrected L2-normalized Phase-1 pipeline
# can execute 50 finite optimizer steps from immutable upstream inputs and
# archive coherent artifacts. REPLAY_REFERENCE_ATTEMPT additionally validates
# only bit-exact deterministic-pair evidence against one completed prior run;
# neither mode establishes convergence, deterministic throughput, benchmark
# performance, or clinical validity.
set -euo pipefail

WORK="${WORK:-/tmp/geno-correction-control}"
CARBON_DIR="${CARBON_DIR:-/carbon}"
MIN_CUDA_VRAM_GB="120"
COMMIT_SHA="${COMMIT_SHA:?COMMIT_SHA is required}"
CONTAINER_IMAGE="${CONTAINER_IMAGE:-ghcr.io/astral-sh/uv@sha256:35b0aa516fbcf6f18624919cfc38fa02ab3458e0ffcd3c03e932051b37f315db}"

CONFIG="${CONFIG:-configs/correction_control/train-carbon-500m-snv-l2-smoke-v1.yaml}"
SPEC_SRC="${SPEC_SRC:-configs/correction_control/dataset-snapshot-snv-l2-smoke-v1.json}"
RUN_ID="correction-control-l2-p1-smoke-v1"
SNAPSHOT_ID="geno-lewm-data-correction-control-l2-p1-proof-v1"
RUN_ATTEMPT="${RUN_ATTEMPT:-1}"
RUN_NAME="${RUN_NAME:-geno-lewm-l2-p1-smoke-${COMMIT_SHA:0:12}-50-r${RUN_ATTEMPT}}"
REPLAY_REFERENCE_ATTEMPT="${REPLAY_REFERENCE_ATTEMPT:-}"
REFERENCE_RUN_NAME=""
UPLOAD_REPO="${UPLOAD_REPO:-abdelstark/geno-lewm-runs}"

CORPUS_REVISION="${CORPUS_REVISION:-cb4c13a78102933b3a6ac65734d326f7b431d9b7}"
CARBON_CONFIG="${CARBON_CONFIG:-eukaryote_generator_10B_subset}"
CARBON_SOURCE="${CARBON_SOURCE:-eukaryotic_genes}"
MAX_WINDOWS="${MAX_WINDOWS:-512}"
CLINVAR_LINES="${CLINVAR_LINES:-60000}"
GNOMAD_LINES="${GNOMAD_LINES:-60000}"
STEPS="${STEPS:-50}"
TUPLE_THROUGHPUT_SAMPLES="${TUPLE_THROUGHPUT_SAMPLES:-400}"
MIN_TUPLES_PER_SECOND="${MIN_TUPLES_PER_SECOND:-5000}"
WINDOW_BP="${WINDOW_BP:-4096}"
HOLDOUT_CHROM="${HOLDOUT_CHROM:-22}"

CLINVAR_URL="${CLINVAR_URL:-https://ftp.ncbi.nlm.nih.gov/pub/clinvar/vcf_GRCh38/archive_2.0/2026/clinvar_20260415.vcf.gz}"
GNOMAD_URL="${GNOMAD_URL:-https://storage.googleapis.com/gcp-public-data--gnomad/release/4.1/vcf/exomes/gnomad.exomes.v4.1.sites.chr22.vcf.bgz?generation=1713312296186865}"
GNOMAD_METADATA_URL="https://storage.googleapis.com/storage/v1/b/gcp-public-data--gnomad/o/release%2F4.1%2Fvcf%2Fexomes%2Fgnomad.exomes.v4.1.sites.chr22.vcf.bgz?generation=1713312296186865"

CLINVAR_MD5="e63b5c3a046010c098cc70e81bebaa8d"
GNOMAD_GENERATION="1713312296186865"
GNOMAD_MD5="dcf191563e69054a71bd4dc77862799a"
GNOMAD_SIZE_BYTES="5060347554"

RUN_PROTECTED=0

log() { echo "=== $* ==="; }

upload_partial_run_on_failure() {
  local rc=$?
  if [ "$rc" -ne 0 ] && [ "$RUN_PROTECTED" -eq 0 ] \
    && [ -f "$WORK/run/predictor_checkpoint.pt" ]; then
    log "upload failed-run checkpoint and evidence to $UPLOAD_REPO/$RUN_NAME/run-partial"
    hf upload "$UPLOAD_REPO" "$WORK/run" "$RUN_NAME/run-partial" --repo-type model || true
  fi
  return "$rc"
}
trap upload_partial_run_on_failure EXIT

verify_digest() {
  local path=$1
  local algorithm=$2
  local expected=$3
  python - "$path" "$algorithm" "$expected" <<'PY'
import hashlib
import sys
from pathlib import Path

path = Path(sys.argv[1])
algorithm = sys.argv[2]
expected = sys.argv[3]
hasher = hashlib.md5(usedforsecurity=False) if algorithm == "md5" else hashlib.sha256()
with path.open("rb") as stream:
    for chunk in iter(lambda: stream.read(1024 * 1024), b""):
        hasher.update(chunk)
observed = hasher.hexdigest()
if observed != expected:
    raise SystemExit(
        f"FATAL: {path.name} {algorithm} mismatch: expected {expected}, observed {observed}"
    )
print(f"{path.name} {algorithm} verified: {observed}")
PY
}

log "validate immutable correction-control launch before writes"
test -n "${HF_TOKEN:-}" || { echo "FATAL: HF_TOKEN is required" >&2; exit 1; }
set +e
JOB_CONTRACT_REPORT="$(
  python -m tools.research.correction_control_preflight \
    --repo-root . \
    --config "$CONFIG" \
    --snapshot "$SPEC_SRC" \
    --expected-commit-sha "$COMMIT_SHA" \
    --run-name "$RUN_NAME" \
    --run-attempt "$RUN_ATTEMPT" \
    --steps "$STEPS" \
    --max-windows "$MAX_WINDOWS" \
    --clinvar-lines "$CLINVAR_LINES" \
    --gnomad-lines "$GNOMAD_LINES" \
    --tuple-throughput-samples "$TUPLE_THROUGHPUT_SAMPLES" \
    --window-bp "$WINDOW_BP" \
    --holdout-chrom "$HOLDOUT_CHROM" \
    --carbon-model-dir "$CARBON_DIR" \
    --carbon-config "$CARBON_CONFIG" \
    --carbon-source "$CARBON_SOURCE" \
    --corpus-revision "$CORPUS_REVISION" \
    --container-image "$CONTAINER_IMAGE" \
    --clinvar-url "$CLINVAR_URL" \
    --gnomad-url "$GNOMAD_URL"
)"
job_contract_rc=$?
set -e
if [ "$job_contract_rc" -ne 0 ]; then
  printf '%s\n' "$JOB_CONTRACT_REPORT" >&2
  exit "$job_contract_rc"
fi

REMOTE_AUDIT_URL="https://huggingface.co/$UPLOAD_REPO/resolve/main/$RUN_NAME/evidence/state_contract_audit.json"
REMOTE_STATUS="$(
  curl -sS -L --retry 3 \
    -H "Authorization: Bearer $HF_TOKEN" \
    -o /dev/null \
    -w '%{http_code}' \
    "$REMOTE_AUDIT_URL"
)"
case "$REMOTE_STATUS" in
  404) ;;
  200)
    echo "FATAL: immutable run namespace already exists: $RUN_NAME" >&2
    exit 1
    ;;
  *)
    echo "FATAL: could not establish remote namespace availability (HTTP $REMOTE_STATUS)" >&2
    exit 1
    ;;
esac

if [ -n "$REPLAY_REFERENCE_ATTEMPT" ]; then
  case "$REPLAY_REFERENCE_ATTEMPT" in
    *[!0-9]* | 0 | 0*)
      echo "FATAL: REPLAY_REFERENCE_ATTEMPT must be a positive integer" >&2
      exit 1
      ;;
  esac
  if [ "$REPLAY_REFERENCE_ATTEMPT" -ge "$RUN_ATTEMPT" ]; then
    echo "FATAL: REPLAY_REFERENCE_ATTEMPT must precede RUN_ATTEMPT" >&2
    exit 1
  fi

  REFERENCE_RUN_NAME="geno-lewm-l2-p1-smoke-${COMMIT_SHA:0:12}-50-r${REPLAY_REFERENCE_ATTEMPT}"
  REFERENCE_POSTFLIGHT_PATH="$REFERENCE_RUN_NAME/run/correction_control/correction_control_postflight.json"
  REFERENCE_POSTFLIGHT_URL="https://huggingface.co/$UPLOAD_REPO/resolve/main/$REFERENCE_POSTFLIGHT_PATH"
  REFERENCE_POSTFLIGHT_STATUS="$(
    curl -sS -L --retry 3 \
      -H "Authorization: Bearer $HF_TOKEN" \
      -o /dev/null \
      -w '%{http_code}' \
      "$REFERENCE_POSTFLIGHT_URL"
  )"
  if [ "$REFERENCE_POSTFLIGHT_STATUS" != "200" ]; then
    echo "FATAL: completed replay reference is unavailable (HTTP $REFERENCE_POSTFLIGHT_STATUS)" >&2
    exit 1
  fi
fi

test -d "$CARBON_DIR" || { echo "FATAL: Carbon checkpoint is not mounted at $CARBON_DIR" >&2; exit 1; }
python - "$MIN_CUDA_VRAM_GB" <<'PY'
import sys

import torch

minimum_gb = float(sys.argv[1])
if not torch.cuda.is_available():
    raise SystemExit("FATAL: CUDA is required for the correction-control smoke run")
properties = torch.cuda.get_device_properties(0)
observed_gb = properties.total_memory / (1024**3)
print("torch", torch.__version__, "cuda_device", properties.name, f"{observed_gb:.1f} GiB")
if "H200" not in properties.name:
    raise SystemExit(f"FATAL: correction-control job requires H200, observed {properties.name}")
if observed_gb < minimum_gb:
    raise SystemExit(
        f"FATAL: CUDA device has {observed_gb:.1f} GiB; need at least {minimum_gb:.1f} GiB"
    )
PY
nvidia-smi

mkdir -p \
  "$WORK/downloads" \
  "$WORK/evidence" \
  "$WORK/inputs/clinvar" \
  "$WORK/inputs/gnomad" \
  "$WORK/inputs/carbon"
printf '%s\n' "$JOB_CONTRACT_REPORT" > "$WORK/evidence/job_contract_preflight.json"
cp "$SPEC_SRC" "$WORK/dataset-snapshot-snv.json"

log "audit corrected Carbon state contract at ${WINDOW_BP} bp"
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
  python -m tools.research.state_contract_audit \
    --carbon-model-dir "$CARBON_DIR" \
    --commit-sha "$COMMIT_SHA" \
    --output-json "$WORK/evidence/state_contract_audit.json" \
    --window-bp "$WINDOW_BP" \
    --device cuda \
    --no-trust-remote-code
hf upload \
  "$UPLOAD_REPO" \
  "$WORK/evidence/state_contract_audit.json" \
  "$RUN_NAME/evidence/state_contract_audit.json" \
  --repo-type model

log "stage and verify archived ClinVar GRCh38"
CLINVAR_SOURCE="$WORK/downloads/clinvar_20260415.vcf.gz"
CLINVAR_OUT="$WORK/inputs/clinvar/clinvar-2026-04-15-snv.vcf.gz"
curl -fsSL --retry 5 --retry-all-errors "$CLINVAR_URL" -o "$CLINVAR_SOURCE"
verify_digest "$CLINVAR_SOURCE" md5 "$CLINVAR_MD5"
set +o pipefail
zcat "$CLINVAR_SOURCE" 2>/dev/null \
  | awk -F'\t' -v c="$HOLDOUT_CHROM" '/^#/ || ($1 != c && $1 != "chr" c)' \
  | head -n "$CLINVAR_LINES" \
  | gzip -n > "$CLINVAR_OUT"
set -o pipefail
test -s "$CLINVAR_OUT" || { echo "FATAL: ClinVar holdout filter produced no rows" >&2; exit 1; }
CLINVAR_OBSERVED_LINES="$(zcat "$CLINVAR_OUT" 2>/dev/null | wc -l | tr -d ' ')"
test "$CLINVAR_OBSERVED_LINES" = "$CLINVAR_LINES" || {
  echo "FATAL: expected $CLINVAR_LINES ClinVar lines, observed $CLINVAR_OBSERVED_LINES" >&2
  exit 1
}

log "verify generation-pinned gnomAD object metadata"
GNOMAD_METADATA="$WORK/evidence/gnomad_object_metadata.json"
curl -fsSL --retry 5 --retry-all-errors "$GNOMAD_METADATA_URL" -o "$GNOMAD_METADATA"
python - "$GNOMAD_METADATA" "$GNOMAD_GENERATION" "$GNOMAD_MD5" "$GNOMAD_SIZE_BYTES" <<'PY'
import base64
import json
import sys
from pathlib import Path

payload = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
expected_generation, expected_md5, expected_size = sys.argv[2:]
observed_md5 = base64.b64decode(payload.get("md5Hash", ""), validate=True).hex()
observed = {
    "generation": payload.get("generation"),
    "md5": observed_md5,
    "size": payload.get("size"),
}
expected = {
    "generation": expected_generation,
    "md5": expected_md5,
    "size": expected_size,
}
if observed != expected:
    raise SystemExit(f"FATAL: gnomAD object identity mismatch: expected {expected}, observed {observed}")
print("gnomAD object identity verified", json.dumps(observed, sort_keys=True))
PY

log "stage gnomAD chr22 prefix ($GNOMAD_LINES decompressed lines)"
GNOMAD_OUT="$WORK/inputs/gnomad/gnomad-v4.1-snv.vcf.gz"
set +o pipefail
curl -fsSL --retry 5 "$GNOMAD_URL" \
  | zcat 2>/dev/null \
  | head -n "$GNOMAD_LINES" \
  | gzip -n > "$GNOMAD_OUT"
set -o pipefail
test -s "$GNOMAD_OUT" || { echo "FATAL: gnomAD subset is empty" >&2; exit 1; }
GNOMAD_OBSERVED_LINES="$(zcat "$GNOMAD_OUT" 2>/dev/null | wc -l | tr -d ' ')"
test "$GNOMAD_OBSERVED_LINES" = "$GNOMAD_LINES" || {
  echo "FATAL: expected $GNOMAD_LINES gnomAD lines, observed $GNOMAD_OBSERVED_LINES" >&2
  exit 1
}

log "stage pinned Carbon corpus windows"
CARBON_OUT="$WORK/inputs/carbon/source-mix-windows.jsonl"
set +e
python -m tools.data.carbon_windows \
  --revision "$CORPUS_REVISION" \
  --dataset-config "$CARBON_CONFIG" \
  --default-source "$CARBON_SOURCE" \
  --max-windows "$MAX_WINDOWS" \
  --window-bp "$WINDOW_BP" \
  --output "$CARBON_OUT"
carbon_rc=$?
set -e
test -s "$CARBON_OUT" || {
  echo "FATAL: Carbon window staging produced no rows (rc=$carbon_rc)" >&2
  exit 1
}
CARBON_OBSERVED_WINDOWS="$(wc -l < "$CARBON_OUT" | tr -d ' ')"
test "$CARBON_OBSERVED_WINDOWS" = "$MAX_WINDOWS" || {
  echo "FATAL: expected $MAX_WINDOWS Carbon windows, observed $CARBON_OBSERVED_WINDOWS" >&2
  exit 1
}

log "write verified upstream identity receipt"
python - \
  "$WORK/evidence/source_identity_report.json" \
  "$CLINVAR_SOURCE" "$CLINVAR_OUT" \
  "$GNOMAD_METADATA" "$GNOMAD_OUT" \
  "$CARBON_OUT" \
  "$CLINVAR_MD5" "$GNOMAD_GENERATION" "$GNOMAD_MD5" "$GNOMAD_SIZE_BYTES" \
  "$CORPUS_REVISION" "$CARBON_CONFIG" "$CARBON_SOURCE" \
  "$CLINVAR_URL" "$GNOMAD_URL" "$COMMIT_SHA" "$RUN_NAME" "$SNAPSHOT_ID" \
  "$CLINVAR_LINES" "$GNOMAD_LINES" "$WINDOW_BP" <<'PY'
import base64
import gzip
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

(
    output,
    clinvar_source,
    clinvar_filtered,
    gnomad_metadata,
    gnomad_subset,
    carbon,
    expected_clinvar_md5,
    expected_gnomad_generation,
    expected_gnomad_md5,
    expected_gnomad_size,
    corpus_revision,
    carbon_config,
    carbon_source,
    clinvar_url,
    gnomad_url,
    commit_sha,
    run_name,
    snapshot_id,
    expected_clinvar_lines,
    expected_gnomad_lines,
    window_bp,
) = sys.argv[1:]


def digest(path: str, algorithm: str) -> str:
    hasher = hashlib.md5(usedforsecurity=False) if algorithm == "md5" else hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def identity(path: str) -> dict[str, object]:
    value = Path(path)
    return {
        "path": value.name,
        "sha256": f"sha256:{digest(path, 'sha256')}",
        "size_bytes": value.stat().st_size,
    }


metadata = json.loads(Path(gnomad_metadata).read_text(encoding="utf-8"))
observed = {
    "clinvar_md5": digest(clinvar_source, "md5"),
    "gnomad_generation": metadata.get("generation"),
    "gnomad_md5": base64.b64decode(metadata.get("md5Hash", ""), validate=True).hex(),
    "gnomad_size_bytes": metadata.get("size"),
}
expected = {
    "clinvar_md5": expected_clinvar_md5,
    "gnomad_generation": expected_gnomad_generation,
    "gnomad_md5": expected_gnomad_md5,
    "gnomad_size_bytes": expected_gnomad_size,
}
if observed != expected:
    raise SystemExit(f"FATAL: source identity mismatch: expected {expected}, observed {observed}")
with gzip.open(gnomad_subset, "rt", encoding="utf-8", errors="strict") as stream:
    gnomad_lines = sum(1 for _ in stream)
with gzip.open(clinvar_filtered, "rt", encoding="utf-8", errors="strict") as stream:
    clinvar_lines = sum(1 for _ in stream)
with Path(carbon).open("rt", encoding="utf-8") as stream:
    carbon_windows = sum(1 for _ in stream)
if clinvar_lines != int(expected_clinvar_lines) or gnomad_lines != int(expected_gnomad_lines):
    raise SystemExit(
        "FATAL: staged line-count mismatch: "
        f"ClinVar {clinvar_lines}/{expected_clinvar_lines}, "
        f"gnomAD {gnomad_lines}/{expected_gnomad_lines}"
    )
payload = {
    "schema_version": "1.0.0",
    "generated_by": "tools.jobs.proof_run.source_identity",
    "generated_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
    "ok": True,
    "commit_sha": commit_sha,
    "run_name": run_name,
    "dataset_snapshot_id": snapshot_id,
    "training_contract": {
        "active_window_source": "carbon",
        "window_bp": int(window_bp),
        "action_sub_encoders": ["snv"],
        "actions_per_window": 8,
        "absolute_variant_fallback": "synthetic_snv",
    },
    "sources": {
        "carbon_corpus": {
            "revision": corpus_revision,
            "dataset_config": carbon_config,
            "default_source": carbon_source,
            "windows": carbon_windows,
            "artifact": identity(carbon),
        },
        "clinvar": {
            "url": clinvar_url,
            "md5": observed["clinvar_md5"],
            "subset_lines": clinvar_lines,
            "archive": identity(clinvar_source),
            "filtered_artifact": identity(clinvar_filtered),
        },
        "gnomad": {
            "url": gnomad_url,
            "generation": observed["gnomad_generation"],
            "md5": observed["gnomad_md5"],
            "size_bytes": int(str(observed["gnomad_size_bytes"])),
            "subset_lines": gnomad_lines,
            "subset_artifact": identity(gnomad_subset),
        },
    },
    "claim_boundary": (
        "This report verifies source object identity and staged-file provenance only; "
        "it does not establish dataset representativeness or model quality."
    ),
}
Path(output).write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
PY

log "build correction-control dataset snapshot"
python -m tools.release.dataset_snapshot \
  --spec-json "$WORK/dataset-snapshot-snv.json" \
  --dataset-dir "$WORK/dataset"

log "generic tuple-builder throughput gate ($TUPLE_THROUGHPUT_SAMPLES samples)"
python -m tools.data.tuple_throughput \
  --dataset-dir "$WORK/dataset" \
  --samples "$TUPLE_THROUGHPUT_SAMPLES" \
  --min-tuples-per-second "$MIN_TUPLES_PER_SECOND" \
  --output "$WORK/evidence/tuple_throughput_report.json"

log "Carbon training preflight"
geno-lewm-train --carbon-preflight \
  --run-dir "$WORK/run" \
  --dataset-dir "$WORK/dataset" \
  --carbon-model-dir "$CARBON_DIR" \
  --training-config "$CONFIG" \
  --min-cuda-vram-gb "$MIN_CUDA_VRAM_GB" \
  --no-banner --quiet

log "train corrected Phase-1 predictor ($STEPS steps)"
geno-lewm-train --carbon-train \
  --run-dir "$WORK/run" \
  --dataset-dir "$WORK/dataset" \
  --carbon-model-dir "$CARBON_DIR" \
  --training-config "$CONFIG" \
  --steps "$STEPS" \
  --min-cuda-vram-gb "$MIN_CUDA_VRAM_GB" \
  --package-release-run \
  --no-banner --quiet

mkdir -p "$WORK/run/correction_control"
cp "$WORK/evidence/job_contract_preflight.json" "$WORK/run/correction_control/"
cp "$WORK/evidence/source_identity_report.json" "$WORK/run/correction_control/"
cp "$WORK/evidence/state_contract_audit.json" "$WORK/run/correction_control/"
cp "$WORK/evidence/tuple_throughput_report.json" "$WORK/run/correction_control/"
cp "$WORK/dataset/dataset_snapshot_report.json" "$WORK/run/correction_control/"
cp "$WORK/dataset/dataset_input_check_report.json" "$WORK/run/correction_control/"

log "validate completed correction-control artifact set"
python -m tools.research.correction_control_postflight \
  --training-run-json "$WORK/run/training_run.json" \
  --metrics-json "$WORK/run/metrics.json" \
  --training-config "$WORK/run/training_config.effective.yaml" \
  --checkpoint "$WORK/run/predictor_checkpoint.pt" \
  --state-contract-audit-json "$WORK/run/correction_control/state_contract_audit.json" \
  --job-contract-preflight-json "$WORK/run/correction_control/job_contract_preflight.json" \
  --source-identity-report-json "$WORK/run/correction_control/source_identity_report.json" \
  --dataset-manifest-json "$WORK/run/dataset_manifest.json" \
  --dataset-snapshot-report-json "$WORK/run/correction_control/dataset_snapshot_report.json" \
  --training-preflight-report-json "$WORK/run/training_preflight_report.json" \
  --tuple-throughput-report-json "$WORK/run/correction_control/tuple_throughput_report.json" \
  --expected-commit-sha "$COMMIT_SHA" \
  --expected-run-id "$RUN_ID" \
  --expected-dataset-snapshot-id "$SNAPSHOT_ID" \
  --output-json "$WORK/run/correction_control/correction_control_postflight.json"

if [ -n "$REPLAY_REFERENCE_ATTEMPT" ]; then
  log "validate deterministic replay against completed reference $REFERENCE_RUN_NAME"
  hf download "$UPLOAD_REPO" \
    --repo-type model \
    --include "$REFERENCE_RUN_NAME/run/**" \
    --local-dir "$WORK/replay-reference"
  python -m tools.research.correction_control_replay \
    --reference-run-dir "$WORK/replay-reference/$REFERENCE_RUN_NAME/run" \
    --candidate-run-dir "$WORK/run" \
    --reference-run-name "$REFERENCE_RUN_NAME" \
    --candidate-run-name "$RUN_NAME" \
    --expected-commit-sha "$COMMIT_SHA" \
    --output-json "$WORK/run/correction_control/deterministic_replay_report.json"
fi

log "export validated deploy checkpoint"
geno-lewm-export \
  --checkpoint "$WORK/run/predictor_checkpoint.pt" \
  --output-dir "$WORK/model" \
  --no-banner --quiet

log "author non-release correction-control model evidence manifest"
python -m tools.research.correction_control_model_manifest author \
  --model-dir "$WORK/model" \
  --training-run-json "$WORK/run/training_run.json" \
  --training-config "$WORK/run/training_config.effective.yaml" \
  --checkpoint "$WORK/run/predictor_checkpoint.pt" \
  --state-contract-audit-json "$WORK/run/correction_control/state_contract_audit.json" \
  --dataset-manifest-json "$WORK/run/dataset_manifest.json" \
  --correction-control-postflight-json \
    "$WORK/run/correction_control/correction_control_postflight.json" \
  --export-report-json "$WORK/model/export_report.json" \
  --manifest-json "$WORK/model/manifest.json"

log "revalidate correction-control model evidence manifest"
python -m tools.research.correction_control_model_manifest validate \
  --model-dir "$WORK/model" \
  --training-run-json "$WORK/run/training_run.json" \
  --training-config "$WORK/run/training_config.effective.yaml" \
  --checkpoint "$WORK/run/predictor_checkpoint.pt" \
  --state-contract-audit-json "$WORK/run/correction_control/state_contract_audit.json" \
  --dataset-manifest-json "$WORK/run/dataset_manifest.json" \
  --correction-control-postflight-json \
    "$WORK/run/correction_control/correction_control_postflight.json" \
  --export-report-json "$WORK/model/export_report.json" \
  --manifest-json "$WORK/model/manifest.json" \
  --validation-report-json "$WORK/model/manifest_validation.json"

CONTROL_ARTIFACTS=(
  job_contract_preflight.json
  source_identity_report.json
  state_contract_audit.json
  dataset_snapshot_report.json
  dataset_input_check_report.json
  tuple_throughput_report.json
  correction_control_postflight.json
)
if [ -n "$REPLAY_REFERENCE_ATTEMPT" ]; then
  CONTROL_ARTIFACTS+=(deterministic_replay_report.json)
fi
(
  cd "$WORK/run/correction_control"
  sha256sum "${CONTROL_ARTIFACTS[@]}" > SHA256SUMS
)

log "upload deploy and dataset artifacts"
hf upload "$UPLOAD_REPO" "$WORK/model" "$RUN_NAME/model" --repo-type model
hf upload "$UPLOAD_REPO" "$WORK/dataset" "$RUN_NAME/dataset" --repo-type model

log "upload validated run last as the completion marker"
hf upload "$UPLOAD_REPO" "$WORK/run" "$RUN_NAME/run" --repo-type model
RUN_PROTECTED=1

echo "GENO_LEWM_CORRECTION_CONTROL_OK $RUN_NAME"
