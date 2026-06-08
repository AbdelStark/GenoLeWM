#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
#
# First-experiment / serious-completion training run orchestration for
# Hugging Face Jobs.
#
# Stages real inputs (ClinVar + a gnomAD chr-subset + pinned Carbon corpus
# windows), builds the dataset snapshot, trains the JEPA predictor on real
# Carbon-500M features (GPU), exports the deploy checkpoint, and uploads the
# checkpoint + dataset to the Hub. The expensive checkpoint is uploaded right
# after export so a later-step failure never discards it.
#
# Run inside a Job from a fresh clone of the repo:
#   git clone --depth 1 https://github.com/AbdelStark/GenoLeWM /repo
#   cd /repo && pip install -e ".[train]"
#   bash tools/jobs/proof_run.sh
#
# Use an A100/H100-class CUDA runner. Carbon-500M must be mounted read-only at
# $CARBON_DIR (default /carbon):
#   hf jobs run -v hf://HuggingFaceBio/Carbon-500M:/carbon ...
set -euo pipefail

WORK="${WORK:-/tmp/geno}"
CARBON_DIR="${CARBON_DIR:-/carbon}"
MIN_CUDA_VRAM_GB="${MIN_CUDA_VRAM_GB:-40}"
CORPUS_REVISION="${CORPUS_REVISION:-cb4c13a78102933b3a6ac65734d326f7b431d9b7}"
CARBON_CONFIG="${CARBON_CONFIG:-eukaryote_generator_10B_subset}"
CARBON_SOURCE="${CARBON_SOURCE:-eukaryotic_genes}"
MAX_WINDOWS="${MAX_WINDOWS:-20000}"
GNOMAD_LINES="${GNOMAD_LINES:-60000}"
STEPS="${STEPS:-20000}"
TUPLE_THROUGHPUT_SAMPLES="${TUPLE_THROUGHPUT_SAMPLES:-4096}"
MIN_TUPLES_PER_SECOND="${MIN_TUPLES_PER_SECOND:-5000}"
# Carbon window width fed to the encoder. The default 12288 bp makes each step a
# very long-sequence (O(n^2) attention) forward through Carbon-500M (~9 s/step on
# a100); the proof uses a smaller window for a fast, completable run. The EVAL
# job MUST use the same WINDOW_BP so train/score window latents are comparable.
WINDOW_BP="${WINDOW_BP:-4096}"
UPLOAD_REPO="${UPLOAD_REPO:-abdelstark/geno-lewm-runs}"
RUN_NAME="${RUN_NAME:-geno-lewm-proof}"
UPLOAD_PROGRESS="${UPLOAD_PROGRESS:-0}"
PARTIAL_UPLOAD_INTERVAL_SECONDS="${PARTIAL_UPLOAD_INTERVAL_SECONDS:-0}"
PARTIAL_UPLOAD_SUBPATH="${PARTIAL_UPLOAD_SUBPATH:-$RUN_NAME/run-progress}"
CLINVAR_URL="${CLINVAR_URL:-https://ftp.ncbi.nlm.nih.gov/pub/clinvar/vcf_GRCh38/clinvar.vcf.gz}"
GNOMAD_URL="${GNOMAD_URL:-https://storage.googleapis.com/gcp-public-data--gnomad/release/4.1/vcf/exomes/gnomad.exomes.v4.1.sites.chr22.vcf.bgz}"
FASTA22_URL="${FASTA22_URL:-https://ftp.ensembl.org/pub/release-110/fasta/homo_sapiens/dna/Homo_sapiens.GRCh38.dna.chromosome.22.fa.gz}"

CONFIG="${CONFIG:-configs/first_experiment/train-carbon-500m-snv.yaml}"
SPEC_SRC="${SPEC_SRC:-configs/first_experiment/dataset-snapshot-snv.json}"
PROGRESS_UPLOAD_PID=""

log() { echo "=== $* ==="; }

stop_progress_uploads() {
  if [ -n "$PROGRESS_UPLOAD_PID" ]; then
    kill "$PROGRESS_UPLOAD_PID" 2>/dev/null || true
    wait "$PROGRESS_UPLOAD_PID" 2>/dev/null || true
    PROGRESS_UPLOAD_PID=""
  fi
}

upload_run_progress_loop() {
  local checkpoint="$WORK/run/predictor_checkpoint.pt"
  local last_hash=""
  local hash=""
  while sleep "$PARTIAL_UPLOAD_INTERVAL_SECONDS"; do
    if [ ! -f "$checkpoint" ]; then
      continue
    fi
    hash="$(sha256sum "$checkpoint" 2>/dev/null | awk '{print $1}')" || hash=""
    if [ -z "$hash" ] || [ "$hash" = "$last_hash" ]; then
      continue
    fi
    log "upload progress checkpoint to $UPLOAD_REPO/$PARTIAL_UPLOAD_SUBPATH"
    hf upload "$UPLOAD_REPO" "$WORK/run" "$PARTIAL_UPLOAD_SUBPATH" --repo-type model || true
    last_hash="$hash"
  done
}

start_progress_uploads() {
  if [ "$UPLOAD_PROGRESS" != "1" ]; then
    return
  fi
  if ! [[ "$PARTIAL_UPLOAD_INTERVAL_SECONDS" =~ ^[0-9]+$ ]] \
    || [ "$PARTIAL_UPLOAD_INTERVAL_SECONDS" -le 0 ]; then
    echo "FATAL: PARTIAL_UPLOAD_INTERVAL_SECONDS must be a positive integer when UPLOAD_PROGRESS=1" >&2
    exit 1
  fi
  log "enable progress checkpoint uploads every ${PARTIAL_UPLOAD_INTERVAL_SECONDS}s to $UPLOAD_REPO/$PARTIAL_UPLOAD_SUBPATH"
  upload_run_progress_loop &
  PROGRESS_UPLOAD_PID="$!"
}

upload_partial_run_on_failure() {
  rc=$?
  stop_progress_uploads
  if [ "$rc" -ne 0 ] && [ -f "$WORK/run/predictor_checkpoint.pt" ]; then
    log "upload partial run checkpoint to $UPLOAD_REPO/$RUN_NAME/run-partial (rc=$rc)"
    hf upload "$UPLOAD_REPO" "$WORK/run" "$RUN_NAME/run-partial" --repo-type model || true
  fi
  return "$rc"
}
trap upload_partial_run_on_failure EXIT

log "proof run: $RUN_NAME (steps=$STEPS windows=$MAX_WINDOWS config=$CONFIG spec=$SPEC_SRC)"
test "$CARBON_DIR" = "/carbon" || { echo "FATAL: coherent release config expects Carbon mounted at /carbon"; exit 1; }
python - <<'PY'
import os
import sys

import torch

min_gb = float(os.environ.get("MIN_CUDA_VRAM_GB", "40"))
print("torch", torch.__version__, "cuda", torch.cuda.is_available())
if not torch.cuda.is_available():
    sys.exit("FATAL: CUDA is required for the first-experiment Carbon training proof")
props = torch.cuda.get_device_properties(0)
observed_gb = props.total_memory / (1024**3)
print("cuda_device", props.name, f"{observed_gb:.1f} GiB")
if observed_gb < min_gb:
    sys.exit(
        f"FATAL: CUDA device has {observed_gb:.1f} GiB; "
        f"need at least {min_gb:.1f} GiB for this proof"
    )
PY
nvidia-smi

mkdir -p "$WORK/inputs/clinvar" "$WORK/inputs/gnomad" "$WORK/inputs/carbon" "$WORK/inputs/reference"
cp "$SPEC_SRC" "$WORK/dataset-snapshot-snv.json"

log "stage ClinVar GRCh38"
CLINVAR_OUT="$WORK/inputs/clinvar/clinvar-2026-04-15-snv.vcf.gz"
curl -fsSL "$CLINVAR_URL" -o "$CLINVAR_OUT"

# Hold out the gnomAD training chromosome from the ClinVar eval split so the
# training edit source and the eval labels are disjoint by construction — no
# train/eval leakage (the snapshot's leakage gate enforces this).
HOLDOUT_CHROM="${HOLDOUT_CHROM:-22}"
log "hold out chr$HOLDOUT_CHROM from ClinVar eval (gnomAD edits use it)"
set +o pipefail
zcat "$CLINVAR_OUT" 2>/dev/null \
  | awk -F'\t' -v c="$HOLDOUT_CHROM" '/^#/ || ($1 != c && $1 != "chr" c)' \
  | gzip > "$CLINVAR_OUT.tmp"
set -o pipefail
test -s "$CLINVAR_OUT.tmp" || { echo "ClinVar holdout filter produced empty file"; exit 1; }
mv "$CLINVAR_OUT.tmp" "$CLINVAR_OUT"
echo "ClinVar (chr$HOLDOUT_CHROM held out): $(zcat "$CLINVAR_OUT" 2>/dev/null | grep -cv '^#') variant rows"

log "stage gnomAD chr22 subset ($GNOMAD_LINES lines)"
# bgzip is gzip-compatible; take the header + first variant rows, then re-gzip.
# `head` closes the pipe early on purpose, so curl/zcat get SIGPIPE (curl exit
# 23) — disable pipefail for this line and assert the subset is non-empty.
GNOMAD_OUT="$WORK/inputs/gnomad/gnomad-v4.1-snv.vcf.gz"
set +o pipefail
curl -fsSL "$GNOMAD_URL" | zcat 2>/dev/null | head -n "$GNOMAD_LINES" | gzip > "$GNOMAD_OUT"
set -o pipefail
test -s "$GNOMAD_OUT" || { echo "gnomAD subset is empty"; exit 1; }
echo "gnomAD subset: $(zcat "$GNOMAD_OUT" 2>/dev/null | wc -l) lines"

log "stage GRCh38 chr22 reference FASTA for placed windows"
FASTA22_OUT="$WORK/inputs/reference/Homo_sapiens.GRCh38.dna.chromosome.22.fa.gz"
curl -fsSL "$FASTA22_URL" -o "$FASTA22_OUT"
echo "FASTA chr22 header: $(zcat "$FASTA22_OUT" 2>/dev/null | head -1)"

log "stage Carbon corpus windows"
# The HF `datasets` streaming reader occasionally segfaults during interpreter
# finalization *after* the JSONL is fully written; tolerate a non-zero exit and
# validate the output file instead.
CARBON_OUT="$WORK/inputs/carbon/source-mix-windows.jsonl"
set +e
python -m tools.data.carbon_windows \
  --revision "$CORPUS_REVISION" --dataset-config "$CARBON_CONFIG" \
  --default-source "$CARBON_SOURCE" --max-windows "$MAX_WINDOWS" \
  --window-bp "$WINDOW_BP" --output "$CARBON_OUT"
cw_rc=$?
set -e
test -s "$CARBON_OUT" || { echo "carbon windows output is empty (rc=$cw_rc)"; exit 1; }
echo "carbon windows: $(wc -l < "$CARBON_OUT") lines (tool rc=$cw_rc)"

log "build dataset snapshot"
python -m tools.release.dataset_snapshot \
  --spec-json "$WORK/dataset-snapshot-snv.json" --dataset-dir "$WORK/dataset"

log "tuple-builder throughput gate"
python -m tools.data.tuple_throughput \
  --dataset-dir "$WORK/dataset" \
  --samples "$TUPLE_THROUGHPUT_SAMPLES" \
  --min-tuples-per-second "$MIN_TUPLES_PER_SECOND"

log "carbon preflight"
geno-lewm-train --carbon-preflight \
  --run-dir "$WORK/run" --dataset-dir "$WORK/dataset" \
  --carbon-model-dir "$CARBON_DIR" --training-config "$CONFIG" \
  --min-cuda-vram-gb "$MIN_CUDA_VRAM_GB" --no-banner --quiet

log "carbon-train ($STEPS steps on cuda)"
start_progress_uploads
geno-lewm-train --carbon-train \
  --run-dir "$WORK/run" --dataset-dir "$WORK/dataset" \
  --carbon-model-dir "$CARBON_DIR" --training-config "$CONFIG" \
  --steps "$STEPS" --min-cuda-vram-gb "$MIN_CUDA_VRAM_GB" \
  --package-release-run --no-banner --quiet
stop_progress_uploads

# Upload the run dir (which holds predictor_checkpoint.pt) FIRST, before export.
# Training is the expensive step; protecting its output immediately means a
# later-step failure (or job timeout during export/upload) never discards it.
log "upload run (checkpoint) to $UPLOAD_REPO to protect the trained artifact"
hf upload "$UPLOAD_REPO" "$WORK/run" "$RUN_NAME/run" --repo-type model

log "export deploy checkpoint"
geno-lewm-export --checkpoint "$WORK/run/predictor_checkpoint.pt" \
  --output-dir "$WORK/model" --no-banner --quiet

log "upload model + dataset to $UPLOAD_REPO"
hf upload "$UPLOAD_REPO" "$WORK/model" "$RUN_NAME/model" --repo-type model
hf upload "$UPLOAD_REPO" "$WORK/dataset" "$RUN_NAME/dataset" --repo-type model

echo "GENO_LEWM_PROOF_OK $RUN_NAME"
