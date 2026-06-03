#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
#
# First-experiment PROOF run orchestration for Hugging Face Jobs.
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
# Carbon-500M must be mounted read-only at $CARBON_DIR (default /carbon):
#   hf jobs run -v hf://HuggingFaceBio/Carbon-500M:/carbon ...
set -euo pipefail

WORK="${WORK:-/tmp/geno}"
CARBON_DIR="${CARBON_DIR:-/carbon}"
CORPUS_REVISION="${CORPUS_REVISION:-cb4c13a78102933b3a6ac65734d326f7b431d9b7}"
CARBON_CONFIG="${CARBON_CONFIG:-eukaryote_generator_10B_subset}"
CARBON_SOURCE="${CARBON_SOURCE:-eukaryotic_genes}"
MAX_WINDOWS="${MAX_WINDOWS:-20000}"
GNOMAD_LINES="${GNOMAD_LINES:-60000}"
STEPS="${STEPS:-2000}"
UPLOAD_REPO="${UPLOAD_REPO:-abdelstark/geno-lewm-runs}"
RUN_NAME="${RUN_NAME:-geno-lewm-proof}"
CLINVAR_URL="${CLINVAR_URL:-https://ftp.ncbi.nlm.nih.gov/pub/clinvar/vcf_GRCh38/clinvar.vcf.gz}"
GNOMAD_URL="${GNOMAD_URL:-https://storage.googleapis.com/gcp-public-data--gnomad/release/4.1/vcf/exomes/gnomad.exomes.v4.1.sites.chr22.vcf.bgz}"

CONFIG="configs/first_experiment/train-carbon-500m-snv.yaml"
SPEC_SRC="configs/first_experiment/dataset-snapshot-snv.json"

log() { echo "=== $* ==="; }

log "proof run: $RUN_NAME (steps=$STEPS windows=$MAX_WINDOWS)"
python -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available())" || true
nvidia-smi || true

mkdir -p "$WORK/inputs/clinvar" "$WORK/inputs/gnomad" "$WORK/inputs/carbon"
cp "$SPEC_SRC" "$WORK/dataset-snapshot-snv.json"

log "stage ClinVar GRCh38"
curl -fsSL "$CLINVAR_URL" -o "$WORK/inputs/clinvar/clinvar-2026-04-15-snv.vcf.gz"

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

log "stage Carbon corpus windows"
# The HF `datasets` streaming reader occasionally segfaults during interpreter
# finalization *after* the JSONL is fully written; tolerate a non-zero exit and
# validate the output file instead.
CARBON_OUT="$WORK/inputs/carbon/source-mix-windows.jsonl"
set +e
python -m tools.data.carbon_windows \
  --revision "$CORPUS_REVISION" --dataset-config "$CARBON_CONFIG" \
  --default-source "$CARBON_SOURCE" --max-windows "$MAX_WINDOWS" \
  --output "$CARBON_OUT"
cw_rc=$?
set -e
test -s "$CARBON_OUT" || { echo "carbon windows output is empty (rc=$cw_rc)"; exit 1; }
echo "carbon windows: $(wc -l < "$CARBON_OUT") lines (tool rc=$cw_rc)"

log "build dataset snapshot"
python -m tools.release.dataset_snapshot \
  --spec-json "$WORK/dataset-snapshot-snv.json" --dataset-dir "$WORK/dataset"

log "carbon-train ($STEPS steps on $(python -c 'import torch;print("cuda" if torch.cuda.is_available() else "cpu")'))"
geno-lewm-train --carbon-train \
  --run-dir "$WORK/run" --dataset-dir "$WORK/dataset" \
  --carbon-model-dir "$CARBON_DIR" --training-config "$CONFIG" \
  --steps "$STEPS" --package-release-run --no-banner --quiet

log "export deploy checkpoint"
geno-lewm-export --checkpoint "$WORK/run/predictor_checkpoint.pt" \
  --output-dir "$WORK/model" --no-banner --quiet

log "upload checkpoint + run + dataset to $UPLOAD_REPO (protect the trained artifact first)"
hf upload "$UPLOAD_REPO" "$WORK/model" "$RUN_NAME/model" --repo-type model
hf upload "$UPLOAD_REPO" "$WORK/run" "$RUN_NAME/run" --repo-type model
hf upload "$UPLOAD_REPO" "$WORK/dataset" "$RUN_NAME/dataset" --repo-type model

echo "GENO_LEWM_PROOF_OK $RUN_NAME"
