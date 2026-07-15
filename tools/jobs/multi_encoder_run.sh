#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
#
# Multi-encoder edit-response geometry replication (R6) — HF Jobs runner.
#
# Companion to tools/jobs/edit_response_run.sh (R1). R1 measured the frozen
# Carbon-500M encoder and found that a single-base edit displaces the pooled
# state delta = s(alt) - s(ref) informatively (ClinVar pathogenic-vs-benign
# AUROC ~0.92 after controlling for mutation spectrum and genomic region) while
# that displacement is NOT predictable from (pooled reference state, action) --
# the best learned predictor reached 0.65 against a 0.93 ceiling.
#
# On one checkpoint that is an anecdote. If it is a fact about genomic
# foundation models as a class, it must reproduce on encoders that share
# neither Carbon's architecture, its tokenizer, nor its training corpus. This
# job re-measures the same geometry on two deliberately dissimilar encoders,
# over the EXACT R1 variant set, so the comparison is a clean join:
#
#   * nt-v2-100m-multi     -- BERT-like masked LM, 6-mer tokenizer, d=512.
#   * hyenadna-medium-450k -- implicit-convolution LM, char tokenizer, d=256.
#
# Nothing is trained here; these are direct measurements of frozen encoders.
#
# Both encoders come from the HF hub, so no volume mount is needed (unlike the
# Carbon jobs' `-v hf://HuggingFaceBio/Carbon-500M:/carbon:ro`).
#
# Expected runtime on a100-large: ~40-70 min (~$2-3). Bounded by 2 forwards per
# variant per encoder over ~13k SNVs. HyenaDNA dominates: at 1bp/token a 4096bp
# window is 4097 tokens against NT's ~683, so it runs with a smaller batch.
set -euo pipefail

log() { printf '\n=== %s ===\n' "$*"; }

# --- knobs (override via env) --------------------------------------------
IN_REPO="${IN_REPO:-abdelstark/geno-lewm-edit-response}"
IN_RUN="${IN_RUN:-r1-edit-response-e23fdf9}"
OUT_REPO="${OUT_REPO:-abdelstark/geno-lewm-edit-response}"
RUN_NAME="${RUN_NAME:-r6-multi-encoder-$(git -C "${REPO_DIR:-.}" rev-parse --short HEAD 2>/dev/null || echo local)}"
WINDOW_BP="${WINDOW_BP:-4096}"
POOL_RADII="${POOL_RADII:-0,8,64,256}"
DTYPE="${DTYPE:-fp32}"
ENCODERS="${ENCODERS:-nt-v2-100m-multi hyenadna-medium-450k}"
FASTA_URL="${FASTA_URL:-https://ftp.ensembl.org/pub/release-110/fasta/homo_sapiens/dna/Homo_sapiens.GRCh38.dna.primary_assembly.fa.gz}"

# Per-encoder batch size: HyenaDNA's char tokenizer makes a 4096bp window 6x
# longer in tokens than NT's, so a shared batch size would either waste the GPU
# on NT or OOM on HyenaDNA.
BATCH_NT="${BATCH_NT:-32}"
BATCH_HYENA="${BATCH_HYENA:-4}"

WORK="${WORK:-/workspace/multi-encoder}"
INPUTS="$WORK/inputs"
OUT="$WORK/out"
mkdir -p "$INPUTS" "$OUT"

# --- 0. pin transformers to the 4.x line ----------------------------------
# nucleotide-transformer-v2 ships trust_remote_code modeling that imports
# `find_pruneable_heads_and_indices` from transformers.pytorch_utils, which 5.x
# removed -- the model raises ImportError on transformers 5. Its 4.x-era
# EsmConfig also lacks `rope_theta`, which 5.x's built-in ESM now requires.
# HyenaDNA loads on both lines, so 4.x is the only version that runs the pair.
log "pin transformers to the 4.x line (nucleotide-transformer-v2 requires it)"
pip install -q "transformers>=4.45,<5"
python -c "import transformers; print('transformers', transformers.__version__)"

# --- 1. stage reference FASTA --------------------------------------------
log "download Ensembl GRCh38 primary_assembly FASTA (~882MB gz)"
FASTA_GZ="$INPUTS/GRCh38.primary_assembly.fa.gz"
FASTA="$INPUTS/GRCh38.primary_assembly.fa"
curl -fsSL "$FASTA_URL" -o "$FASTA_GZ"
gunzip -f "$FASTA_GZ"
test -s "$FASTA" || { echo "FATAL: FASTA empty"; exit 1; }
echo "FASTA header: $(head -1 "$FASTA")"

# --- 2. reuse the EXACT R1 variant set ------------------------------------
# Reusing R1's published variants.jsonl verbatim is what makes this a
# replication rather than a new study: the join against the Carbon rows is
# exact, so any difference is the encoder and not the variant sample.
log "fetch R1 variants.jsonl from $IN_REPO/$IN_RUN"
VARIANTS="$INPUTS/variants.jsonl"
hf download "$IN_REPO" "$IN_RUN/variants.jsonl" --repo-type dataset --local-dir "$INPUTS/dl"
cp "$INPUTS/dl/$IN_RUN/variants.jsonl" "$VARIANTS"
echo "R1 variants: $(wc -l < "$VARIANTS")"

# --- 3. run spectroscopy once per encoder ---------------------------------
# One table per encoder, deliberately: edit_response_analysis buckets rows by
# pool_radius alone and has no notion of encoder_id, so a shared table would
# blend NT's 512-dim rows with HyenaDNA's 256-dim rows under the same
# variant_id. The encoder_id column identifies each file's producer.
for ENCODER in $ENCODERS; do
  case "$ENCODER" in
    hyenadna*) BATCH="$BATCH_HYENA" ;;
    *)         BATCH="$BATCH_NT" ;;
  esac
  log "spectroscopy: $ENCODER (window=$WINDOW_BP radii=$POOL_RADII batch=$BATCH)"
  EMB="$OUT/edit_response_embeddings.$ENCODER.parquet"
  SUMMARY="$OUT/edit_response_summary.$ENCODER.json"
  python -m tools.research.multi_encoder_spectroscopy \
    --variants "$VARIANTS" \
    --reference-fasta "$FASTA" \
    --out-embeddings "$EMB" \
    --out-summary "$SUMMARY" \
    --encoder "$ENCODER" \
    --window-bp "$WINDOW_BP" \
    --pool-radii "$POOL_RADII" \
    --dtype "$DTYPE" \
    --batch-size "$BATCH" \
    --device cuda
  echo "summary ($ENCODER):"
  python -c "import json;s=json.load(open('$SUMMARY'));print(json.dumps({'counts':s['counts'],'per_pool_radius':s['per_pool_radius']}, indent=2))"
done

# --- 4. upload ------------------------------------------------------------
log "upload results to $OUT_REPO/$RUN_NAME"
cp "$VARIANTS" "$OUT/variants.jsonl"
for ENCODER in $ENCODERS; do
  hf upload "$OUT_REPO" "$OUT/edit_response_embeddings.$ENCODER.parquet" \
    "$RUN_NAME/edit_response_embeddings.$ENCODER.parquet" --repo-type dataset
  hf upload "$OUT_REPO" "$OUT/edit_response_summary.$ENCODER.json" \
    "$RUN_NAME/edit_response_summary.$ENCODER.json" --repo-type dataset
done
hf upload "$OUT_REPO" "$OUT/variants.jsonl" "$RUN_NAME/variants.jsonl" --repo-type dataset
log "DONE — results at https://huggingface.co/datasets/$OUT_REPO/tree/main/$RUN_NAME"
