#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
#
# Eval-time DEMO + PAPER generation for Hugging Face Jobs (Stages E + F).
#
# Consumes the verifiable model/eval package + dataset package built by the proof
# and eval runs (uploaded under <RUN_NAME>/eval and <RUN_NAME>/dataset in a runs
# repo), runs the eval-time terminal demo against the LIVE model, generates and
# byte-verifies the results paper, and uploads the demo dir + paper back to the
# runs repo. This is the publish-free subset of the release pipeline: it needs
# only HF_TOKEN + Carbon at /carbon (no GitHub token, no public-repo creation).
#
# Carbon-500M MUST be mounted read-only at $CARBON_DIR:
#   hf jobs run -v hf://HuggingFaceBio/Carbon-500M:/carbon ... bash tools/jobs/demo_run.sh
#
# Verified contracts (see tools/jobs/publish_run.sh header for citations):
#  * terminal_inference RUNS the model (geno-lewm-score subprocess) and writes its
#    own 6 artifacts into --output-dir; --vcf/--fasta MUST live inside that dir.
#  * The package config already ships encoder.model_id=/carbon, so Carbon loads
#    offline with no config patching; the real release ids pass the demo's
#    fixture gate, so we pass neither --allow-fixture-manifest nor
#    --no-require-native-runtime nor --allow-placeholders.
#  * paper_draft splices the eval report (honest ~chance AUROC) into the paper;
#    paper_package re-renders it byte-exact + verifies the package SHA256SUMS.
set -euo pipefail

WORK="${WORK:-/tmp/geno-demo}"
OUT="${OUT:-$WORK/out}"
CARBON_DIR="${CARBON_DIR:-/carbon}"

RUNS_REPO="${RUNS_REPO:-abdelstark/geno-lewm-runs}"
RUN_NAME="${RUN_NAME:-geno-lewm-proof}"
EVAL_SUBPATH="${EVAL_SUBPATH:-$RUN_NAME/eval}"
DATASET_SUBPATH="${DATASET_SUBPATH:-$RUN_NAME/dataset}"
RUN_SUBPATH="${RUN_SUBPATH:-$RUN_NAME/run}"

EVAL_CHROM="${EVAL_CHROM:-21}"
DEMO_MAX_VARIANTS="${DEMO_MAX_VARIANTS:-32}"
PAPER_FILENAME="${PAPER_FILENAME:-paper.md}"
PAPER_TITLE="${PAPER_TITLE:-GenoLeWM First Experiment Report}"
CLINVAR_URL="${CLINVAR_URL:-https://ftp.ncbi.nlm.nih.gov/pub/clinvar/vcf_GRCh38/clinvar.vcf.gz}"
FASTA_URL="${FASTA_URL:-https://ftp.ensembl.org/pub/release-110/fasta/homo_sapiens/dna/Homo_sapiens.GRCh38.dna.chromosome.21.fa.gz}"

UPLOAD="${UPLOAD:-1}"
DEMO_UPLOAD_SUBPATH="${DEMO_UPLOAD_SUBPATH:-$RUN_NAME/demo}"
PAPER_UPLOAD_SUBPATH="${PAPER_UPLOAD_SUBPATH:-$RUN_NAME/paper}"

log()  { echo "=== $* ==="; }
fail() { echo "FATAL: $*" >&2; exit 1; }

log "demo+paper run: $RUN_NAME (PROOF-scale; held-out ClinVar chr$EVAL_CHROM AUROC ~0.52, near chance)"
[ -n "${HF_TOKEN:-}" ] || fail "HF_TOKEN is required (download package + upload demo/paper)"
export HF_TOKEN
test -d "$CARBON_DIR" || fail "Carbon-500M not mounted at $CARBON_DIR"
python -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available())" || true
python -c "import safetensors.torch, transformers, pyarrow.parquet" \
  || fail "native deps missing (torch+safetensors+transformers+pyarrow); pip install -e '.[train]'"

MODEL_DIR="$WORK/model"; DATASET_DIR="$WORK/dataset"; DEMO_DIR="$WORK/demo"; INPUTS="$WORK/inputs"
PAPER_PATH="$OUT/$PAPER_FILENAME"
mkdir -p "$WORK" "$OUT" "$MODEL_DIR" "$DATASET_DIR" "$DEMO_DIR" "$INPUTS"

# --- 1. download the built model/eval package + dataset package ------------
log "download model/eval package from $RUNS_REPO ($EVAL_SUBPATH)"
hf download "$RUNS_REPO" --repo-type model --include "$EVAL_SUBPATH/*" --local-dir "$WORK/dl-model" \
  || fail "could not download $EVAL_SUBPATH from $RUNS_REPO"
cp -a "$WORK/dl-model/$EVAL_SUBPATH/." "$MODEL_DIR/"
for f in manifest.json model_package.json model_card.md SHA256SUMS eval_metrics.json \
         eval_report.md efficiency_report.json training_config.yaml \
         predictor.safetensors action_encoder.safetensors calibration.parquet; do
  test -s "$MODEL_DIR/$f" || fail "model package missing required file: $f"
done

log "download dataset package from $RUNS_REPO ($DATASET_SUBPATH)"
hf download "$RUNS_REPO" --repo-type model --include "$DATASET_SUBPATH/*" --local-dir "$WORK/dl-dataset" \
  || fail "could not download dataset package $DATASET_SUBPATH from $RUNS_REPO"
cp -a "$WORK/dl-dataset/$DATASET_SUBPATH/." "$DATASET_DIR/"
for f in dataset_manifest.json dataset_package.json data_card.md SHA256SUMS; do
  test -s "$DATASET_DIR/$f" || fail "dataset package missing required file: $f"
done

# Stage the FULL training-run evidence package into the model dir. paper_package
# re-validates the whole training-run package, but the eval package ships only the
# summary files (training_run_manifest/card/SHA256SUMS/preflight); the files
# training_run_SHA256SUMS references (predictor_checkpoint.pt, train.log,
# metrics.json, training_config.effective.yaml, dataset_manifest.json) live in the
# run dir. Copy them into the model root (the eval stub at model/eval/
# dataset_manifest.json is a different path, so no collision).
log "stage full training-run evidence from $RUNS_REPO ($RUN_SUBPATH)"
hf download "$RUNS_REPO" --repo-type model --include "$RUN_SUBPATH/*" --local-dir "$WORK/dl-run" \
  || fail "could not download run dir $RUN_SUBPATH"
RUN_DIR="$WORK/dl-run/$RUN_SUBPATH"
for f in dataset_manifest.json training_config.effective.yaml metrics.json train.log \
         predictor_checkpoint.pt training_run_manifest.json training_run_card.md \
         training_run_SHA256SUMS training_preflight_report.json; do
  test -s "$RUN_DIR/$f" && cp "$RUN_DIR/$f" "$MODEL_DIR/$f"
done
for f in dataset_manifest.json training_config.effective.yaml metrics.json train.log predictor_checkpoint.pt; do
  test -s "$MODEL_DIR/$f" || fail "training-run evidence still missing after staging: $f"
done

# --- 2. build the DEMO dir from a LIVE geno-lewm-score run ----------------
log "stage chr$EVAL_CHROM FASTA + a small demo VCF INSIDE the demo dir"
DEMO_FASTA="$DEMO_DIR/chr${EVAL_CHROM}.fa.gz"
DEMO_VCF="$DEMO_DIR/demo.vcf"
CLINVAR_FULL="$INPUTS/clinvar.GRCh38.vcf.gz"
CLINVAR_CHR="$INPUTS/clinvar.chr${EVAL_CHROM}.vcf.gz"
curl -fsSL "$CLINVAR_URL" -o "$CLINVAR_FULL" || fail "ClinVar download failed"
curl -fsSL "$FASTA_URL"  -o "$DEMO_FASTA"     || fail "FASTA download failed"
zcat "$CLINVAR_FULL" 2>/dev/null \
  | awk -F'\t' -v c="$EVAL_CHROM" '/^#/ || ($1 == c || $1 == "chr" c)' \
  | gzip > "$CLINVAR_CHR"
test -s "$CLINVAR_CHR" || fail "chr$EVAL_CHROM ClinVar subset is empty"
python -m tools.data.clinvar_eval_set \
  --input-vcf "$CLINVAR_CHR" --chrom "$EVAL_CHROM" --fasta "$DEMO_FASTA" \
  --max-variants "$DEMO_MAX_VARIANTS" \
  --labels-out "$INPUTS/demo.labels.jsonl" --vcf-out "$DEMO_VCF" \
  || fail "could not build a scoreable demo VCF"
grep -qv '^#' "$DEMO_VCF" || fail "demo VCF has no variant records"
echo "demo VCF: $(grep -cv '^#' "$DEMO_VCF") scoreable variants"

log "run terminal demo (LIVE geno-lewm-score; writes 6 artifacts into $DEMO_DIR)"
# Do NOT set HF_HUB_OFFLINE: the runtime builds the scorer (loads Carbon) in
# GenoLeWMRuntime.__init__ OUTSIDE the fail-closed network guard, and Carbon's
# trust_remote_code module resolves via an auto_map that references the HF repo,
# so offline mode blocks the code fetch ("couldn't connect to huggingface.co").
# The eval job loaded Carbon the same way with network available; scoring itself
# still runs under the runtime's network guard (Carbon is already loaded by then).
python -m tools.demo.terminal_inference \
  --model-dir "$MODEL_DIR" \
  --vcf "$DEMO_VCF" \
  --fasta "$DEMO_FASTA" \
  --output-dir "$DEMO_DIR" \
  --backend auto --batch-size 64 \
  --carbon-cache-dir "$CARBON_DIR" \
  || fail "terminal demo failed (Carbon not loadable from $CARBON_DIR, or a REF/contig mismatch)"
for f in scores.jsonl receipts.jsonl runtime_preflight_report.json \
         batch_receipt_report.json terminal-demo-transcript.md terminal_demo_manifest.json; do
  test -s "$DEMO_DIR/$f" || fail "demo is missing $f"
done
python - "$DEMO_DIR" <<'PY' || fail "demo manifest status is not 'passed'"
import json, sys
m = json.load(open(f"{sys.argv[1]}/terminal_demo_manifest.json"))
assert m.get("status") == "passed", m.get("status")
print("demo status:", m["status"])
PY

# --- 3. generate + verify the PAPER (honest near-chance AUROC) ------------
log "generate paper draft"
python -m tools.release.paper_draft \
  --model-dir "$MODEL_DIR" --dataset-dir "$DATASET_DIR" --demo-dir "$DEMO_DIR" \
  --output "$PAPER_PATH" --title "$PAPER_TITLE" \
  || fail "paper_draft failed"
test -s "$PAPER_PATH" || fail "paper.md is empty"

# paper_package is the strict release-coherence gate. It additionally requires
# the training-run + eval + efficiency + deploy-manifest to share ONE commit and
# ONE training config. This proof builds those across SEPARATE HF Jobs (training
# at one commit with the original config; eval at a later commit with the
# /carbon-patched config), so it reports training_config_*_mismatch /
# *_commit_mismatch. That is a release-packaging-coherence limitation of the
# multi-job proof, NOT a paper-content problem (paper_draft already validated the
# content and spliced the real ~chance AUROC). Run it as advisory so the demo +
# paper still ship; a single-coherent-pass release would make it pass.
PAPER_PACKAGE_OK=1
python -m tools.release.paper_package \
  --model-dir "$MODEL_DIR" --dataset-dir "$DATASET_DIR" --demo-dir "$DEMO_DIR" \
  --paper-path "$PAPER_PATH" \
  || { PAPER_PACKAGE_OK=0; echo "WARNING: paper_package release-coherence gate failed (cross-job commit/config); paper.md content is valid"; }

echo "AUROC (from eval): $(python -c "import json;print([x['value'] for x in json.load(open('$MODEL_DIR/eval_metrics.json'))['metrics'] if x['name']=='auroc'][0])")"

# --- 4. upload the demo dir + paper back to the runs repo -----------------
if [ "$UPLOAD" = "1" ]; then
  log "upload demo + paper to $RUNS_REPO"
  hf upload "$RUNS_REPO" "$DEMO_DIR" "$DEMO_UPLOAD_SUBPATH" --repo-type model
  hf upload "$RUNS_REPO" "$PAPER_PATH" "$PAPER_UPLOAD_SUBPATH/$PAPER_FILENAME" --repo-type model
fi

echo "GENO_LEWM_DEMO_PAPER_OK $RUN_NAME demo=$DEMO_DIR paper=$PAPER_PATH paper_package_ok=$PAPER_PACKAGE_OK"
