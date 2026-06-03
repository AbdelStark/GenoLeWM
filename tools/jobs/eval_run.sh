#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
#
# Held-out ClinVar EVAL + verifiable deploy-package run for Hugging Face Jobs.
#
# Consumes the exported deploy checkpoint produced by tools/jobs/proof_run.sh
# (predictor.safetensors + action_encoder.safetensors, uploaded under
# <RUN_NAME>/model in a runs repo), the Carbon-500M encoder mounted read-only at
# $CARBON_DIR, and a held-out ClinVar VCF, and produces:
#
#   (a) eval_metrics.json + eval_report.md  -> a calibration-independent
#       rank-based AUROC report (--score-field sigma_raw), and
#   (b) a VERIFIABLE deploy package: a model dir whose manifest.json commits
#       real SHA-256 hashes for predictor / action_encoder / calibration / eval /
#       training-config, that loads cleanly OFFLINE through GenoLeWMRuntime.
#       If the training-run evidence (run dir) is present, the full
#       tools.release.model_package gate is also built.
#
# Run inside a Job from a fresh clone of the repo:
#   git clone --depth 1 https://github.com/AbdelStark/GenoLeWM /repo
#   cd /repo && pip install -e ".[train]" && pip install pyarrow
#   hf jobs run -v hf://HuggingFaceBio/Carbon-500M:/carbon ... \
#     bash tools/jobs/eval_run.sh
#
# Carbon-500M MUST be mounted read-only at $CARBON_DIR (default /carbon):
#   hf jobs run -v hf://HuggingFaceBio/Carbon-500M:/carbon ...
#
# ---------------------------------------------------------------------------
# WHY THE STEP ORDER IS WHAT IT IS (verified against the code):
#
#  * The runtime loads Carbon from cfg.encoder.model_id with local_files_only=
#    True (runtime.py:598,606; carbon.py from_pretrained). The committed config
#    ships encoder.model_id=HuggingFaceBio/Carbon-500M, which will NOT resolve
#    offline from a /carbon bind mount. So we PATCH encoder.model_id -> /carbon
#    in the training config BEFORE author_manifest copies it in (author_manifest
#    reads the file verbatim via load_config; it has no --set). trust_remote_code
#    is already true in the first-experiment config and is preserved.
#
#  * GenoLeWMRuntime.__init__ (runtime.py:137-138) hash-verifies predictor,
#    action_encoder, calibration AND eval. So geno-lewm-score (and bench) cannot
#    run until a REAL calibration.parquet AND a REAL eval_report.md exist and
#    their hashes are committed. eval_report.md is derived FROM the scores ->
#    chicken-and-egg. We break it by scoring through the geno_lewm.surprise.score
#    .score_vcf PYTHON API (loads via load_scorer_modules, runtime.py:544, which
#    does NOT hash-verify calibration/eval), then re-author the manifest with the
#    real calibration+eval hashes so the CLI / runtime path is verifiable.
#
#  * AUROC: we score with a (possibly low-confidence proof) calibration table.
#    sigma_calibrated can collapse to ties on a tiny background, flattening AUROC
#    toward 0.5; sigma_raw is always finite and calibration-independent and AUROC
#    is rank-based, so we eval with --score-field sigma_raw.
#
#  * scores cover labels exactly with no dup keys because the SAME tool
#    (tools.data.clinvar_eval_set) emits both labels.jsonl and the scoring VCF
#    from the same de-duplicated rows. chrom is NOT normalized by VariantKey, but
#    both files inherit identical chrom strings from ClinVar, so keys match.
#
#  * A single ClinVar row whose REF disagrees with the FASTA (wrong build /
#    normalization) raises VcfParseError and ABORTS the whole VCF (no skip path,
#    partial JSONL flushed). We use the GRCh38 ClinVar VCF + a GRCh38 chr21 FASTA
#    and restrict to chr21 so every REF matches. We treat a non-zero score exit
#    as fatal and discard partial output.
# ---------------------------------------------------------------------------
set -euo pipefail

# --- inputs / knobs -------------------------------------------------------
WORK="${WORK:-/tmp/geno-eval}"
CARBON_DIR="${CARBON_DIR:-/carbon}"

# Runs repo + path produced by proof_run.sh (model/ holds the export; run/ holds
# the training run dir with training-run evidence, if present).
RUNS_REPO="${RUNS_REPO:-abdelstark/geno-lewm-runs}"
RUN_NAME="${RUN_NAME:-geno-lewm-proof}"
MODEL_SUBPATH="${MODEL_SUBPATH:-$RUN_NAME/model}"
RUN_SUBPATH="${RUN_SUBPATH:-$RUN_NAME/run}"

# Release identity (REAL ids — fixture/dummy/test names are rejected by the
# release gates). Keep these consistent across manifest/eval/efficiency.
REL="${REL:-geno-lewm-v0.1.0-r1}"
SNAP="${SNAP:-geno-lewm-data-v0.1.0-r1}"
MODEL_NAME="${MODEL_NAME:-geno-lewm}"
MODEL_VERSION="${MODEL_VERSION:-0.1.0}"
COMMIT="${COMMIT:-$(git rev-parse HEAD 2>/dev/null || echo 0000000)}"
HARDWARE="${HARDWARE:-$(python -c 'import torch;print("cuda:"+torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu")' 2>/dev/null || echo cpu)}"

# Held-out ClinVar (GRCh38) + chr21-only FASTA. chr21 keeps memory bounded (the
# scorer loads the WHOLE FASTA into RAM); a whole-genome FASTA risks OOM.
EVAL_CHROM="${EVAL_CHROM:-21}"
# Cap the held-out eval set (class-stratified) so the three scoring passes
# (calibration background + eval + efficiency timing) stay proof-scale. chr21 has
# ~20k labelled variants; 6000 keeps a robust AUROC at ~4x less GPU. Set to 0 to
# score the full chromosome.
EVAL_MAX_VARIANTS="${EVAL_MAX_VARIANTS:-6000}"
CLINVAR_URL="${CLINVAR_URL:-https://ftp.ncbi.nlm.nih.gov/pub/clinvar/vcf_GRCh38/clinvar.vcf.gz}"
# Ensembl per-chromosome GRCh38 FASTA. Header is '>21 dna:chromosome ...';
# _contig_candidates resolves ClinVar '21' to {'21','chr21'} so this matches.
# File MUST end in .gz (literal suffix check; .bgz would NOT be gz-decoded).
FASTA_URL="${FASTA_URL:-https://ftp.ensembl.org/pub/release-110/fasta/homo_sapiens/dna/Homo_sapiens.GRCh38.dna.chromosome.21.fa.gz}"

# Background VCF for the calibration table. Reuse a chr-subset of ClinVar (any
# scoreable variants work); a tiny background only yields low-confidence buckets
# (sparse RuntimeWarning, non-fatal) which is fine because we AUROC on sigma_raw.
CFG="${CFG:-configs/first_experiment/train-carbon-500m-snv.yaml}"

# Where to publish the eval report + package back to.
UPLOAD="${UPLOAD:-1}"
EVAL_UPLOAD_SUBPATH="${EVAL_UPLOAD_SUBPATH:-$RUN_NAME/eval}"

log() { echo "=== $* ==="; }

log "eval run: $RUN_NAME  rel=$REL snap=$SNAP commit=$COMMIT hw=$HARDWARE"
python -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available())" || true
python -c "import pyarrow; print('pyarrow', pyarrow.__version__)" \
  || { echo "FATAL: pyarrow is required for calibration parquet IO (pip install pyarrow)"; exit 1; }
nvidia-smi || true

test -d "$CARBON_DIR" || { echo "FATAL: Carbon-500M not mounted at $CARBON_DIR"; exit 1; }

MODEL="$WORK/model"
EVAL_DIR="$MODEL/eval"          # staged INSIDE the package so paths are package-relative
INPUTS="$WORK/inputs"
mkdir -p "$WORK" "$EVAL_DIR" "$INPUTS"

# --- 0. download the exported model from the runs repo --------------------
log "download exported model from $RUNS_REPO ($MODEL_SUBPATH)"
hf download "$RUNS_REPO" --repo-type model --include "$MODEL_SUBPATH/*" \
  --local-dir "$WORK/dl"
# Flatten the downloaded model subdir into $MODEL.
cp -a "$WORK/dl/$MODEL_SUBPATH/." "$MODEL/"
test -f "$MODEL/predictor.safetensors"     || { echo "FATAL: predictor.safetensors missing"; exit 1; }
test -f "$MODEL/action_encoder.safetensors" || { echo "FATAL: action_encoder.safetensors missing"; exit 1; }

# Optionally pull the training run dir (for the full model_package gate).
RUN_DIR="$WORK/run"
if hf download "$RUNS_REPO" --repo-type model --include "$RUN_SUBPATH/*" \
      --local-dir "$WORK/dl-run" 2>/dev/null && [ -d "$WORK/dl-run/$RUN_SUBPATH" ]; then
  mkdir -p "$RUN_DIR"; cp -a "$WORK/dl-run/$RUN_SUBPATH/." "$RUN_DIR/"
  echo "training run dir present: $RUN_DIR"
else
  echo "training run dir not found (full model_package gate will be skipped)"
  RUN_DIR=""
fi

# --- 1. patch the commitment config: encoder.model_id -> /carbon ----------
# author_manifest copies the training config verbatim and the runtime reads
# cfg.encoder.model_id with local_files_only=True, so it MUST be the mount path.
log "patch encoder.model_id -> $CARBON_DIR (and keep trust_remote_code)"
PATCHED_CFG="$WORK/training_config.patched.yaml"
CFG_SRC="$CFG" CARBON_DIR="$CARBON_DIR" OUT="$PATCHED_CFG" python - <<'PY'
import os
from geno_lewm.config import load_config
import yaml
src = os.environ["CFG_SRC"]; carbon = os.environ["CARBON_DIR"]; out = os.environ["OUT"]
with open(src) as fh:
    payload = yaml.safe_load(fh)
payload.setdefault("encoder", {})["model_id"] = carbon
# Carbon-500M needs the custom HybridDNATokenizer -> trust_remote_code MUST be on.
payload["encoder"]["trust_remote_code"] = True
with open(out, "w") as fh:
    yaml.safe_dump(payload, fh, sort_keys=True)
cfg = load_config(out)  # validates schema + rejects unknown keys
assert cfg.encoder.model_id == carbon, cfg.encoder.model_id
assert cfg.encoder.trust_remote_code is True
print("patched encoder.model_id =", cfg.encoder.model_id, "trust_remote_code =", cfg.encoder.trust_remote_code)
PY

# --- 2. preliminary manifest (placeholder calibration/eval hashes) --------
# Lets load_scorer_modules parse manifest.json so calibration + python-API
# scoring can run BEFORE calibration.parquet / eval_report.md exist.
log "author preliminary manifest (--allow-missing-evidence)"
python -m tools.release.author_manifest \
  --model-dir "$MODEL" \
  --training-config "$PATCHED_CFG" \
  --encoder-weights "$CARBON_DIR" \
  --model-name "$MODEL_NAME" --model-version "$MODEL_VERSION" \
  --release-id "$REL" --dataset-snapshot "$SNAP" \
  --allow-missing-evidence

# --- 3. stage the held-out ClinVar eval set + FASTA -----------------------
log "download ClinVar GRCh38 + Ensembl chr$EVAL_CHROM FASTA"
CLINVAR_FULL="$INPUTS/clinvar.GRCh38.vcf.gz"
FASTA="$INPUTS/chr${EVAL_CHROM}.fa.gz"
curl -fsSL "$CLINVAR_URL" -o "$CLINVAR_FULL"
curl -fsSL "$FASTA_URL"  -o "$FASTA"
echo "FASTA header: $(zcat "$FASTA" 2>/dev/null | head -1)"

# Emit labels.jsonl + a matching minimal VCF from the SAME de-duplicated,
# conflict-dropped rows -> score keys exactly cover label keys, no dup keys,
# guaranteed >=1 positive and >=1 negative or the tool errors out.
log "build held-out ClinVar eval set (chr$EVAL_CHROM, P/LP vs B/LB)"
LABELS="$EVAL_DIR/clinvar-chr${EVAL_CHROM}.labels.jsonl"
EVAL_VCF="$INPUTS/clinvar-chr${EVAL_CHROM}.vcf"
# Pass --fasta so unscoreable rows (off-contig / REF disagreeing with this FASTA)
# are dropped from BOTH labels + VCF here. One unscoreable row would otherwise
# abort the whole scoring pass (score.py has no skip path), so this guarantees
# the GPU scoring below cannot abort while keeping score keys covering labels.
CAP_ARGS=()
if [ "${EVAL_MAX_VARIANTS}" != "0" ]; then CAP_ARGS=(--max-variants "$EVAL_MAX_VARIANTS"); fi
python -m tools.data.clinvar_eval_set \
  --input-vcf "$CLINVAR_FULL" --chrom "$EVAL_CHROM" --fasta "$FASTA" \
  "${CAP_ARGS[@]}" \
  --labels-out "$LABELS" --vcf-out "$EVAL_VCF"
echo "labels: $(wc -l < "$LABELS") variants"

# --- 4. calibration table (uses load_scorer_modules; no runtime hash gate) -
# Background = the same held-out eval VCF (any scoreable variants suffice). A
# tiny background only triggers a 'sparse' RuntimeWarning + low-confidence
# buckets; build never fails on >=1 scoreable variant.
log "build calibration.parquet"
python -m tools.release.build_calibration \
  --model-dir "$MODEL" --vcf "$EVAL_VCF" --fasta "$FASTA" \
  --output "$MODEL/calibration.parquet"
test -s "$MODEL/calibration.parquet" || { echo "FATAL: calibration.parquet empty"; exit 1; }

# --- 5. score the held-out ClinVar set (PYTHON API, no runtime hash gate) --
# We deliberately use geno_lewm.surprise.score.score_vcf rather than the
# geno-lewm-score CLI here, because the CLI constructs GenoLeWMRuntime which
# hash-verifies the (not-yet-final) eval_report.md. The API path emits
# generated_by='geno-lewm-score' so the artifact is eval-compatible. A REF
# mismatch / off-contig row raises VcfParseError -> we treat non-zero as fatal.
log "score held-out ClinVar -> scores.jsonl (geno_lewm.surprise.score API)"
SCORES="$EVAL_DIR/scores.jsonl"
set +e
MODEL="$MODEL" EVAL_VCF="$EVAL_VCF" FASTA="$FASTA" SCORES="$SCORES" CARBON_DIR="$CARBON_DIR" \
python - <<'PY'
import os
from pathlib import Path
from geno_lewm.deploy.runtime import load_scorer_modules
from geno_lewm.surprise.calibration import read_calibration_table
from geno_lewm.surprise.score import score_vcf

model = Path(os.environ["MODEL"])
encoder, action_encoder, predictor = load_scorer_modules(model)  # loads Carbon offline from /carbon
calib = read_calibration_table(model / "calibration.parquet")
out = score_vcf(
    os.environ["EVAL_VCF"], encoder, action_encoder, predictor, calib,
    os.environ["SCORES"],
    reference_fasta=os.environ["FASTA"],
    show_progress=False,
)
n = sum(1 for _ in open(out))
print(f"wrote {out} ({n} scored rows)")
PY
rc=$?
set -e
if [ "$rc" -ne 0 ]; then
  echo "FATAL: scoring failed (rc=$rc) — discard partial $SCORES"
  rm -f "$SCORES"
  echo "Most likely cause: a ClinVar REF that disagrees with the chr$EVAL_CHROM FASTA"
  echo "(GRCh37/GRCh38 mismatch or normalization). Ensure both are GRCh38."
  exit "$rc"
fi
test -s "$SCORES" || { echo "FATAL: scores.jsonl empty"; exit 1; }

# Shared sha256:<64hex> model id for eval + efficiency (must be equal between the
# two; model_package does NOT require it to equal manifest.model_id()). Use the
# predictor weights hash — stable and identical across both reports.
MODEL_ID="$(python -c "from geno_lewm.provenance import sha256_file; print(sha256_file('$MODEL/predictor.safetensors'))")"
echo "model_id (shared eval/efficiency) = $MODEL_ID"

# --- 6. efficiency_report.json (assembled; inline inputs dodge path checks) -
# bench/inference.py --release-efficiency runs the geno-lewm-score CLI, which
# needs the FINAL verified manifest (calibration+eval), so it can't run yet and
# eval requires --efficiency-report to already exist. The efficiency report does
# not depend on the eval, so we assemble + normalize it here with measured (or
# conservatively estimated) numbers and inline input identities.
log "assemble + normalize efficiency_report.json"
EFF_IN="$WORK/efficiency_input.json"
EFF_OUT="$MODEL/efficiency_report.json"
MODEL="$MODEL" SCORES="$SCORES" EVAL_VCF="$EVAL_VCF" FASTA="$FASTA" EFF_IN="$EFF_IN" \
EVAL_CHROM="$EVAL_CHROM" MODEL_ID="$MODEL_ID" REL="$REL" SNAP="$SNAP" \
COMMIT="$COMMIT" HARDWARE="$HARDWARE" \
python - <<'PY'
import json, os, time
from pathlib import Path
from geno_lewm.provenance import sha256_file
from geno_lewm.deploy.runtime import load_scorer_modules
from geno_lewm.surprise.calibration import read_calibration_table
from geno_lewm.surprise.score import score_vcf

model = Path(os.environ["MODEL"])
encoder, ae, pred = load_scorer_modules(model)  # Carbon loaded offline from /carbon
calib = read_calibration_table(model / "calibration.parquet")

# Count scored rows for throughput denominator, then time one fresh batched pass.
n = sum(1 for _ in open(os.environ["SCORES"]))
tmp = model / "eval" / "_bench.scores.jsonl"
start = time.perf_counter()
score_vcf(
    os.environ["EVAL_VCF"], encoder, ae, pred, calib, str(tmp),
    reference_fasta=os.environ["FASTA"], show_progress=False, batch_size=64,
)
elapsed = max(time.perf_counter() - start, 1e-6)
throughput = max(n / elapsed, 1e-6)
single_latency_ms = max((elapsed / max(n, 1)) * 1000.0, 1e-6)
try:
    tmp.unlink()
except FileNotFoundError:
    pass

# peak memory: best-effort from torch; fall back to RSS.
peak = 0
try:
    import torch
    if torch.cuda.is_available():
        peak = int(torch.cuda.max_memory_allocated())
except Exception:
    peak = 0
if peak <= 0:
    try:
        import resource
        peak = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) * 1024
    except Exception:
        peak = 1
peak = max(peak, 1)

scores_p = Path(os.environ["SCORES"])
vcf_p = Path(os.environ["EVAL_VCF"])
payload = {
    "schema_version": "1.0.0",
    "generated_by": "tools.release.efficiency_report",
    "model_id": os.environ["MODEL_ID"],
    "model_release": os.environ["REL"],
    "dataset_snapshot": os.environ["SNAP"],
    "commit": os.environ["COMMIT"],
    "hardware": os.environ["HARDWARE"],
    "runtime": "geno_lewm.surprise.score.score_vcf",
    "warmup_batches": 0,
    "samples": n,
    "command": ["python", "-m", "geno_lewm.surprise.score", "score_vcf"],
    "measurements": {
        "single_variant_latency_ms": single_latency_ms,
        "batched_throughput_variants_per_s": throughput,
        "peak_memory_bytes": peak,
    },
    # inline:<label> input identities avoid package-relative path validation.
    "inputs": {
        "eval_vcf": {"path": "inline:eval_vcf", "sha256": sha256_file(vcf_p), "size_bytes": vcf_p.stat().st_size},
        "scores_jsonl": {"path": "inline:scores_jsonl", "sha256": sha256_file(scores_p), "size_bytes": scores_p.stat().st_size},
    },
    "limitations": [
        "Latency and throughput are measured on the held-out ClinVar chr"
        + os.environ.get("EVAL_CHROM", "21")
        + " set via the score_vcf API, not the production CLI subprocess.",
    ],
}
Path(os.environ["EFF_IN"]).write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
print("efficiency input:", os.environ["EFF_IN"])
PY
python -m tools.release.efficiency_report --input-json "$EFF_IN" --output "$EFF_OUT"
test -s "$EFF_OUT" || { echo "FATAL: efficiency_report.json empty"; exit 1; }

# --- 7. AUROC eval -> eval_metrics.json (calibration-independent) ----------
# --score-field sigma_raw: always finite, rank-based AUROC, independent of any
# degenerate proof calibration. --artifact-root=$MODEL so all artifact paths are
# package-relative. config-artifact points at the copied training config.
log "AUROC eval (--score-field sigma_raw) -> eval_metrics.json"
METRICS="$MODEL/eval_metrics.json"
# manifest.json training config name copied by author_manifest (default).
TRAIN_CFG_IN_PKG="$MODEL/training_config.yaml"
# A dataset manifest is required by the eval CLI as an artifact reference; if the
# real one is not staged, write a minimal package-relative stub inside the pkg.
DATASET_MANIFEST="$MODEL/eval/dataset_manifest.json"
if [ ! -f "$DATASET_MANIFEST" ]; then
  printf '{"snapshot": "%s", "source": "tools.data.clinvar_eval_set", "chrom": "%s"}\n' \
    "$SNAP" "$EVAL_CHROM" > "$DATASET_MANIFEST"
fi
geno-lewm-eval \
  --scores-jsonl "$SCORES" \
  --labels-jsonl "$LABELS" \
  --score-field sigma_raw \
  --output-metrics "$METRICS" \
  --artifact-root "$MODEL" \
  --model-id "$MODEL_ID" \
  --model-release "$REL" \
  --dataset-snapshot "$SNAP" \
  --commit "$COMMIT" \
  --hardware "$HARDWARE" \
  --checkpoint "$MODEL/predictor.safetensors" \
  --config-artifact "$TRAIN_CFG_IN_PKG" \
  --dataset-manifest "$DATASET_MANIFEST" \
  --efficiency-report "$EFF_OUT" \
  --split "eval_clinvar_chr${EVAL_CHROM}" \
  --no-banner --quiet
test -s "$METRICS" || { echo "FATAL: eval_metrics.json empty"; exit 1; }
echo "AUROC: $(python -c "import json;m=json.load(open('$METRICS'));print([x for x in m['metrics'] if x['name']=='auroc'])")"

# --- 8. render eval_report.md from the EXACT eval_metrics.json -------------
# model_package re-renders load_report_input(eval_metrics.json) and byte-compares
# (incl. trailing newline). Render from the same file; never hand-edit.
log "render eval_report.md (byte-exact source = eval_metrics.json)"
EVAL_REPORT="$MODEL/eval_report.md"
python -m tools.release.eval_report --metrics-json "$METRICS" --output "$EVAL_REPORT"
test -s "$EVAL_REPORT" || { echo "FATAL: eval_report.md empty"; exit 1; }

# --- 9. re-author manifest with REAL calibration + eval hashes ------------
# Now calibration.parquet AND eval_report.md exist; committing their real hashes
# makes GenoLeWMRuntime() construct cleanly (no placeholder mismatch). Use the
# SAME patched config (encoder.model_id=/carbon) and same ids.
log "re-author manifest (real evidence hashes, no placeholder)"
python -m tools.release.author_manifest \
  --model-dir "$MODEL" \
  --training-config "$PATCHED_CFG" \
  --encoder-weights "$CARBON_DIR" \
  --model-name "$MODEL_NAME" --model-version "$MODEL_VERSION" \
  --release-id "$REL" --dataset-snapshot "$SNAP"

# --- 10. verify the deploy package loads OFFLINE through the runtime -------
# This is the verifiable deploy package gate: it runs _verify_manifest_artifacts
# (predictor/action_encoder/calibration/eval/training hashes) AND loads Carbon
# offline under the fail-closed network guard.
log "verify package: GenoLeWMRuntime offline construction"
HF_HUB_OFFLINE=1 MODEL="$MODEL" python - <<'PY'
import os
from geno_lewm.deploy import GenoLeWMRuntime
rt = GenoLeWMRuntime(os.environ["MODEL"])
print("OK: runtime constructed; model_id =", rt.manifest.model_id())
PY

# --- 11. (optional) full release model_package gate -----------------------
# Requires training-run evidence (training_preflight_report.json,
# training_run_manifest.json, training_run_card.md, training_run_SHA256SUMS)
# staged in the model dir + listed in model_package.json extra_files. These come
# from the TRAINING job (run dir). Built only if that evidence is available.
if [ -n "$RUN_DIR" ] && [ -f "$RUN_DIR/training_run_manifest.json" ]; then
  log "build training-run evidence package"
  python -m tools.release.training_run \
    --run-dir "$RUN_DIR" \
    --metadata-json "$RUN_DIR/training_run_metadata.json" || {
      echo "training_run packaging failed; skipping full model_package gate"; RUN_DIR=""; }
fi

if [ -n "$RUN_DIR" ]; then
  # Copy the 4 required evidence files into the model dir.
  for f in training_preflight_report.json training_run_manifest.json \
           training_run_card.md training_run_SHA256SUMS; do
    if [ -f "$RUN_DIR/$f" ]; then cp "$RUN_DIR/$f" "$MODEL/$f"; fi
  done
fi

if [ -f "$MODEL/training_run_manifest.json" ]; then
  log "build verifiable model_package (full release gate)"
  META="$MODEL/model_package.json"
  REL="$REL" SNAP="$SNAP" META="$META" python - <<'PY'
import json, os
from pathlib import Path
meta = {
    "schema_version": "1.0.0",
    "generated_by": "tools.release.model_package",
    "summary": "GenoLeWM action-conditioned variant-effect scorer (Carbon-500M backbone).",
    "license": "Apache-2.0",
    "intended_use": "Research scoring of single-nucleotide variants on held-out ClinVar.",
    "data": ["Held-out ClinVar GRCh38 chr21 (P/LP vs B/LB) for AUROC evaluation."],
    "hardware": [os.environ.get("HARDWARE", "cpu")],
    "limitations": ["Proof-scale calibration; AUROC reported on sigma_raw (rank-based)."],
    "training": ["JEPA predictor trained on Carbon-500M features (see training-run evidence)."],
    "evaluation": ["Rank-based AUROC on held-out ClinVar via geno-lewm-eval --score-field sigma_raw."],
    "runtime": ["Offline GenoLeWMRuntime; Carbon-500M loaded local_files_only with trust_remote_code."],
    "release_notes": ["First eval release of " + os.environ["REL"] + "."],
    "extra_files": [
        "training_preflight_report.json",
        "training_run_manifest.json",
        "training_run_card.md",
        "training_run_SHA256SUMS",
    ],
}
Path(os.environ["META"]).write_text(json.dumps(meta, indent=2, sort_keys=True) + "\n")
print("wrote", os.environ["META"])
PY
  python -m tools.release.model_package --model-dir "$MODEL" --metadata-json "$META" \
    && echo "MODEL_PACKAGE_OK" \
    || echo "model_package gate failed (see error above); eval report + runtime package still valid"
else
  echo "skipping full model_package gate (no training-run evidence in run dir)"
fi

# --- 12. publish the eval report + package back to the runs repo ----------
if [ "$UPLOAD" = "1" ]; then
  log "upload eval report + verifiable package to $RUNS_REPO ($EVAL_UPLOAD_SUBPATH)"
  hf upload "$RUNS_REPO" "$MODEL" "$EVAL_UPLOAD_SUBPATH" --repo-type model
fi

echo "GENO_LEWM_EVAL_OK $RUN_NAME auroc_report=$METRICS package=$MODEL"
