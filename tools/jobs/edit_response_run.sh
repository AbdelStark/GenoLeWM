#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
#
# Edit-response geometry study (R1) — HF Jobs runner.
#
# Measures the interventional edit-response geometry of the frozen Carbon-500M
# encoder over REAL, coordinate-grounded variants: for each variant we encode
# the reference window and the single-base-edited window and study the pooled
# displacement delta = s(alt) - s(ref) across a pooling-radius grid. Unlike the
# invalidated v0.2.1 training run, nothing is trained here — this is a direct
# measurement of the frozen encoder, so no model-quality claim is confounded by
# an optimizer. Output is a per-(variant, pool_radius) embedding table plus a
# provenance/aggregate summary, uploaded to a content-addressed dataset repo.
#
# Variant sets (all real, coordinate-grounded against GRCh38):
#   * ClinVar P/LP vs B/LB  -> pathogenicity AUROC of the displacement.
#   * BRCA2 saturation genome editing (Sahu et al. 2025, MaveDB
#     urn:mavedb:00001242-a-1) -> Spearman of displacement vs functional score.
#
# This exercises geno_lewm/tools code paths only; it stages public upstream data
# (NCBI ClinVar, Ensembl GRCh38 FASTA, MaveDB) and never trains or fine-tunes.
#
# Expected runtime on a100-large: ~15-25 min (~$1). Bounded by the Carbon
# forward passes over ~2 * N_variants windows.
set -euo pipefail

log() { printf '\n=== %s ===\n' "$*"; }

# --- knobs (override via env) --------------------------------------------
BRANCH="${BRANCH:-feat/edit-response-geometry}"
OUT_REPO="${OUT_REPO:-abdelstark/geno-lewm-edit-response}"
RUN_NAME="${RUN_NAME:-r1-edit-response-$(git -C "${REPO_DIR:-.}" rev-parse --short HEAD 2>/dev/null || echo local)}"
WINDOW_BP="${WINDOW_BP:-4096}"
POOL_RADII="${POOL_RADII:-0,8,64,256}"
STATE_LAYER="${STATE_LAYER:-20}"
DTYPE="${DTYPE:-bf16}"
BATCH_SIZE="${BATCH_SIZE:-64}"
MAX_CLINVAR="${MAX_CLINVAR:-8000}"
MAX_BRCA2="${MAX_BRCA2:-6000}"
CARBON_DIR="${CARBON_DIR:-/carbon}"
CARBON_REV="${CARBON_REV:-5d31d59b3c845b288a13aedb1358934196852eec}"

CLINVAR_URL="${CLINVAR_URL:-https://ftp.ncbi.nlm.nih.gov/pub/clinvar/vcf_GRCh38/clinvar.vcf.gz}"
FASTA_URL="${FASTA_URL:-https://ftp.ensembl.org/pub/release-110/fasta/homo_sapiens/dna/Homo_sapiens.GRCh38.dna.primary_assembly.fa.gz}"
MAVEDB_URN="${MAVEDB_URN:-urn:mavedb:00001242-a-1}"
MAVEDB_API="${MAVEDB_API:-https://api.mavedb.org/api/v1}"

WORK="${WORK:-/workspace/edit-response}"
INPUTS="$WORK/inputs"
OUT="$WORK/out"
mkdir -p "$INPUTS" "$OUT"

# --- 0. sanity: Carbon mounted read-only ---------------------------------
test -f "$CARBON_DIR/config.json" || { echo "FATAL: Carbon not mounted at $CARBON_DIR"; exit 1; }
log "Carbon runtime at $CARBON_DIR"

# --- 1. stage upstream data ----------------------------------------------
log "download Ensembl GRCh38 primary_assembly FASTA (~882MB gz)"
FASTA_GZ="$INPUTS/GRCh38.primary_assembly.fa.gz"
FASTA="$INPUTS/GRCh38.primary_assembly.fa"
curl -fsSL "$FASTA_URL" -o "$FASTA_GZ"
gunzip -f "$FASTA_GZ"
echo "FASTA header: $(head -1 "$FASTA")"
test -s "$FASTA" || { echo "FATAL: FASTA empty"; exit 1; }

log "download NCBI ClinVar GRCh38 VCF (~192MB)"
CLINVAR="$INPUTS/clinvar.GRCh38.vcf.gz"
curl -fsSL "$CLINVAR_URL" -o "$CLINVAR"
test -s "$CLINVAR" || { echo "FATAL: ClinVar empty"; exit 1; }

log "download MaveDB BRCA2 sGE scores + VRS mappings ($MAVEDB_URN)"
BRCA2_SCORES="$INPUTS/brca2_scores.csv"
BRCA2_MAPPED="$INPUTS/brca2_mapped.json"
curl -fsSL "$MAVEDB_API/score-sets/$MAVEDB_URN/scores" -o "$BRCA2_SCORES"
curl -fsSL "$MAVEDB_API/score-sets/$MAVEDB_URN/mapped-variants" -o "$BRCA2_MAPPED"
test -s "$BRCA2_SCORES" && test -s "$BRCA2_MAPPED" || { echo "FATAL: MaveDB download empty"; exit 1; }

# --- 2. build ClinVar labelled eval set (genome-wide, FASTA-scoreable) ----
log "build ClinVar P/LP-vs-B/LB set (genome-wide, class-stratified, max=$MAX_CLINVAR)"
CLINVAR_LABELS="$INPUTS/clinvar.labels.jsonl"
CLINVAR_VCF="$INPUTS/clinvar.eval.vcf"
python -m tools.data.clinvar_eval_set \
  --input-vcf "$CLINVAR" \
  --labels-out "$CLINVAR_LABELS" \
  --vcf-out "$CLINVAR_VCF" \
  --fasta "$FASTA" \
  --max-variants "$MAX_CLINVAR"
echo "ClinVar labelled rows: $(wc -l < "$CLINVAR_LABELS")"

# --- 3. unify all variant sources into one variants.jsonl -----------------
log "unify ClinVar + BRCA2 into variants.jsonl"
VARIANTS="$INPUTS/variants.jsonl"
python - "$CLINVAR_LABELS" "$BRCA2_SCORES" "$BRCA2_MAPPED" "$VARIANTS" "$MAX_BRCA2" <<'PY'
import json, sys
from pathlib import Path
from tools.data.clinvar_eval_set import _binary_label
from tools.release.v02_benchmark_inputs import load_brca2_rows

clinvar_labels, brca2_scores, brca2_mapped, out_path, max_brca2 = sys.argv[1:6]
records = []

# ClinVar -> clinvar_path / clinvar_benign
#
# SNV-only. ClinVar ships indels and MNVs under the same clinical-significance
# codes, but they are ~43% of pathogenic against ~5% of benign and displace the
# pooled state ~3.7x further than a single-base edit, so their length change --
# not their biology -- tracks the label. This study measures single-base edit
# response, so multi-base alleles are out of scope rather than merely awkward.
n_path = n_benign = n_non_snv = 0
for line in Path(clinvar_labels).read_text().splitlines():
    line = line.strip()
    if not line:
        continue
    row = json.loads(line)
    binary = _binary_label(row["clinical_significance"])
    if binary is None:
        continue
    if len(row["ref"]) != 1 or len(row["alt"]) != 1:
        n_non_snv += 1
        continue
    group = "clinvar_path" if binary else "clinvar_benign"
    n_path += binary
    n_benign += (not binary)
    records.append({
        "chrom": str(row["chrom"]), "pos": int(row["pos"]),
        "ref": row["ref"], "alt": row["alt"],
        "label": group, "label_group": group,
        "region": None, "gene": None,
        "variant_id": f'clinvar:{row["chrom"]}:{row["pos"]}:{row["ref"]}:{row["alt"]}',
    })

# BRCA2 sGE -> brca2_sge with continuous functional score
brca2_rows, skipped = load_brca2_rows(
    scores_csv=Path(brca2_scores),
    mapped_variants_json=Path(brca2_mapped),
    limit=int(max_brca2),
    seed=0,
)
for r in brca2_rows:
    records.append({
        "chrom": str(r.key.chrom), "pos": int(r.key.pos),
        "ref": r.key.ref, "alt": r.key.alt,
        "label": "brca2_sge", "label_group": "brca2_sge",
        "continuous_score": float(r.functional_score),
        "region": None, "gene": "BRCA2",
        "variant_id": f"brca2:{r.source_id}",
    })

with Path(out_path).open("w") as fh:
    for rec in records:
        fh.write(json.dumps(rec, sort_keys=True) + "\n")

print(f"ClinVar path={n_path} benign={n_benign} dropped_non_snv={n_non_snv} | "
      f"BRCA2 kept={len(brca2_rows)} skipped={skipped}")
print(f"total variants: {len(records)}")
PY
echo "unified variants: $(wc -l < "$VARIANTS")"

# --- 4. run edit-response spectroscopy (single Carbon forward per window) --
log "run edit-response spectroscopy (window=$WINDOW_BP radii=$POOL_RADII layer=$STATE_LAYER)"
EMB="$OUT/edit_response_embeddings.parquet"
SUMMARY="$OUT/edit_response_summary.json"
python -m tools.research.edit_response_spectroscopy \
  --variants "$VARIANTS" \
  --reference-fasta "$FASTA" \
  --out-embeddings "$EMB" \
  --out-summary "$SUMMARY" \
  --window-bp "$WINDOW_BP" \
  --pool-radii "$POOL_RADII" \
  --state-layer "$STATE_LAYER" \
  --dtype "$DTYPE" \
  --batch-size "$BATCH_SIZE" \
  --model-id "$CARBON_DIR" \
  --revision "$CARBON_REV"
echo "summary:"; python -c "import json;print(json.dumps(json.load(open('$SUMMARY'))['per_pool_radius'], indent=2))"

# --- 5. upload results to the content-addressed dataset repo --------------
log "upload results to $OUT_REPO/$RUN_NAME"
cp "$VARIANTS" "$OUT/variants.jsonl"
hf upload "$OUT_REPO" "$OUT/edit_response_embeddings.parquet" "$RUN_NAME/edit_response_embeddings.parquet" --repo-type dataset
hf upload "$OUT_REPO" "$OUT/edit_response_summary.json"      "$RUN_NAME/edit_response_summary.json"      --repo-type dataset
hf upload "$OUT_REPO" "$OUT/variants.jsonl"                  "$RUN_NAME/variants.jsonl"                  --repo-type dataset
log "DONE — results at https://huggingface.co/datasets/$OUT_REPO/tree/main/$RUN_NAME"
