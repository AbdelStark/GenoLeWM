#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
#
# Carbon zero-shot delta-log-likelihood baseline (R5) — HF Jobs runner.
#
# Companion to tools/jobs/edit_response_run.sh (R1). R1 measured the pooled
# edit-response geometry delta = s(alt) - s(ref) of the frozen Carbon-500M
# encoder and found that ||delta|| separates ClinVar pathogenic from benign SNVs
# at AUROC ~0.92 after controlling for mutation spectrum and genomic region.
#
# That result is only novel if the geometry is not simply a restatement of the
# model's own likelihood surprise: autoregressive DNA models already yield a
# standard zero-shot variant-effect score, delta-logLik = logP(alt) - logP(ref),
# and embedding-distance VEP (BEND) is a known baseline. This job computes that
# baseline on EXACTLY the same variants so the two can be compared, correlated,
# and ensembled. The decisive question is orthogonality: does the edit-response
# geometry carry pathogenicity signal that the likelihood does not?
#
# It also emits the local reference k-mer context around each variant, which is
# required to control for the Carbon 6bp-token confound (a single SNV rewrites
# one 6-mer token; ||delta|| could reflect that token pair rather than biology).
#
# Inputs are reused verbatim from the R1 run (same variants.jsonl), so the join
# is exact and no re-encoding is needed. Nothing is trained here.
#
# Expected runtime on a100-large: ~30-50 min (~$2). Bounded by 2 forwards per
# variant (ref + alt) over ~13k SNVs.
set -euo pipefail

log() { printf '\n=== %s ===\n' "$*"; }

# --- knobs (override via env) --------------------------------------------
IN_REPO="${IN_REPO:-abdelstark/geno-lewm-edit-response}"
IN_RUN="${IN_RUN:-r1-edit-response-e23fdf9}"
OUT_REPO="${OUT_REPO:-abdelstark/geno-lewm-edit-response}"
RUN_NAME="${RUN_NAME:-r5-delta-loglik-$(git -C "${REPO_DIR:-.}" rev-parse --short HEAD 2>/dev/null || echo local)}"
WINDOW_BP="${WINDOW_BP:-4096}"
# fp32, NOT bf16. A window log-likelihood sums ~10^3 token terms to about -5000, while the
# quantity of interest -- logP(alt) - logP(ref) for one base -- is of order 1. bf16 resolves
# only ~16 at that magnitude, so the difference quantizes onto a grid coarser than the signal:
# the first bf16 run of this job returned 90.7% exact zeros across just 18 distinct values,
# every one a multiple of 16. geno_lewm.carbon_zero_shot now accumulates in fp32 regardless,
# but a reference baseline should not depend on that cast to be correct, and the released
# v0.2.1 suite pins fp32 here for the same reason.
DTYPE="${DTYPE:-fp32}"
KMER_FLANK="${KMER_FLANK:-5}"
CARBON_DIR="${CARBON_DIR:-/carbon}"
CARBON_REV="${CARBON_REV:-5d31d59b3c845b288a13aedb1358934196852eec}"
FASTA_URL="${FASTA_URL:-https://ftp.ensembl.org/pub/release-110/fasta/homo_sapiens/dna/Homo_sapiens.GRCh38.dna.primary_assembly.fa.gz}"

WORK="${WORK:-/workspace/delta-loglik}"
INPUTS="$WORK/inputs"
OUT="$WORK/out"
mkdir -p "$INPUTS" "$OUT"

test -f "$CARBON_DIR/config.json" || { echo "FATAL: Carbon not mounted at $CARBON_DIR"; exit 1; }
log "Carbon runtime at $CARBON_DIR"

# --- 1. stage reference FASTA --------------------------------------------
log "download Ensembl GRCh38 primary_assembly FASTA (~882MB gz)"
FASTA_GZ="$INPUTS/GRCh38.primary_assembly.fa.gz"
FASTA="$INPUTS/GRCh38.primary_assembly.fa"
curl -fsSL "$FASTA_URL" -o "$FASTA_GZ"
gunzip -f "$FASTA_GZ"
test -s "$FASTA" || { echo "FATAL: FASTA empty"; exit 1; }

# --- 2. reuse the EXACT R1 variant set ------------------------------------
log "fetch R1 variants.jsonl from $IN_REPO/$IN_RUN"
VARIANTS_ALL="$INPUTS/variants.all.jsonl"
hf download "$IN_REPO" "$IN_RUN/variants.jsonl" --repo-type dataset --local-dir "$INPUTS/dl"
cp "$INPUTS/dl/$IN_RUN/variants.jsonl" "$VARIANTS_ALL"
echo "R1 variants: $(wc -l < "$VARIANTS_ALL")"

# --- 3. restrict to SNVs (R1 confound fix) + emit VCF + k-mer context ------
# R1 shipped indels in the ClinVar arm: they are 43% of pathogenic vs 5% of
# benign and inflate ||delta|| 3.7x, which confounds the label. The study is
# about single-base edits, so the SNV restriction is the correct scope.
log "filter to SNVs, emit VCF and local k-mer context (flank=$KMER_FLANK)"
VCF="$INPUTS/variants.snv.vcf"
CONTEXT="$OUT/variant_context.jsonl"
python - "$VARIANTS_ALL" "$FASTA" "$VCF" "$CONTEXT" "$KMER_FLANK" <<'PY'
import json, sys
from pathlib import Path
from geno_lewm.surprise.score import _load_reference_fasta

src, fasta_path, vcf_out, ctx_out, flank_text = sys.argv[1:6]
flank = int(flank_text)
refs = _load_reference_fasta(fasta_path)

rows = [json.loads(x) for x in Path(src).read_text().splitlines() if x.strip()]
snv = [r for r in rows if len(r["ref"]) == 1 and len(r["alt"]) == 1]
print(f"total={len(rows)} snv={len(snv)} dropped_non_snv={len(rows) - len(snv)}")

kept, mismatch = [], 0
with Path(ctx_out).open("w") as ch:
    for r in snv:
        seq = refs.get(str(r["chrom"]))
        if seq is None:
            continue
        i = int(r["pos"]) - 1  # VCF POS is 1-based
        if not (0 <= i < len(seq)):
            continue
        if seq[i].upper() != r["ref"].upper():
            mismatch += 1
            continue
        lo, hi = max(0, i - flank), min(len(seq), i + flank + 1)
        ch.write(json.dumps({
            "variant_id": r["variant_id"],
            "chrom": str(r["chrom"]), "pos": int(r["pos"]),
            "ref": r["ref"], "alt": r["alt"],
            "label_group": r["label_group"],
            "ref_context": seq[lo:hi].upper(),
            "context_flank": flank,
            "variant_offset_in_context": i - lo,
        }, sort_keys=True) + "\n")
        kept.append(r)

with Path(vcf_out).open("w") as vh:
    vh.write("##fileformat=VCFv4.2\n#CHROM\tPOS\tID\tREF\tALT\n")
    for r in kept:
        vh.write(f'{r["chrom"]}\t{r["pos"]}\t{r["variant_id"]}\t{r["ref"]}\t{r["alt"]}\n')
print(f"kept={len(kept)} fasta_ref_mismatch={mismatch}")
PY
echo "VCF records: $(( $(wc -l < "$VCF") - 2 ))"

# --- 4. Carbon zero-shot delta-logLik on the same variants -----------------
log "score Carbon delta-logLik (window=$WINDOW_BP dtype=$DTYPE)"
SCORES="$OUT/carbon_zero_shot_scores.jsonl"
META="$OUT/carbon_zero_shot_metadata.json"
CACHE="$OUT/logp_cache.jsonl"
python - "$VCF" "$FASTA" "$SCORES" "$META" "$CACHE" "$WINDOW_BP" "$DTYPE" "$CARBON_DIR" "$CARBON_REV" <<'PY'
import sys
from geno_lewm.carbon_zero_shot import load_carbon_logp_scorer, write_carbon_zero_shot_scores

vcf, fasta, scores, meta, cache, window_bp, dtype, model_dir, revision = sys.argv[1:10]
# Carbon ships custom modeling code, and unlike the encoder path -- which uses
# the repo's own CarbonDNATokenizer and AutoModel -- likelihood scoring needs
# AutoModelForCausalLM to execute it. Opting in is the repo's sanctioned
# pattern precisely because the revision below is pinned, so the code being
# trusted is a fixed, audited commit rather than whatever HEAD becomes.
scorer = load_carbon_logp_scorer(
    model_dir,
    revision=revision,
    dtype=dtype,
    device="cuda",
    trust_remote_code=True,
)
summary = write_carbon_zero_shot_scores(
    vcf_path=vcf,
    fasta_path=fasta,
    output_scores=scores,
    scorer=scorer,
    carbon_model=model_dir,
    carbon_revision=revision,
    window_bp=int(window_bp),
    logp_cache_jsonl=cache,
    metadata_output=meta,
)
print(f"records={summary.records} new_logp_evals={summary.new_logp_evaluations}")
PY

# --- 5. upload ------------------------------------------------------------
log "upload results to $OUT_REPO/$RUN_NAME"
hf upload "$OUT_REPO" "$SCORES"  "$RUN_NAME/carbon_zero_shot_scores.jsonl" --repo-type dataset
hf upload "$OUT_REPO" "$META"    "$RUN_NAME/carbon_zero_shot_metadata.json" --repo-type dataset
hf upload "$OUT_REPO" "$CONTEXT" "$RUN_NAME/variant_context.jsonl"          --repo-type dataset
log "DONE — results at https://huggingface.co/datasets/$OUT_REPO/tree/main/$RUN_NAME"
