#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
#
# Full v0.2 benchmark-suite run for Hugging Face Jobs.
#
# This consumes the model/eval package produced by tools/jobs/eval_run.sh and
# the dataset package produced by tools/jobs/proof_run.sh, stages the public
# chr13 v0.2 benchmark input bundle, reruns GenoLeWM scores, Carbon baselines,
# rollout-fidelity rows, aggregate eval, efficiency, rollout-speed, and the
# final release-input readiness gate.
set -euo pipefail

WORK="${WORK:-/tmp/geno-v02-suite}"
ROOT="${ROOT:-$WORK/suite}"
CARBON_DIR="${CARBON_DIR:-/carbon}"

RUNS_REPO="${RUNS_REPO:-abdelstark/geno-lewm-runs}"
RUN_NAME="${RUN_NAME:-geno-lewm-proof}"
MODEL_SUBPATH="${MODEL_SUBPATH:-$RUN_NAME/eval}"
DATASET_SUBPATH="${DATASET_SUBPATH:-$RUN_NAME/dataset}"

INPUT_BUNDLE_REPO="${INPUT_BUNDLE_REPO:-abdelstark/geno-lewm-runs}"
INPUT_BUNDLE_SUBPATH="${INPUT_BUNDLE_SUBPATH:-geno-lewm-v02-autonomous/9bec68ad04f2787dd0dfdf42d116050061ad53f6/suite-r11}"
REFERENCE_FASTA_URL="${REFERENCE_FASTA_URL:-https://ftp.ensembl.org/pub/release-110/fasta/homo_sapiens/dna/Homo_sapiens.GRCh38.dna.chromosome.13.fa.gz}"

REL="${REL:-geno-lewm-v0.2.1-r1}"
SNAP="${SNAP:-geno-lewm-data-v0.2.1-r1}"
COMMIT="${COMMIT:-$(git rev-parse HEAD 2>/dev/null || echo unknown)}"
HARDWARE="${HARDWARE:-$(python - <<'PY' 2>/dev/null || echo unknown
import platform
try:
    import torch
    device = "cuda:" + torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"
except Exception:
    device = "unknown"
print(f"{device} ({platform.platform(terse=True)})")
PY
)}"

EFF_SAMPLES="${EFF_SAMPLES:-1}"
EFF_WARMUP="${EFF_WARMUP:-0}"
EFF_BATCH_SIZE="${EFF_BATCH_SIZE:-8}"
ROLLOUT_ITERS="${ROLLOUT_ITERS:-20}"
ROLLOUT_WARMUP="${ROLLOUT_WARMUP:-5}"
UPLOAD="${UPLOAD:-1}"
SUITE_UPLOAD_SUBPATH="${SUITE_UPLOAD_SUBPATH:-$RUN_NAME/suite}"

ACCEPTED_BY="${ACCEPTED_BY:-AbdelStark/GenoLeWM maintainer workflow}"
ACCEPTED_AT="${ACCEPTED_AT:-2026-06-08T08:14:56Z}"
DECISION_URL="${DECISION_URL:-https://github.com/AbdelStark/GenoLeWM/issues/42#issuecomment-4646678395}"
RESCOPE_RATIONALE="${RESCOPE_RATIONALE:-The H200 benchmark at branch head may miss the original K=20 RFC-0004 target at training-shaped dimensions. v0.2 benchmark readiness should publish measured speed values and preserve any K=20 miss as a negative finding instead of blocking VEP and rollout-fidelity evidence.}"
REPLACEMENT_TARGET="${REPLACEMENT_TARGET:-For v0.2, publish measured K=5 and K=20 autoregressive rollout speedups with any K=20 target miss recorded; retain true KV-cache speed-target closure for a later RFC-0004 implementation.}"

log() { echo "=== $* ==="; }
fail() { echo "FATAL: $*" >&2; exit 1; }

copy_downloaded_subpath() {
  local source_root="$1"
  local subpath="$2"
  local dest="$3"
  test -d "$source_root/$subpath" || fail "downloaded subpath missing: $subpath"
  mkdir -p "$dest"
  cp -a "$source_root/$subpath/." "$dest/"
}

log "v0.2 suite run: run=$RUN_NAME rel=$REL snap=$SNAP commit=$COMMIT"
[ -n "${HF_TOKEN:-}" ] || fail "HF_TOKEN is required for HF downloads/uploads"
export HF_TOKEN
test -d "$CARBON_DIR" || fail "Carbon-500M not mounted at $CARBON_DIR"
python -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available())" || true
python -c "import safetensors.torch, transformers, pyarrow.parquet" \
  || fail "native deps missing; install with python -m pip install -e '.[train]'"
nvidia-smi || true

rm -rf "$ROOT"
mkdir -p "$ROOT/model" "$ROOT/dataset" "$ROOT/benchmark_inputs" "$ROOT/eval" \
  "$ROOT/cache" "$ROOT/configs" "$ROOT/bench" "$ROOT/carbon"
ln -sfn "$CARBON_DIR" "$ROOT/carbon/500m"

log "download model/eval package from $RUNS_REPO ($MODEL_SUBPATH)"
hf download "$RUNS_REPO" --repo-type model --include "$MODEL_SUBPATH/*" --local-dir "$WORK/dl-model"
copy_downloaded_subpath "$WORK/dl-model" "$MODEL_SUBPATH" "$ROOT/model"
for f in manifest.json predictor.safetensors action_encoder.safetensors calibration.parquet \
         eval_metrics.json eval_report.md efficiency_report.json; do
  test -s "$ROOT/model/$f" || fail "model/eval package missing $f"
done
TRAIN_CFG="$(python - "$ROOT/model" <<'PY'
import json, sys
manifest = json.load(open(f"{sys.argv[1]}/manifest.json"))
print(manifest["training"]["config_file"])
PY
)"
test -s "$ROOT/model/$TRAIN_CFG" || fail "model package missing training config $TRAIN_CFG"

log "download dataset package from $RUNS_REPO ($DATASET_SUBPATH)"
hf download "$RUNS_REPO" --repo-type model --include "$DATASET_SUBPATH/*" --local-dir "$WORK/dl-dataset"
copy_downloaded_subpath "$WORK/dl-dataset" "$DATASET_SUBPATH" "$ROOT/dataset"
for f in dataset_manifest.json dataset_package.json data_card.md SHA256SUMS; do
  test -s "$ROOT/dataset/$f" || fail "dataset package missing $f"
done

log "stage previous public v0.2 input bundle from $INPUT_BUNDLE_REPO ($INPUT_BUNDLE_SUBPATH)"
hf download "$INPUT_BUNDLE_REPO" --repo-type model \
  --include "$INPUT_BUNDLE_SUBPATH/**" \
  --local-dir "$WORK/dl-inputs"
copy_downloaded_subpath "$WORK/dl-inputs" "$INPUT_BUNDLE_SUBPATH/benchmark_inputs" "$ROOT/benchmark_inputs"
cp "$WORK/dl-inputs/$INPUT_BUNDLE_SUBPATH"/eval/*.labels.jsonl "$ROOT/eval/"
cp "$WORK/dl-inputs/$INPUT_BUNDLE_SUBPATH"/eval/*example_specs.jsonl "$ROOT/eval/"
cp "$WORK/dl-inputs/$INPUT_BUNDLE_SUBPATH"/eval/v02_rollout_inputs_report.json "$ROOT/eval/"
copy_downloaded_subpath "$WORK/dl-inputs" "$INPUT_BUNDLE_SUBPATH/cache" "$ROOT/cache"

log "stage chr13 reference FASTA"
curl -fsSL "$REFERENCE_FASTA_URL" -o "$WORK/chr13.fa.gz"
gzip -dc "$WORK/chr13.fa.gz" > "$ROOT/benchmark_inputs/reference.fa"
test -s "$ROOT/benchmark_inputs/reference.fa" || fail "reference FASTA is empty"

log "extract efficiency single-variant window"
read -r EFF_VARIANT EFF_WINDOW_START EFF_WINDOW < <(
  python - "$ROOT/benchmark_inputs/clinvar_coding.efficiency.vcf" \
    "$ROOT/benchmark_inputs/reference.fa" <<'PY'
import sys
from pathlib import Path

vcf = Path(sys.argv[1])
fasta = Path(sys.argv[2])
chrom = pos = ref = alt = None
for line in vcf.read_text(encoding="utf-8").splitlines():
    if not line or line.startswith("#"):
        continue
    fields = line.split("\t")
    chrom, pos, ref, alt = fields[0], int(fields[1]), fields[3], fields[4].split(",")[0]
    break
if chrom is None:
    raise SystemExit("no variant rows in efficiency VCF")
sequence = "".join(line.strip().upper() for line in fasta.read_text(encoding="utf-8").splitlines() if not line.startswith(">"))
window_bp = 12288
start = max(0, pos - 1 - window_bp // 2)
end = start + window_bp
if end > len(sequence):
    end = len(sequence)
    start = max(0, end - window_bp)
window = sequence[start:end]
if len(window) != window_bp:
    raise SystemExit(f"expected {window_bp}bp window, observed {len(window)}")
print(f"{chrom}:{pos}:{ref}:{alt}\t{start}\t{window}")
PY
)
test -n "$EFF_VARIANT" || fail "could not extract efficiency variant"
test -n "$EFF_WINDOW_START" || fail "could not extract efficiency window start"
test -n "$EFF_WINDOW" || fail "could not extract efficiency window"

log "write v0.2 efficiency report"
(
  cd "$ROOT"
  python -m bench.inference --release-efficiency \
    --model-dir model \
    --vcf benchmark_inputs/clinvar_coding.efficiency.vcf \
    --fasta benchmark_inputs/reference.fa \
    --variant "$EFF_VARIANT" \
    --window "$EFF_WINDOW" \
    --window-start-bp "$EFF_WINDOW_START" \
    --output-json model/efficiency_report.v02.json \
    --backend auto \
    --batch-size "$EFF_BATCH_SIZE" \
    --samples "$EFF_SAMPLES" \
    --warmup "$EFF_WARMUP" \
    --commit-sha "$COMMIT" \
    --dataset-snapshot "$SNAP"
)

log "write benchmark suite manifest"
MODEL_ID="$(python - "$ROOT/model/manifest.json" <<'PY'
import sys
from geno_lewm.provenance import load_manifest
print(load_manifest(sys.argv[1]).model_id())
PY
)"
ROOT="$ROOT" MODEL_ID="$MODEL_ID" REL="$REL" SNAP="$SNAP" COMMIT="$COMMIT" HARDWARE="$HARDWARE" \
python - <<'PY'
import json
import os
from pathlib import Path

root = Path(os.environ["ROOT"])
template = json.loads(Path("configs/first_experiment/v0.2_benchmark_suite.template.json").read_text(encoding="utf-8"))
template["identity"] = {
    "model_id": os.environ["MODEL_ID"],
    "model_release": os.environ["REL"],
    "dataset_snapshot": os.environ["SNAP"],
    "commit": os.environ["COMMIT"],
    "hardware": os.environ["HARDWARE"],
}
template["artifacts"] = {
    "model_dir": "model",
    "checkpoint": "model/predictor.safetensors",
    "config": "model/training_config.effective.yaml",
    "dataset_manifest": "dataset/dataset_manifest.json",
    "efficiency_report": "model/efficiency_report.v02.json",
}
template["aggregate"]["metrics_json"] = "model/eval_metrics.v02.json"
template["aggregate"]["report_md"] = "model/eval_report.v02.md"
for benchmark in template["benchmarks"]:
    baseline = benchmark.get("carbon_baseline")
    if isinstance(baseline, dict):
        baseline.update(
            {
                "carbon_model_dir": "carbon/500m",
                "carbon_revision": "5d31d59b3c845b288a13aedb1358934196852eec",
                "device": "cuda",
                "dtype": "fp32",
                "trust_remote_code": True,
                "allow_network_download": False,
                "logp_cache_jsonl": "cache/carbon_zero_shot_logp.jsonl",
            }
        )
    if benchmark["kind"] == "vep":
        benchmark["bootstrap_resamples"] = 20
        benchmark["bootstrap_seed"] = 20260608
    if benchmark["kind"] == "rollout":
        benchmark["recall_k"] = 4
template.pop("readiness", None)
path = root / "v0.2_benchmark_suite.autonomous.json"
path.write_text(json.dumps(template, indent=2, sort_keys=True) + "\n", encoding="utf-8")
print(path)
PY

log "run benchmark suite"
(
  cd "$ROOT"
  python -m tools.release.v02_benchmark_suite \
    --manifest v0.2_benchmark_suite.autonomous.json \
    --output-report model/v0.2_benchmark_suite_report.json \
    --execute | tee model/v0.2_benchmark_suite.stdout.log
)

log "run AR rollout speed benchmark"
set +e
(
  cd "$ROOT"
  python -m bench.rollout \
    --k 5 --k 20 \
    --batch-size 8 \
    --d-state 1024 \
    --d-action 64 \
    --d-hidden 1024 \
    --n-heads 8 \
    --n-cross-layers 6 \
    --n-self-layers 1 \
    --ffn-dim 4096 \
    --iters "$ROLLOUT_ITERS" \
    --warmup "$ROLLOUT_WARMUP" \
    --seed 20260606 \
    --device cuda \
    --dtype bf16 \
    --output-json bench/rollout.ar_speed.json \
    --require-targets
)
ROLLOUT_RC=$?
set -e
test -s "$ROOT/bench/rollout.ar_speed.json" || fail "rollout speed report was not written"
if [ "$ROLLOUT_RC" -ne 0 ]; then
  log "record accepted #42 rollout-speed scope decision"
  (
    cd "$ROOT"
    python -m tools.release.rollout_speed_scope \
      --rollout-speed-report bench/rollout.ar_speed.json \
      --output bench/rollout_speed_scope.json \
      --accepted-by "$ACCEPTED_BY" \
      --accepted-at "$ACCEPTED_AT" \
      --decision-url "$DECISION_URL" \
      --rationale "$RESCOPE_RATIONALE" \
      --replacement-target "$REPLACEMENT_TARGET" \
      --issue-ref '#42' \
      --issue-ref '#197'
  )
fi

log "run final v0.2 readiness gate"
READINESS_ARGS=(
  python -m tools.release.v02_benchmark_readiness
  --metrics-json model/eval_metrics.v02.json
  --rollout-speed-report bench/rollout.ar_speed.json
  --efficiency-report model/efficiency_report.v02.json
  --suite-report model/v0.2_benchmark_suite_report.json
  --output model/v0.2_benchmark_readiness_report.json
  --require-ok
  --require-release-inputs
)
if [ -s "$ROOT/bench/rollout_speed_scope.json" ]; then
  READINESS_ARGS+=(--rollout-speed-scope-report bench/rollout_speed_scope.json)
fi
(
  cd "$ROOT"
  "${READINESS_ARGS[@]}"
)

log "upload v0.2 suite evidence"
if [ "$UPLOAD" = "1" ]; then
  hf upload "$RUNS_REPO" "$ROOT" "$SUITE_UPLOAD_SUBPATH" --repo-type model \
    --exclude "carbon/**" \
    --commit-message "Upload v0.2 suite evidence for $RUN_NAME"
fi

echo "GENO_LEWM_V02_SUITE_OK $RUN_NAME readiness=$ROOT/model/v0.2_benchmark_readiness_report.json"
