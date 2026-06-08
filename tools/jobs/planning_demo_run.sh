#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
#
# Manifest-runtime planning demo for Hugging Face Jobs.
#
# This is the #204 planning/showcase runner: it consumes a packaged model/eval
# directory, creates a deterministic multi-SNV target window from a public chr13
# reference FASTA, runs geno-lewm-plan with real local model inference, and
# writes a public-safe transcript plus machine-readable planning manifest.
set -euo pipefail

WORK="${WORK:-/tmp/geno-planning-demo}"
OUT="${OUT:-$WORK/out}"
CARBON_DIR="${CARBON_DIR:-/carbon}"

RUNS_REPO="${RUNS_REPO:-abdelstark/geno-lewm-runs}"
RUN_NAME="${RUN_NAME:-geno-lewm-proof}"
MODEL_SUBPATH="${MODEL_SUBPATH:-$RUN_NAME/eval}"
DEMO_UPLOAD_SUBPATH="${DEMO_UPLOAD_SUBPATH:-$RUN_NAME/planning-demo}"

REFERENCE_FASTA_URL="${REFERENCE_FASTA_URL:-https://ftp.ensembl.org/pub/release-110/fasta/homo_sapiens/dna/Homo_sapiens.GRCh38.dna.chromosome.13.fa.gz}"
WINDOW_BP="${WINDOW_BP:-12288}"
WINDOW_START_1BASED="${WINDOW_START_1BASED:-32352000}"
PLAN_HORIZON="${PLAN_HORIZON:-3}"
PLAN_ITERATIONS="${PLAN_ITERATIONS:-4}"
PLAN_SAMPLES="${PLAN_SAMPLES:-128}"
PLAN_ELITE="${PLAN_ELITE:-16}"
PLAN_SEED="${PLAN_SEED:-20260608}"
UPLOAD="${UPLOAD:-1}"

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

log "planning demo: run=$RUN_NAME model=$MODEL_SUBPATH"
[ -n "${HF_TOKEN:-}" ] || fail "HF_TOKEN is required for HF downloads/uploads"
export HF_TOKEN
test -d "$CARBON_DIR" || fail "Carbon-500M not mounted at $CARBON_DIR"
python -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available())" || true
python -c "import safetensors.torch, transformers, pyarrow.parquet" \
  || fail "native deps missing; install with python -m pip install -e '.[train]'"
nvidia-smi || true

rm -rf "$OUT"
mkdir -p "$OUT/model" "$OUT/demo"

log "download model/eval package from $RUNS_REPO ($MODEL_SUBPATH)"
hf download "$RUNS_REPO" --repo-type model --include "$MODEL_SUBPATH/*" --local-dir "$WORK/dl-model"
copy_downloaded_subpath "$WORK/dl-model" "$MODEL_SUBPATH" "$OUT/model"
for f in manifest.json predictor.safetensors action_encoder.safetensors calibration.parquet; do
  test -s "$OUT/model/$f" || fail "model package missing $f"
done

log "stage deterministic source and target FASTA windows"
curl -fsSL "$REFERENCE_FASTA_URL" -o "$WORK/chr13.fa.gz"
WORK="$WORK" WINDOW_BP="$WINDOW_BP" WINDOW_START_1BASED="$WINDOW_START_1BASED" OUT="$OUT" python - <<'PY'
import json
import os
import gzip
from pathlib import Path

out = Path(os.environ["OUT"])
window_bp = int(os.environ["WINDOW_BP"])
start = int(os.environ["WINDOW_START_1BASED"])
raw = gzip.open(Path(os.environ.get("WORK", "/tmp/geno-planning-demo")) / "chr13.fa.gz", "rt", encoding="utf-8")
sequence = "".join(line.strip().upper() for line in raw if not line.startswith(">"))
raw.close()
offset = start - 1
window = sequence[offset : offset + window_bp]
if len(window) != window_bp:
    raise SystemExit(f"expected {window_bp}bp window, observed {len(window)}")
target = list(window)
edits = []
positions = (window_bp // 3, window_bp // 2, (2 * window_bp) // 3)
bases = "ACGT"
for rel_pos in positions:
    ref = target[rel_pos]
    alt = next(base for base in bases if base != ref)
    target[rel_pos] = alt
    edits.append(
        {
            "chrom": "13",
            "pos": start + rel_pos,
            "rel_pos": rel_pos,
            "ref": ref,
            "alt": alt,
        }
    )
demo = out / "demo"
demo.mkdir(parents=True, exist_ok=True)
(demo / "source_window.fa").write_text(">chr13_source_window\n" + window + "\n", encoding="utf-8")
(demo / "target_window.fa").write_text(">chr13_target_window\n" + "".join(target) + "\n", encoding="utf-8")
(demo / "target_edits.json").write_text(json.dumps(edits, indent=2, sort_keys=True) + "\n", encoding="utf-8")
print(json.dumps({"window_bp": window_bp, "start_1based": start, "target_edits": edits}, sort_keys=True))
PY

PLAN_JSON="$OUT/demo/plan.json"
PLAN_STDOUT="$OUT/demo/plan.stdout.json"
TRANSCRIPT="$OUT/demo/planning-demo-transcript.md"
MANIFEST="$OUT/demo/planning_demo_manifest.json"
PLAN_CMD=(
  geno-lewm-plan
  --quiet
  --no-banner
  --model-dir model
  --backend auto
  --window-fasta demo/source_window.fa
  --target-fasta demo/target_window.fa
  --horizon "$PLAN_HORIZON"
  --iterations "$PLAN_ITERATIONS"
  --samples "$PLAN_SAMPLES"
  --elite "$PLAN_ELITE"
  --edit-types snv
  --edge-margin 64
  --position-bin-bp 8
  --seed "$PLAN_SEED"
  --output demo/plan.json
)

log "run geno-lewm-plan in manifest_runtime mode"
(
  cd "$OUT"
  "${PLAN_CMD[@]}" | tee "$PLAN_STDOUT"
)
test -s "$PLAN_JSON" || fail "plan.json was not written"

log "write transcript and planning demo manifest"
OUT="$OUT" PLAN_CMD_JSON="$(printf '%s\n' "${PLAN_CMD[@]}" | python -c 'import json,sys; print(json.dumps([line.rstrip(\"\\n\") for line in sys.stdin]))')" \
python - <<'PY'
import json
import os
from datetime import datetime, timezone
from pathlib import Path

from geno_lewm.provenance import sha256_file

out = Path(os.environ["OUT"])
demo = out / "demo"
model = out / "model"
plan = json.loads((demo / "plan.json").read_text(encoding="utf-8"))
stdout = json.loads((demo / "plan.stdout.json").read_text(encoding="utf-8"))
command = json.loads(os.environ["PLAN_CMD_JSON"])

def identity(path: Path, label: str) -> dict[str, object]:
    return {"path": label, "sha256": sha256_file(path), "size_bytes": path.stat().st_size}

transcript = demo / "planning-demo-transcript.md"
transcript.write_text(
    "\n".join(
        [
            "# GenoLeWM Planning Demo",
            "",
            f"Generated: {datetime.now(timezone.utc).isoformat().replace('+00:00', 'Z')}",
            "",
            "This demo runs manifest-runtime planning from local model artifacts. It is not clinical, deployment-readiness, privacy-assurance, or runtime-assurance evidence.",
            "",
            "## Command",
            "",
            "```bash",
            " ".join(command),
            "```",
            "",
            "## Result",
            "",
            f"- evaluation_mode: {plan.get('evaluation_mode')}",
            f"- best_distance: {stdout.get('best_distance')}",
            f"- n_evaluations: {stdout.get('n_evaluations')}",
            f"- stopped_reason: {stdout.get('stopped_reason')}",
            "",
        ]
    )
    + "\n",
    encoding="utf-8",
)
manifest = {
    "schema_version": "1.0.0",
    "generated_by": "tools.jobs.planning_demo_run",
    "generated_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
    "status": "passed" if plan.get("evaluation_mode") == "manifest_runtime" else "failed",
    "command": command,
    "claim_boundary": (
        "This planning demo is research evidence only. It does not establish clinical utility, "
        "deployment readiness, privacy assurance, runtime assurance, or useful planning behavior."
    ),
    "negative_findings": [
        "The demo records one deterministic synthetic multi-SNV target window and does not prove useful planning behavior.",
        "Planning quality must be judged from the measured best distance and generated edit sequence.",
    ],
    "artifacts": {
        "model_manifest": identity(model / "manifest.json", "model/manifest.json"),
        "predictor": identity(model / "predictor.safetensors", "model/predictor.safetensors"),
        "action_encoder": identity(model / "action_encoder.safetensors", "model/action_encoder.safetensors"),
        "source_window": identity(demo / "source_window.fa", "demo/source_window.fa"),
        "target_window": identity(demo / "target_window.fa", "demo/target_window.fa"),
        "target_edits": identity(demo / "target_edits.json", "demo/target_edits.json"),
        "plan": identity(demo / "plan.json", "demo/plan.json"),
        "plan_stdout": identity(demo / "plan.stdout.json", "demo/plan.stdout.json"),
        "transcript": identity(transcript, "demo/planning-demo-transcript.md"),
    },
    "plan_summary": {
        "evaluation_mode": plan.get("evaluation_mode"),
        "model_id": plan.get("runtime", {}).get("model_id"),
        "best_distance": plan.get("result", {}).get("best_distance"),
        "best_objective": plan.get("result", {}).get("best_objective"),
        "n_evaluations": plan.get("result", {}).get("n_evaluations"),
        "elapsed_seconds": plan.get("result", {}).get("elapsed_seconds"),
        "stopped_reason": plan.get("result", {}).get("stopped_reason"),
        "best_edits": plan.get("result", {}).get("best_edits"),
    },
}
if manifest["status"] != "passed":
    raise SystemExit("planning demo did not use manifest_runtime mode")
(demo / "planning_demo_manifest.json").write_text(
    json.dumps(manifest, indent=2, sort_keys=True) + "\n",
    encoding="utf-8",
)
print(json.dumps({"status": manifest["status"], "best_distance": manifest["plan_summary"]["best_distance"]}, sort_keys=True))
PY

if [ "$UPLOAD" = "1" ]; then
  log "upload planning demo to $RUNS_REPO ($DEMO_UPLOAD_SUBPATH)"
  hf upload "$RUNS_REPO" "$OUT/demo" "$DEMO_UPLOAD_SUBPATH" --repo-type model \
    --commit-message "Upload planning demo for $RUN_NAME"
fi

echo "GENO_LEWM_PLANNING_DEMO_OK $RUN_NAME manifest=$MANIFEST"
