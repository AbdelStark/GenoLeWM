#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Pinned Carbon state-normalization audit for Hugging Face Jobs.
set -euo pipefail

CARBON_DIR="${CARBON_DIR:-/carbon}"
COMMIT_SHA="${COMMIT_SHA:?COMMIT_SHA is required}"
OUTPUT_JSON="${OUTPUT_JSON:-/tmp/geno-state-contract/state_contract_audit.json}"
UPLOAD_REPO="${UPLOAD_REPO:-abdelstark/geno-lewm-runs}"
UPLOAD_PATH="${UPLOAD_PATH:-state-contract-audits/$COMMIT_SHA/state_contract_audit.json}"
UPLOAD="${UPLOAD:-1}"

test -d "$CARBON_DIR" || { echo "FATAL: Carbon checkpoint is not mounted at $CARBON_DIR" >&2; exit 1; }
mkdir -p "$(dirname "$OUTPUT_JSON")"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

python -m tools.research.state_contract_audit \
  --carbon-model-dir "$CARBON_DIR" \
  --commit-sha "$COMMIT_SHA" \
  --output-json "$OUTPUT_JSON" \
  --device cuda

if [ "$UPLOAD" = "1" ]; then
  test -n "${HF_TOKEN:-}" || { echo "FATAL: HF_TOKEN is required when UPLOAD=1" >&2; exit 1; }
  hf upload "$UPLOAD_REPO" "$OUTPUT_JSON" "$UPLOAD_PATH" --repo-type model
fi

echo "GENO_LEWM_STATE_CONTRACT_AUDIT_OK $OUTPUT_JSON"
