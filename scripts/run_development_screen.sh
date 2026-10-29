#!/usr/bin/env bash
# Development B/R screen (both models). Requires GPU + HF access. Not for CI.
# OUTPUT_PROTOCOL defaults to native_v1. For a forced A/B first token on every
# model in this script: OUTPUT_PROTOCOL=ab_first_token_v1 bash scripts/run_development_screen.sh
# Gemma only: see the ab_first_token_v1 command in docs/DATASET_PREPARATION.md.
# Ministral needs mistral-common: pip install -r requirements.txt
# or pip install -e ".[inference]" before this script.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
INPUT="${1:-data/processed/development_inputs.jsonl}"
PROTOCOL="${OUTPUT_PROTOCOL:-native_v1}"

python3 scripts/adapt_hf_dataset.py --split development --out "$INPUT"

for MODEL in gemma ministral; do
  OUT="runs/dev-screen-${MODEL}-BR"
  echo "=== ${MODEL}: B/R only → ${OUT} ==="
  python3 scripts/run_inference.py \
    --model "$MODEL" \
    --variants B R \
    --input "$INPUT" \
    --output-protocol "$PROTOCOL" \
    --out "$OUT"
  python3 scripts/audit_smoke.py "$OUT" || true
done
