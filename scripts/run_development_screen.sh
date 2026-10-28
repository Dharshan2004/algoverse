#!/usr/bin/env bash
# Development B/R screen (both models). Requires GPU + HF access. Not for CI.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
INPUT="${1:-data/processed/development_inputs.jsonl}"

python3 scripts/adapt_sabika_hf.py --split development --out "$INPUT"

for MODEL in gemma ministral; do
  OUT="runs/dev-screen-${MODEL}-BR"
  echo "=== ${MODEL}: B/R only → ${OUT} ==="
  python3 scripts/run_inference.py \
    --model "$MODEL" \
    --variants B R \
    --input "$INPUT" \
    --out "$OUT"
  python3 scripts/audit_smoke.py "$OUT" || true
done
