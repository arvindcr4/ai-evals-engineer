#!/usr/bin/env bash
# 03 Calibrated LLM-as-a-Judge — offline demo with a simulated judge whose
# biases are known, so you can see the audit recover them and calibration remove them.
# Swap in a real judge with:  --llm openai:gpt-4o-mini --judge-family gpt --rejudge
set -euo pipefail
cd "$(dirname "$0")"
OUT=${OUT:-out}
mkdir -p "$OUT"

echo "== 1. Build 500 human-labelled anchors (length and model style independent of quality)"
uv run --no-sync evalkit calibrated-judge make-anchors --n 500 --seed 0 --out anchors.jsonl

echo; echo "== 2. Audit the judge (both orders + pointwise scores)"
uv run --no-sync evalkit calibrated-judge audit --anchors anchors.jsonl --pointwise \
  --judgments "$OUT/judgments.jsonl" --rejudge \
  --out-md "$OUT/audit.md" --out-json "$OUT/audit.json"

echo; echo "== 3. Fit corrections on 60% of anchors, evaluate on the held-out 40%"
uv run --no-sync evalkit calibrated-judge calibrate --anchors anchors.jsonl \
  --judgments "$OUT/judgments.jsonl" --out "$OUT/calibration.json" \
  --out-md "$OUT/calibration.md" --out-json "$OUT/calibration_report.json"

echo; echo "== 4. Control: an unbiased judge should audit clean"
uv run --no-sync evalkit calibrated-judge audit --anchors anchors.jsonl \
  --sim-position-bias 0 --sim-verbosity-bias 0 --sim-self-bias 0 --sim-overconfidence 1 \
  --out-md "$OUT/audit_unbiased.md" | sed -n '1,20p'
