#!/usr/bin/env bash
# End-to-end demo: generate -> validate+label -> coverage -> probe the toy system.
set -euo pipefail
cd "$(dirname "$0")"
OUT=out
mkdir -p "$OUT"

uv run --no-sync evalkit edge-case-gen generate \
  --spec spec.yaml --out "$OUT/candidates.jsonl" --llm "${LLM:-mock}" --n-llm 4 --seed 7

uv run --no-sync evalkit edge-case-gen validate \
  --spec spec.yaml --in "$OUT/candidates.jsonl" --out "$OUT/golden.jsonl" \
  --rejects "$OUT/rejects.jsonl" --oracle expense_system.py:reference

uv run --no-sync evalkit edge-case-gen coverage \
  --spec spec.yaml --in "$OUT/golden.jsonl" --json "$OUT/coverage.json" --min-pairwise 0.95

uv run --no-sync evalkit edge-case-gen probe \
  --spec spec.yaml --in "$OUT/golden.jsonl" --system expense_system.py:naive_triage \
  --out "$OUT/probe.json" --show 10 || true

# The fixed system (validate first, then apply policy) passes the same golden set.
uv run --no-sync evalkit edge-case-gen probe \
  --spec spec.yaml --in "$OUT/golden.jsonl" --system expense_system.py:robust_triage \
  --out "$OUT/probe-robust.json" --show 3 --max-fail-rate 0.0
