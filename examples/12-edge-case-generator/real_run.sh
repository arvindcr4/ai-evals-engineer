#!/usr/bin/env bash
# Real-model run: rule cases + DeepSeek-written cases -> validate (claim check,
# oracle labels cross-checked by DeepSeek) -> coverage -> probe the toy systems.
# Needs DEEPSEEK_API_KEY in ~/TradingAgents/.env (never printed). Cost ~$0.03.
set -euo pipefail
cd "$(dirname "$0")"
set -a; source ~/TradingAgents/.env; set +a
LLM=${LLM:-deepseek:deepseek-flash}
OUT=out/real
mkdir -p "$OUT"

uv run --no-sync evalkit edge-case-gen generate \
  --spec spec.yaml --out "$OUT/candidates.jsonl" --llm "$LLM" --n-llm "${N_LLM:-10}" \
  --seed 7 --temperature 0 | tee "$OUT/generate.txt"

uv run --no-sync evalkit edge-case-gen validate \
  --spec spec.yaml --in "$OUT/candidates.jsonl" --out "$OUT/golden.jsonl" \
  --rejects "$OUT/rejects.jsonl" --oracle expense_system.py:reference \
  --label-llm "$LLM" | tee "$OUT/validate.txt"

uv run --no-sync evalkit edge-case-gen coverage \
  --spec spec.yaml --in "$OUT/golden.jsonl" --json "$OUT/coverage.json" \
  | tee "$OUT/coverage.txt"

uv run --no-sync evalkit edge-case-gen probe \
  --spec spec.yaml --in "$OUT/golden.jsonl" --system expense_system.py:naive_triage \
  --out "$OUT/probe.json" --show 10 | tee "$OUT/probe.txt" || true

uv run --no-sync evalkit edge-case-gen probe \
  --spec spec.yaml --in "$OUT/golden.jsonl" --system expense_system.py:robust_triage \
  --out "$OUT/probe-robust.json" --show 5 | tee "$OUT/probe-robust.txt" || true
