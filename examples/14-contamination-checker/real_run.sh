#!/usr/bin/env bash
# Real-model run of the contamination checker against the DeepSeek API.
#  1. the demo scan with deepseek-flash as the LLM judge for suspicious items;
#  2. a leak-recall test: deepseek-flash writes 5 kinds of leak (+ a same-topic
#     non-leak) for 40 eval items (12 demo + 28 model-written); every detector is
#     scored per kind and the judge rates every (item, document) pair directly.
# Needs DEEPSEEK_API_KEY in ~/TradingAgents/.env (never printed). Cost ~ $0.05.
set -euo pipefail
cd "$(dirname "$0")/../.."
set -a; source ~/TradingAgents/.env; set +a
EX=examples/14-contamination-checker
OUT=$EX/out/real
MODEL="${EVALKIT_REAL_LLM:-deepseek:deepseek-flash}"
mkdir -p "$OUT"

echo "== 1. index + scan the demo corpus, judge suspicious items with $MODEL"
uv run --no-sync evalkit contamination index --eval "$EX/eval.jsonl" --out "$OUT/index.json"
uv run --no-sync evalkit contamination scan --index "$OUT/index.json" \
  --train "$EX/train.jsonl" --out "$OUT/scan-judged" --llm "$MODEL"

echo; echo "== 2. leak-recall test: model-written leaks per kind, detectors + judge"
# --reuse keeps an existing out/real/leaktest/{eval,leaks}.jsonl (re-scores for free);
# delete that directory to regenerate.
uv run --no-sync evalkit contamination leaktest --eval "$EX/eval.jsonl" --synth 28 \
  --llm "$MODEL" --judge "$MODEL" --out "$OUT/leaktest" --reuse

if [[ "${REAL_RUN_PROBE:-0}" == 1 ]]; then
  echo; echo "== 3. threshold probe: same items + leaks, --cosine-suspicious 0.25 (~\$0.035)"
  uv run --no-sync evalkit contamination leaktest --eval "$OUT/leaktest/eval.jsonl" \
    --leaks "$OUT/leaktest/leaks.jsonl" --llm "$MODEL" --judge "$MODEL" \
    --cosine-suspicious 0.25 --out "$OUT/leaktest-cos025"
fi
