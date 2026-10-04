#!/usr/bin/env bash
# 13 Context Window Eviction Tester — real-model run (DeepSeek V4.1 Flash as reader + summarizer).
# Needs DEEPSEEK_API_KEY in ~/TradingAgents/.env. Approx. cost at Oct 2026 peak prices: ~$0.65.
set -euo pipefail
cd "$(dirname "$0")"
set -a; source ~/TradingAgents/.env; set +a
E="uv run --no-sync evalkit context-eviction"
LLM=deepseek:deepseek-flash
mkdir -p out/real

echo "== 400-token budget: 5 strategies x 3 noise levels x 3 paired trials (~\$0.45) =="
$E run --llm $LLM --strategies full,fifo,window,summary,retrieval --noise 50,200,800 \
    --trials 3 --seed 0 --workers 8 | tee out/real/sweep-budget400.txt

echo "== 120-token budget, 8 facts, 30% hard noise, 2 trials (~\$0.20) =="
$E run --llm $LLM --strategies window,summary,retrieval --noise 50,800 --budget 120 \
    --facts 8 --hard-noise 0.3 --trials 2 --workers 8 | tee out/real/tight-budget120.txt

echo "== offline idealized reader/summarizer on the same grids, for comparison (free) =="
$E run --strategies full,fifo,window,summary,retrieval --noise 50,200,800 --trials 3 --seed 0 \
    | tee out/real/offline-budget400.txt
$E run --strategies window,summary,retrieval --noise 50,800 --budget 120 --facts 8 \
    --hard-noise 0.3 --trials 2 | tee out/real/offline-budget120.txt

echo "== per-probe view: real summarizer, checked-in 120-noise scenario, 200-token budget (~\$0.01) =="
$E inspect --scenario scenario-noise120.json --strategy summary --budget 200 --llm $LLM \
    --show-context | tee out/real/inspect-summary-noise120.txt
