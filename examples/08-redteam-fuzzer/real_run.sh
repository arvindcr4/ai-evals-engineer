#!/usr/bin/env bash
# Agent Red-Team Fuzzer (system 08) against a REAL policy model (DeepSeek).
#
# Defensive testing of our own toy agent: deepseek-flash replaces the
# deterministic toy policy; the guardrail layer, tools and oracles are unchanged.
#
#   1. adaptive campaigns (seed sweep + 30 feedback-driven mutations per level),
#      mock policy vs deepseek-flash
#   2. paired replay: the exact 39 cases the mock executed at `none`, replayed at
#      every level under both policies (--iterations 0), so per-class attack
#      success is compared on identical inputs
#
# Needs DEEPSEEK_API_KEY (loaded from ~/TradingAgents/.env; never printed).
# Spend is capped with --max-cost-usd; the whole script costs well under $1.
set -euo pipefail
cd "$(dirname "$0")/../.."
set -a; source ~/TradingAgents/.env; set +a

EX=examples/08-redteam-fuzzer
OUT=$EX/out/real
mkdir -p "$OUT" "$EX/real_output"
LLM=deepseek:deepseek-flash
RT="uv run --no-sync evalkit redteam-fuzzer"

echo "# 1a. adaptive campaign, mock policy (offline baseline)"
$RT run --guardrail all --iterations 30 --seed 1 --seeds $EX/seeds.jsonl --out "$OUT/mock-adaptive"

echo "# 1b. adaptive campaign, $LLM policy"
$RT run --guardrail all --iterations 30 --seed 1 --seeds $EX/seeds.jsonl \
  --llm $LLM --max-cost-usd 0.45 --out "$OUT/real-adaptive"

echo "# 2a. paired replay of the mock's 39 executed cases, mock policy"
$RT run --guardrail all --iterations 0 --seeds "$OUT/mock-adaptive/cases-none.jsonl" \
  --out "$OUT/mock-paired"

echo "# 2b. paired replay, $LLM policy"
$RT run --guardrail all --iterations 0 --seeds "$OUT/mock-adaptive/cases-none.jsonl" \
  --llm $LLM --max-cost-usd 0.45 --out "$OUT/real-paired"

echo "# 3. comparisons"
$RT compare "$OUT/mock-adaptive/findings.jsonl" "$OUT/real-adaptive/findings.jsonl" \
  --labels mock deepseek-flash --out "$OUT/compare-adaptive.md"
$RT compare "$OUT/mock-paired/findings.jsonl" "$OUT/real-paired/findings.jsonl" \
  --labels mock deepseek-flash --out "$OUT/compare-paired.md"

# Small, commit-worthy copies (scorecards + comparisons; no raw logs).
cp "$OUT/real-adaptive/scorecard.md" "$EX/real_output/scorecard-real-adaptive.md"
cp "$OUT/real-paired/scorecard.md" "$EX/real_output/scorecard-real-paired.md"
cp "$OUT/compare-adaptive.md" "$OUT/compare-paired.md" "$EX/real_output/"
echo "artifacts in $OUT, summaries in $EX/real_output"
