#!/usr/bin/env bash
# Shadow Routing Comparator against the REAL DeepSeek API.
#
# Primary (serves users): deepseek-v4-pro  — the expensive incumbent.
# Candidate (shadow):     deepseek-flash   — the cheaper model we want to swap in.
# 60 distinct dev-support questions (system prompt: <=120 words, temperature 0,
# max_tokens 400), shadow rate 0.75 -> 47 sticky-sampled pairs.
# Judged twice: by v4-pro (the primary's family -> may self-prefer) and by flash
# (the candidate -> may self-prefer); agreement between them bounds that bias.
# Spend: ~$0.10 total at Oct 2026 DeepSeek peak prices.
set -euo pipefail
cd "$(dirname "$0")/../.."
EX=examples/02-shadow-router
OUT=$EX/out/real
mkdir -p "$OUT"
set -a; source ~/TradingAgents/.env; set +a   # provides DEEPSEEK_API_KEY (never printed)

uv run --no-sync evalkit shadow-router simulate "$EX/real_requests.jsonl" \
  --primary deepseek:deepseek-v4-pro --candidate deepseek:deepseek-flash \
  --rate 0.75 --timeout 60 --log "$OUT/pairs.jsonl"

uv run --no-sync evalkit shadow-router report "$OUT/pairs.jsonl" \
  --judge deepseek:deepseek-v4-pro --out "$OUT/report_judge_pro"

uv run --no-sync evalkit shadow-router report "$OUT/pairs.jsonl" \
  --judge deepseek:deepseek-flash --out "$OUT/report_judge_flash"
