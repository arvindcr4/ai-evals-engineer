#!/usr/bin/env bash
# Shadow Routing Comparator demo: replay 1,200 prod requests through the proxy
# (5% sticky shadow sample to a cheaper candidate), then build the report.
set -euo pipefail
cd "$(dirname "$0")/../.."
EX=examples/02-shadow-router

uv run --no-sync evalkit shadow-router simulate "$EX/requests.jsonl" \
  --primary demo:primary --candidate demo:candidate --rate 0.05 \
  --log "$EX/out/pairs.jsonl"

uv run --no-sync evalkit shadow-router report "$EX/out/pairs.jsonl" \
  --judge mock --out "$EX/out/report"

# Real models instead (OpenAI-compatible):
#   evalkit shadow-router serve --primary openai:gpt-4.1 --candidate deepseek:deepseek-chat --rate 0.05
#   evalkit shadow-router report shadow_pairs.jsonl --judge openai:gpt-4.1-mini
