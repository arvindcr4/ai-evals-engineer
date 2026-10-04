#!/usr/bin/env bash
# 07 Statistical Significance Engine — end-to-end demo (offline, deterministic).
set -euo pipefail
cd "$(dirname "$0")"

echo "== 1. Is ft-v2 really better than gpt-base? (500 items, 50 prompt templates as clusters)"
uv run --no-sync evalkit significance compare --results results.jsonl --a gpt-base --b ft-v2

echo; echo "== 2. Same question on a 50-item smoke set (a 2-point delta here is noise)"
uv run --no-sync evalkit significance compare --results small_results.jsonl --a gpt-base --b ft-v2

echo; echo "== 3. All candidates vs gpt-base, Holm-corrected for 3 comparisons"
uv run --no-sync evalkit significance compare --results results.jsonl --a gpt-base

echo; echo "== 4. Leaderboard with significance tiers"
uv run --no-sync evalkit significance leaderboard --results results.jsonl

echo; echo "== 5. How many items do we need to see +2 pts on an 80% baseline?"
uv run --no-sync evalkit significance power --baseline 0.8 --delta 0.02
uv run --no-sync evalkit significance power --baseline 0.8 --delta 0.02 --paired --discordance 0.08 --n 500
