#!/usr/bin/env bash
# Real-model DPO flywheel run against DeepSeek: teacher = deepseek-v4-pro (stronger,
# writes the "chosen" answers), judge = deepseek-flash (cheap rubric scorer, T=0).
# Needs DEEPSEEK_API_KEY in ~/TradingAgents/.env. Writes to out/real/.
set -euo pipefail
cd "$(dirname "$0")"
set -a; source ~/TradingAgents/.env; set +a

TEACHER=${TEACHER:-deepseek:deepseek-v4-pro}
JUDGE=${JUDGE:-deepseek:deepseek-flash}
OUT=out/real
rm -rf "$OUT" && mkdir -p "$OUT"
COMMON=(--golden golden.jsonl --teacher "$TEACHER" --judge "$JUDGE" --workers 6)

echo "== 1. ingest day-1 feedback"
uv run --no-sync evalkit dpo-flywheel ingest feedback_day1.jsonl --store "$OUT/feedback.jsonl"

echo "== 2. build-pairs with the real teacher + judge (cache: $OUT/llm_cache.jsonl)"
uv run --no-sync evalkit dpo-flywheel build-pairs --store "$OUT/feedback.jsonl" "${COMMON[@]}" \
  --cache "$OUT/llm_cache.jsonl" --out "$OUT/pairs" | tee "$OUT/build_pairs.json"

echo "== 2b. same build again: must be served from the cache (0 API calls, same sha)"
uv run --no-sync evalkit dpo-flywheel build-pairs --store "$OUT/feedback.jsonl" "${COMMON[@]}" \
  --cache "$OUT/llm_cache.jsonl" --out "$OUT/pairs_rerun" | tee "$OUT/build_pairs_rerun.json"

echo "== 3. nightly run #1 (dry-run backend, gate at 10 new pairs; reuses the same cache)"
mkdir -p "$OUT/flywheel" && cp "$OUT/llm_cache.jsonl" "$OUT/flywheel/llm_cache.jsonl"
uv run --no-sync evalkit dpo-flywheel nightly --store "$OUT/feedback.jsonl" --root "$OUT/flywheel" \
  "${COMMON[@]}" --min-new-pairs 10 --date 2026-10-03 --dry-run | tee "$OUT/nightly_1.json"

echo "== 4. day 2 feedback -> nightly (only the new thumbs-downs hit the API)"
uv run --no-sync evalkit dpo-flywheel ingest feedback_day2.jsonl --store "$OUT/feedback.jsonl"
uv run --no-sync evalkit dpo-flywheel nightly --store "$OUT/feedback.jsonl" --root "$OUT/flywheel" \
  "${COMMON[@]}" --min-new-pairs 10 --date 2026-10-04 --dry-run | tee "$OUT/nightly_2.json"

echo "== 5. same day, threshold 2 -> trains, gate decides (real teacher yields fewer new pairs than the mock: near-dups)"
uv run --no-sync evalkit dpo-flywheel nightly --store "$OUT/feedback.jsonl" --root "$OUT/flywheel" \
  "${COMMON[@]}" --min-new-pairs 2 --date 2026-10-04 --dry-run | tee "$OUT/nightly_3.json"

# Refresh the committed summaries (no key, no cache) only when asked: SAVE=1 bash real_run.sh
# Note: run.sh does `rm -rf out`, which also deletes out/real and its completion cache.
if [[ "${SAVE:-0}" == 1 ]]; then
  mkdir -p real_output
  cp "$OUT"/build_pairs.json "$OUT"/build_pairs_rerun.json "$OUT"/nightly_[123].json real_output/
  cp "$OUT/pairs/manifest.json" "$OUT/pairs/pairs.jsonl" real_output/
  cp "$OUT/flywheel/adapters/2026-10-04/plan.json" real_output/plan_2026-10-04.json
  echo "saved summaries to real_output/ (capture the log with: SAVE=1 bash real_run.sh 2>&1 | tee real_output/real_run.log)"
fi
