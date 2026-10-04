#!/usr/bin/env bash
# End-to-end DPO flywheel demo: ingest → build pairs → nightly (dry backend) → day-2 gate.
set -euo pipefail
cd "$(dirname "$0")"
OUT=out
rm -rf "$OUT" && mkdir -p "$OUT"

echo "== 1. ingest day-1 feedback (2 invalid rows, 1 duplicate expected)"
uv run --no-sync evalkit dpo-flywheel ingest feedback_day1.jsonl --store "$OUT/feedback.jsonl"

echo "== 2. build preference pairs (mock teacher, heuristic judge, golden decontamination)"
uv run --no-sync evalkit dpo-flywheel build-pairs --store "$OUT/feedback.jsonl" \
  --golden golden.jsonl --out "$OUT/pairs"
head -n 2 "$OUT/pairs/train.jsonl"

echo "== 3. nightly run #1 (dry-run backend, gate at 10 new pairs)"
uv run --no-sync evalkit dpo-flywheel nightly --store "$OUT/feedback.jsonl" --root "$OUT/flywheel" \
  --golden golden.jsonl --min-new-pairs 10 --date 2026-10-03 --dry-run

echo "== 4. day 2: only a few new thumbs-downs arrive -> nightly should skip"
uv run --no-sync evalkit dpo-flywheel ingest feedback_day2.jsonl --store "$OUT/feedback.jsonl"
uv run --no-sync evalkit dpo-flywheel nightly --store "$OUT/feedback.jsonl" --root "$OUT/flywheel" \
  --golden golden.jsonl --min-new-pairs 10 --date 2026-10-04 --backend dry

echo "== 5. same day, lower threshold -> trains, gate decides promotion vs the incumbent"
uv run --no-sync evalkit dpo-flywheel nightly --store "$OUT/feedback.jsonl" --root "$OUT/flywheel" \
  --golden golden.jsonl --min-new-pairs 3 --date 2026-10-04 --backend dry

ls -R "$OUT/flywheel" | head -n 20
