#!/usr/bin/env bash
# Production Drift Monitor demo: 30 days of synthetic traffic (quality decays
# from day 20), nightly-job backfill at a 5% sample, alerts + trend report.
set -euo pipefail
cd "$(dirname "$0")/../.."
EX=examples/09-drift-monitor
OUT=$EX/out
mkdir -p "$OUT"
# Clear previous demo output but keep out/real/ (written by real_run.sh, costs API money).
find "$OUT" -mindepth 1 -maxdepth 1 ! -name real -exec rm -rf {} +

uv run --no-sync evalkit drift-monitor generate --out "$OUT/logs" \
  --start 2026-09-01 --days 30 --per-day 2000 --decay-day 20 --seed 7

# Days 1-29 as a backfill (alerts to a JSONL file) ...
uv run --no-sync evalkit drift-monitor backfill --logs "$OUT/logs" --db "$OUT/drift.sqlite" \
  --from 2026-09-01 --to 2026-09-29 --rate 0.05 --judge mock --sink "jsonl:$OUT/alerts.jsonl"

# ... then day 30 exactly as the nightly cron/systemd job would run it.
uv run --no-sync evalkit drift-monitor run --logs "$OUT/logs" --db "$OUT/drift.sqlite" \
  --date 2026-09-30 --rate 0.05 --judge mock \
  --sink stdout --sink "jsonl:$OUT/alerts.jsonl" --sink webhook \
  || echo "(run exited non-zero)"

uv run --no-sync evalkit drift-monitor report --db "$OUT/drift.sqlite" --out "$OUT/report"
