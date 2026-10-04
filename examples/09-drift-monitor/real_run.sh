#!/usr/bin/env bash
# Real-model run of the drift monitor: DeepSeek V4.1 Flash as the LLM judge
# (thinking off, temperature 0) on 7 healthy + 6 decayed days around the
# decay onset (2026-09-21), 80 sampled records/day. ~2.1k judge calls, ~$0.18.
#
#   A  grounded traffic + real judge   (the headline run)
#   B  grounded traffic + mock judge   (offline comparison on identical samples)
#   C  filler traffic   + real judge   (the original demo data: shows the floor effect)
#
# The API key comes from ~/TradingAgents/.env (DEEPSEEK_API_KEY); it is never printed.
set -euo pipefail
cd "$(dirname "$0")/../.."
EX=examples/09-drift-monitor
OUT=$EX/out/real
rm -rf "$OUT" && mkdir -p "$OUT"
set -a; source ~/TradingAgents/.env; set +a

FROM=2026-09-14; TO=2026-09-26
COMMON=(--from "$FROM" --to "$TO" --rate 0.05 --max-samples 80 --baseline-days 7)
JUDGE=deepseek:deepseek-flash

for style in grounded filler; do
  uv run --no-sync evalkit drift-monitor generate --out "$OUT/logs-$style" \
    --start 2026-09-01 --days 30 --per-day 2000 --decay-day 20 --seed 7 --style "$style"
done

run() {  # name logs judge
  uv run --no-sync evalkit drift-monitor backfill --logs "$OUT/logs-$2" --db "$OUT/$1.sqlite" \
    "${COMMON[@]}" --judge "$3" --workers 8 --sink "jsonl:$OUT/$1.alerts.jsonl" \
    | tee "$OUT/$1.backfill.txt"
  uv run --no-sync evalkit drift-monitor report --db "$OUT/$1.sqlite" --out "$OUT/$1-report" >/dev/null
}

run A-grounded-deepseek grounded "$JUDGE"
run B-grounded-mock     grounded mock
run C-filler-deepseek   filler   "$JUDGE"

# Nightly-job form for the last day (what cron would run), against store A.
uv run --no-sync evalkit drift-monitor run --logs "$OUT/logs-grounded" --db "$OUT/A-grounded-deepseek.sqlite" \
  --date "$TO" --rate 0.05 --max-samples 80 --baseline-days 7 --judge "$JUDGE" --workers 8 \
  --sink stdout | tee "$OUT/A-nightly-$TO.txt"

# Small, commit-worthy copies (no logs, no sqlite, no secrets) -> real_output/.
R=$EX/real_output
mkdir -p "$R"
for n in A-grounded-deepseek B-grounded-mock C-filler-deepseek; do
  cp "$OUT/$n.backfill.txt" "$OUT/$n.alerts.jsonl" "$R/"
  cp "$OUT/$n-report/drift_report.md" "$R/$n.report.md"
done
cp "$OUT/A-grounded-deepseek-report/drift_report.html" "$R/A-grounded-deepseek.report.html"
cp "$OUT/A-nightly-$TO.txt" "$R/"
head -n 3 "$OUT/logs-grounded/2026-09-25.jsonl" > "$R/traffic-grounded.2026-09-25.head.jsonl"
