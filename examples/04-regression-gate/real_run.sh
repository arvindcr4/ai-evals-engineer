#!/usr/bin/env bash
# 04 CI/CD Regression Gate against the REAL DeepSeek API.
# The system under test is a triage prompt to deepseek-flash (thinking off, temperature 0).
#   1. baseline   : llm_suite.yaml on "main"
#   2. same prompt: re-run of llm_suite.yaml                 -> expect PASS (exit 0)
#   3. bad prompt : llm_suite_degraded.yaml ("simplified")   -> expect FAIL on task success
#   4. model swap : same prompt, --llm deepseek-flash+think  -> latency/cost visible in the report
# ~320 API calls; the recorded run cost $0.036 at peak prices (real_output/summary.json). Needs DEEPSEEK_API_KEY in ~/TradingAgents/.env.
set -uo pipefail
cd "$(dirname "$0")"
set -a; source ~/TradingAgents/.env; set +a
OUT=${OUT:-out/real}
mkdir -p "$OUT"
RG="uv run --no-sync evalkit regression-gate"

echo "================ 1. baseline (main prompt)"
$RG baseline --suite llm_suite.yaml --out "$OUT/baseline.json"

gate() {  # gate <label> <suite> [extra args]
  local label=$1 suite=$2; shift 2
  echo; echo "================ $label"
  $RG run --suite "$suite" --baseline "$OUT/baseline.json" \
    --out "$OUT/results-$label.json" --report "$OUT/report-$label.md" "$@" > /dev/null
  echo "-> exit code $?"
}
gate same-prompt  llm_suite.yaml
gate bad-prompt   llm_suite_degraded.yaml
gate think        llm_suite.yaml --llm deepseek:deepseek-flash+think
