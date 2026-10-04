#!/usr/bin/env bash
# 04 CI/CD Regression Gate — runs the toy triage suite against the committed
# baseline for four builds of the system under test and shows the gate's verdict.
set -uo pipefail
cd "$(dirname "$0")"
OUT=${OUT:-out}
mkdir -p "$OUT"

gate() {  # gate <SUT_VERSION> <label>
  echo; echo "================ SUT_VERSION=$1 ($2)"
  SUT_VERSION=$1 uv run --no-sync evalkit regression-gate run \
    --suite suite.yaml --baseline baseline.json \
    --out "$OUT/results-v$1.json" --report "$OUT/report-v$1.md"
  echo "-> exit code $?"
}

# To refresh the baseline on main:
#   SUT_VERSION=1 uv run --no-sync evalkit regression-gate baseline --suite suite.yaml --out baseline.json
gate 1   "unchanged main build"
gate 1.1 "harmless refactor that fixes an item"
gate 2   "bad refactor: refund routing + cents dropped"
gate 3   "same answers, 3x slower"

echo; echo "================ compare two saved runs directly (v1.1 as the new baseline, v2 as candidate)"
uv run --no-sync evalkit regression-gate compare --suite suite.yaml \
  --baseline "$OUT/results-v1.1.json" --candidate "$OUT/results-v2.json" > /dev/null
echo "-> exit code $?"
