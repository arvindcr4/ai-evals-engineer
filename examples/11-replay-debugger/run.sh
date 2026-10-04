#!/usr/bin/env bash
# 11 Counterfactual Replay Debugger — two buggy systems, one culprit each.
set -euo pipefail
cd "$(dirname "$0")"
OUT=${OUT:-out}; mkdir -p "$OUT"
E="uv run --no-sync evalkit replay-debugger"

echo "== A. sloppy model (hallucinates FX parity), correct tools =="
$E record --tasks tasks.jsonl --llm toy:sloppy --tools toy --out "$OUT/sloppy.jsonl"
echo
$E show --cassettes "$OUT/sloppy.jsonl" --task-id widgets-eur
echo
echo "-- bisect: swap each node with the careful model / oracle tools --"
$E bisect --cassettes "$OUT/sloppy.jsonl" --llm toy:sloppy --tools toy \
  --ref-llm toy --oracle-tools toy --task-id widgets-eur
echo
echo "-- manual counterfactual: rewrite node 2 by hand, replay the rest --"
$E replay --cassettes "$OUT/sloppy.jsonl" --task-id widgets-eur --llm toy:sloppy --tools toy \
  --at 2 --output 'CALL fx_rate {"base": "USD", "quote": "EUR"}'
echo
echo "== B. careful model, stale FX tool =="
$E record --tasks tasks.jsonl --llm toy --tools toy:stale_fx --out "$OUT/stale.jsonl"
echo
$E bisect --cassettes "$OUT/stale.jsonl" --llm toy --tools toy:stale_fx \
  --ref-llm toy --oracle-tools toy
echo "cassettes in $OUT"
