#!/usr/bin/env bash
# Dataset contamination checker demo: index the eval set, scan a training corpus
# with planted leaks, adjudicate suspicious items, then write decontaminated corpora.
set -euo pipefail
cd "$(dirname "$0")/../.."
EX=examples/14-contamination-checker
OUT=$EX/out
mkdir -p "$OUT"

echo "== 1. index the golden eval set (13-gram + MinHash + hashed embeddings)"
uv run --no-sync evalkit contamination index --eval "$EX/eval.jsonl" --out "$OUT/index.json"

echo; echo "== 2. scan the fine-tuning corpus (JSONL text + chat messages)"
uv run --no-sync evalkit contamination scan --index "$OUT/index.json" \
  --train "$EX/train.jsonl" --out "$OUT/scan" --clean-eval "$OUT/eval.clean.jsonl"

echo; echo "== 3. n-gram only (the classic GPT-3 check) - what it misses"
uv run --no-sync evalkit contamination scan --index "$OUT/index.json" --methods ngram \
  --train "$EX/train.jsonl" --out "$OUT/scan-ngram-only"

echo; echo "== 4. adjudicate suspicious items with an LLM judge (offline mock by default)"
uv run --no-sync evalkit contamination scan --index "$OUT/index.json" \
  --train "$EX/train.jsonl" --out "$OUT/scan-judged" --llm "${EVALKIT_LLM:-mock}"

echo; echo "== 5. decontaminate: drop contaminated docs, and a flagged copy at suspicious level"
uv run --no-sync evalkit contamination decontaminate --index "$OUT/index.json" \
  --train "$EX/train.jsonl" --out "$OUT/train.decontaminated.jsonl" --mode drop
uv run --no-sync evalkit contamination decontaminate --index "$OUT/index.json" \
  --train "$EX/train.jsonl" --out "$OUT/train.flagged.jsonl" --mode flag --level suspicious

echo; echo "== 6. CI gate: non-zero exit when anything is contaminated"
if uv run --no-sync evalkit contamination scan --index "$OUT/index.json" \
     --train "$EX/train.jsonl" --out "$OUT/scan" --fail-on contaminated > /dev/null; then
  echo "gate passed"
else
  echo "gate FAILED as expected (exit 1)"
fi
echo; echo "Report: $OUT/scan/report.md"
