#!/usr/bin/env bash
# End-to-end demo: perturb a clean QA set, run a naive and a grounded RAG, report.
set -euo pipefail
cd "$(dirname "$0")"
OUT=out
mkdir -p "$OUT"

uv run --no-sync evalkit rag-adversarial perturb \
  --dataset qa.jsonl --out "$OUT/cases.jsonl" --distractors 3 --seed 0

# Offline baselines: the confident liar vs the grounded abstainer.
uv run --no-sync evalkit rag-adversarial run --cases "$OUT/cases.jsonl" \
  --system naive --out "$OUT/results-naive.jsonl"
uv run --no-sync evalkit rag-adversarial run --cases "$OUT/cases.jsonl" \
  --system grounded --out "$OUT/results-grounded.jsonl"

# LLM adapter path (citation-required prompt). Offline mock by default;
# set RAG_LLM=openai:gpt-4o-mini (or any get_llm spec) to test a real model.
uv run --no-sync evalkit rag-adversarial run --cases "$OUT/cases.jsonl" \
  --system llm --llm "${RAG_LLM:-mock}" --out "$OUT/results-llm.jsonl"

uv run --no-sync evalkit rag-adversarial report \
  "$OUT/results-naive.jsonl" "$OUT/results-grounded.jsonl" "$OUT/results-llm.jsonl" \
  --md "$OUT/report.md" --json "$OUT/report.json"

# CI-style gate: the grounded system must keep its lie rate under 5%.
uv run --no-sync evalkit rag-adversarial report "$OUT/results-grounded.jsonl" \
  --max-lie-rate 0.05 > /dev/null && echo "gate: grounded lie rate OK"
