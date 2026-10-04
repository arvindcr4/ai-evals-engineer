#!/usr/bin/env bash
# Cost-Quality Pareto demo: 4 tenants x 7 configs x 60 tasks of eval results,
# frontier + System-One -> System-Two router curve, then a price-drop what-if.
set -euo pipefail
cd "$(dirname "$0")/../.."
EX=examples/10-pareto-dashboard

# results.jsonl is committed; regenerate it with:
#   uv run --no-sync evalkit pareto generate --out "$EX/results.jsonl" --tasks 60 --seed 11

uv run --no-sync evalkit pareto build "$EX/results.jsonl" \
  --router s1-mini+rag,s2-large+reasoning --out "$EX/out"

# What if the frontier model's price drops 75%?
uv run --no-sync evalkit pareto whatif "$EX/results.jsonl" \
  --config s2-frontier --factor 0.25 \
  --router s1-mini+rag,s2-large+reasoning --out "$EX/out"
