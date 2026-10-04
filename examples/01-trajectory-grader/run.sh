#!/usr/bin/env bash
# 01 Trajectory Grading Engine — grade 8 refund-agent runs against the DAG spec.
set -euo pipefail
cd "$(dirname "$0")"
uv run --no-sync evalkit trajectory-grader check-spec --spec spec.yaml
echo
uv run --no-sync evalkit trajectory-grader grade --spec spec.yaml --trajectories runs.jsonl
echo
uv run --no-sync evalkit trajectory-grader grade --spec spec.yaml --trajectories runs.jsonl --json \
  | uv run --no-sync python -c "import json,sys; d=json.load(sys.stdin); print('JSON summary:', d['summary'])"
# CI usage: non-zero exit when any run fails
if uv run --no-sync evalkit trajectory-grader grade --spec spec.yaml --trajectories runs.jsonl --quiet --strict >/dev/null; then
  echo "strict: all runs passed"
else
  echo "strict: exit 1 (failing runs present) — as expected for this dataset"
fi
