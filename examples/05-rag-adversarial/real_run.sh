#!/usr/bin/env bash
# Real-model run: DeepSeek as the RAG answerer (citation-required prompt) vs the
# offline naive / grounded baselines, on the demo set and the harder 30-item set.
# Needs DEEPSEEK_API_KEY (loaded from ~/TradingAgents/.env; never printed).
set -euo pipefail
cd "$(dirname "$0")"
set -a; source ~/TradingAgents/.env; set +a
OUT=out/real
mkdir -p "$OUT" real_output
RUN="uv run --no-sync evalkit rag-adversarial"

# 1) Demo set (10 items -> 80 cases), flash x3 repeats.
$RUN perturb --dataset qa.jsonl --out "$OUT/cases.jsonl" --distractors 3 --seed 0
for s in naive grounded; do
  $RUN run --cases "$OUT/cases.jsonl" --system "$s" --out "$OUT/demo-$s.jsonl"
done
$RUN run --cases "$OUT/cases.jsonl" --system llm --llm deepseek:deepseek-flash \
  --workers 8 --repeats 3 --out "$OUT/demo-flash.jsonl"
$RUN report "$OUT"/demo-{naive,grounded,flash}.jsonl \
  --md real_output/report-demo.md --json real_output/report-demo.json > /dev/null

# 2) Hard set (30 items with same-type confounders and near-miss entities in context).
uv run --no-sync python make_qa_hard.py
$RUN perturb --dataset qa_hard.jsonl --out "$OUT/cases-hard.jsonl" --distractors 4 --seed 0
for s in naive grounded; do
  $RUN run --cases "$OUT/cases-hard.jsonl" --system "$s" --out "$OUT/hard-$s.jsonl"
done
$RUN run --cases "$OUT/cases-hard.jsonl" --system llm --llm deepseek:deepseek-flash \
  --workers 8 --repeats 3 --out "$OUT/hard-flash.jsonl"
$RUN run --cases "$OUT/cases-hard.jsonl" --system llm --llm deepseek:deepseek-v4-pro \
  --workers 8 --out "$OUT/hard-pro.jsonl"
$RUN report "$OUT"/hard-{naive,grounded,flash,pro}.jsonl \
  --md real_output/report-hard.md --json real_output/report-hard.json > /dev/null

# Small, key-free sample of the model's non-correct replies for the write-up.
uv run --no-sync python - <<'PY'
import json
from pathlib import Path
rows = []
for name in ("hard-flash", "hard-pro", "demo-flash"):
    for line in Path(f"out/real/{name}.jsonl").read_text().splitlines():
        r = json.loads(line)
        if not r["correct"]:
            rows.append({k: r[k] for k in ("system", "case_id", "expected", "response", "lie",
                                           "malformed", "abstained")} | {"repeat": r["meta"].get("repeat", 0)})
Path("real_output/failures.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
print(f"{len(rows)} non-correct replies -> real_output/failures.jsonl")
PY
echo "reports: real_output/report-demo.md real_output/report-hard.md"
