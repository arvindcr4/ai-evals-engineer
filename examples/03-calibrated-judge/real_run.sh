#!/usr/bin/env bash
# 03 Calibrated LLM-as-a-Judge — real-model run: DeepSeek V4.1 Flash (thinking off) as the judge
# on the same 500 synthetic anchors as run.sh, both orders + pointwise scores (~2,000 calls,
# ~$0.25). The key is loaded from ~/TradingAgents/.env and never printed.
#
# Judge family: DeepSeek is none of the synthetic anchor families (atlas/nova/orion are just
# stylistic openers), so the main audit/calibration use --judge-family none (no self-preference
# term). The family audits in step 3 re-use the cached verdicts (no API calls) as a *style
# preference probe*: does the judge favour responses that open with a given family's style?
set -euo pipefail
cd "$(dirname "$0")"
set -a; source ~/TradingAgents/.env; set +a
OUT=${OUT:-out/real}
JUDGE=${JUDGE:-deepseek:deepseek-flash}
mkdir -p "$OUT"
[ -f anchors.jsonl ] || uv run --no-sync evalkit calibrated-judge make-anchors --n 500 --seed 0 --out anchors.jsonl

echo "== 1. Audit $JUDGE on 500 anchors (both orders + pointwise)"
uv run --no-sync evalkit calibrated-judge audit --anchors anchors.jsonl --llm "$JUDGE" \
  --judge-family none --pointwise --workers 8 \
  --judgments "$OUT/judgments.jsonl" ${REJUDGE:+--rejudge} \
  --out-md "$OUT/audit.md" --out-json "$OUT/audit.json"

echo; echo "== 2. Calibrate: fit on 60%, report held-out 40% (cached verdicts, no API calls)"
uv run --no-sync evalkit calibrated-judge calibrate --anchors anchors.jsonl --llm "$JUDGE" \
  --judge-family none --judgments "$OUT/judgments.jsonl" --out "$OUT/calibration.json" \
  --out-md "$OUT/calibration.md" --out-json "$OUT/calibration_report.json"

echo; echo "== 3. Style-preference probe per anchor family (cached verdicts)"
for fam in atlas nova orion; do
  uv run --no-sync evalkit calibrated-judge audit --anchors anchors.jsonl --llm "$JUDGE" \
    --judge-family "$fam" --judgments "$OUT/judgments.jsonl" \
    --out-json "$OUT/audit_style_$fam.json" --out-md "$OUT/audit_style_$fam.md" >/dev/null
  uv run --no-sync python -c "import json,sys; s=json.load(open(sys.argv[1]))['raw']['self_preference']; print(f\"{sys.argv[2]:6s} judge own-style win {s['judge_own_win_rate']:.1%} vs human {s['human_own_win_rate']:.1%} gap {s['gap']*100:+.1f} pts (n={s['n']})\")" "$OUT/audit_style_$fam.json" "$fam"
done
