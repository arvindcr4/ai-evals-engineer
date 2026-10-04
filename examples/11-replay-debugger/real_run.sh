#!/usr/bin/env bash
# 11 Counterfactual Replay Debugger — real-model run on DeepSeek (agent = deepseek-flash).
# Needs DEEPSEEK_API_KEY in ~/TradingAgents/.env (never printed). ~$0.02 per full run.
set -euo pipefail
cd "$(dirname "$0")"
set -a; source ~/TradingAgents/.env; set +a
OUT=${OUT:-out/real}; mkdir -p "$OUT"
E() { uv run --no-sync evalkit replay-debugger "$@"; }
AGENT=deepseek:deepseek-flash
REF=deepseek:deepseek-v4-pro
TASKS=tasks_real.jsonl   # 16 tasks: 4 items x 4 currencies, 4 phrasings

echo "== 1. record, correct tools (twice, to measure temperature-0 reproducibility) =="
E record --tasks $TASKS --llm $AGENT --tools toy --out "$OUT/clean.jsonl" | tee "$OUT/1_record_clean.txt"
E record --tasks $TASKS --llm $AGENT --tools toy --out "$OUT/clean2.jsonl" | tail -1
uv run --no-sync python - "$OUT/clean.jsonl" "$OUT/clean2.jsonl" <<'PY' | tee "$OUT/1b_reproducibility.txt"
import sys
from evalkit.replay_debugger import load_cassettes
a, b = (load_cassettes(p) for p in sys.argv[1:3])
same = [x.task_id for x, y in zip(a, b) if [n.output for n in x.nodes] == [n.output for n in y.nodes]]
hops = sum(n.kind == "llm" for c in a for n in c.nodes)
diff_hops = sum(nx.output != ny.output for x, y in zip(a, b)
                for nx, ny in zip(x.nodes, y.nodes) if nx.kind == "llm")
print(f"byte-identical re-record at temperature 0: {len(same)}/{len(a)} runs; "
      f"{diff_hops}/{hops} aligned LLM hops differ")
PY

echo; echo "== 2. record, stale FX tool injected (toy:stale_fx) =="
E record --tasks $TASKS --llm $AGENT --tools toy:stale_fx --out "$OUT/stale.jsonl" | tee "$OUT/2_record_stale.txt"
E show --cassettes "$OUT/stale.jsonl" > "$OUT/2_show_stale.txt"

echo; echo "== 3. bisect, reference = deepseek-v4-pro, oracle tools = toy =="
E bisect --cassettes "$OUT/stale.jsonl" --llm $AGENT --tools toy:stale_fx \
  --ref-llm $REF --oracle-tools toy | tee "$OUT/3_bisect_pro.txt"

echo; echo "== 4. bisect, reference = the agent model itself (nondeterminism control) =="
E bisect --cassettes "$OUT/stale.jsonl" --llm $AGENT --tools toy:stale_fx \
  --ref-llm $AGENT --oracle-tools toy | tee "$OUT/4_bisect_self.txt"

echo; echo "== 5. manual counterfactuals on widgets-eur-3 =="
# 5a: inject the stale rate into a clean, passing run at the fx_rate node
E replay --cassettes "$OUT/clean.jsonl" --task-id widgets-eur-3 --llm $AGENT --tools toy \
  --at 3 --output-json 1.087 | tee "$OUT/5a_inject_stale.txt"
# 5b: repair the stale run by swapping the fx_rate node with the oracle tool, live after
E replay --cassettes "$OUT/stale.jsonl" --task-id widgets-eur-3 --llm $AGENT --tools toy:stale_fx \
  --at 3 --with-tools toy --after live | tee "$OUT/5b_oracle_fx_live.txt"
# 5c: regenerate the calc hop with the stronger model: does pro notice the inverted rate?
E replay --cassettes "$OUT/stale.jsonl" --task-id widgets-eur-3 --llm $AGENT --tools toy:stale_fx \
  --at 4 --with-llm $REF | tee "$OUT/5c_pro_calc_hop.txt"
echo "outputs in $OUT"
