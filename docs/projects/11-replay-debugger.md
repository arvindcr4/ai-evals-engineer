# 11 — Counterfactual Replay Debugger

> Record every hop, swap one node, replay the rest, and see which step caused the hallucination.

## What and why

When an agent returns a wrong answer after seven hops, the cause could be the
model's third decision, a stale tool result at hop four, or the final
formatting step. Re-running the whole agent with a change confounds that
question: sampling noise, other tools and other hops all move together. This
system records a run as a **cassette** of nodes (every LLM hop and every tool
call, with full inputs and outputs). It then replays the run
**counterfactually**: the prefix is served bit-for-bit from the cassette, exactly
one node is substituted, and the rest of the run continues. `bisect` automates
this by swapping each node in turn for a known-good alternative and reporting
which *single* swap turns FAIL into PASS.

## Design

```
record:  agent loop ──Runtime──▶ live LLM / tools ──▶ Cassette (= core Trajectory JSONL)
replay:  agent loop ──Runtime──▶ [0..k-1] cassette │ [k] Override │ [k+1..] cached-or-live
bisect:  for k in nodes: Override(k, known-good alternative) → replay → checker(answer)?
```

- **Agent** (`agent.py`). A minimal ReAct loop with a two-verb protocol,
  `CALL <tool> {json}` and `FINAL <answer>`. Every side effect goes through a
  `Runtime` (`llm(messages)` / `tool(name, args)`), which is what makes recording
  and replay transparent to the agent. Tool exceptions are returned to the agent
  as `ERROR:` observations instead of crashing the run.
- **Cassette** (`cassette.py`). A `Node` stores `index`, `kind` (`llm`/`tool`),
  name, full input, output, source (`live`/`cassette`/`override`), and tokens and
  cost. `Node.key` is a content hash of the input. Cassettes serialize as core
  `Trajectory` JSONL (`llm`, `tool_call` and `final` steps), so the trajectory
  grader (#01) and other systems can consume them directly.
- **Replay** (`replay.py`). `_ReplayRuntime` serves nodes `< k` from the cassette
  and checks that the agent asks for the *same* input at each one. If the agent
  code or prompt changed, the inputs differ and `ReplayDivergence` is raised
  rather than replaying something meaningless. Node `k` comes from an `Override`:
  a literal output, a different model (`--with-llm`), or a different tool
  implementation (`--with-tools`). Nodes after `k` run in one of two modes:
  - `cached` (default): serve a recorded output whenever the node's input is
    byte-identical to one in the cassette, otherwise go live. Untouched branches
    cost nothing, and only the consequences of the swap are recomputed.
  - `live`: re-execute everything after the swap.
- **Bisect**. For each node, the known-good alternative comes from a reference
  model (`--ref-llm`) regenerating the hop on the *exact recorded messages*, or
  from oracle tools (`--oracle-tools`). Alternatives identical to the recording
  are skipped (`identical`), because that node cannot be the cause. Each
  remaining node is replayed and checked. Results are `flipped`,
  `still_failing`, `identical`, `no_alternative` or `error`. The root cause is
  the earliest flipped node. If no single swap repairs the run, bisect says so:
  there are interacting faults, or no oracle covers the faulty node.

**Toy world** for offline determinism: tools `lookup_price`, `fx_rate`, `calc`
(an AST-restricted arithmetic evaluator), and two scripted policies served
through `MockLLM(responder=...)`. `toy` is careful; `toy:sloppy` skips the FX
lookup and hallucinates parity (rate 1.0), but uses an FX observation if one is
in its context, the way a real model would. The faulty toolset `toy:stale_fx`
serves an inverted rate. Any `get_llm` spec (`openai:…`, `deepseek:…`,
`<base_url>|<model>`) can be the agent, the swap model or the reference model.
The system prompt explains the protocol to a real model.

## How to run

```bash
E="uv run --no-sync evalkit replay-debugger"
$E record --tasks examples/11-replay-debugger/tasks.jsonl --llm toy:sloppy --tools toy --out out.jsonl
$E show   --cassettes out.jsonl --task-id widgets-eur
$E replay --cassettes out.jsonl --task-id widgets-eur --llm toy:sloppy --at 2 \
          --output 'CALL fx_rate {"base": "USD", "quote": "EUR"}'      # or --with-llm toy / --with-tools toy
$E bisect --cassettes out.jsonl --llm toy:sloppy --tools toy --ref-llm toy --oracle-tools toy
examples/11-replay-debugger/run.sh
```

## Sample output (from `run.sh`)

**A: hallucinating model, correct tools.** Bisect points at the LLM hop:

```
widgets-eur         5 nodes  answer=12.75      expected=11.73      FAIL
...
widgets-eur: baseline FAIL; swapping each node with its known-good alternative
   node  0 llm  toy-sloppy   identical      alt='CALL lookup_price {"item": "widget"}'
   node  1 tool lookup_price identical      alt='4.25'
   node  2 llm  toy-sloppy   flipped        alt='CALL fx_rate {"base": "USD", "quote": "EUR"}' -> answer 11.73
   node  3 tool calc         identical      alt='12.75'
   node  4 llm  toy-sloppy   still_failing  alt='CALL fx_rate {"base": "USD", "quote": "EUR"}' -> answer 12.75
   ROOT CAUSE: node 2 (llm toy-sloppy)

replay with node 2 swapped, after=cached (PASS, 4 live calls):
    0 llm  cassette toy-sloppy   CALL lookup_price {"item": "widget"}
    1 tool cassette lookup_price lookup_price({"item": "widget"}) -> 4.25
>>  2 llm  override toy-sloppy   CALL fx_rate {"base": "USD", "quote": "EUR"}
    3 tool live     fx_rate      fx_rate({"base": "USD", "quote": "EUR"}) -> 0.92
    4 llm  live     toy-sloppy   CALL calc {"expr": "3 * 4.25 * 0.92"}
    5 tool live     calc         calc({"expr": "3 * 4.25 * 0.92"}) -> 11.73
    6 llm  live     toy-sloppy   FINAL 11.73
```

Node 4 is `still_failing`: fixing the final hop alone does not help. That is why
bisecting single nodes beats "swap the last thing that looked wrong".

**B: careful model, stale tool.** Every LLM hop is `identical` to the reference
model, and the tool node flips:

```
widgets-eur: baseline FAIL; swapping each node with its known-good alternative
   node  2 llm  toy-careful  identical      alt='CALL fx_rate {"base": "USD", "quote": "EUR"}'
   node  3 tool fx_rate      flipped        alt='0.92' -> answer 11.73
   node  4 llm  toy-careful  identical      alt='CALL calc {"expr": "3 * 4.25 * 1.087"}'
   ROOT CAUSE: node 3 (tool fx_rate)
...
root causes isolated: 3
```

The run that passes (`sprockets-usd`, where the currency is USD) is skipped by
bisect automatically.

## Limitations

- Single-node bisection. When two faults interact (sloppy model *and* stale
  tool), no single swap repairs the run, and the debugger reports exactly that.
  A pairwise or delta-debugging search would be the next step.
- The known-good alternative is only as good as the reference model or oracle
  tools. Without an oracle for a node, that node is never blamed
  (`no_alternative`).
- `cached` mode matches on exact input hashes. A reference model that phrases
  the same action differently forces everything after it to run live.
- The checker is numeric-tolerance or substring match. Plug in an LLM judge
  (`bisect(checker=...)`) for free-form answers.
- The bundled agent is a single-thread ReAct loop. Agents built on other
  frameworks need a thin adapter that routes their model and tool calls through
  a `Runtime`.

## Real-model run (DeepSeek, Oct 2026)

```bash
examples/11-replay-debugger/real_run.sh     # sources ~/TradingAgents/.env, writes out/real/
```

- **Agent model:** `deepseek:deepseek-flash` (thinking off, temperature 0) on every hop.
- **Reference model for bisect:** `deepseek:deepseek-v4-pro`. A second bisect uses the agent
  model itself as its own reference, as a nondeterminism control.
- **Tasks:** `examples/11-replay-debugger/tasks_real.jsonl` has 16 tasks (4 items × 4 currencies,
  4 phrasings, quantities 1–25). Each row's `expected` field equals `expected_answer()` for its
  input; the checker passes answers within ±0.011.
- **Fault:** the real model never produced a wrong answer with correct tools, so the failures
  are injected through the `toy:stale_fx` tool, which serves an inverted rate rounded to
  4 dp.
- **Spend:** about **$0.025** in total, including an initial 4-task probe at the provider-default
  temperature (the probe's outputs were not saved, so its share is an estimate). A full
  `real_run.sh` costs about $0.021 (sum of the printed costs): 3 × 16-task records at $0.0036 each,
  the pro-ref bisect at $0.0072, the self-ref bisect at $0.0024 and three replays under $0.001.
- Sample outputs are in `examples/11-replay-debugger/real_output/`.

**Results (n = 16 tasks per record; small sample):**

| run | pass | notes |
|---|---|---|
| record, correct tools | **16/16** | protocol followed on every hop. `widgets-gbp-2` answered 6.72 vs expected 6.71 (exact value 6.715; within the ±0.011 tolerance) |
| re-record, same settings | 16/16 | 15/16 runs byte-identical; 1 of 60 aligned LLM hops differed (`sprockets-usd-25` node 0: `"sprockets"` vs `"sprocket"`). Recomputed from `out/real/clean*.jsonl`; only the summary line is saved in `real_output/1b_reproducibility.txt` |
| record, stale FX tool | **10/16** | 4 USD tasks unaffected. 6 of 12 non-USD tasks still pass because the model *divides* by the inverted rate (10/12 divided; INR still fails on rounding) |
| bisect, ref = v4-pro | 6/6 root causes = node 3 `fx_rate` | every LLM hop `identical` to pro at temperature 0 except one `still_failing` hop with a different spelling |
| bisect, ref = flash itself | 6/6 = node 3 `fx_rate` | 0 `unstable` nodes at temperature 0 |

Excerpt (`real_output/3_bisect_pro.txt`):

```
gadgets-eur-7: baseline FAIL; swapping each node with its known-good alternative
   node  2 llm  deepseek-flash identical      alt='CALL fx_rate {"base": "USD", "quote": "EUR"}'
   node  3 tool fx_rate      flipped        alt='0.92' -> answer 77.28
   node  4 llm  deepseek-flash identical      alt='CALL calc {"expr": "7 * 12.0 * 1.087"}'
   ROOT CAUSE: node 3 (tool fx_rate)
sprockets-inr-20: baseline FAIL; ...
   node  0 llm  deepseek-flash still_failing  alt='CALL lookup_price {"item": "sprocket"}' -> answer 1333.33
   node  3 tool fx_rate      flipped        alt='83.1' -> answer 1329.60
   ROOT CAUSE: node 3 (tool fx_rate)
root causes isolated: 6  (api cost $0.00722)
```

**Comparison with the offline mock.**

- **What carries over.** The mechanics transfer unchanged: prefix served from the cassette,
  one node swapped, consequences re-run, and the stale tool isolated as the earliest
  flipped node in 6/6 failing runs with either reference. Cached mode keeps replays cheap:
  each manual replay made 2–3 live calls, under $0.0004.
- **The model partly defends against the fault.** The scripted `toy` policy always
  multiplies by whatever rate it gets, so the stale tool breaks 3/3 non-USD toy tasks.
  deepseek-flash divided by the stale rate in **10 of 12** non-USD tasks
  (EUR 3/4, GBP 3/4, INR 4/4; see `real_output/2_show_stale.txt`):
  - For EUR and GBP, dividing undoes the inversion, so 6 tasks pass. The two that
    multiplied (`gadgets-eur-7`, `sprockets-gbp-13`) fail, with no obvious phrasing
    pattern.
  - In the probe run (outputs not saved; quoted from the run log) flash said why it divided for INR: *"the quote returned 0.012
    (which appears to be USD per INR)"*.
  - All 4 INR runs still fail only because the stale tool rounded 1/83.1 to 0.012.
    1/0.012 = 83.33, which is ~0.3% high.

  Bisect still blames the tool correctly in every case. The interesting real-model
  signal is *which* tasks the fault reaches, and the mock cannot show that.
- **The stronger model is no better at this hop.** Swapping the calc hop of
  `widgets-eur-3`'s stale run to deepseek-v4-pro (`5c_pro_calc_hop.txt`) made the run
  *worse*. Pro multiplied by 1.087 → 13.86 (FAIL), where flash had divided → 11.73
  (PASS). In the bisect, pro reproduced flash's hop verbatim on every other LLM node.
  On this toy, "known-good reference model" adds little beyond the oracle tool. One
  task on one hop, so this is anecdotal.
- **Hallucinated parity was not observed.** The hallucinating-model failure of scenario A
  (`toy:sloppy` skipping `fx_rate`) never appeared: flash called `fx_rate` on all 12
  non-USD tasks and skipped it correctly on the 4 USD tasks. That scenario stays
  mock-only.

**Integration bugs found and fixed (regression tests in `tests/test_replay_debugger_real.py`):**

1. **Prose before the action.** On the stale INR task flash replied with a reasoning sentence,
   a blank line, then `CALL calc {...}`. `parse_action` only read the first line, so
   it returned an `ERROR`, cost an extra hop, and added a spurious LLM node to the cassette. The
   parser now strips code fences and markdown decoration (`**FINAL**`, backticks), accepts
   `FINAL:`, and takes the *first* protocol line. Prose such as "The FINAL answer is 3" is
   still rejected.
2. **No temperature control.** Record, replay, override and bisect called the API at the
   provider default temperature. In the probe run the "reference" (same model) re-sampled the
   calc hop of `widgets-eur` as `/ 1.087` instead of `* 1.087` and was reported as a
   **second culprit** (probe outputs not saved). That verdict came from sampling noise, not a fix. All live hops now pass
   `temperature=0` (CLI `--temperature`, default 0; `None` keeps the provider default).
3. **Same-model re-samples treated as known-good.** Temperature 0 is not bit-reproducible
   (1/60 hops differed between two records). When `--ref-llm` is the model that recorded the
   hop and its output differs, bisect now reports the node as `unstable`. The node is
   replayed but never blamed. This did not trigger in the final temperature-0 run. The
   guard exists for the probe-run case above.
4. **Cost accounting.** An `--with-llm` override node lost its tokens and cost. Bisect's
   reference calls and the live hops of its replays were not counted anywhere, and no
   command reported spend. Override nodes now keep their cost, `ReplayResult.cost_usd` and
   `BisectReport.cost_usd` exist, and `record`, `replay` and `bisect` print API cost. With
   `toy` models the printed cost is MockLLM's notional price, not real spend.

No core bugs were found. `OpenAICompatibleLLM` passed `temperature` through and its retry logic
was never exercised: there were no 429s at this volume.
