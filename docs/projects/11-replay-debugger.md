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
