# 13 — Context Window Eviction Tester

> Context overflow is the silent killer of long-running agentic workflows.

## What and why

A harness that floods an agent's working memory with noise and then checks
whether the memory strategy kept the facts that matter. It measures three
things:

- **Recall**: answers with the *current* value of a fact.
- **Stale rate**: answers with a value that a later update superseded. This is
  worse than "unknown", because the agent is confidently wrong.
- **Budget compliance**: the share of checkpoints at which the rendered context
  fits the token budget.

Each metric is swept over noise levels, so you get a degradation curve per
strategy instead of a single anecdote.

## Design

```
generate(noise, seed) ─▶ Scenario(system prompt, turns, probes)
                             │ stream turns
                             ▼
               Memory strategy (budget) ── context(question) ──▶ reader LLM ──▶ correct / stale / miss
```

**Scenarios** (`scenario.py`). Each scenario is fully determined by its seed.
Six *slots* (home address, manager, wifi password, …) get an initial value
planted at depths spread evenly across `fact_span` (0–60% of the stream by
default). Three slots get an **update** ("Correction, my car is now a white
Golf.") spread across `update_span` (60–95%). Two facts live in the **pinned
system prompt** (name, timezone), and `--pinned-updates` of those are later
changed in conversation. That catches strategies that pin the system prompt
but evict the correction. Noise turns are topic chit-chat. A `--hard-noise`
fraction of user turns are distractors: either slot vocabulary without a fact
("remind me to sort the paperwork about the home address") or **colleague facts**
that share the fact template but not the subject ("my colleague's manager is
Luis Romero"). Scenarios round-trip to JSON (`export`, `inspect --scenario`).

**Strategies** (`memory.py`). Every strategy renders the context the model
would actually see. Tokens are counted as whitespace tokens, which is
deterministic and model-agnostic.

| name | behaviour |
|---|---|
| `full` | keep everything (unbounded baseline; shows what the budget costs) |
| `fifo` | truncate oldest first, **system prompt included** |
| `window` | pinned system prompt + newest turns that fit |
| `summary` | pinned system prompt + rolling LLM summary of evicted turns (half the window is summarized on each overflow) + recent turns; a summary over its cap (40% of budget) is hard-truncated to its newest words |
| `retrieval` | pinned system prompt + recent turns (30% of budget) + top-k older turns by cosine over a signed feature-hashed bag-of-words+bigram embedding (numpy), restored to chronological order |

**Reader and scoring** (`harness.py`). Each probe ("What is the user's
manager?") is sent with the rendered context through the `LLM` interface.
Offline, the reader is a `MockLLM` whose responder is a deterministic extractive
reader: it returns the latest value stated for exactly that attribute, or
`unknown`. The offline summarizer is an extractive `MockLLM` that keeps fact
sentences and lets the latest value win. Offline scores therefore measure
exactly what the memory retained, not reader quality. With `--llm <spec>`, one
real model plays both reader and summarizer, using the same prompts. Answers are
classified by normalized containment: current value → correct, superseded value
→ stale, otherwise miss.

`sweep` is **paired**: every strategy sees the same scenarios at each noise
level, so differences between strategies are not sampling noise. It reports
recall, stale and miss rates, recall on pinned and on updated facts, budget
compliance, peak context tokens and summarizer calls per run.

## How to run

```bash
uv run --no-sync evalkit context-eviction run --strategies fifo,window,summary,retrieval \
    --noise 0,50,200,800 [--budget 400] [--trials 3] [--seed 0] [--hard-noise 0.1] [--json]
uv run --no-sync evalkit context-eviction inspect --strategy window --noise 800 [--show-context]
uv run --no-sync evalkit context-eviction export --noise 120 --seed 7 --out scenario.json
uv run --no-sync evalkit context-eviction run --llm openai:gpt-4o-mini --noise 0,200 --trials 1
examples/13-context-eviction/run.sh
```

## Sample output (from `run.sh`)

```
budget = 400 tokens; recall/stale/miss over all probes; pinned = system-prompt facts, upd = updated facts
strategy   noise recall  stale   miss pinned    upd  budget   peak    llm
-------------------------------------------------------------------------
fifo          50   0.38   0.00   0.62   0.50   0.75   100%    400    0.0
fifo         800   0.00   0.00   1.00   0.00   0.00   100%    400    0.0
full         200   1.00   0.00   0.00   1.00   1.00     8%   4666    0.0
full         800   1.00   0.00   0.00   1.00   1.00     2%  18512    0.0
retrieval    800   0.96   0.00   0.04   1.00   1.00   100%    400    0.0
summary      800   1.00   0.00   0.00   1.00   1.00   100%    400  103.3
window        50   0.50   0.00   0.50   1.00   0.75   100%    400    0.0
window       800   0.12   0.12   0.75   0.50   0.00   100%    400    0.0

recall curve over noise levels:
  full       |████|  0:1.00 50:1.00 200:1.00 800:1.00
  fifo       |█▃▁ |  0:1.00 50:0.38 200:0.12 800:0.00
  window     |█▄▂▁|  0:1.00 50:0.50 200:0.25 800:0.12
  summary    |████|  0:1.00 50:1.00 200:1.00 800:1.00
  retrieval  |████|  0:1.00 50:1.00 200:1.00 800:0.96

== tight budget (120 tokens), 8 facts, 30% hard noise ==
retrieval    800   0.70   0.00   0.30   1.00   0.83   100%    120    0.0
summary      800   0.80   0.00   0.20   1.00   1.00   100%    120  603.3
window       800   0.10   0.10   0.80   0.50   0.00   100%    120    0.0
```

(Rows abridged; `run.sh` prints the full 5×4 grid.) What the results show:

- `fifo` loses the system prompt first: pinned recall falls to 0 at 800 noise.
- `window` keeps the pinned prompt but goes **stale**. The timezone correction
  scrolls out of the window, the pinned old value stays, and the reader answers
  confidently with the old timezone.
- `full` is perfect only because it ignores the budget (2% compliance).
- `summary` holds 100% recall at a 400-token budget for about 100 summarizer
  calls per 800-turn run. At 120 tokens its capped summary starts dropping facts.
- `retrieval` costs no LLM calls. It degrades under heavy, vocabulary-matched
  distractor noise at small budgets (0.70 recall).

## Limitations

- Offline results use an *ideal* extractive reader and summarizer, so they
  bound what each strategy *retains*. Real-model runs add reader and summarizer
  errors: lossy or hallucinated summaries, and ignoring updates. That is where
  stale rates get interesting for `summary`.
- Facts follow a small set of templates, and the extractive components depend on
  them. Paraphrased facts ("we moved to …") need `--llm` with a real reader.
- Token counting is by whitespace and does not match any tokenizer. The budget
  is relative, not a model's real context limit.
- Hashed bag-of-words retrieval is a stand-in for dense embeddings. Plug any
  `embedder(text) -> unit vector` into `RetrievalMemory`.
- Probes are single-hop lookups. Multi-hop state (for example, "which of my
  two managers approved X") is not modeled.
