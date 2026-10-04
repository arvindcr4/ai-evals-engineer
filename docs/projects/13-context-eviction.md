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
| `summary` | pinned system prompt + rolling LLM summary of evicted turns (half the window is summarized on each overflow) + recent turns; a summary over its cap (40% of budget) is cut by whole sentences: non-fact sentences first, then the oldest facts (see fix 1 below) |
| `retrieval` | pinned system prompt + recent turns (30% of budget) + top-k older turns by cosine over a signed feature-hashed bag-of-words+bigram embedding (numpy), restored to chronological order |

**Reader and scoring** (`harness.py`). Each probe ("What is the user's
manager?") is sent with the rendered context through the `LLM` interface.
Offline, the reader is a `MockLLM` whose responder is a deterministic extractive
reader: it returns the latest value stated for exactly that attribute, or
`unknown`. The offline summarizer is an extractive `MockLLM` that keeps fact
sentences and lets the latest value win. Offline scores therefore measure
exactly what the memory retained, not reader quality. With `--llm <spec>`, one
real model plays both reader and summarizer, using the same prompts at
temperature 0; `--reader` / `--summarizer` override one role (e.g. a real
summarizer with the ideal reader), `--workers N` runs jobs in parallel threads,
and every run prints per-role calls, tokens and USD cost. Answers are
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
examples/13-context-eviction/real_run.sh   # DeepSeek, ~$0.65
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

## Real-model run (DeepSeek, Oct 2026)

**Command** (`examples/13-context-eviction/real_run.sh`; outputs in `out/real/`,
committed copies in `examples/13-context-eviction/real_output/`):

```bash
set -a; source ~/TradingAgents/.env; set +a
E="uv run --no-sync evalkit context-eviction"
$E run --llm deepseek:deepseek-flash --strategies full,fifo,window,summary,retrieval \
    --noise 50,200,800 --trials 3 --seed 0 --workers 8
$E run --llm deepseek:deepseek-flash --strategies window,summary,retrieval --noise 50,800 \
    --budget 120 --facts 8 --hard-noise 0.3 --trials 2 --workers 8   # (window/retrieval also ran with 3 trials)
```

**Models.** `deepseek:deepseek-flash` (DeepSeek V4.1 Flash, thinking off,
temperature 0) as both reader and summarizer. No second model was needed.

**Sample size.** Small. 400-token budget: 5 strategies × 3 noise levels × 3
paired scenarios, 8 probes each, so 24 probes per cell. 120-token budget: 10
probes per scenario; 3 scenarios per cell for window/retrieval and 2 for the
final summary run, so 20–30 probes per cell. One probe is worth 3–5 points of
recall, so read differences under about 10 points as noise.

**API cost.** **$0.92** summed from the `llm usage` lines of the four saved
real-model outputs (`sweep-budget400` $0.4495, `tight-budget120-trials3-precapfix`
$0.2790, `tight-budget120-summary-trials2` $0.1785,
`inspect-summary-noise120-precapfix` $0.0111; `Completion.cost_usd` at peak
prices). The original run reported about $0.94, which also includes an unsaved
debugging run. A later verification re-run of the inspect example
(`inspect-summary-noise120-postfix.txt`) cost another $0.01. The 400-budget sweep cost $0.45: $0.27 of that was the reader
on the `full` strategy's 18k-token contexts, and $0.18 was 533 summarizer calls.
The final 120-budget summary run is almost all summarizer calls: 1,335 calls
for $0.18 over 4 scenario runs (2 noise levels × 2 trials).

### Results

400-token budget, real reader + summarizer (`real_output/sweep-budget400.txt`):

```
strategy   noise recall  stale   miss pinned    upd  budget   peak    llm
fifo          50   0.38   0.00   0.62   0.50   0.75   100%    400    0.0
fifo         800   0.00   0.00   1.00   0.00   0.00   100%    400    0.0
full         800   1.00   0.00   0.00   1.00   1.00     2%  18512    0.0
retrieval    800   0.96   0.00   0.04   1.00   1.00   100%    400    0.0
summary       50   1.00   0.00   0.00   1.00   1.00   100%    399    6.3
summary      200   1.00   0.00   0.00   1.00   1.00   100%    400   30.0
summary      800   0.96   0.00   0.04   1.00   0.92   100%    400  141.3
window       200   0.25   0.00   0.75   1.00   0.25   100%    400    0.0
window       800   0.12   0.12   0.75   0.50   0.00   100%    400    0.0
llm usage: reader deepseek-flash: 360 calls, 893988+809 tok, $0.2692;
           summarizer deepseek-flash: 533 calls, 229810+92843 tok, $0.1804
```

120-token budget, 8 facts, 30% hard noise: real model vs the idealized offline
reader and summarizer on the same paired scenarios:

| strategy | noise | recall, offline ideal | recall, DeepSeek flash | stale (real) | summarizer calls/run (ideal → real) |
|---|---|---|---|---|---|
| window | 50 | 0.20 | 0.20 | 0.00 | – |
| window | 800 | 0.10 | 0.10 | 0.10 | – |
| retrieval | 50 | 1.00 | 1.00 | 0.00 | – |
| retrieval | 800 | 0.70 | 0.70 | 0.00 | – |
| summary | 50 | 0.80 | **0.75** | 0.00 | 39.5 → 43.0 |
| summary | 800 | 0.80 | **0.45** | 0.00 | 599 → 624.5 |

The window and retrieval rows are 3 trials, and real and ideal agree exactly.
The summary rows are 2 trials (`real_output/tight-budget120-summary-trials2.txt`
vs `offline-budget120-trials2.txt`). The earlier 3-trial summary run, made
before the case-sensitivity fix, gave 0.73 / 0.43, which is within noise of the
fixed run.

### What the real model changed

- **The reader is not the bottleneck.** Wherever the summarizer is not
  involved, DeepSeek-flash scored **identically** to the extractive oracle
  reader in every cell (full, fifo, window and retrieval at 400 tokens; window
  and retrieval at 120 tokens, the only ones run there). It answered with
  just the value, used the latest update, ignored "my colleague's manager is …"
  distractors, and said `unknown` when the fact was gone, including on 18.5k-token
  `full` contexts. The `window` stale answers (old pinned timezone) also
  reproduce exactly: a real reader is as confidently wrong as the oracle when the
  correction has been evicted.
- **The summarizer is where real models lose facts.** At a 400-token budget
  (160-word notes cap) the real summarizer nearly matches the ideal one: 0.96 vs
  1.00 at 800 noise, a single missed update across 24 probes. It needs 141 vs 103
  calls per run because its notes are longer. At a 120-token budget (48-word cap)
  recall drops from the ideal 0.80 to **0.45** at 800 noise. The notes pad
  themselves with small talk the prompt says to drop ("the user's interests
  include jazz history, film photography, …", "the user's sister is curious about
  car buying", "the user's colleague's car is a white Golf"). Over about 600
  sequential re-summarizations, facts fall out and never come back.
- **No stale answers came from the summarizer.** Across all real runs it never
  kept a superseded value over its update, so the stale failure mode we
  expected from real summaries did not show up at this sample size. Its errors
  were all **omissions** (misses).
- Reader replies were bare values: 809 output tokens over 360 calls, and every
  answer we inspected was just the value. We saw no code fences, `ANSWER:`
  echoes or JSON wrapping. The summaries we logged (one scenario, 49 merges) were
  plain sentences without bullets, so the cleaners below are defensive.

### Integration bugs found and fixed (regression tests in `tests/test_context_eviction_real.py`)

1. **Summary truncation deleted the oldest facts.** The old cap kept the newest
   *words* (`words[-cap:]`). The real summarizer routinely exceeds the cap and
   puts its filler list *after* the facts, so the truncation cut the oldest fact
   sentences and kept the filler. Because each merge only sees the cut notes,
   the loss was permanent. On the checked-in 120-noise scenario at budget 200,
   this made monthly budget and dentist `unknown`. Fix: `cap_summary` drops whole
   sentences, non-fact sentences first, then the oldest facts, and cuts word-wise
   only when a single sentence is still over the cap.
2. **Fact matching was case-sensitive.** The real summarizer writes "The user's
   manager is …" with a capital T. `FACT_RE` only matched lowercase, so the new
   cap treated every real fact as filler and dropped car and manager on the same
   scenario (`real_output/inspect-summary-noise120-precapfix.txt`, recall 0.75).
   The extractive reader would also have missed every capitalized fact. Fix:
   `FACT_RE` is case-insensitive and attributes are compared lowercased.
3. **Article-sensitive scoring.** The reader answered `blue Corolla` for
   expected `a blue Corolla`, and that was scored a *miss*. `classify` now ignores
   a/an/the.
4. **No temperature control or cost accounting.** Reader and summarizer calls
   now pass `temperature=0`. `CostMeter` totals calls, tokens and
   `cost_usd` per role, and the CLI prints them (`usage` in `--json`).
5. **Too slow against a real API.** A 120-budget, 800-noise summary run is more
   than 600 sequential calls. `sweep(..., workers=N)` runs (strategy, scenario)
   jobs in threads, and a test checks that parallel results equal sequential
   ones. The 400-budget grid took 3.5 minutes with `--workers 8`.
6. Defensive: `clean_summary` strips fences, `UPDATED NOTES:` labels and bullets,
   and `clean_answer` strips fences, `ANSWER:` echoes and quotes. Neither was
   observed in this run.

Re-running the checked-in inspect example after fixes 1–3
(`real_output/inspect-summary-noise120-postfix.txt`, 48 summarizer calls) gives
recall **1.00** where the pre-fix run gave 0.75: car and manager survive, and the
reader's `blue Corolla` now scores correct. This is one scenario, so it shows
the fixes work, not how large their effect is.

Note on fix 1: `cap_summary` decides what counts as a "fact" with `FACT_RE`,
the same sentence template the scenario generator uses. So the `summary`
strategy is really "LLM summarizer + a template-aware trimmer", and its recall
is an upper bound for a trimmer that cannot recognise facts. That only matters
when the model overruns the word cap, which is the 120-token case.

Fixes 1 and 2 changed no offline numbers, because the ideal summarizer only
emits lowercase fact sentences, and the offline tables above are unchanged.
Caveat: the 400-budget real sweep and the 3-trial tight run were made **before
fix 2**. That bug only matters when the notes overflow their cap, which is rare
at a 160-word cap, so those summary rows may be slightly pessimistic. The final
tight-budget summary numbers (0.75 / 0.45) were run after all fixes.

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
