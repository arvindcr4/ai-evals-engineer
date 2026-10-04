# 04 — CI/CD Regression Gate

> GitHub Action that runs parameterized evals on every PR and blocks merges if task success drops
> >2% or latency spikes. Evals that don't block deploys are just dashboards.

## What and why

Teams run evals, look at a dashboard, and ship anyway. `evalkit regression-gate` makes the eval a
merge check: every PR runs the suite, the result is compared case by case with a baseline
committed from `main`, and the job exits non-zero when quality or latency regresses. The PR gets a
Markdown report saying exactly which cases broke and why.

The gate is deliberately hard to fool in both directions:

- a drop only blocks when it is **over the threshold and statistically significant** (it reuses
  `evalkit.significance`), so a one-case wobble on a small suite doesn't block a PR;
- the significance test is **clustered by dataset item**, so parameter variants of the same item
  (e.g. `tier=fast` and `tier=accurate`) don't count as independent evidence;
- latency blocks on a relative p95 rise **and** an absolute floor, so sub-millisecond jitter
  can't fail a build.

## Design

```
suite.yaml ──► Suite (items × param matrix = cases, scorers, gate policy)
                 │
target ──────────┤  callable "file.py:fn" / "pkg.mod:fn"  or  LLM prompt template via get_llm
                 ▼
              runner ──► RunResult JSON (per case: success, per-check got/expected, latency, error)
                 │
baseline.json ───┤
                 ▼
               gate ──► GateReport: PASS/FAIL + reasons, Markdown, exit code, $GITHUB_STEP_SUMMARY
```

| Module | Responsibility |
|---|---|
| `suite.py` | YAML loading, item × parameter-matrix expansion (`t01[tier=fast]`), scorers: `exact`, `contains`, `regex`, `json_field` (parses JSON, also inside ``` fences), `numeric` (abs/rel tolerance). Dotted field paths into outputs. Unknown scorer types / gate keys fail loudly. |
| `runner.py` | Executes the target per case, times it (or uses the LLM's reported latency), turns exceptions into failing cases instead of crashing the gate, writes results JSON. |
| `gate.py` | Pairs cases by id, paired+clustered significance test of success, p95 latency check, newly failing/passing/missing/added cases, Markdown PR report with a stable HTML marker. |
| `cli.py` | `run` (optionally gating against `--baseline`), `compare` (two saved runs), `baseline` (run on main or promote a results file). |

Gate policy (`gate:` in the suite, overridable by flags):

| Key | Default | Meaning |
|---|---|---|
| `max_success_drop` | 0.02 | absolute drop in task success (fraction) |
| `require_significance` | true | also require p < `alpha` and a CI excluding 0 |
| `alpha` | 0.05 | two-sided significance level |
| `max_p95_latency_increase` | 0.25 | relative p95 rise |
| `min_latency_increase_ms` | 5 | absolute floor for the latency check |

A suite targeting an LLM instead of a function:

```yaml
target: {llm: openai:gpt-4o-mini, system: "Classify the ticket.", prompt: "Ticket: {text}\nTier: {tier}"}
scorers: [{type: contains, expected_field: category}]
```

`--llm <spec>` overrides the model (default: suite value, then `$EVALKIT_LLM`, then the offline mock).

## GitHub Action

`.github/workflows/eval-gate.yml` runs on every `pull_request`: checks out, sets up uv, runs
`evalkit regression-gate run --suite examples/04-regression-gate/suite.yaml --baseline
examples/04-regression-gate/baseline.json`, writes the report to the job summary, uploads the
results, comments on the PR (`gh pr comment --edit-last --create-if-none`, skipped for fork PRs
whose token is read-only), and fails the job when the gate fails. Make the `eval-gate` check
required in branch protection to block merges. Refresh the baseline on `main` with
`SUT_VERSION=1 uv run evalkit regression-gate baseline --suite ... --out .../baseline.json`.

## How to run

```bash
examples/04-regression-gate/run.sh
# or individually
SUT_VERSION=2 uv run --no-sync evalkit regression-gate run --suite examples/04-regression-gate/suite.yaml \
  --baseline examples/04-regression-gate/baseline.json --report report.md --out results.json
uv run --no-sync evalkit regression-gate compare --baseline a.json --candidate b.json --suite suite.yaml
```

The example system under test (`examples/04-regression-gate/sut.py`) is a rule-based
support-ticket triager returning `{category, priority, order_id, amount}` on 40 tickets × 2 tiers
= 80 cases. `SUT_VERSION` selects a build: `1` (main), `1.1` (harmless refactor that fixes an
item), `2` (refund routed to shipping, cents dropped), `3` (same answers, 3× slower).

## Sample output

From `run.sh`:

```
================ SUT_VERSION=1 (unchanged main build)          -> PASS, exit 0
================ SUT_VERSION=1.1 (harmless refactor)            -> PASS, exit 0
================ SUT_VERSION=2 (bad refactor)
regression-gate: FAIL: task success dropped 20.0 pts (limit 2.0), significant at p=0.0064   -> exit 1
================ SUT_VERSION=3 (same answers, 3x slower)
regression-gate: FAIL: p95 latency rose +193% (4.2 → 12.2 ms, limit +25%)                    -> exit 1
```

The PR comment for version 2:

```markdown
## Eval regression gate: **FAIL — merge blocked**

Suite `support-ticket-triage` · 80 cases

> baseline beats candidate by +20.0 pts ± 12.5 (95% CI [10.0, 35.0], p=0.006, n=80 paired, 40 clusters)

| Metric | Baseline | Candidate | Δ | Limit |
|---|---:|---:|---:|---|
| Task success | 86.2% | 66.2% | -20.0 pts | drop ≤ 2.0 pts or not significant |
| Latency p50 | 3.2 ms | 3.1 ms | -0.1 ms | — |
| Latency p95 | 4.2 ms | 4.1 ms | -0% | rise ≤ 25% (or < 2 ms) |
| Errors | 0 | 0 | +0 | — |
| `amount` | 100.0% | 80.0% | -20.0 pts | — |
| `category` | 86.2% | 71.2% | -15.0 pts | — |
| `order_id` | 100.0% | 100.0% | +0.0 pts | — |
| `priority` | 100.0% | 100.0% | +0.0 pts | — |

**Blocking:**
- task success dropped 20.0 pts (limit 2.0), significant at p=0.0064

16 newly failing cases (collapsed):
| `t01[tier=fast]` | `category`: 'shipping' → 'billing'; `amount`: 49.0 → 49.99 |
| `t07[tier=fast]` | `amount`: 15.0 → 15.5 |
| … |
```

## Limitations

- Latency in a committed baseline comes from the machine that produced it; the relative threshold
  plus absolute floor absorbs runner noise, but for real services prefer running the baseline and
  candidate on the same runner (e.g. a matrix job on `main` and the PR head).
- The significance test is two-sided at `alpha`; a one-sided "is it worse?" test would block slightly more often.
- Cases are paired by id; renamed cases show up as "missing" + "new" and are not gated.
- Only success and p95 latency are gated; cost and token budgets would be easy additions to the policy.
- Targets run sequentially in-process; there is no sandboxing of the target callable.

## Real-model run (DeepSeek, Oct 2026)

The system under test becomes a **prompt**: `examples/04-regression-gate/llm_suite.yaml` sends each
of the same 40 tickets (× `channel: [email, chat]` = 80 cases, 40 clusters) to
`deepseek:deepseek-flash` (DeepSeek V4.1 Flash, thinking off, `temperature: 0`, `max_tokens: 200`,
8 concurrent workers) and scores the four JSON fields with `json_field` scorers. Three candidates
are gated against a baseline produced the same way:

| Run | What changed | Purpose |
|---|---|---|
| `same-prompt` | nothing (re-run of `llm_suite.yaml`) | must PASS |
| `bad-prompt` | `llm_suite_degraded.yaml`: a "simplified" system prompt that drops the priority rules, the billing hint and the JSON example and asks for a one-sentence rationale first | must FAIL |
| `think` | same prompt, `--llm deepseek:deepseek-flash+think` (model swap) | shows latency/cost/truncation in the report |

```bash
examples/04-regression-gate/real_run.sh      # sources ~/TradingAgents/.env, writes out/real/
# = for each run:
uv run --no-sync evalkit regression-gate baseline --suite llm_suite.yaml --out out/real/baseline.json
uv run --no-sync evalkit regression-gate run --suite llm_suite_degraded.yaml \
  --baseline out/real/baseline.json --out out/real/results-bad-prompt.json --report out/real/report-bad-prompt.md
uv run --no-sync evalkit regression-gate run --suite llm_suite.yaml --llm deepseek:deepseek-flash+think ...
```

**Sample size / cost:** 4 runs × 80 calls = 320 API calls, 0 API errors. The recorded run cost
**$0.036** (baseline $0.0075, same-prompt $0.0075, bad-prompt $0.0077, think $0.0135; costs are
from `Completion.cost_usd` at peak prices; `real_output/summary.json` → `total_cost_usd`). Including
the first full run (before the truncation fix) and two 4–6-case probes, total spend for this system
was **≈ $0.075** (agent estimate; those extra runs were not saved).

Console (`real_output/console.txt`, verbatim; Markdown reports go to `--report`):

```
================ 1. baseline (main prompt)
support-ticket-triage-llm: 80 cases, success 100.0%, p95 973.3 ms, errors 0, cost $0.0075
baseline written to out/real/baseline.json
================ same-prompt
support-ticket-triage-llm: 80 cases, success 100.0%, p95 893.1 ms, errors 0, cost $0.0075
regression-gate: PASS
-> exit code 0
================ bad-prompt
support-ticket-triage-llm: 80 cases, success 58.8%, p95 1161.2 ms, errors 0, cost $0.0077
regression-gate: FAIL: task success dropped 41.2 pts (limit 2.0), significant at p=0.0002
-> exit code 1
================ think
support-ticket-triage-llm: 80 cases, success 96.2%, p95 1261.3 ms, errors 2, cost $0.0135
regression-gate: PASS
-> exit code 0
```

The `think` report's verdict line: −3.8 pts, 95% CI [−14.4, 0.0], p=0.497, not significant.

Excerpt of the `bad-prompt` PR report (`real_output/report-bad-prompt.md`):

```markdown
| Build | Model | Prompt |
| baseline | `deepseek-flash` | `bc9d99be02ee` |
| candidate | `deepseek-flash` | `88701f736160` |

> baseline beats candidate by +41.2 pts ± 14.2 (95% CI [27.5, 55.9], p<0.001, n=80 paired, 40 clusters)

| Task success | 100.0% | 58.8% | -41.2 pts | drop ≤ 2.0 pts or not significant |
| Latency p95  | 973.3 ms | 1161.2 ms | +19% | rise ≤ 50% (or < 1500 ms) |
| Cost         | $0.0075 | $0.0077 | +3% | — |
| `priority`   | 100.0% | 58.8% | -41.2 pts | — |    (category, order_id, amount: 100% → 100%)

| `t02[channel=email]` | `priority`: 'high' → 'normal' |
```

What the real model showed:

- **Determinism holds at temperature 0, mostly.** The same-prompt re-run reproduced all 80 outputs
  byte-for-byte, so the PASS is exact (Δ 0.0, CI [0, 0]); only latency moved (p95 −8%). The
  degraded prompt, which produces longer free-text outputs, scored 60.0% in an earlier full run
  (not saved; only the 58.8% run is in `real_output/`) and 58.8% in the recorded one — one case
  flipped — so "temperature 0" is not a determinism guarantee for longer
  generations.
- **The degraded prompt fails in one specific way.** Without the priority rules the model
  over-escalates: all 33 failures are `priority` 'high' where 'normal' was expected (20 of 40
  tickets; recomputable from `real_output/results-bad-prompt.json`, e.g. "My package has not arrived yet" → high). Category, order id and amount stayed at
  100% — the model gets those from the text without being told, which the rule-based mock
  cannot. Every degraded output was "reasoning sentence + ```json fence```", so the parsing fixes
  below are what make this a quality signal instead of a parse-failure signal.
- **Model swap to thinking mode** costs +80% and +30% p95 here and *passes*: 3 newly failing
  cases on 80 is not significant (p=0.50, minimum detectable effect ≈ 7.8 pts). Two of the three
  are not wrong answers at all — the reasoning tokens used the whole `max_tokens=200` budget
  (`TruncatedOutputError`, one returned `''`), and the report now says so instead of listing four
  failed checks.
- **Latency is network-bound.** p95 swung −8% between two identical runs and +19–30% for the
  changed builds; the mock suite's 25% / 2 ms limits would flap on a real API, so the LLM suite
  uses `max_p95_latency_increase: 0.5` and `min_latency_increase_ms: 1500`. With those limits
  neither candidate trips the latency gate — the thinking-mode slowdown here is real but small. Note that
  both limits must be exceeded, so with a ~0.97 s baseline p95 the 1500 ms floor dominates: the
  gate only trips above ~2.47 s p95 (+154%); the 50% relative limit is inert at this latency.

Comparison with the offline mock run (`run.sh`): the rule-based SUT starts at 86.2% and its bad
build loses 20 pts of task success (`amount` −20, `category` −15); the LLM starts at 100% (this 40-ticket set is easy
for an LLM — a ceiling effect, so the baseline has no headroom to show improvements) and its bad
build loses 41 pts on `priority` only. Both FAILs are significant with 40 clusters; the mock's
latency failure (v3, 3× slower) has no real-model counterpart here — the thinking build did not
cross the absolute 1.5 s floor. Small sample: 40 items, one run per build; a 1–2 case wobble
(as seen between the two degraded runs) is within noise.

Bugs found and fixed in `regression_gate` by the real run (regression tests in
`tests/test_regression_gate_llm.py`, responders copied from real outputs):

1. **JSON after prose was a total miss.** `json_field` only parsed bare or fenced JSON; the real
   "One sentence of reasoning.\n{...}" reply failed all four checks although every field was right.
   New `extract_json` handles bare, fenced and embedded JSON (last object wins). `exact`,
   `numeric` and `regex` scorers with a `field` now also parse text outputs, and `numeric` accepts
   `"$1,234.50"`.
2. **Generation settings were silently ignored.** `temperature`, `max_tokens`, … in the suite's
   `target:` were never sent to the model, so "temperature 0" baselines ran with default sampling.
   They are now forwarded; unknown target keys (`temprature`) raise.
3. **Prompt templates with literal JSON crashed.** `str.format` raised on `{"category": ...}` in
   a prompt; `render_template` substitutes only `{identifier}` placeholders (unknown ones raise).
4. **No token/cost accounting.** Cases now record `tokens_in/out` and `cost_usd`; the summary,
   CLI line and PR report show cost; results files written before this still load.
5. **Truncation looked like a wrong answer.** `finish_reason == "length"` now raises
   `TruncatedOutputError` (tokens still billed) so it shows as an error, and the report warns when
   the candidate has more errored cases than the baseline.
6. **Provenance.** Run `meta` records the model (incl. `+think`, which `Completion.model` drops),
   a 12-char prompt fingerprint and the generation settings; the report shows a Build table.
7. **Sequential calls.** `workers` (target key or `--workers`) runs cases concurrently; order is
   preserved (the real runs used 8 workers).

Sample outputs (reports, console, `summary.json` with per-run summaries, failure patterns and a
few raw model outputs, plus the full per-case results files `baseline.json` and
`results-{same-prompt,bad-prompt,think}.json`) are in `examples/04-regression-gate/real_output/`.
The three reports regenerate offline, with no API calls, from those files:
`evalkit regression-gate compare --suite llm_suite.yaml --baseline real_output/baseline.json
--candidate real_output/results-think.json`.
