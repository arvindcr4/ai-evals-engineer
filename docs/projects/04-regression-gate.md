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
