# 10 — Cost-Quality Pareto Dashboard

> Unit economics decide if your AI product survives the next pricing drop.

## What and why

A visualiser for the exact trade-off between inference cost and task success,
**per tenant**. It takes raw eval results for every model/config you run,
computes success rates with confidence intervals, cost per task and cost per
*successful* task, finds the non-dominated (Pareto) configs, simulates
System-One → System-Two escalation routers, and answers "what if this model's
price drops 75%?"

The cheapest model per token is rarely the cheapest per solved task. The best
model is often not worth 4× the price on easy tenants and is the only option
on hard ones. That is a per-tenant decision, and it changes every time a
provider re-prices.

## Design

```
results.jsonl ─▶ summarize()  ─▶ per (tenant, config): n, success, Wilson CI, cost/task,
                                  cost/success, p95 latency, on_frontier
             ─▶ simulate_router(cheap, expensive, thresholds) ─▶ blended curve
             ─▶ apply_price_factor(pattern, ×f) ─▶ frontier_diff(before, after)
             ─▶ dashboard: pareto.html (inline SVG + vanilla JS) · pareto.md · pareto.json
```

**Input rows**: `{tenant, config, task_id, success, cost_usd, latency_s,
confidence?}`. All configs for a tenant are expected to have run on the same
`task_id`s, which is what makes per-task routing simulation possible.

**Frontier** (`analysis.pareto_frontier`). Sort by cost ascending (ties broken
toward higher success), then keep a point iff its success strictly beats every
cheaper point. This is O(n log n), and a test checks it against a brute-force
dominance check on random point sets. The chart draws the frontier as a
**staircase** (for a discrete set of options, spending more never buys less
success), with dominated configs greyed out.

**Statistics.** Success rates carry Wilson 95% intervals, drawn as whiskers.
With 60 tasks per tenant they are ±10pp, which is the point: two configs whose
whiskers overlap are not reliably ordered, however the dots look.
Cost per success = mean cost / success rate (∞ when nothing succeeds).

**Router simulation** (`simulate_router`). Each task goes to the cheap
(System-One) config first. If that config's self-reported `confidence` is below
threshold *t*, the task escalates to the expensive (System-Two) config and
takes its outcome. Sweeping *t* over [0, 1] traces a curve from "never
escalate" to "always escalate". By default the cascade pays for both calls on
escalation, since the cheap attempt is sunk. `--no-cascade` models a
pre-router that only pays for the expensive call. Rows without `confidence`
fall back to a hash-random router, which interpolates linearly. Router points
that beat the config frontier are flagged, and the dashboard reports the
cheapest threshold within 2pp of always escalating. If no threshold saves money
at parity, it says so instead.

**What-if** (`apply_price_factor` + `frontier_diff`). Scale `cost_usd` for every
config matching a glob (`s2-frontier`, `s2-*`) by a factor, recompute, and
report which configs joined or left each tenant's frontier and how cost per
success moved. The dashboard draws the old frontier as a dotted ghost line.

**Dashboard** (`dashboard.py`). One self-contained HTML file: tenant `<select>`
(remembered in the URL hash), KPI tiles (best quality, value pick = cheapest
config within 2pp of best, router at parity), a log-cost × success scatter with
CI whiskers, frontier staircase, router curve and hover tooltips, plus config
and router tables. Colours are CSS variables with a dark-mode set under
`prefers-color-scheme` and `data-theme`. Identity never relies on colour
alone: every config is direct-labelled, and the labels avoid each other and the
data marks. On narrow screens the chart scrolls horizontally instead of
shrinking to illegibility.

**Synthetic generator** (`generate.py`). 4 tenants (support → legal, increasing
difficulty and token footprint) × 7 configs: `s1-nano`, `s1-mini`,
`s1-mini+rag` (2.5× input tokens), `s2-legacy` (expensive *and* weaker, so it is
always dominated), `s2-large`, `s2-large+reasoning` (4× output tokens), and
`s2-frontier`. Prices are per 1M tokens. A task's latent difficulty is shared
across configs, so outcomes are correlated like real evals, and confidence is a
noisy, roughly calibrated function of each config's true success probability.

## How to run

```bash
examples/10-pareto-dashboard/run.sh

uv run --no-sync evalkit pareto generate --out results.jsonl --tasks 120 --seed 11
uv run --no-sync evalkit pareto build results.jsonl --router s1-mini+rag,s2-large+reasoning --out out/
uv run --no-sync evalkit pareto whatif results.jsonl --config 's2-*' --factor 0.5 --out out/
```

Without `--router`, each tenant gets an automatic pair: the best frontier config
costing ≤10% of the top config, escalating to the top config. Open
`out/pareto.html` in a browser.

## Sample output

From `examples/10-pareto-dashboard/run.sh` (full files in `sample_output/`, and
`sample_output/pareto.html` is the dashboard):

```
globex-support     frontier: s1-nano < s1-mini < s1-mini+rag < s2-large < s2-large+reasoning
initech-finance    frontier: s1-nano < s1-mini < s1-mini+rag < s2-large < s2-large+reasoning < s2-frontier
acme-legal         frontier: s1-nano < s1-mini < s1-mini+rag < s2-large < s2-large+reasoning < s2-frontier
umbrella-health    frontier: s1-nano < s1-mini < s1-mini+rag < s2-large < s2-large+reasoning
```

```
## globex-support
- best quality: **s2-large+reasoning** (96.7%)
- router `s1-mini+rag` → `s2-large+reasoning` at threshold 0.80: 95.0% success, 32% escalated,
  $0.0072/task vs $0.0175 always-escalate

| config | success (95% CI) | cost/task | cost/success | p95 latency | frontier |
|---|---:|---:|---:|---:|:---:|
| s1-nano | 61.7% (49%–73%) | $0.0002 | $0.0004 | 1.2s | ● |
| s1-mini | 75.0% (63%–84%) | $0.0005 | $0.0006 | 1.7s | ● |
| s1-mini+rag | 88.3% (78%–94%) | $0.0009 | $0.0011 | 3.5s | ● |
| s2-large | 93.3% (84%–97%) | $0.0079 | $0.0085 | 3.6s | ● |
| s2-legacy | 88.3% (78%–94%) | $0.0146 | $0.0165 | 6.7s |  |
| s2-large+reasoning | 96.7% (89%–99%) | $0.0166 | $0.0171 | 15.6s | ● |
| s2-frontier | 96.7% (89%–99%) | $0.0559 | $0.0578 | 13.0s |  |

## initech-finance
- best quality: **s2-frontier** (88.3%)
- value pick (≤2pp off best): s2-large+reasoning · 73% cheaper
- router `s1-mini+rag` → `s2-large+reasoning`: no threshold stays within 2pp of always-escalate at lower cost
```

What-if, `--config s2-frontier --factor 0.25`:

```
- globex-support: joined frontier ['s2-frontier']; left ['s2-large+reasoning']; cost/success s2-frontier $0.0578 → $0.0144
- initech-finance: joined frontier —; left ['s2-large+reasoning']; cost/success s2-frontier $0.1146 → $0.0286
- acme-legal: joined frontier —; left ['s2-large+reasoning']; cost/success s2-frontier $0.2914 → $0.0729
- umbrella-health: joined frontier ['s2-frontier']; left —; cost/success s2-frontier $0.1724 → $0.0431
```

The router finding is the interesting one. For the support tenant, escalating
only the 32% of tasks where the cheap model is unsure gives 95% success at 41%
of the always-escalate cost. For finance, the cheap model's confidence isn't
good enough to route on, so the router saves nothing at parity. Same configs,
different tenant, different answer. A 75% price cut on the frontier model
knocks the reasoning config off three of the four frontiers.

## Limitations

- Success is binary. Graded scores (0–1) would need a mean ± bootstrap CI in
  place of Wilson.
- Frontier membership ignores the CIs. A config can be "dominated" by a point
  estimate that is statistically indistinguishable from it. Use 07
  (significance) before acting on a close call.
- The router uses a single scalar confidence and a single threshold.
  Learned routers and multi-tier cascades are not modelled. Router costs ignore
  the classifier's own cost.
- What-if scales observed cost linearly. It does not model tokenizer
  differences, cached-input discounts or rate-limit-driven retries.
- Latency is reported as p95 per config but is not an axis of the frontier.
