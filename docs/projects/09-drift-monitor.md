# 09 — Production Drift Monitor

> Golden datasets go stale. Live traffic doesn't.

## What and why

A nightly job that samples 5% of yesterday's production traffic, scores it
offline, stores daily metrics in SQLite and compares each day to a rolling
baseline. It alerts when refusal rate, answer length, tool errors, latency,
cost, an LLM-judge score, or the *shape* of the traffic (answer-length
histogram, tool mix) drifts, before users start complaining.

A golden eval set tells you how the model does on the questions you thought of
last quarter. A drift monitor tells you how it is doing on the questions people
asked last night.

## Design

```
logs/YYYY-MM-DD.jsonl ─▶ JsonlDirSource.read(day)
        ─▶ hash_sample(rate=5%, bottom-k cap) ─▶ scorers ─▶ aggregate()
        ─▶ MetricStore (sqlite: runs, metrics, dists, alerts)
        ─▶ detect(day): z-score · CUSUM · PSI vs rolling baseline
        ─▶ sinks: stdout · jsonl:<path> · webhook[:<url>]
```

**Source and sampling** (`source.py`). `TrafficSource` is a two-method protocol
(`dates()`, `read(day)`), and `JsonlDirSource` reads a date-partitioned log
directory. Sampling is **bottom-k hash sampling**: keep records whose
`sha256(salt, request_id)` falls under `rate`, and optionally keep only the k
smallest hashes (`--max-samples`). The result is a uniform reservoir that does
not depend on file order and is identical on every rerun.

**Scorers** (`scorers.py`). Each scorer maps a record to named values, with
`None` meaning not applicable:

| scorer | values |
|---|---|
| `RefusalScorer` | `refusal` (regex: "I'm sorry", "I can't help", "as an AI", ...) |
| `LengthScorer` | `answer_chars`, `empty` |
| `ToolScorer` | `tool_error` (failed / total calls; `None` if no tools), `tool_calls` |
| `LatencyCostScorer` | `latency_s`, `cost_usd` |
| `JudgeScorer` | `judge_score` ∈ [0,1] from a 1–5 rating by any `get_llm` model (`--judge mock` = offline heuristic judge), plus `judge_cost_usd` and `judge_unscored` |

`aggregate` reduces a day to `refusal_rate, empty_rate, answer_chars_mean,
tool_error_rate, tool_calls_mean, cost_usd_mean, judge_score, latency_p50,
latency_p95`, plus two distributions: `answer_length` (7 fixed bins) and
`tool_mix`.

**Detection** (`detect.py`). The baseline is the last 14 stored days before the
target date, **excluding days that raised critical alerts**, so a sustained
regression keeps alerting instead of quietly becoming the new normal. At least
5 baseline days are required.

- **z-score**: `(x − μ) / σ`, where σ is floored at 5% of μ and, for rate
  metrics, at max(binomial SE at today's sample size, 0.01). Without the floor,
  a 2-in-120 empty-answer day against a near-zero baseline would page someone.
  Warn at |z| ≥ 3, critical at ≥ 4, and **only in the bad direction**: a
  falling refusal rate is not an incident.
- **CUSUM**: a one-sided cumulative sum of standardised deviations (k = 0.5,
  h = 5) over the last 7 days. It catches slow drift that never trips a
  single-day z-score. It is reported only when z-score didn't already fire.
- **PSI**: `Σ (a−e)·ln(a/e)` between the pooled baseline histogram and today's,
  with **additive smoothing (+0.5 per bin)**. A rare bin that happens to be
  empty in a ~100-record sample would otherwise get ε mass and dominate the
  index. Warn ≥ 0.2, critical ≥ 0.3. The message names the bin that moved most.

**Store** (`store.py`). Tables `runs`, `metrics(date, metric, value)`,
`dists(date, name, counts-json)`, `alerts`. Re-running a date replaces its
rows, so `run` and `backfill` are idempotent. `backfill` processes dates in
order, because each day's baseline depends on earlier days' results.

**Sinks** (`alerts.py`). `stdout`, `jsonl:<path>`, `webhook:<url>` (posts a
Slack incoming-webhook payload with a header block plus one section per alert,
via httpx). Plain `webhook` builds the payload and prints it as a dry run.

**Synthetic data** (`generate.py`). 30 days × 2,000 requests. From day index 20
(2026-09-21), a severity ramp reaches full strength after 5 days. Refusals rise
from 3% to 15%, empties from 0.5% to 3.5%, tool errors from 4% to 18%,
answers shrink by 45%, latency grows by 60%, and the tool mix shifts toward a
new `browse` tool.

## How to run

```bash
examples/09-drift-monitor/run.sh        # generate → backfill → nightly run → report

uv run --no-sync evalkit drift-monitor generate --out logs/
uv run --no-sync evalkit drift-monitor backfill --logs logs/ --db drift.sqlite \
    --from 2026-09-01 --to 2026-09-29 --judge mock --sink jsonl:alerts.jsonl
uv run --no-sync evalkit drift-monitor run --logs logs/ --db drift.sqlite \
    --date 2026-09-30 --sink stdout --sink webhook:https://hooks.slack.com/services/... \
    --fail-on-critical
uv run --no-sync evalkit drift-monitor report --db drift.sqlite --out report/
```

`run` defaults `--date` to yesterday (UTC). Use `--judge openai:gpt-4.1-mini`
for a real judge. Its cost is bounded by the sample size, not by traffic.

### Scheduling

systemd (user units):

```ini
# ~/.config/systemd/user/evalkit-drift.service
[Unit]
Description=evalkit nightly drift monitor

[Service]
Type=oneshot
WorkingDirectory=/srv/evalkit
Environment=EVALKIT_API_KEY=...
Environment=SLACK_WEBHOOK_URL=https://hooks.slack.com/services/...
ExecStart=/usr/bin/env uv run --no-sync evalkit drift-monitor run \
  --logs /var/log/assistant --db /srv/evalkit/drift.sqlite --judge openai:gpt-4.1-mini \
  --sink jsonl:/srv/evalkit/alerts.jsonl --sink webhook:${SLACK_WEBHOOK_URL}

# ~/.config/systemd/user/evalkit-drift.timer
[Unit]
Description=Run evalkit drift monitor nightly

[Timer]
OnCalendar=*-*-* 02:15:00 UTC
Persistent=true

[Install]
WantedBy=timers.target
```

`systemctl --user enable --now evalkit-drift.timer`. `Persistent=true` catches
up on a run missed while the machine was off. Or with cron:

```cron
15 2 * * *  cd /srv/evalkit && uv run --no-sync evalkit drift-monitor run --logs /var/log/assistant --db drift.sqlite --sink jsonl:alerts.jsonl --sink "webhook:$SLACK_WEBHOOK_URL" >> drift.log 2>&1
```

## Sample output

From `examples/09-drift-monitor/run.sh` (full files in `sample_output/`):

```
2026-09-13: sampled 93/2000 -> OK | judge 0.944 $0.0044
2026-09-14: sampled 97/2000 -> 0 critical / 1 warning | judge 0.887 $0.0047
2026-09-15: sampled 101/2000 -> OK | judge 0.968 $0.0049
...
2026-09-20: sampled 121/2000 -> OK | judge 0.959 $0.0059
2026-09-21: sampled 108/2000 -> 1 critical / 1 warning | judge 0.875 $0.0049
2026-09-22: sampled 100/2000 -> 3 critical / 3 warning | judge 0.877 $0.0043
2026-09-23: sampled 97/2000 -> 6 critical / 2 warning | judge 0.827 $0.0040
...
[CRITICAL] 2026-09-30 zscore refusal_rate 0.1919 vs baseline 0.03201 (z=+8.0)
[WARNING ] 2026-09-30 cusum  empty_rate drifting up for 7 days (CUSUM=17.2)
[CRITICAL] 2026-09-30 zscore answer_chars_mean 293.5 vs baseline 612 (z=-10.4)
[CRITICAL] 2026-09-30 zscore tool_error_rate 0.2569 vs baseline 0.04397 (z=+10.3)
[CRITICAL] 2026-09-30 zscore latency_p50 2.964 vs baseline 1.73 (z=+11.8)
[CRITICAL] 2026-09-30 zscore judge_score 0.7247 vs baseline 0.9542 (z=-4.8)
[CRITICAL] 2026-09-30 psi    answer_length distribution shifted (PSI=1.47; '500-999' 50% -> 13%)
[CRITICAL] 2026-09-30 psi    tool_mix distribution shifted (PSI=2.98; 'browse' 0% -> 39%)
webhook payload (dry run, no URL):
{ "text": "Drift monitor 2026-09-30: 8 critical, 1 warning", "blocks": [ ... ] }
```

| metric | baseline (first 7d) | latest | change |
|---|---:|---:|---:|
| answer_chars_mean | 612.5 | 293.5 | -52.1% |
| judge_score | 0.9628 | 0.7247 | -24.7% |
| refusal_rate | 0.02686 | 0.1919 | +614.6% |
| tool_error_rate | 0.02688 | 0.2569 | +856.0% |

The decay starts on 2026-09-21. On that same night, at 20% severity, the
tool-mix PSI is already **critical** (0.34), a full day before any scalar
metric goes critical. The judge score is the last signal to react: cheap
heuristics plus distribution tests beat waiting for an LLM judge. The single
warning on 2026-09-14 is a real 3.7σ sampling blip (9 refusals in 97 records).
It stayed a warning, didn't persist, and CUSUM didn't follow it, which is the
intended behaviour for noise.

## Real-model run (DeepSeek, Oct 2026)

```bash
examples/09-drift-monitor/real_run.sh   # sources ~/TradingAgents/.env for DEEPSEEK_API_KEY
# core of it, per configuration:
uv run --no-sync evalkit drift-monitor generate --out logs-grounded --style grounded --seed 7 \
    --start 2026-09-01 --days 30 --per-day 2000 --decay-day 20
uv run --no-sync evalkit drift-monitor backfill --logs logs-grounded --db A.sqlite \
    --from 2026-09-14 --to 2026-09-26 --rate 0.05 --max-samples 80 --baseline-days 7 \
    --judge deepseek:deepseek-flash --workers 8 --sink jsonl:A.alerts.jsonl
```

- **Judge:** `deepseek:deepseek-flash` (V4.1 Flash, thinking off, temperature 0,
  `max_tokens=16`). No other model involved.
- **Window:** 7 healthy days (09-14 → 09-20) + 6 decayed days (09-21 → 09-26,
  severity 20% → 100%), 80 sampled records/day = 1,040 records per run.
- **Configurations:** A = grounded traffic + real judge; B = the same samples
  with the offline mock judge; C = the original filler traffic + real judge.
  Plus a single `run --date 2026-09-26` (the nightly-job form) against store A.
- **Cost:** A $0.086, C $0.086, nightly run $0.006 → **$0.18 per full
  `real_run.sh`**. Real spend this session ≈ **$0.36** (the script ran twice;
  the second run reproduced the first exactly, at temperature 0) plus ~$0.003 of
  probing. ~0.7 s per call; 8 worker threads → ~1 min per 13-day backfill.
- Outputs: `examples/09-drift-monitor/real_output/` (backfill logs, Markdown
  reports, alert JSONL, HTML report for A, nightly-run stdout).

### What the real judge revealed first: the demo data is unjudgeable

The first probe (the original prompt, original filler traffic) returned `1`
for **every** record, healthy or decayed. That was correct: the synthetic
"healthy" answers are random generic sentences ("I pulled the latest numbers
from the warehouse. Three accounts account for most of the variance. ...") that
never address the question. The mock judge only looks at length, refusals and
tool errors, so it rated them 5/5. Run C confirms it at scale: the real-judge
score sits at **0.000–0.009 on every day**, with no room to fall, and never
alerts. (The other detectors still catch the decay in C. Only the judge is
blind.)

So the generator gained `--style grounded` (on-topic answers from a
per-question pool with concrete figures; during decay, up to 35% of answers
become fluent answers to a *different* question). Default `filler` is unchanged,
and the offline demo's sampled traffic, metric values and alerts are unchanged
(the summary line and report table now also show the new `judge_cost_usd` /
`judge_unscored_rate` metrics; with `--judge mock` the cost is `MockLLM`'s
simulated price, not real spend).

### Results (grounded traffic)

| day | severity | sampled on-topic / off-topic / refusal / empty | real judge (A) | mock judge (B) |
|---|---:|---|---:|---:|
| 09-14 | 0 | 78 / 0 / 1 / 1 | 0.975 | 0.966 |
| 09-17 | 0 | 76 / 0 / 3 / 1 | 0.950 | 0.934 |
| 09-20 | 0 | 74 / 0 / 5 / 1 | 0.925 | 0.909 |
| 09-21 | 0.2 | 68 / 6 / 6 / 0 | 0.850 | 0.909 |
| 09-22 | 0.4 | 63 / 8 / 6 / 3 | 0.787 **warn z=−3.5** | 0.850 |
| 09-23 | 0.6 | 61 / 10 / 8 / 1 | 0.762 **crit z=−4.0** | 0.866 |
| 09-24 | 0.8 | 55 / 13 / 9 / 3 | 0.681 **crit z=−5.7** | 0.809 warn (CUSUM) |
| 09-25 | 1.0 | 42 / 20 / 16 / 2 | 0.512 **crit z=−9.3** | 0.734 **crit z=−4.4** |
| 09-26 | 1.0 | 40 / 25 / 15 / 0 | 0.500 **crit z=−9.5** | 0.781 warn z=−3.4 |

(Healthy days 09-15/16/18/19 are in `real_output/`. All OK, no judge alerts.)

```
[CRITICAL] 2026-09-26 zscore judge_score 0.5 vs baseline 0.9554 (z=-9.5)
[CRITICAL] 2026-09-26 psi    tool_mix distribution shifted (PSI=2.60; 'browse' 0% -> 39%)
2026-09-26: sampled 80/2000 -> 8 critical / 1 warning | judge 0.500 $0.0063
```

**Headline: yes, the real judge score detects the decay**: a warning one day
after onset (09-22, severity 40%), critical from 09-23, and z = −9.5 at full
severity. It reacts two days earlier than the mock judge on identical samples
and with roughly twice the effect size. No judge alerts fired on the healthy
days.

**Comparison with the mock, honestly stated:**

- The real judge behaved as a near-binary relevance classifier. Its daily
  score equals the "share of on-topic answers" oracle on 11 of 13 days, and on
  the other two it is off by one record (0.681 vs 0.688, 0.512 vs 0.525). It
  gave 5 to on-topic answers, 1 to off-topic answers and refusals, and almost
  never anything in between.
- It caught the **off-topic** failure mode, which the mock cannot see by
  construction. That is why its drop is larger and earlier. The mock's signal
  comes only from refusals, short answers and tool errors.
- It **ignored tool errors**. On-topic answers with "Tool errors during the
  run: 1" still scored 5, while the mock subtracts a point per error. Tool
  failures are still covered by `tool_error_rate`.
- It did **not** penalise short on-topic answers (162 chars → 5). The mock
  gives 3 to anything under 120 chars.
- It was not first to fire: on 09-21 the tool-mix PSI (critical) and latency
  (warning) fired, but neither judge did. The original conclusion that cheap
  heuristics and distribution tests react first still holds. The judge adds a
  *quality* signal those can't provide, not speed.
- Caveats: one seed, 80 records/day, synthetic traffic whose failure modes the
  judge finds easy (off-topic answers are blatant here). Subtle real-world
  degradations (plausible but wrong figures, partial answers) would give a
  noisier, mid-scale judge signal that this run does not test.

### Integration bugs fixed (regression tests in `tests/test_drift_monitor_real.py`)

1. **First-digit parsing.** `re.search("[1-5]")` took the first digit, so
   replies like "On a 1-5 scale this is a 4" parsed as 1. Replaced with
   `parse_rating`, which prefers `SCORE: n` (last one wins), then `n/5` /
   `n out of 5`, then a single unambiguous standalone digit, and returns
   `None` for ambiguous replies. The prompt now asks for `SCORE: <1-5>`.
2. **Role confusion on empty answers.** Given an empty answer, the real model
   *answered the user* ("I don't have access to your company's internal
   documentation...") instead of grading. Empty answers now score 0 without a
   call (the rubric fixes them at 1/5). A system prompt tells the judge never
   to answer the request itself, and request/answer are delimited by
   `<request>` / `<answer>` tags, with closing tags inside the content
   neutralised.
3. **Non-deterministic sampling.** The judge was called without
   `temperature`. It now sends `temperature=0, max_tokens=16`, and the rerun
   reproduced every daily score exactly.
4. **One failed call killed the nightly job.** Any exception from
   `llm.complete` propagated out of `run_day`. Failures now yield `None` and
   count into the new `judge_unscored_rate` metric. That rate was 0% in this
   run, with no parse failures across ~2,100 calls.
5. **No cost accounting.** Judge spend was invisible. There is now a per-day
   `judge_cost_usd` metric (sum of `Completion.cost_usd`), shown in the
   `backfill`/`run` summary line.
6. **Sequential scoring.** ~0.7 s per call × 80 records × 13 days. The new
   `--workers N` (`MonitorConfig.workers`) scores records in a thread pool
   with order-preserving results, so the metrics match sequential scoring
   (tested).
7. The `examples/09-drift-monitor/run.sh` cleanup deleted the paid
   `out/real/` results. It now keeps `out/real/`.

## Limitations

- ~100 sampled records per day is a small sample. The SD floors and PSI
  smoothing keep the false-alarm rate low, but a seed sweep still shows an
  occasional warning on healthy days. Raise `--rate` for low-traffic products.
- The baseline is purely trailing. There is no weekly seasonality model, so a
  product with strong weekday/weekend differences should use
  `--baseline-days 28` or per-weekday baselines.
- Days with critical alerts are excluded from the baseline until someone
  re-runs them. There is no "acknowledge and re-baseline" command yet.
- Multiple-testing correction is not applied across the ~11 tests per day.
  Thresholds are set conservatively instead.
- The refusal regex is English-only.
- With a real judge, cost scales with the sample: about $0.0001 per judged record
  on deepseek-flash, so 100 records a day costs about $0.01 a night.
