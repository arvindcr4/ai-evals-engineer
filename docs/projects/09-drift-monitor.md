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
| `JudgeScorer` | `judge_score` ∈ [0,1] from a 1–5 rating by any `get_llm` model (`--judge mock` = offline heuristic judge) |

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
2026-09-13: sampled 93/2000 -> OK
2026-09-14: sampled 97/2000 -> 0 critical / 1 warning
2026-09-15: sampled 101/2000 -> OK
...
2026-09-20: sampled 121/2000 -> OK
2026-09-21: sampled 108/2000 -> 1 critical / 1 warning
2026-09-22: sampled 100/2000 -> 3 critical / 3 warning
2026-09-23: sampled 97/2000 -> 6 critical / 2 warning
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
