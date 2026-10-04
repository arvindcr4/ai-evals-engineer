# 02 — Shadow Routing Comparator

> You can't A/B test non-determinism safely without shadow mode.

## What and why

An OpenAI-compatible proxy that sits in front of your production model. Every
request is answered by the **primary** model, exactly as before. A sticky,
hash-sampled slice of traffic (5% by default) is also mirrored to a
**candidate** model in the background. Both answers are logged as a pair, and a
`report` command diffs the pairs into a cost/quality verdict.

Users never see the candidate's output, never wait for it, and never feel its
failures, so you can evaluate a cheaper or newer model on real traffic before an
A/B test exposes anyone to it.

## Design

```
client ──POST /v1/chat/completions──▶ FastAPI ──▶ ShadowRouter.handle()
                                                   │ primary.complete()  ──▶ response to client
                                                   │ should_shadow(key)? ──▶ orchestrator pool
                                                   ▼                         │ candidate.complete()
                                            shadow_pairs.jsonl ◀─────────────┘ (timeout, errors caught)
shadow_pairs.jsonl ──▶ build_report() ──▶ report.md / report.html / report.json
```

**Sampling** (`proxy.py`). `sample_key` picks a sticky key: `X-User-Id` header,
else the body's `user` field, else `X-Request-Id`, else a hash of the
conversation. `should_shadow(key, rate, salt)` is `sha256(salt, key) mod 1e6 <
rate·1e6`: deterministic across processes and restarts, so a given user is
always in or out of the cohort. Changing `--salt` reshuffles the cohort.

**Off the latency path.** The primary result is returned as soon as it exists.
The shadow call is submitted to an orchestrator thread pool, which runs the
candidate on a second pool and waits at most `--timeout` seconds. Exceptions,
timeouts (wall clock, or a reported latency above the timeout) and log-write
failures are caught, logged and recorded in the pair's `shadow.error`. A
primary failure returns 502 and is never shadowed. `GET /shadow/stats` exposes
counters (`requests`, `sampled`, `shadow_errors`, ...).

**Pair record**: `{ts, request_id, sample_key, messages, primary: Side, shadow:
Side}` where `Side = {model, text, tokens_in, tokens_out, latency_s, cost_usd,
error}`.

**Report** (`report.py`), computed over pairs where both sides succeeded:

| metric | how |
|---|---|
| exact agreement | normalised text (lowercase, alphanumeric tokens) equal |
| near agreement | token Jaccard ≥ 0.6 |
| mean Jaccard / similarity | set Jaccard; `difflib` ratio on the normalised **word** sequence (`autojunk` off) |
| judge win rate | pairwise LLM judge on every non-identical pair, **run in both A/B orders**; a win or loss counts only when both orders agree, otherwise it's a tie (cancels position bias). Wilson 95% CI on decided pairs. Judge calls run at temperature 0 and see the request's system prompt; order flips, slot-A share (position bias), unparsed verdicts, judge errors and judge spend are reported |
| cost delta | mean cost/request, shadow vs primary |
| latency | p50 / p95 per side and deltas |
| error rate | shadow errors / all sampled pairs |
| truncated | share of answers with upstream `finish_reason == "length"` (also passed through in the proxy response) |

Verdict rules: `BLOCK` if the candidate error rate exceeds `--max-error-rate`
(2%) or the judge's win-rate CI lies entirely below 50%; `PROMOTE` if the CI is
entirely above 50%, or the candidate is cheaper with a ≥90% non-loss rate;
otherwise `HOLD`. `--fail-on-block` makes the report exit 2 for CI use.

**Offline demo models** (`demo.py`). `demo:primary` and `demo:candidate` are
template models whose output is a pure function of the prompt. The candidate
answers verbatim ~60% of the time, paraphrases or truncates ~30%, hedges
("I'm not certain...") ~8%, and throws a 503 ~3%, at ~6% of the primary's price.
Any `get_llm` spec (`openai:gpt-4.1-mini`, `deepseek:deepseek-chat`,
`http://localhost:8000/v1|qwen`) works for `--primary`, `--candidate` and
`--judge`. `--judge mock` uses an offline heuristic judge (prefers the answer
that covers more of the question; penalises hedges).

## How to run

```bash
# offline end-to-end demo (1,200 requests, 5% shadow rate)
examples/02-shadow-router/run.sh

# real proxy
uv run --no-sync evalkit shadow-router serve \
  --primary openai:gpt-4.1 --candidate deepseek:deepseek-chat --rate 0.05 \
  --log shadow_pairs.jsonl --port 8787
# point your OpenAI client's base_url at http://127.0.0.1:8787/v1

uv run --no-sync evalkit shadow-router simulate requests.jsonl --rate 0.05 --log pairs.jsonl
uv run --no-sync evalkit shadow-router report pairs.jsonl --judge openai:gpt-4.1-mini --out report/
```

`requests.jsonl` rows are `{"request_id"?, "user"?, "messages": [...]}` (or
`{"prompt": "..."}`). `simulate` sends them through the real FastAPI app via
`TestClient`, so it exercises the same code path as `serve`.

## Sample output

From `examples/02-shadow-router/run.sh` (full files in `sample_output/`):

```
replayed 1200 requests (1200 ok, 0 failed) in 2.67s; shadowed 59 (4.9%), 1 shadow errors -> examples/02-shadow-router/out/pairs.jsonl
# Shadow report: `demo-small` vs `demo-large`

**Verdict: BLOCK** — judge says candidate loses significantly (win-rate CI below 50%)

59 shadow pairs logged, 58 comparable (both sides succeeded).

| metric | primary | shadow | delta |
|---|---:|---:|---:|
| mean cost / request | $0.000328 | $0.000019 | -94.3% |
| latency p50 | 0.944s | 0.493s | -0.451s |
| latency p95 | 1.261s | 0.631s | -0.630s |
| mean output tokens | 29.9 | 28.5 | |
| error rate | 0.0% | 1.7% | |
| truncated (finish=length) | 0.0% | 0.0% | |

## Agreement

- exact (normalised): **56.9%**
- near (token Jaccard ≥ 0.6): **89.7%**
- mean token Jaccard: 0.855; mean similarity: 0.894

## Judge (`mock-judge`, both orders)

- shadow wins 0, losses 15, ties 43
- win rate (decided pairs) **0.0%** (95% CI 0.0%–20.4%)
- non-loss rate 74.1%
- order flips (verdict changed with A/B order, scored tie) 0; unparsed verdicts 0; judge errors 0
- slot-A share of decisive single-order verdicts 50% (0.5 = no position bias)
```

The candidate is 94% cheaper and twice as fast, and 90% of its answers are near
matches, but the judge finds 15 real losses (hedged or truncated answers), so
the verdict is BLOCK. A "90% agreement" headline alone would have hidden that.

## Limitations

- Pairs are only comparable for single-turn semantics: the candidate sees the
  conversation the primary was given, never its own earlier turns. Multi-turn
  divergence needs a full trajectory replay (see 11 replay-debugger).
- Tool-calling requests are mirrored as plain chat. Executing the candidate's
  tool calls would touch production side effects, so it is deliberately not
  done.
- Lexical agreement metrics undercount paraphrases. The judge is what measures
  quality, and an LLM judge has its own biases. Calibrate it first (03).
- The in-process thread pools are not a durable queue. If the process dies,
  in-flight shadow calls are lost. That's acceptable for sampling, not for
  billing.
- Streaming (`stream: true`) is not supported. Responses are returned whole.

## Real-model run (DeepSeek, Oct 2026)

```bash
examples/02-shadow-router/real_run.sh     # sources ~/TradingAgents/.env for DEEPSEEK_API_KEY
```

which runs:

```bash
evalkit shadow-router simulate examples/02-shadow-router/real_requests.jsonl \
  --primary deepseek:deepseek-v4-pro --candidate deepseek:deepseek-flash \
  --rate 0.75 --timeout 60 --log out/real/pairs.jsonl
evalkit shadow-router report out/real/pairs.jsonl --judge deepseek:deepseek-v4-pro --out out/real/report_judge_pro
evalkit shadow-router report out/real/pairs.jsonl --judge deepseek:deepseek-flash  --out out/real/report_judge_flash
```

**Setup.** The primary (serves users) is `deepseek-v4-pro`, the expensive
incumbent. The candidate (shadow) is `deepseek-flash`, the cheaper model you
would want to swap in, which is the usual reason to run shadow mode. Both run
with thinking off. `real_requests.jsonl` holds the first 60 *distinct* questions
from the demo traffic, each with a support-chat system prompt ("answer in at
most 120 words"), `temperature: 0` and `max_tokens: 400`. A 0.75 sticky shadow
rate sampled **47 pairs**. Each report was judged twice: once by v4-pro (the
primary's model, so it could favour itself) and once by flash (the candidate's
model, so it could favour itself). When the two judges agree, self-preference
does not explain the result.

**Cost.** `real_run.sh` costs about **$0.11** (60 pro + 47 flash answers ≈ $0.05;
v4-pro judge $0.052; flash judge $0.012; 94 judge calls each). All the
development runs together, including a pre-fix run and judge re-runs, cost
about **$0.33**. Outputs are in `examples/02-shadow-router/real_output/`: the
two final reports, the full 47-pair log (`pairs.jsonl`), the first run's pair
log and its pre-fix / fixed-judge reports (`pairs.run1_prefix.jsonl`,
`report_run1_*.md`), the first judging pass (`first_judging.log`) and
`analysis.txt`, which `real_analysis.py` recomputes from the two pair logs
without API calls.

**Result.** Excerpt from `real_output/simulate.txt` and `real_output/report_judge_pro.md`:

```
replayed 60 requests (60 ok, 0 failed) in 172.96s; shadowed 47 (78.3%), 0 shadow errors
**Verdict: PROMOTE** — judge says candidate wins significantly

| metric | primary | shadow | delta |
|---|---:|---:|---:|
| mean cost / request | $0.000692 | $0.000189 | -72.7% |
| latency p50 | 2.900s | 1.412s | -1.489s |
| latency p95 | 3.638s | 1.801s | -1.837s |
| mean output tokens | 161.3 | 146.8 | |
| error rate | 0.0% | 0.0% | |
| truncated (finish=length) | 0.0% | 0.0% | |

- exact (normalised): 0.0%;  near (token Jaccard ≥ 0.6): 0.0%
- mean token Jaccard: 0.277; mean similarity: 0.221
```

| judge | shadow wins | losses | ties | order flips | slot-A share | win rate (95% CI) | verdict |
|---|---:|---:|---:|---:|---:|---|---|
| deepseek-v4-pro | 20 | 1 | 26 | 26 | 67% | 95.2% (77.3–99.2%) | PROMOTE |
| deepseek-flash  | 37 | 0 | 10 | 8 | 46% | 100% (90.6–100%) | PROMOTE |

Both judges prefer the cheaper candidate, so the PROMOTE is not just the flash
judge favouring its own model. The v4-pro judge even rules against v4-pro's own
answers. One reason is that flash follows the 120-word limit better: 1 of 47
flash answers runs over it, against 7 of 47 v4-pro answers, and v4-pro averages
100 words to flash's 88 (`analysis.txt`). Spot checks show content differences
too. On `r00000` flash correctly notes that `CancelledError` is not caught by
`except Exception` since Python 3.8. The single judged loss
(`r00040`) is a real flash error: flash says nginx `limit_req` is enforced
per worker, but the zones live in shared memory.

**How this differs from the offline mock.**

| | mock (`demo:*`, `--judge mock`) | real DeepSeek |
|---|---|---|
| verdict | BLOCK: the candidate hedges or truncates in 15/58 pairs | PROMOTE: the candidate is cheaper, faster and judged better |
| exact / near agreement | 56.9% / 89.7% | **0% / 0%** |
| candidate errors, timeouts | 1.7% (injected 503s) | 0 in 47 calls |
| judge position bias | none (symmetric heuristic) | v4-pro picks slot A 67% of the time; 26/47 pairs flip with order |
| determinism | pure function of the prompt | at `temperature: 0`, **0/47** answers were byte-identical across two runs on either model |

What the real run showed:

- **Lexical agreement says little here.** Re-running the *same* model on the
  same 47 prompts at temperature 0 gives a mean word-sequence similarity of
  only 0.47 (v4-pro) and 0.54 (flash), against 0.22 across the two models.
  Exact and near agreement are 0% on real free-form answers. Only the judge
  separates quality. Agreement numbers need a same-model re-run as their noise
  floor before anyone reads them. (The re-run is the first run in
  `pairs.run1_prefix.jsonl`: same 47 prompts, models and parameters.)
- **The judge has a strong position bias and is not fully deterministic.**
  Running both orders was essential, because the v4-pro judge flipped on 26 of
  47 pairs, all of which count as ties. Re-judging the same pairs at
  temperature 0 changed the tally from 21/1/25 to 20/1/26, and the flash
  judge's from 35/0/12 to 37/0/10 (`first_judging.log` vs the final reports).
- **The judge must see the system prompt.** Before the fix, the judge saw only
  the user question and ran at default temperature. On the first run's 47
  pairs it scored 12 wins, 7 losses and 28 ties for flash, and the verdict was
  HOLD (`report_run1_prefix_judge_pro.md`). With the fix it scored 21/4/22 on
  those same pairs (`report_run1_fixed_judge_pro.md`), because it could
  now penalise answers that ignored the length limit.
- **Things the real models did not do in this small sample:** errors,
  timeouts (p95 was 3.6s against the 60s timeout), truncation, and
  unparseable judge verdicts (0 unparsed in every report).
  The proxy's error handling and the stricter parser were therefore only
  exercised by tests here.

**Bugs fixed for real outputs** (regression tests in
`tests/test_shadow_router_real.py`):

1. `similarity` used `difflib` on character strings with the default
   `autojunk`. On answers longer than 200 characters, that heuristic marks
   every common character as junk, so mean similarity fell to **0.064** and
   clear paraphrases scored 0.00. It now compares word sequences with
   `autojunk=False`, giving 0.22 on the same pairs.
2. The judge ran at the provider's default temperature and saw no system
   prompt. It now runs at `temperature=0` with `max_tokens=16` and is shown
   the system instructions.
3. Verdict parsing took the first `A`, `B` or `TIE` anywhere in the reply, so
   "A and B are both fine" counted as A. Ambiguous replies are now counted as
   `unparsed_verdicts` and scored as ties. The parser also accepts `**B**`,
   `Verdict: B` and `B, because …`.
4. A judge exception aborted the whole report, throwing away pairs that had
   already been paid for. Failures are now recorded as `error` outcomes and
   left out of the rates.
5. The HOLD reason was wrong. A candidate that was 73% cheaper but below the
   90% non-loss bar was reported as having "no cost win". The reason now
   states the real cause.
6. The proxy hard-coded `finish_reason: "stop"` even when the upstream answer
   was truncated. It now passes `finish_reason` through and logs it per side.
   The report shows a truncation rate.
7. The disagreements table was sorted by lexical similarity. With real
   answers that sort is close to noise, so the table now lists judged losses
   first.
8. The judge's own spend (calls, tokens, cost) and its position bias
   (order flips, slot-A share) were invisible. Both are now in the report.

**Caveats.** This is a small sample: 47 pairs, one prompt style, 60 distinct
questions. Both models are DeepSeek, so the judge is never from an
independent model family. Costs use cache-miss input prices. DeepSeek reports
`prompt_cache_hit_tokens` and `core/llm.py` ignores them, so input cost is a
slight overestimate. The unsampled primary calls are not logged, so their
spend appears only in the total above. Do not treat this as evidence that
flash beats v4-pro in general. It shows that, on short dev-support answers
under a word limit, this pipeline found a cheaper candidate it could not
reject.
