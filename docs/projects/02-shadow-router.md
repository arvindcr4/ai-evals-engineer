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
| mean Jaccard / similarity | set Jaccard; `difflib` ratio on normalised text |
| judge win rate | pairwise LLM judge on every non-identical pair, **run in both A/B orders**; a win or loss counts only when both orders agree, otherwise it's a tie (cancels position bias). Wilson 95% CI on decided pairs |
| cost delta | mean cost/request, shadow vs primary |
| latency | p50 / p95 per side and deltas |
| error rate | shadow errors / all sampled pairs |

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

## Agreement

- exact (normalised): **56.9%**
- near (token Jaccard ≥ 0.6): **89.7%**
- mean token Jaccard: 0.855; mean similarity: 0.912

## Judge (`mock-judge`, both orders)

- shadow wins 0, losses 15, ties 43
- win rate (decided pairs) **0.0%** (95% CI 0.0%–20.4%)
- non-loss rate 74.1%
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
