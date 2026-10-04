# Shadow report: `deepseek-flash` vs `deepseek-v4-pro`

**Verdict: PROMOTE** — judge says candidate wins significantly

47 shadow pairs logged, 47 comparable (both sides succeeded).

| metric | primary | shadow | delta |
|---|---:|---:|---:|
| mean cost / request | $0.000692 | $0.000189 | -72.7% |
| latency p50 | 2.900s | 1.412s | -1.489s |
| latency p95 | 3.638s | 1.801s | -1.837s |
| mean output tokens | 161.3 | 146.8 | |
| error rate | 0.0% | 0.0% | |
| truncated (finish=length) | 0.0% | 0.0% | |

## Agreement

- exact (normalised): **0.0%**
- near (token Jaccard ≥ 0.6): **0.0%**
- mean token Jaccard: 0.277; mean similarity: 0.221

## Judge (`deepseek-v4-pro`, both orders)

- shadow wins 20, losses 1, ties 26
- win rate (decided pairs) **95.2%** (95% CI 77.3%–99.2%)
- non-loss rate 97.9%
- order flips (verdict changed with A/B order, scored tie) 26; unparsed verdicts 0; judge errors 0
- slot-A share of decisive single-order verdicts 67% (0.5 = no position bias)
- judge spend: 94 calls, $0.0518

## Largest disagreements

| request | sim | judge | primary | shadow |
|---|---:|---|---|---|
| r00040 | 0.14 | loss | Nginx rate limiting typically "breaks" under load due to the **burst** and **nodelay** pa… | Nginx `limit_req` is per-worker, not global. Under load, each worker enforces its own buc… |
| r00023 | 0.09 | win | gRPC deadline propagation automatically carries the timeout from an incoming request to a… | gRPC deadlines propagate automatically: when a client sets a deadline, the server receive… |
| r00068 | 0.11 | win | Postgres connection pooling breaks under load typically due to:  1. **Pool exhaustion** –… | Pooling usually breaks because of **exhaustion and queueing**, not pooling itself:  - **P… |
| r00051 | 0.11 | win | For React hydration errors, follow these practices:  1. **Avoid browser-only APIs during … | Fix the root cause: server and client must render identical HTML on first pass.  - Don't … |
| r00052 | 0.14 | win | First, check the test logs for the **failure reason** and **timing**. Look for:  1. **Sha… | Check the failure pattern first: is it the same test, same step, or random? Then verify: … |
| r00002 | 0.15 | win | Vector index recall degrades under load primarily due to resource contention and algorith… | Recall drops under load mainly because the index is being searched *approximately* under … |
| r00021 | 0.16 | win | CI flaky tests break under load because resource contention exposes hidden assumptions in… | Flaky tests under load usually stem from timing and resource contention, not logic bugs: … |
| r00055 | 0.17 | win | First, verify your **embedding model matches the one used at index time** — a mismatch is… | Check the basics first:  1. **Ground truth quality** — verify your recall calculation use… |
