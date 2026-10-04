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

## Largest disagreements

| request | sim | judge | primary | shadow |
|---|---:|---|---|---|
| r00566 | 0.39 | loss | For feature flag cleanup, start with the smallest change that reproduces the behaviour. D… | I'm not certain about feature flag cleanup; it depends on your setup. |
| r01012 | 0.46 | loss | For fastapi dependency injection, pin versions so the result is reproducible. Document th… | I'm not certain about fastapi dependency injection; it depends on your setup. |
| r00146 | 0.68 | loss | For jwt token rotation, pin versions so the result is reproducible. Teams that skip this … | In short: For jwt token rotation, pin versions so the result is reproducible. |
| r00500 | 0.69 | loss | For kubernetes pod eviction, keep the rollback path one command away. Teams that skip thi… | In short: For kubernetes pod eviction, keep the rollback path one command away. |
| r00299 | 0.72 | loss | For ci flaky tests, pin versions so the result is reproducible. Document the decision nex… | In short: For ci flaky tests, pin versions so the result is reproducible. |
| r01194 | 0.72 | loss | For ci flaky tests, pin versions so the result is reproducible. Document the decision nex… | In short: For ci flaky tests, pin versions so the result is reproducible. |
| r00237 | 0.74 | loss | For celery retry backoff, pin versions so the result is reproducible. Document the decisi… | In short: For celery retry backoff, pin versions so the result is reproducible. |
| r00274 | 0.74 | loss | For celery retry backoff, pin versions so the result is reproducible. Document the decisi… | In short: For celery retry backoff, pin versions so the result is reproducible. |

## Shadow errors (first few)

- `r00507`: CandidateError: upstream 503 from candidate endpoint
