<!-- evalkit-regression-gate -->
## Eval regression gate: **PASS**

Suite `support-ticket-triage-llm` · 80 cases

| Build | Model | Prompt |
|---|---|---|
| baseline | `deepseek-flash` | `bc9d99be02ee` |
| candidate | `deepseek-flash` | `bc9d99be02ee` |

> No significant difference between baseline and candidate: candidate − baseline = +0.0 pts (95% CI [0.0, 0.0], p=1.000, n=80 paired, 40 clusters).

| Metric | Baseline | Candidate | Δ | Limit |
|---|---:|---:|---:|---|
| Task success | 100.0% | 100.0% | +0.0 pts | drop ≤ 2.0 pts or not significant |
| Latency p50 | 691.8 ms | 644.0 ms | -47.8 ms | — |
| Latency p95 | 973.3 ms | 893.1 ms | -8% | rise ≤ 50% (or < 1500 ms) |
| Errors | 0 | 0 | +0 | — |
| Cost | $0.0075 | $0.0075 | +0% | — |
| `amount` | 100.0% | 100.0% | +0.0 pts | — |
| `category` | 100.0% | 100.0% | +0.0 pts | — |
| `order_id` | 100.0% | 100.0% | +0.0 pts | — |
| `priority` | 100.0% | 100.0% | +0.0 pts | — |
