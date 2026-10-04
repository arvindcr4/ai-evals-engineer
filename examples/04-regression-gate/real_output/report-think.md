<!-- evalkit-regression-gate -->
## Eval regression gate: **PASS**

Suite `support-ticket-triage-llm` · 80 cases

| Build | Model | Prompt |
|---|---|---|
| baseline | `deepseek-flash` | `bc9d99be02ee` |
| candidate | `deepseek-flash+think` | `bc9d99be02ee` |

> No significant difference between baseline and candidate: candidate − baseline = -3.8 pts (95% CI [-14.4, 0.0], p=0.497, n=80 paired, 40 clusters). Smallest effect detectable at this n with 80% power ≈ 7.8 pts.

| Metric | Baseline | Candidate | Δ | Limit |
|---|---:|---:|---:|---|
| Task success | 100.0% | 96.2% | -3.8 pts | drop ≤ 2.0 pts or not significant |
| Latency p50 | 691.8 ms | 899.7 ms | +207.9 ms | — |
| Latency p95 | 973.3 ms | 1261.3 ms | +30% | rise ≤ 50% (or < 1500 ms) |
| Errors | 0 | 2 | +2 | — |
| Cost | $0.0075 | $0.0135 | +80% | — |
| `amount` | 100.0% | 97.5% | -2.5 pts | — |
| `category` | 100.0% | 97.5% | -2.5 pts | — |
| `order_id` | 100.0% | 97.5% | -2.5 pts | — |
| `priority` | 100.0% | 96.2% | -3.8 pts | — |

> Warning: 2 more errored cases than the baseline — check for API/infra failures before reading the drop as a quality regression.

<details><summary>3 newly failing cases</summary>

| Case | Failed check: got → expected |
|---|---|
| `t03[channel=email]` | error: TruncatedOutputError: output hit max_tokens=200 (200 tokens out); partial text: '' |
| `t03[channel=chat]` | `priority`: 'high' → 'normal' |
| `t31[channel=chat]` | error: TruncatedOutputError: output hit max_tokens=200 (200 tokens out); partial text: '{"category": "billing", "priority": "normal", "' |

</details>
