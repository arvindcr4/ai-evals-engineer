<!-- evalkit-regression-gate -->
## Eval regression gate: **FAIL — merge blocked**

Suite `support-ticket-triage-llm` · 80 cases

| Build | Model | Prompt |
|---|---|---|
| baseline | `deepseek-flash` | `bc9d99be02ee` |
| candidate | `deepseek-flash` | `88701f736160` |

> baseline beats candidate by +41.2 pts ± 14.2 (95% CI [27.5, 55.9], p<0.001, n=80 paired, 40 clusters)

| Metric | Baseline | Candidate | Δ | Limit |
|---|---:|---:|---:|---|
| Task success | 100.0% | 58.8% | -41.2 pts | drop ≤ 2.0 pts or not significant |
| Latency p50 | 691.8 ms | 832.7 ms | +140.9 ms | — |
| Latency p95 | 973.3 ms | 1161.2 ms | +19% | rise ≤ 50% (or < 1500 ms) |
| Errors | 0 | 0 | +0 | — |
| Cost | $0.0075 | $0.0077 | +3% | — |
| `amount` | 100.0% | 100.0% | +0.0 pts | — |
| `category` | 100.0% | 100.0% | +0.0 pts | — |
| `order_id` | 100.0% | 100.0% | +0.0 pts | — |
| `priority` | 100.0% | 58.8% | -41.2 pts | — |

**Blocking:**
- task success dropped 41.2 pts (limit 2.0), significant at p=0.0002

<details><summary>33 newly failing cases</summary>

| Case | Failed check: got → expected |
|---|---|
| `t02[channel=email]` | `priority`: 'high' → 'normal' |
| `t02[channel=chat]` | `priority`: 'high' → 'normal' |
| `t03[channel=email]` | `priority`: 'high' → 'normal' |
| `t03[channel=chat]` | `priority`: 'high' → 'normal' |
| `t04[channel=email]` | `priority`: 'high' → 'normal' |
| `t04[channel=chat]` | `priority`: 'high' → 'normal' |
| `t06[channel=email]` | `priority`: 'high' → 'normal' |
| `t06[channel=chat]` | `priority`: 'high' → 'normal' |
| `t08[channel=email]` | `priority`: 'high' → 'normal' |
| `t10[channel=email]` | `priority`: 'high' → 'normal' |
| … 23 more | |

</details>
