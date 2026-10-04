# Drift monitor report

13 days monitored (2026-09-14 → 2026-09-26), 39 alerts; first alert on **2026-09-21**.

| metric | baseline (first 7d) | latest | change |
|---|---:|---:|---:|
| answer_chars_mean | 388.2 | 272.6 | -29.8% |
| cost_usd_mean | 0.002227 | 0.002086 | -6.3% |
| empty_rate | 0.007143 | 0 | -100.0% |
| judge_cost_usd | 0.003212 | 0.002878 | -10.4% |
| judge_score | 0.9437 | 0.7812 | -17.2% |
| judge_unscored_rate | 0 | 0 | +0.0% |
| latency_p50 | 1.694 | 2.563 | +51.3% |
| latency_p95 | 2.884 | 4.761 | +65.1% |
| refusal_rate | 0.0375 | 0.1875 | +400.0% |
| tool_calls_mean | 1.068 | 1.025 | -4.0% |
| tool_error_rate | 0.04088 | 0.142 | +247.3% |

## Alerts

| date | severity | method | message |
|---|---|---|---|
| 2026-09-21 | warning | zscore | latency_p50 2.058 vs baseline 1.694 (z=+3.2) |
| 2026-09-21 | critical | psi | tool_mix distribution shifted (PSI=0.44; 'browse' 0% -> 9%) |
| 2026-09-22 | warning | zscore | empty_rate 0.0375 vs baseline 0.007143 (z=+3.0) |
| 2026-09-22 | critical | zscore | latency_p50 2.207 vs baseline 1.694 (z=+4.6) |
| 2026-09-22 | critical | zscore | tool_error_rate 0.1636 vs baseline 0.04088 (z=+5.5) |
| 2026-09-22 | critical | psi | tool_mix distribution shifted (PSI=1.10; 'browse' 0% -> 19%) |
| 2026-09-23 | warning | psi | answer_length distribution shifted (PSI=0.20; '250-499' 93% -> 80%) |
| 2026-09-23 | critical | zscore | latency_p50 2.36 vs baseline 1.694 (z=+5.9) |
| 2026-09-23 | warning | cusum | latency_p95 drifting up for 7 days (CUSUM=6.0) |
| 2026-09-23 | warning | cusum | refusal_rate drifting up for 7 days (CUSUM=5.7) |
| 2026-09-23 | warning | cusum | tool_error_rate drifting up for 7 days (CUSUM=6.2) |
| 2026-09-23 | critical | psi | tool_mix distribution shifted (PSI=1.76; 'browse' 0% -> 29%) |
| 2026-09-24 | critical | zscore | answer_chars_mean 306.6 vs baseline 388.2 (z=-4.2) |
| 2026-09-24 | warning | psi | answer_length distribution shifted (PSI=0.23; '250-499' 93% -> 79%) |
| 2026-09-24 | warning | zscore | empty_rate 0.0375 vs baseline 0.007143 (z=+3.0) |
| 2026-09-24 | warning | cusum | judge_score drifting down for 7 days (CUSUM=5.4) |
| 2026-09-24 | critical | zscore | latency_p50 2.583 vs baseline 1.694 (z=+7.9) |
| 2026-09-24 | critical | zscore | latency_p95 4.241 vs baseline 2.884 (z=+4.7) |
| 2026-09-24 | warning | zscore | refusal_rate 0.1125 vs baseline 0.0375 (z=+3.5) |
| 2026-09-24 | critical | zscore | tool_error_rate 0.1724 vs baseline 0.04088 (z=+5.9) |
| 2026-09-24 | critical | psi | tool_mix distribution shifted (PSI=2.40; 'browse' 0% -> 36%) |
| 2026-09-25 | critical | zscore | answer_chars_mean 245 vs baseline 388.2 (z=-7.4) |
| 2026-09-25 | critical | psi | answer_length distribution shifted (PSI=1.10; '250-499' 93% -> 51%) |
| 2026-09-25 | warning | cusum | empty_rate drifting up for 7 days (CUSUM=6.4) |
| 2026-09-25 | critical | zscore | judge_score 0.7344 vs baseline 0.9437 (z=-4.4) |
| 2026-09-25 | critical | zscore | latency_p50 2.79 vs baseline 1.694 (z=+9.8) |
| 2026-09-25 | critical | zscore | latency_p95 5.011 vs baseline 2.884 (z=+7.3) |
| 2026-09-25 | critical | zscore | refusal_rate 0.2 vs baseline 0.0375 (z=+7.7) |
| 2026-09-25 | critical | zscore | tool_error_rate 0.1667 vs baseline 0.04088 (z=+5.7) |
| 2026-09-25 | critical | psi | tool_mix distribution shifted (PSI=2.13; 'browse' 0% -> 33%) |
| 2026-09-26 | critical | zscore | answer_chars_mean 272.6 vs baseline 388.2 (z=-6.0) |
| 2026-09-26 | critical | psi | answer_length distribution shifted (PSI=0.75; '250-499' 93% -> 61%) |
| 2026-09-26 | warning | cusum | empty_rate drifting up for 7 days (CUSUM=5.2) |
| 2026-09-26 | warning | zscore | judge_score 0.7812 vs baseline 0.9437 (z=-3.4) |
| 2026-09-26 | critical | zscore | latency_p50 2.563 vs baseline 1.694 (z=+7.8) |
| 2026-09-26 | critical | zscore | latency_p95 4.761 vs baseline 2.884 (z=+6.5) |
| 2026-09-26 | critical | zscore | refusal_rate 0.1875 vs baseline 0.0375 (z=+7.1) |
| 2026-09-26 | critical | zscore | tool_error_rate 0.142 vs baseline 0.04088 (z=+4.6) |
| 2026-09-26 | critical | psi | tool_mix distribution shifted (PSI=2.60; 'browse' 0% -> 39%) |
