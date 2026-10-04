# Drift monitor report

13 days monitored (2026-09-14 → 2026-09-26), 35 alerts; first alert on **2026-09-21**.

| metric | baseline (first 7d) | latest | change |
|---|---:|---:|---:|
| answer_chars_mean | 604.6 | 304 | -49.7% |
| cost_usd_mean | 0.002486 | 0.002133 | -14.2% |
| empty_rate | 0.005357 | 0.0375 | +600.0% |
| judge_cost_usd | 0.007128 | 0.005573 | -21.8% |
| judge_score | 0.003125 | 0.009375 | +200.0% |
| judge_unscored_rate | 0 | 0 | +0.0% |
| latency_p50 | 1.732 | 2.563 | +48.0% |
| latency_p95 | 3.05 | 3.762 | +23.3% |
| refusal_rate | 0.03393 | 0.15 | +342.1% |
| tool_calls_mean | 1.137 | 1.05 | -7.7% |
| tool_error_rate | 0.05766 | 0.1603 | +177.9% |

## Alerts

| date | severity | method | message |
|---|---|---|---|
| 2026-09-21 | warning | zscore | answer_chars_mean 510.7 vs baseline 604.6 (z=-3.1) |
| 2026-09-21 | warning | psi | tool_mix distribution shifted (PSI=0.27; 'sql' 27% -> 19%) |
| 2026-09-22 | warning | zscore | latency_p50 2.221 vs baseline 1.739 (z=+3.8) |
| 2026-09-22 | warning | zscore | latency_p95 3.983 vs baseline 3.092 (z=+3.4) |
| 2026-09-22 | critical | psi | tool_mix distribution shifted (PSI=0.50; 'browse' 1% -> 15%) |
| 2026-09-23 | critical | zscore | answer_chars_mean 398.1 vs baseline 592 (z=-4.9) |
| 2026-09-23 | critical | psi | answer_length distribution shifted (PSI=0.50; '500-999' 48% -> 28%) |
| 2026-09-23 | critical | zscore | latency_p50 2.816 vs baseline 1.739 (z=+8.5) |
| 2026-09-23 | critical | zscore | latency_p95 4.294 vs baseline 3.092 (z=+4.6) |
| 2026-09-23 | critical | zscore | refusal_rate 0.15 vs baseline 0.02857 (z=+6.5) |
| 2026-09-23 | warning | zscore | tool_error_rate 0.1323 vs baseline 0.05161 (z=+3.3) |
| 2026-09-23 | critical | psi | tool_mix distribution shifted (PSI=0.81; 'browse' 1% -> 22%) |
| 2026-09-24 | critical | zscore | answer_chars_mean 366.4 vs baseline 592 (z=-5.7) |
| 2026-09-24 | critical | psi | answer_length distribution shifted (PSI=0.81; '500-999' 48% -> 21%) |
| 2026-09-24 | critical | zscore | latency_p50 2.565 vs baseline 1.739 (z=+6.6) |
| 2026-09-24 | critical | zscore | latency_p95 4.371 vs baseline 3.092 (z=+4.9) |
| 2026-09-24 | critical | zscore | refusal_rate 0.1125 vs baseline 0.02857 (z=+4.5) |
| 2026-09-24 | critical | zscore | tool_error_rate 0.1901 vs baseline 0.05161 (z=+5.6) |
| 2026-09-24 | critical | psi | tool_mix distribution shifted (PSI=1.76; 'browse' 1% -> 39%) |
| 2026-09-25 | critical | zscore | answer_chars_mean 345.7 vs baseline 592 (z=-6.2) |
| 2026-09-25 | critical | psi | answer_length distribution shifted (PSI=0.85; '500-999' 48% -> 21%) |
| 2026-09-25 | warning | cusum | empty_rate drifting up for 7 days (CUSUM=6.2) |
| 2026-09-25 | critical | zscore | latency_p50 2.966 vs baseline 1.739 (z=+9.7) |
| 2026-09-25 | critical | zscore | latency_p95 5.499 vs baseline 3.092 (z=+9.3) |
| 2026-09-25 | critical | zscore | refusal_rate 0.15 vs baseline 0.02857 (z=+6.5) |
| 2026-09-25 | warning | cusum | tool_error_rate drifting up for 7 days (CUSUM=10.4) |
| 2026-09-25 | critical | psi | tool_mix distribution shifted (PSI=2.39; 'browse' 1% -> 50%) |
| 2026-09-26 | critical | zscore | answer_chars_mean 304 vs baseline 592 (z=-7.2) |
| 2026-09-26 | critical | psi | answer_length distribution shifted (PSI=1.39; '500-999' 48% -> 12%) |
| 2026-09-26 | warning | cusum | empty_rate drifting up for 7 days (CUSUM=8.1) |
| 2026-09-26 | critical | zscore | latency_p50 2.563 vs baseline 1.739 (z=+6.5) |
| 2026-09-26 | warning | cusum | latency_p95 drifting up for 7 days (CUSUM=23.3) |
| 2026-09-26 | critical | zscore | refusal_rate 0.15 vs baseline 0.02857 (z=+6.5) |
| 2026-09-26 | critical | zscore | tool_error_rate 0.1603 vs baseline 0.05161 (z=+4.4) |
| 2026-09-26 | critical | psi | tool_mix distribution shifted (PSI=1.43; 'browse' 1% -> 35%) |
