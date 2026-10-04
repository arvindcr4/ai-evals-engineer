# Drift monitor report

30 days monitored (2026-09-01 → 2026-09-30), 80 alerts; first alert on **2026-09-14**.

| metric | baseline (first 7d) | latest | change |
|---|---:|---:|---:|
| answer_chars_mean | 612.5 | 293.5 | -52.1% |
| cost_usd_mean | 0.002448 | 0.002213 | -9.6% |
| empty_rate | 0.003081 | 0.0303 | +883.5% |
| judge_cost_usd | 0.00479 | 0.003564 | -25.6% |
| judge_score | 0.9628 | 0.7247 | -24.7% |
| judge_unscored_rate | 0 | 0 | +0.0% |
| latency_p50 | 1.723 | 2.964 | +72.0% |
| latency_p95 | 2.971 | 5.061 | +70.3% |
| refusal_rate | 0.02686 | 0.1919 | +614.6% |
| tool_calls_mean | 1.061 | 1.202 | +13.3% |
| tool_error_rate | 0.02688 | 0.2569 | +856.0% |

## Alerts

| date | severity | method | message |
|---|---|---|---|
| 2026-09-14 | warning | zscore | refusal_rate 0.09278 vs baseline 0.02934 (z=+3.7) |
| 2026-09-21 | warning | zscore | refusal_rate 0.09259 vs baseline 0.03201 (z=+3.0) |
| 2026-09-21 | critical | psi | tool_mix distribution shifted (PSI=0.34; 'sql' 29% -> 20%) |
| 2026-09-22 | critical | zscore | answer_chars_mean 474.2 vs baseline 612 (z=-4.5) |
| 2026-09-22 | warning | psi | answer_length distribution shifted (PSI=0.29; '500-999' 50% -> 38%) |
| 2026-09-22 | critical | zscore | latency_p50 2.236 vs baseline 1.73 (z=+4.8) |
| 2026-09-22 | warning | zscore | latency_p95 4.004 vs baseline 3.016 (z=+3.7) |
| 2026-09-22 | warning | zscore | tool_error_rate 0.1127 vs baseline 0.04397 (z=+3.4) |
| 2026-09-22 | critical | psi | tool_mix distribution shifted (PSI=1.09; 'browse' 0% -> 16%) |
| 2026-09-23 | critical | zscore | answer_chars_mean 413.3 vs baseline 612 (z=-6.5) |
| 2026-09-23 | critical | psi | answer_length distribution shifted (PSI=0.50; '500-999' 50% -> 30%) |
| 2026-09-23 | warning | cusum | empty_rate drifting up for 7 days (CUSUM=5.7) |
| 2026-09-23 | critical | zscore | latency_p50 2.7 vs baseline 1.73 (z=+9.3) |
| 2026-09-23 | critical | zscore | latency_p95 4.161 vs baseline 3.016 (z=+4.3) |
| 2026-09-23 | critical | zscore | refusal_rate 0.1237 vs baseline 0.03201 (z=+4.6) |
| 2026-09-23 | warning | zscore | tool_error_rate 0.1228 vs baseline 0.04397 (z=+3.8) |
| 2026-09-23 | critical | psi | tool_mix distribution shifted (PSI=1.47; 'browse' 0% -> 22%) |
| 2026-09-24 | critical | zscore | answer_chars_mean 349.4 vs baseline 612 (z=-8.6) |
| 2026-09-24 | critical | psi | answer_length distribution shifted (PSI=1.13; '500-999' 50% -> 19%) |
| 2026-09-24 | warning | cusum | empty_rate drifting up for 7 days (CUSUM=7.4) |
| 2026-09-24 | warning | zscore | judge_score 0.7812 vs baseline 0.9542 (z=-3.6) |
| 2026-09-24 | critical | zscore | latency_p50 2.596 vs baseline 1.73 (z=+8.3) |
| 2026-09-24 | critical | zscore | latency_p95 4.261 vs baseline 3.016 (z=+4.7) |
| 2026-09-24 | critical | zscore | refusal_rate 0.1583 vs baseline 0.03201 (z=+6.3) |
| 2026-09-24 | critical | zscore | tool_error_rate 0.1762 vs baseline 0.04397 (z=+7.1) |
| 2026-09-24 | critical | psi | tool_mix distribution shifted (PSI=2.67; 'browse' 0% -> 36%) |
| 2026-09-25 | critical | zscore | answer_chars_mean 339.2 vs baseline 612 (z=-8.9) |
| 2026-09-25 | critical | psi | answer_length distribution shifted (PSI=1.09; '500-999' 50% -> 20%) |
| 2026-09-25 | warning | zscore | empty_rate 0.04124 vs baseline 0.002441 (z=+3.9) |
| 2026-09-25 | warning | zscore | judge_score 0.768 vs baseline 0.9542 (z=-3.9) |
| 2026-09-25 | critical | zscore | latency_p50 2.874 vs baseline 1.73 (z=+11.0) |
| 2026-09-25 | critical | zscore | latency_p95 5.348 vs baseline 3.016 (z=+8.8) |
| 2026-09-25 | critical | zscore | refusal_rate 0.1546 vs baseline 0.03201 (z=+6.1) |
| 2026-09-25 | critical | zscore | tool_error_rate 0.1446 vs baseline 0.04397 (z=+4.8) |
| 2026-09-25 | critical | psi | tool_mix distribution shifted (PSI=3.85; 'browse' 0% -> 49%) |
| 2026-09-26 | critical | zscore | answer_chars_mean 288.7 vs baseline 612 (z=-10.6) |
| 2026-09-26 | critical | psi | answer_length distribution shifted (PSI=1.67; '500-999' 50% -> 12%) |
| 2026-09-26 | warning | zscore | empty_rate 0.03846 vs baseline 0.002441 (z=+3.6) |
| 2026-09-26 | critical | zscore | judge_score 0.7524 vs baseline 0.9542 (z=-4.2) |
| 2026-09-26 | critical | zscore | latency_p50 2.601 vs baseline 1.73 (z=+8.4) |
| 2026-09-26 | critical | zscore | latency_p95 4.416 vs baseline 3.016 (z=+5.3) |
| 2026-09-26 | critical | zscore | refusal_rate 0.1731 vs baseline 0.03201 (z=+7.0) |
| 2026-09-26 | critical | zscore | tool_error_rate 0.1737 vs baseline 0.04397 (z=+6.5) |
| 2026-09-26 | critical | psi | tool_mix distribution shifted (PSI=3.15; 'browse' 0% -> 41%) |
| 2026-09-27 | critical | zscore | answer_chars_mean 306.2 vs baseline 612 (z=-10.0) |
| 2026-09-27 | critical | psi | answer_length distribution shifted (PSI=1.64; '500-999' 50% -> 9%) |
| 2026-09-27 | warning | zscore | empty_rate 0.03409 vs baseline 0.002441 (z=+3.2) |
| 2026-09-27 | warning | zscore | judge_score 0.7955 vs baseline 0.9542 (z=-3.3) |
| 2026-09-27 | critical | zscore | latency_p50 2.822 vs baseline 1.73 (z=+10.5) |
| 2026-09-27 | critical | zscore | latency_p95 4.83 vs baseline 3.016 (z=+6.8) |
| 2026-09-27 | critical | zscore | refusal_rate 0.1364 vs baseline 0.03201 (z=+5.2) |
| 2026-09-27 | critical | zscore | tool_error_rate 0.2083 vs baseline 0.04397 (z=+7.5) |
| 2026-09-27 | critical | psi | tool_mix distribution shifted (PSI=3.18; 'browse' 0% -> 41%) |
| 2026-09-28 | critical | zscore | answer_chars_mean 299.2 vs baseline 612 (z=-10.2) |
| 2026-09-28 | critical | psi | answer_length distribution shifted (PSI=1.68; '500-999' 50% -> 11%) |
| 2026-09-28 | warning | cusum | empty_rate drifting up for 7 days (CUSUM=14.7) |
| 2026-09-28 | warning | zscore | judge_score 0.8083 vs baseline 0.9542 (z=-3.1) |
| 2026-09-28 | critical | zscore | latency_p50 3.072 vs baseline 1.73 (z=+12.9) |
| 2026-09-28 | critical | zscore | latency_p95 4.556 vs baseline 3.016 (z=+5.8) |
| 2026-09-28 | critical | zscore | refusal_rate 0.1553 vs baseline 0.03201 (z=+6.1) |
| 2026-09-28 | critical | zscore | tool_error_rate 0.1618 vs baseline 0.04397 (z=+5.8) |
| 2026-09-28 | critical | psi | tool_mix distribution shifted (PSI=3.15; 'browse' 0% -> 41%) |
| 2026-09-29 | critical | zscore | answer_chars_mean 270.9 vs baseline 612 (z=-11.1) |
| 2026-09-29 | critical | psi | answer_length distribution shifted (PSI=1.86; '500-999' 50% -> 10%) |
| 2026-09-29 | critical | zscore | empty_rate 0.04545 vs baseline 0.002441 (z=+4.3) |
| 2026-09-29 | critical | zscore | judge_score 0.6795 vs baseline 0.9542 (z=-5.8) |
| 2026-09-29 | critical | zscore | latency_p50 3.029 vs baseline 1.73 (z=+12.5) |
| 2026-09-29 | critical | zscore | latency_p95 5.458 vs baseline 3.016 (z=+9.2) |
| 2026-09-29 | critical | zscore | refusal_rate 0.2455 vs baseline 0.03201 (z=+10.6) |
| 2026-09-29 | critical | zscore | tool_error_rate 0.1496 vs baseline 0.04397 (z=+5.4) |
| 2026-09-29 | critical | psi | tool_mix distribution shifted (PSI=3.49; 'browse' 0% -> 45%) |
| 2026-09-30 | critical | zscore | answer_chars_mean 293.5 vs baseline 612 (z=-10.4) |
| 2026-09-30 | critical | psi | answer_length distribution shifted (PSI=1.47; '500-999' 50% -> 13%) |
| 2026-09-30 | warning | cusum | empty_rate drifting up for 7 days (CUSUM=17.2) |
| 2026-09-30 | critical | zscore | judge_score 0.7247 vs baseline 0.9542 (z=-4.8) |
| 2026-09-30 | critical | zscore | latency_p50 2.964 vs baseline 1.73 (z=+11.8) |
| 2026-09-30 | critical | zscore | latency_p95 5.061 vs baseline 3.016 (z=+7.7) |
| 2026-09-30 | critical | zscore | refusal_rate 0.1919 vs baseline 0.03201 (z=+8.0) |
| 2026-09-30 | critical | zscore | tool_error_rate 0.2569 vs baseline 0.04397 (z=+10.3) |
| 2026-09-30 | critical | psi | tool_mix distribution shifted (PSI=2.98; 'browse' 0% -> 39%) |
