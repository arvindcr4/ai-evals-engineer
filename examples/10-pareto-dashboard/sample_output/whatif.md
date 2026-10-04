# What-if: s2-frontier price × 0.25

What-if: **s2-frontier price × 0.25**. Dotted line in the dashboard = frontier before the change.

- **globex-support**: joined frontier ['s2-frontier']; left ['s2-large+reasoning']; cost/success s2-frontier $0.0578 → $0.0144
- **initech-finance**: joined frontier —; left ['s2-large+reasoning']; cost/success s2-frontier $0.1146 → $0.0286
- **acme-legal**: joined frontier —; left ['s2-large+reasoning']; cost/success s2-frontier $0.2914 → $0.0729
- **umbrella-health**: joined frontier ['s2-frontier']; left —; cost/success s2-frontier $0.1724 → $0.0431

## globex-support

- best quality: **s2-frontier** (96.7%)
- value pick (≤2pp off best): s2-frontier · nothing cheaper is close
- router `s1-mini+rag` → `s2-large+reasoning` at threshold 0.80: 95.0% success, 32% escalated, $0.0072/task vs $0.0175 always-escalate

| config | success (95% CI) | cost/task | cost/success | p95 latency | frontier |
|---|---:|---:|---:|---:|:---:|
| s1-nano | 61.7% (49%–73%) | $0.0002 | $0.0004 | 1.2s | ● |
| s1-mini | 75.0% (63%–84%) | $0.0005 | $0.0006 | 1.7s | ● |
| s1-mini+rag | 88.3% (78%–94%) | $0.0009 | $0.0011 | 3.5s | ● |
| s2-large | 93.3% (84%–97%) | $0.0079 | $0.0085 | 3.6s | ● |
| s2-frontier | 96.7% (89%–99%) | $0.0140 | $0.0144 | 13.0s | ● |
| s2-legacy | 88.3% (78%–94%) | $0.0146 | $0.0165 | 6.7s |  |
| s2-large+reasoning | 96.7% (89%–99%) | $0.0166 | $0.0171 | 15.6s |  |

## initech-finance

- best quality: **s2-frontier** (88.3%)
- value pick (≤2pp off best): s2-frontier · nothing cheaper is close
- router `s1-mini+rag` → `s2-large+reasoning`: no threshold stays within 2pp of always-escalate at lower cost

| config | success (95% CI) | cost/task | cost/success | p95 latency | frontier |
|---|---:|---:|---:|---:|:---:|
| s1-nano | 45.0% (33%–58%) | $0.0004 | $0.0008 | 1.3s | ● |
| s1-mini | 58.3% (46%–70%) | $0.0009 | $0.0015 | 1.8s | ● |
| s1-mini+rag | 73.3% (61%–83%) | $0.0018 | $0.0024 | 2.7s | ● |
| s2-large | 85.0% (74%–92%) | $0.0145 | $0.0171 | 4.2s | ● |
| s2-frontier | 88.3% (78%–94%) | $0.0253 | $0.0286 | 10.5s | ● |
| s2-legacy | 68.3% (56%–79%) | $0.0269 | $0.0393 | 6.3s |  |
| s2-large+reasoning | 86.7% (76%–93%) | $0.0277 | $0.0320 | 14.1s |  |

## acme-legal

- best quality: **s2-frontier** (75.0%)
- value pick (≤2pp off best): s2-frontier · nothing cheaper is close
- router `s1-mini+rag` → `s2-large+reasoning` at threshold 0.65: 61.7% success, 77% escalated, $0.0464/task vs $0.0590 always-escalate

| config | success (95% CI) | cost/task | cost/success | p95 latency | frontier |
|---|---:|---:|---:|---:|:---:|
| s1-nano | 21.7% (13%–34%) | $0.0008 | $0.0037 | 1.3s | ● |
| s1-mini | 33.3% (23%–46%) | $0.0019 | $0.0057 | 1.8s | ● |
| s1-mini+rag | 41.7% (30%–54%) | $0.0041 | $0.0098 | 3.0s | ● |
| s2-large | 55.0% (42%–67%) | $0.0315 | $0.0572 | 3.9s | ● |
| s2-frontier | 75.0% (63%–84%) | $0.0546 | $0.0729 | 11.4s | ● |
| s2-large+reasoning | 63.3% (51%–74%) | $0.0549 | $0.0867 | 15.2s |  |
| s2-legacy | 43.3% (32%–56%) | $0.0590 | $0.136 | 5.8s |  |

## umbrella-health

- best quality: **s2-large+reasoning** (81.7%)
- value pick (≤2pp off best): s2-large+reasoning · nothing cheaper is close
- router `s1-mini+rag` → `s2-large+reasoning` at threshold 0.75: 81.7% success, 78% escalated, $0.0285/task vs $0.0366 always-escalate

| config | success (95% CI) | cost/task | cost/success | p95 latency | frontier |
|---|---:|---:|---:|---:|:---:|
| s1-nano | 18.3% (11%–30%) | $0.0005 | $0.0026 | 1.3s | ● |
| s1-mini | 35.0% (24%–48%) | $0.0011 | $0.0032 | 1.8s | ● |
| s1-mini+rag | 46.7% (35%–59%) | $0.0023 | $0.0050 | 2.5s | ● |
| s2-large | 55.0% (42%–67%) | $0.0187 | $0.0341 | 4.3s | ● |
| s2-frontier | 73.3% (61%–83%) | $0.0316 | $0.0431 | 10.8s | ● |
| s2-large+reasoning | 81.7% (70%–89%) | $0.0343 | $0.0420 | 16.2s | ● |
| s2-legacy | 41.7% (30%–54%) | $0.0346 | $0.0830 | 5.8s |  |
