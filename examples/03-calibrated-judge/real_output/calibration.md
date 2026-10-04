# Judge calibration (held-out anchors)

Fitted on 300 anchors, evaluated on 200 held-out anchors · judge family `n/a`

| Judge | Agreement | κ | Verbosity excess | Length coef | Self-pref gap | ECE |
|---|---:|---:|---:|---:|---:|---:|
| raw judge (AB order) | 82.0% | 0.698 | -3.1 pts | -1.08 | — | 0.069 |
| swap-aggregated | 81.0% | 0.675 | -4.3 pts | -1.84 | — | 0.093 |
| swap + length | 80.0% | 0.648 | +0.1 pts | -0.30 | — | 0.050 |
| swap + length + self (full) | 80.0% | 0.648 | +0.1 pts | -0.30 | — | 0.050 |

Correction: `P(A wins) = σ(bias=+0.17, judge_logit=+1.85, log_len_ratio=+2.19, self_family=+0.00)`, tie band ±0.01.
Position bias is removed by construction (both orders are averaged); raw consistency on the test split was 84.5%.
