# Judge bias audit

500 human-labelled anchors · judge family `n/a`

| Measure | Value | Ideal |
|---|---:|---:|
| Agreement with humans (3-way) | 79.6% | high |
| Agreement on decisive pairs | 93.5% | high |
| Cohen's κ | 0.657 | → 1 |
| Position consistency (same verdict after swap) | 85.0% | 100% |
| First-slot win rate | 55.0% | 50% |
| Judge prefers longer | 47.5% | = human (51.0%) |
| Verbosity excess over humans | -3.5 pts | 0 |
| Length logit coef (controls for human pref) | -1.31 | 0 |
| Self-preference: own-family win rate, judge vs human | — vs — (n=0) | equal |
| Self-preference gap | — | 0 |
| Confidence ECE | 0.057 | 0 |
| Agreement after swap-aggregation | 80.8% | — |
| Pointwise score vs human: Pearson / Spearman | 0.780 / 0.840 | → 1 |
| Pointwise length slope (points per log-word, human-controlled) | -1.46 | 0 |

## Verbosity by length gap

| |log len ratio| | n | Judge prefers longer | Humans prefer longer |
|---|---:|---:|---:|
| [0, 0.2) | 114 | 50.0% | 50.0% |
| [0.2, 0.5) | 155 | 45.2% | 45.8% |
| [0.5, 1) | 123 | 49.6% | 56.1% |
| [1, inf) | 10 | 50.0% | 80.0% |

## Confidence reliability

| Confidence bin | n | Stated | Actual |
|---|---:|---:|---:|
| 0.60–0.65 | 39 | 60.0% | 76.9% |
| 0.70–0.75 | 34 | 70.0% | 88.2% |
| 0.80–0.85 | 45 | 83.8% | 91.1% |
| 0.90–0.95 | 94 | 90.0% | 93.6% |
| 0.95–1.00 | 193 | 96.1% | 97.9% |
