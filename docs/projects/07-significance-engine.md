# 07 — Statistical Significance Engine

> Bootstrapping utility that outputs "Model B wins by 3.2% ± 1.1% (p<0.05)" instead of raw
> accuracy scores. A 2-point delta on 50 samples is noise.

## What and why

Leaderboards and eval dashboards report point estimates. Two models at 80.0% and 82.0% look
different, but on a 50-item set that gap is one or two coin flips. `evalkit.significance` turns
per-item results into a verdict sentence a reviewer can act on:

```
ft-v2 beats gpt-base by +4.4 pts ± 2.9 (95% CI [1.5, 7.3], p=0.007, n=500 paired, 50 clusters)
No significant difference between gpt-base and ft-v2: ft-v2 − gpt-base = +2.0 pts (95% CI [-6.0, 12.0], p=1.000, n=50 paired). Smallest effect detectable at this n with 80% power ≈ 12.6 pts.
```

It also answers the question you should ask before running an eval: how many items do I need?

## Design

Input is JSONL, one row per (model, item): `{item_id, model, score[, cluster][, metric]}`.
Scores can be 0/1 (accuracy), in [0, 1] (reported as points ×100), or any real number.
Repeated rows for the same (model, item) are averaged (e.g. several samples per prompt).

| Piece | Where | Notes |
|---|---|---|
| Paired bootstrap CI | `stats.bootstrap_mean` on per-item differences | Percentile and **BCa** (bias-corrected, accelerated via jackknife). Vectorised with multinomial resampling weights, chunked to bound memory. |
| Clustered bootstrap | same function, `clusters=` | Resamples whole clusters (conversation, prompt template, source doc) and uses the ratio-of-sums mean. Correlated items no longer masquerade as independent evidence. |
| Paired permutation test | `stats.paired_permutation_test` | Sign-flip per item, or per cluster when clustered. |
| McNemar exact test | `stats.mcnemar_exact` | Used for 0/1 scores without clusters; exact binomial on discordant pairs via `lgamma`. |
| Unpaired fallback | `bootstrap_unpaired`, `unpaired_permutation_test` | When models share no items. |
| Multiple comparisons | `stats.adjust_pvalues` | Bonferroni, Holm (FWER), Benjamini–Hochberg (FDR). Applied automatically when `compare` is given several candidates and in `leaderboard`. |
| Power analysis | `power_two_proportions`, `required_sample_size`, `minimum_detectable_effect` | Unpaired two-proportion z-test; paired McNemar approximation with a `discordance` (fraction of items the models disagree on). |
| Verdict | `compare.Comparison.verdict()` | "Significant" requires both p < α **and** a CI that excludes 0; when the two disagree the verdict says "borderline". Non-significant verdicts include the MDE at the current n. |
| Leaderboard | `compare.leaderboard` | All pairwise comparisons, corrected together; models that are not significantly worse than a tier's leader share the tier. |

Normal quantiles use Acklam's approximation refined by one Newton step on `math.erf`, so no SciPy.

## How to run

```bash
# Compare two models (positive delta = B better)
uv run --no-sync evalkit significance compare --results results.jsonl --a gpt-base --b ft-v2
# Several candidates against one baseline, Holm-corrected (omit --b for "all others")
uv run --no-sync evalkit significance compare --results results.jsonl --a gpt-base --correction holm
# Leaderboard with significance tiers
uv run --no-sync evalkit significance leaderboard --results results.jsonl
# Sample size / MDE
uv run --no-sync evalkit significance power --baseline 0.8 --delta 0.02 [--paired --discordance 0.08] [--n 500]
# Synthetic data with known true accuracies
uv run --no-sync evalkit significance simulate --model a=0.8 --model b=0.83 --n 500 --clusters 50 --out r.jsonl
```

Flags: `--metric` (score field, or metric name in long-format rows), `--n-boot`, `--confidence`,
`--ci bca|percentile`, `--no-clusters`, `--json`.

From Python:

```python
from evalkit.significance import compare, load_scores
from evalkit.core.trajectory import read_jsonl
c = compare(load_scores(read_jsonl("results.jsonl")), "gpt-base", "ft-v2")
c.significant, c.delta, c.ci.low, c.ci.high, c.p_value, c.verdict()
```

## Sample output

`examples/07-significance-engine/run.sh` (500 items in 50 template clusters, plus a 50-item smoke set):

```
== 1. Is ft-v2 really better than gpt-base? (500 items, 50 prompt templates as clusters)
ft-v2 beats gpt-base by +4.4 pts ± 2.9 (95% CI [1.5, 7.3], p=0.007, n=500 paired, 50 clusters)
  gpt-base: 80.0  ft-v2: 84.4  test=cluster-permutation  ci=bca

== 2. Same question on a 50-item smoke set (a 2-point delta here is noise)
No significant difference between gpt-base and ft-v2: ft-v2 − gpt-base = +2.0 pts (95% CI [-6.0, 12.0], p=1.000, n=50 paired). Smallest effect detectable at this n with 80% power ≈ 12.6 pts.
  gpt-base: 88.0  ft-v2: 90.0  test=mcnemar-exact  ci=bca

== 3. All candidates vs gpt-base, Holm-corrected for 3 comparisons
gpt-base beats distilled by +5.6 pts ± 2.6 (95% CI [2.9, 8.2], p=0.002, n=500 paired, 50 clusters)
ft-v2 beats gpt-base by +4.4 pts ± 2.9 (95% CI [1.5, 7.3], p=0.013, n=500 paired, 50 clusters)
No significant difference between gpt-base and ft-v2-lite: ft-v2-lite − gpt-base = +0.4 pts (95% CI [-3.1, 3.9], p=0.906, n=500 paired, 50 clusters). Smallest effect detectable at this n with 80% power ≈ 5.0 pts.

== 4. Leaderboard with significance tiers
| Rank | Tier | Model | Score | 95% CI | Δ vs top | p (adj) | n |
|---:|---:|---|---:|---|---:|---:|---:|
| 1 | 1 | ft-v2 | 84.4 | [81.2, 87.6] | +0.0 | — | 500 |
| 2 | 2 | ft-v2-lite | 80.4 | [76.8, 83.8] | -4.0 | 0.036 | 500 |
| 3 | 2 | gpt-base | 80.0 | [76.4, 83.4] | -4.4 | 0.020 | 500 |
| 4 | 3 | distilled | 74.4 | [70.6, 78.2] | -10.0 | 0.001 | 500 |

== 5. How many items do we need to see +2 pts on an 80% baseline?
To detect 80.0% → 82.0% (+2.0 pts) with 80% power at α=0.05: n = 6039 items (unpaired, per model).
To detect 80.0% → 82.0% (+2.0 pts) with 80% power at α=0.05: n = 1568 items (paired (McNemar)).
With n=500 (paired (McNemar)), the minimum detectable effect is 3.5 pts at 80% power.
Power to detect +2.0 pts at n=500: 35.2%.
```

## Validation

`tests/test_significance.py` checks, among others:

- BCa and percentile CIs cover the true mean difference of a skewed distribution at ≈95% over 200 simulated datasets;
- the permutation test's false-positive rate under the null is ≈α (2–8.5% over 400 null datasets), and McNemar's is ≤8%;
- McNemar, Holm, BH and Bonferroni against hand-computed values; sample size against the textbook 50%→60% example (≈385/arm);
- a 2-point delta on 50 items is reported as "No significant difference";
- clustered resampling widens the CI (>1.3×) and raises p when whole clusters move together.

## Limitations

- The leaderboard's per-model CI is an item-level percentile bootstrap; only the pairwise tests are cluster-aware.
- BCa on tiny 0/1 samples can be narrower than the exact McNemar test implies; such cases are labelled "borderline" rather than silently declared wins.
- Multiplicity correction adjusts p-values but not CIs (no simultaneous intervals).
- Paired power needs a discordance estimate; without one it assumes independent errors, which overstates the required n for highly correlated models.
- Tiering is greedy (leader-based), not a full compact-letter display.
