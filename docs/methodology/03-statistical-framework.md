# Stop reporting raw accuracy

*Methodology teardown, part 3: the statistical framework behind every number evalkit prints.*

"ft-v2 scores 84.4, gpt-base scores 80.0" is not a result. It is two point estimates with no error bars, no notion of which items they share, and no statement of how many other comparisons were run that day.

This post covers the statistics evalkit applies where a number becomes a decision: comparing models (`src/evalkit/significance/`), blocking merges (`src/evalkit/regression_gate/`), drift alerts (`src/evalkit/drift_monitor/`), cost-quality choices (`src/evalkit/pareto/`), and whether a benchmark number is valid at all (`src/evalkit/contamination/`). Every number comes from re-running the committed examples (commands at the end).

**About the data.** Nothing here is production data. Every example is offline and synthetic: the model results (`gpt-base`, `ft-v2`, …) are a committed file of 0/1 scores, not outputs of real models; the regression gate runs a toy triage function; the drift traffic comes from a generator with a planted decay and is scored by a deterministic mock judge; the Pareto results come from `evalkit pareto generate`; the contamination corpus has planted leaks. The numbers show how the statistics behave, not how any real model performs.

## 1. The unit of analysis is the item, and sometimes the cluster

Two models scored on the same 500 items are **not** two independent samples: a hard item is hard for both. Comparing two marginal proportions uses the wrong variance. Pairing removes the shared difficulty (section 5: ~2.3x fewer items at the discordance we measured, ~3.9x under an optimistic assumption).

Items are also not always independent of *each other*. In `examples/07-significance-engine/results.jsonl` the 500 items carry 50 cluster labels standing in for prompt templates. Items from one template share failure modes, so they are not 500 independent pieces of evidence. The engine resamples **clusters** rather than items whenever a `cluster` column is present. Paired, clustered and plain bootstraps all run on one engine, `_boot_means` in `src/evalkit/significance/stats.py`:

```python
w = rng.multinomial(g, probs, size=stop - start).astype(float)
out[start:stop] = (w @ sums) / (w @ counts)
```

Each bootstrap replicate draws multinomial weights over *g* clusters and computes a ratio-of-sums mean. An unclustered bootstrap is the special case where every item is its own cluster. A paired CI is the same call applied to per-item differences `xb - xa`.

### Worked example: ft-v2 vs gpt-base

```
$ evalkit significance compare --results results.jsonl --a gpt-base --b ft-v2
ft-v2 beats gpt-base by +4.4 pts ± 2.9 (95% CI [1.5, 7.3], p=0.007, n=500 paired, 50 clusters)
  gpt-base: 80.0  ft-v2: 84.4  test=cluster-permutation  ci=bca
```

The same data with `--no-clusters`:

```
ft-v2 beats gpt-base by +4.4 pts ± 3.2 (95% CI [1.2, 7.6], p=0.010, n=500 paired)
  gpt-base: 80.0  ft-v2: 84.4  test=mcnemar-exact  ci=bca
```

Surprise: clustering made the interval *narrower* (bootstrap SE 0.0148 vs 0.0163: an SE ratio of ≈0.91, a variance design effect of ≈0.83). Clustering widens intervals only when clusters carry correlated *differences*; here the template effect shifts both models and mostly cancels in the paired difference. To check, we ran the simulator with and without a per-cluster shift (`simulate_results(..., n_clusters=50, cluster_sd=σ, seed=3)`):

| cluster_sd | clustered CI | unclustered CI | clustered p | unclustered p |
|---|---|---|---|---|
| 0.0 | [2.3, 7.5] | [2.0, 7.8] | 0.0014 | 0.0022 |
| 1.0 | [1.0, 8.0] | [1.4, 7.8] | 0.0189 | 0.0067 |

With real cluster correlation, the unclustered analysis overstates the evidence by almost 3x in p-value (one seeded draw, so read it as an illustration, not a rate). **Our rule: if you have a natural grouping (conversation, template, source document, parameter variant of one item), declare it.**

## 2. Intervals: BCa by default

The default interval is bias-corrected and accelerated (BCa). The bias term `z0` is the normal quantile of the share of replicates below the point estimate, with ties counted half. The acceleration `a` comes from a **cluster-deletion jackknife** (`_bca_bounds` in `stats.py`):

```python
dev = jack.mean() - jack
denom = 6.0 * (np.sum(dev**2) ** 1.5)
acc = float(np.sum(dev**3) / denom) if denom > 0 else 0.0
...
adj = norm_cdf(z0 + (z0 + z) / (1 - acc * (z0 + z)))
```

Normal quantiles are inline (Acklam plus one Newton step; no scipy). On well-behaved data BCa and percentile agree: `--ci percentile` gives the identical [1.5, 7.3]. BCa earns its keep on skewed statistics near 0% or 100%, and has a failure mode covered in section 4.

## 3. Tests: McNemar when you can, permutation otherwise

`compare()` in `src/evalkit/significance/compare.py` picks the test from the data:

| Design | Test | Why |
|---|---|---|
| paired, 0/1 scores, no clusters | exact McNemar | only discordant items carry information; exact binomial tail via `lgamma` |
| paired, clusters or continuous scores | sign-flip permutation over clusters (or items) | under H0, A/B labels are exchangeable within each cluster |
| no shared items | label-shuffle permutation + unpaired percentile bootstrap | fallback; you lose pairing's variance reduction |

The McNemar run (`--no-clusters --json`) reports `a_only_correct: 23, b_only_correct: 45`: 68 discordant items out of 500. The other 432 say nothing about which model is better.

**A significant result needs both conditions at once:**

```python
@property
def significant(self) -> bool:
    return self.p_value < self.alpha and self.ci_excludes_zero
```

When the test and the bootstrap CI disagree, the verdict string says `(borderline: test and CI disagree)` and declares no winner. We prefer a loud "we don't know" to a quiet coin flip.

## 4. Multiple comparisons

Three candidates against a baseline is three tests. The CLI corrects automatically whenever more than one candidate is compared (several `--b`, or none, which means all others). Here are the p-values under each correction (from `evalkit significance compare --results results.jsonl --a gpt-base --correction <method>`):

| Correction | distilled vs base | ft-v2 vs base | ft-v2-lite vs base |
|---|---|---|---|
| none | <0.001 | 0.007 | 0.906 |
| Holm (default) | 0.002 | 0.013 | 0.906 |
| Benjamini–Hochberg | 0.002 | 0.010 | 0.906 |
| Bonferroni | 0.002 | 0.020 | 1.000 |

Holm is the default: it controls family-wise error like Bonferroni and is never less powerful. BH is for large sweeps where FDR is the right target.

The leaderboard Holm-corrects all pairwise tests together and groups models into **tiers**: a model stays in the current tier unless significantly worse than its leader:

| Rank | Tier | Model | Score | 95% CI | Δ vs top | p (adj) |
|---:|---:|---|---:|---|---:|---:|
| 1 | 1 | ft-v2 | 84.4 | [81.2, 87.6] | +0.0 | — |
| 2 | 2 | ft-v2-lite | 80.4 | [76.8, 83.8] | -4.0 | 0.036 |
| 3 | 2 | gpt-base | 80.0 | [76.4, 83.4] | -4.4 | 0.020 |
| 4 | 3 | distilled | 74.4 | [70.6, 78.2] | -10.0 | 0.001 |

Rows 1 and 3: the per-model CIs overlap ([81.2, 87.6] vs [76.4, 83.4]), yet the paired test separates them at p=0.020. **"The error bars overlap, so it's a tie" is wrong for paired data.**

### What we got wrong

- **The leaderboard's per-model CIs are unclustered percentile intervals** (`bootstrap_mean(scores, None, ..., "percentile", ...)` in `leaderboard()`). They ignore the template clustering that the pairwise tests respect. They should take the cluster column too.
- **Corrected p-values travel with uncorrected 95% CIs.** After Holm, ft-v2's p becomes 0.013 but its CI stays at the 95% level. A simultaneous (e.g. Bonferroni-adjusted 98.3%) interval would be consistent with the test. Today the `significant` check can disagree with itself under correction.
- **BCa misbehaves on near-degenerate data.** In the regression gate's v1.1 run, exactly one item (two cases, one cluster) flipped from fail to pass. BCa reported `CI [0.0, 12.5]`. Re-running that exact vector (`[1,1]+[0]*78`, 40 clusters) through `bootstrap_mean` gives a percentile interval of [0.0, 7.5]. The bootstrap distribution has only seven distinct values, and the jackknife is dominated by the one non-zero cluster: acceleration a≈0.16 with z0≈0.12 moves the upper quantile from 0.975 to 0.9994, which lands on the rare 12.5 replicates (17 of 5,000). We would fall back to percentile when fewer than about 5 clusters have a nonzero difference.

## 5. Power and MDE: decide n before you run

A more common mistake than a wrong test is a suite that could never have detected the effect anyone cared about. `evalkit significance power` answers this up front:

```
$ evalkit significance power --baseline 0.8 --delta 0.02
n = 6039 items (unpaired, per model)
$ evalkit significance power --baseline 0.8 --delta 0.02 --paired --discordance 0.08 --n 500
n = 1568 items (paired (McNemar))
With n=500 (paired (McNemar)), the minimum detectable effect is 3.5 pts at 80% power.
Power to detect +2.0 pts at n=500: 35.2%.
```

Paired power uses the McNemar normal approximation from `power_two_proportions`:

```python
z = (delta * math.sqrt(n) - za * math.sqrt(psi)) / math.sqrt(psi - delta**2)
```

Here `psi` is the **discordance rate**, the fraction of items where the models disagree, and it drives the whole calculation:

| Discordance ψ | n to detect 80→82% (paired) | Source |
|---|---:|---|
| default: p1(1−p2)+p2(1−p1) = 0.308 | 6042 | independent errors, worst case |
| 0.136 | 2667 | observed in the ft-v2 run (68/500) |
| 0.08 | 1568 | an optimistic, highly correlated pair |

The default ψ assumes independent errors, giving essentially the *unpaired* answer (6042 vs 6039). Deliberately: no measured discordance, no pairing discount. At the observed ψ=0.136, n=500 had 76.2% power for the +4.4 points we found and 22.7% for +2. The 50-item smoke set (`small_results.jsonl`):

```
No significant difference between gpt-base and ft-v2: ft-v2 − gpt-base = +2.0 pts
(95% CI [-6.0, 12.0], p=1.000, n=50 paired). Smallest effect detectable at this n
with 80% power ≈ 12.6 pts.
```

Two vs three discordant items. Every "no difference" verdict prints its MDE so nobody reads "not significant" as "equivalent".

## 6. The CI gate: effect size AND significance

The regression gate (`src/evalkit/regression_gate/gate.py`) blocks a merge on task success only when two conditions hold. The drop must exceed a practical threshold, and it must be significant on a paired test clustered by dataset item:

```python
drop = -comparison.delta
if drop > policy.max_success_drop:
    if not policy.require_significance:
        reasons.append(...)
    elif comparison.significant:
        reasons.append(...)
```

Clustering by `item_id` matters here because the suite is a parameter matrix. `examples/04-regression-gate/suite.yaml` crosses 40 tickets with `tier: [fast, accurate]`, giving 80 cases but only 40 clusters. Two parameter variants of one ticket are not two pieces of evidence. The policy in that suite is:

```yaml
gate:
  max_success_drop: 0.02
  alpha: 0.05
  require_significance: true
  max_p95_latency_increase: 0.25
  min_latency_increase_ms: 2
```

Latency has its own two-part rule: p95 must rise by more than 25% *and* by at least `min_latency_increase_ms`, so jitter under 2 ms on a ~4 ms baseline cannot fail a build. Results across the four builds (p95 is wall-clock on a toy function and moves by ±0.1 ms between runs):

| Build | Success | Δ | Gate stats | p95 | Verdict |
|---|---:|---:|---|---|---|
| v1 (unchanged) | 86.2% | +0.0 | CI [0.0, 0.0], p=1.000 | 4.2 ms | PASS |
| v1.1 (fixes one item) | 88.8% | +2.5 | CI [0.0, 12.5], p=1.000 | 4.1 ms | PASS |
| v2 (bad refactor) | 66.2% | −20.0 | drop CI [10.0, 35.0], p=0.006 | 4.2 ms | **FAIL** (exit 1) |
| v3 (3x slower) | 86.2% | +0.0 | — | 4.2 → 12.2 ms (+192%) | **FAIL** (exit 1) |

v1.1's p=1.000 is correct: with one cluster carrying all the signal, every sign flip gives the same |mean|. Likewise v2's 16 newly failing cases come in pairs (`t01[tier=fast]`, `t01[tier=accurate]`, …) from 8 tickets; case-level resampling would double-count them.

### What we'd change

**The gate is underpowered against small regressions, and it fails open.** `require_significance: true` means a drop that is real but not significant *passes*. At n=80 cases (40 independent items) and baseline 86.2%:

```
$ evalkit significance power --baseline 0.862 --delta -0.05 --paired --n 80
With n=80 (paired (McNemar)), the minimum detectable effect is 12.0 pts at 80% power.
Power to detect -5.0 pts at n=80: 13.3%.
$ evalkit significance power --baseline 0.862 --delta -0.05 --paired --n 80 --discordance 0.05
With n=80 (paired (McNemar)), the minimum detectable effect is 9.5 pts at 80% power.
Power to detect -5.0 pts at n=80: 51.6%.
```

(These use the McNemar approximation, not the cluster-permutation test the gate actually runs, and the MDE is computed for an upward change.) The 2-point threshold is largely decorative. With the default independent-errors discordance, a 5-point regression passes ~87% of the time; even with an optimistic ψ=0.05 it passes about half the time. Both figures treat the 80 cases as independent. At the honest n=40 the power drops to 8.7% and 28.8%. The fix is **non-inferiority**: pass only if the CI's bound on Δ sits above −`max_success_drop`. That fails *closed* on a too-small suite and forces the suite-size conversation up front. We would also go one-sided; the gate only cares about drops but spends α on both tails.

## 7. Drift: z-score, CUSUM, PSI, and the false-alarm budget

Production monitoring has no paired items, just a daily hash-sampled slice of traffic. The example uses synthetic traffic (5% of 2,000 requests per day, so 87–121 scored records). `src/evalkit/drift_monitor/detect.py` runs three detectors against a rolling 14-day baseline. A detector stays silent until at least 5 baseline days exist:

| Detector | Statistic | Defaults (`DetectorConfig`) | Catches |
|---|---|---|---|
| z-score | (today − μ) / σ, bad direction only | warn 3.0, crit 4.0 | step changes |
| CUSUM | `s = max(0, s + dir·(v−μ)/σ − k)` over the last 7 days | k=0.5, h=5.0 | slow decay no single day trips |
| PSI | Σ (a−e)·ln(a/e), additive smoothing α=0.5 | warn 0.2, crit 0.3 | shape shifts (answer length, tool mix) |

Three design choices do most of the false-alarm control:

1. **σ floors.** σ is floored at 5% of the mean (`std_floor_frac`). For rate metrics it is also floored at the binomial SE of *today's* sample size, √(p(1−p)/n), with an absolute minimum of 0.01. A refusal rate that sat near 3% for two weeks has a tiny empirical σ, and on 2026-09-14 (9 refusals in 97 records) the unfloored z is +5.4, a critical, where the floored z is +3.7, a warning.
2. **Direction.** Only the bad direction alerts. A falling refusal rate is not an incident.
3. **No baseline poisoning.** Days that raised a critical alert are excluded from future baselines, so a sustained regression keeps alerting instead of becoming the new normal.

In the 30-day example, quality decays from day index 20 (2026-09-21, ramping to full strength over 5 days). The run produced 80 alerts. The first one was **2026-09-14, a false alarm**: `refusal_rate 0.09278 vs baseline 0.02934 (z=+3.7)`. Real detection began on day 1 of the decay: `tool_mix distribution shifted (PSI=0.34; 'sql' 29% -> 20%)`. The judge score (here a mock judge), the metric everyone wants to watch, first alerted on 09-24 (z=−3.6, a warning), three days after the lexical and distributional metrics.

We replayed the stored daily metrics under different configs (a scratch script calling `detect()` day by day on a copy of the SQLite store):

| Config | Pre-decay alerts (09-06 … 09-20) | First alert after decay |
|---|---:|---|
| default (z 3/4, PSI 0.2/0.3, h=5) | 1 warning | 09-21 |
| z_warn = 2.0 | 3 | 09-21 |
| z_warn = 2.5 | 2 | 09-21 |
| z_warn = 3.5 | 1 | 09-21 |
| psi_warn = 0.1 | 2 | 09-21 |
| z_warn = 4.0, z_crit = 5.0 | **0** | 09-21 |
| no σ floors (rate floor and `std_floor_frac` off) | **7**, including 2 criticals | 09-21 |

The σ floors matter more than the thresholds: removing them yields seven false alarms, including criticals on `refusal_rate` and `judge_score`. And z_warn=4.0 loses no detection time here because PSI catches the onset; several different detectors let you set each one conservatively.

### What we got wrong

Eight scalar metrics through z-score and CUSUM plus two PSI distributions is 18 tests a night, a multiple-comparison problem nobody corrects for. The thresholds are textbook values, never calibrated on a held-out healthy period. The sweep above is the calibration we should have run first; it belongs in CI, replaying the last N healthy days against a false-alarm budget.

## 8. Cost-quality decisions: Wilson intervals and a frontier you shouldn't trust blindly

`src/evalkit/pareto/analysis.py` attaches a Wilson score interval to every success rate:

```python
centre = (p + z * z / (2 * n)) / denom
half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
```

We use Wilson rather than the normal approximation because these are small samples (60 tasks per tenant) and rates near the edges. Some configs in this data sit at 96.7%, where the Wald upper bound is 101.2%; at 0% or 100% Wald collapses to zero width.

The frontier itself (`pareto_frontier`) is computed on **point estimates**. For the `initech-finance` tenant, the dashboard marks six of seven configs as frontier members:

| config | success (Wilson 95%) | cost/task | frontier |
|---|---|---:|:---:|
| s2-large | 85.0% (74%–92%) | $0.0145 | ● |
| s2-large+reasoning | 86.7% (76%–93%) | $0.0277 | ● |
| s2-frontier | 88.3% (78%–94%) | $0.101 | ● |

We fed the same per-task results into the significance engine as paired 0/1 scores:

```
No significant difference between s2-large and s2-frontier: +3.3 pts
(95% CI [-8.3, 15.0], p=0.774, n=60 paired). Smallest effect detectable ≈ 16.1 pts.
```

The Holm-corrected leaderboard over all 21 pairs puts **five configs in tier 1**, from s2-frontier (88.3%) down to s2-legacy (68.3%). Only s1-mini and s1-nano separate. The "frontier" includes a 7x price difference that the data cannot tell apart. The `recommend()` value pick (cheapest config within 2 points of the best) is a step in the right direction, but a 2-point tolerance is tighter than the 16-point MDE.

### What we'd change

Draw the frontier on tiers, not point estimates. Within each significance tier, keep only the cheapest config, then compute dominance across tiers. The same applies to the router simulation: its "95.0% success at 32% escalated" for `globex-support` is a point on a 21-threshold sweep chosen on the same 60 tasks it is reported on. That is selection on the test set. Pick the threshold on a split and report it on the holdout.

## 9. Contamination is a validity precondition, not a metric

A significant +4.4 points on a contaminated set measures memorisation. We treat a contamination scan as a **gate that runs before** the comparison, not a column in the report.

The thresholds live in `Thresholds` in `src/evalkit/contamination/scanner.py`, and the cosine defaults are set per embedder in `embed.py`:

| Signal | Suspicious | Contaminated |
|---|---|---|
| 13-gram overlap (share of the item's n-grams) | ≥ 1 hit | ≥ 0.5 |
| MinHash containment (3-word shingles, 128 perms) | ≥ 0.5 | ≥ 0.8 |
| Cosine, hashed lexical embedder | ≥ 0.4 | ≥ 0.8 |
| Cosine, OpenAI embedder | ≥ 0.85 | ≥ 0.95 |

The per-embedder thresholds exist because, as the docstring puts it, "lexical embeddings score paraphrases far lower than neural ones (~0.45 vs ~0.9)". A single global cosine cutoff is a bug.

On the 12-item example with planted leaks, the three methods together flag 4 contaminated and 4 suspicious items. **The classic 13-gram-only check flags 3 contaminated and 1 suspicious, and misses `q03-edited` entirely.** That item is a lightly edited copy with 0% 13-gram overlap that the other methods catch at containment 0.67 and cosine 0.80. Suspicious items can be adjudicated with this judge prompt, verbatim from `src/evalkit/contamination/judge.py`:

```
You audit evaluation datasets for training-data contamination.
EVAL ITEM:
{item}

TRAINING PASSAGE:
{passage}

Does the training passage contain this eval item, a paraphrase of it, or its answer in a way that would let a model memorise it? Reply with YES or NO on the first line, then one short sentence of justification.
```

In the offline demo, adjudication upgraded 3 of the 4 suspicious items (to 7 contaminated). The default judge is a deterministic `MockLLM` responder that answers YES when ≥55% of the item's content words recur. It tests the plumbing, not the judgment. With `--fail-on contaminated` the scan exits 1, and that exit code should sit upstream of every comparison in this post.

What we'd change: the 0.5 n-gram and 0.8 containment cutoffs are inherited conventions, never validated against labelled leaks vs coincidental overlaps. We would also report on both the full set and the decontaminated subset (`--clean-eval` already writes it) and test the delta between them.

## 10. The checklist

Before any eval number goes into a decision doc: **Valid?** (contamination scan passed) · **Paired?** · **Clustered?** (what is the independent unit) · **Powered?** (MDE at the *measured* discordance) · **Corrected?** (how many comparisons) · **Practical?** (does the CI's far end clear a threshold someone would act on).

## Reproducibility

All commands run from the repo root with the locked environment. Outputs go to gitignored `out/` directories.

```bash
# Significance engine: paired/clustered compare, Holm, leaderboard, power
bash examples/07-significance-engine/run.sh
cd examples/07-significance-engine
uv run --no-sync evalkit significance compare --results results.jsonl --a gpt-base --b ft-v2 --no-clusters
uv run --no-sync evalkit significance compare --results results.jsonl --a gpt-base --b ft-v2 --ci percentile
for c in none holm bh bonferroni; do
  uv run --no-sync evalkit significance compare --results results.jsonl --a gpt-base --correction $c; done
uv run --no-sync evalkit significance compare --results results.jsonl --a gpt-base --b ft-v2 --no-clusters --json
uv run --no-sync evalkit significance power --baseline 0.8 --delta 0.02 --paired
uv run --no-sync evalkit significance power --baseline 0.8 --delta 0.02 --paired --discordance 0.136 --n 500
uv run --no-sync evalkit significance power --baseline 0.862 --delta -0.05 --paired --n 80
uv run --no-sync evalkit significance power --baseline 0.8 --delta 0.044 --paired --discordance 0.136 --n 500
uv run --no-sync evalkit significance compare --results small_results.jsonl --a gpt-base --b ft-v2 --json
uv run --no-sync evalkit significance power --baseline 0.862 --delta -0.05 --paired --n 80 --discordance 0.05
uv run --no-sync evalkit significance power --baseline 0.862 --delta -0.05 --paired --n 40
uv run --no-sync evalkit significance power --baseline 0.862 --delta -0.05 --paired --n 40 --discordance 0.05
cd ../..

# Cluster-correlation simulation (section 1)
uv run --no-sync python -c "
from evalkit.significance.compare import simulate_results, load_scores, compare
for sd in (0.0, 1.0):
    t = load_scores(simulate_results({'a': 0.8, 'b': 0.84}, n_items=500, n_clusters=50, cluster_sd=sd, seed=3))
    for cl in (True, False):
        c = compare(t, 'a', 'b', clustered=cl); print(sd, cl, c.ci.low, c.ci.high, c.p_value)"

# Degenerate BCa vs percentile (section 4)
uv run --no-sync python -c "
from evalkit.significance.stats import bootstrap_mean
d = [1, 1] + [0] * 78; cl = ['t0', 't0'] + [f't{i//2+1}' for i in range(78)]
for m in ('bca', 'percentile'): print(m, bootstrap_mean(d, cl, 5000, 0.95, m))"

# Regression gate: four builds
bash examples/04-regression-gate/run.sh

# Drift monitor: 30 days, decay from day index 20 (2026-09-21)
bash examples/09-drift-monitor/run.sh
# Threshold sweep: copy out/drift.sqlite, clear the alerts table, then call
# evalkit.drift_monitor.detect.detect(day, store, DetectorConfig(...)) for each stored date,
# calling store.save_alerts(day, alerts) after each day (baseline exclusion reads them).
# "No σ floors" = DetectorConfig(std_floor_frac=0.0, rate_sd_floor=0.0, rate_metrics=frozenset()).

# Pareto + Wilson, then the same tenant through the significance engine
bash examples/10-pareto-dashboard/run.sh
python3 -c "
import json
for l in open('examples/10-pareto-dashboard/results.jsonl'):
    r = json.loads(l)
    if r['tenant'] == 'initech-finance':
        print(json.dumps({'item_id': r['task_id'], 'model': r['config'], 'score': int(bool(r['success']))}))
" > /tmp/initech.jsonl
uv run --no-sync evalkit significance compare --results /tmp/initech.jsonl --a s2-large --b s2-frontier
uv run --no-sync evalkit significance leaderboard --results /tmp/initech.jsonl

# Contamination precondition
bash examples/14-contamination-checker/run.sh
```

All seeds are fixed (`seed=0` by default in the significance engine, `--seed 7` for drift traffic, `--seed 11` for the committed Pareto results), so every number above reproduces exactly except the regression gate's wall-clock latencies, which move by about ±0.1 ms.
