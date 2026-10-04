# 03 — Calibrated LLM-as-a-Judge

> Evaluation pipeline measuring the judge's verbosity, position and self-preference biases against
> 500 human-labeled anchors. Uncalibrated judges are expensive vibes; calibrated judges are
> instruments.

## What and why

LLM judges are known to prefer the first response shown, prefer longer responses, and prefer
outputs from their own model family. If you use a judge to pick between models or prompts, those
biases leak straight into your decisions. `evalkit calibrated-judge` treats the judge like a
measuring instrument: it is checked against a ruler (human-labelled anchors), its systematic
errors are quantified, and a correction is fitted and validated on held-out anchors.

## Design

**Anchors** (`anchors.py`) are pairwise comparisons with a human verdict:
`{id, prompt, response_a, response_b, model_a, model_b, human_pref: a|b|tie, human_score_a?, human_score_b?}`.
`make-anchors` builds a seeded synthetic set of 500 where the nuisance variables are controlled:
quality is the number of substantive facts a response covers (what humans reward); length is a
sentence budget drawn *independently* of quality and filled with generic filler of the same average
word count; model family shows up as a stylistic opener ("Certainly! Great question.", "Quick
answer.", ...), which is how real judges recognise their own outputs. Bring your own human labels
in the same JSONL format to audit a real judge.

**Judges** (`judge.py`): `PairwiseJudge` and `PointwiseJudge` prompt any `LLM` and parse JSON
verdicts robustly (fenced JSON, `[[A]]`, garbage → tie). Model names are never shown to the
judge. `SimulatedJudge` is a `MockLLM` responder that reads the same prompts and answers like a
flawed judge with dial-in `position_bias`, `verbosity_bias`, `self_bias`, `noise` and
`overconfidence`, so tests can verify that the audit recovers injected biases.

**Audit** (`audit.py`): every anchor is judged in both orders (AB and BA).

| Measure | How |
|---|---|
| Agreement, Cohen's κ | 3-way (a/b/tie) vs human; also agreement on decisive pairs |
| Position bias | consistency after swapping, flip rate, first-slot win rate (ideal 50%) |
| Verbosity bias | P(judge prefers longer) vs P(human prefers longer) overall and by \|log length ratio\| bucket; logistic coefficient of log length ratio on the judge's choice **controlling for the human verdict** (ideal 0) |
| Self-preference | own-family win rate per the judge vs per humans on pairs with exactly one own-family response |
| Calibration | ECE + reliability table of stated confidence vs actual agreement |
| Pointwise (optional) | Pearson/Spearman vs human scores, and length slope of judge score controlling for human score |

**Calibration** (`calibrate.py`): three stacked mitigations, fitted on a train split (default 60%)
and evaluated on the held-out rest with the same metrics:

1. *swap-and-aggregate* — average P(A wins) over both orders, cancelling position bias by construction;
2. *length control* — a log-length-ratio term in a logistic correction (the idea behind length-controlled win rates);
3. *self-family control* — a term for "exactly one response is from the judge's family".

The correction `P(A wins) = σ(w·[1, judge_logit, log_len_ratio, self_family])` is fitted by
Newton/IRLS on decisive anchors, with a tie band chosen on the train split. It is saved as JSON
and applied at inference time by `CalibratedJudge.compare(prompt, a, b, model_a, model_b)`.

## How to run

```bash
examples/03-calibrated-judge/run.sh
# step by step
uv run --no-sync evalkit calibrated-judge make-anchors --n 500 --seed 0 --out anchors.jsonl
uv run --no-sync evalkit calibrated-judge audit --anchors anchors.jsonl --pointwise --judgments j.jsonl --out-md audit.md
uv run --no-sync evalkit calibrated-judge calibrate --anchors anchors.jsonl --judgments j.jsonl --out calibration.json
# a real judge (judge family is used for the self-preference measurement)
uv run --no-sync evalkit calibrated-judge audit --anchors anchors.jsonl --llm openai:gpt-4o-mini --judge-family gpt --judgments j.jsonl --rejudge
```

`--judgments` caches the two-order verdicts so `calibrate` reuses the audit's judge calls.
Simulated-judge knobs: `--sim-position-bias --sim-verbosity-bias --sim-self-bias --sim-noise --sim-overconfidence`.

## Sample output

Audit of the default simulated judge (family `nova`) on 500 anchors:

```
| Agreement with humans (3-way) | 68.8% | high |
| Agreement on decisive pairs | 84.7% | high |
| Cohen's κ | 0.467 | → 1 |
| Position consistency (same verdict after swap) | 67.0% | 100% |
| First-slot win rate | 63.6% | 50% |
| Judge prefers longer | 63.8% | = human (53.9%) |
| Verbosity excess over humans | +9.9 pts | 0 |
| Length logit coef (controls for human pref) | +4.18 | 0 |
| Self-preference: own-family win rate, judge vs human | 69.0% vs 54.4% (n=308) | equal |
| Self-preference gap | +14.6 pts | 0 |
| Confidence ECE | 0.092 | 0 |
| Pointwise score vs human: Pearson / Spearman | 0.644 / 0.691 | → 1 |
| Pointwise length slope (points per log-word, human-controlled) | +2.72 | 0 |
```

Calibration, fitted on 300 anchors and evaluated on 200 held-out anchors:

```
| Judge | Agreement | κ | Verbosity excess | Length coef | Self-pref gap | ECE |
|---|---:|---:|---:|---:|---:|---:|
| raw judge (AB order) | 68.5% | 0.459 | +10.1 pts | +3.88 | +15.6 pts | 0.091 |
| swap-aggregated | 73.5% | 0.541 | +9.9 pts | +3.48 | +18.0 pts | 0.061 |
| swap + length | 73.5% | 0.534 | +2.1 pts | +0.86 | +21.6 pts | 0.058 |
| swap + length + self (full) | 77.0% | 0.596 | +2.6 pts | +0.96 | +11.2 pts | 0.058 |

Correction: P(A wins) = σ(bias=+0.27, judge_logit=+1.80, log_len_ratio=-4.48, self_family=-2.33), tie band ±0.00.
```

The control run (same judge with all biases set to 0) audits clean: first-slot win rate 50.4%,
verbosity excess −1.4 pts, length coefficient −0.38, self-preference gap −2.8 pts.

Swap-aggregation fixes position bias but does nothing for length or self-preference; the length
term takes out most of the verbosity excess; the self-family term roughly halves the remaining
self-preference gap. Agreement goes from 68.5% to 77.0% on held-out anchors.

## Validation

`tests/test_calibrated_judge.py` checks that the anchor generator keeps length uncorrelated with
quality (|r| < 0.15); that an unbiased simulated judge audits clean; that a biased one is caught on
every axis and that self-preference disappears when audited against the wrong family; that the
verbosity coefficient rises monotonically with the injected bias; and that calibration improves
held-out agreement by ≥4 pts and κ, more than halves the length coefficient, shrinks the
self-preference gap, and learns weights with the expected signs.

## Limitations

- The synthetic anchors are a controlled testbed, not a substitute for real human labels; with
  real data, length and quality are correlated, so "excess" verbosity depends on how well the
  human verdict controls for quality.
- Self-preference uses a coarse family signal (one indicator); real self-recognition is graded.
- The fitted correction cannot recover information the judge never produced: with an
  overconfident judge whose probabilities are saturated, a residual self-preference gap remains.
- Calibrated decisions use a single tie band; held-out ECE improves over the raw judge but is
  noisy at n=200.
- Each anchor costs two judge calls (both orders), four with `--pointwise`. With deepseek-flash, a full 500-anchor audit with pointwise scores cost about $0.20 (see below).

## Real-model run (DeepSeek, Oct 2026)

```bash
examples/03-calibrated-judge/real_run.sh          # sources ~/TradingAgents/.env, writes out/real/
# the core command it runs:
uv run --no-sync evalkit calibrated-judge audit --anchors anchors.jsonl \
  --llm deepseek:deepseek-flash --judge-family none --pointwise --workers 8 \
  --judgments out/real/judgments.jsonl --out-md out/real/audit.md --out-json out/real/audit.json
uv run --no-sync evalkit calibrated-judge calibrate --anchors anchors.jsonl \
  --llm deepseek:deepseek-flash --judge-family none --judgments out/real/judgments.jsonl \
  --out out/real/calibration.json
```

- **Judge:** `deepseek:deepseek-flash` (DeepSeek V4.1 Flash, thinking off, temperature 0). No
  second model was needed.
- **Sample:** all 500 synthetic anchors (`make-anchors --n 500 --seed 0`), both orders plus
  pointwise scores for both responses: 2,000 calls, 0 failures.
- **Cost:** main run 572,440 input + 20,098 output tokens = **$0.196**; with the parser probe and
  a 40-anchor determinism re-run the total spend was about **$0.21**.
- **Judge family:** DeepSeek belongs to none of the synthetic families (`atlas`/`nova`/`orion` are
  only stylistic openers), so the main audit and calibration use `--judge-family none`, with no
  self-preference term. Rerunning the audit from the cached verdicts with each family name
  measures *style preference* instead, at no extra API cost.
- Small committed outputs are in `examples/03-calibrated-judge/real_output/`: `audit.md/json`,
  `calibration.md/json`, `calibration_report.json`, `style_probe.txt`, and the 500 cached
  verdicts in `judgments.jsonl` (82 KB), so calibration can be reproduced offline.

Audit (500 anchors):

```
| Agreement with humans (3-way) | 79.6% | high |
| Agreement on decisive pairs | 93.5% | high |
| Cohen's κ | 0.657 | → 1 |
| Position consistency (same verdict after swap) | 85.0% | 100% |
| First-slot win rate | 55.0% | 50% |
| Judge prefers longer | 47.5% | = human (51.0%) |
| Verbosity excess over humans | -3.5 pts | 0 |
| Length logit coef (controls for human pref) | -1.31 | 0 |
| Confidence ECE | 0.057 | 0 |
| Agreement after swap-aggregation | 80.8% | — |
| Pointwise score vs human: Pearson / Spearman | 0.780 / 0.840 | → 1 |
| Pointwise length slope (points per log-word, human-controlled) | -1.46 | 0 |

| Confidence bin | n | Stated | Actual |
| 0.60–0.65 | 39 | 60.0% | 76.9% |
| 0.70–0.75 | 34 | 70.0% | 88.2% |
| 0.80–0.85 | 45 | 83.8% | 91.1% |
| 0.90–0.95 | 94 | 90.0% | 93.6% |
| 0.95–1.00 | 193 | 96.1% | 97.9% |
```

Style probe from the cached verdicts (own-style win rate, judge vs human): atlas 44.7% vs 44.9%
(−0.2 pts, n=302), nova 52.6% vs 54.4% (−1.8 pts, n=308), orion 53.9% vs 51.0% (+2.9 pts, n=206).

Calibration (fit on 300, evaluated on 200 held-out anchors):

```
| Judge | Agreement | κ | Verbosity excess | Length coef | Self-pref gap | ECE |
|---|---:|---:|---:|---:|---:|---:|
| raw judge (AB order) | 82.0% | 0.698 | -3.1 pts | -1.08 | — | 0.069 |
| swap-aggregated | 81.0% | 0.675 | -4.3 pts | -1.84 | — | 0.093 |
| swap + length | 80.0% | 0.648 | +0.1 pts | -0.30 | — | 0.050 |
| swap + length + self (full) | 80.0% | 0.648 | +0.1 pts | -0.30 | — | 0.050 |

Correction: P(A wins) = σ(bias=+0.17, judge_logit=+1.85, log_len_ratio=+2.19, self_family=+0.00), tie band ±0.01.
```

(With `--judge-family none` the "full" row is the same as "swap + length", because the self
term is always zero.)

**Comparison with the offline simulated judge.** The simulated judge is built to be badly biased:
68.8% agreement, a 63.6% first-slot win rate, +9.9 pts verbosity excess, overconfidence (ECE
0.092), and a +14.6 pt self-preference gap. Calibration raised its held-out agreement from 68.5%
to 77.0%. The real judge shows a different picture:

- **More accurate:** 79.6% three-way agreement, 93.5% on decisive pairs, and κ 0.66. Pointwise
  scores track human scores (Spearman 0.84). The anchors make this an easy task: quality is a
  count of on-topic facts, and the filler sentences are obviously empty. This result does not
  show that the judge is this good on real human preference data.
- **Some position bias:** 55.0% first-slot wins and 85% consistency after a swap. At least part
  of the inconsistency is sampling noise, not order: re-judging 40 anchors at temperature 0
  matched both earlier verdicts on only 90% of them. DeepSeek is not deterministic at T=0, so the
  cache file (`--judgments`) is what makes the audit and the calibration reproducible.
- **Slightly anti-verbose, not verbose:** the judge preferred the longer response 47.5% of the
  time against 51.0% for humans. The length coefficient is negative (−1.31), and so is the
  pointwise length slope (−1.46 points per log-word). Our anchors pad length with generic filler,
  and this judge appears to penalise that padding. The opposite bias is common on real data,
  where length and substance go together. On the held-out split, the length term moved the
  verbosity excess from −3.1 to +0.1 pts and the coefficient from −1.08 to −0.30.
- **Underconfident, not overconfident:** in every bin, stated confidence is below actual accuracy
  (for example, 60% stated vs 77% actual). The fitted `judge_logit` weight of +1.85 stretches the
  probabilities, and held-out ECE drops from 0.069 to 0.050.
- **No measurable style preference:** all three gaps are within ±3 pts, which is noise at
  n≈200–300. True self-preference cannot be tested here, because no anchor response was written
  by DeepSeek.
- **Calibration does not improve agreement for this judge.** Held-out agreement went from 82.0%
  (raw) to 80.0% (swap + length). A paired bootstrap of the difference over the 200 test anchors
  gives −2.0 pts, 95% CI [−6.5, +2.5]: no significant change in either direction.
  Swap-aggregation also did not help agreement (81.0%) and made ECE worse (0.093), because
  averaging a confident verdict with a tie or a flipped verdict pulls the probability toward 0.5.
  When a judge is already fairly unbiased on this data, the calibration step mainly removes the
  remaining length bias and fixes the confidence scale. It does not raise accuracy. A real gain
  in agreement would need anchors where this judge actually fails.

**Real-model integration fixes (each covered by `tests/test_calibrated_judge_real.py`):**
judge calls run concurrently (`--workers`, results kept in anchor order); an anchor whose calls
still fail after the client's retries is dropped and counted, so one failure no longer aborts a
run that has already paid for 1,000+ calls; API usage (calls, tokens, `cost_usd`) is tracked
thread-safely, printed, and saved under `usage` in the report JSON; `--judge-family none` is
supported for judges outside the anchor families; and the parsers now accept percentage
confidences (`85` → 0.85, where before it was clamped to 1.0), winner strings such as
`"Response B"`, `"a"` and `"both"`, prose or other brace objects before the verdict JSON, and
pointwise replies that mention the "1-10" scale. In this run DeepSeek always returned clean JSON
(`{"winner": "A", "confidence": 0.95}`, `{"score": 3}`), including ties that carry a confidence
(`{"winner": "tie", "confidence": 0.85}`), so the parser changes are defensive and were not
needed for this particular run.
