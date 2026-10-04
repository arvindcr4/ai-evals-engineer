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
- Each anchor costs two judge calls (both orders), four with `--pointwise`.
