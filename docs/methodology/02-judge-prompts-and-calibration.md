# Your LLM judge is an instrument; calibrate it

*Methodology teardown 2 of 3: the exact judge prompts in evalkit, how we measure their biases, and
how we correct them. Every number below comes from a command listed in the
[Reproducibility](#reproducibility) section, run against this repo on 2026-10-04. All of them are
offline: the anchors are synthetic, the "human" labels are simulated, and every judge is a seeded
mock. They show that the measurement and correction pipeline works on biases we injected
ourselves. They are not measurements of a production judge.*

Treat an LLM judge like a thermometer, not an oracle: check it against known references, write
down how far off it is, and correct for that. Its errors are not random. They point in a
direction: toward the first slot, toward the longer answer, toward its own model family's voice.
This post quotes every judge prompt in the codebase, describes the anchor set and bias audit, and
shows the logistic recalibration with held-out before/after numbers and what it does not fix.

## 1. The prompts, verbatim

Six modules call an LLM to make a judgment. They don't share a prompt: the output format follows
from what the downstream code does with the answer.

### 03 calibrated_judge: pairwise with stated confidence

`src/evalkit/calibrated_judge/judge.py`:

```python
PAIRWISE_SYSTEM = (
    "You are an impartial evaluator. Compare two assistant responses to the same question. "
    "Judge helpfulness, correctness and completeness. Do not let response order, length or "
    "style influence you."
)
PAIRWISE_TEMPLATE = """[Question]
{prompt}

[Response A]
{a}
[End of Response A]

[Response B]
{b}
[End of Response B]

Which response is better? Reply with JSON only:
{{"winner": "A" | "B" | "tie", "confidence": <probability your verdict is right, 0.5-1.0>}}"""
```

and the pointwise variant used for the score-level audit:

```python
POINTWISE_SYSTEM = "You are a strict evaluator. Rate the response from 1 (useless) to 10 (ideal)."
POINTWISE_TEMPLATE = """[Question]
{prompt}

[Response]
{response}
[End of Response]

Reply with JSON only: {{"score": <integer 1-10>}}"""
```

Two choices here matter. The system prompt tells the judge not to be swayed by order, length or
style. **We do not count that instruction as a mitigation.** Note that our own numbers can't
test it: the simulated judge below ignores the instructions by construction, so its bias says
nothing about whether the sentence helps a real model. We have not run that ablation on a real
judge. We keep the sentence because it costs nothing, and we measure bias as if it weren't there. Second, we ask for a
`confidence`. That is not because we believe it. The recalibration needs a continuous signal to
regress on, and a bare A/B/tie gives it three values to work with.

Model names never appear in the prompt. Self-preference has to come from style, not from a label.

The parser (`parse_pairwise`) tries JSON first, falls back to `[[A]]`/`[[B]]`/`[[C]]` or
`winner: A` (with a default confidence of 0.75), and treats anything it can't parse as
`("tie", 0.5)`. It also clamps JSON confidence into [0.5, 1.0]. A judge
that breaks the format is scored as abstaining, never as a vote for slot A.

### 02 shadow_router: single token, both orders

`src/evalkit/shadow_router/report.py`:

```python
JUDGE_PROMPT = """You are comparing two assistant answers to the same user request.
Pick the answer that is more correct, specific and helpful. Reply with exactly one
token: A, B, or TIE.

Question:
{question}

Answer A:
{a}

Answer B:
{b}
"""
```

There is no confidence field here, because the decision is a promote/hold/block gate on a win
rate, not a per-pair probability. Position bias is handled structurally in `judge_pair`. The
judge runs twice, once with the primary answer as A and once with the shadow answer as A, and
a pair only counts as a win or a loss **if both orders agree**. Any disagreement becomes a tie:

```python
shadow_first = {"A": "loss", "B": "win", "TIE": "tie"}[first]
shadow_second = {"A": "win", "B": "loss", "TIE": "tie"}[second]
return shadow_first if shadow_first == shadow_second else "tie"
```

The verdict uses a Wilson 95% interval on decided pairs: `BLOCK` if the upper bound is below
0.5, `PROMOTE` if the lower bound is above 0.5. (An error rate above the limit also blocks, and a
cheaper candidate with a non-loss rate of at least 90% can be promoted on cost.)

### 06 dpo_flywheel: pointwise 0–10 for pair margins

`src/evalkit/dpo_flywheel/pairs.py`:

```python
JUDGE_PROMPT = """You grade an assistant reply for helpfulness, correctness and safety.
Prompt:
{prompt}

Reply:
{response}

Answer with one line: SCORE: <0-10>"""
```

The score is used as a difference. `margin = judge(chosen) - judge(rejected)`, and a pair is
kept only when `margin >= 1.0` (`FilterConfig.min_margin`), unless it came from a human
correction (`trust_corrections: True`). The offline default is `HeuristicJudge`, a 0–10 rubric
in code: it starts at 4, adds up to +3 for content-word coverage of the prompt, +2 for 12–300
words (+0.5 for 5–11, −1.5 otherwise), −2 when fewer than half the words are unique and −4 for a
refusal, then clamps to [0, 10].

### 09 drift_monitor: 1–5 with tool errors in the prompt

`src/evalkit/drift_monitor/scorers.py`:

```python
JUDGE_PROMPT = """Rate the assistant's answer to the user on a 1-5 scale
(5 = fully correct and helpful, 1 = useless, refused or empty).
Tool errors during the run: {tool_errors}.
Reply with a single digit.

User: {input}

Assistant: {output}
"""
```

The reply is parsed with `re.search(r"[1-5]", reply)` and mapped to `(d - 1) / 4` in [0, 1].
Unparseable replies count as `None` and are skipped rather than scored 0. The tool-error count
is put into the prompt on purpose, since a drift monitor cares about the whole run and not just
the final text. That also means the judge is told a fact it is expected to penalise, and
section 6 comes back to this.

### 14 contamination: binary adjudicator

`src/evalkit/contamination/judge.py`:

```python
PROMPT = (
    "You audit evaluation datasets for training-data contamination.\n"
    "EVAL ITEM:\n{item}\n\nTRAINING PASSAGE:\n{passage}\n\n"
    "Does the training passage contain this eval item, a paraphrase of it, or its answer "
    "in a way that would let a model memorise it? Reply with YES or NO on the first line, "
    "then one short sentence of justification."
)
```

Only items already flagged `suspicious` by n-gram, MinHash or embedding evidence are sent to it,
and the only thing it can do is upgrade: `YES` changes the status to `contaminated`, `NO` leaves
it suspicious. A judge that can only add to the flagged set can't hide a leak.

### 12 edge_case_gen: label proposer with a confidence floor

`src/evalkit/edge_case_gen/llm_gen.py`:

```python
LABEL_TEMPLATE = """TASK: propose_label
Task: {desc}
Allowed labels: {labels}
INPUT_JSON:
{inp}
END_INPUT
Return: {{"label": "<one allowed label>", "confidence": <0..1>}}"""
```

This is a judge in the labelling sense. `propose_labels` (in `validate.py`) sends a case to
human review when the LLM supplied the label (no oracle) with confidence below
`min_confidence=0.7`, when the LLM disagrees with the oracle, or when the case is on the
`semantic` axis.

### Summary

| Module | Format | Position control | Length control | Downstream use |
|---|---|---|---|---|
| 03 calibrated_judge | pairwise JSON + confidence | swap-and-average | fitted log-length term | P(A wins) |
| 02 shadow_router | pairwise, one token | both orders must agree | none | Wilson CI on win rate |
| 06 dpo_flywheel | pointwise 0–10 | n/a | heuristic fallback only: 12–300 word band | margin ≥ 1.0 filter |
| 09 drift_monitor | pointwise 1–5 | n/a | none | z-score vs baseline |
| 14 contamination | YES/NO | n/a | n/a | upgrade-only |
| 12 edge_case_gen | label + confidence | n/a | n/a | review if conf < 0.7 |

Only module 03 is calibrated. The rest of this post is about why that one gets the full
treatment, and section 6 covers what we'd change in the others.

## 2. The ruler: anchor-set design

A calibration is only as good as the thing you calibrate against. An anchor
(`src/evalkit/calibrated_judge/anchors.py`) is a pairwise comparison with a human verdict:

```
{id, prompt, response_a, response_b, model_a, model_b,
 human_pref: "a" | "b" | "tie", human_score_a?, human_score_b?}
```

`make_anchors` builds a seeded set of 500 in which each nuisance variable is **controlled
independently of quality**. Without that independence, you can't tell bias apart from
correlation.

- **Quality** is the number of substantive facts a response covers, out of six per topic, across
  ten topics. Each side draws its count uniformly from 0 to 6.
- **Length** is a separate sentence budget, drawn uniformly from 6 to 18 sentences
  (`min_sentences=6, max_sentences=18`). It is topped up with generic filler sentences whose
  average word count is close to a fact sentence's (9.0 vs 9.6 words). So a long answer is not, on
  average, a better one; the 0.6-word gap leaves a very small residual correlation.
- **Model family** appears only as a stylistic opener: `"Quick answer."` (atlas),
  `"Certainly! Great question."` (nova), `"Here's a breakdown."` (orion). That is how real judges
  recognise their own family: by voice, not by name.
- **Human verdict**: utility is `facts + N(0, 0.6)` for each side. If the two utilities are within
  0.5 of each other, the verdict is `tie`.

The seed-0 set has 224 `a`, 214 `b` and 62 `tie` verdicts. Ties are 12.4% of the set, which
matters later. Remember these are simulated humans: the "human" labels are a noisy function of the
same fact counts we generated.

The judge under test is `SimulatedJudge`, a deterministic `MockLLM` responder. It reads the real
prompt above, parses out the two response blocks, and answers like a flawed judge. Its decision
logit for "the first slot wins" is:

```
z = skill·Δfacts + verbosity_bias·log(len1/len2) + position_bias
    + self_bias·(own1 − own2) + noise·ε
confidence = σ(overconfidence · |z|)
```

The defaults are `skill=0.7, position_bias=1.0, verbosity_bias=2.5, self_bias=1.5, noise=0.8,
overconfidence=2.0, tie_band=0.25`, and the judge family is `nova`. We simulate because the
injected biases are known, so we can check that the audit finds them and that the calibration
removes them. The CLI accepts `--llm openai:gpt-4o-mini --judge-family gpt` on the same anchors,
but we did not run that for this post, and two caveats apply. No anchor model belongs to a `gpt`
family (they are atlas, nova and orion), so self-preference can't be measured on these anchors.
And agreement with simulated humans says little about a real judge. For a real judge you need
real human labels and anchors that include its own family's outputs.

## 3. Measuring the biases

Every anchor is judged twice, in AB order and in BA order. Every metric comes from
`decision_metrics` in `src/evalkit/calibrated_judge/audit.py`, so the raw, swapped and calibrated
judges are all scored with exactly the same yardsticks.

**Position.** We measure *consistency*, the share of anchors where the AB and BA verdicts name
the same response, and the *first-slot win rate* across all decisive verdicts in both orders.
The ideal values are 100% and 50%.

**Verbosity.** "The judge prefers longer answers" is meaningless on its own, because sometimes
the longer answer really is better. So we compare against humans in two ways:

- the excess `P(judge picks longer) − P(human picks longer)`, overall and by |log length ratio|
  bucket;
- a logistic regression of the judge's choice on `[1, human_sign, log_len_ratio, self_family]`.
  The `log_len_ratio` coefficient is the length effect **after controlling for the human
  verdict**, and the ideal is 0.

**Self-preference.** On pairs where exactly one side comes from the judge's family, we compare
the judge's own-family win rate with the humans' own-family win rate on the same pairs. Ties
count 0.5. The ideal gap is 0.

**Confidence.** ECE over ten equal-width bins on [0.5, 1.0], computed on pairs where both the
judge and the human were decisive.

Here is the default simulated judge on all 500 anchors
(`examples/03-calibrated-judge/out/audit.md`). The decision rows use the AB order only:

| Measure | Value | Ideal |
|---|---:|---:|
| Agreement with humans (3-way) | 68.8% | high |
| Agreement on decisive pairs | 84.7% | high |
| Cohen's κ | 0.467 | → 1 |
| Position consistency | 67.0% | 100% |
| First-slot win rate | 63.6% | 50% |
| Judge prefers longer | 63.8% | = human (53.9%) |
| Verbosity excess over humans | +9.9 pts | 0 |
| Length logit coef (human-controlled) | +4.18 | 0 |
| Own-family win rate, judge vs human | 69.0% vs 54.4% (n=308) | equal |
| Self-preference gap | +14.6 pts | 0 |
| Confidence ECE | 0.092 | 0 |
| Pointwise length slope (pts per log-word) | +2.72 | 0 |

The bucketed verbosity table shows why a single aggregate number misleads:

| \|log len ratio\| | n | Judge prefers longer | Humans prefer longer |
|---|---:|---:|---:|
| [0, 0.2) | 125 | 49.6% | 54.4% |
| [0.2, 0.5) | 150 | 57.3% | 46.7% |
| [0.5, 1) | 115 | 77.4% | 60.0% |
| [1, inf) | 9 | 100.0% | 88.9% |

Near-equal lengths show no excess (the judge is 4.8 points *below* humans). From a log ratio of
0.2 the judge leans 10.6 points further toward the longer answer than humans do, and once one
answer is about 1.6× longer the gap is 17.4 points. (Bucket rows count only pairs where both judge
and human were decisive.)

The reliability table is where overconfidence shows. 281 of the 402 decisive verdicts claimed
confidence of 0.95 or more. They were right 93.2% of the time, against a stated average of
99.1%. In the 0.70–0.75 bin the judge was right 53.3% of the time, close to a coin flip.

**The control matters as much as the result.** We ran the same judge with `--sim-position-bias 0
--sim-verbosity-bias 0 --sim-self-bias 0 --sim-overconfidence 1`. It audits clean: first-slot
win rate 50.4%, verbosity excess −1.4 pts, length coefficient −0.38, self-preference gap −2.8
pts. If your audit flags an unbiased judge, the problem is your audit.

## 4. The recalibration

`src/evalkit/calibrated_judge/calibrate.py` stacks three mitigations. They are fitted on 60% of
the anchors (300) and evaluated on the other 40% (200):

1. **Swap-and-aggregate.** Average P(A wins) across both orders. This cancels position bias by
   construction.
2. **Length control.** Add a `log_len_ratio` term to a logistic correction. This is the idea
   behind length-controlled win rates.
3. **Self-family control.** Add a term that is +1 when only A comes from the judge's family and
   −1 when only B does.

The features are built like this:

```python
def feature_row(ab_p_a, ba_p_a, log_len_ratio, self_diff):
    return [1.0, float((_logit(ab_p_a) + _logit(ba_p_a)) / 2), log_len_ratio, float(self_diff)]
```

`_logit` clips p to [0.02, 0.98]. The model is an L2-regularised logistic regression
(`l2=1e-2`, intercept unpenalised), fitted by Newton/IRLS on the *decisive* human labels only. A
tie band is then grid-searched over 0.00–0.25 to maximise three-way agreement on the training
split. The result is a JSON file you can review in a pull request (`out/calibration.json`):

```
P(A wins) = σ(0.27 + 1.80·judge_logit − 4.48·log_len_ratio − 2.33·self_family), tie band ±0.00
```

All signs are as expected: the length and self-family weights are negative, so they push against
what the judge adds. They don't cancel it exactly; section 5 shows the self-family weight is less
than half of what full cancellation would need.

### Before and after, held out

| Judge (200 held-out anchors) | Agreement | κ | Verbosity excess | Length coef | Self-pref gap | ECE |
|---|---:|---:|---:|---:|---:|---:|
| raw judge (AB order) | 68.5% | 0.459 | +10.1 pts | +3.88 | +15.6 pts | 0.091 |
| swap-aggregated | 73.5% | 0.541 | +9.9 pts | +3.48 | +18.0 pts | 0.061 |
| swap + length | 73.5% | 0.534 | +2.1 pts | +0.86 | +21.6 pts | 0.058 |
| swap + length + self (full) | 77.0% | 0.596 | +2.6 pts | +0.96 | +11.2 pts | 0.058 |

How to read it:

- **Swapping gives +5 points of agreement on its own.** It fixes position. It also lowers ECE
  (0.091 → 0.061), because averaging two orders pulls extreme confidences toward 0.5. The
  verbosity excess barely moves (+10.1 → +9.9).
- **The length term removes about 80% of the verbosity excess** (+10.1 → +2.1 pts) and cuts the
  human-controlled length coefficient from +3.88 to +0.86. Agreement stays flat at this step,
  which is the honest result: you're correcting a bias, not adding signal.
- **Self-preference gets *worse* before it gets better** (+15.6 → +18.0 → +21.6). Our reading:
  once position noise and length effects are taken out, the judge's remaining systematic
  preference shows up more cleanly in its decisions. Order and length noise had been masking
  part of it. If you fix one bias, re-measure all the others.
- **The full model reaches 77.0% / κ 0.596**, up from 68.5% / 0.459.

### One seed is an anecdote

Seed 0 is the number in `docs/projects/03-calibrated-judge.md`. Here are three more. `--seed` changes both the train/test
split and the simulated judge's noise draw.

| Seed | Raw agreement | Full agreement | Raw self-pref gap | Full self-pref gap | Raw ECE | Full ECE |
|---|---:|---:|---:|---:|---:|---:|
| 0 | 68.5% | 77.0% | +15.6 | +11.2 | 0.091 | 0.058 |
| 1 | 66.0% | 77.0% | +17.1 | +2.1 | 0.160 | 0.065 |
| 2 | 71.5% | 80.5% | +12.4 | +0.4 | 0.103 | 0.057 |
| 3 | 71.0% | 79.5% | +11.5 | −1.3 | 0.129 | 0.046 |

The agreement gain holds up: +8.5 to +11.0 points on every seed. The self-preference fix varies.
Seed 0 happens to be the *worst* case for residual self-preference. We publish it anyway rather
than switching the headline to seed 2.

## 5. Honest limitations

**Residual self-preference is real, and our metric for it is noisy.** Two pieces of evidence.
First, the fitted self-family weight is −2.33. If the judge's logit were linear, full
cancellation would need roughly `−1.80 × overconfidence × self_bias = −1.80 × 2.0 × 1.5 = −5.4`.
The fitted weight is less than half that (the length weight, −4.48 against a naive −9.0, under-
corrects the same way). Our best explanation, which we haven't isolated with an ablation, is the
`[0.02, 0.98]` clip: 260 of 500 AB verdicts (and 270 of 500 BA verdicts) hit it, so `judge_logit`
saturates, and a linear correction on a saturated feature under-corrects. Second, we re-ran the
calibration with `--sim-self-bias 0`, a judge that has *no* self-preference. Its measured held-out
gap still ranged from −8.8 pts (raw) to +4.8 pts (full), a 13.6-point spread with no bias to find.
The test split has only 125 own-family pairs. We haven't computed a confidence interval, but that
control suggests a gap of about 10 points on this split size is not evidence of anything. To make
claims you need more anchors or a bootstrap CI on the gap, and we don't compute that CI yet.

**Overconfident judges stay overconfident in the parts we don't model.** ECE falls from 0.091 to
0.058 but doesn't reach 0. Even the bias-free control judge has an ECE of 0.101, because its
`confidence = σ(|z|)` is not the probability of agreeing with humans: it ignores the judge's own
noise term and the human label noise. "Unbiased" does not
mean "calibrated". Never pass a judge's self-reported confidence downstream without refitting it.

**The calibrated judge almost never says "tie".** The tie band came out at ±0.00 on seeds 0, 2
and 3 (±0.10 on seed 1). The logistic fit only sees decisive labels. The band search does see the
training ties, but widening the band loses more decisive agreements than it gains on ties. On the
seed-0 test split humans say tie 14% of the time, so three-way agreement is capped at 86% before
any other error. A better design
is an ordinal model (Bradley–Terry–Davidson or cumulative logit) that treats ties as their own
outcome.

**The self-family feature needs provenance at inference time.**
`CalibratedJudge.compare(..., model_a, model_b)` needs to know which model produced each answer.
In shadow routing and A/B tests we know this. When grading arbitrary text, we don't, and the
correction quietly falls back to `self_family = 0`.

**Synthetic anchors test the pipeline, not a real judge.** Every number in sections 3 and 4 is
from a simulated judge on simulated labels. The simulated judge's biases are linear in exactly the
features we correct for, which flatters the correction. Real judges have
biases we haven't named, such as markdown formatting, hedging, or agreement with the question's
framing.

## 6. What we got wrong elsewhere, and what we'd change

**02 shadow_router has no length control.** Its offline `heuristic_judge` scores
`len(qt & set(toks)) + min(len(toks), 30) / 10 - hedge`, so it rewards length up to 30 tokens.
In the (synthetic) demo run the candidate went 0 wins, 15 losses, 43 ties: win rate 0.0%, 95% CI
0.0%–20.4%, verdict `BLOCK`. 10 of the 15 losses were truncated `"In short: ..."` answers, so
"longer wins" happened to be the right call here. But a real judge plugged into that path inherits
whatever verbosity bias it has, uncorrected. *Change:* wrap `judge_pair` in `CalibratedJudge`.
The provenance it needs (primary vs candidate model) is already in the logged pairs.

**06 dpo_flywheel silently mixes scales.** When `LLMJudge` can't parse `SCORE: n`, it falls back
to `HeuristicJudge`. If one side of a pair parses and the other doesn't, the margin compares two
different instruments. In the demo every kept margin was well above the 1.0 floor (correction
2.5–8.5, similar_up 3.5–6.7, teacher 4.5–7.9), so the floor never decided anything. *Change:*
log the judge source for each score and drop mixed-source pairs. Also, `WinRateEvaluator` has
its judge hard-typed as `HeuristicJudge`. The promotion gate should accept the same judge the
pair builder uses.

**09 drift_monitor's judge lags and double-counts.** In the synthetic demo, decay began on
2026-09-21 (the 0-based `decay_day` 20). `judge_score` was the **last** metric to alert, on
2026-09-24 (z=−3.6). Tool mix and refusal rate alerted on 09-21 (refusal rate also had a
pre-decay false alarm on 09-14). Answer length, tool errors and latency alerted on 09-22, and
empty rate on 09-23. The offline judge caps
answers under 120 characters at a score of 3 and subtracts tool errors that we pasted into its
prompt. Partly, then, it re-measures signals we already track, just later. *Change:* drop the
tool-error line from the prompt, and alert on the judge score only after regressing out
length. Otherwise the judge is an expensive length metric.

**14 contamination's offline adjudicator works on a cliff.** The mock judge returns YES at ≥55%
content-word overlap. In the demo it upgraded three items (`suspicious` 4 → 1) and left
`q04-paraphrase` suspicious at 50% overlap (as the mock reports it, rounded). That is the right *shape* for an upgrade-only
judge. But any real model put in that slot should be audited on known paraphrase and
non-paraphrase pairs first, in the same way as section 3.

**General rule we'd now enforce:** no module may consume a judge score unless that judge has a
`calibration.json` fitted on anchors from the same distribution. Today only module 03 meets that
bar.

## Reproducibility

All commands are run from the repo root with the locked environment. Outputs go to gitignored
`out/` directories. Note that the 03 script also writes `examples/03-calibrated-judge/anchors.jsonl`.

```bash
# 03: anchors, audit, held-out calibration, unbiased control
bash examples/03-calibrated-judge/run.sh
cat examples/03-calibrated-judge/out/{audit.md,calibration.md,calibration.json}

# seed sweep (seed changes split + simulated noise)
cd examples/03-calibrated-judge
for s in 1 2 3; do
  uv run --no-sync evalkit calibrated-judge calibrate --anchors anchors.jsonl \
    --judgments out/j_seed$s.jsonl --rejudge --seed $s
done
# self-preference noise floor: judge with no self-bias
uv run --no-sync evalkit calibrated-judge calibrate --anchors anchors.jsonl \
  --sim-self-bias 0 --judgments out/j_noself.jsonl --rejudge
cd ../..

# 02 shadow router (mock judge, both orders)
bash examples/02-shadow-router/run.sh   # report in examples/02-shadow-router/out/report/

# 06 DPO flywheel (heuristic judge margins)
bash examples/06-dpo-flywheel/run.sh    # examples/06-dpo-flywheel/out/pairs/{manifest.json,pairs.jsonl}

# 09 drift monitor (mock judge at 5% sample)
bash examples/09-drift-monitor/run.sh   # alerts in examples/09-drift-monitor/out/alerts.jsonl

# 14 contamination (mock adjudicator)
bash examples/14-contamination-checker/run.sh

# run against a real judge on the same anchors (NOT run for this post; needs an API key,
# and self-preference is unmeasurable with --judge-family gpt on these anchors)
uv run --no-sync evalkit calibrated-judge audit --anchors examples/03-calibrated-judge/anchors.jsonl \
  --llm openai:gpt-4o-mini --judge-family gpt --judgments j.jsonl --rejudge --pointwise
```

The source files quoted are `calibrated_judge/{judge,anchors,audit,calibrate}.py`,
`shadow_router/report.py`, `dpo_flywheel/{pairs,filters,nightly}.py`,
`drift_monitor/scorers.py`, `contamination/judge.py` and `edge_case_gen/{llm_gen,validate}.py`,
all under `src/evalkit/`.
