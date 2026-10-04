# 12 — Synthetic Edge-Case Generator

> A pipeline that uses a frontier model to systematically generate out-of-distribution
> inputs and boundary conditions for your golden dataset. Human annotators miss the
> edge cases that break production.

`evalkit edge-case-gen` turns a small YAML **task spec** (task description, input schema,
seed examples) into a labelled, deduplicated, provenance-tagged **golden-set JSONL** of edge
cases. It also reports how much of the input space the set covers and runs a system under
test against it, so you can see which edge cases break that system.

## Why

Golden sets are usually sampled from production traffic or written by annotators. Both
sources favour typical inputs. The inputs that actually page you are rare ones: a
`null` amount, a date in `DD/MM/YYYY`, a Cyrillic `а` that looks like a Latin `a`, an
empty note, a valid category combined with an out-of-range amount. Covering them
needs a systematic enumeration, not more sampling.

## Design

```
spec.yaml ──► generate ──► candidates.jsonl ──► validate ──► golden.jsonl ──► coverage
              (rules + LLM)                     (schema, dedupe,            └──► probe <system>
                                                 novelty, labels)
```

| Module | Role |
|---|---|
| `spec.py` | `TaskSpec`/`FieldSpec` (string, integer, number, date, enum, boolean, with min/max, lengths, choices, pattern), YAML loader, `validate_input`. It rejects a spec whose own seeds break the schema. |
| `generators.py` | `RuleGenerator`, one deterministic method per axis, plus `field_levels()`, which gives the named test levels for each field (`nominal`, `min`, `below_min`, `over_max_length`, `invalid_choice`, `date_format`, …). |
| `pairwise.py` | AETG-style greedy all-pairs **covering array** and a pairwise-coverage metric. |
| `llm_gen.py` | `LLMGenerator`: a structured JSON prompt per axis, a fence- and prose-tolerant JSON parser, and an offline `mock_responder` that returns plausible JSON. It also contains `llm_label`, which proposes labels. |
| `claims.py` | `claim_holds`: mechanically checks that a case really contains the edge its category names (zero-width/RTL/combining/fullwidth/emoji characters, markup, `at_max_length`, `above_max`, …). Returns `None` for categories that cannot be checked. |
| `validate.py` | `Validator` (structural rejects, exact and near-duplicate removal, novelty against seeds) and `propose_labels` (schema, then oracle, then LLM, plus the human-review flag). |
| `coverage.py` | Counts per axis, category, field and generator, missing expected categories, and pairwise level coverage (overall and per axis). |
| `probe.py` | Runs a callable on the golden set and records pass, fail or crash. It groups failures by axis and category and marks a failure as *provisional* when its label still needs review. |

**Generation axes** (all rule-based and deterministic for a given `--seed`):

- **boundary**: min, max, below_min, above_max, zero, negative, overflow (`2**63`, `1e308`,
  `9999-12-31`), empty, at/over max length, whitespace-only, missing field, null, wrong type
  (`"1,234.5"` as a string), invalid enum choice, invalid date (`YYYY-02-30`), and leap day.
- **format**: homoglyphs, RTL override plus Arabic text, emoji, odd whitespace (tabs, NBSP,
  newlines), casing, deterministic typos, mixed languages, zero-width joins, combining marks,
  full-width digits, and alternative date formats.
- **semantic**: ambiguous (numbers replaced with "a few"), contradictory (text that asserts a
  different value than a structured field), multi-intent, out-of-scope (taken from the spec or
  generic), and negated.
- **adversarial but benign**: long input, nested and escaped quoting, markup or JSON inside
  text, instruction-like text, and delimiter collisions.
- **combinatorial**: a strength-2 covering array over every field's levels. For the example
  spec (6 fields, 4–7 levels each, 372 level pairs), 39 rows cover every pair. One-factor
  boundary cases alone reach only about 21 %.

**LLM generator.** Pass `--llm <spec>` to add model-written cases per axis, using
`get_llm` (`openai:gpt-4o-mini`, `deepseek:…`, `base|model`, …). The prompt contains the
spec as JSON and asks for `{"cases":[{"input","category","description"}]}`. Unparseable
replies and malformed cases are skipped with a warning; they never crash the run. With
`--llm mock`, the deterministic responder reads the same prompt and returns axis-specific
cases that the rules do not produce (hypothetical or sarcastic phrasing, a bidi number,
JSON inside the text, a plural enum value, and others).

**Validation rules:**

- Rejected: a non-object input, unknown fields, an exact copy of a seed, an exact
  duplicate, a near duplicate, or a case with novelty below `--min-novelty`.
  - **Near duplicate** means 4-gram shingle Jaccard ≥ `--near-dup` on the text field, but
    only within a bucket of identical structured fields, the same schema-error kinds and the
    same text-length scale. Without the bucket, `at_max_length` and `over_max_length`
    differ by one character and would collapse into one case. Shingles are NFC-normalised
    but whitespace is *not* collapsed, so a CRLF variant and an ideographic-space variant of
    the same note stay distinct format cases.
- **Kept on purpose:** schema-*invalid* cases. Testing those is the point. Each one carries
  `schema_valid` and `schema_errors`.
- **Labels:**
  - A schema-invalid case gets `spec.invalid_label` (source `schema`).
  - Any other case is labelled by the reference oracle (`--oracle file.py:fn`).
  - Without an oracle, the LLM proposes the label. With an oracle, the LLM cross-checks it.
  - `needs_human_review` is set, with reasons, when there is no label, the oracle crashed,
    the oracle and the LLM disagree, the LLM's confidence is below `--min-confidence`, or the
    case is on the semantic axis. Semantic cases are always flagged because an oracle that
    reads structured fields cannot judge what the text means.
- **Provenance**: generator (`rule:<axis>` or `llm:<model>`), seed index, spec name,
  12-character spec fingerprint, and RNG seed. The ID is a stable hash of the input, axis and
  category.

## How to run

```bash
examples/12-edge-case-generator/run.sh          # fully offline (LLM=mock)
LLM=openai:gpt-4o-mini examples/12-edge-case-generator/run.sh

uv run --no-sync evalkit edge-case-gen generate --spec spec.yaml --out cands.jsonl [--llm mock] [--axes boundary,format] [--seed 7]
uv run --no-sync evalkit edge-case-gen validate --spec spec.yaml --in cands.jsonl --out golden.jsonl --oracle expense_system.py:reference [--label-llm mock]
uv run --no-sync evalkit edge-case-gen coverage --spec spec.yaml --in golden.jsonl [--min-pairwise 0.95]
uv run --no-sync evalkit edge-case-gen probe    --spec spec.yaml --in golden.jsonl --system expense_system.py:naive_triage [--max-fail-rate 0.05]
```

`coverage --min-pairwise` and `probe --max-fail-rate` exit with status 1 when the threshold is
missed, so either can act as a CI gate. The probe failure rate is computed over *scored*
cases (labelled ones plus any crash); unlabelled cases that ran cleanly are reported
separately and never count as passes, so an unlabelled golden set cannot pass the gate.

The example is an **expense-claim triage** task: approve, escalate, reject or invalid, with a
note, amount, currency, date, category and receipt flag. `expense_system.py` has three
functions:

- `reference`: the written policy.
- `naive_triage`: plausible production code that agrees with `reference` on every seed.
- `robust_triage`: validates the claim first, then applies the policy.

## Sample output (real run, `examples/12-edge-case-generator/run.sh`)

```
generated 127 candidate cases -> out/candidates.jsonl
  llm:mock-1=16, rule:adversarial=5, rule:boundary=37, rule:format=10, rule:pairwise=39, rule:semantic=20
validated: kept 126, rejected 1 {'duplicate': 1}
  schema-invalid kept on purpose: 72; needs_human_review: 24 -> out/golden.jsonl
coverage for expense-claim-triage: 126 cases
  by axis:     adversarial=9, boundary=37, combinatorial=39, format=17, semantic=24
  by field:    amount=12, category=4, currency=5, employee_note=54, expense_date=9, has_receipt=3
  generators:  llm:mock-1=16, rule:adversarial=5, rule:boundary=37, rule:format=9, rule:pairwise=39, rule:semantic=20
  schema:      valid=54 invalid=72
  labels:      invalid=72, approve=44, reject=2, escalate=8
  review:      24 flagged needs_human_review
  categories:  100% of expected axis categories hit
  pairwise:    372/372 level pairs = 100.0%
    adversarial    alone: 2.7%
    boundary       alone: 21.5%
    combinatorial  alone: 100.0%
    format         alone: 9.4%
    semantic       alone: 2.7%
probe: 126 cases, 30 wrong, 44 crashed (58.7% of 126 scored, 0 unlabeled); confirmed-label subset: 71.6% of 102
  by axis:
    adversarial     33.3%  (fail=2 crash=1 / 9)
    boundary        81.1%  (fail=16 crash=14 / 37)
    combinatorial   74.4%  (fail=10 crash=19 / 39)
    format          64.7%  (fail=2 crash=9 / 17)
    semantic         4.2%  (fail=0 crash=1 / 24)
  worst categories:
    boundary/missing                   100.0% of 6
    boundary/null                      100.0% of 6
    boundary/above_max                 100.0% of 3
    boundary/invalid_choice            100.0% of 3
    boundary/wrong_type                100.0% of 3
    boundary/invalid_date              100.0% of 2
    boundary/overflow                  100.0% of 2
    adversarial/long_input             100.0% of 1
    adversarial/nested_quoting         100.0% of 1
    adversarial/very_long_word         100.0% of 1
  sample failures:
    ec-7694597147 boundary/over_max_length: expected=invalid got=approve
    ec-e3b1f4e578 format/homoglyph: UnicodeEncodeError: 'latin-1' codec can't encode characters in position 1-2: ordinal not in range(256) (line 28)
    ec-4fe5cbaa15 boundary/null: AttributeError: 'NoneType' object has no attribute 'strip' (line 25)
    ec-937930e8b5 boundary/whitespace_only: expected=approve got=invalid
    ec-76b2835213 boundary/wrong_type: AttributeError: 'int' object has no attribute 'strip' (line 25)
    ec-4796a94408 boundary/missing: KeyError: 'employee_note' (line 25)
    ec-87eb7df606 boundary/below_min: expected=invalid got=approve
    ec-1c717cebb2 boundary/above_max: expected=invalid got=reject
    ec-7926be15c7 boundary/null: TypeError: float() argument must be a string or a real number, not 'NoneType' (line 32)
    ec-d54db5bdc1 boundary/zero: expected=invalid got=approve
probe: 126 cases, 0 wrong, 0 crashed (0.0% of 126 scored, 0 unlabeled); confirmed-label subset: 0.0% of 102   # robust_triage
```

`naive_triage` agrees with the reference on every seed, yet it fails 58.7 % of the generated
edge cases:

- It crashes on non-Latin-1 text.
- It approves negative and above-maximum amounts.
- It crashes on `null` values and missing fields.

The `whitespace_only` failure is a different kind of finding: it exposes a **spec gap**. The
schema allows a note of five spaces, but the naive code calls it invalid. That is exactly the
question a human reviewer should settle before the label goes into the golden set.

## Tests

`tests/test_edge_case_gen.py` (23 tests) and `tests/test_edge_case_gen_adversarial.py`
(8 regression tests from an adversarial review) cover:

- Schema validation of every field type, including bool-as-int and impossible dates.
- Covering-array completeness, size bounds and determinism.
- Determinism of each generator axis.
- Robustness of the LLM path: garbage replies, malformed cases, out-of-set labels.
- Dedupe, novelty and near-duplicate bucketing.
- Labelling precedence: schema, then oracle, then LLM, with disagreement and crash review
  flags.
- Pairwise coverage with and without the combinatorial axis.
- Probe outcome classification (pass, fail, crash).
- The example system's seeds-pass, edges-fail property, and a full end-to-end CLI run.
- Review regressions: optional fields absent from every seed, unlabelled cases diluting the
  `--max-fail-rate` gate, `--help` rendering, whitespace variants surviving near-dup, bare
  JSON-array LLM replies, and stale labels on re-validation.

## Limitations

- The schema is flat. There are no nested objects or lists, and no cross-field constraints
  such as "end date ≥ start date". Cross-field contradictions are only exercised through the
  semantic text rewrites.
- Pairwise coverage is strength 2 over spec-derived levels. Interactions between three or
  more factors are not guaranteed. The levels come from the first seed's values (or, for an optional field the seed omits, another seed's value or a type default).
- Semantic cases keep the oracle's structured-field label, so they always go to human review
  instead of being auto-labelled. The mock LLM's labels are hash-based placeholders. Use a
  real model with `--label-llm` for meaningful cross-checks.
- Near-duplicate detection is lexical (character shingles), not embedding-based. Paraphrases
  survive as separate cases.
- `field_levels` uses fixed perturbation tables (homoglyphs, languages, emoji). They are
  broad, but they do not cover every locale.

## Real-model run (DeepSeek, Oct 2026)

**Command:** `examples/12-edge-case-generator/real_run.sh`. It sources `~/TradingAgents/.env`
and writes to `out/real/`; small copies are in `examples/12-edge-case-generator/real_output/`.

```bash
evalkit edge-case-gen generate --spec spec.yaml --out out/real/candidates.jsonl \
  --llm deepseek:deepseek-flash --n-llm 10 --seed 7 --temperature 0
evalkit edge-case-gen validate --spec spec.yaml --in out/real/candidates.jsonl \
  --out out/real/golden.jsonl --rejects out/real/rejects.jsonl \
  --oracle expense_system.py:reference --label-llm deepseek:deepseek-flash
evalkit edge-case-gen coverage --spec spec.yaml --in out/real/golden.jsonl --json out/real/coverage.json
evalkit edge-case-gen probe --spec spec.yaml --in out/real/golden.jsonl --system expense_system.py:naive_triage
evalkit edge-case-gen probe --spec spec.yaml --in out/real/golden.jsonl --system expense_system.py:robust_triage
```

**Models:** `deepseek-flash` with thinking off is used for both roles: generator (`--llm`,
10 cases per axis over 4 axes, plus one repair round) and label cross-checker
(`--label-llm`, temperature 0). The reference oracle remains the primary labeller.
**Sample:** 40 LLM candidates (6 generation calls) and 70 label calls. This is one run on one
small spec, so the numbers below are anecdotes, not estimates.
**Cost:** the final run cost $0.0189 ($0.0138 generation + $0.0051 labels; 22.1k input and
10.2k output tokens). All development runs together cost about $0.10.

### Results (final run)

```
generated 151 candidate cases -> out/real/candidates.jsonl
  llm:deepseek-flash=40, rule:adversarial=5, rule:boundary=37, rule:format=10, rule:pairwise=39, rule:semantic=20
  llm usage: 6 calls, 9332 in / 9173 out tokens, $0.0138; phantom cases 12, repaired 8 (resent/duplicate replies dropped: 4)
  llm warning: adversarial: reply truncated at the max-token limit
  llm warning: adversarial: malformed/truncated JSON, salvaged 2 complete cases (Invalid \uXXXX escape: ...)
validated: kept 137, rejected 14 {'duplicate': 1, 'copy_of_seed': 12, 'near_duplicate': 1}
  label llm: 70 calls, 12787 in / 1071 out tokens, $0.0051; oracle/llm disagreements: 29
  llm vs rules: 27 llm cases; novelty vs rule cases mean 0.39 (median 0.45, min 0.05); novelty vs seeds llm 0.4165 vs rule 0.2646
    12 llm cases in 12 categories no rule produced: adversarial/numeric_ambiguity, adversarial/receipt_contradiction,
    boundary/escalation_threshold_exact, boundary/other_category_escalation, boundary/over_escalation_threshold,
    boundary/receipt_present_over_threshold, boundary/receipt_threshold_exact, boundary/under_threshold_no_receipt,
    format/amount_currency_mismatch, format/currency_symbol_in_note, format/note_amount_conflict, format/zero_width_space
probe (naive_triage): 137 cases, 27 wrong, 49 crashed (55.5% of 137 scored); confirmed-label subset: 73.1% of 93
  by generator: llm=40.7% of 27, rule=59.1% of 110
probe (robust_triage): 137 cases, 0 wrong, 0 crashed
```

| | offline mock (`run.sh`) | DeepSeek flash (`real_run.sh`) |
|---|---|---|
| LLM candidates → kept | 16 → 16 | 40 → 27 (68 %) |
| Rejected LLM candidates | 0 | 12 copies of a seed (all 12 first-pass phantoms; the originals stay in the candidate file even when a repair replaced them), 1 near-dup (a repaired `rtl_override`); 0 `claim_not_in_input` |
| Phantom cases on first pass (input identical to a seed or claimed edge absent) | 0 | 12 of 32 (38 %): 7 format cases that described an edge but returned the seed unchanged, plus 5 boundary "control cases" the model copied from seeds on purpose; the repair round produced 8 valid replacements (all 7 format, 1 boundary), 7 of which were kept |
| Novelty vs rule cases (mean / median) | 0.18 / 0.16 | 0.39 / 0.45 |
| Novelty vs seeds, LLM vs rule cases | 0.14 vs 0.26 | 0.42 vs 0.26 |
| LLM cases in categories no rule produces | 11 of 16 | 12 of 27 |
| naive_triage failure rate on LLM cases | 56.2 % of 16 | 40.7 % of 27 |
| robust_triage failure rate | 0 % | 0 % |
| Pairwise coverage | 100 % | 100 % (from the rule covering array, not from the LLM) |

### What the real model revealed

The fixes below are each covered by a regression test in `tests/test_edge_case_gen_real.py`.

1. **Phantom perturbations.** This was the main finding. In the final run the generator
   flagged 12 of 32 first-pass cases: 7 format cases *describe* an edge (Cyrillic homoglyph,
   RTL override, zero-width space, combining accent, fullwidth, emoji, mixed language) but
   return the seed input unchanged, and 5 boundary cases are seed copies the model labelled
   "control case". All 12 were rejected as `copy_of_seed`; none reached the claim check in
   that run. The development runs (not saved) also produced phantoms whose input *was*
   changed but still lacked the claimed edge (4 of 10 format cases in an early probe).
   Examples from those runs: "zero-width space (U+200B) after 'Café'" on plain text; "combining
   acute on e" with no combining mark; "date written as 14/03/2026" with the seed's ISO date
   unchanged; `at_max_length` "exactly 280 characters" on a 479-character note; an
   `instruction_like` case whose only change was the currency `XEU`. The mock never does
   this, so the pipeline would have labelled these cases and counted them as coverage of
   categories they do not test.
   Fixes:
   - The prompt now asks for `\uXXXX` escapes and says the input must really contain the
     edge.
   - `claims.py` checks the claim mechanically. The validator rejects failures as
     `claim_not_in_input`, and copies of a seed are rejected as before.
   - `LLMGenerator` sends one repair turn listing the phantom cases.
   - Every rule-generated case is asserted to pass its own claim check.
2. **The repair reply resends the whole list.** Asked for replacements, the model returned
   all N cases again plus the fixes. The generator now keeps only unseen inputs, at most one
   per phantom, and counts the rest as `resent`.
3. **Degenerate, truncated replies.** At least one axis per run (adversarial or format) runs
   into the token cap, either by repeating cases or by writing a very long note. The JSON then
   ends mid-`\u` escape, and the whole axis used to be lost as "unparseable". Now
   `salvage_cases` recovers every complete case object, `finish_reason=length` is reported,
   `max_tokens` is capped at 4000, and the prompt caps strings at 300 characters.
4. **Accounting.** `generate` and `validate` now print calls, tokens and `$` from
   `Completion.cost_usd`. Before this, label-LLM spend was invisible.
5. **Determinism.** `--temperature` exists now; the old value was a hard-coded 0.9. Even at
   temperature 0, DeepSeek's outputs differ between runs: the 5 development runs (outputs not
   saved) kept 18–31 LLM cases with different category mixes. Reproducibility therefore comes from the
   committed JSONL, not from re-running.
6. **Offline mock gap.** The mock's `smart_quotes` mutation is itself a phantom whenever the
   seed text has no " the ". The claim check catches it on the test spec. The mock now also
   answers a repair turn with an empty list instead of crashing.

**What the LLM cases add.** They are clearly more novel than the rule cases (0.39 mean
distance from the nearest rule case, against 0.18 for the mock). They also concentrate on
*policy* edges that the schema-driven rules cannot see, because the rules do not read the
policy text:
- exactly 75 USD without a receipt;
- exactly 1000 USD;
- a foreign currency that crosses a threshold only after conversion;
- a note whose amount or currency contradicts the structured fields.

They found one realistic production bug the rules hit only through synthetic tables. The
model habitually writes em-dashes (`—`), `€` and `₹`, and 11 of the 27 LLM cases crashed
`naive_triage`'s latin-1 audit log, including boundary cases that were not meant to test
encoding at all.

**What they don't add.** They contributed nothing to pairwise coverage, which comes entirely
from the covering array. They rarely produced schema-invalid inputs on purpose (null,
missing, wrong-type), and those inputs are the bulk of `naive_triage`'s crashes. The LLM
cases' lower failure rate (40.7 % against 59.1 % for rule cases) reflects this; it is not
evidence that the LLM cases are worse. Axis drift also occurs: the model filed
`bidi_override` and `invisible_unicode` under the semantic axis in one run.

**LLM as label cross-checker.** It is unreliable here. On 29 of 70 schema-valid cases, the
model disagreed with the written-policy oracle. In 18 of them it said `escalate` where the
policy says `approve`, including plain 142.50 USD meals claims with a receipt. It also called
exactly 75 USD without a receipt `reject`, although the policy says "over 75". Every
disagreement I spot-checked was a model error. The disagreements are correctly routed to
`needs_human_review`, but they inflate the review queue. With this model, use the LLM only
as a second opinion; do not let it be the primary labeller.
