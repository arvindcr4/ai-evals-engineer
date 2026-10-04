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
