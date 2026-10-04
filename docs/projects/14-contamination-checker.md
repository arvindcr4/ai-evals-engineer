# 14 — Dataset Contamination Checker

> N-gram and embedding similarity scanner that ensures your golden eval dataset
> hasn't leaked into your fine-tuning corpora. Evaluating on training data gives
> false confidence.

Package: `evalkit.contamination` · CLI: `evalkit contamination index|scan|decontaminate`

## What and why

A model that saw an eval item during fine-tuning can answer it from memory, so
the score measures recall, not capability. Leaks are rarely clean copies: a
scraped forum post restates the question, a chat log includes it as a user turn,
a study guide edits a few words, or a bulletin quotes just the answer. One
detector is not enough, so this system runs three with different trade-offs and
reports which one caught each item:

| detector | catches | misses | cost |
|---|---|---|---|
| **n-gram** (13-gram, GPT-3 style) | verbatim copies, quoted spans, partial overlaps | any edit every <13 words, paraphrases, short items | very cheap |
| **MinHash + LSH** (3-word shingles) | lightly edited copies | paraphrases | cheap |
| **embedding** (hashed TF-IDF, or a real `/embeddings` model) | paraphrases, short Q/A facts | — (noisiest; needs a higher bar) | moderate |

Statuses are graded per eval item: **contaminated** (strong evidence),
**suspicious** (needs a look), **clean**. Every flag carries a human-readable
reason and the training doc id(s) responsible.

## Design

- **Normalization** (`text.py`): Unicode NFKC, casefold, every non-alphanumeric
  run collapsed to one space, so `TRAIN LEAVES PUNE!!!` matches `train leaves Pune`.
- **Hashed n-grams**: each token gets a stable 64-bit blake2b hash (bounded LRU
  cache); n-gram hashes are a polynomial rolling hash computed vectorised in
  numpy. The eval index maps n-gram hash → eval items. Items shorter than
  `--n` but at least `--min-n` (8) tokens use their full length as n; shorter
  ones skip the n-gram check (recorded as a note) and rely on the other detectors.
- **Per item n-gram stats**: union of matched n-gram positions across the whole
  corpus (overlap ratio), the longest run of n-grams consecutive in *both* the
  item and one training doc (converted to a common word span), the number of
  matching docs, and their ids (list capped by `max_matches`, count is not).
- **Windows**: training docs are cut into overlapping word windows for MinHash
  and embeddings. Auto size is 1.5× the 90th-percentile eval item length with a
  quarter-window stride, so a verbatim copy always fits inside one window and a
  long document does not dilute similarity.
- **MinHash + LSH** (`minhash.py`): 128 universal hashes `(a·x+b) mod (2³¹−1)`
  in numpy (products fit in uint64), 64 bands × 2 rows for high recall
  (P[candidate | J=0.2] ≈ 0.93). Candidates are verified on the full signature
  and scored as estimated **containment** of the eval item in the window
  (`|A∩B| = J(|A|+|B|)/(1+J)`), not Jaccard, since windows are longer than items.
- **Embeddings** (`embed.py`): `HashedEmbedder` maps each content word to a
  cached sparse vector (the word + its character 4-grams, sign-hashed into 4096
  dims, IDF fitted on the eval set), sums them with sublinear TF and
  L2-normalises; cosine is a numpy matmul per batch of windows. Word bigrams are
  deliberately excluded (paraphrases don't keep them). `OpenAIEmbedder` is the
  hook for real embeddings via any OpenAI-compatible `/embeddings` endpoint
  (`--embedder openai:text-embedding-3-small` or `--embedder 'http://host/v1|bge-m3'`).
- **Calibrated thresholds**: cosine thresholds come from the embedder unless set:
  lexical hashed vectors score the demo's hand-written paraphrases ≈0.45 against
  ≤0.27 for its hand-written same-topic distractors (suspicious 0.40 / contaminated
  0.80; model-written same-topic text reaches 0.73, see the real-model run below); neural embedders default to
  0.85 / 0.95. Other defaults: n-gram overlap ≥ 50% → contaminated, any 13-gram
  hit → suspicious; containment ≥ 0.8 → contaminated, ≥ 0.5 → suspicious.
- **Streaming and memory**: records are read lazily (JSONL with configurable
  `--text-field`s, chat `messages`, or plain text one doc per line) and processed
  in batches of `batch_docs`; memory is the eval index + one batch + per-item
  aggregates. Throughput on a laptop: ~1.5M tokens/s n-gram only, ~110k tokens/s
  with all three detectors on the hashed embedder.
- **Optional LLM adjudication** (`judge.py`, `--llm <spec>`): each suspicious
  item's best-matching training window goes to a model asking whether it is a
  restatement; `YES` upgrades to contaminated, `NO` is recorded. Offline,
  `--llm mock` uses a deterministic content-word-overlap responder.
- **Outputs**: `report.json` (summary, corpus stats, config, per-item results),
  `report.md`, an optional `--clean-eval` file with only the clean eval rows, and
  `decontaminate` writing the training corpus with flagged docs dropped
  (`--mode drop`) or annotated with `_contamination: {level, eval_ids}`
  (`--mode flag`), at `--level contaminated|suspicious`. `--fail-on` makes `scan`
  a CI gate.

## How to run

```bash
./examples/14-contamination-checker/run.sh

# or step by step
uv run --no-sync evalkit contamination index --eval eval.jsonl --out index.json
uv run --no-sync evalkit contamination scan --index index.json \
    --train corpus-*.jsonl --text-field text --out out/ --fail-on contaminated
uv run --no-sync evalkit contamination decontaminate --index index.json \
    --train corpus.jsonl --out corpus.clean.jsonl --mode drop --level suspicious
# real embeddings + real judge
uv run --no-sync evalkit contamination scan --eval eval.jsonl --train corpus.jsonl \
    --embedder openai:text-embedding-3-small --llm openai:gpt-4o-mini --out out/
```

```python
from evalkit.contamination import ContaminationScanner, EvalIndex
from evalkit.contamination.text import iter_records

idx = EvalIndex.from_jsonl("eval.jsonl")
report = ContaminationScanner(idx).scan(iter_records("train.jsonl"))
print(report.summary)
```

## Example

`examples/14-contamination-checker/` has 12 eval items and a 15-doc training
corpus with planted leaks: two verbatim copies inside longer docs (`q01`, `q02`
— the latter with "Top answer:" inserted between Q and A), a chat-format copy
(`q07`), a light edit that changes one word every ~10 so no 13-gram survives
(`q03`), two paraphrases (`q04`, `q05`), a doc quoting only the answer (`q06`),
a two-word Q/A fact present in a quiz (`q12`), plus same-topic distractors for
the clean items (`q08`–`q11`).

Which method catches what (from the real run):

| item | planted as | n-gram only | all detectors | + mock judge |
|---|---|---|---|---|
| q01, q02, q07 | verbatim / chat copy | contaminated | contaminated | contaminated |
| q03 | light edit | **clean (missed)** | contaminated (MinHash 0.67 + cosine 0.80) | contaminated |
| q04, q05 | paraphrase | **clean (missed)** | suspicious (cosine 0.46 / 0.48) | q05 upgraded, q04 kept suspicious |
| q06 | answer quoted | suspicious (33% of 13-grams) | suspicious | upgraded |
| q12 | short fact | **skipped (too short)** | suspicious (cosine 0.51) | upgraded |
| q08–q11 | same-topic distractors | clean | clean | clean |

Real `scan` output:

```
12 eval items: 4 contaminated, 4 suspicious, 4 clean (by method: {'ngram': 4, 'minhash': 4, 'embedding': 8, 'judge': 0}) -> examples/14-contamination-checker/out/scan/report.md
  contaminated q01-exact      via ngram,minhash,embedding
  contaminated q02-exact      via ngram,minhash,embedding
  contaminated q03-edited     via minhash,embedding
  suspicious   q04-paraphrase via embedding
  suspicious   q05-paraphrase via embedding
  suspicious   q06-partial    via ngram,embedding
  contaminated q07-chat       via ngram,minhash,embedding
  suspicious   q12-short      via embedding
```

`--methods ngram`:

```
12 eval items: 3 contaminated, 1 suspicious, 8 clean (by method: {'ngram': 4, 'minhash': 0, 'embedding': 0, 'judge': 0})
```

Report excerpt (`out/scan/report.md`):

```
| id | status | methods | n-gram overlap | longest span | containment | cosine | docs |
|---|---|---|---|---|---|---|---|
| q01-exact | **contaminated** | ngram, minhash, embedding | 100% | 38 | 1.00 | 0.75 | web-0001 |
| q07-chat | **contaminated** | ngram, minhash, embedding | 100% | 35 | 0.97 | 0.96 | chat-0007 |
| q02-exact | **contaminated** | ngram, minhash, embedding | 62% | 27 | 0.92 | 0.90 | web-0002 |
| q03-edited | **contaminated** | minhash, embedding | 0% | 0 | 0.67 | 0.80 | web-0003 |
| q06-partial | **suspicious** | ngram, embedding | 33% | 22 | 0.45 | 0.50 | web-0006 |
| q12-short | **suspicious** | embedding | 0% | 0 | 0.00 | 0.51 | web-0014 |
| q05-paraphrase | **suspicious** | embedding | 0% | 0 | 0.00 | 0.48 | web-0005 |
| q04-paraphrase | **suspicious** | embedding | 0% | 0 | 0.00 | 0.46 | web-0004 |

- **q06-partial** (suspicious)
  - [suspicious] partial overlap: 10/30 13-grams (33%), longest common span 22 words
  - [suspicious] embedding cosine 0.50 with web-0006
```

Decontamination:

```
{"documents": 15, "kept": 11, "dropped": 4, "flagged": 0, "dropped_ids": ["web-0001", "web-0002", "web-0003", "chat-0007"]}
{"documents": 15, "kept": 15, "dropped": 0, "flagged": 8, "dropped_ids": []}
```

Tests: `uv run --no-sync pytest tests/test_contamination*.py -q` (24 tests:
normalization, hash stability, chat/JSONL/text readers, MinHash estimate
accuracy, containment math, LSH recall/precision, embedder ranking and the
`/embeddings` hook via a fake transport, each detector's catch/miss behaviour,
thresholds, index round-trip and scan-time overrides, decontaminate drop/flag,
judge upgrade/reject, and the CLI end-to-end including the `--fail-on` gate),
plus `tests/test_contamination_real.py` (23 real-model regressions: judge verdict
parsing, judge cost accounting, judge API failures, leak-generation JSON extraction
and retry, recall scoring and the `leaktest` CLI), and
`tests/test_contamination_adversarial.py` (11 regressions: OpenAI-style
content-part messages, non-object JSONL rows, non-contiguous span inflation,
uncapped doc counts, degenerate configs, empty eval set, blocked embedding
similarity, and refusing to decontaminate a corpus onto itself).

## Real-model run (DeepSeek, Oct 2026)

**What `--llm` does here.** The scanner itself never calls a model; `--llm` only
adjudicates *suspicious* items (`judge.py`). There are no embeddings at DeepSeek,
so the embedding detector stays on the hashed lexical embedder. To measure
detector recall on *real* paraphrases instead of the 2 hand-written ones, a new
`evalkit contamination leaktest` subcommand (`leakgen.py`) has the model write,
for every eval item, five documents (`light_edit`, `paraphrase`,
`heavy_paraphrase`, `answer_only`, and a same-topic `negative` that should not
leak), adds a locally built `verbatim` copy as a positive control, scans one
corpus per kind, runs the judge exactly as `scan --llm` does, and also asks the
judge about every (item, document) pair directly.

Command (`examples/14-contamination-checker/real_run.sh`, outputs in `out/real/`,
committed excerpts in `real_output/`):

```bash
# from the repo root; EX=examples/14-contamination-checker, OUT=$EX/out/real
set -a; source ~/TradingAgents/.env; set +a
uv run --no-sync evalkit contamination index --eval $EX/eval.jsonl --out $OUT/index.json
uv run --no-sync evalkit contamination scan --index $OUT/index.json \
  --train $EX/train.jsonl --out $OUT/scan-judged --llm deepseek:deepseek-flash
uv run --no-sync evalkit contamination leaktest --eval $EX/eval.jsonl --synth 28 \
  --llm deepseek:deepseek-flash --judge deepseek:deepseek-flash --out $OUT/leaktest --reuse
# threshold probe on the SAME items and leaks (REAL_RUN_PROBE=1 in real_run.sh):
uv run --no-sync evalkit contamination leaktest --eval $OUT/leaktest/eval.jsonl \
  --leaks $OUT/leaktest/leaks.jsonl --llm deepseek:deepseek-flash \
  --judge deepseek:deepseek-flash --cosine-suspicious 0.25 --out $OUT/leaktest-cos025
```

`real_output/` holds the full items (`leaktest.eval.jsonl`) and leaks
(`leaktest.leaks.jsonl`), so every detector column below can be recomputed offline
for free: pass `--llm mock` and leave out `--judge`. The judge columns
are in `leaktest*.summary.json`. The per-item pipeline verdicts on non-leaks, and
on leaks the judge did not upgrade, are in `pipeline-judge-verdicts.json`.

- Models: `deepseek-flash` (thinking off, temperature 0) as leak writer, eval-item
  writer (`--synth 28`) and judge. The judge is the same model that wrote the
  leaks, so the judge columns are optimistic; no second model family was used.
- Sample: 40 eval items (the 12 demo items + 28 model-written ones in
  `real_output/leaktest.eval.jsonl`) x 6 kinds = 240 documents. Small: one leak
  per item per kind, so every percentage below has a +/- ~10-15 pt interval.
- Total API spend for this system: **about $0.11**. Recorded in saved files:
  demo judge $0.0003 (`scan-judged.report.md`), the re-score $0.0303
  (`leaktest.recall.md`, 341 calls) and the 0.25 threshold probe $0.0340
  (`leaktest-cos025.recall.md`, 381 calls), $0.065 in all. Not in any saved file
  (the re-score overwrote that run's `recall.*`), so taken from the console:
  the first generation + scoring run, $0.044, and a 3-item smoke run plus two
  format probes, about $0.004. The $0.044 is plausible: about $0.02 of generation
  (about 16k output tokens in `leaktest.leaks.jsonl`) plus about $0.03 of judging,
  the same as the re-score.

**Demo corpus with the real judge** (`real_output/scan-judged.report.md`):

```
judge deepseek-flash: 4 call(s), 4 YES / 0 NO / 0 unparsed / 0 error(s), $0.0003
12 eval items: 8 contaminated, 0 suspicious, 4 clean (by method: {'ngram': 4, 'minhash': 4, 'embedding': 8, 'judge': 4})
```

All four suspicious items (q04/q05 paraphrases, q06 answer quoted, q12 short fact)
were upgraded, each with a correct one-sentence reason ("The passage explicitly
states 'the capital of peru is lima'..."). The offline overlap mock upgraded
three and kept q04 (a real paraphrase) as suspicious, so the real judge is
strictly better on this corpus; the clean items never reach the judge.

**Leak recall** (`real_output/leaktest.recall.md`, default thresholds):

| kind | items | any detector | contaminated | n-gram | MinHash | embedding | own-doc cosine min / median / max | scan + judge -> contaminated | judge YES on every pair |
|---|---|---|---|---|---|---|---|---|---|
| verbatim | 40 | 100% | 100% | 98% | 100% | 100% | 0.52 / 0.90 / 0.93 | 100% | 100% (40/40) |
| light_edit | 40 | 100% | 57% | 50% | 70% | 100% | 0.40 / 0.81 / 0.96 | 100% | 100% (40/40) |
| paraphrase | 40 | 98% | 12% | 0% | 0% | 98% | 0.39 / 0.65 / 0.85 | 98% | 100% (40/40) |
| heavy_paraphrase | 40 | 28% | 0% | 0% | 0% | 28% | 0.07 / 0.31 / 0.72 | 28% | 98% (39/40) |
| answer_only | 40 | 95% | 28% | 10% | 8% | 95% | 0.31 / 0.67 / 0.95 | 95% | 98% (39/40) |
| negative | 40 | 28% | 0% | 0% | 0% | 28% | 0.07 / 0.32 / 0.73 | 2% | 2% (1/40) |

Same leaks with `--cosine-suspicious 0.25` (`real_output/leaktest-cos025.recall.md`):

| kind | any detector | scan + judge -> contaminated |
|---|---|---|
| paraphrase | 100% | 100% |
| heavy_paraphrase | 72% | 70% |
| answer_only | 100% | 100% |
| negative | 70% | 8% (3/40) |

What the real model revealed, compared with the offline mock and the hand-planted demo:

- **The detector story holds for ordinary paraphrases.** n-gram and MinHash catch
  0% of model-written paraphrases; the hashed embedder flags 98%, as the demo's two
  hand-written paraphrases suggested. n-gram alone misses half of the light edits.
- **Heavy paraphrases defeat every detector.** Prose restatements with fresh
  vocabulary ("twenty minutes to ten", "seventy-eight kilometres per hour") score a
  median cosine of 0.31, the same as same-topic non-leaks (0.32). No cosine
  threshold separates them: the documented calibration "paraphrases ~0.45 vs <=0.27
  for same-topic distractors" came from hand-picked examples and does not hold for
  model-written same-topic text (negatives reach 0.73). The default bar flags 28% of
  heavy paraphrases and 28% of non-leaks alike.
- **The judge supplies the precision, the embedder caps the recall.** The judge says
  YES to 98-100% of every leak kind and to 1/40 non-leaks, so scan + judge keeps
  only 1 of 11 suspicious non-leaks. Because only suspicious items reach the judge,
  end-to-end recall on heavy paraphrases is the embedder's 28%. Lowering
  `--cosine-suspicious` to 0.25 sends 70% of non-leaks to the judge (141 instead
  of 101 pipeline judge calls) and lifts heavy-paraphrase recall to 70%, with 8%
  of negatives flagged at the end. Only part of the rise from 2% to 8% comes from the threshold.
  s04 reached the judge only at 0.25. s13 was already suspicious at the default
  bar (same cosine, 0.44), but the judge said NO in the default run and YES in the
  probe.
- **The "false positives" were mostly mislabeled, but the judge is not consistent
  on them.** The three negatives convicted in the probe each give away part or all
  of the answer, so the generator broke its own "do not reveal the answer" rule. s24
  names Marbury v. Madison. s04 says Marshall Plan aid went to Western Europe, but
  not its purpose. s13 names Washington and 1789, but leaves the link between them
  implicit. Only s24 is a clear-cut leak. The same judge answered NO on the
  full s04 and s13 documents in the pairwise pass of both runs (1/40 negatives YES).
  For s13 in the default run it even claimed the passage does not "give the year
  1789", which it does. So read the 8% as 1 clear leak + 2 borderline cases on
  which the judge flips, not as 3 judge-confirmed leaks. Its NO verdicts on leaks
  (pairwise s20 heavy_paraphrase and s15 answer_only in the re-score, and s20 + s23
  heavy_paraphrase in the probe; pipeline s23 in the probe) were borderline "general
  explanation, not the item" cases.
- **Not deterministic at temperature 0.** Two saved runs judged the same 240
  pairs: the re-score (`leaktest.summary.json`) and the probe
  (`leaktest-cos025.summary.json`). Between them, 2 pairwise verdicts flipped:
  heavy_paraphrase s23 went YES -> NO (39 -> 38/40), and answer_only s15 went
  NO -> YES (39 -> 40/40). The pipeline verdict on negative s13 also flipped, as
  above. The first, unsaved run had shown heavy_paraphrase 38/40 and answer_only
  40/40. Report judge numbers as +/- a couple of items.
- **Output format.** deepseek-flash started every judge reply with a bare `YES`
  or `NO`: 0 unparsed across the 726 judge calls in saved files (4 demo + 341
  re-score + 381 probe; 101 + 141 pipeline replies in `pipeline-judge-verdicts.json`
  and the summaries). Generation returned plain JSON (no fences) with 0 failures
  (all 200 model-written documents are present in `leaktest.leaks.jsonl`). The
  parsing fixes below are hardening for other models.

Code changes from the real run (each with a regression in `tests/test_contamination_real.py`):

1. Judge verdict parsing only accepted a bare first-line `YES`; `**YES**`,
   `Verdict: NO`, fenced or trailing verdicts were silently treated as NO. Now
   `parse_verdict` handles them and an unreadable reply is counted as `unparsed`
   and left suspicious with a note.
2. The judge spent money with no record of it. `report.json` now carries
   `judge_usage` (calls, tokens, USD, YES/NO/unparsed/error counts), shown in
   `report.md` and on the CLI.
3. One failed judge call (timeout after retries, 4xx) aborted the whole scan with no
   report written. Errors are now recorded per item and the scan completes.
4. New `leaktest` (generation with fenced/prose-tolerant JSON extraction and one
   repair retry, per-kind recall, pipeline and pairwise judge scoring).

## Limitations

- The hashed embedder is lexical: it finds paraphrases that reuse content
  words (and morphological variants via char n-grams) but not translations or
  restatements with fresh vocabulary (28% recall on model-written heavy
  paraphrases in the real run above, indistinguishable from same-topic text). Use `--embedder` with a real model for
  that, and recalibrate `--cosine*` on a sample of known leaks/non-leaks.
- Cosine thresholds for the hashed embedder were calibrated on small examples
  (the real run found same-topic non-leaks up to 0.73, median 0.32);
  same-topic corpora (e.g. a medical eval vs a medical corpus) raise the
  background and need a higher `--cosine-suspicious`.
- MinHash estimates have sampling noise (±~0.05 at 128 permutations); items
  near a threshold can flip with `--num-perm`/seed.
- Evidence snippets are the normalized (lowercased, punctuation-stripped) text.
- Python-level per-window work makes the all-detector scan ~110k tokens/s; for
  billion-token corpora run `--methods ngram,minhash` first or shard the corpus
  across processes and merge the JSON reports.
- The mock judge is a deterministic overlap heuristic for offline demos, not
  a substitute for a real model's judgement.
