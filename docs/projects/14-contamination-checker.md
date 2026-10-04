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
  lexical hashed vectors score paraphrases ≈0.45 against ≤0.27 for same-topic
  distractors (suspicious 0.40 / contaminated 0.80); neural embedders default to
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
plus `tests/test_contamination_adversarial.py` (11 regressions: OpenAI-style
content-part messages, non-object JSONL rows, non-contiguous span inflation,
uncapped doc counts, degenerate configs, empty eval set, blocked embedding
similarity, and refusing to decontaminate a corpus onto itself).

## Limitations

- The hashed embedder is lexical: it finds paraphrases that reuse content
  words (and morphological variants via char n-grams) but not translations or
  restatements with fresh vocabulary. Use `--embedder` with a real model for
  that, and recalibrate `--cosine*` on a sample of known leaks/non-leaks.
- Cosine thresholds for the hashed embedder were calibrated on small examples;
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
