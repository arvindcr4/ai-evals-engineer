# 06 — Automated DPO Flywheel

> Pipeline capturing user thumbs-downs, auto-formatting them into preference pairs and
> triggering a nightly LoRA fine-tune. Production edge cases are your highest-leverage
> training data.

## What and why

Every thumbs-down in production marks a prompt where the model failed. That is the
data hardest to get any other way. `evalkit.dpo_flywheel` turns those signals into a
TRL-ready DPO dataset every night. It filters the data hard enough to train on safely,
and promotes the resulting LoRA adapter only if it beats the current one on an eval.

```
POST /feedback ─┐
ingest JSONL ───┴─► feedback.jsonl ─► PairBuilder ─► filters ─► datasets/<date>/train.jsonl
 (append-only,      (cursor = line)   correction       PII scrub     + manifest.json
  deduped by id)                      similar_up       length/refusal + train_config.yaml
                                      teacher+judge    margin, decon      │
                                                       dedupe/near-dup    ▼
                                    state.json ◄── promotion gate ◄── Trainer (dry | trl)
```

## Design

**Feedback ingestion** (`feedback.py`). A `FeedbackEvent` has
`{conversation_id, messages, response, rating, correction?, model, ts}`. Ratings in
many spellings (`up/down`, `thumbs_up`, `±1`, booleans) are normalized. Any unknown
field goes into `meta`. The `event_id` is a stable content hash, so replaying a batch
or retrying a POST does not create duplicates. `FeedbackStore` is an append-only JSONL
file guarded by a thread lock plus `flock`, which makes it safe with several uvicorn
workers. Because the file only grows, a line offset is a valid cursor.
`create_app(store)` serves `POST /feedback` (201, or 422 with the validation error),
`GET /stats` and `GET /healthz`.

**Pair construction** (`pairs.py`). Strategies are tried in order for each thumbs-down
(the order is configurable with `--strategies`):

1. `correction`: the user's own edit becomes `chosen` and the bad response becomes `rejected`.
2. `similar_up`: a thumbs-up answer to a near-identical prompt (word Jaccard ≥ 0.6)
   from another conversation becomes `chosen`.
3. `teacher`: a stronger model (`--teacher <spec>`, any `get_llm` spec) writes N
   candidates. A judge scores them and the best one becomes `chosen`.

Each pair records its judge margin, `score(chosen) − score(rejected)`. The default
judge is an offline `HeuristicJudge`, a 0–10 rubric that scores topical coverage,
sane length, no refusal and no repetition. `--judge <spec>` swaps in an `LLMJudge`,
which parses `SCORE: x` and falls back to the heuristic if the reply cannot be parsed.
In offline mode, `mock` teacher specs get a deterministic responder that writes
on-topic answers, so the whole pipeline produces real output.

**Quality filters** (`filters.py`). These run in this order, and each drop is counted
under a named reason:

| step | reason key |
|---|---|
| PII scrub: emails, Luhn-valid card numbers, 10–13-digit phone numbers (dotted-quad IPs left alone) → `[EMAIL]`/`[CARD]`/`[PHONE]` (a rewrite, not a drop) | `pii_redactions` |
| empty prompt, chosen or rejected | `empty` |
| chosen == rejected after normalization | `identical` |
| chosen word count outside [3, 800], or prompt too long | `length` |
| chosen is a refusal | `refusal` |
| judge margin < `--min-margin` (user corrections are trusted by default) | `low_margin` |
| prompt overlaps the golden eval set (exact normalized match; ≥50% word-5-gram overlap measured against the prompt *or* the golden item, so an eval item pasted into a long prompt is caught; 2–4-word golden items by phrase containment) | `contaminated` |
| normalized hash of prompt+chosen already seen | `duplicate` |
| 3-shingle Jaccard ≥ 0.85 with a kept pair | `near_duplicate` |

Scrubbing runs first, so dedupe and decontamination see the same text that will be
trained on.

**Output.** `train.jsonl` holds TRL `DPOTrainer` standard-format rows
`{prompt, chosen, rejected}`. Multi-turn context is rendered as a `User:/Assistant:`
transcript. `pairs.jsonl` holds the same rows plus provenance: strategy, source event,
margin and donor/teacher. `manifest.json` records event counts, raw and kept counts
per strategy, drop reasons, PII redactions, the filter config and a SHA-256 over the
training rows. The hash is reproducible: the same events and config give the same hash.

**Nightly orchestrator** (`nightly.py`). `run_nightly` works through these steps:

1. Rebuild the cumulative dataset from all events.
2. Count the kept pairs that came from events after the cursor. If there are fewer
   than `--min-new-pairs`, skip the run and leave the cursor where it is, so feedback
   keeps building toward the threshold.
3. Write the versioned `datasets/<YYYY-MM-DD>[-N]/`, including a `train_config.yaml`
   with LoRA r/alpha/dropout/target modules, DPO beta, lr, batch size and max length.
4. Call the pluggable `Trainer`:
   - `DryRunTrainer` (default) validates every row and writes `plan.json` with rows,
     approximate tokens, effective batch and optimizer steps.
   - `TRLTrainer` runs `DPOTrainer` with a PEFT `LoraConfig`. Its imports are lazy,
     and it raises a clear error if the `train` extra is missing.
5. Gate on the result. An `Evaluator` scores the incumbent (base, or the last promoted
   adapter) and the candidate. The candidate is promoted only if it wins by at least
   `--min-win`. If training fails, nothing is promoted and the cursor does not move.
   Shipped evaluators:
   - `MockEvaluator`: deterministic and seeded, with gain proportional to dataset size.
   - `WinRateEvaluator`: head-to-head between a base and a candidate endpoint on the
     golden prompts, scored by a judge.

## How to run

```bash
bash examples/06-dpo-flywheel/run.sh          # full offline demo, writes examples/06-dpo-flywheel/out/

uv run --no-sync evalkit dpo-flywheel ingest events.jsonl --store data/feedback.jsonl [--strict]
uv run --no-sync evalkit dpo-flywheel serve --store data/feedback.jsonl --port 8706
curl -XPOST localhost:8706/feedback -H 'content-type: application/json' -d \
  '{"conversation_id":"c1","messages":[{"role":"user","content":"How do I reset my password?"}],
    "response":"Contact support.","rating":"down","correction":"Settings → Security → Reset password."}'
uv run --no-sync evalkit dpo-flywheel build-pairs --store data/feedback.jsonl --golden golden.jsonl \
  --out data/pairs [--teacher openai:gpt-4.1] [--judge openai:gpt-4.1-mini] [--min-margin 1.0]
uv run --no-sync evalkit dpo-flywheel nightly --store data/feedback.jsonl --root data/flywheel \
  --golden golden.jsonl --min-new-pairs 200 [--dry-run | --backend trl] [--base-model ...]
```

`--teacher none` turns off regeneration (corrections and similar-up only). Real
training needs `uv sync --extra train` and a GPU.

### Nightly schedule

systemd user timer (`~/.config/systemd/user/dpo-flywheel.{service,timer}`):

```ini
# dpo-flywheel.service
[Unit]
Description=evalkit DPO flywheel nightly run

[Service]
Type=oneshot
WorkingDirectory=%h/developer/ai-evals-engineer
ExecStart=/usr/bin/env uv run --no-sync evalkit dpo-flywheel nightly \
  --store data/feedback.jsonl --root data/flywheel --golden data/golden.jsonl \
  --min-new-pairs 200 --backend trl --teacher openai:gpt-4.1

# dpo-flywheel.timer
[Unit]
Description=Run the DPO flywheel at 02:30 every night

[Timer]
OnCalendar=*-*-* 02:30:00
Persistent=true
RandomizedDelaySec=10m

[Install]
WantedBy=timers.target
```

Enable it with `systemctl --user enable --now dpo-flywheel.timer`. The cron equivalent is:

```cron
30 2 * * * cd ~/developer/ai-evals-engineer && uv run --no-sync evalkit dpo-flywheel nightly --store data/feedback.jsonl --root data/flywheel --golden data/golden.jsonl --min-new-pairs 200 --backend trl >> data/flywheel/nightly.log 2>&1
```

The command exits 0 for `skipped`, `promoted` and `rejected`, and 2 for
`train_failed`, so the timer's failure state reflects real breakage.

## Sample output (real run of `examples/06-dpo-flywheel/run.sh`)

The example has 35 day-1 events: 20 support topics, with a third of them carrying
user corrections, a third having a thumbs-up neighbour and a third left for the
teacher. It also includes a PII-laden refund request, a replayed event, a near-repeat
prompt, a golden-set leak, a no-op correction, a refusal "correction" and two
malformed rows.

```
== 1. ingest day-1 feedback (2 invalid rows, 1 duplicate expected)
  rejected line 34: messages must contain at least one user turn
  rejected line 35: unrecognised rating: 'meh'
{"read": 35, "valid": 33, "invalid": 2, "written": 32, "duplicates": 1, "store": "out/feedback.jsonl"}
== 2. build preference pairs (mock teacher, heuristic judge, golden decontamination)
{
  "train": "out/pairs/train.jsonl",
  "pairs_kept": 21,
  "pairs_raw": 25,
  "by_strategy_kept": {"correction": 8, "similar_up": 7, "teacher": 6},
  "drops": {"contaminated": 1, "identical": 1, "refusal": 1, "duplicate": 1},
  "pii_redactions": {"email": 1, "card": 1, "phone": 1},
  "data_sha256": "03c8f004e8512b8a"
}
{"prompt": "How do I reset my password?", "chosen": "Go to Settings, choose Security, click Reset password and follow the emailed link; it expires after 30 minutes.", "rejected": "I'm sorry, but I can't help with that request."}
== 3. nightly run #1 (dry-run backend, gate at 10 new pairs)
  "events_total": 32, "events_new": 32, "pairs_total": 21, "pairs_new": 21,
  "dataset": "out/flywheel/datasets/2026-10-03",
  "train": {"backend": "dry", "status": "planned", "metrics": {"rows": 21, "steps": 2, "problems": 0}},
  "gate": {"incumbent": "base", "incumbent_score": 0.7, "candidate_score": 0.7607, "min_win": 0.01, "promoted": true},
  "status": "promoted"
== 4. day 2: only a few new thumbs-downs arrive -> nightly should skip
  "events_total": 38, "events_new": 6, "pairs_total": 25, "pairs_new": 4,
  "status": "skipped", "reason": "only 4 new pairs (< 10)"
== 5. same day, lower threshold -> trains, gate decides promotion vs the incumbent
  "dataset": "out/flywheel/datasets/2026-10-04",
  "gate": {"incumbent": "out/flywheel/adapters/2026-10-03", "incumbent_score": 0.7607,
           "candidate_score": 0.7876, "promoted": true},
  "status": "promoted"
```

The nightly JSON above is trimmed. `adapters/2026-10-03/plan.json` reports 21 rows,
about 1.1k tokens, an effective batch of 16 and 2 optimizer steps, along with the
full LoRA/DPO config.

## Limitations

- Near-duplicate detection compares every pair with every other pair using exact
  shingle Jaccard. That is fine up to tens of thousands of pairs; beyond that, swap in
  MinHash/LSH.
- The phone regex is digit-count based, so a bare 10–13-digit number such as a Unix
  timestamp or an order id is also redacted as `[PHONE]`.
- If a trainer raises (OOM, missing deps), the nightly run records `train_failed`
  with the error and leaves the cursor in place, so the next night retries.
- The PII regexes cover emails, cards and phone numbers only. Names, addresses and
  free-text secrets need an NER-based scrubber (for example Presidio) before this runs
  on real traffic.
- `similar_up` matches by lexical overlap, not embeddings. Paraphrases with little
  word overlap are missed.
- The heuristic judge is a rubric proxy. For real data, use `--judge` with a strong
  model and calibrate it (see system 03).
- `MockEvaluator` only exercises the promotion logic. A real gate must serve the
  adapter (for example vLLM with LoRA) and use `WinRateEvaluator` or a
  task-specific eval.
- Thumbs-downs are noisy, and some are aimed at the product rather than the answer.
  The margin filter and trusted-correction flag reduce this noise but do not
  remove it.
- The TRL backend targets the current `DPOTrainer(processing_class=...)` API. Older
  TRL versions need `tokenizer=`.

## Real-model run (DeepSeek, Oct 2026)

The offline demo above uses a mock teacher and a heuristic judge. This section is the
same pipeline against the real DeepSeek API.

```bash
bash examples/06-dpo-flywheel/real_run.sh   # sources ~/TradingAgents/.env, writes out/real/
# the core step it runs:
uv run --no-sync evalkit dpo-flywheel build-pairs --store out/real/feedback.jsonl --golden golden.jsonl \
  --teacher deepseek:deepseek-v4-pro --judge deepseek:deepseek-flash --workers 6 \
  --cache out/real/llm_cache.jsonl --out out/real/pairs
```

- **Teacher:** `deepseek-v4-pro`, thinking off, T=0.8, 3 candidates per thumbs-down,
  `max_tokens` 600.
- **Judge:** `deepseek-flash`, thinking off, T=0, `SCORE: <0-10>` rubric.
- **Data:** the example feedback, which is small. Day 1 has 35 rows (33 valid, 32 stored,
  25 thumbs-downs). Day 2 adds 6 near-repeat thumbs-downs.
- **Cost:** the final committed run cost **$0.0169** (day-1 build $0.0108, day-2
  increment $0.0061, and $0 for the cache-served rebuild and nightly #1). Development
  runs (a raw-output probe, one unmetered first run, and five full runs while fixing
  prompts) bring the total API spend to about **$0.12**. Wall time for the day-1 build
  fell from 142 s (sequential) to about 12 s (6 workers).
- **Outputs:** `examples/06-dpo-flywheel/real_output/`, which holds the CLI JSON for every step,
  `manifest.json`, all 21 provenance pairs, the dry-run `plan.json` and the run log.

### Results

`build-pairs`, day 1 (from `real_output/build_pairs.json`):

| | mock teacher + heuristic judge | DeepSeek teacher + judge |
|---|---|---|
| pairs raw / kept | 25 / 21 | 25 / 21 |
| kept by strategy (correction / similar_up / teacher) | 8 / 7 / 6 | 8 / 7 / 6 |
| drops | contaminated 1, identical 1, refusal 1, duplicate 1 | same |
| judge margin, correction (min / median / max) | 2.5 / 4.75 / 8.5 | 4 / 6 / 7 |
| judge margin, similar_up | 3.5 / 6.25 / 6.7 | 4 / 7 / 7 |
| judge margin, teacher | 4.5 / 6.1 / 7.9 | 6 / 6.5 / 8 |
| judge parse failures | n/a | 0 / 60 calls |
| API calls (teacher / judge) | 18 mock / none (heuristic) | 18 / 60 (+8 in-run cache hits) |
| rebuild of the same data | new hash each time with a sampling teacher | same `data_sha256` `d781bf94…`, 0 calls |

Nightly (dry-run backend; the gate is still `MockEvaluator`):

| run | mock | DeepSeek |
|---|---|---|
| #1, day 1, min 10 new | 21 new pairs → promoted (0.700 → 0.761) | 21 new → promoted (0.700 → 0.761), $0 (cache) |
| #2, day 2, min 10 new | 4 new → skipped | **2 new** → skipped (near_duplicate 4), $0.0061 |
| #3, day 2, lower min | min 3: trains, promoted 0.761 → 0.788 | min 2: trains, promoted 0.761 → 0.788, $0 |

Teacher-chosen rows from the real run (`real_output/pairs.jsonl`):

```
6.0 [8,6,8] How do I rotate my webhook secret? | rejected: Please contact support.
  chosen: Go to Settings, choose Webhooks, click Rotate secret next to the webhook, and confirm.
          The old secret stops working immediately.
8.0 [9,9,9] Why is my API key not working? | rejected: That is not possible.
  chosen: I can help troubleshoot, but I'll need a bit more detail. ... `401 Unauthorized` or `403 Forbidden`? ...
```

### What the real model revealed (and what was fixed)

1. **The nightly job re-bought the whole history every night, and the dataset was not
   reproducible.** `run_nightly` rebuilds the cumulative dataset, so every thumbs-down
   ever seen went back to the T=0.8 teacher each night. Spend grew linearly with history,
   and `data_sha256` changed with no new data. The mock hid both problems because it is
   free and deterministic. Fix: `MeteredLLM` (`llm_cache.py`), a persistent completion
   cache (`--cache`, default `<root>/llm_cache.jsonl` for `nightly`). With it, day 2 paid
   only for its 6 new events, and a rebuild reproduces the hash with 0 calls.
2. **Nothing recorded cost.** The manifest, `build-pairs` output and nightly record now
   carry `llm_usage` (calls, cache hits, errors, tokens and `cost_usd` per role; a model
   shared by both roles is counted once), along with `margins_raw/kept` per strategy,
   `judge_parse_failures` and `unpaired`.
3. **One API error killed the whole run, and the run was slow.** Pair building was
   sequential (142 s) and any exception aborted the night. It now uses a `--workers`
   thread pool that keeps event order. A failure after core retries costs that one
   event (`teacher_error` in `unpaired`), not the whole run.
4. **Raw PII went to the API.** Scrubbing ran only after pairs were built, so the judge
   (and the teacher, for teacher pairs) received unscrubbed emails and card numbers. The
   real run made this visible: a phone number from one user's correction appeared in
   another prompt's teacher answer. LLM inputs are now scrubbed before sending
   (`scrub_llm_inputs`, on by default).
5. **The ungrounded teacher wrote generic answers.** The first real run produced chosen
   texts like "the exact process varies by service (Google, Apple, Microsoft…)". The judge
   scored them 9/10 against "Please contact support", so they passed every filter. That is
   bad DPO data for a product assistant. The teacher now receives the 3 most similar
   trusted answers (thumbs-ups and user corrections, scrubbed) as example conversations
   (`--teacher-refs`).
6. **The grounded teacher leaked its prompt context.** The first grounding prompt called
   the references "verified answers", and 13 of 18 teacher candidates echoed it ("I don't
   have a verified answer for that…", "…in the verified product documentation I've been
   given"). An instruction not to mention them did not help. Two changes fixed it:
   - The block was reframed as "Example conversations", with the instruction "write only
     the reply the user sees". That brought leaks to 0 of 18.
   - `leaks_context` now drops leaking candidates before judging, and the `context_leak`
     filter is a backstop for teacher pairs.
7. **Judge parsing was hardened, but the real judge never needed it.** deepseek-flash
   replied with exactly `SCORE: N` 60 out of 60 times. The parser still accepts
   `**SCORE:** 8`, `Score = 7/10` and reasoning-then-verdict (a bare `7/10` with no
   `score`/`rating` label is still a parse failure) (last score wins). A parse failure is
   now counted, because a silent fallback would mix the heuristic scale into a margin.
   The judge call is now capped with `max_tokens` 64.

Regression tests for each fix are in `tests/test_dpo_flywheel_real.py`, using MockLLM
responders that copy the real outputs.

### Honest comparison with the mock

- **The headline counts are identical (21 kept, 6 per strategy), but that is mostly
  structural.** 15 of the 21 pairs (corrections and thumbs-up donors) never touch the
  teacher. The judge only sets their margin, and no pair came near `--min-margin 1`, so
  `low_margin` dropped nothing in either run. On this data the judge does not act as a
  filter.
- **The real judge is coarse.** Scores are integers, and the 3 teacher candidates tied in
  5 of 6 events, and in the sixth (`[8, 6, 8]`) the first still tied for best, so the
  "best of N" pick was the first candidate in all 6. Teacher margins sit at 6–8; all
  kept margins are 4–8. The mock heuristic spreads scores more, but they mean less.
- **The real model's main risk is confident fabrication, and nothing here catches it.**
  With product-style examples, the teacher writes plausible menu paths for features the
  examples never covered ("Settings → Webhooks → Rotate secret… stops working
  immediately"), and the judge scores them 8. The earlier prompt hedged honestly but
  leaked its context. Neither prompt is free of trade-offs. Teacher pairs for features
  with no trusted answer should be reviewed by a person, or produced by a teacher with
  real docs or retrieval, before they train a production model. The mock teacher cannot
  show this failure.
- **Day 2 differed.** The real grounded teacher reproduces the user correction almost word
  for word for a near-repeat prompt ("Quick question: how do I add a teammate…"). The
  near-dup filter then removed 4 of the 6 day-2 pairs, against 2 with the mock (mock
  `near_duplicate 2`, 4 new pairs; `out/flywheel/state.json` of `run.sh`), so only
  2 new pairs arrived and the nightly threshold in `real_run.sh` step 5 was lowered to 2.
  This is the filter doing its job, and it shows that the near-repeat traffic adds little
  new signal.
- **The promotion gate is still `MockEvaluator`.** The gate numbers are identical to the
  mock by construction. Nothing was trained, because the dry-run backend was used. A real
  gate needs a served adapter and `WinRateEvaluator`.
- **The cache only makes old events free while their teacher prompt is unchanged.** The
  teacher prompt embeds the 3 most similar trusted answers from the *whole* cumulative
  store, so a new thumbs-up or correction that ranks into an old event's top 3 changes
  that prompt, misses the cache, re-buys the event and changes its chosen text (and the
  dataset hash). The example day 2 has only thumbs-downs, which never become references,
  so this run did not show it. A mock probe with one new thumbs-up and no new
  thumbs-down re-paid 2 old teacher events (6 calls). Freezing each event's reference set
  at first build, or drawing references only from earlier events, would fix it.
- **Not in `real_output/`:** the development-run figures (the 142 s → 12 s wall time,
  the 13/18 leak rate, the generic 9/10 answers, the cross-user phone number, and the
  ~$0.12 total spend) come from earlier runs whose outputs were not saved. Everything in
  the tables above is recomputable from `real_output/`.
- **The sample is tiny** (25 thumbs-downs, 6 teacher pairs, one run per prompt version).
  The leak rates (13/18 → 0/18) and the tie rates come from single runs and are
  directional only.
