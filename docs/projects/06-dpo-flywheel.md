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
  "data_sha256": "2987204611155387"
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
