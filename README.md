# evalkit — fifteen AI-evals systems

Working implementations of the fifteen projects in
[Suraj Sharma's "As an AI Evals Engineer, you must build these projects"](https://x.com/suraj_sharma14/status/2106715961403273433):
systems that measure, secure and ship agents, not dashboards.

Everything runs **offline and deterministically** with a seeded mock model, so every test and
demo is reproducible on a laptop. Any OpenAI-compatible model (OpenAI, DeepSeek, vLLM, Ollama,
OpenRouter…) plugs in with `--llm <spec>`.

```bash
uv sync
uv run pytest -q                       # 295 tests
uv run evalkit -h                      # one CLI, fourteen systems
bash examples/07-significance-engine/run.sh
```

## The systems

| # | System | CLI | What it proves | Docs |
|---|---|---|---|---|
| 01 | Trajectory Grading Engine | `evalkit trajectory-grader` | Grades every tool call against a DAG spec; fails the run on a hallucinated parameter or a skipped safety check, and names the first failing step. | [docs](docs/projects/01-trajectory-grader.md) |
| 02 | Shadow Routing Comparator | `evalkit shadow-router` | An OpenAI-compatible proxy mirrors a sticky 5% of traffic to a candidate model off the user's latency path, then writes a cost/quality/latency report with a PROMOTE / HOLD / BLOCK verdict. | [docs](docs/projects/02-shadow-router.md) |
| 03 | Calibrated LLM-as-a-Judge | `evalkit calibrated-judge` | Measures position, verbosity and self-preference bias against 500 labelled anchors, fits a logistic correction and reports held-out before/after. | [docs](docs/projects/03-calibrated-judge.md) |
| 04 | CI/CD Regression Gate | `evalkit regression-gate` + [`eval-gate.yml`](.github/workflows/eval-gate.yml) | Parameterised evals on every PR. A merge is blocked only when success drops more than 2 pts **and** the drop is significant, or when p95 latency spikes. | [docs](docs/projects/04-regression-gate.md) |
| 05 | RAG Adversarial Harness | `evalkit rag-adversarial` | Eight perturbations (distractors, gold removal, contradictions, stale docs, citation shuffles, entity swaps, injected instructions) that test whether the model abstains, and a measured "confident lie" rate. | [docs](docs/projects/05-rag-adversarial.md) |
| 06 | Automated DPO Flywheel | `evalkit dpo-flywheel` | Thumbs-down feedback becomes filtered, PII-scrubbed, decontaminated DPO pairs for a nightly LoRA run (TRL backend or dry run), with a promote-only-if-better gate. | [docs](docs/projects/06-dpo-flywheel.md) |
| 07 | Statistical Significance Engine | `evalkit significance` | Paired, clustered and BCa bootstrap, permutation and McNemar tests, Holm and BH correction, and power/MDE, ending in a sentence such as *"B beats A by +4.4 pts ± 2.9 (95% CI [1.5, 7.3], p=0.007)"*. | [docs](docs/projects/07-significance-engine.md) |
| 08 | Agent Red-Team Fuzzer | `evalkit redteam-fuzzer` | A mutation fuzzer that injects prompt injections (direct, via tool output, via docs, via another agent), malformed tool schemas and loop inducers, with canary/forbidden-tool oracles. | [docs](docs/projects/08-redteam-fuzzer.md) |
| 09 | Production Drift Monitor | `evalkit drift-monitor` | Nightly 5% sampling, a SQLite metric store, z-score / CUSUM / PSI against a rolling baseline, and alerts to stdout, JSONL or a webhook. | [docs](docs/projects/09-drift-monitor.md) |
| 10 | Cost-Quality Pareto Dashboard | `evalkit pareto` | Per-tenant Pareto frontiers with Wilson CIs, a simulated cheap-then-escalate router (System-One → System-Two) and price-drop what-ifs, rendered as a self-contained HTML dashboard. | [docs](docs/projects/10-pareto-dashboard.md) |
| 11 | Counterfactual Replay Debugger | `evalkit replay-debugger` | Records every LLM hop and tool output, swaps one node, replays the rest, and bisects to the single step that caused the failure. | [docs](docs/projects/11-replay-debugger.md) |
| 12 | Synthetic Edge-Case Generator | `evalkit edge-case-gen` | Boundary, format, semantic-OOD and pairwise-combinatorial cases, from rules and an LLM, validated, deduped and coverage-reported. | [docs](docs/projects/12-edge-case-generator.md) |
| 13 | Context Window Eviction Tester | `evalkit context-eviction` | Plants facts and fact updates in noise, then scores FIFO, window, summary and retrieval memory on recall, stale answers and token budget. | [docs](docs/projects/13-context-eviction.md) |
| 14 | Dataset Contamination Checker | `evalkit contamination` | 13-gram overlap, MinHash-LSH and hashed-embedding similarity over streamed training corpora, with clean / suspicious / contaminated verdicts and a decontaminate step. | [docs](docs/projects/14-contamination-checker.md) |
| 15 | Public Eval Methodology Teardown | — | Three deep-dives on the grading rubrics, judge prompts and statistical framework used above. | [methodology](docs/methodology/) |

## Layout

```
src/evalkit/core/        LLM client (mock + OpenAI-compatible), Trajectory/Step model, JSONL I/O
src/evalkit/<system>/    one package per system, each with its own cli.py
tests/                   pytest suite for every system (incl. adversarial-review regression tests)
examples/NN-<slug>/      sample data + run.sh that drives the CLI end-to-end
docs/projects/           design notes, sample output and limitations per system
docs/methodology/        project 15: the three methodology deep-dives
```

## Using a real model

```bash
export OPENAI_API_KEY=...            # or DEEPSEEK_API_KEY / EVALKIT_API_KEY
uv run evalkit calibrated-judge audit --llm openai:gpt-4o-mini ...
uv run evalkit rag-adversarial run --llm deepseek:deepseek-chat ...
uv run evalkit shadow-router serve --llm "http://localhost:8000/v1|my-model" ...
```

`EVALKIT_LLM` sets the default spec. The DPO flywheel's TRL training backend needs the optional
extra: `uv sync --extra train`.

## Honesty note

Every number in the docs comes from the bundled **synthetic** datasets and the seeded mock model.
The numbers show that the machinery works and that the metrics catch the failures they were built
to catch. They are not benchmark claims about any real model. Each system's doc has a
*Limitations* section.

## License

MIT
