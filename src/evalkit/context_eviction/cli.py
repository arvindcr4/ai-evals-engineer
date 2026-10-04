"""`evalkit context-eviction run|inspect` commands."""

from __future__ import annotations

import argparse
import json

from evalkit.context_eviction.harness import (
    cells_to_dicts,
    default_reader,
    format_table,
    run_scenario,
    sweep,
)
from evalkit.context_eviction.memory import STRATEGIES, make_memory
from evalkit.context_eviction.scenario import Scenario, generate
from evalkit.core.llm import LLM, get_llm


def _models(spec: str | None) -> tuple[LLM, LLM | None]:
    """``mock`` → offline extractive reader + summarizer; otherwise one real model for both."""
    if spec is None or spec == "mock" or spec.startswith("mock:"):
        return default_reader(), None
    llm = get_llm(spec)
    return llm, llm


def _ints(s: str) -> list[int]:
    return [int(x) for x in s.split(",") if x.strip()]


def _run(a: argparse.Namespace) -> int:
    strategies = [s.strip() for s in a.strategies.split(",") if s.strip()]
    unknown = set(strategies) - set(STRATEGIES)
    if unknown:
        raise SystemExit(f"unknown strategies {sorted(unknown)}; choose from {sorted(STRATEGIES)}")
    reader, summarizer = _models(a.llm)
    cells = sweep(strategies, _ints(a.noise), budget=a.budget, trials=a.trials, seed=a.seed,
                  reader=reader, summarizer=summarizer, n_facts=a.facts, n_updates=a.updates,
                  pinned_updates=a.pinned_updates, hard_noise=a.hard_noise)
    if a.json:
        print(json.dumps({"budget": a.budget, "cells": cells_to_dicts(cells)}, indent=2))
    else:
        print(format_table(cells, a.budget))
    return 0


def _scenario(a: argparse.Namespace) -> Scenario:
    if getattr(a, "scenario", None):
        return Scenario.load(a.scenario)
    return generate(a.noise, n_facts=a.facts, n_updates=a.updates, seed=a.seed,
                    pinned_updates=a.pinned_updates, hard_noise=a.hard_noise)


def _export(a: argparse.Namespace) -> int:
    sc = _scenario(a)
    sc.save(a.out)
    kinds = [t.kind for t in sc.turns]
    print(f"wrote {a.out}: {len(sc.turns)} turns ({kinds.count('fact')} facts, "
          f"{kinds.count('update')} updates, {kinds.count('distractor')} distractors), "
          f"{len(sc.probes)} probes")
    return 0


def _inspect(a: argparse.Namespace) -> int:
    sc = _scenario(a)
    reader, summarizer = _models(a.llm)
    mem = make_memory(a.strategy, sc.system, a.budget, llm=summarizer)
    print(f"scenario seed={sc.seed} noise={sc.noise}: {len(sc.turns)} turns; planted:")
    for t in sc.turns:
        if t.kind in ("fact", "update"):
            print(f"  turn {t.index:>4} [{t.kind}] {t.text}")
    res, answers = run_scenario(sc, mem, reader)
    print(f"\n{a.strategy}: recall {res.recall:.2f}, stale {res.stale_rate:.2f}, "
          f"budget compliance {res.compliance:.0%}, peak {res.peak_tokens} tokens")
    for row in answers:
        print(f"  {row['slot']:<18} {row['verdict']:<8} answer={row['answer']!r} "
              f"expected={row['expected']!r}")
    if a.show_context:
        print("\ncontext for the last probe:")
        print("\n".join("  " + x for x in mem.context(sc.probes[-1].question)))
    return 0


def register(subparsers: argparse._SubParsersAction) -> None:
    p = subparsers.add_parser(
        "context-eviction", help="13 noise-flood memory strategies and score fact recall"
    )
    sub = p.add_subparsers(dest="cmd", required=True)

    def common(sp: argparse.ArgumentParser) -> None:
        sp.add_argument("--budget", type=int, default=400, help="context budget in tokens")
        sp.add_argument("--seed", type=int, default=0)
        sp.add_argument("--facts", type=int, default=6)
        sp.add_argument("--updates", type=int, default=3)
        sp.add_argument("--pinned-updates", type=int, default=1,
                        help="system-prompt facts later changed in conversation")
        sp.add_argument("--hard-noise", type=float, default=0.1,
                        help="fraction of user noise turns that are slot-vocabulary distractors")
        sp.add_argument("--llm", default="mock", help="reader/summarizer model spec")

    r = sub.add_parser("run", help="sweep strategies over noise levels")
    r.add_argument("--strategies", default="fifo,window,summary,retrieval")
    r.add_argument("--noise", default="0,50,200,800", help="comma-separated noise-turn counts")
    r.add_argument("--trials", type=int, default=3)
    r.add_argument("--json", action="store_true")
    common(r)
    r.set_defaults(func=_run)

    i = sub.add_parser("inspect", help="one scenario, one strategy, per-probe answers")
    i.add_argument("--strategy", default="retrieval")
    i.add_argument("--noise", type=int, default=200)
    i.add_argument("--scenario", help="load a scenario JSON (from `export`) instead of generating")
    i.add_argument("--show-context", action="store_true")
    common(i)
    i.set_defaults(func=_inspect)

    e = sub.add_parser("export", help="write a generated scenario to JSON for inspection/reuse")
    e.add_argument("--noise", type=int, default=200)
    e.add_argument("--out", required=True)
    common(e)
    e.set_defaults(func=_export)
