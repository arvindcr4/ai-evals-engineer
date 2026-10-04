"""`evalkit context-eviction run|inspect` commands."""

from __future__ import annotations

import argparse
import json

from evalkit.context_eviction.harness import (
    CostMeter,
    cells_to_dicts,
    default_reader,
    format_table,
    run_scenario,
    sweep,
)
from evalkit.context_eviction.memory import STRATEGIES, extractive_summarizer, make_memory
from evalkit.context_eviction.scenario import Scenario, generate
from evalkit.core.llm import MockLLM, get_llm


def _is_mock(spec: str | None) -> bool:
    return spec is None or spec == "mock" or spec.startswith("mock:")


def _models(a: argparse.Namespace) -> tuple[CostMeter, CostMeter]:
    """Reader and summarizer, each wrapped in a cost meter.

    ``mock`` → the offline extractive reader / summarizer (the idealized
    components); otherwise a real model. ``--llm`` sets both roles and
    ``--reader`` / ``--summarizer`` override one, so reader and summarizer
    errors can be separated.
    """
    r_spec = a.reader or a.llm
    s_spec = a.summarizer or a.llm
    reader = default_reader() if _is_mock(r_spec) else get_llm(r_spec)
    summarizer = (MockLLM(model="mock-summarizer", responder=extractive_summarizer)
                  if _is_mock(s_spec) else get_llm(s_spec))
    return CostMeter(reader), CostMeter(summarizer)


def _cost_line(meters: dict[str, CostMeter]) -> str:
    parts = [f"{role} {m.model}: {m.calls} calls, {m.tokens_in}+{m.tokens_out} tok, "
             f"${m.cost_usd:.4f}" for role, m in meters.items()]
    total = sum(m.cost_usd for m in meters.values())
    return "llm usage: " + "; ".join(parts) + f"; total ${total:.4f}"


def _ints(s: str) -> list[int]:
    return [int(x) for x in s.split(",") if x.strip()]


def _run(a: argparse.Namespace) -> int:
    strategies = [s.strip() for s in a.strategies.split(",") if s.strip()]
    unknown = set(strategies) - set(STRATEGIES)
    if unknown:
        raise SystemExit(f"unknown strategies {sorted(unknown)}; choose from {sorted(STRATEGIES)}")
    reader, summarizer = _models(a)
    cells = sweep(strategies, _ints(a.noise), budget=a.budget, trials=a.trials, seed=a.seed,
                  reader=reader, summarizer=summarizer, n_facts=a.facts, n_updates=a.updates,
                  pinned_updates=a.pinned_updates, hard_noise=a.hard_noise, workers=a.workers)
    meters = {"reader": reader, "summarizer": summarizer}
    if a.json:
        print(json.dumps({"budget": a.budget, "cells": cells_to_dicts(cells),
                          "usage": {k: m.to_dict() for k, m in meters.items()}}, indent=2))
    else:
        print(format_table(cells, a.budget))
        print(_cost_line(meters))
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
    reader, summarizer = _models(a)
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
    print(_cost_line({"reader": reader, "summarizer": summarizer}))
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
        sp.add_argument("--reader", help="reader model spec (overrides --llm)")
        sp.add_argument("--summarizer", help="summarizer model spec (overrides --llm)")

    r = sub.add_parser("run", help="sweep strategies over noise levels")
    r.add_argument("--strategies", default="fifo,window,summary,retrieval")
    r.add_argument("--noise", default="0,50,200,800", help="comma-separated noise-turn counts")
    r.add_argument("--trials", type=int, default=3)
    r.add_argument("--json", action="store_true")
    r.add_argument("--workers", type=int, default=1,
                   help="parallel (strategy, scenario) jobs; use ~8 with a real API")
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
