"""`evalkit redteam-fuzzer run|report`."""

from __future__ import annotations

import argparse
from pathlib import Path

from evalkit.core.llm import MockLLM, get_llm
from evalkit.core.trajectory import read_jsonl, write_jsonl
from evalkit.redteam_fuzzer.agent import toy_responder
from evalkit.redteam_fuzzer.attacks import AttackCase
from evalkit.redteam_fuzzer.fuzzer import run_campaign
from evalkit.redteam_fuzzer.report import (
    campaign_from_findings,
    render_scorecard,
    write_scorecard,
)

_LEVELS = ("none", "basic", "hardened")


def _non_negative(v: str) -> int:
    n = int(v)
    if n < 0:
        raise argparse.ArgumentTypeError("must be >= 0")
    return n


def _run(a: argparse.Namespace) -> int:
    levels = list(_LEVELS) if a.guardrail == "all" else [a.guardrail]
    campaigns = []
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    all_rows: list[dict] = []
    llm = None if a.llm == "mock" else get_llm(a.llm)
    seeds = [AttackCase.from_dict(r) for r in read_jsonl(a.seeds)] if a.seeds else None
    for level in levels:
        camp = run_campaign(level, iterations=a.iterations, seed=a.seed,
                            step_budget=a.step_budget, token_budget=a.token_budget,
                            llm=llm, seeds=seeds)
        campaigns.append(camp)
        rows = [f.to_dict() for f in camp.findings]
        # Summary marker row: carries the run count so `report` can rescore.
        rows.append({"guardrail": level, "_runs": camp.runs, "_failed_runs": camp.failed_runs})
        all_rows.extend(rows)
        write_jsonl(out / f"findings-{level}.jsonl", rows)
        print(f"[{level}] runs={camp.runs} findings={len(camp.findings)} "
              f"attack_success={camp.attack_success_rate:.3f} "
              f"resilience={camp.resilience_score:.3f} "
              f"severities={dict(sorted(camp.severity_counts.items()))}")
    write_jsonl(out / "findings.jsonl", all_rows)
    path = write_scorecard(campaigns, out)
    print(f"wrote {path}")
    return 0


def _report(a: argparse.Namespace) -> int:
    rows = read_jsonl(a.findings)
    campaigns = campaign_from_findings(rows)
    order = {lvl: i for i, lvl in enumerate(_LEVELS)}
    campaigns.sort(key=lambda c: order.get(c.guardrail, 99))
    md = render_scorecard(campaigns)
    if a.out:
        path = write_scorecard(campaigns, a.out)
        print(f"wrote {path}")
    else:
        print(md)
    return 0


def _selftest(a: argparse.Namespace) -> int:
    """Print the raw policy action for a single seeded step (debug aid)."""
    llm = MockLLM(model="toy", responder=toy_responder)
    msg = [{"role": "system", "content": "SECRET=x"},
           {"role": "user", "content": f"TASK: {a.task}"}]
    print(llm.complete(msg).text)
    return 0


def register(subparsers) -> None:
    p = subparsers.add_parser("redteam-fuzzer", help="08 agent red-team fuzzer")
    sub = p.add_subparsers(dest="cmd", required=True)

    r = sub.add_parser("run", help="fuzz a guardrail level (or all) and score resilience")
    r.add_argument("--guardrail", choices=[*_LEVELS, "all"], default="all")
    r.add_argument("--iterations", type=_non_negative, default=60, help="mutation iterations per level")
    r.add_argument("--seed", type=int, default=0)
    r.add_argument("--step-budget", type=int, default=12)
    r.add_argument("--token-budget", type=int, default=20_000)
    r.add_argument("--llm", default="mock",
                   help="policy LLM spec driving the toy agent (default: deterministic toy policy)")
    r.add_argument("--seeds", default="", help="seed-corpus JSONL of AttackCases (default: built-in)")
    r.add_argument("--out", default="redteam_out", help="output directory")
    r.set_defaults(func=_run)

    rp = sub.add_parser("report", help="render a scorecard from a findings JSONL")
    rp.add_argument("findings", help="findings.jsonl written by run")
    rp.add_argument("--out", default="", help="output dir (prints to stdout if omitted)")
    rp.set_defaults(func=_report)

    st = sub.add_parser("selftest", help="show the toy policy's action for one task")
    st.add_argument("task")
    st.set_defaults(func=_selftest)


__all__ = ["register"]
