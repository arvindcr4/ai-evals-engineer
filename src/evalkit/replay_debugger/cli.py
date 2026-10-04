"""`evalkit replay-debugger record|show|replay|bisect` commands.

``--llm`` takes ``toy`` / ``toy:sloppy`` (scripted offline policies) or any
:func:`evalkit.core.llm.get_llm` spec; ``--tools`` takes ``toy`` or ``toy:stale_fx``.
"""

from __future__ import annotations

import argparse
import json
import sys

from evalkit.core.trajectory import read_jsonl
from evalkit.replay_debugger.agent import numeric_checker, resolve_llm, resolve_tools
from evalkit.replay_debugger.cassette import Cassette, load_cassettes, save_cassettes
from evalkit.replay_debugger.replay import (
    Override,
    Replayer,
    bisect,
    format_bisect,
    format_nodes,
    record,
)


def _passed(c: Cassette) -> bool | None:
    return None if c.expected is None else numeric_checker(c.expected)(c.final_answer)


def _status(ok: bool | None) -> str:
    return {True: "PASS", False: "FAIL", None: "n/a"}[ok]


def _record(a: argparse.Namespace) -> int:
    llm, tools = resolve_llm(a.llm), resolve_tools(a.tools)
    out = []
    for row in read_jsonl(a.tasks):
        c = record(row["input"], llm, tools, task_id=row["task_id"], max_hops=a.max_hops,
                   meta={"expected": row.get("expected"), "llm": a.llm, "tools": a.tools})
        out.append(c)
        print(f"{c.task_id:<18} {len(c.nodes):>2} nodes  answer={c.final_answer!s:<10} "
              f"expected={c.expected!s:<10} {_status(_passed(c))}")
    save_cassettes(a.out, out)
    print(f"wrote {len(out)} cassettes -> {a.out}")
    return 0


def _pick(a: argparse.Namespace) -> list[Cassette]:
    cs = load_cassettes(a.cassettes)
    if getattr(a, "task_id", None):
        cs = [c for c in cs if c.task_id == a.task_id]
        if not cs:
            sys.exit(f"no cassette with task_id {a.task_id!r}")
    return cs


def _show(a: argparse.Namespace) -> int:
    for c in _pick(a):
        print(f"[{c.task_id}] {c.task}  ({_status(_passed(c))})")
        print(format_nodes(c))
    return 0


def _replay(a: argparse.Namespace) -> int:
    c = _pick(a)[0]
    if a.output is not None:
        ov = Override(a.at, output=a.output)
    elif a.output_json is not None:
        ov = Override(a.at, output=json.loads(a.output_json))
    elif a.with_llm:
        ov = Override(a.at, llm=resolve_llm(a.with_llm))
    elif a.with_tools:
        ov = Override(a.at, tools=resolve_tools(a.with_tools))
    else:
        sys.exit("give one of --output, --output-json, --with-llm, --with-tools")
    res = Replayer(resolve_llm(a.llm), resolve_tools(a.tools), after=a.after,
                   max_hops=a.max_hops).replay(c, ov)
    print(f"[{c.task_id}] original ({_status(_passed(c))}):")
    print(format_nodes(c, mark=a.at))
    print(f"\nreplay with node {a.at} swapped, after={a.after} "
          f"({_status(_passed(res.replayed))}, {res.live_calls} live calls):")
    print(format_nodes(res.replayed, mark=a.at))
    if a.out:
        save_cassettes(a.out, [res.replayed])
    return 0


def _bisect(a: argparse.Namespace) -> int:
    llm, tools = resolve_llm(a.llm), resolve_tools(a.tools)
    ref = resolve_llm(a.ref_llm) if a.ref_llm else None
    oracle = resolve_tools(a.oracle_tools) if a.oracle_tools else None
    found = 0
    for c in _pick(a):
        if c.expected is None:
            print(f"{c.task_id}: no expected answer recorded — skipped")
            continue
        r = bisect(c, numeric_checker(c.expected), llm=llm, tools=tools, ref_llm=ref,
                   oracle_tools=oracle, after=a.after, max_hops=a.max_hops)
        if r.baseline_passed and not a.verbose:
            continue
        print(format_bisect(r))
        found += r.root_cause is not None
    print(f"\nroot causes isolated: {found}")
    return 0


def register(subparsers: argparse._SubParsersAction) -> None:
    p = subparsers.add_parser(
        "replay-debugger", help="11 record agent runs, swap one node, replay, bisect the culprit"
    )
    sub = p.add_subparsers(dest="cmd", required=True)

    def system_args(sp: argparse.ArgumentParser) -> None:
        sp.add_argument("--llm", default="toy", help="agent model (toy, toy:sloppy, or LLM spec)")
        sp.add_argument("--tools", default="toy", help="toolset (toy, toy:stale_fx)")
        sp.add_argument("--max-hops", type=int, default=8)

    r = sub.add_parser("record", help="run tasks live and save cassettes")
    r.add_argument("--tasks", required=True, help="JSONL with task_id, input, expected")
    r.add_argument("--out", required=True)
    system_args(r)
    r.set_defaults(func=_record)

    s = sub.add_parser("show", help="print the recorded node graph")
    s.add_argument("--cassettes", required=True)
    s.add_argument("--task-id")
    s.set_defaults(func=_show)

    rp = sub.add_parser("replay", help="swap node --at and replay the rest")
    rp.add_argument("--cassettes", required=True)
    rp.add_argument("--task-id")
    rp.add_argument("--at", type=int, required=True)
    g = rp.add_mutually_exclusive_group()
    g.add_argument("--output", help="literal replacement output (LLM text or tool result)")
    g.add_argument("--output-json", help="replacement output parsed as JSON")
    g.add_argument("--with-llm", help="regenerate the node with this model")
    g.add_argument("--with-tools", help="re-execute the node with this toolset")
    rp.add_argument("--after", choices=["cached", "live"], default="cached")
    rp.add_argument("--out", help="write the replayed cassette here")
    system_args(rp)
    rp.set_defaults(func=_replay)

    b = sub.add_parser("bisect", help="find the single node whose fix flips FAIL to PASS")
    b.add_argument("--cassettes", required=True)
    b.add_argument("--task-id")
    b.add_argument("--ref-llm", help="known-good model for LLM nodes")
    b.add_argument("--oracle-tools", help="known-good toolset for tool nodes")
    b.add_argument("--after", choices=["cached", "live"], default="cached")
    b.add_argument("--verbose", action="store_true", help="also list passing runs")
    system_args(b)
    b.set_defaults(func=_bisect)
