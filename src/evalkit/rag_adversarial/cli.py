"""`evalkit rag-adversarial perturb|run|report`."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from evalkit.core.llm import MockLLM, get_llm
from evalkit.core.trajectory import read_jsonl

from .data import load_cases, load_items, save
from .perturb import OPERATORS, PerturbConfig, perturb
from .rag import LLMRAG, BaselineRAG, RAGSystem, mock_rag_llm
from .scoring import run_cases, summarize, to_markdown


def _perturb(args: argparse.Namespace) -> int:
    items = load_items(args.dataset)
    ops = args.ops.split(",") if args.ops and args.ops != "all" else None
    cfg = PerturbConfig(seed=args.seed, n_distractors=args.distractors)
    cases = perturb(items, ops, cfg)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    save(args.out, cases)
    counts: dict[str, int] = {}
    for c in cases:
        counts[c.perturbation] = counts.get(c.perturbation, 0) + 1
    print(f"{len(items)} items -> {len(cases)} cases -> {args.out}")
    for op, n in counts.items():
        print(f"  {op:<17} {n}")
    return 0


def build_system(kind: str, llm_spec: str | None, top_k: int) -> tuple[RAGSystem, str]:
    if kind in ("naive", "grounded"):
        rag = BaselineRAG(mode=kind, top_k=top_k)  # type: ignore[arg-type]
        return rag, rag.name
    llm = get_llm(llm_spec)
    if isinstance(llm, MockLLM) and llm.responder is None:
        mode = "naive" if "naive" in llm.model else "grounded"
        llm = mock_rag_llm(mode)  # type: ignore[arg-type]
    adapter = LLMRAG(llm)
    return adapter, adapter.name


def _run(args: argparse.Namespace) -> int:
    cases = load_cases(args.cases)[: args.limit or None]
    system, name = build_system(args.system, args.llm, args.top_k)
    results = run_cases(cases, system, args.name or name)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    save(args.out, results)
    lies = sum(r.lie for r in results)
    correct = sum(r.correct for r in results)
    print(f"{args.name or name}: {len(results)} cases, {correct} correct, {lies} lies -> {args.out}")
    return 0


def _report(args: argparse.Namespace) -> int:
    rows = [r for path in args.results for r in read_jsonl(path)]
    if not rows:
        print("no scored results found in: " + ", ".join(args.results))
        return 1
    summary = summarize(rows)
    md = to_markdown(summary)
    if args.md:
        Path(args.md).parent.mkdir(parents=True, exist_ok=True)
        Path(args.md).write_text(md)
    if args.json:
        Path(args.json).parent.mkdir(parents=True, exist_ok=True)
        Path(args.json).write_text(json.dumps(summary, indent=2))
    print(md)
    if args.max_lie_rate is not None:
        worst = max(s["overall"]["lie_rate"] or 0.0 for s in summary.values())
        if worst > args.max_lie_rate:
            print(f"FAIL: lie rate {worst:.1%} exceeds --max-lie-rate {args.max_lie_rate:.1%}")
            return 1
    return 0


def register(subparsers: argparse._SubParsersAction) -> None:
    p = subparsers.add_parser(
        "rag-adversarial", help="05 RAG adversarial harness: perturb, run, report",
        description="Stress a RAG system with adversarial context; measure confident lies.",
    )
    sub = p.add_subparsers(dest="cmd", required=True)

    pp = sub.add_parser("perturb", help="expand a clean QA set into adversarial cases")
    pp.add_argument("--dataset", required=True, help="QA items JSONL")
    pp.add_argument("--out", required=True, help="cases JSONL to write")
    pp.add_argument("--ops", default="all", help=f"comma list from {','.join(OPERATORS)}")
    pp.add_argument("--distractors", type=int, default=3, help="distractor docs to inject")
    pp.add_argument("--seed", type=int, default=0)
    pp.set_defaults(func=_perturb)

    pr = sub.add_parser("run", help="run a RAG system over cases and score each one")
    pr.add_argument("--cases", required=True)
    pr.add_argument("--out", required=True, help="scored results JSONL")
    pr.add_argument("--system", choices=["naive", "grounded", "llm"], default="grounded",
                    help="offline baseline modes or an LLM via --llm")
    pr.add_argument("--llm", default=None,
                    help="LLM spec for --system llm (mock, mock:naive, openai:gpt-4o-mini, ...)")
    pr.add_argument("--name", default=None, help="label for this system in reports")
    pr.add_argument("--top-k", type=int, default=6)
    pr.add_argument("--limit", type=int, default=0)
    pr.set_defaults(func=_run)

    rp = sub.add_parser("report", help="aggregate results into Markdown + JSON tables")
    rp.add_argument("results", nargs="+", help="one or more results JSONL files")
    rp.add_argument("--md", default=None)
    rp.add_argument("--json", default=None)
    rp.add_argument("--max-lie-rate", type=float, default=None,
                    help="exit 1 if any system's lie rate exceeds this (CI gate)")
    rp.set_defaults(func=_report)
