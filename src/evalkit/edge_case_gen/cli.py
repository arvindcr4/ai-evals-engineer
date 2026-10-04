"""`evalkit edge-case-gen generate|validate|coverage|probe`."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from evalkit.core.llm import get_llm
from evalkit.core.trajectory import read_jsonl, write_jsonl
from evalkit.edge_case_gen.cases import AXES, EdgeCase
from evalkit.edge_case_gen.coverage import coverage_report, format_report
from evalkit.edge_case_gen.generators import RuleGenerator
from evalkit.edge_case_gen.llm_gen import LLMGenerator
from evalkit.edge_case_gen.probe import format_summary, probe, summarize
from evalkit.edge_case_gen.spec import load_spec
from evalkit.edge_case_gen.validate import Validator, load_callable, propose_labels


def _cases(path: str) -> list[EdgeCase]:
    return [EdgeCase.from_dict(d) for d in read_jsonl(path)]


def _spec_dir(spec_path: str) -> Path:
    return Path(spec_path).resolve().parent


def _write_json(path: str | None, obj: dict) -> None:
    if path:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(json.dumps(obj, indent=2, ensure_ascii=False, default=str))


def cmd_generate(a: argparse.Namespace) -> int:
    spec = load_spec(a.spec)
    axes = [x.strip() for x in a.axes.split(",") if x.strip()] if a.axes else list(AXES)
    bad = [x for x in axes if x not in AXES]
    if bad:
        raise SystemExit(f"unknown axes: {bad}; choose from {AXES}")
    cases = [] if a.no_rules else RuleGenerator(spec, seed=a.seed).generate(axes)
    errors: list[str] = []
    if a.llm:
        gen = LLMGenerator(spec, get_llm(a.llm), n_per_axis=a.n_llm,
                           temperature=a.temperature)
        cases += gen.generate(axes)
        errors = gen.errors
        u = gen.usage
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    write_jsonl(a.out, cases)
    per_gen: dict[str, int] = {}
    for c in cases:
        g = c.provenance["generator"]
        per_gen[g] = per_gen.get(g, 0) + 1
    print(f"generated {len(cases)} candidate cases -> {a.out}")
    print("  " + ", ".join(f"{k}={v}" for k, v in sorted(per_gen.items())))
    if a.llm:
        print(f"  llm usage: {u['calls']} calls, {u['tokens_in']} in / {u['tokens_out']} out "
              f"tokens, ${u['cost_usd']:.4f}; phantom cases {gen.stats['phantom']}, "
              f"repaired {gen.stats['repaired']} (resent/duplicate replies dropped: "
              f"{gen.stats['resent']})")
    for e in errors:
        print(f"  llm warning: {e}")
    return 0


def cmd_validate(a: argparse.Namespace) -> int:
    spec = load_spec(a.spec)
    rep = Validator(spec, near_dup=a.near_dup, min_novelty=a.min_novelty,
                    check_claims=not a.no_claim_check).run(_cases(a.inp))
    oracle = load_callable(a.oracle, _spec_dir(a.spec)) if a.oracle else None
    llm = get_llm(a.label_llm) if a.label_llm else None
    usage = propose_labels(spec, rep.kept, oracle=oracle, llm=llm,
                           min_confidence=a.min_confidence)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    write_jsonl(a.out, rep.kept)
    if a.rejects:
        write_jsonl(a.rejects, rep.rejected)
    s = rep.summary()
    print(f"validated: kept {s['kept']}, rejected {s['rejected']} {s['rejected_by_reason']}")
    print(f"  schema-invalid kept on purpose: {s['schema_invalid_kept']}; "
          f"needs_human_review: {s['needs_human_review']} -> {a.out}")
    if llm is not None:
        dis = sum(any(r.startswith("oracle_llm_disagree") for r in c.review_reasons)
                  for c in rep.kept)
        print(f"  label llm: {usage['calls']} calls, {usage['tokens_in']} in / "
              f"{usage['tokens_out']} out tokens, ${usage['cost_usd']:.4f}; "
              f"oracle/llm disagreements: {dis}")
    return 0


def cmd_coverage(a: argparse.Namespace) -> int:
    spec = load_spec(a.spec)
    rep = coverage_report(spec, _cases(a.inp))
    print(format_report(rep))
    _write_json(a.json, rep)
    if a.min_pairwise is not None and rep["pairwise_coverage"] < a.min_pairwise:
        print(f"FAIL: pairwise coverage {rep['pairwise_coverage']:.1%} < {a.min_pairwise:.1%}")
        return 1
    return 0


def cmd_probe(a: argparse.Namespace) -> int:
    system = load_callable(a.system, _spec_dir(a.spec))
    summary = summarize(probe(system, _cases(a.inp)))
    print(format_summary(summary, show=a.show))
    _write_json(a.out, summary)
    rate = summary["overall"]["failure_rate"]
    if a.max_fail_rate is not None and rate > a.max_fail_rate:
        print(f"FAIL: failure rate {rate:.1%} > {a.max_fail_rate:.1%}")
        return 1
    return 0


def register(subparsers: argparse._SubParsersAction) -> None:
    p = subparsers.add_parser("edge-case-gen", help="12 synthetic edge-case generator for golden sets")
    sub = p.add_subparsers(dest="cmd", required=True)

    g = sub.add_parser("generate", help="generate candidate edge cases from a task spec")
    g.add_argument("--spec", required=True)
    g.add_argument("--out", required=True)
    g.add_argument("--axes", help=f"comma list from {','.join(AXES)} (default: all)")
    g.add_argument("--seed", type=int, default=0)
    g.add_argument("--llm", help="also generate with a model, e.g. mock or openai:gpt-4o-mini")
    g.add_argument("--n-llm", type=int, default=4, help="LLM cases per axis")
    g.add_argument("--temperature", type=float, default=0.9,
                   help="sampling temperature for --llm (0 for reproducible real runs)")
    g.add_argument("--no-rules", action="store_true", help="skip rule-based generators")
    g.set_defaults(func=cmd_generate)

    v = sub.add_parser("validate", help="schema-check, dedupe, novelty-filter and label cases")
    v.add_argument("--spec", required=True)
    v.add_argument("--in", dest="inp", required=True)
    v.add_argument("--out", required=True)
    v.add_argument("--rejects", help="write rejected cases + reasons here")
    v.add_argument("--oracle", help="reference labeler <file.py|module>:<fn>")
    v.add_argument("--label-llm", help="LLM spec for label proposals / cross-checks")
    v.add_argument("--near-dup", type=float, default=0.95)
    v.add_argument("--min-novelty", type=float, default=0.0)
    v.add_argument("--min-confidence", type=float, default=0.7)
    v.add_argument("--no-claim-check", action="store_true",
                   help="keep cases whose input lacks the edge their category claims")
    v.set_defaults(func=cmd_validate)

    c = sub.add_parser("coverage", help="coverage report per axis/category + pairwise %%")
    c.add_argument("--spec", required=True)
    c.add_argument("--in", dest="inp", required=True)
    c.add_argument("--json", help="also write the report as JSON")
    c.add_argument("--min-pairwise", type=float, help="exit 1 below this pairwise coverage")
    c.set_defaults(func=cmd_coverage)

    pr = sub.add_parser("probe", help="run a system on the golden set, report failing edge cases")
    pr.add_argument("--spec", required=True)
    pr.add_argument("--in", dest="inp", required=True)
    pr.add_argument("--system", required=True, help="<file.py|module>:<fn> taking the input dict")
    pr.add_argument("--out", help="write the JSON summary here")
    pr.add_argument("--show", type=int, default=8)
    pr.add_argument("--max-fail-rate", type=float, help="exit 1 above this failure rate")
    pr.set_defaults(func=cmd_probe)
