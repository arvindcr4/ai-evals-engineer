"""`evalkit regression-gate run|compare|baseline`."""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

from evalkit.core.llm import get_llm
from evalkit.regression_gate.gate import evaluate_gate, write_step_summary
from evalkit.regression_gate.runner import RunResult, run_suite
from evalkit.regression_gate.suite import GatePolicy, Suite


def _policy(suite_gate: GatePolicy, args: argparse.Namespace) -> GatePolicy:
    p = GatePolicy(**vars(suite_gate))
    if args.max_drop is not None:
        p.max_success_drop = args.max_drop
    if args.max_latency_increase is not None:
        p.max_p95_latency_increase = args.max_latency_increase
    if args.no_significance:
        p.require_significance = False
    return p


def _gate(baseline: RunResult, candidate: RunResult, policy: GatePolicy,
          args: argparse.Namespace) -> int:
    report = evaluate_gate(baseline, candidate, policy)
    md = report.to_markdown()
    if args.report:
        Path(args.report).write_text(md)
    write_step_summary(md)
    print(md)
    verdict = "PASS" if report.passed else "FAIL: " + "; ".join(report.reasons)
    print(f"regression-gate: {verdict}", file=sys.stderr)
    return report.exit_code


def _run(args: argparse.Namespace) -> RunResult:
    suite = Suite.load(args.suite)
    llm = get_llm(args.llm) if args.llm else None
    result = run_suite(suite, llm=llm, workers=getattr(args, "workers", None))
    if args.out:
        result.save(args.out)
    s = result.summary()
    cost = f", cost ${s['cost_usd']:.4f}" if s["tokens_in"] or s["cost_usd"] else ""
    print(f"{suite.name}: {s['n_cases']} cases, success {s['success_rate']:.1%}, "
          f"p95 {s['latency_p95_s'] * 1000:.1f} ms, errors {s['errors']}{cost}", file=sys.stderr)
    return result


def _cmd_run(args: argparse.Namespace) -> int:
    result = _run(args)
    if not args.baseline:
        return 0
    if not Path(args.baseline).exists():
        print(f"no baseline at {args.baseline}; run `regression-gate baseline` on main first",
              file=sys.stderr)
        return 0 if args.allow_missing_baseline else 1
    policy = _policy(Suite.load(args.suite).gate, args)
    return _gate(RunResult.load(args.baseline), result, policy, args)


def _cmd_compare(args: argparse.Namespace) -> int:
    gate = Suite.load(args.suite).gate if args.suite else GatePolicy()
    return _gate(RunResult.load(args.baseline), RunResult.load(args.candidate),
                 _policy(gate, args), args)


def _cmd_baseline(args: argparse.Namespace) -> int:
    if args.from_results:
        shutil.copyfile(args.from_results, args.out)
    elif not args.suite:
        print("baseline needs --suite or --from-results", file=sys.stderr)
        return 2
    else:
        _run(args)
    print(f"baseline written to {args.out}", file=sys.stderr)
    return 0


def register(subparsers: argparse._SubParsersAction) -> None:
    p = subparsers.add_parser("regression-gate", help="04 CI/CD eval regression gate")
    sub = p.add_subparsers(dest="cmd", required=True)

    def policy_flags(sp: argparse.ArgumentParser) -> None:
        sp.add_argument("--max-drop", type=float, help="max success drop, e.g. 0.02")
        sp.add_argument("--max-latency-increase", type=float, help="max p95 rise, e.g. 0.25")
        sp.add_argument("--no-significance", action="store_true",
                        help="block on any drop over threshold, significant or not")
        sp.add_argument("--report", help="write the Markdown report here")

    r = sub.add_parser("run", help="run a suite; gate against --baseline if given")
    r.add_argument("--suite", required=True)
    r.add_argument("--out", help="write results JSON")
    r.add_argument("--baseline", help="baseline results JSON to gate against")
    r.add_argument("--allow-missing-baseline", action="store_true")
    r.add_argument("--llm", help="LLM spec for llm targets (default: suite / $EVALKIT_LLM)")
    r.add_argument("--workers", type=int, help="concurrent cases (default: target setting, 1)")
    policy_flags(r)
    r.set_defaults(func=_cmd_run)

    c = sub.add_parser("compare", help="gate two existing results files")
    c.add_argument("--baseline", required=True)
    c.add_argument("--candidate", required=True)
    c.add_argument("--suite", help="suite YAML to read the gate policy from")
    policy_flags(c)
    c.set_defaults(func=_cmd_compare)

    b = sub.add_parser("baseline", help="run the suite (or promote results) as the new baseline")
    b.add_argument("--suite")
    b.add_argument("--out", required=True)
    b.add_argument("--from-results", help="promote an existing results file instead of running")
    b.add_argument("--llm")
    b.add_argument("--workers", type=int)
    b.set_defaults(func=_cmd_baseline)
