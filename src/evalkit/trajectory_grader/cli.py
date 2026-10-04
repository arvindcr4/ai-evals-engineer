"""`evalkit trajectory-grader ...` commands."""

from __future__ import annotations

import argparse
import json

from evalkit.core.trajectory import Trajectory, read_jsonl
from evalkit.trajectory_grader.grader import TrajectoryGrader, format_report, summarize
from evalkit.trajectory_grader.spec import load_spec


def _grade(args: argparse.Namespace) -> int:
    spec = load_spec(args.spec)
    grader = TrajectoryGrader(spec)
    reports = [grader.grade(Trajectory.from_dict(d)) for d in read_jsonl(args.trajectories)]
    if args.json:
        s = summarize(reports)
        print(json.dumps({
            "spec": spec.name,
            "summary": {"n": s.n, "passed": s.passed, "mean_score": s.mean_score,
                        "verdicts": s.verdict_counts},
            "runs": [r.to_dict() for r in reports],
        }, indent=2))
    else:
        print(format_report(reports, verbose=not args.quiet))
    return 1 if args.strict and not all(r.passed for r in reports) else 0


def _check_spec(args: argparse.Namespace) -> int:
    spec = load_spec(args.spec)
    print(f"spec {spec.name!r}: {len(spec.tools)} tools, DAG order "
          f"{' -> '.join(spec.topological_order())}")
    print(f"mandatory: {spec.mandatory}  safety: {spec.safety_tools}  forbidden: {spec.forbidden}")
    return 0


def register(subparsers: argparse._SubParsersAction) -> None:
    p = subparsers.add_parser(
        "trajectory-grader", help="01 step-level grading of agent tool calls against a DAG spec"
    )
    sub = p.add_subparsers(dest="cmd", required=True)

    g = sub.add_parser("grade", help="grade trajectories (JSONL) against a spec (YAML)")
    g.add_argument("--spec", required=True)
    g.add_argument("--trajectories", required=True)
    g.add_argument("--json", action="store_true", help="emit machine-readable JSON")
    g.add_argument("--quiet", action="store_true", help="table only, no per-step issues")
    g.add_argument("--strict", action="store_true", help="exit 1 if any run fails (for CI)")
    g.set_defaults(func=_grade)

    c = sub.add_parser("check-spec", help="validate a spec and print its DAG order")
    c.add_argument("--spec", required=True)
    c.set_defaults(func=_check_spec)
