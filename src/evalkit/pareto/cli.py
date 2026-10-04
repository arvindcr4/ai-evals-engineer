"""`evalkit pareto generate|build|whatif`."""

from __future__ import annotations

import argparse
import math

from evalkit.core.trajectory import read_jsonl
from evalkit.pareto.analysis import apply_price_factor, frontier_diff, summarize
from evalkit.pareto.dashboard import build_views, write_outputs
from evalkit.pareto.generate import generate


def _router(spec: str | None) -> tuple[str, str] | None:
    if not spec:
        return None
    parts = [p.strip() for p in spec.split(",")]
    if len(parts) != 2:
        raise SystemExit("--router expects CHEAP,EXPENSIVE")
    return parts[0], parts[1]


def _generate(a: argparse.Namespace) -> int:
    n = generate(a.out, a.tasks, a.seed)
    print(f"wrote {n} eval rows to {a.out}")
    return 0


def _build(a: argparse.Namespace) -> int:
    rows = read_jsonl(a.input)
    views = build_views(rows, _router(a.router), cascade=not a.no_cascade)
    tenants = len(views)
    configs = len({r["config"] for r in rows})
    paths = write_outputs(
        views,
        a.out,
        "Cost-quality Pareto frontier",
        f"{len(rows)} eval results · {tenants} tenants · {configs} configs",
    )
    for v in views:
        front = [s.config for s in v.stats if s.on_frontier]
        print(f"{v.tenant:18} frontier: {' < '.join(front)}")
    print("wrote " + ", ".join(str(p) for p in paths.values()))
    return 0


def _whatif(a: argparse.Namespace) -> int:
    rows = read_jsonl(a.input)
    if not any(_match(r["config"], a.config) for r in rows):
        raise SystemExit(f"no config matches {a.config!r}")
    changed = apply_price_factor(rows, a.config, a.factor)
    views = build_views(changed, _router(a.router), cascade=not a.no_cascade, ghost_rows=rows)
    diff = frontier_diff(summarize(rows), summarize(changed))
    label = f"{a.config} price × {a.factor:g}"
    notes = [
        f"What-if: **{label}**. Dotted line in the dashboard = frontier before the change.",
        "",
    ]
    for tenant, d in diff.items():
        moved = ", ".join(
            f"{c} {_fmt(o)} → {_fmt(n)}" for c, (o, n) in d["cost_per_success"].items()
        )
        notes.append(
            f"- **{tenant}**: joined frontier {d['joined'] or '—'}; left {d['left'] or '—'}; "
            f"cost/success {moved}"
        )
    paths = write_outputs(
        views,
        a.out,
        f"What-if: {label}",
        "frontier after the price change",
        notes,
        {"whatif": {"config": a.config, "factor": a.factor, "diff": diff}},
        ghost_label="frontier before",
        stem="whatif",
    )
    print("\n".join(notes))
    print("wrote " + ", ".join(str(p) for p in paths.values()))
    return 0


def _match(name: str, pattern: str) -> bool:
    import fnmatch

    return fnmatch.fnmatchcase(name, pattern)


def _fmt(x: float) -> str:
    return "∞" if math.isinf(x) else f"${x:.4f}"


def register(subparsers) -> None:
    p = subparsers.add_parser("pareto", help="10 cost-quality Pareto dashboard")
    sub = p.add_subparsers(dest="cmd", required=True)

    g = sub.add_parser("generate", help="synthetic eval results (tenants × configs × tasks)")
    g.add_argument("--out", required=True, help="output JSONL")
    g.add_argument("--tasks", type=int, default=120, help="tasks per tenant")
    g.add_argument("--seed", type=int, default=11)
    g.set_defaults(func=_generate)

    for name, fn, helptext in (
        ("build", _build, "dashboard + summary from eval results"),
        ("whatif", _whatif, "re-price a config and show how the frontier moves"),
    ):
        s = sub.add_parser(name, help=helptext)
        s.add_argument("input", help="JSONL rows {tenant, config, task_id, success, cost_usd, ...}")
        s.add_argument("--out", default="pareto_out", help="output directory")
        s.add_argument(
            "--router",
            default=None,
            help="CHEAP,EXPENSIVE configs for the escalation router (default: auto)",
        )
        s.add_argument(
            "--no-cascade",
            action="store_true",
            help="router pre-routes (escalated tasks don't pay for the cheap call)",
        )
        if name == "whatif":
            s.add_argument("--config", required=True, help="config name or glob, e.g. 's2-*'")
            s.add_argument("--factor", type=float, required=True, help="price multiplier, e.g. 0.5")
        s.set_defaults(func=fn)
