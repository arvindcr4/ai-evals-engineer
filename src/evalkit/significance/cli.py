"""`evalkit significance compare|power|leaderboard|simulate`."""

from __future__ import annotations

import argparse
import json
import sys

from evalkit.core.trajectory import read_jsonl, write_jsonl
from evalkit.significance.compare import (
    compare,
    leaderboard,
    leaderboard_markdown,
    load_scores,
    simulate_results,
)
from evalkit.significance.stats import (
    adjust_pvalues,
    minimum_detectable_effect,
    power_two_proportions,
    required_sample_size,
)


def _cmd_compare(args: argparse.Namespace) -> int:
    table = load_scores(read_jsonl(args.results), args.metric)
    others = args.b or [m for m in sorted(table) if m != args.a]
    comps = [compare(table, args.a, b, args.n_boot, args.confidence, seed=args.seed,
                     clustered=not args.no_clusters, ci_method=args.ci) for b in others]
    if len(comps) > 1:
        for c, p in zip(comps, adjust_pvalues([c.p_value for c in comps], args.correction)):
            c.extra["p_raw"] = c.p_value
            c.p_value = p
    if args.json:
        print(json.dumps([c.to_dict() for c in comps], indent=2, default=float))
    else:
        for c in comps:
            print(c.verdict())
            print(f"  {c.model_a}: {c.mean_a * c.scale:.1f}  {c.model_b}: "
                  f"{c.mean_b * c.scale:.1f}  test={c.test}  ci={c.ci.method}"
                  + (f"  correction={args.correction}" if len(comps) > 1 else ""))
    return 0


def _cmd_power(args: argparse.Namespace) -> int:
    p1, d = args.baseline, args.delta
    kw = {"alpha": args.alpha, "paired": args.paired, "discordance": args.discordance}
    design = "paired (McNemar)" if args.paired else "unpaired, per model"
    out: dict = {"baseline": p1, "alpha": args.alpha, "power_target": args.power,
                 "design": design}
    if d is not None:
        n = required_sample_size(p1, d, power=args.power, **kw)
        out.update(delta=d, required_n=n)
        print(f"To detect {p1 * 100:.1f}% → {(p1 + d) * 100:.1f}% ({d * 100:+.1f} pts) with "
              f"{args.power:.0%} power at α={args.alpha}: n = {n} items ({design}).")
    if args.n is not None:
        mde = minimum_detectable_effect(args.n, p1, power=args.power, **kw)
        out.update(n=args.n, mde=mde)
        print(f"With n={args.n} ({design}), the minimum detectable effect is "
              f"{mde * 100:.1f} pts at {args.power:.0%} power.")
        if d is not None:
            pw = power_two_proportions(args.n, p1, p1 + d, **kw)
            out.update(power_at_n=pw)
            print(f"Power to detect {d * 100:+.1f} pts at n={args.n}: {pw:.1%}.")
    if args.json:
        print(json.dumps(out, indent=2))
    return 0


def _cmd_leaderboard(args: argparse.Namespace) -> int:
    table = load_scores(read_jsonl(args.results), args.metric)
    rows, pairs, adj = leaderboard(table, args.n_boot, args.confidence, args.correction,
                                   args.seed)
    if args.json:
        print(json.dumps({"rows": [r.__dict__ for r in rows],
                          "pairs": [dict(c.to_dict(), p_adjusted=p)
                                    for c, p in zip(pairs, adj)]}, indent=2, default=float))
        return 0
    scale = pairs[0].scale if pairs else 100.0
    print(leaderboard_markdown(rows, scale))
    print(f"\nTiers: models in the same tier are not significantly different from the tier "
          f"leader ({args.correction}-corrected over {len(pairs)} pairwise tests).")
    return 0


def _cmd_simulate(args: argparse.Namespace) -> int:
    accs = {}
    for spec in args.model:
        name, acc = spec.split("=", 1)
        accs[name] = float(acc)
    rows = simulate_results(accs, args.n, args.clusters, args.cluster_sd, args.seed)
    write_jsonl(args.out, rows)
    print(f"wrote {len(rows)} rows for {len(accs)} models to {args.out}", file=sys.stderr)
    return 0


def register(subparsers: argparse._SubParsersAction) -> None:
    p = subparsers.add_parser("significance", help="07 bootstrap CIs, p-values and power")
    sub = p.add_subparsers(dest="cmd", required=True)

    def common(sp: argparse.ArgumentParser) -> None:
        sp.add_argument("--results", required=True, help="JSONL {item_id, model, score[, cluster]}")
        sp.add_argument("--metric", default="score", help="score field (or metric name)")
        sp.add_argument("--n-boot", type=int, default=10_000)
        sp.add_argument("--confidence", type=float, default=0.95)
        sp.add_argument("--seed", type=int, default=0)
        sp.add_argument("--correction", choices=["none", "bonferroni", "holm", "bh"],
                        default="holm")
        sp.add_argument("--json", action="store_true")

    c = sub.add_parser("compare", help="paired comparison of model B (or several) vs A")
    common(c)
    c.add_argument("--a", required=True, help="baseline model")
    c.add_argument("--b", action="append", help="candidate model(s); default: all others")
    c.add_argument("--ci", choices=["bca", "percentile"], default="bca")
    c.add_argument("--no-clusters", action="store_true", help="ignore the cluster column")
    c.set_defaults(func=_cmd_compare)

    w = sub.add_parser("power", help="required sample size / minimum detectable effect")
    w.add_argument("--baseline", type=float, required=True, help="baseline accuracy in (0,1)")
    w.add_argument("--delta", type=float, help="target improvement, e.g. 0.02")
    w.add_argument("--n", type=int, help="available items (reports MDE)")
    w.add_argument("--alpha", type=float, default=0.05)
    w.add_argument("--power", type=float, default=0.8)
    w.add_argument("--paired", action="store_true", help="both models on the same items")
    w.add_argument("--discordance", type=float,
                   help="paired: fraction of items where models disagree")
    w.add_argument("--json", action="store_true")
    w.set_defaults(func=_cmd_power)

    lb = sub.add_parser("leaderboard", help="rank N models into significance tiers")
    common(lb)
    lb.set_defaults(func=_cmd_leaderboard, n_boot=5_000)

    s = sub.add_parser("simulate", help="write synthetic 0/1 results with known accuracies")
    s.add_argument("--model", action="append", required=True, help="name=accuracy")
    s.add_argument("--n", type=int, default=500)
    s.add_argument("--clusters", type=int)
    s.add_argument("--cluster-sd", type=float, default=0.0)
    s.add_argument("--seed", type=int, default=0)
    s.add_argument("--out", required=True)
    s.set_defaults(func=_cmd_simulate)
