"""`evalkit calibrated-judge make-anchors|audit|calibrate`."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from evalkit.calibrated_judge.anchors import load_anchors, make_anchors, save_anchors
from evalkit.calibrated_judge.audit import (
    Judgment,
    audit,
    audit_markdown,
    collect_judgments,
    load_judgments,
    save_judgments,
)
from evalkit.calibrated_judge.calibrate import calibrate, calibration_markdown
from evalkit.calibrated_judge.judge import (
    PairwiseJudge,
    PointwiseJudge,
    SimulatedJudge,
    Usage,
    sim_judge_from_args,
)
from evalkit.core.llm import LLM, get_llm


def _judge_llm(args: argparse.Namespace) -> LLM:
    if args.llm in (None, "sim"):
        return sim_judge_from_args(args).as_llm()
    return get_llm(args.llm)


def _family(args: argparse.Namespace) -> str | None:
    """``--judge-family none`` → no self-preference term (judge is in no anchor family)."""
    fam = (args.judge_family or "").strip().lower()
    return None if fam in ("", "none", "null") else fam


def _judgments(args: argparse.Namespace, anchors: list, pointwise: bool = False
               ) -> tuple[list[Judgment], Usage | None]:
    if args.judgments and Path(args.judgments).exists() and not args.rejudge:
        print(f"reusing judgments from {args.judgments}", file=sys.stderr)
        return load_judgments(args.judgments), None
    llm = _judge_llm(args)
    pair = PairwiseJudge(llm)
    point = PointwiseJudge(llm) if pointwise else None
    step = max(1, len(anchors) // 10)

    def progress(done: int, total: int) -> None:
        if done % step == 0 or done == total:
            spent = pair.usage.cost_usd + (point.usage.cost_usd if point else 0.0)
            print(f"  judged {done}/{total} anchors (${spent:.4f})", file=sys.stderr)

    rows = collect_judgments(anchors, pair, point, workers=args.workers,
                             progress=progress if args.workers > 1 else None)
    usage = pair.usage.merge(point.usage) if point else pair.usage
    if args.judgments:
        save_judgments(args.judgments, rows)
    print(f"judged {len(rows)}/{len(anchors)} anchors × 2 orders with {llm.model}: "
          f"{usage.calls} calls, {usage.tokens_in}+{usage.tokens_out} tokens, "
          f"${usage.cost_usd:.4f}, {usage.errors} failed anchors", file=sys.stderr)
    if not rows:
        raise SystemExit("every judge call failed; check the model spec / API key")
    return rows, usage


def _write(md: str, payload: dict, args: argparse.Namespace) -> None:
    if args.out_md:
        Path(args.out_md).write_text(md)
    if args.out_json:
        Path(args.out_json).write_text(json.dumps(payload, indent=2, default=float) + "\n")
    print(md)


def _cmd_make(args: argparse.Namespace) -> int:
    anchors = make_anchors(args.n, args.seed)
    save_anchors(args.out, anchors)
    rates = {k: sum(a.human_pref == k for a in anchors) for k in ("a", "b", "tie")}
    print(f"wrote {len(anchors)} anchors to {args.out} (human prefs {rates})", file=sys.stderr)
    return 0


def _cmd_audit(args: argparse.Namespace) -> int:
    anchors = load_anchors(args.anchors)
    rows, usage = _judgments(args, anchors, pointwise=args.pointwise)
    rep = audit(anchors, rows, _family(args))
    if usage is not None:
        rep["usage"] = usage.to_dict()
    _write(audit_markdown(rep), rep, args)
    return 0


def _cmd_calibrate(args: argparse.Namespace) -> int:
    anchors = load_anchors(args.anchors)
    rows, usage = _judgments(args, anchors)
    model, rep = calibrate(anchors, rows, _family(args), args.train_frac, args.seed)
    if usage is not None:
        rep["usage"] = usage.to_dict()
    if args.out:
        model.save(args.out)
        print(f"calibration model written to {args.out}", file=sys.stderr)
    _write(calibration_markdown(rep), rep, args)
    return 0


def register(subparsers: argparse._SubParsersAction) -> None:
    p = subparsers.add_parser("calibrated-judge", help="03 measure and correct LLM-judge bias")
    sub = p.add_subparsers(dest="cmd", required=True)

    m = sub.add_parser("make-anchors", help="synthesise a seeded human-labelled anchor set")
    m.add_argument("--n", type=int, default=500)
    m.add_argument("--seed", type=int, default=0)
    m.add_argument("--out", required=True)
    m.set_defaults(func=_cmd_make)

    defaults = SimulatedJudge()

    def judge_flags(sp: argparse.ArgumentParser) -> None:
        sp.add_argument("--anchors", required=True)
        sp.add_argument("--llm", default="sim",
                        help="judge model spec (default: built-in simulated biased judge)")
        sp.add_argument("--judge-family", default="nova",
                        help="model family of the judge, for self-preference; 'none' when "
                        "the judge belongs to no anchor family (e.g. a real API judge)")
        sp.add_argument("--workers", type=int, default=1,
                        help="concurrent anchors when calling an API judge")
        sp.add_argument("--judgments", help="cache file: reused if present, else written")
        sp.add_argument("--rejudge", action="store_true", help="ignore an existing cache")
        sp.add_argument("--seed", type=int, default=0)
        sp.add_argument("--out-md")
        sp.add_argument("--out-json")
        for name in ("position_bias", "verbosity_bias", "self_bias", "noise",
                     "overconfidence"):
            sp.add_argument(f"--sim-{name.replace('_', '-')}", type=float,
                            default=getattr(defaults, name), dest=f"sim_{name}",
                            help=f"simulated judge {name} (default {getattr(defaults, name)})")

    a = sub.add_parser("audit", help="measure agreement + position/verbosity/self biases")
    judge_flags(a)
    a.add_argument("--pointwise", action="store_true", help="also collect 1-10 scores")
    a.set_defaults(func=_cmd_audit)

    c = sub.add_parser("calibrate", help="fit bias corrections, report held-out improvement")
    judge_flags(c)
    c.add_argument("--train-frac", type=float, default=0.6)
    c.add_argument("--out", help="write the fitted correction (JSON)")
    c.set_defaults(func=_cmd_calibrate)
