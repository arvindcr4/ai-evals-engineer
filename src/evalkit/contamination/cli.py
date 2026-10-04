"""`evalkit contamination index|scan|decontaminate`."""

from __future__ import annotations

import argparse
import json
from itertools import chain
from pathlib import Path

from evalkit.contamination.judge import adjudicate, judge_llm
from evalkit.contamination.report import to_markdown
from evalkit.contamination.scanner import (
    METHODS,
    ContaminationScanner,
    EvalIndex,
    ScanConfig,
    Thresholds,
    decontaminate,
)
from evalkit.contamination.text import DEFAULT_TRAIN_FIELDS, iter_records


def _methods(args: argparse.Namespace) -> tuple[str, ...]:
    return tuple(m.strip() for m in (args.methods or ",".join(METHODS)).split(",") if m.strip())


def _config(args: argparse.Namespace) -> ScanConfig:
    return ScanConfig(
        n=args.n, min_n=min(args.min_n, args.n), shingle=args.shingle, num_perm=args.num_perm,
        bands=args.bands, rows=args.rows, window=args.window, stride=args.stride,
        embedder=args.embedder, methods=_methods(args),
    )


def _thresholds(args: argparse.Namespace) -> Thresholds:
    return Thresholds(
        overlap_contaminated=args.overlap,
        containment_contaminated=args.containment,
        containment_suspicious=args.containment_suspicious,
        cosine_contaminated=args.cosine,
        cosine_suspicious=args.cosine_suspicious,
    )


def _index(args: argparse.Namespace) -> EvalIndex:
    if getattr(args, "index", None):
        overrides: dict = {"methods": _methods(args)} if args.methods else {}
        if args.window:
            overrides.update(window=args.window, stride=args.stride)
        if args.embedder_set:
            overrides["embedder"] = args.embedder
        return EvalIndex.load(args.index, **overrides)
    if not args.eval:
        raise SystemExit("error: pass --eval <jsonl> or --index <index.json>")
    return EvalIndex.from_jsonl(args.eval, args.eval_field or None, args.id_field, _config(args))


def _records(args: argparse.Namespace):
    fields = args.text_field or None
    return chain.from_iterable(
        iter_records(p, fields, args.train_id_field, DEFAULT_TRAIN_FIELDS) for p in args.train
    )


def _cmd_index(args: argparse.Namespace) -> int:
    idx = _index(args)
    idx.save(args.out)
    print(json.dumps({"out": args.out, **idx.to_dict()["stats"]}))
    return 0


def _cmd_scan(args: argparse.Namespace) -> int:
    idx = _index(args)
    scanner = ContaminationScanner(idx, _thresholds(args))
    report = scanner.scan(_records(args))
    if args.llm:
        adjudicate(report, idx, judge_llm(args.llm))
        ju = report.judge_usage
        print(
            f"judge {ju['model']}: {ju['calls']} call(s), {ju['yes']} YES / {ju['no']} NO / "
            f"{ju['unparsed']} unparsed / {ju['errors']} error(s), ${ju['cost_usd']:.4f}"
        )
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "report.json").write_text(json.dumps(report.to_dict(), indent=2), encoding="utf-8")
    (out / "report.md").write_text(to_markdown(report), encoding="utf-8")
    if args.clean_eval:
        keep = set(report.clean_ids())
        rows = (
            (r.id, r.raw) for r in iter_records(args.eval, args.eval_field, args.id_field)
        ) if args.eval else ((i, {"id": i, "text": t}) for i, t in zip(idx.ids, idx.texts))
        with open(args.clean_eval, "w", encoding="utf-8") as fh:
            for rid, raw in rows:
                if rid in keep and raw is not None:
                    fh.write(json.dumps(raw, ensure_ascii=False) + "\n")
    s = report.summary
    print(
        f"{s['items']} eval items: {s['contaminated']} contaminated, "
        f"{s['suspicious']} suspicious, {s['clean']} clean "
        f"(by method: {s['flagged_by_method']}) -> {out}/report.md"
    )
    for r in report.items:
        if r.status != "clean":
            print(f"  {r.status:<12} {r.id:<14} via {','.join(r.methods)}")
    if args.fail_on:
        bad = s["contaminated"] + (s["suspicious"] if args.fail_on == "suspicious" else 0)
        return 1 if bad else 0
    return 0


def _cmd_decontaminate(args: argparse.Namespace) -> int:
    out = Path(args.out).resolve()
    if any(Path(p).resolve() == out for p in args.train):
        raise SystemExit("error: --out must differ from every --train file (it would be truncated)")
    idx = _index(args)
    scanner = ContaminationScanner(idx, _thresholds(args))
    stats = decontaminate(scanner, _records(args), args.out, args.mode, args.level)
    print(json.dumps(stats))
    return 0


def _cmd_leaktest(args: argparse.Namespace) -> int:
    from evalkit.contamination import leakgen
    from evalkit.core.llm import get_llm

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    usage = leakgen.Usage()
    gen = get_llm(args.llm)
    if not args.eval:
        raise SystemExit("error: leaktest needs --eval <jsonl>")
    rows = [
        r.raw if isinstance(r.raw, dict) else {"id": r.id, "text": r.text}
        for r in iter_records(args.eval, args.eval_field, args.id_field)
    ]
    eval_path = out / "eval.jsonl"
    if args.synth:
        if eval_path.exists() and args.reuse:
            rows = [json.loads(ln) for ln in eval_path.read_text().splitlines() if ln.strip()]
        else:
            rows += leakgen.synth_items(gen, args.synth, usage)
    eval_path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))
    args.eval, args.index = str(eval_path), None
    idx = _index(args)
    items = list(zip(idx.ids, idx.texts))
    leaks_path = Path(args.leaks) if args.leaks else out / "leaks.jsonl"
    if leaks_path.exists() and (args.leaks or args.reuse):
        leaks = [json.loads(ln) for ln in leaks_path.read_text().splitlines() if ln.strip()]
    else:
        leaks = leakgen.generate_leaks(items, gen, usage)
        leaks_path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in leaks))
    jl = judge_llm(args.judge) if args.judge else None
    result = {
        "items": len(items), "generator": gen.model,
        "detectors": leakgen.detector_recall(idx, leaks, _thresholds(args), jl, usage),
    }
    if jl is not None:
        result["judge_model"] = jl.model
        result["judge"] = leakgen.judge_pairs(idx, leaks, jl, usage)
    result["usage"] = usage.to_dict()
    (out / "recall.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    md = leakgen.to_markdown(result)
    (out / "recall.md").write_text(md, encoding="utf-8")
    print(md)
    return 0


def _common(p: argparse.ArgumentParser, need_train: bool) -> None:
    src = p.add_argument_group("eval set")
    src.add_argument("--eval", help="eval set JSONL")
    src.add_argument("--index", help="index JSON written by `contamination index`")
    src.add_argument("--eval-field", action="append", help="eval text field (repeatable)")
    src.add_argument("--id-field", default="id")
    if need_train:
        tr = p.add_argument_group("training corpus")
        tr.add_argument("--train", nargs="+", required=True, help="JSONL / text corpora")
        tr.add_argument("--text-field", action="append",
                        help="training text field (repeatable; default: text/prompt/"
                             "completion/... plus chat `messages`)")
        tr.add_argument("--train-id-field", default="id")
    cfg = p.add_argument_group("detectors")
    cfg.add_argument("--n", type=int, default=13, help="word n-gram size")
    cfg.add_argument("--min-n", type=int, default=8,
                     help="items shorter than this skip the n-gram check")
    cfg.add_argument("--shingle", type=int, default=3)
    cfg.add_argument("--num-perm", type=int, default=128)
    cfg.add_argument("--bands", type=int, default=64)
    cfg.add_argument("--rows", type=int, default=2)
    cfg.add_argument("--window", type=int, default=0,
                     help="training window in words (0 = auto from eval lengths)")
    cfg.add_argument("--stride", type=int, default=0, help="window stride (0 = window/4)")
    cfg.add_argument("--embedder", default=None,
                     help="hashed | hashed:<dim> | openai:<model> | <base_url>|<model>")
    cfg.add_argument("--methods", default=None,
                     help=f"comma list of {','.join(METHODS)} (default: all)")
    th = p.add_argument_group("thresholds")
    d = Thresholds()
    th.add_argument("--overlap", type=float, default=d.overlap_contaminated)
    th.add_argument("--containment", type=float, default=d.containment_contaminated)
    th.add_argument("--containment-suspicious", type=float, default=d.containment_suspicious)
    th.add_argument("--cosine", type=float, default=None,
                    help="cosine for contaminated (default: embedder-calibrated)")
    th.add_argument("--cosine-suspicious", type=float, default=None)


def _run(cmd):
    def run(args: argparse.Namespace) -> int:
        args.embedder_set = args.embedder is not None
        args.embedder = args.embedder or "hashed"
        return cmd(args)

    return run


def register(subparsers: argparse._SubParsersAction) -> None:
    p = subparsers.add_parser(
        "contamination", help="14 dataset contamination checker (n-gram / MinHash / embedding)",
        description=__doc__,
    )
    sub = p.add_subparsers(dest="contamination_cmd", required=True)

    pi = sub.add_parser("index", help="build and save an eval-set index")
    _common(pi, need_train=False)
    pi.add_argument("--out", required=True)
    pi.set_defaults(func=_run(_cmd_index))

    ps = sub.add_parser("scan", help="scan training corpora for eval leakage")
    _common(ps, need_train=True)
    ps.add_argument("--out", required=True, help="output dir for report.json / report.md")
    ps.add_argument("--llm", default=None,
                    help="adjudicate suspicious items with an LLM (mock, openai:<model>, ...)")
    ps.add_argument("--clean-eval", help="also write the eval rows judged clean to this JSONL")
    ps.add_argument("--fail-on", choices=["suspicious", "contaminated"],
                    help="exit 1 if any item reaches this status (CI gate)")
    ps.set_defaults(func=_run(_cmd_scan))

    pl = sub.add_parser(
        "leaktest", help="model-written leaks per kind -> per-detector recall (+ judge)"
    )
    _common(pl, need_train=False)
    pl.add_argument("--out", required=True, help="output dir (eval.jsonl, leaks.jsonl, recall.*)")
    pl.add_argument("--llm", default=None, help="leak generator model (mock, deepseek:<model>, ...)")
    pl.add_argument("--judge", default=None, help="also judge every (item, leak) pair directly")
    pl.add_argument("--synth", type=int, default=0, help="add N model-written eval items")
    pl.add_argument("--leaks", default=None, help="reuse an existing leaks.jsonl")
    pl.add_argument("--reuse", action="store_true",
                    help="reuse eval.jsonl/leaks.jsonl already in --out (no regeneration)")
    pl.set_defaults(func=_run(_cmd_leaktest))

    pd = sub.add_parser("decontaminate", help="write a filtered training corpus")
    _common(pd, need_train=True)
    pd.add_argument("--out", required=True, help="output JSONL")
    pd.add_argument("--mode", choices=["drop", "flag"], default="drop")
    pd.add_argument("--level", choices=["suspicious", "contaminated"], default="contaminated")
    pd.set_defaults(func=_run(_cmd_decontaminate))
