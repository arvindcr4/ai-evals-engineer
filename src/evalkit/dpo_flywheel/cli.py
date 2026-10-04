"""`evalkit dpo-flywheel ingest|serve|build-pairs|nightly`."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import date
from pathlib import Path

from evalkit.dpo_flywheel.feedback import FeedbackStore, create_app, load_events
from evalkit.dpo_flywheel.filters import FilterConfig
from evalkit.dpo_flywheel.nightly import (
    TRAINERS,
    MockEvaluator,
    NightlyConfig,
    TrainConfig,
    run_nightly,
)
from evalkit.dpo_flywheel.pairs import (
    STRATEGIES,
    HeuristicJudge,
    LLMJudge,
    PairBuilder,
    build_dataset,
    resolve_llm,
    write_dataset,
)


def _golden(path: str | None) -> list[str]:
    if not path:
        return []
    out = []
    for line in Path(path).read_text().splitlines():
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            out.append(line.strip())
            continue
        if isinstance(row, dict):
            out.append(row.get("prompt") or row.get("question") or row.get("input") or "")
        else:
            out.append(str(row))
    return [g for g in out if g]


def _builder(args) -> PairBuilder:
    strategies = tuple(s.strip() for s in args.strategies.split(",") if s.strip())
    bad = set(strategies) - set(STRATEGIES)
    if bad:
        raise SystemExit(f"unknown strategies: {sorted(bad)}")
    teacher = None if args.teacher == "none" else resolve_llm(args.teacher or args.llm, teacher=True)
    judge = LLMJudge(resolve_llm(args.judge)) if args.judge else HeuristicJudge()
    return PairBuilder(strategies=strategies, teacher=teacher, judge=judge, n_candidates=args.n_candidates)


def _filters(args) -> FilterConfig:
    return FilterConfig(min_margin=args.min_margin, scrub_pii=not args.no_scrub)


def _cmd_ingest(args) -> int:
    events, bad = load_events(args.input)
    written = FeedbackStore(args.store).append(events)
    for line_no, err in bad:
        print(f"  rejected line {line_no}: {err}", file=sys.stderr)
    print(json.dumps({"read": len(events) + len(bad), "valid": len(events), "invalid": len(bad),
                      "written": written, "duplicates": len(events) - written, "store": args.store}))
    return 1 if bad and args.strict else 0


def _cmd_serve(args) -> int:
    import uvicorn

    uvicorn.run(create_app(FeedbackStore(args.store)), host=args.host, port=args.port)
    return 0


def _cmd_build(args) -> int:
    events = FeedbackStore(args.store).read()
    result = build_dataset(events, _builder(args), _filters(args), _golden(args.golden))
    path = write_dataset(result, args.out)
    m = result.manifest
    print(json.dumps({"train": str(path), "pairs_kept": m["pairs_kept"], "pairs_raw": m["pairs_raw"],
                      "by_strategy_kept": m["by_strategy_kept"], "drops": m["drops"],
                      "pii_redactions": m["pii_redactions"], "data_sha256": m["data_sha256"][:16]}, indent=2))
    return 0


def _cmd_nightly(args) -> int:
    backend = "dry" if args.dry_run else args.backend
    train = TrainConfig(base_model=args.base_model, lora_r=args.lora_r, lora_alpha=args.lora_alpha,
                        beta=args.beta, learning_rate=args.lr)
    cfg = NightlyConfig(
        root=Path(args.root),
        store=Path(args.store),
        min_new_pairs=args.min_new_pairs,
        min_win=args.min_win,
        run_date=date.fromisoformat(args.date) if args.date else None,
        golden=_golden(args.golden),
        train=train,
        filters=_filters(args),
    )
    record = run_nightly(cfg, _builder(args), TRAINERS[backend](), MockEvaluator(seed=args.seed))
    print(json.dumps(record, indent=2))
    return 0 if record["status"] in {"skipped", "promoted", "rejected"} else 2


def _pair_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--store", required=True, help="feedback JSONL store")
    p.add_argument("--golden", help="eval set (JSONL with prompt/question/input, or plain lines) for decontamination")
    p.add_argument("--strategies", default=",".join(STRATEGIES), help="ordered, comma-separated")
    p.add_argument("--llm", default=None, help="default model spec (used as teacher when --teacher unset)")
    p.add_argument("--teacher", default=None, help="teacher LLM spec, or 'none' to disable regeneration")
    p.add_argument("--judge", default=None, help="judge LLM spec (default: offline heuristic rubric)")
    p.add_argument("--n-candidates", type=int, default=3)
    p.add_argument("--min-margin", type=float, default=1.0, help="judge score gap chosen - rejected")
    p.add_argument("--no-scrub", action="store_true", help="disable PII scrubbing")


def register(subparsers) -> None:
    root = subparsers.add_parser("dpo-flywheel", help="06 thumbs-down → DPO pairs → nightly LoRA")
    sub = root.add_subparsers(dest="dpo_cmd", required=True)

    p = sub.add_parser("ingest", help="validate feedback events and append them to the store")
    p.add_argument("input", help="JSONL feedback events")
    p.add_argument("--store", required=True)
    p.add_argument("--strict", action="store_true", help="exit 1 if any line is invalid")
    p.set_defaults(func=_cmd_ingest)

    p = sub.add_parser("serve", help="run the POST /feedback endpoint")
    p.add_argument("--store", required=True)
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8706)
    p.set_defaults(func=_cmd_serve)

    p = sub.add_parser("build-pairs", help="construct + filter preference pairs")
    _pair_args(p)
    p.add_argument("--out", required=True, help="output dataset directory")
    p.set_defaults(func=_cmd_build)

    p = sub.add_parser("nightly", help="gate, version, train and promote")
    _pair_args(p)
    p.add_argument("--root", required=True, help="flywheel state directory")
    p.add_argument("--backend", choices=sorted(TRAINERS), default="dry")
    p.add_argument("--dry-run", action="store_true", help="force the dry backend")
    p.add_argument("--min-new-pairs", type=int, default=20)
    p.add_argument("--min-win", type=float, default=0.01, help="required eval gain to promote")
    p.add_argument("--date", help="run date YYYY-MM-DD (default today)")
    p.add_argument("--base-model", default=TrainConfig.base_model)
    p.add_argument("--lora-r", type=int, default=16)
    p.add_argument("--lora-alpha", type=int, default=32)
    p.add_argument("--beta", type=float, default=0.1)
    p.add_argument("--lr", type=float, default=5e-6)
    p.add_argument("--seed", type=int, default=0)
    p.set_defaults(func=_cmd_nightly)
