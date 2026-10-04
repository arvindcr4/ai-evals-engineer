"""`evalkit shadow-router serve|simulate|report`."""

from __future__ import annotations

import argparse

from evalkit.core.trajectory import read_jsonl
from evalkit.shadow_router.demo import resolve_llm
from evalkit.shadow_router.proxy import ShadowConfig, ShadowRouter, create_app
from evalkit.shadow_router.report import build_report, default_judge, render_markdown, write_report


def _common(p: argparse.ArgumentParser) -> None:
    p.add_argument("--primary", default="demo:primary", help="primary LLM spec (serves users)")
    p.add_argument("--candidate", default="demo:candidate", help="candidate LLM spec (shadow)")
    p.add_argument("--rate", type=float, default=0.05, help="shadow sample rate (default 0.05)")
    p.add_argument("--salt", default="shadow-v1", help="sampling salt; change to reshuffle cohort")
    p.add_argument("--log", default="shadow_pairs.jsonl", help="pair log JSONL path")
    p.add_argument("--timeout", type=float, default=30.0, help="candidate timeout seconds")


def _config(a: argparse.Namespace) -> ShadowConfig:
    return ShadowConfig(rate=a.rate, salt=a.salt, log_path=a.log, candidate_timeout_s=a.timeout)


def _serve(a: argparse.Namespace) -> int:
    import uvicorn

    router = ShadowRouter(resolve_llm(a.primary), resolve_llm(a.candidate), _config(a))
    uvicorn.run(create_app(router), host=a.host, port=a.port)
    return 0


def _simulate(a: argparse.Namespace) -> int:
    from evalkit.shadow_router.simulate import simulate

    res = simulate(a.requests, resolve_llm(a.primary), resolve_llm(a.candidate), _config(a))
    print(
        f"replayed {res.requests} requests ({res.ok} ok, {res.failed} failed) in {res.wall_s:.2f}s; "
        f"shadowed {res.sampled} ({res.sampled / max(res.requests, 1):.1%}), "
        f"{res.shadow_errors} shadow errors -> {res.pairs_path}"
    )
    return 0 if res.failed == 0 else 1


def _report(a: argparse.Namespace) -> int:
    pairs = read_jsonl(a.pairs)
    judge = None
    if a.judge:
        judge = default_judge() if a.judge == "mock" else resolve_llm(a.judge)
    report = build_report(pairs, judge=judge, max_error_rate=a.max_error_rate)
    paths = write_report(report, a.out)
    print(render_markdown(report))
    print("wrote " + ", ".join(str(p) for p in paths.values()))
    return 2 if a.fail_on_block and report.verdict == "BLOCK" else 0


def register(subparsers) -> None:
    p = subparsers.add_parser("shadow-router", help="02 shadow routing comparator")
    sub = p.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("serve", help="run the OpenAI-compatible shadow proxy")
    _common(s)
    s.add_argument("--host", default="127.0.0.1")
    s.add_argument("--port", type=int, default=8787)
    s.set_defaults(func=_serve)

    m = sub.add_parser("simulate", help="replay a JSONL of prod requests through the proxy")
    _common(m)
    m.add_argument("requests", help="JSONL of requests ({messages} or {prompt})")
    m.set_defaults(func=_simulate)

    r = sub.add_parser("report", help="diff logged pairs into a cost/quality report")
    r.add_argument("pairs", help="pair log JSONL written by serve/simulate")
    r.add_argument("--out", default="shadow_report", help="output directory")
    r.add_argument("--judge", default="mock", help="judge LLM spec; 'mock' = offline, '' = off")
    r.add_argument("--max-error-rate", type=float, default=0.02)
    r.add_argument("--fail-on-block", action="store_true", help="exit 2 when verdict is BLOCK")
    r.set_defaults(func=_report)
