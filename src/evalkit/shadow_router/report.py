"""Turn logged shadow pairs into a cost/quality comparison report.

Agreement is measured three ways (normalised exact match, token Jaccard and a
character-level similarity ratio). An optional pairwise LLM judge runs every
non-identical pair in both A/B orders; a pair counts as a win only when the
judge agrees with itself across orders, which cancels position bias. Output is
a JSON-able :class:`ShadowReport` rendered to Markdown and self-contained HTML.
"""

from __future__ import annotations

import html
import math
import re
from dataclasses import asdict, dataclass, field
from difflib import SequenceMatcher
from pathlib import Path

import numpy as np

from evalkit.core.llm import LLM, Message, MockLLM

JACCARD_AGREE = 0.6
_WORD = re.compile(r"[a-z0-9]+")


def normalize(text: str) -> str:
    return " ".join(_WORD.findall(text.lower()))


def token_jaccard(a: str, b: str) -> float:
    ta, tb = set(_WORD.findall(a.lower())), set(_WORD.findall(b.lower()))
    if not ta and not tb:
        return 1.0
    return len(ta & tb) / len(ta | tb)


def similarity(a: str, b: str) -> float:
    return SequenceMatcher(None, normalize(a), normalize(b)).ratio()


def wilson(successes: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (0.0, 1.0)
    p = successes / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (max(0.0, centre - half), min(1.0, centre + half))


JUDGE_PROMPT = """You are comparing two assistant answers to the same user request.
Pick the answer that is more correct, specific and helpful. Reply with exactly one
token: A, B, or TIE.

Question:
{question}

Answer A:
{a}

Answer B:
{b}
"""


def heuristic_judge(messages: list[Message]) -> str:
    """Offline judge: prefer the substantive answer that covers more of the question."""
    body = messages[-1]["content"]
    q = body.split("Question:\n", 1)[-1].split("\n\nAnswer A:\n", 1)[0]
    a = body.split("\n\nAnswer A:\n", 1)[-1].split("\n\nAnswer B:\n", 1)[0]
    b = body.split("\n\nAnswer B:\n", 1)[-1]
    qt = set(_WORD.findall(q.lower()))

    def score(ans: str) -> float:
        toks = _WORD.findall(ans.lower())
        hedge = 2.0 if re.search(r"not certain|depends on your setup|cannot", ans.lower()) else 0
        return len(qt & set(toks)) + min(len(toks), 30) / 10 - hedge

    sa, sb = score(a), score(b)
    if abs(sa - sb) < 0.5:
        return "TIE"
    return "A" if sa > sb else "B"


def default_judge() -> LLM:
    return MockLLM(model="mock-judge", responder=heuristic_judge)


def _parse_verdict(text: str) -> str:
    m = re.search(r"\b(TIE|A|B)\b", text.strip().upper())
    return m.group(1) if m else "TIE"


def judge_pair(judge: LLM, question: str, primary: str, shadow: str) -> str:
    """Return ``win``/``loss``/``tie`` for the shadow answer, order-debiased."""
    first = _parse_verdict(
        judge.complete(
            [
                {
                    "role": "user",
                    "content": JUDGE_PROMPT.format(question=question, a=primary, b=shadow),
                }
            ]
        ).text
    )
    second = _parse_verdict(
        judge.complete(
            [
                {
                    "role": "user",
                    "content": JUDGE_PROMPT.format(question=question, a=shadow, b=primary),
                }
            ]
        ).text
    )
    shadow_first = {"A": "loss", "B": "win", "TIE": "tie"}[first]
    shadow_second = {"A": "win", "B": "loss", "TIE": "tie"}[second]
    return shadow_first if shadow_first == shadow_second else "tie"


@dataclass
class SideStats:
    model: str
    mean_cost_usd: float
    total_cost_usd: float
    mean_tokens_out: float
    latency_p50_s: float
    latency_p95_s: float
    error_rate: float


@dataclass
class ShadowReport:
    n_pairs: int
    n_compared: int
    primary: SideStats
    shadow: SideStats
    exact_agreement: float
    near_agreement: float
    mean_jaccard: float
    mean_similarity: float
    cost_delta_pct: float
    latency_p50_delta_s: float
    latency_p95_delta_s: float
    judge: dict | None
    verdict: str
    reasons: list[str] = field(default_factory=list)
    similarity_hist: list[int] = field(default_factory=list)
    disagreements: list[dict] = field(default_factory=list)
    errors: list[dict] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)


def _question(messages: list[Message]) -> str:
    return next((m["content"] for m in reversed(messages) if m.get("role") == "user"), "")


def _side_stats(sides: list[dict], ok: list[dict], n: int) -> SideStats:
    lat = np.array([s["latency_s"] for s in ok]) if ok else np.zeros(1)
    cost = [s["cost_usd"] for s in ok]
    return SideStats(
        model=sides[0]["model"] if sides else "",
        mean_cost_usd=float(np.mean(cost)) if cost else 0.0,
        total_cost_usd=float(np.sum(cost)) if cost else 0.0,
        mean_tokens_out=float(np.mean([s["tokens_out"] for s in ok])) if ok else 0.0,
        latency_p50_s=float(np.percentile(lat, 50)),
        latency_p95_s=float(np.percentile(lat, 95)),
        error_rate=sum(1 for s in sides if s.get("error")) / n if n else 0.0,
    )


def build_report(
    pairs: list[dict],
    judge: LLM | None = None,
    max_error_rate: float = 0.02,
    top_k: int = 8,
) -> ShadowReport:
    """Compute agreement, judge win rate, cost and latency deltas, and a verdict."""
    n = len(pairs)
    ok = [p for p in pairs if not p["primary"].get("error") and not p["shadow"].get("error")]
    if not ok:
        raise ValueError(f"no comparable pairs among {n} logged pairs")
    prim = _side_stats([p["primary"] for p in pairs], [p["primary"] for p in ok], n)
    shad = _side_stats([p["shadow"] for p in pairs], [p["shadow"] for p in ok], n)

    exact, jac, sim, rows = 0, [], [], []
    for p in ok:
        a, b = p["primary"]["text"], p["shadow"]["text"]
        same = normalize(a) == normalize(b)
        exact += same
        j, s = token_jaccard(a, b), similarity(a, b)
        jac.append(j)
        sim.append(s)
        rows.append((s, same, p))

    judge_summary, decided = None, 0
    if judge is not None:
        tally = {"win": 0, "loss": 0, "tie": 0}
        for s, same, p in rows:
            outcome = (
                "tie"
                if same
                else judge_pair(
                    judge, _question(p["messages"]), p["primary"]["text"], p["shadow"]["text"]
                )
            )
            tally[outcome] += 1
            p["_judge"] = outcome
        decided = tally["win"] + tally["loss"]
        lo, hi = wilson(tally["win"], decided)
        judge_summary = {
            "model": judge.model,
            **tally,
            "win_rate": tally["win"] / decided if decided else 0.5,
            "win_rate_ci95": [lo, hi],
            "non_loss_rate": (tally["win"] + tally["tie"]) / len(rows),
        }

    cost_delta = (
        (shad.mean_cost_usd - prim.mean_cost_usd) / prim.mean_cost_usd * 100
        if (prim.mean_cost_usd)
        else 0.0
    )
    reasons: list[str] = []
    if shad.error_rate > max_error_rate:
        verdict = "BLOCK"
        reasons.append(f"candidate error rate {shad.error_rate:.1%} > {max_error_rate:.0%}")
    elif judge_summary and decided and judge_summary["win_rate_ci95"][1] < 0.5:
        verdict = "BLOCK"
        reasons.append("judge says candidate loses significantly (win-rate CI below 50%)")
    elif judge_summary and decided and judge_summary["win_rate_ci95"][0] > 0.5:
        verdict = "PROMOTE"
        reasons.append("judge says candidate wins significantly")
    elif cost_delta < 0 and (judge_summary is None or judge_summary["non_loss_rate"] >= 0.9):
        verdict = "PROMOTE"
        reasons.append(f"candidate {abs(cost_delta):.0f}% cheaper at quality parity")
    else:
        verdict = "HOLD"
        reasons.append("no significant quality difference and no cost win; collect more pairs")
    if (
        judge_summary
        and decided
        and verdict != "BLOCK"
        and judge_summary["loss"] > judge_summary["win"]
    ):
        reasons.append(f"{judge_summary['loss']} judged losses — review disagreements first")

    hist = np.histogram(sim, bins=10, range=(0, 1))[0].tolist()
    worst = sorted((r for r in rows if not r[1]), key=lambda r: r[0])[:top_k]
    return ShadowReport(
        n_pairs=n,
        n_compared=len(ok),
        primary=prim,
        shadow=shad,
        exact_agreement=exact / len(ok),
        near_agreement=sum(j >= JACCARD_AGREE for j in jac) / len(ok),
        mean_jaccard=float(np.mean(jac)),
        mean_similarity=float(np.mean(sim)),
        cost_delta_pct=cost_delta,
        latency_p50_delta_s=shad.latency_p50_s - prim.latency_p50_s,
        latency_p95_delta_s=shad.latency_p95_s - prim.latency_p95_s,
        judge=judge_summary,
        verdict=verdict,
        reasons=reasons,
        similarity_hist=hist,
        disagreements=[
            {
                "request_id": p["request_id"],
                "question": _question(p["messages"]),
                "primary": p["primary"]["text"],
                "shadow": p["shadow"]["text"],
                "similarity": round(s, 3),
                "judge": p.get("_judge"),
            }
            for s, _, p in worst
        ],
        errors=[
            {"request_id": p["request_id"], "error": p["shadow"]["error"]}
            for p in pairs
            if p["shadow"].get("error")
        ][:top_k],
    )


def render_markdown(r: ShadowReport) -> str:
    p, s = r.primary, r.shadow
    lines = [
        f"# Shadow report: `{s.model}` vs `{p.model}`",
        "",
        f"**Verdict: {r.verdict}** — " + "; ".join(r.reasons),
        "",
        f"{r.n_pairs} shadow pairs logged, {r.n_compared} comparable (both sides succeeded).",
        "",
        "| metric | primary | shadow | delta |",
        "|---|---:|---:|---:|",
        (
            f"| mean cost / request | ${p.mean_cost_usd:.6f} | ${s.mean_cost_usd:.6f} "
            f"| {r.cost_delta_pct:+.1f}% |"
        ),
        (
            f"| latency p50 | {p.latency_p50_s:.3f}s | {s.latency_p50_s:.3f}s "
            f"| {r.latency_p50_delta_s:+.3f}s |"
        ),
        (
            f"| latency p95 | {p.latency_p95_s:.3f}s | {s.latency_p95_s:.3f}s "
            f"| {r.latency_p95_delta_s:+.3f}s |"
        ),
        f"| mean output tokens | {p.mean_tokens_out:.1f} | {s.mean_tokens_out:.1f} | |",
        f"| error rate | {p.error_rate:.1%} | {s.error_rate:.1%} | |",
        "",
        "## Agreement",
        "",
        f"- exact (normalised): **{r.exact_agreement:.1%}**",
        f"- near (token Jaccard ≥ {JACCARD_AGREE}): **{r.near_agreement:.1%}**",
        f"- mean token Jaccard: {r.mean_jaccard:.3f}; mean similarity: {r.mean_similarity:.3f}",
    ]
    if r.judge:
        j = r.judge
        lines += [
            "",
            f"## Judge (`{j['model']}`, both orders)",
            "",
            f"- shadow wins {j['win']}, losses {j['loss']}, ties {j['tie']}",
            (
                f"- win rate (decided pairs) **{j['win_rate']:.1%}** "
                f"(95% CI {j['win_rate_ci95'][0]:.1%}–{j['win_rate_ci95'][1]:.1%})"
            ),
            f"- non-loss rate {j['non_loss_rate']:.1%}",
        ]
    if r.disagreements:
        lines += [
            "",
            "## Largest disagreements",
            "",
            "| request | sim | judge | primary | shadow |",
            "|---|---:|---|---|---|",
        ]
        for d in r.disagreements:
            lines.append(
                f"| {d['request_id']} | {d['similarity']:.2f} | {d['judge'] or ''} "
                f"| {_cell(d['primary'])} | {_cell(d['shadow'])} |"
            )
    if r.errors:
        lines += ["", "## Shadow errors (first few)", ""]
        lines += [f"- `{e['request_id']}`: {e['error']}" for e in r.errors]
    return "\n".join(lines) + "\n"


def _cell(text: str, n: int = 90) -> str:
    text = text.replace("|", "\\|").replace("\n", " ")
    return text if len(text) <= n else text[: n - 1] + "…"


_CSS = """
:root{--bg:#fcfcfb;--card:#ffffff;--ink:#0b0b0b;--ink2:#52514e;--line:#e4e3df;
--s1:#2a78d6;--s2:#eb6834;--good:#0ca30c;--warn:#fab219;--bad:#d03b3b}
@media (prefers-color-scheme: dark){:root:not([data-theme="light"]){--bg:#1a1a19;--card:#232322;
--ink:#fff;--ink2:#c3c2b7;--line:#383835;--s1:#3987e5;--s2:#d95926}}
:root[data-theme="dark"]{--bg:#1a1a19;--card:#232322;--ink:#fff;--ink2:#c3c2b7;--line:#383835;
--s1:#3987e5;--s2:#d95926}
body{margin:0;background:var(--bg);color:var(--ink);font:15px/1.5 system-ui,sans-serif}
main{max-width:960px;margin:0 auto;padding:24px 16px}
h1{font-size:22px;margin:0 0 4px}h2{font-size:17px;margin:28px 0 8px}
.muted{color:var(--ink2)}.tiles{display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));
gap:12px;margin:16px 0}.tile{background:var(--card);border:1px solid var(--line);border-radius:10px;
padding:12px}.tile b{display:block;font-size:22px}.badge{display:inline-block;padding:2px 10px;
border-radius:999px;font-weight:600;color:#fff}.PROMOTE{background:var(--good)}
.HOLD{background:#8a6d00}.BLOCK{background:var(--bad)}
table{border-collapse:collapse;width:100%;background:var(--card);font-size:14px}
th,td{border-bottom:1px solid var(--line);padding:6px 8px;text-align:left;vertical-align:top}
td.n{text-align:right;font-variant-numeric:tabular-nums}.scroll{overflow-x:auto}
svg{max-width:680px;display:block}svg text{fill:var(--ink2);font-size:11px}.legend span{display:inline-flex;align-items:center;
gap:6px;margin-right:14px}.sw{width:10px;height:10px;border-radius:2px;display:inline-block}
"""


def _hist_svg(hist: list[int]) -> str:
    w, h, pad = 560, 180, 28
    top = max(hist) or 1
    bw = (w - 2 * pad) / len(hist)
    bars = []
    for i, c in enumerate(hist):
        bh = (h - 2 * pad) * c / top
        x, y = pad + i * bw + 1, h - pad - bh
        bars.append(
            f'<rect x="{x:.1f}" y="{y:.1f}" width="{bw - 2:.1f}" height="{bh:.1f}" rx="3" '
            f'fill="var(--s1)"><title>similarity {i / 10:.1f}–{(i + 1) / 10:.1f}: {c} pairs'
            f"</title></rect>"
        )
        if c:
            bars.append(
                f'<text x="{x + bw / 2 - 1:.1f}" y="{y - 4:.1f}" text-anchor="middle">{c}</text>'
            )
    ticks = "".join(
        f'<text x="{pad + i * bw:.1f}" y="{h - 10}" text-anchor="middle">{i / 10:.1f}</text>'
        for i in range(0, 11, 2)
    )
    return (
        f'<svg viewBox="0 0 {w} {h}" width="100%" role="img" aria-label="similarity histogram">'
        f'<line x1="{pad}" x2="{w - pad}" y1="{h - pad}" y2="{h - pad}" stroke="var(--line)"/>'
        + "".join(bars)
        + ticks
        + "</svg>"
    )


def _latency_svg(r: ShadowReport) -> str:
    w, h, pad = 560, 150, 70
    groups = [
        ("p50", r.primary.latency_p50_s, r.shadow.latency_p50_s),
        ("p95", r.primary.latency_p95_s, r.shadow.latency_p95_s),
    ]
    top = max(max(a, b) for _, a, b in groups) or 1
    out = []
    for i, (label, a, b) in enumerate(groups):
        y0 = 20 + i * 60
        out.append(f'<text x="8" y="{y0 + 20}">{label}</text>')
        for j, (v, col) in enumerate(((a, "--s1"), (b, "--s2"))):
            bw = (w - pad - 60) * v / top
            y = y0 + j * 22
            out.append(
                f'<rect x="{pad}" y="{y}" width="{bw:.1f}" height="18" rx="3" fill="var({col})">'
                f"<title>{'primary' if j == 0 else 'shadow'} {label}: {v:.3f}s</title></rect>"
                f'<text x="{pad + bw + 6:.1f}" y="{y + 13}">{v:.2f}s</text>'
            )
    return f'<svg viewBox="0 0 {w} {h}" width="100%" role="img" aria-label="latency">{"".join(out)}</svg>'


def render_html(r: ShadowReport) -> str:
    e = html.escape
    tiles = [
        ("cost delta", f"{r.cost_delta_pct:+.1f}%"),
        ("exact agreement", f"{r.exact_agreement:.0%}"),
        ("near agreement", f"{r.near_agreement:.0%}"),
        ("shadow error rate", f"{r.shadow.error_rate:.1%}"),
        ("p95 latency delta", f"{r.latency_p95_delta_s:+.2f}s"),
    ]
    if r.judge:
        tiles.insert(3, ("judge win rate", f"{r.judge['win_rate']:.0%}"))
    tile_html = "".join(
        f'<div class="tile"><span class="muted">{k}</span><b>{v}</b></div>' for k, v in tiles
    )
    dis = "".join(
        f"<tr><td>{e(d['request_id'])}</td><td class=n>{d['similarity']:.2f}</td>"
        f"<td>{e(d['judge'] or '')}</td><td>{e(d['primary'])}</td><td>{e(d['shadow'])}</td></tr>"
        for d in r.disagreements
    )
    legend = (
        '<div class="legend muted"><span><i class="sw" style="background:var(--s1)"></i>'
        f'primary ({e(r.primary.model)})</span><span><i class="sw" style="background:var(--s2)">'
        f"</i>shadow ({e(r.shadow.model)})</span></div>"
    )
    return f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Shadow Routing Report</title><style>{_CSS}</style></head><body><main>
<h1>Shadow report: {e(r.shadow.model)} vs {e(r.primary.model)}</h1>
<p><span class="badge {r.verdict}">{r.verdict}</span> <span class="muted">{e("; ".join(r.reasons))}</span></p>
<p class="muted">{r.n_pairs} pairs logged · {r.n_compared} comparable</p>
<div class="tiles">{tile_html}</div>
<h2>Answer similarity distribution</h2>{_hist_svg(r.similarity_hist)}
<h2>Latency</h2>{legend}{_latency_svg(r)}
<h2>Largest disagreements</h2><div class="scroll"><table><thead><tr><th>request</th><th>sim</th>
<th>judge</th><th>primary</th><th>shadow</th></tr></thead><tbody>{dis}</tbody></table></div>
</main></body></html>
"""


def write_report(r: ShadowReport, out_dir: str | Path) -> dict[str, Path]:
    import json

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    paths = {"md": out / "report.md", "html": out / "report.html", "json": out / "report.json"}
    paths["md"].write_text(render_markdown(r))
    paths["html"].write_text(render_html(r))
    paths["json"].write_text(json.dumps(r.to_dict(), indent=2))
    return paths
