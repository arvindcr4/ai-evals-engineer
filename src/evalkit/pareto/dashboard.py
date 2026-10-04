"""Self-contained HTML dashboard, Markdown summary and JSON for Pareto results.

One inline-SVG scatter per tenant (log-cost x axis, success-rate y axis with
Wilson whiskers), a staircase frontier through the non-dominated configs,
dominated configs greyed out, the System-One → System-Two router curve, and
(for what-ifs) the previous frontier as a ghost line. A tiny vanilla-JS layer
handles the tenant selector and hover tooltips.
"""

from __future__ import annotations

import html
import json
import math
from dataclasses import dataclass, field
from pathlib import Path

from evalkit.pareto.analysis import (
    ConfigStats,
    RouterPoint,
    mark_router_frontier,
    pareto_frontier,
    recommend,
    simulate_router,
    summarize,
)


@dataclass
class TenantView:
    tenant: str
    stats: list[ConfigStats]
    router: list[RouterPoint] = field(default_factory=list)
    ghost: list[tuple[float, float]] = field(default_factory=list)
    recommendation: dict = field(default_factory=dict)


def auto_router_pair(stats: list[ConfigStats], cost_ratio: float = 0.1) -> tuple[str, str]:
    """Cheap = best frontier config costing ≤ ``cost_ratio`` × the top config; expensive = top."""
    front = [s for s in stats if s.on_frontier]
    top = max(front, key=lambda s: s.success_rate)
    cheap_pool = [s for s in front if s.mean_cost <= cost_ratio * top.mean_cost] or front[:1]
    cheap = max(cheap_pool, key=lambda s: s.success_rate)
    return cheap.config, top.config


def build_views(
    rows: list[dict],
    router: tuple[str, str] | None = None,
    cascade: bool = True,
    ghost_rows: list[dict] | None = None,
) -> list[TenantView]:
    """Summaries, router curves and optional ghost frontiers for every tenant."""
    summary = summarize(rows)
    ghost_summary = summarize(ghost_rows) if ghost_rows else {}
    views = []
    for tenant, stats in summary.items():
        names = {s.config for s in stats}
        pair = router if router and set(router) <= names else auto_router_pair(stats)
        curve = simulate_router(rows, tenant, pair[0], pair[1], cascade=cascade)
        mark_router_frontier(stats, curve)
        ghost = []
        if tenant in ghost_summary:
            g = ghost_summary[tenant]
            idx = pareto_frontier([(s.mean_cost, s.success_rate) for s in g])
            ghost = [(g[i].mean_cost, g[i].success_rate) for i in idx]
        views.append(TenantView(tenant, stats, curve, ghost, recommend(stats)))
    return views


def best_router(view: TenantView) -> RouterPoint | None:
    """Cheapest router point within 2pp of always-escalate, if it actually saves money."""
    if not view.router:
        return None
    always = view.router[-1]
    ok = [p for p in view.router if p.success_rate >= always.success_rate - 0.02]
    best = min(ok, key=lambda p: p.mean_cost)
    return best if best.escalation_rate < 1.0 and best.mean_cost < always.mean_cost else None


def _value_text(rec: dict) -> str:
    if rec["value_pick"] == rec["best_quality"]:
        return f"{rec['value_pick']} · nothing cheaper is close"
    return f"{rec['value_pick']} · {rec['savings_vs_best']:.0%} cheaper"


def _money(x: float) -> str:
    if x == math.inf:
        return "∞"
    return f"${x:.4f}" if x < 0.1 else f"${x:.3f}"


# ---------- SVG -------------------------------------------------------------

W, H = 720, 400
ML, MR, MT, MB = 60, 24, 18, 46


def _scales(view: TenantView):
    costs = [s.mean_cost for s in view.stats] + [p.mean_cost for p in view.router]
    costs += [c for c, _ in view.ghost]
    lo, hi = math.log10(min(costs)), math.log10(max(costs))
    pad = max(0.15, (hi - lo) * 0.06)
    x0, x1 = lo - pad, hi + pad
    ys = [s.ci_lo for s in view.stats] + [p.success_rate for p in view.router]
    y0 = max(0.0, math.floor((min(ys) - 0.03) * 10) / 10)
    y1 = min(1.0, math.ceil((max(s.ci_hi for s in view.stats) + 0.02) * 10) / 10)

    def sx(c: float) -> float:
        return ML + (W - ML - MR) * (math.log10(c) - x0) / (x1 - x0)

    def sy(v: float) -> float:
        return H - MB - (H - MT - MB) * (v - y0) / (y1 - y0)

    return sx, sy, (x0, x1), (y0, y1)


def _staircase(points: list[tuple[float, float]], sx, sy) -> str:
    pts = sorted(points)
    d = []
    for i, (c, v) in enumerate(pts):
        x, y = sx(c), sy(v)
        if i == 0:
            d.append(f"M{x:.1f},{y:.1f}")
        else:
            d.append(f"H{x:.1f}V{y:.1f}")
    return " ".join(d)


def _tip(lines: list[str]) -> str:
    return html.escape(" | ".join(lines), quote=True)


def _scatter(view: TenantView) -> str:
    sx, sy, (x0, x1), (y0, y1) = _scales(view)
    label = f"cost vs success for {html.escape(view.tenant)}"
    out = [f'<svg viewBox="0 0 {W} {H}" width="100%" role="img" aria-label="{label}">']
    for k in range(math.floor(x0), math.ceil(x1) + 1):
        for m in (1, 2, 5):
            v = m * 10.0**k
            if not x0 <= math.log10(v) <= x1:
                continue
            x = sx(v)
            out.append(f'<line class="grid" x1="{x:.1f}" x2="{x:.1f}" y1="{MT}" y2="{H - MB}"/>')
            out.append(
                f'<text class="tick" x="{x:.1f}" y="{H - MB + 16}" '
                f'text-anchor="middle">{_money(v)}</text>'
            )
    yv = y0
    while yv <= y1 + 1e-9:
        y = sy(yv)
        out.append(f'<line class="grid" x1="{ML}" x2="{W - MR}" y1="{y:.1f}" y2="{y:.1f}"/>')
        out.append(
            f'<text class="tick" x="{ML - 8}" y="{y + 4:.1f}" text-anchor="end">{yv:.0%}</text>'
        )
        yv += 0.1
    out.append(
        f'<text class="axis" x="{(ML + W - MR) / 2}" y="{H - 8}" text-anchor="middle">'
        "mean cost per task (log scale)</text>"
    )
    out.append(
        f'<text class="axis" transform="translate(14 {(MT + H - MB) / 2}) rotate(-90)" '
        'text-anchor="middle">task success rate</text>'
    )

    if view.ghost:
        out.append(f'<path class="ghost" d="{_staircase(view.ghost, sx, sy)}"/>')
    front = [(s.mean_cost, s.success_rate) for s in view.stats if s.on_frontier]
    out.append(f'<path class="frontier" d="{_staircase(front, sx, sy)}"/>')
    if view.router:
        pts = " ".join(f"{sx(p.mean_cost):.1f},{sy(p.success_rate):.1f}" for p in view.router)
        out.append(f'<polyline class="router" points="{pts}"/>')
        for p in view.router:
            tip = _tip(
                [
                    p.config,
                    f"success {p.success_rate:.1%}",
                    f"cost {_money(p.mean_cost)}",
                    f"escalated {p.escalation_rate:.0%}",
                ]
            )
            cls = "rpt on" if p.on_frontier else "rpt"
            out.append(
                f'<circle class="{cls}" data-tip="{tip}" cx="{sx(p.mean_cost):.1f}" '
                f'cy="{sy(p.success_rate):.1f}" r="{4 if p.on_frontier else 3}"/>'
            )

    placed: list[tuple[float, float, float, float]] = [
        (sx(p.mean_cost) - 4, sy(p.success_rate) - 4, sx(p.mean_cost) + 4, sy(p.success_rate) + 4)
        for p in view.router
    ] + [
        (sx(s.mean_cost) - 7, sy(s.success_rate) - 7, sx(s.mean_cost) + 7, sy(s.success_rate) + 7)
        for s in view.stats
    ]
    for s in sorted(view.stats, key=lambda s: -s.success_rate):
        x, y = sx(s.mean_cost), sy(s.success_rate)
        cls = "pt on" if s.on_frontier else "pt off"
        out.append(
            f'<line class="ci" x1="{x:.1f}" x2="{x:.1f}" y1="{sy(s.ci_lo):.1f}" '
            f'y2="{sy(s.ci_hi):.1f}"/>'
        )
        tip = _tip(
            [
                s.config,
                f"success {s.success_rate:.1%} (95% CI {s.ci_lo:.0%}–{s.ci_hi:.0%})",
                f"cost/task {_money(s.mean_cost)}",
                f"cost/success {_money(s.cost_per_success)}",
                f"p95 {s.p95_latency:.1f}s",
                "frontier" if s.on_frontier else "dominated",
            ]
        )
        out.append(
            f'<circle class="{cls}" data-tip="{tip}" cx="{x:.1f}" cy="{y:.1f}" '
            f'r="{6 if s.on_frontier else 5}"/>'
        )
        lw = 6.4 * len(s.config)
        for dx, dy, anchor in (
            (9, -8, "start"),
            (9, 14, "start"),
            (0, -12, "middle"),
            (-9, -8, "end"),
            (-9, 14, "end"),
            (0, 22, "middle"),
            (9, -22, "start"),
            (-9, 28, "end"),
        ):
            lx0 = {"start": x + dx, "end": x + dx - lw, "middle": x - lw / 2}[anchor]
            box = (lx0, y + dy - 10, lx0 + lw, y + dy + 2)
            inside = ML <= box[0] and box[2] <= W - MR and MT <= box[1]
            clash = any(
                not (box[2] < b[0] or box[0] > b[2] or box[3] < b[1] or box[1] > b[3])
                for b in placed
            )
            if inside and not clash:
                break
        placed.append(box)
        out.append(
            f'<text class="lbl {"on" if s.on_frontier else "off"}" x="{x + dx:.1f}" '
            f'y="{y + dy:.1f}" text-anchor="{anchor}">{html.escape(s.config)}</text>'
        )
    out.append("</svg>")
    return "".join(out)


# ---------- page ------------------------------------------------------------

_CSS = """
:root{color-scheme:light;--bg:#f6f5f2;--card:#fcfcfb;--ink:#0b0b0b;--ink2:#52514e;--ink3:#8a8984;
--line:#e4e3df;--s1:#2a78d6;--s2:#eb6834;--off:#b9b8b2}
@media (prefers-color-scheme: dark){:root:not([data-theme="light"]){color-scheme:dark;--bg:#121211;
--card:#1a1a19;--ink:#fff;--ink2:#c3c2b7;--ink3:#8f8e86;--line:#383835;--s1:#3987e5;--s2:#d95926;
--off:#5c5b56}}
:root[data-theme="dark"]{color-scheme:dark;--bg:#121211;--card:#1a1a19;--ink:#fff;--ink2:#c3c2b7;
--ink3:#8f8e86;--line:#383835;--s1:#3987e5;--s2:#d95926;--off:#5c5b56}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--ink);
font:15px/1.5 system-ui,-apple-system,"Segoe UI",sans-serif}
main{max-width:980px;margin:0 auto;padding:24px 16px 48px}h1{font-size:22px;margin:0 0 2px}
h2{font-size:16px;margin:24px 0 8px}.muted{color:var(--ink2)}
.bar{display:flex;flex-wrap:wrap;gap:12px;align-items:center;margin:16px 0}
select{font:inherit;padding:6px 10px;border-radius:8px;border:1px solid var(--line);
background:var(--card);color:var(--ink)}
.card{background:var(--card);border:1px solid var(--line);border-radius:12px;padding:16px}
.kpis{display:grid;grid-template-columns:repeat(auto-fit,minmax(200px,1fr));gap:12px;margin-bottom:12px}
.kpi{background:var(--card);border:1px solid var(--line);border-radius:12px;padding:12px}
.kpi span{color:var(--ink2);font-size:13px}.kpi b{display:block;font-size:18px;margin-top:2px}
.legend{display:flex;flex-wrap:wrap;gap:16px;font-size:13px;color:var(--ink2);margin:4px 0 8px}
.legend i{display:inline-block;vertical-align:middle;margin-right:6px}
.sw-dot{width:10px;height:10px;border-radius:50%}.sw-line{width:18px;height:0;border-top:2px solid}
svg .grid{stroke:var(--line);stroke-width:1}svg .tick,svg .axis{fill:var(--ink3);font-size:11px}
svg .axis{fill:var(--ink2);font-size:12px}
svg .frontier{fill:none;stroke:var(--s1);stroke-width:2}
svg .ghost{fill:none;stroke:var(--ink3);stroke-width:1.5;stroke-dasharray:3 4}
svg .router{fill:none;stroke:var(--s2);stroke-width:2;stroke-dasharray:6 4}
svg .rpt{fill:var(--s2);stroke:var(--card);stroke-width:1.5}
svg .pt{stroke:var(--card);stroke-width:2}svg .pt.on{fill:var(--s1)}svg .pt.off{fill:var(--off)}
svg .ci{stroke:var(--ink3);stroke-width:1.5;opacity:.6}
svg .lbl{font-size:11.5px;paint-order:stroke;stroke:var(--card);stroke-width:4px;
stroke-linejoin:round}svg .lbl.on{fill:var(--ink)}svg .lbl.off{fill:var(--ink3)}
svg [data-tip]{cursor:pointer}svg [data-tip]:hover{stroke:var(--ink);stroke-width:2}
.chart{min-width:560px}.tenant{display:none}.tenant.active{display:block}
.scroll{overflow-x:auto}table{border-collapse:collapse;width:100%;font-size:13.5px}
th,td{padding:7px 10px;border-bottom:1px solid var(--line);text-align:left;white-space:nowrap}
th{color:var(--ink2);font-weight:600}td.n,th.n{text-align:right;font-variant-numeric:tabular-nums}
tr.off td,tr.off .muted{color:var(--ink3)}
#tip{position:fixed;pointer-events:none;background:var(--ink);color:var(--bg);font-size:12px;
padding:6px 8px;border-radius:6px;max-width:320px;display:none;z-index:9;line-height:1.4}
"""

_JS = """
const sel=document.getElementById('tenant');
function show(t){document.querySelectorAll('.tenant').forEach(s=>s.classList.toggle('active',s.dataset.t===t));
 sel.value=t;try{history.replaceState(null,'','#'+encodeURIComponent(t))}catch(e){}}
sel.addEventListener('change',e=>show(e.target.value));
const h=decodeURIComponent(location.hash.slice(1));show([...sel.options].some(o=>o.value===h)?h:sel.value);
const tip=document.getElementById('tip');
document.addEventListener('mousemove',e=>{const t=e.target.closest&&e.target.closest('[data-tip]');
 if(!t){tip.style.display='none';return}
 tip.innerHTML=t.dataset.tip.split(' | ').map((s,i)=>i?s:'<b>'+s+'</b>').join('<br>');
 tip.style.display='block';tip.style.left=Math.min(e.clientX+14,innerWidth-330)+'px';
 tip.style.top=(e.clientY+14)+'px'});
"""


def _tenant_section(v: TenantView, ghost_label: str | None) -> str:
    e = html.escape
    rec = v.recommendation
    br = best_router(v)
    kpis = [
        ("best quality", f"{rec['best_quality']} · {rec['best_success']:.0%}"),
        ("value pick (≤2pp off best)", _value_text(rec)),
    ]
    if not br and v.router:
        kpis.append(("router at parity", "no saving vs always escalating"))
    if br:
        kpis.append(
            (
                "router at parity",
                (
                    f"t={br.threshold:.2f} · {br.escalation_rate:.0%} escalated · "
                    f"{_money(br.mean_cost)}/task"
                ),
            )
        )
    kpi_html = "".join(f'<div class="kpi"><span>{k}</span><b>{e(val)}</b></div>' for k, val in kpis)
    legend = [
        '<span><i class="sw-dot" style="background:var(--s1)"></i>frontier config</span>',
        '<span><i class="sw-dot" style="background:var(--off)"></i>dominated config</span>',
    ]
    if v.router:
        r0 = v.router[0]
        legend.append(
            f'<span><i class="sw-line" style="border-color:var(--s2);border-top-style:dashed">'
            f"</i>router {e(r0.cheap)} → {e(r0.expensive)}</span>"
        )
    if v.ghost:
        legend.append(
            '<span><i class="sw-line" style="border-color:var(--ink3);'
            f'border-top-style:dotted"></i>{e(ghost_label or "previous frontier")}</span>'
        )
    rows = "".join(
        f'<tr class="{"on" if s.on_frontier else "off"}"><td>{e(s.config)}</td><td class=n>{s.n}</td>'
        f"<td class=n>{s.success_rate:.1%} <span class=muted>({s.ci_lo:.0%}–{s.ci_hi:.0%})</span></td>"
        f"<td class=n>{_money(s.mean_cost)}</td><td class=n>{_money(s.cost_per_success)}</td>"
        f"<td class=n>{s.p95_latency:.1f}s</td><td>{'● frontier' if s.on_frontier else 'dominated'}"
        f"</td></tr>"
        for s in v.stats
    )
    rrows = (
        "".join(
            f"<tr><td class=n>{p.threshold:.2f}</td><td class=n>{p.escalation_rate:.0%}</td>"
            f"<td class=n>{p.success_rate:.1%}</td><td class=n>{_money(p.mean_cost)}</td>"
            f"<td class=n>{_money(p.cost_per_success)}</td><td class=n>{p.p95_latency:.1f}s</td></tr>"
            for p in v.router
            if p.on_frontier
        )
        or "<tr><td colspan=6 class=muted>no router threshold beats the config frontier</td></tr>"
    )
    return f"""<section class="tenant" data-t="{e(v.tenant)}">
<div class="kpis">{kpi_html}</div>
<div class="card"><div class="legend">{"".join(legend)}</div>
<div class="scroll"><div class="chart">{_scatter(v)}</div></div></div>
<h2>Configurations</h2><div class="card scroll"><table><thead><tr><th>config</th><th class=n>tasks</th>
<th class=n>success (95% CI)</th><th class=n>cost / task</th><th class=n>cost / success</th>
<th class=n>p95 latency</th><th>status</th></tr></thead><tbody>{rows}</tbody></table></div>
<h2>Router thresholds on the frontier</h2><div class="card scroll"><table><thead><tr>
<th class=n>threshold</th><th class=n>escalated</th><th class=n>success</th><th class=n>cost / task</th>
<th class=n>cost / success</th><th class=n>p95 latency</th></tr></thead><tbody>{rrows}</tbody></table></div>
</section>"""


def render_html(
    views: list[TenantView], title: str, subtitle: str = "", ghost_label: str | None = None
) -> str:
    e = html.escape
    opts = "".join(f'<option value="{e(v.tenant)}">{e(v.tenant)}</option>' for v in views)
    sections = "".join(_tenant_section(v, ghost_label) for v in views)
    return f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Cost-Quality Pareto</title><style>{_CSS}</style></head><body><main>
<h1>{e(title)}</h1><p class="muted">{e(subtitle)}</p>
<div class="bar"><label for="tenant" class="muted">Tenant</label><select id="tenant">{opts}</select></div>
{sections}<div id="tip"></div></main><script>{_JS}</script></body></html>
"""


def render_markdown(views: list[TenantView], title: str, notes: list[str] | None = None) -> str:
    lines = [f"# {title}", ""] + (notes or [])
    for v in views:
        rec, br = v.recommendation, best_router(v)
        if lines[-1]:
            lines.append("")
        lines += [
            f"## {v.tenant}",
            "",
            f"- best quality: **{rec['best_quality']}** ({rec['best_success']:.1%})",
            f"- value pick (≤2pp off best): {_value_text(rec)}",
        ]
        if br:
            lines.append(
                f"- router `{br.cheap}` → `{br.expensive}` at threshold {br.threshold:.2f}: "
                f"{br.success_rate:.1%} success, {br.escalation_rate:.0%} escalated, "
                f"{_money(br.mean_cost)}/task vs {_money(v.router[-1].mean_cost)} always-escalate"
            )
        elif v.router:
            r0 = v.router[0]
            lines.append(
                f"- router `{r0.cheap}` → `{r0.expensive}`: no threshold stays within 2pp of "
                "always-escalate at lower cost"
            )
        lines += [
            "",
            "| config | success (95% CI) | cost/task | cost/success | p95 latency | frontier |",
            "|---|---:|---:|---:|---:|:---:|",
        ]
        lines += [
            f"| {s.config} | {s.success_rate:.1%} ({s.ci_lo:.0%}–{s.ci_hi:.0%}) | "
            f"{_money(s.mean_cost)} | {_money(s.cost_per_success)} | {s.p95_latency:.1f}s | "
            f"{'●' if s.on_frontier else ''} |"
            for s in v.stats
        ]
    return "\n".join(lines) + "\n"


def to_json(views: list[TenantView], extra: dict | None = None) -> dict:
    def clean(d: dict) -> dict:
        return {k: (None if isinstance(x, float) and math.isinf(x) else x) for k, x in d.items()}

    return {
        "tenants": {
            v.tenant: {
                "configs": [clean(s.to_dict()) for s in v.stats],
                "router": [clean(p.to_dict()) for p in v.router],
                "recommendation": v.recommendation,
            }
            for v in views
        },
        **(extra or {}),
    }


def write_outputs(
    views: list[TenantView],
    out_dir: str | Path,
    title: str,
    subtitle: str = "",
    notes: list[str] | None = None,
    extra: dict | None = None,
    ghost_label: str | None = None,
    stem: str = "pareto",
) -> dict[str, Path]:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    paths = {"html": out / f"{stem}.html", "md": out / f"{stem}.md", "json": out / f"{stem}.json"}
    paths["html"].write_text(render_html(views, title, subtitle, ghost_label))
    paths["md"].write_text(render_markdown(views, title, notes))
    paths["json"].write_text(json.dumps(to_json(views, extra), indent=2, default=str))
    return paths
