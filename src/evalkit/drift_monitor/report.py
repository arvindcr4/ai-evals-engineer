"""Markdown + self-contained HTML trend report over the metrics store."""

from __future__ import annotations

import html
from pathlib import Path

from evalkit.drift_monitor.detect import DIRECTION
from evalkit.drift_monitor.store import MetricStore

_CSS = """
:root{--bg:#fcfcfb;--card:#fff;--ink:#0b0b0b;--ink2:#52514e;--line:#e4e3df;--s1:#2a78d6;
--warn:#fab219;--crit:#d03b3b}
@media (prefers-color-scheme: dark){:root:not([data-theme="light"]){--bg:#1a1a19;--card:#232322;
--ink:#fff;--ink2:#c3c2b7;--line:#383835;--s1:#3987e5}}
:root[data-theme="dark"]{--bg:#1a1a19;--card:#232322;--ink:#fff;--ink2:#c3c2b7;--line:#383835;
--s1:#3987e5}
body{margin:0;background:var(--bg);color:var(--ink);font:15px/1.5 system-ui,sans-serif}
main{max-width:1040px;margin:0 auto;padding:24px 16px}h1{font-size:22px;margin:0}
h2{font-size:17px;margin:28px 0 8px}.muted{color:var(--ink2)}
.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(300px,1fr));gap:12px}
.card{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:10px 12px}
.card h3{font-size:14px;margin:0 0 4px;font-weight:600}svg text{fill:var(--ink2);font-size:10px}
table{border-collapse:collapse;width:100%;background:var(--card);font-size:13px}
th,td{border-bottom:1px solid var(--line);padding:5px 8px;text-align:left}
.sev{font-weight:600}.critical{color:var(--crit)}.warning{color:#b07a00}
.scroll{overflow-x:auto}
"""


def _spark(dates: list[str], series: dict[str, float], alerts: dict[str, str]) -> str:
    w, h, px, py = 300, 110, 34, 14
    pts = [(i, series[d]) for i, d in enumerate(dates) if d in series]
    if len(pts) < 2:
        return "<p class=muted>not enough data</p>"
    vals = [v for _, v in pts]
    lo, hi = min(vals), max(vals)
    span = (hi - lo) or abs(hi) or 1.0

    def xy(i: int, v: float) -> tuple[float, float]:
        x = px + (w - px - 6) * i / max(len(dates) - 1, 1)
        y = h - py - (h - 2 * py) * (v - lo) / span
        return x, y

    path = " ".join(
        f"{'M' if j == 0 else 'L'}{x:.1f},{y:.1f}"
        for j, (x, y) in enumerate(xy(i, v) for i, v in pts)
    )
    marks = []
    for i, v in pts:
        d = dates[i]
        x, y = xy(i, v)
        sev = alerts.get(d)
        r, fill = (
            (4.5, f"var(--{'crit' if sev == 'critical' else 'warn'})") if sev else (2, "var(--s1)")
        )
        marks.append(
            f'<circle cx="{x:.1f}" cy="{y:.1f}" r="{r}" fill="{fill}" stroke="var(--card)" '
            f'stroke-width="1.5"><title>{d}: {v:.4g}{" — " + sev if sev else ""}</title></circle>'
        )
    axis = (
        f'<text x="2" y="{py + 3}">{hi:.3g}</text><text x="2" y="{h - py + 3}">{lo:.3g}</text>'
        f'<text x="{px}" y="{h - 1}">{dates[0][5:]}</text>'
        f'<text x="{w - 6}" y="{h - 1}" text-anchor="end">{dates[-1][5:]}</text>'
    )
    return (
        f'<svg viewBox="0 0 {w} {h}" width="100%" role="img">'
        f'<line x1="{px}" x2="{w - 6}" y1="{h - py}" y2="{h - py}" stroke="var(--line)"/>'
        f'<path d="{path}" fill="none" stroke="var(--s1)" stroke-width="2"/>'
        + "".join(marks)
        + axis
        + "</svg>"
    )


def _alert_index(alerts: list[dict]) -> dict[str, dict[str, str]]:
    idx: dict[str, dict[str, str]] = {}
    for a in alerts:
        cur = idx.setdefault(a["metric"], {})
        if cur.get(a["date"]) != "critical":
            cur[a["date"]] = a["severity"]
    return idx


def render_markdown(store: MetricStore) -> str:
    dates = store.dates()
    alerts = store.alerts()
    first = next((a["date"] for a in alerts), None)
    lines = [
        "# Drift monitor report",
        "",
        f"{len(dates)} days monitored ({dates[0]} → {dates[-1]}), {len(alerts)} alerts"
        + (f"; first alert on **{first}**." if first else "; no alerts."),
        "",
        "| metric | baseline (first 7d) | latest | change |",
        "|---|---:|---:|---:|",
    ]
    for m in store.metric_names():
        s = store.series(m)
        vals = [s[d] for d in dates if d in s]
        if not vals:
            continue
        base = sum(vals[:7]) / len(vals[:7])
        change = (vals[-1] - base) / base * 100 if base else 0.0
        lines.append(f"| {m} | {base:.4g} | {vals[-1]:.4g} | {change:+.1f}% |")
    if alerts:
        lines += [
            "",
            "## Alerts",
            "",
            "| date | severity | method | message |",
            "|---|---|---|---|",
        ]
        lines += [
            f"| {a['date']} | {a['severity']} | {a['method']} | {a['message']} |" for a in alerts
        ]
    return "\n".join(lines) + "\n"


def render_html(store: MetricStore) -> str:
    e = html.escape
    dates = store.dates()
    alerts = store.alerts()
    idx = _alert_index(alerts)
    metrics = [m for m in DIRECTION if m in store.metric_names()]
    cards = "".join(
        f'<div class="card"><h3>{e(m)}</h3>'
        f'<div class="muted">{"lower" if DIRECTION[m] > 0 else "higher"} is better</div>'
        f"{_spark(dates, store.series(m), idx.get(m, {}))}</div>"
        for m in metrics
    )
    rows = "".join(
        f"<tr><td>{a['date']}</td><td class='sev {a['severity']}'>{a['severity']}</td>"
        f"<td>{e(a['metric'])}</td><td>{a['method']}</td><td>{e(a['message'])}</td></tr>"
        for a in alerts
    )
    first = next((a["date"] for a in alerts), "none")
    return f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Drift Monitor Report</title><style>{_CSS}</style></head><body><main>
<h1>Production drift monitor</h1>
<p class="muted">{len(dates)} days · {e(dates[0])} → {e(dates[-1])} · {len(alerts)} alerts ·
first alert {e(first)} · large dots = alert days (red critical, amber warning)</p>
<div class="grid">{cards}</div>
<h2>Alerts</h2><div class="scroll"><table><thead><tr><th>date</th><th>severity</th><th>metric</th>
<th>method</th><th>message</th></tr></thead><tbody>{rows}</tbody></table></div>
</main></body></html>
"""


def write_report(store: MetricStore, out_dir: str | Path) -> dict[str, Path]:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    if not store.dates():
        raise ValueError("metrics store is empty; run `run` or `backfill` first")
    paths = {"md": out / "drift_report.md", "html": out / "drift_report.html"}
    paths["md"].write_text(render_markdown(store))
    paths["html"].write_text(render_html(store))
    return paths
