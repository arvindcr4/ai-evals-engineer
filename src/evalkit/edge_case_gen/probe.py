"""Probe a system under test with the golden edge-case set.

Runs every case through the system, compares the output with the proposed
label, and groups failures and crashes by axis and category so you see which
kind of edge case breaks it. Cases still flagged for human review are scored
but reported as provisional.
"""

from __future__ import annotations

import traceback
from collections import defaultdict
from collections.abc import Callable
from dataclasses import asdict, dataclass
from typing import Any

from evalkit.edge_case_gen.cases import EdgeCase


@dataclass
class ProbeResult:
    id: str
    axis: str
    category: str
    status: str  # pass | fail | crash | unlabeled
    expected: str | None
    got: Any
    provisional: bool
    error: str | None = None
    generator: str = ""


def probe(system: Callable[[dict], Any], cases: list[EdgeCase]) -> list[ProbeResult]:
    out = []
    for c in cases:
        err, got = None, None
        try:
            got = system(dict(c.input))
            if c.label is None:
                status = "unlabeled"
            else:
                status = "pass" if str(got) == c.label else "fail"
        except Exception as e:  # noqa: BLE001 - crashes are findings, not errors
            status = "crash"
            tb = traceback.extract_tb(e.__traceback__)[-1]
            err = f"{type(e).__name__}: {e} (line {tb.lineno})"
        gen = str(c.provenance.get("generator", "")).split(":", 1)[0]
        out.append(ProbeResult(c.id, c.axis, c.category, status, c.label,
                               got, c.needs_human_review, err, gen))
    return out


def summarize(results: list[ProbeResult]) -> dict:
    def tally(rs: list[ProbeResult]) -> dict:
        # Unlabeled cases that ran cleanly have no verdict, so they are left out
        # of the denominator rather than silently counted as passes.
        n = len(rs)
        bad = sum(1 for r in rs if r.status in ("fail", "crash"))
        scored = n - sum(r.status == "unlabeled" for r in rs)
        return {"n": n, "scored": scored, "unlabeled": n - scored,
                "fail": sum(r.status == "fail" for r in rs),
                "crash": sum(r.status == "crash" for r in rs),
                "failure_rate": round(bad / scored, 4) if scored else 0.0}

    by_axis: dict[str, list[ProbeResult]] = defaultdict(list)
    by_cat: dict[str, list[ProbeResult]] = defaultdict(list)
    by_gen: dict[str, list[ProbeResult]] = defaultdict(list)
    for r in results:
        by_gen[r.generator or "?"].append(r)
        by_axis[r.axis].append(r)
        by_cat[f"{r.axis}/{r.category}"].append(r)
    confirmed = [r for r in results if not r.provisional]
    worst = sorted(((k, tally(v)) for k, v in by_cat.items()),
                   key=lambda kv: (-kv[1]["failure_rate"], -kv[1]["n"], kv[0]))
    return {
        "overall": tally(results),
        "confirmed_only": tally(confirmed),
        "by_axis": {k: tally(v) for k, v in sorted(by_axis.items())},
        "by_generator": {k: tally(v) for k, v in sorted(by_gen.items())},
        "worst_categories": [{"category": k, **t} for k, t in worst if t["failure_rate"] > 0],
        "failures": [asdict(r) for r in results if r.status in ("fail", "crash")],
    }


def format_summary(s: dict, show: int = 8) -> str:
    o, c = s["overall"], s["confirmed_only"]
    lines = [
        (f"probe: {o['n']} cases, {o['fail']} wrong, {o['crash']} crashed "
         f"({o['failure_rate']:.1%} of {o['scored']} scored, {o['unlabeled']} unlabeled); "
         f"confirmed-label subset: {c['failure_rate']:.1%} of {c['scored']}"),
        "  by axis:",
    ]
    for ax, t in s["by_axis"].items():
        lines.append(f"    {ax:<14} {t['failure_rate']:>6.1%}  (fail={t['fail']} crash={t['crash']}"
                     f" / {t['n']})")
    if len(s.get("by_generator", {})) > 1:
        lines.append("  by generator: " + ", ".join(
            f"{g}={t['failure_rate']:.1%} of {t['scored']}" for g, t in s["by_generator"].items()))
    lines.append("  worst categories:" + ("" if s["worst_categories"] else " none"))
    for w in s["worst_categories"][:show]:
        lines.append(f"    {w['category']:<34} {w['failure_rate']:>6.1%} of {w['n']}")
    lines.append("  sample failures:" + ("" if s["failures"] else " none"))
    for f in s["failures"][:show]:
        why = f["error"] or f"expected={f['expected']} got={f['got']}"
        tag = " [provisional]" if f["provisional"] else ""
        lines.append(f"    {f['id']} {f['axis']}/{f['category']}: {why}{tag}")
    return "\n".join(lines)
