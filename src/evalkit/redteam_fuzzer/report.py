"""Render fuzzing campaigns into a Markdown resilience scorecard.

The scorecard compares guardrail levels side by side: resilience score, the
worst surviving finding, and a breakdown by oracle. It is built either from
in-memory :class:`~evalkit.redteam_fuzzer.fuzzer.CampaignResult` objects or
reloaded from a findings JSONL (grouped by the ``guardrail`` field).
"""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path

from evalkit.redteam_fuzzer.fuzzer import SEVERITY_ORDER, CampaignResult, Finding

_ORACLES = [
    "canary_leaked",
    "forbidden_tool_invoked",
    "unhandled_exception",
    "schema_violation_propagated",
    "no_termination",
    "token_budget_exceeded",
]


def render_scorecard(campaigns: list[CampaignResult]) -> str:
    """Markdown scorecard for one or more guardrail campaigns."""
    lines = ["# Agent Red-Team Fuzzer — Resilience Scorecard", ""]
    lines.append(f"Guardrail levels fuzzed: {', '.join(c.guardrail for c in campaigns)}")
    lines.append("")
    lines.append("| guardrail | runs | attack success | findings | critical | high | medium "
                 "| resilience |")
    lines.append("|---|---|---|---|---|---|---|---|")
    for c in campaigns:
        lines.append(
            f"| {c.guardrail} | {c.runs} | {c.attack_success_rate:.1%} | {len(c.findings)} | "
            f"{c.severity_counts.get('critical', 0)} | {c.severity_counts.get('high', 0)} | "
            f"{c.severity_counts.get('medium', 0)} | {c.resilience_score:.3f} |"
        )
    lines.append("")
    spend = [c for c in campaigns if c.cost_usd > 0 or c.infra_errors]
    if spend:
        total = sum(c.cost_usd for c in campaigns)
        calls = sum(c.llm_calls for c in campaigns)
        infra = sum(c.infra_errors for c in campaigns)
        lines.append(f"Policy-model calls: {calls}, API cost: ${total:.4f}, "
                     f"runs dropped for provider errors: {infra}")
        lines.append("")

    lines.append("## Findings by oracle")
    lines.append("")
    header = "| oracle | " + " | ".join(c.guardrail for c in campaigns) + " |"
    lines.append(header)
    lines.append("|" + "---|" * (len(campaigns) + 1))
    for oracle in _ORACLES:
        cells = [str(c.oracle_counts.get(oracle, 0)) for c in campaigns]
        lines.append(f"| {oracle} | " + " | ".join(cells) + " |")
    lines.append("")

    classes = sorted({k for c in campaigns for k in c.class_stats})
    if classes:
        lines.append("## Attack success by class (failed runs / runs)")
        lines.append("")
        lines.append("| class | " + " | ".join(c.guardrail for c in campaigns) + " |")
        lines.append("|" + "---|" * (len(campaigns) + 1))
        for cls in classes:
            lines.append(f"| {cls} | " + " | ".join(_cell(c.class_stats.get(cls)) for c in campaigns)
                         + " |")
        lines.append("")

    for c in campaigns:
        lines.append(f"## {c.guardrail}: top reproducers")
        lines.append("")
        if not c.findings:
            lines.append("_No findings — all attacks were blocked._")
            lines.append("")
            continue
        for f in c.findings[:5]:
            rep = f.reproducer
            lines.append(
                f"- **{f.severity.upper()} {f.oracle}** `{f.case_id}` ({f.detail}) — "
                f"vector=`{rep['vector']}` family=`{rep['family']}` "
                f"lineage=`{'/'.join(rep['lineage']) or 'seed'}`"
            )
        lines.append("")
    return "\n".join(lines)


def _cell(st: list[int] | None) -> str:
    if not st or not st[0]:
        return "–"
    return f"{st[1]}/{st[0]} ({st[1] / st[0]:.0%})"


def summary_row(c: CampaignResult) -> dict:
    """The per-level marker row appended to a findings JSONL (lets ``report``
    rescore without the live campaign)."""
    return {"guardrail": c.guardrail, "_runs": c.runs, "_failed_runs": c.failed_runs,
            "_infra_errors": c.infra_errors, "_llm_calls": c.llm_calls,
            "_cost_usd": round(c.cost_usd, 6), "_class_stats": c.class_stats}


def render_comparison(baseline: list[CampaignResult], other: list[CampaignResult],
                      labels: tuple[str, str] = ("mock", "real")) -> str:
    """Side-by-side attack success per guardrail level and attack class."""
    a = {c.guardrail: c for c in baseline}
    b = {c.guardrail: c for c in other}
    levels = [lvl for lvl in a if lvl in b]
    la, lb = labels
    lines = [f"# Red-team fuzzer: {la} vs {lb}", ""]
    lines.append(f"| guardrail | {la} attack success | {lb} attack success | {la} resilience "
                 f"| {lb} resilience |")
    lines.append("|---|---|---|---|---|")
    for lvl in levels:
        lines.append(f"| {lvl} | {a[lvl].attack_success_rate:.1%} ({a[lvl].runs} runs) | "
                     f"{b[lvl].attack_success_rate:.1%} ({b[lvl].runs} runs) | "
                     f"{a[lvl].resilience_score:.3f} | {b[lvl].resilience_score:.3f} |")
    lines.append("")
    lines.append("## Oracle counts (deduped findings)")
    lines.append("")
    lines.append("| oracle | " + " | ".join(f"{lvl} {la} / {lb}" for lvl in levels) + " |")
    lines.append("|" + "---|" * (len(levels) + 1))
    for oracle in _ORACLES:
        cells = [f"{a[lvl].oracle_counts.get(oracle, 0)} / {b[lvl].oracle_counts.get(oracle, 0)}"
                 for lvl in levels]
        lines.append(f"| {oracle} | " + " | ".join(cells) + " |")
    lines.append("")
    classes = sorted({k for lvl in levels for c in (a[lvl], b[lvl]) for k in c.class_stats})
    if classes:
        lines.append("## Attack success by class (failed / runs)")
        lines.append("")
        lines.append("| class | " + " | ".join(f"{lvl} {la} | {lvl} {lb}" for lvl in levels) + " |")
        lines.append("|" + "---|" * (2 * len(levels) + 1))
        for cls in classes:
            cells = [f"{_cell(a[lvl].class_stats.get(cls))} | {_cell(b[lvl].class_stats.get(cls))}"
                     for lvl in levels]
            lines.append(f"| {cls} | " + " | ".join(cells) + " |")
        lines.append("")
    return "\n".join(lines)


def campaign_from_findings(rows: list[dict]) -> list[CampaignResult]:
    """Reconstruct per-guardrail campaigns from a findings JSONL."""
    by_guard: dict[str, list[Finding]] = defaultdict(list)
    runs: dict[str, int] = {}
    failed: dict[str, int] = {}
    extra: dict[str, dict] = {}
    for r in rows:
        if "_runs" in r:  # summary marker rows (also surface zero-finding levels)
            runs[r["guardrail"]] = r["_runs"]
            failed[r["guardrail"]] = r.get("_failed_runs", 0)
            extra[r["guardrail"]] = r
            by_guard.setdefault(r["guardrail"], [])
            continue
        by_guard[r["guardrail"]].append(Finding(**{k: r[k] for k in (
            "oracle", "severity", "guardrail", "detail", "reproducer", "case_id")}))
    out: list[CampaignResult] = []
    for guard, findings in by_guard.items():
        findings.sort(key=lambda f: SEVERITY_ORDER[f.severity], reverse=True)
        # Without a summary row the run count is unknown; fall back to one run
        # per finding (every logged run failed), the pessimistic reading.
        n = runs.get(guard, len(findings))
        res = CampaignResult(guardrail=guard, seed=0, iterations=0, runs=n,
                             failed_runs=failed.get(guard, n))
        meta = extra.get(guard, {})
        res.infra_errors = meta.get("_infra_errors", 0)
        res.llm_calls = meta.get("_llm_calls", 0)
        res.cost_usd = meta.get("_cost_usd", 0.0)
        res.class_stats = {k: list(v) for k, v in meta.get("_class_stats", {}).items()}
        res.findings = findings
        for f in findings:
            res.oracle_counts[f.oracle] = res.oracle_counts.get(f.oracle, 0) + 1
            res.severity_counts[f.severity] = res.severity_counts.get(f.severity, 0) + 1
        out.append(res)
    return out


def write_scorecard(campaigns: list[CampaignResult], out_dir: str | Path) -> Path:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    path = out / "scorecard.md"
    path.write_text(render_scorecard(campaigns))
    return path
