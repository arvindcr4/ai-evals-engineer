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

    lines.append("## Findings by oracle")
    lines.append("")
    header = "| oracle | " + " | ".join(c.guardrail for c in campaigns) + " |"
    lines.append(header)
    lines.append("|" + "---|" * (len(campaigns) + 1))
    for oracle in _ORACLES:
        cells = [str(c.oracle_counts.get(oracle, 0)) for c in campaigns]
        lines.append(f"| {oracle} | " + " | ".join(cells) + " |")
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


def campaign_from_findings(rows: list[dict]) -> list[CampaignResult]:
    """Reconstruct per-guardrail campaigns from a findings JSONL."""
    by_guard: dict[str, list[Finding]] = defaultdict(list)
    runs: dict[str, int] = {}
    failed: dict[str, int] = {}
    for r in rows:
        if "_runs" in r:  # summary marker rows (also surface zero-finding levels)
            runs[r["guardrail"]] = r["_runs"]
            failed[r["guardrail"]] = r.get("_failed_runs", 0)
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
