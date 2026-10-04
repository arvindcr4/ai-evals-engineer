"""Markdown rendering of a :class:`ScanReport`."""

from __future__ import annotations

from evalkit.contamination.scanner import ScanReport

_ORDER = {"contaminated": 0, "suspicious": 1, "clean": 2}


def _cell(text: str, limit: int = 90) -> str:
    text = " ".join(text.split()).replace("|", "\\|")
    return text if len(text) <= limit else text[: limit - 1] + "…"


def to_markdown(report: ScanReport, max_rows: int = 200) -> str:
    """Human-readable report: summary, per-method catch counts, flagged items."""
    s, c, cfg = report.summary, report.corpus, report.config
    lines = [
        "# Contamination report",
        "",
        (
            f"Scanned **{c['documents']}** training documents ({c['tokens']:,} tokens, "
            f"{c['windows']:,} windows) against **{s['items']}** eval items "
            f"in {c['seconds']}s."
        ),
        "",
        "| status | items |",
        "|---|---|",
        f"| contaminated | {s['contaminated']} |",
        f"| suspicious | {s['suspicious']} |",
        f"| clean | {s['clean']} |",
        "",
        (
            f"Contamination rate **{s['contamination_rate']:.1%}**, flag rate "
            f"{s['flag_rate']:.1%}; {c['flagged_documents']} training document(s) implicated."
        ),
        "",
        "Flagged by method: "
        + ", ".join(f"{m} {k}" for m, k in s["flagged_by_method"].items())
        + f" (n={cfg['n']}, shingle={cfg['shingle']}, embedder={cfg['embedder']}).",
        "",
    ]
    flagged = sorted(
        (r for r in report.items if r.status != "clean"),
        key=lambda r: (_ORDER[r.status], -r.overlap_ratio, -r.max_containment, -r.max_cosine),
    )
    if not flagged:
        lines.append("No eval item shows evidence of leakage.")
        return "\n".join(lines) + "\n"
    lines += [
        "## Flagged items",
        "",
        "| id | status | methods | n-gram overlap | longest span | containment | cosine | docs |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for r in flagged[:max_rows]:
        docs = sorted({*r.ngram_docs, *(d for d in (r.containment_doc, r.cosine_doc) if d)})
        lines.append(
            f"| {r.id} | **{r.status}** | {', '.join(r.methods)} | {r.overlap_ratio:.0%} | "
            f"{r.longest_common_ngram} | {r.max_containment:.2f} | {r.max_cosine:.2f} | "
            f"{_cell(', '.join(docs), 40)} |"
        )
    lines += ["", "## Reasons", ""]
    for r in flagged[:max_rows]:
        lines.append(f"- **{r.id}** ({r.status})")
        lines += [f"  - {reason}" for reason in r.reasons]
        if r.judge:
            lines.append(f"  - judge: {r.judge}")
        if r.evidence:
            lines.append(f"  - evidence: \"{_cell(r.evidence, 160)}\"")
    return "\n".join(lines) + "\n"
