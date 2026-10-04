"""Synthetic production traffic with a quality regression partway through.

Before ``decay_day`` the assistant is healthy. From then on a severity ramp
(full strength after ``ramp_days``) raises refusals, empty answers and tool
errors, shortens answers, slows responses and shifts the tool mix toward a
new ``browse`` tool — the shape of a bad prompt or model rollout.

``style="filler"`` (default, used by the offline demo) builds answers from
generic sentences that never address the question: fine for heuristic
scorers and the mock judge, but a real LLM judge correctly rates every one of
them 1/5, so the judge score has no headroom to fall. ``style="grounded"``
answers each question from a topical pool, and decay additionally produces
*off-topic* answers (a fluent answer to a different question) — a quality
failure that length/refusal heuristics cannot see but a real judge can.
"""

from __future__ import annotations

import json
from datetime import date, timedelta
from pathlib import Path

import numpy as np

QUESTIONS = [
    "Summarise last quarter's churn by region",
    "Which invoices are overdue more than 60 days?",
    "Draft a reply to the customer asking about refunds",
    "Compare plan A and plan B pricing for 200 seats",
    "Find the top 5 SKUs by margin this month",
    "Explain why the nightly ETL failed",
    "What is our SLA for enterprise support tickets?",
    "Convert 1,250 EUR to USD at today's rate",
]
SENTENCES = [
    "I pulled the latest numbers from the warehouse.",
    "The main driver is a drop in renewals among mid-market accounts.",
    "Three accounts account for most of the variance.",
    "Here is the breakdown by segment with the relevant totals.",
    "I double-checked the figures against the finance export.",
    "Let me know if you want this as a spreadsheet.",
    "The calculation uses the rate published this morning.",
    "Two items need a manual follow-up from the account owner.",
]
REFUSALS = [
    "I'm sorry, but I can't help with that request.",
    "I am unable to access that information right now.",
]
# On-topic answer sentences per QUESTIONS entry (same order), for style="grounded".
GROUNDED: list[list[str]] = [
    [
        "Q2 churn was 4.1% overall, up from 3.6% in Q1.",
        "EMEA had the highest churn at 5.8%, driven by mid-market non-renewals.",
        "North America was flat at 3.2%, and APAC improved to 3.9% from 4.4%.",
        "Most of the EMEA increase came from 14 accounts on the legacy Starter plan.",
        "Excluding those accounts, EMEA churn would have been 4.0%.",
        "I can split this further by plan or by account owner if useful.",
    ],
    [
        "There are 23 invoices more than 60 days overdue, totalling $184,300.",
        "The largest is INV-20931 for Brightwave Ltd at $41,200, now 97 days past due.",
        "Nine of the 23 belong to three customers: Brightwave, Kestrel Foods and Orion Labs.",
        "Seven invoices are over 90 days and should go to collections under the policy.",
        "Reminders were sent for all 23 at day 30; 11 have no reply logged.",
        "I can export the full list with contact owners to a spreadsheet.",
    ],
    [
        "Here is a draft: Hi Dana, thanks for reaching out about a refund.",
        "Annual plans can be refunded pro rata within 30 days of renewal, and yours renewed 12 days ago.",
        "I have started a refund of $1,140 for the unused 11 months to your original card.",
        "You should see it within 5 to 7 business days, and I will confirm by email once it posts.",
        "If you would rather switch to monthly billing instead, I can arrange that today.",
        "Best regards, the Support Team.",
    ],
    [
        "For 200 seats, plan A costs $18 per seat per month, or $43,200 a year.",
        "Plan B costs $24 per seat with a 15% volume discount above 150 seats, about $48,960 a year.",
        "Plan B adds SSO, audit logs and a 99.9% uptime SLA, which plan A lacks.",
        "Plan A is $5,760 a year cheaper at 200 seats.",
        "If you need SSO, plan A plus the SSO add-on ($3 per seat) comes to $50,400, so plan B is cheaper.",
        "My recommendation is plan B if SSO or the SLA matters, otherwise plan A.",
    ],
    [
        "The top 5 SKUs by gross margin this month are listed below.",
        "1. SKU-4471 Pro Annual: 82% margin on $96k revenue.",
        "2. SKU-1020 Analytics add-on: 79% on $41k. 3. SKU-3305 Seat pack 50: 74% on $63k.",
        "4. SKU-2218 Priority support: 71% on $28k. 5. SKU-5102 API tier 2: 68% on $35k.",
        "Together they contribute 61% of this month's gross profit.",
        "SKU-2218 moved into the top 5 after the support price change on the 1st.",
    ],
    [
        "The nightly ETL failed at 02:14 UTC in the load_orders step.",
        "The upstream orders table gained a nullable discount_code column yesterday.",
        "The loader's strict schema check rejected it, so the job aborted after 3 retries.",
        "No data was lost; the staging tables still hold last night's extract.",
        "Adding the column to the warehouse schema and re-running load_orders should fix it.",
        "I would also switch the schema check to warn on additive changes so this does not recur.",
    ],
    [
        "Our enterprise SLA is a 1-hour first response for Severity 1 tickets, 24/7.",
        "Severity 2 tickets get a 4-hour first response during business hours.",
        "Severity 3 and 4 tickets get a response within 1 business day.",
        "Severity 1 issues also get hourly status updates until resolved.",
        "Missing the Sev 1 target in a month triggers a 5% service credit.",
        "The full terms are in section 7 of the enterprise master agreement.",
    ],
    [
        "At today's reference rate of 1 EUR = 1.0842 USD, 1,250 EUR is 1,355.25 USD.",
        "The rate is the ECB reference published at 14:15 CET this afternoon.",
        "Card or bank conversions usually add a 1 to 3% spread, so expect roughly 1,370 to 1,395 USD.",
        "Yesterday's rate was 1.0817, so the euro gained about 0.2% overnight.",
        "Let me know if you need the conversion locked for an invoice.",
    ],
]
BASE_TOOLS = {"search": 0.5, "sql": 0.3, "calculator": 0.2}
DRIFT_TOOLS = {"search": 0.3, "sql": 0.2, "calculator": 0.1, "browse": 0.4}


def severity(day_idx: int, decay_day: int, ramp_days: int = 5) -> float:
    if day_idx < decay_day:
        return 0.0
    return min(1.0, (day_idx - decay_day + 1) / ramp_days)


def _answer(rng: np.random.Generator, target_chars: int) -> str:
    parts: list[str] = []
    while sum(len(p) + 1 for p in parts) < target_chars:
        parts.append(SENTENCES[rng.integers(len(SENTENCES))])
    return " ".join(parts)


def _grounded_answer(q: int, target_chars: int) -> str:
    """Opening sentence of question ``q``'s pool, then further sentences in order until
    ``target_chars`` (never repeats a sentence, so the answer stays coherent)."""
    pool = GROUNDED[q]
    parts = [pool[0]]
    for sent in pool[1:]:
        if sum(len(p) + 1 for p in parts) >= target_chars:
            break
        parts.append(sent)
    return " ".join(parts)


STYLES = ("filler", "grounded")


def generate_day(
    day: str,
    day_idx: int,
    n: int,
    decay_day: int,
    rng: np.random.Generator,
    style: str = "filler",
) -> list[dict]:
    if style not in STYLES:
        raise ValueError(f"unknown style {style!r}; expected one of {STYLES}")
    s = severity(day_idx, decay_day)
    weekend = date.fromisoformat(day).weekday() >= 5
    mix = {k: (1 - s) * BASE_TOOLS.get(k, 0) + s * DRIFT_TOOLS.get(k, 0) for k in DRIFT_TOOLS}
    names, probs = list(mix), np.array(list(mix.values()))
    probs = probs / probs.sum()
    rows = []
    for i in range(n):
        n_tools = int(rng.choice([0, 1, 2, 3], p=[0.3, 0.4, 0.2, 0.1]))
        tool_calls = [
            {"name": str(rng.choice(names, p=probs)), "ok": bool(rng.random() > 0.04 + 0.14 * s)}
            for _ in range(n_tools)
        ]
        q = int(rng.integers(len(QUESTIONS))) if style == "grounded" else -1
        r = rng.random()
        if r < 0.005 + 0.03 * s:
            output = ""
        elif r < 0.035 + 0.15 * s:
            output = REFUSALS[rng.integers(len(REFUSALS))]
        else:
            target = int(rng.lognormal(np.log(550 * (1 - 0.45 * s)), 0.45))
            if style == "grounded":
                # Decay also brings off-topic answers: fluent, but for another question.
                off = rng.random() < 0.35 * s
                aq = (q + 1 + int(rng.integers(len(QUESTIONS) - 1))) % len(QUESTIONS) if off else q
                output = _grounded_answer(aq, target)
            else:
                output = _answer(rng, target)
        latency = float(rng.lognormal(np.log(1.8 * (1 + 0.6 * s) * (0.9 if weekend else 1)), 0.35))
        tokens = len(output) // 4 + 300 + 150 * n_tools
        rows.append(
            {
                "request_id": f"{day}-{i:05d}",
                "ts": f"{day}T{int(rng.integers(24)):02d}:{int(rng.integers(60)):02d}:00Z",
                "user": f"u{int(rng.integers(5000)):04d}",
                "model": "assistant-prod",
                "input": QUESTIONS[q if q >= 0 else rng.integers(len(QUESTIONS))],
                "output": output,
                "tool_calls": tool_calls,
                "latency_s": round(latency, 3),
                "cost_usd": round(tokens * 4e-6, 6),
            }
        )
    return rows


def generate(
    out_dir: str | Path,
    start: str = "2026-09-01",
    days: int = 30,
    per_day: int = 2000,
    decay_day: int = 20,
    seed: int = 7,
    style: str = "filler",
) -> list[Path]:
    """Write ``days`` daily JSONL files to ``out_dir``; returns their paths."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    d0 = date.fromisoformat(start)
    paths = []
    for i in range(days):
        day = (d0 + timedelta(days=i)).isoformat()
        path = out / f"{day}.jsonl"
        with open(path, "w") as f:
            f.writelines(
                json.dumps(row) + "\n"
                for row in generate_day(day, i, per_day, decay_day, rng, style)
            )
        paths.append(path)
    return paths
