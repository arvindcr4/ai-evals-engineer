"""Synthetic production traffic with a quality regression partway through.

Before ``decay_day`` the assistant is healthy. From then on a severity ramp
(full strength after ``ramp_days``) raises refusals, empty answers and tool
errors, shortens answers, slows responses and shifts the tool mix toward a
new ``browse`` tool — the shape of a bad prompt or model rollout.
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


def generate_day(
    day: str, day_idx: int, n: int, decay_day: int, rng: np.random.Generator
) -> list[dict]:
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
        r = rng.random()
        if r < 0.005 + 0.03 * s:
            output = ""
        elif r < 0.035 + 0.15 * s:
            output = REFUSALS[rng.integers(len(REFUSALS))]
        else:
            target = int(rng.lognormal(np.log(550 * (1 - 0.45 * s)), 0.45))
            output = _answer(rng, target)
        latency = float(rng.lognormal(np.log(1.8 * (1 + 0.6 * s) * (0.9 if weekend else 1)), 0.35))
        tokens = len(output) // 4 + 300 + 150 * n_tools
        rows.append(
            {
                "request_id": f"{day}-{i:05d}",
                "ts": f"{day}T{int(rng.integers(24)):02d}:{int(rng.integers(60)):02d}:00Z",
                "user": f"u{int(rng.integers(5000)):04d}",
                "model": "assistant-prod",
                "input": QUESTIONS[rng.integers(len(QUESTIONS))],
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
                json.dumps(row) + "\n" for row in generate_day(day, i, per_day, decay_day, rng)
            )
        paths.append(path)
    return paths
