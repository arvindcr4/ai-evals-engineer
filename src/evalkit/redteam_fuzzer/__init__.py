"""Agent Red-Team Fuzzer.

An automated harness that injects prompt injections (direct and indirect),
malformed tool schemas, infinite loops and resource-exhaustion payloads into a
tool-using agent to test its guardrails. A deterministic toy target with
``none`` / ``basic`` / ``hardened`` guardrail levels makes the resilience
differences visible; a seeded mutation loop re-mutates cases that broke
something, and oracles flag leaked canaries, forbidden-tool invocations,
crashes, propagated schema violations and non-termination. Output is a
findings JSONL (each row a minimal reproducer) plus a Markdown scorecard.
"""

from evalkit.redteam_fuzzer.agent import (
    CANARY,
    FORBIDDEN_TOOLS,
    AgentResult,
    GuardConfig,
    TargetAgent,
    ToyAgent,
)
from evalkit.redteam_fuzzer.attacks import AttackCase, mutate, seed_corpus
from evalkit.redteam_fuzzer.fuzzer import (
    CampaignResult,
    Finding,
    evaluate,
    run_campaign,
)
from evalkit.redteam_fuzzer.report import render_scorecard

__all__ = [
    "CANARY",
    "FORBIDDEN_TOOLS",
    "AgentResult",
    "AttackCase",
    "CampaignResult",
    "Finding",
    "GuardConfig",
    "TargetAgent",
    "ToyAgent",
    "evaluate",
    "mutate",
    "render_scorecard",
    "run_campaign",
    "seed_corpus",
]
