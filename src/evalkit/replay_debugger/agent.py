"""A minimal ReAct-style agent loop plus a deterministic toy world.

The agent talks a two-verb protocol, one action per LLM hop::

    CALL <tool> {"json": "args"}
    FINAL <answer>

All side effects go through a :class:`Runtime`, which is what makes recording
and counterfactual replay possible: the loop never calls a model or tool
directly. The toy world (a price/FX calculator) and its scripted policies give
offline, deterministic failures to debug — a policy that hallucinates an
exchange rate and a tool that serves a stale one.
"""

from __future__ import annotations

import ast
import json
import operator
import re
from collections.abc import Callable
from typing import Any, Protocol

from evalkit.core.llm import LLM, Message, MockLLM, get_llm

Tool = Callable[..., Any]

SYSTEM_PROMPT = """You are a careful shopping assistant. Answer by calling tools.
Tools:
- lookup_price {"item": str} -> unit price in USD
- fx_rate {"base": "USD", "quote": str} -> units of quote per 1 base
- calc {"expr": str} -> arithmetic result
Reply with exactly one line, either
CALL <tool> {json args}
or
FINAL <answer as a number rounded to 2 decimals>"""

_ACTION = re.compile(r"^\s*(CALL\s+(\w+)\s*(\{.*\})|FINAL\s+(.+))\s*$", re.DOTALL)


class Runtime(Protocol):
    def llm(self, messages: list[Message]) -> str: ...

    def tool(self, name: str, args: dict[str, Any]) -> Any: ...


def parse_action(text: str) -> tuple[str, str, Any]:
    """Parse one hop into ``("call", tool, args)``, ``("final", "", answer)`` or ``("error", ...)``."""
    line = next((ln for ln in text.strip().splitlines() if ln.strip()), "")
    for candidate in (text.strip(), line):
        m = _ACTION.match(candidate)
        if not m:
            continue
        if m.group(4) is not None:
            return "final", "", m.group(4).strip()
        try:
            return "call", m.group(2), json.loads(m.group(3))
        except json.JSONDecodeError as e:
            return "error", "", f"bad JSON args: {e}"
    return "error", "", "expected `CALL <tool> {json}` or `FINAL <answer>`"


def run_agent(task: str, rt: Runtime, max_hops: int = 8) -> str | None:
    """Run the agent loop on ``task`` through ``rt``; returns the final answer or None."""
    messages: list[Message] = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": task},
    ]
    for _ in range(max_hops):
        text = rt.llm(list(messages))
        messages.append({"role": "assistant", "content": text})
        kind, name, payload = parse_action(text)
        if kind == "final":
            return str(payload)
        if kind == "call":
            obs = rt.tool(name, payload if isinstance(payload, dict) else {})
            messages.append({"role": "user", "content": f"OBSERVATION {name}: {json.dumps(obs)}"})
        else:
            messages.append({"role": "user", "content": f"ERROR: {payload}"})
    return None


# --- toy world -----------------------------------------------------------------------

PRICES_USD = {"widget": 4.25, "gadget": 12.0, "gizmo": 7.5, "sprocket": 0.8}
FX = {"USD": 1.0, "EUR": 0.92, "GBP": 0.79, "INR": 83.1}
_OPS = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul,
        ast.Div: operator.truediv, ast.USub: operator.neg}


def safe_eval(expr: str) -> float:
    """Evaluate a pure arithmetic expression (no names, calls or attributes)."""
    def ev(n: ast.AST) -> float:
        if isinstance(n, ast.Expression):
            return ev(n.body)
        if isinstance(n, ast.Constant) and isinstance(n.value, (int, float)):
            return n.value
        if isinstance(n, ast.BinOp) and type(n.op) in _OPS:
            return _OPS[type(n.op)](ev(n.left), ev(n.right))
        if isinstance(n, ast.UnaryOp) and type(n.op) in _OPS:
            return _OPS[type(n.op)](ev(n.operand))
        raise ValueError(f"unsupported expression: {ast.dump(n)[:40]}")
    return ev(ast.parse(expr, mode="eval"))


def toy_tools(fault: str | None = None) -> dict[str, Tool]:
    """Toy toolset. ``fault="stale_fx"`` makes ``fx_rate`` serve an inverted (stale) rate."""

    def lookup_price(item: str) -> float:
        key = item.lower().rstrip("s")
        if key not in PRICES_USD:
            raise KeyError(f"unknown item {item!r}")
        return PRICES_USD[key]

    def fx_rate(base: str, quote: str) -> float:
        rate = FX[quote.upper()] / FX[base.upper()]
        if fault == "stale_fx" and quote.upper() != base.upper():
            return round(1 / rate, 4)
        return round(rate, 4)

    def calc(expr: str) -> float:
        return round(safe_eval(expr), 4)

    if fault not in (None, "stale_fx"):
        raise ValueError(f"unknown toy fault {fault!r}")
    return {"lookup_price": lookup_price, "fx_rate": fx_rate, "calc": calc}


_TASK = re.compile(r"(\d+)\s+([a-z]+?)s?\b.*?\b([A-Z]{3})\b")


def parse_task(task: str) -> tuple[int, str, str]:
    m = _TASK.search(task)
    if not m:
        raise ValueError(f"cannot parse toy task: {task!r}")
    return int(m.group(1)), m.group(2), m.group(3)


def expected_answer(task: str) -> float:
    qty, item, cur = parse_task(task)
    return round(qty * PRICES_USD[item] * round(FX[cur], 4), 2)


def toy_policy(sloppy: bool = False) -> Callable[[list[Message]], str]:
    """Scripted policy over the conversation so far.

    The careful policy looks up the price, then the rate, then calculates. The
    sloppy one skips ``fx_rate`` and hallucinates parity (rate 1.0) — but, like a
    real model, it uses an FX observation if one is already in its context.
    """

    def respond(messages: list[Message]) -> str:
        task = next(m["content"] for m in messages if m["role"] == "user")
        try:
            qty, item, cur = parse_task(task)
        except ValueError:
            return "FINAL unknown"
        obs: dict[str, Any] = {}
        for m in messages:
            hit = re.match(r"OBSERVATION (\w+): (.*)", m["content"], re.DOTALL)
            if hit:
                obs[hit.group(1)] = json.loads(hit.group(2))
        if "lookup_price" not in obs:
            return f'CALL lookup_price {{"item": "{item}"}}'
        if "fx_rate" not in obs and not sloppy:
            return f'CALL fx_rate {{"base": "USD", "quote": "{cur}"}}'
        if "calc" not in obs:
            rate = obs.get("fx_rate", 1.0)
            return f'CALL calc {{"expr": "{qty} * {obs["lookup_price"]} * {rate}"}}'
        return f"FINAL {float(obs['calc']):.2f}"

    return respond


def resolve_llm(spec: str | None) -> LLM:
    """``toy`` / ``toy:sloppy`` → scripted toy policies; anything else → :func:`get_llm`."""
    if spec in ("toy", "toy:careful"):
        return MockLLM(model="toy-careful", responder=toy_policy(False))
    if spec == "toy:sloppy":
        return MockLLM(model="toy-sloppy", responder=toy_policy(True))
    return get_llm(spec)


def resolve_tools(spec: str | None) -> dict[str, Tool]:
    """``toy`` / ``toy:stale_fx`` toolsets."""
    spec = spec or "toy"
    if spec == "toy":
        return toy_tools()
    if spec.startswith("toy:"):
        return toy_tools(spec.split(":", 1)[1])
    raise ValueError(f"unknown toolset {spec!r} (expected toy or toy:<fault>)")


def numeric_checker(expected: Any, tol: float = 0.011) -> Callable[[str | None], bool]:
    """Pass if the first number in the answer is within ``tol`` of ``expected``;
    non-numeric expectations fall back to a case-insensitive substring match."""

    def check(answer: str | None) -> bool:
        if answer is None:
            return False
        try:
            target = float(expected)
        except (TypeError, ValueError):
            return str(expected).lower() in answer.lower()
        m = re.search(r"-?\d+(?:\.\d+)?", answer.replace(",", ""))
        return bool(m) and abs(float(m.group()) - target) <= tol

    return check
