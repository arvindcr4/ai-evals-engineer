"""Run memory strategies through scenarios, probe them, and sweep noise levels.

For each (strategy, noise level, trial) the harness streams the scenario into a
fresh memory, checks budget compliance after every turn, then asks each probe
against the context the strategy renders for that question. Answers are scored
as *correct* (current value), *stale* (a superseded value) or *miss*.

Offline, the reader is a deterministic extractive model: it returns the latest
value stated for the asked attribute in the visible context, or ``unknown`` —
so scores measure exactly what the memory kept, not reader quality.
"""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass

from evalkit.context_eviction.memory import FACT_RE, Memory, make_memory
from evalkit.context_eviction.scenario import Probe, Scenario, generate
from evalkit.core.llm import LLM, Message, MockLLM

READER_PROMPT = """Answer the question using only the assistant's memory below.
If a fact was updated, answer with the latest value. If the memory does not
contain the answer, reply exactly "unknown". Reply with the value only.

MEMORY:
{memory}

QUESTION: {question}
ANSWER:"""

_Q = re.compile(r"QUESTION: What is the user's (.+?)\?")


def extractive_reader(messages: list[Message]) -> str:
    """Offline reader: latest stated value for the asked attribute, else ``unknown``."""
    prompt = messages[-1]["content"]
    q = _Q.search(prompt)
    if not q:
        return "unknown"
    memory = prompt.split("MEMORY:", 1)[-1].split("QUESTION:", 1)[0]
    values = [v.strip() for a, v in FACT_RE.findall(memory) if a == q.group(1)]
    return values[-1] if values else "unknown"


def _norm(s: str) -> str:
    return " ".join(re.findall(r"[a-z0-9]+", s.lower()))


def classify(answer: str, probe: Probe) -> str:
    a = f" {_norm(answer)} "
    if f" {_norm(probe.current)} " in a:
        return "correct"
    if any(f" {_norm(v)} " in a for v in probe.stale):
        return "stale"
    return "miss"


@dataclass
class RunResult:
    strategy: str
    noise: int
    seed: int
    correct: int
    stale: int
    miss: int
    n_probes: int
    pinned_correct: int
    n_pinned: int
    updated_correct: int
    n_updated: int
    budget_ok: int
    checkpoints: int
    peak_tokens: int
    llm_calls: int

    @property
    def recall(self) -> float:
        return self.correct / self.n_probes if self.n_probes else 0.0

    @property
    def stale_rate(self) -> float:
        return self.stale / self.n_probes if self.n_probes else 0.0

    @property
    def compliance(self) -> float:
        return self.budget_ok / self.checkpoints if self.checkpoints else 1.0


def run_scenario(scenario: Scenario, memory: Memory, reader: LLM) -> tuple[RunResult, list[dict]]:
    """Stream ``scenario`` into ``memory``, then answer every probe with ``reader``."""
    ok = checks = peak = 0
    for turn in scenario.turns:
        memory.add(turn)
        tok = memory.tokens()
        peak, checks, ok = max(peak, tok), checks + 1, ok + (tok <= memory.budget)

    counts = {"correct": 0, "stale": 0, "miss": 0}
    pinned_ok = updated_ok = 0
    answers: list[dict] = []
    for p in scenario.probes:
        ctx = memory.context(p.question)
        tok = sum(len(x.split()) for x in ctx)
        peak, checks, ok = max(peak, tok), checks + 1, ok + (tok <= memory.budget)
        prompt = READER_PROMPT.format(memory="\n".join(ctx), question=p.question)
        ans = reader.complete([{"role": "user", "content": prompt}]).text.strip()
        verdict = classify(ans, p)
        counts[verdict] += 1
        pinned_ok += p.pinned and verdict == "correct"
        updated_ok += p.updated and verdict == "correct"
        answers.append({"slot": p.slot, "answer": ans, "expected": p.current, "verdict": verdict})

    res = RunResult(
        strategy=memory.name, noise=scenario.noise, seed=scenario.seed,
        correct=counts["correct"], stale=counts["stale"], miss=counts["miss"],
        n_probes=len(scenario.probes),
        pinned_correct=pinned_ok, n_pinned=sum(p.pinned for p in scenario.probes),
        updated_correct=updated_ok, n_updated=sum(p.updated for p in scenario.probes),
        budget_ok=ok, checkpoints=checks, peak_tokens=peak, llm_calls=memory.llm_calls,
    )
    return res, answers


@dataclass
class Cell:
    strategy: str
    noise: int
    trials: int
    recall: float
    stale_rate: float
    miss_rate: float
    pinned_recall: float
    updated_recall: float
    compliance: float
    peak_tokens: int
    llm_calls: float


def _aggregate(rs: list[RunResult]) -> Cell:
    n = len(rs)
    probes = sum(r.n_probes for r in rs)
    pinned = sum(r.n_pinned for r in rs)
    updated = sum(r.n_updated for r in rs)
    return Cell(
        strategy=rs[0].strategy, noise=rs[0].noise, trials=n,
        recall=round(sum(r.correct for r in rs) / probes, 3),
        stale_rate=round(sum(r.stale for r in rs) / probes, 3),
        miss_rate=round(sum(r.miss for r in rs) / probes, 3),
        pinned_recall=round(sum(r.pinned_correct for r in rs) / pinned, 3) if pinned else 0.0,
        updated_recall=round(sum(r.updated_correct for r in rs) / updated, 3) if updated else 0.0,
        compliance=round(sum(r.budget_ok for r in rs) / sum(r.checkpoints for r in rs), 3),
        peak_tokens=max(r.peak_tokens for r in rs),
        llm_calls=round(sum(r.llm_calls for r in rs) / n, 1),
    )


def default_reader() -> LLM:
    return MockLLM(model="mock-extractive-reader", responder=extractive_reader)


def sweep(strategies: list[str], noise_levels: list[int], *, budget: int = 400, trials: int = 3,
          seed: int = 0, reader: LLM | None = None, summarizer: LLM | None = None,
          n_facts: int = 6, n_updates: int = 3, pinned_updates: int = 1,
          hard_noise: float = 0.1) -> list[Cell]:
    """Grid of strategy × noise; every strategy sees the same scenarios (paired comparison)."""
    reader = reader or default_reader()
    cells: list[Cell] = []
    for noise in noise_levels:
        scenarios = [generate(noise, n_facts=n_facts, n_updates=n_updates,
                              pinned_updates=pinned_updates,
                              seed=seed + 7919 * t + noise, hard_noise=hard_noise)
                     for t in range(trials)]
        for name in strategies:
            results = []
            for sc in scenarios:
                mem = make_memory(name, sc.system, budget,
                                  llm=summarizer if name == "summary" else None)
                results.append(run_scenario(sc, mem, reader)[0])
            cells.append(_aggregate(results))
    return cells


_BARS = " ▁▂▃▄▅▆▇█"


def format_table(cells: list[Cell], budget: int) -> str:
    head = (f"{'strategy':<10} {'noise':>5} {'recall':>6} {'stale':>6} {'miss':>6} "
            f"{'pinned':>6} {'upd':>6} {'budget':>7} {'peak':>6} {'llm':>6}")
    lines = [(f"budget = {budget} tokens; recall/stale/miss over all probes; "
             "pinned = system-prompt facts, upd = updated facts"), head, "-" * len(head)]
    for c in sorted(cells, key=lambda c: (c.strategy, c.noise)):
        lines.append(
            f"{c.strategy:<10} {c.noise:>5} {c.recall:>6.2f} {c.stale_rate:>6.2f} "
            f"{c.miss_rate:>6.2f} {c.pinned_recall:>6.2f} {c.updated_recall:>6.2f} "
            f"{c.compliance:>6.0%} {c.peak_tokens:>6} {c.llm_calls:>6.1f}"
        )
    lines.append("\nrecall curve over noise levels:")
    by: dict[str, list[Cell]] = {}
    for c in cells:
        by.setdefault(c.strategy, []).append(c)
    for name, cs in by.items():
        cs.sort(key=lambda c: c.noise)
        spark = "".join(_BARS[round(c.recall * (len(_BARS) - 1))] for c in cs)
        pts = " ".join(f"{c.noise}:{c.recall:.2f}" for c in cs)
        lines.append(f"  {name:<10} |{spark}|  {pts}")
    return "\n".join(lines)


def cells_to_dicts(cells: list[Cell]) -> list[dict]:
    return [asdict(c) for c in cells]
