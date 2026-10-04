"""Record, counterfactually replay and bisect agent runs.

* :func:`record` runs the agent live and captures every node into a cassette.
* :class:`Replayer` re-runs the agent where nodes ``< at`` are served from the
  cassette, node ``at`` is replaced by an :class:`Override` (a literal output, a
  different model, or a different tool implementation) and the remaining nodes
  run either ``live`` or ``cached`` (served from the cassette whenever the node's
  input is byte-identical to a recorded one, live otherwise).
* :func:`bisect` swaps each node of a failing run, one at a time, with a
  known-good alternative (reference model / oracle tools) and reports which single
  swap flips the outcome to pass — the step that caused the failure.
"""

from __future__ import annotations

from collections import defaultdict, deque
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, Literal

from evalkit.core.llm import LLM, Message
from evalkit.replay_debugger.agent import Tool, run_agent
from evalkit.replay_debugger.cassette import Cassette, Node

AfterMode = Literal["live", "cached"]
Checker = Callable[[str | None], bool]


class ReplayDivergence(RuntimeError):
    """The agent asked for a different node than the cassette holds before the swap point."""


@dataclass
class Override:
    """What to put at node ``index``: an explicit ``output``, or regenerate it with
    ``llm`` (for LLM nodes) / ``tools`` (for tool nodes)."""

    index: int
    output: Any = None
    llm: LLM | None = None
    tools: dict[str, Tool] | None = None

    def resolve(self, node: Node) -> Any:
        if self.output is not None:
            return self.output
        if node.kind == "llm" and self.llm is not None:
            return self.llm.complete(node.input["messages"]).text
        if node.kind == "tool" and self.tools is not None:
            return _call_tool(self.tools, node.name, node.input["args"])
        raise ValueError(f"override for node {self.index} ({node.kind}) has nothing to apply")


def _call_tool(tools: dict[str, Tool], name: str, args: dict[str, Any]) -> Any:
    if name not in tools:
        return f"ERROR: unknown tool {name!r}"
    try:
        return tools[name](**args)
    except Exception as e:  # noqa: BLE001 — tool failures are observations, not crashes
        return f"ERROR: {type(e).__name__}: {e}"


def _gen_kwargs(temperature: float | None) -> dict[str, Any]:
    """Sampling kwargs for live LLM calls. Real APIs default to temperature ~1, which
    makes a recorded hop irreproducible; ``None`` leaves the provider default."""
    return {} if temperature is None else {"temperature": temperature}


class _Runtime:
    """Executes nodes and records them; subclasses decide where outputs come from."""

    def __init__(self, llm: LLM, tools: dict[str, Tool], temperature: float | None = 0.0):
        self.llm_impl, self.tools = llm, tools
        self.gen_kwargs = _gen_kwargs(temperature)
        self.nodes: list[Node] = []

    def llm(self, messages: list[Message]) -> str:
        node = Node(len(self.nodes), "llm", self.llm_impl.model, {"messages": messages}, None)
        self._fill(node)
        self.nodes.append(node)
        return str(node.output)

    def tool(self, name: str, args: dict[str, Any]) -> Any:
        node = Node(len(self.nodes), "tool", name, {"args": args}, None)
        self._fill(node)
        self.nodes.append(node)
        return node.output

    def _fill(self, node: Node) -> None:
        self._live(node)

    def _live(self, node: Node, llm: LLM | None = None) -> None:
        node.source = "live"
        if node.kind == "llm":
            c = (llm or self.llm_impl).complete(node.input["messages"], **self.gen_kwargs)
            node.name = c.model
            node.output, node.tokens_in, node.tokens_out, node.cost_usd = (
                c.text, c.tokens_in, c.tokens_out, c.cost_usd)
        else:
            node.output = _call_tool(self.tools, node.name, node.input["args"])


def record(task: str, llm: LLM, tools: dict[str, Tool], *, task_id: str = "task",
           max_hops: int = 8, meta: dict[str, Any] | None = None,
           temperature: float | None = 0.0) -> Cassette:
    """Run the agent live and capture every hop into a :class:`Cassette`."""
    rt = _Runtime(llm, tools, temperature)
    answer = run_agent(task, rt, max_hops=max_hops)
    return Cassette(task_id, task, rt.nodes, answer, llm.model, dict(meta or {}))


class _ReplayRuntime(_Runtime):
    def __init__(self, cassette: Cassette, override: Override, llm: LLM,
                 tools: dict[str, Tool], after: AfterMode, temperature: float | None = 0.0):
        super().__init__(llm, tools, temperature)
        self.cassette, self.override, self.after = cassette, override, after
        self.cache: dict[str, deque[Node]] = defaultdict(deque)
        for n in cassette.nodes[override.index + 1:]:
            self.cache[n.key].append(n)

    def _fill(self, node: Node) -> None:
        i, at = node.index, self.override.index
        if i < at:
            rec = self.cassette.nodes[i]
            if rec.key != node.key:
                raise ReplayDivergence(
                    f"node {i}: agent requested {node.kind}:{node.name} with different input "
                    f"than the cassette ({rec.kind}:{rec.name}) — agent code or prompt changed"
                )
            self._serve(node, rec, "cassette")
        elif i == at:
            if node.kind != self.cassette.nodes[at].kind:
                raise ReplayDivergence(f"node {at} is {node.kind}, cassette has a different kind")
            ov = self.override
            if ov.output is None and node.kind == "llm" and ov.llm is not None:
                self._live(node, llm=ov.llm)  # keeps the swap model's tokens and cost
            else:
                node.output = ov.resolve(node)
            node.source = "override"
        elif self.after == "cached" and self.cache.get(node.key):
            self._serve(node, self.cache[node.key].popleft(), "cassette")
        else:
            self._live(node)

    @staticmethod
    def _serve(node: Node, rec: Node, source: str) -> None:
        node.name, node.output, node.source = rec.name, rec.output, source


@dataclass
class ReplayResult:
    original: Cassette
    replayed: Cassette
    override_index: int

    @property
    def live_calls(self) -> int:
        return sum(1 for n in self.replayed.nodes if n.source == "live")

    @property
    def cost_usd(self) -> float:
        """API spend of this replay (cassette-served nodes are free)."""
        return sum(n.cost_usd for n in self.replayed.nodes if n.source != "cassette")

    @property
    def first_divergence(self) -> int | None:
        """First node whose output differs from the original run."""
        a, b = self.original.nodes, self.replayed.nodes
        for i in range(max(len(a), len(b))):
            if i >= len(a) or i >= len(b) or a[i].key != b[i].key or a[i].output != b[i].output:
                return i
        return None


class Replayer:
    """Counterfactual replay of a cassette with one node swapped."""

    def __init__(self, llm: LLM, tools: dict[str, Tool], *, after: AfterMode = "cached",
                 max_hops: int = 8, temperature: float | None = 0.0):
        self.llm, self.tools, self.after, self.max_hops = llm, tools, after, max_hops
        self.temperature = temperature

    def replay(self, cassette: Cassette, override: Override) -> ReplayResult:
        if not 0 <= override.index < len(cassette.nodes):
            raise IndexError(f"node {override.index} out of range (0..{len(cassette.nodes) - 1})")
        rt = _ReplayRuntime(cassette, override, self.llm, self.tools, self.after,
                            self.temperature)
        answer = run_agent(cassette.task, rt, max_hops=self.max_hops)
        out = Cassette(cassette.task_id, cassette.task, rt.nodes, answer, cassette.model,
                       {**cassette.meta, "override_index": override.index})
        return ReplayResult(cassette, out, override.index)


@dataclass
class NodeTrial:
    index: int
    kind: str
    name: str
    status: str  # flipped | still_failing | unstable | no_alternative | identical | error
    alternative: Any = None
    replay_answer: str | None = None
    detail: str = ""


@dataclass
class BisectReport:
    task_id: str
    baseline_passed: bool
    trials: list[NodeTrial] = field(default_factory=list)
    cost_usd: float = 0.0  # reference-model calls + live replay hops

    @property
    def unstable(self) -> list[int]:
        return [t.index for t in self.trials if t.status == "unstable"]

    @property
    def culprits(self) -> list[int]:
        return [t.index for t in self.trials if t.status == "flipped"]

    @property
    def root_cause(self) -> int | None:
        """Earliest node whose single swap repairs the run."""
        return self.culprits[0] if self.culprits else None


def bisect(cassette: Cassette, checker: Checker, *, llm: LLM, tools: dict[str, Tool],
           ref_llm: LLM | None = None, oracle_tools: dict[str, Tool] | None = None,
           after: AfterMode = "cached", max_hops: int = 8,
           temperature: float | None = 0.0) -> BisectReport:
    """Swap each node with its known-good alternative and record which swaps flip FAIL→PASS.

    LLM nodes are regenerated by ``ref_llm`` on the exact recorded messages; tool
    nodes are re-executed against ``oracle_tools``. Nodes whose alternative equals
    the recorded output are skipped (``identical``) — they cannot be the cause.

    Real models are not bit-reproducible even at temperature 0. When the
    reference model *is* the model that recorded a hop and still answers
    differently, the difference is sampling noise, not a known-good fix: the node
    is replayed and reported as ``unstable`` but never blamed as a culprit. Use a
    different (stronger) reference model to get LLM-node verdicts.
    """
    report = BisectReport(cassette.task_id, checker(cassette.final_answer))
    if report.baseline_passed:
        return report
    replayer = Replayer(llm, tools, after=after, max_hops=max_hops, temperature=temperature)
    for node in cassette.nodes:
        trial = NodeTrial(node.index, node.kind, node.name, "no_alternative")
        if node.kind == "llm" and ref_llm is not None:
            c = ref_llm.complete(node.input["messages"], **_gen_kwargs(temperature))
            report.cost_usd += c.cost_usd
            alt = c.text
        elif node.kind == "tool" and oracle_tools is not None and node.name in oracle_tools:
            alt = _call_tool(oracle_tools, node.name, node.input["args"])
        else:
            report.trials.append(trial)
            continue
        trial.alternative = alt
        if alt == node.output:
            trial.status = "identical"
            report.trials.append(trial)
            continue
        try:
            res = replayer.replay(cassette, Override(node.index, output=alt))
        except ReplayDivergence as e:
            trial.status, trial.detail = "error", str(e)
        else:
            report.cost_usd += res.cost_usd
            trial.replay_answer = res.replayed.final_answer
            passed = checker(trial.replay_answer)
            trial.status = "flipped" if passed else "still_failing"
            trial.detail = f"{res.live_calls} live calls after swap"
            if node.kind == "llm" and ref_llm is not None and ref_llm.model == node.name:
                trial.status = "unstable"
                trial.detail = (f"same model re-sampled a different hop (nondeterministic); "
                                f"replay {'passes' if passed else 'still fails'}")
        report.trials.append(trial)
    return report


def format_nodes(c: Cassette, mark: int | None = None) -> str:
    lines = []
    for n in c.nodes:
        flag = ">>" if n.index == mark else "  "
        lines.append(f"{flag} {n.index:>2} {n.kind:<4} {n.source:<8} {n.name:<12} {n.brief()}")
    lines.append(f"   final answer: {c.final_answer}")
    return "\n".join(lines)


def format_bisect(r: BisectReport) -> str:
    if r.baseline_passed:
        return f"{r.task_id}: baseline already passes — nothing to bisect"
    lines = [f"{r.task_id}: baseline FAIL; swapping each node with its known-good alternative"]
    for t in r.trials:
        alt = "" if t.alternative is None else f" alt={str(t.alternative)[:48]!r}"
        ans = "" if t.replay_answer is None else f" -> answer {t.replay_answer}"
        lines.append(f"   node {t.index:>2} {t.kind:<4} {t.name:<12} {t.status:<14}{alt}{ans}")
    if r.root_cause is None:
        lines.append("   no single-node swap repairs this run (multiple faults or no oracle)")
    else:
        t = next(t for t in r.trials if t.index == r.root_cause)
        lines.append(f"   ROOT CAUSE: node {t.index} ({t.kind} {t.name})"
                     + (f"; also repairing: {r.culprits[1:]}" if len(r.culprits) > 1 else ""))
    if r.unstable:
        lines.append(f"   unstable (re-sampled differently by the same model, not blamed): "
                     f"{r.unstable}")
    return "\n".join(lines)
