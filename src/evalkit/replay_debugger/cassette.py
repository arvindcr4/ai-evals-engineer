"""Cassettes: the recorded node graph of one agent run.

Every LLM hop and every tool call is a :class:`Node` with its full input
(messages or arguments) and output. A cassette serialises to a core
:class:`~evalkit.core.trajectory.Trajectory` (``llm`` / ``tool_call`` /
``final`` steps), so recorded runs can be fed to any other evalkit system.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

from evalkit.core.llm import stable_hash
from evalkit.core.trajectory import Step, Trajectory, read_jsonl, write_jsonl

NodeKind = Literal["llm", "tool"]


@dataclass
class Node:
    index: int
    kind: NodeKind
    name: str
    input: dict[str, Any]
    output: Any
    source: str = "live"  # live | cassette | override
    tokens_in: int = 0
    tokens_out: int = 0
    cost_usd: float = 0.0

    @property
    def key(self) -> str:
        """Content address of the node's input — equal inputs replay equal outputs."""
        payload = json.dumps(self.input, sort_keys=True, default=str)
        return f"{self.kind}:{self.name if self.kind == 'tool' else ''}:{stable_hash(payload):016x}"

    def brief(self, width: int = 70) -> str:
        if self.kind == "llm":
            text = str(self.output)
        else:
            text = f"{self.name}({json.dumps(self.input.get('args', {}))}) -> {json.dumps(self.output)}"
        text = " ".join(text.split())
        return text if len(text) <= width else text[: width - 3] + "..."


@dataclass
class Cassette:
    task_id: str
    task: str
    nodes: list[Node] = field(default_factory=list)
    final_answer: str | None = None
    model: str = ""
    meta: dict[str, Any] = field(default_factory=dict)

    @property
    def expected(self) -> Any:
        return self.meta.get("expected")

    def to_trajectory(self) -> Trajectory:
        steps: list[Step] = []
        for n in self.nodes:
            if n.kind == "llm":
                steps.append(Step(
                    kind="llm", name=n.name, args=n.input, output=n.output,
                    tokens_in=n.tokens_in, tokens_out=n.tokens_out, cost_usd=n.cost_usd,
                    meta={"source": n.source},
                ))
            else:
                steps.append(Step(
                    kind="tool_call", name=n.name, args=n.input.get("args", {}),
                    output=n.output, meta={"source": n.source},
                ))
        steps.append(Step(kind="final", output=self.final_answer))
        return Trajectory(self.task_id, self.task, steps, self.final_answer, self.model,
                          dict(self.meta))

    @classmethod
    def from_trajectory(cls, t: Trajectory) -> Cassette:
        nodes: list[Node] = []
        for s in t.steps:
            if s.kind == "llm":
                nodes.append(Node(len(nodes), "llm", s.name, dict(s.args), s.output,
                                  s.meta.get("source", "live"), s.tokens_in, s.tokens_out,
                                  s.cost_usd))
            elif s.kind == "tool_call":
                nodes.append(Node(len(nodes), "tool", s.name, {"args": dict(s.args)}, s.output,
                                  s.meta.get("source", "live")))
        return cls(t.task_id, t.input, nodes, t.final_answer, t.model, dict(t.meta))


def save_cassettes(path: str | Path, cassettes: list[Cassette]) -> None:
    write_jsonl(path, [c.to_trajectory() for c in cassettes])


def load_cassettes(path: str | Path) -> list[Cassette]:
    return [Cassette.from_trajectory(Trajectory.from_dict(d)) for d in read_jsonl(path)]
