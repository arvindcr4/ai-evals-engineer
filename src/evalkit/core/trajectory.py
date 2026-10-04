"""Agent trajectory data model shared by every evaluator.

A trajectory is the ordered list of steps an agent took for one task: model
messages, tool calls with arguments, and tool outputs. Most systems in evalkit
read and write trajectories as JSONL, one trajectory per line.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable, Literal

StepKind = Literal["llm", "tool_call", "tool_result", "final"]


@dataclass
class Step:
    kind: StepKind
    name: str = ""  # tool name for tool_call/tool_result, model for llm
    args: dict[str, Any] = field(default_factory=dict)
    output: Any = None
    tokens_in: int = 0
    tokens_out: int = 0
    latency_s: float = 0.0
    cost_usd: float = 0.0
    meta: dict[str, Any] = field(default_factory=dict)


@dataclass
class Trajectory:
    task_id: str
    input: str
    steps: list[Step] = field(default_factory=list)
    final_answer: str | None = None
    model: str = ""
    meta: dict[str, Any] = field(default_factory=dict)

    @property
    def tool_calls(self) -> list[Step]:
        return [s for s in self.steps if s.kind == "tool_call"]

    @property
    def cost_usd(self) -> float:
        return sum(s.cost_usd for s in self.steps)

    @property
    def latency_s(self) -> float:
        return sum(s.latency_s for s in self.steps)

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "Trajectory":
        d = dict(d)
        d["steps"] = [Step(**s) for s in d.get("steps", [])]
        return cls(**d)


def write_jsonl(path: str | Path, rows: Iterable[Any]) -> None:
    with open(path, "w") as f:
        for r in rows:
            if hasattr(r, "to_dict"):
                r = r.to_dict()
            f.write(json.dumps(r, default=str) + "\n")


def read_jsonl(path: str | Path) -> list[dict]:
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]
