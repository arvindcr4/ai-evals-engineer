"""Scenario generator: needle facts, superseding updates and noise at controlled depths.

A scenario is a long conversation. A few *slots* (``home address``, ``manager``,
...) get an initial value planted at a chosen depth; some slots are later
*updated* to a new value, which supersedes the old one. Two slots live in the
pinned system prompt. Everything else is noise turns — optionally "hard" noise
that reuses slot vocabulary and colleague facts that share the fact template
but not the subject. At the end, one probe per slot asks for its current value.
"""

from __future__ import annotations

import json
import random
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Literal

Role = Literal["system", "user", "assistant"]

SLOT_VALUES: dict[str, list[str]] = {
    "home address": ["14 Elm Street", "221 Baker Road", "9 Lake View Lane", "77 Hill Crescent",
                     "3 Orchard Close", "58 Harbour Walk"],
    "manager": ["Priya Nair", "Tomas Weber", "Aiko Tanaka", "Grace Okafor", "Luis Romero"],
    "project deadline": ["March 3", "April 18", "June 30", "September 9", "November 21"],
    "favorite airline": ["Lufthansa", "Qantas", "Air India", "KLM", "Emirates"],
    "wifi password": ["tangerine-42", "violet-mango-7", "quartz-river-19", "amber-sky-88"],
    "monthly budget": ["4200 dollars", "3100 dollars", "5600 dollars", "2750 dollars"],
    "car": ["a blue Corolla", "a grey Model 3", "a red Swift", "a white Golf"],
    "dentist": ["Dr Mehta", "Dr Lindqvist", "Dr Osei", "Dr Fontaine"],
}
PINNED_VALUES: dict[str, list[str]] = {
    "timezone": ["IST", "CET", "PST", "JST"],
    "name": ["Arvind", "Maya", "Jonas", "Leila"],
}
TOPICS = ["sourdough starters", "the history of jazz", "mechanical keyboards", "tide pools",
          "Roman aqueducts", "marathon training", "houseplant lighting", "chess openings",
          "volcanic soil", "film photography", "urban beekeeping", "medieval maps",
          "espresso extraction", "bird migration", "solar panels", "origami cranes"]
ADJ = ["surprisingly subtle", "well documented", "hotly debated", "easy to overlook",
       "rewarding to learn", "full of trade-offs", "older than people think"]
FACT_TEMPLATES = ["Quick note for later: my {attr} is {value}.",
                  "Please remember that my {attr} is {value}."]
UPDATE_TEMPLATES = ["Update: my {attr} has changed to {value}.",
                    "Correction, my {attr} is now {value}."]
COLLEAGUE_TEMPLATE = "Unrelated, but my colleague's {attr} is {value}."
HARD_NOISE = ["I keep forgetting the paperwork about the {attr}, remind me to sort it later.",
              "My sister asked how people usually pick a {attr} and I had no answer."]


@dataclass
class Turn:
    role: Role
    text: str
    index: int = 0
    slot: str | None = None
    kind: str = "noise"  # noise | fact | update | distractor | system


@dataclass
class Probe:
    slot: str
    question: str
    current: str
    stale: list[str] = field(default_factory=list)
    pinned: bool = False
    updated: bool = False


@dataclass
class Scenario:
    system: str
    turns: list[Turn]
    probes: list[Probe]
    seed: int
    noise: int

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(asdict(self), indent=1))

    @classmethod
    def load(cls, path: str | Path) -> Scenario:
        d = json.loads(Path(path).read_text())
        return cls(d["system"], [Turn(**t) for t in d["turns"]],
                   [Probe(**p) for p in d["probes"]], d["seed"], d["noise"])


def _noise_turn(rng: random.Random, role: Role) -> str:
    a, b, c = rng.sample(TOPICS, 3)
    if role == "user":
        return rng.choice([
            (f"Can you tell me more about {a}? I was reading about {b} and {c} yesterday "
            f"and got curious how they connect."),
            f"Random question: what makes {a} {rng.choice(ADJ)}? A friend compared it to {b}.",
            f"I spent the weekend on {a} and {b}; any tips before I dive into {c} next?",
        ])
    return rng.choice([
        (f"Sure. {a.capitalize()} is {rng.choice(ADJ)}, and it shares some ideas with {b}. "
        f"Start small, keep notes, and revisit {c} later."),
        (f"Good question. Most guides on {a} skip the basics of {b}, which is "
        f"{rng.choice(ADJ)}. I would compare a few sources first."),
        (f"People often pair {a} with {c}. The key point is that it is {rng.choice(ADJ)}, "
        f"so take your time with it."),
    ])


def generate(noise: int, *, n_facts: int = 6, n_updates: int = 3, pinned_updates: int = 1,
             seed: int = 0, hard_noise: float = 0.1, fact_span: tuple[float, float] = (0.0, 0.6),
             update_span: tuple[float, float] = (0.6, 0.95)) -> Scenario:
    """Build a scenario with ``noise`` noise turns.

    Initial facts are spaced evenly across ``fact_span`` (fractions of the noise
    stream), updates across ``update_span``; ``hard_noise`` is the fraction of
    noise turns replaced by slot-vocabulary distractors. ``pinned_updates``
    system-prompt facts are later changed in conversation, so a strategy that
    pins the system prompt but evicts the update answers with the stale value.
    """
    if n_facts > len(SLOT_VALUES):
        raise ValueError(f"at most {len(SLOT_VALUES)} facts")
    rng = random.Random(seed)
    slots = rng.sample(sorted(SLOT_VALUES), n_facts)
    updated = slots[:min(n_updates, n_facts)]
    values = {s: rng.sample(SLOT_VALUES[s], 2) for s in slots}
    pinned_hist = {k: rng.sample(v, 2) for k, v in PINNED_VALUES.items()}
    pinned = {k: v[0] for k, v in pinned_hist.items()}
    changed_pinned = list(pinned)[:pinned_updates]
    system = ("You are a helpful personal assistant. User profile: "
              + " ".join(f"the user's {k} is {v}." for k, v in pinned.items()))

    stream: list[Turn] = []
    for i in range(noise):
        role: Role = "user" if i % 2 == 0 else "assistant"
        if role == "user" and rng.random() < hard_noise:
            attr = rng.choice(slots)
            if rng.random() < 0.5:
                other = rng.choice([v for v in SLOT_VALUES[attr] if v not in values[attr]]
                                   or SLOT_VALUES[attr])
                stream.append(Turn(role, COLLEAGUE_TEMPLATE.format(attr=attr, value=other),
                                   slot=attr, kind="distractor"))
            else:
                stream.append(Turn(role, rng.choice(HARD_NOISE).format(attr=attr),
                                   slot=attr, kind="distractor"))
        else:
            stream.append(Turn(role, _noise_turn(rng, role)))

    def spread(k: int, span: tuple[float, float]) -> list[int]:
        lo, hi = span
        if k == 1:
            return [round(lo * noise)]
        return [round((lo + (hi - lo) * j / (k - 1)) * noise) for j in range(k)]

    inserts: list[tuple[int, int, Turn]] = []
    for j, (pos, s) in enumerate(zip(spread(len(slots), fact_span), slots)):
        text = rng.choice(FACT_TEMPLATES).format(attr=s, value=values[s][0])
        inserts.append((pos, j, Turn("user", text, slot=s, kind="fact")))
    all_updates = [(s, values[s][1]) for s in updated]
    all_updates += [(k, pinned_hist[k][1]) for k in changed_pinned]
    for j, (pos, (s, new)) in enumerate(zip(spread(len(all_updates), update_span), all_updates)):
        text = rng.choice(UPDATE_TEMPLATES).format(attr=s, value=new)
        inserts.append((pos, 100 + j, Turn("user", text, slot=s, kind="update")))
    for pos, _, turn in sorted(inserts, key=lambda x: (x[0], x[1]), reverse=True):
        stream.insert(pos, turn)
    for i, t in enumerate(stream):
        t.index = i

    probes = []
    for k, v in pinned.items():
        if k in changed_pinned:
            probes.append(Probe(k, f"What is the user's {k}?", pinned_hist[k][1], [v],
                                pinned=True, updated=True))
        else:
            probes.append(Probe(k, f"What is the user's {k}?", v, pinned=True))
    for s in slots:
        if s in updated:
            probes.append(Probe(s, f"What is the user's {s}?", values[s][1], [values[s][0]],
                                updated=True))
        else:
            probes.append(Probe(s, f"What is the user's {s}?", values[s][0]))
    return Scenario(system, stream, probes, seed, noise)
