"""Human-labelled anchors: the ruler a judge is measured against.

An anchor is one pairwise comparison with a human verdict::

    {id, prompt, response_a, response_b, model_a, model_b,
     human_pref: "a" | "b" | "tie", human_score_a?, human_score_b?}

:func:`make_anchors` synthesises a realistic, seeded set where the things a
judge should *not* care about are controlled independently of quality:

* quality = how many substantive facts a response covers (what humans reward);
* length = a sentence budget drawn independently of quality, topped up with
  generic filler of the same average word count as a fact;
* model family = a stylistic opener ("Certainly!", "Quick answer.", ...), which is
  how real judges recognise their own family's outputs.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np

from evalkit.core.trajectory import read_jsonl, write_jsonl

FAMILY_STYLE = {
    "atlas": "Quick answer.",
    "nova": "Certainly! Great question.",
    "orion": "Here's a breakdown.",
}
MODELS = ["atlas-7b", "atlas-70b", "nova-mini", "nova-pro", "orion-1"]

TOPICS: dict[str, list[str]] = {
    "How does HTTPS keep traffic private?": [
        "TLS negotiates a shared session key using asymmetric cryptography.",
        "The server proves its identity with a certificate signed by a trusted CA.",
        "Bulk data is encrypted with a fast symmetric cipher such as AES-GCM.",
        "Message authentication codes detect tampering in transit.",
        "Forward secrecy means a stolen server key cannot decrypt past sessions.",
        "The handshake happens before any HTTP bytes are sent.",
    ],
    "Why do we use a validation set when training models?": [
        "It estimates performance on data the model was not fitted to.",
        "Hyperparameters are tuned against it instead of the test set.",
        "Early stopping watches validation loss to avoid overfitting.",
        "Reusing the test set for tuning leaks information and inflates scores.",
        "A gap between training and validation loss signals overfitting.",
        "Cross-validation rotates the validation fold when data is scarce.",
    ],
    "What causes inflation?": [
        "Demand growing faster than supply pushes prices up.",
        "Rising input costs such as energy get passed on to consumers.",
        "Rapid money-supply growth can reduce the value of each unit of currency.",
        "Expectations of future inflation feed into wage and price setting.",
        "Central banks raise interest rates to cool demand.",
        "Supply shocks can raise prices even when demand is flat.",
    ],
    "How do vaccines train the immune system?": [
        "They expose the body to a harmless antigen from the pathogen.",
        "B cells learn to produce antibodies that bind that antigen.",
        "Memory cells persist and respond faster on later exposure.",
        "T cells help coordinate the response and kill infected cells.",
        "Boosters raise antibody levels that wane over time.",
        "mRNA vaccines instruct cells to make the antigen temporarily.",
    ],
    "What is a database index?": [
        "An index is a separate structure that maps key values to row locations.",
        "B-tree indexes keep keys sorted so range scans are efficient.",
        "Lookups drop from a full table scan to logarithmic time.",
        "Every insert or update must also maintain the index, slowing writes.",
        "Composite indexes serve queries that filter on their leading columns.",
        "Indexes consume extra disk space and memory.",
    ],
    "Why is the sky blue?": [
        "Sunlight contains all visible wavelengths.",
        "Air molecules scatter short wavelengths far more than long ones.",
        "This Rayleigh scattering scales with the inverse fourth power of wavelength.",
        "Scattered blue light reaches our eyes from every direction of the sky.",
        "Sunsets look red because light crosses more atmosphere and loses its blue.",
        "Our eyes are less sensitive to violet, so the sky looks blue rather than violet.",
    ],
    "How should I store user passwords?": [
        "Never store passwords in plain text or with reversible encryption.",
        "Use a slow, salted hash such as Argon2id, scrypt or bcrypt.",
        "A unique per-user salt defeats precomputed rainbow tables.",
        "Tune the work factor so hashing takes tens of milliseconds.",
        "Rehash with stronger parameters when users next log in.",
        "Rate-limit login attempts to slow online guessing.",
    ],
    "What does a load balancer do?": [
        "It spreads incoming requests across multiple backend servers.",
        "Health checks remove failing instances from rotation.",
        "Algorithms include round robin, least connections and consistent hashing.",
        "It lets you scale horizontally by adding servers behind one address.",
        "Layer 7 balancers can route by URL path or header.",
        "Sticky sessions pin a client to one backend when state is local.",
    ],
    "How do noise-cancelling headphones work?": [
        "Microphones sample ambient sound around the ear cup.",
        "A processor generates an inverted waveform of that noise.",
        "The anti-noise destructively interferes with the incoming sound.",
        "It works best on steady low-frequency noise like engine hum.",
        "Passive isolation from the ear cups handles higher frequencies.",
        "Latency must be tiny or the cancellation falls out of phase.",
    ],
    "Why do we sleep?": [
        "Sleep consolidates memories formed during the day.",
        "The glymphatic system clears metabolic waste from the brain during sleep.",
        "Growth hormone release peaks in deep sleep.",
        "Sleep deprivation impairs attention and decision making.",
        "REM sleep is linked to emotional processing.",
        "Circadian rhythms and sleep pressure together time when we feel sleepy.",
    ],
}

FILLER = [
    "This is an important topic that many people find interesting.",
    "There are many factors to consider here.",
    "It is worth keeping in mind that every situation is different.",
    "Overall, it is a nuanced subject with several perspectives.",
    "Understanding this can be very helpful in everyday life.",
    "Experts have discussed this question for a long time.",
    "Of course, there is always more to learn about it.",
    "I hope this explanation gives you a useful overview.",
    "In summary, the details matter quite a lot.",
    "Let me know if you would like me to expand on any part.",
    "Many people ask this question at some point.",
    "The answer depends on a number of different things.",
    "It helps to think about the bigger picture here.",
    "This comes up surprisingly often in practice.",
    "Different sources may describe it in slightly different ways.",
    "There is a lot of interesting history behind this.",
    "Hopefully this makes the overall idea a bit clearer.",
    "It is a good question to be curious about.",
    "Keep in mind that this is a general explanation.",
    "Everyone's experience with this can vary somewhat.",
]

FACT_SET = frozenset(f for facts in TOPICS.values() for f in facts)


def family_of(model: str) -> str:
    return model.split("-", 1)[0].lower()


@dataclass
class Anchor:
    id: str
    prompt: str
    response_a: str
    response_b: str
    model_a: str
    model_b: str
    human_pref: str
    human_score_a: float | None = None
    human_score_b: float | None = None
    meta: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> Anchor:
        known = cls.__dataclass_fields__
        return cls(**{k: v for k, v in d.items() if k in known})

    @property
    def log_len_ratio(self) -> float:
        """log(words in A / words in B); positive when A is longer."""
        return math.log(max(len(self.response_a.split()), 1) / max(len(self.response_b.split()), 1))


def _response(rng: np.random.Generator, model: str, facts: list[str], k: int,
              n_filler: int) -> str:
    chosen = list(rng.choice(facts, size=k, replace=False)) if k else []
    filler = list(rng.choice(FILLER, size=n_filler, replace=False)) if n_filler else []
    body = chosen + filler
    rng.shuffle(body)
    if not body:
        body = ["I'm not sure."]
    return " ".join([FAMILY_STYLE[family_of(model)]] + body)


def _human_score(rng: np.random.Generator, k: int, n_facts: int) -> float:
    return float(np.clip(round(1 + 9 * k / n_facts + rng.normal(0, 0.8)), 1, 10))


def make_anchors(n: int = 500, seed: int = 0, human_noise: float = 0.6,
                 tie_margin: float = 0.5, min_sentences: int = 6,
                 max_sentences: int = 18) -> list[Anchor]:
    """Synthesise ``n`` anchors with known quality, independent length and model identity."""
    if min_sentences < max(len(f) for f in TOPICS.values()) or max_sentences > min_sentences + len(
            FILLER):
        raise ValueError("sentence budget must cover every fact and fit the filler bank")
    rng = np.random.default_rng(seed)
    topics = list(TOPICS)
    anchors: list[Anchor] = []
    for i in range(n):
        q = topics[int(rng.integers(len(topics)))]
        facts = TOPICS[q]
        ma, mb = (str(m) for m in rng.choice(MODELS, size=2, replace=False))
        ka, kb = (int(x) for x in rng.integers(0, len(facts) + 1, size=2))
        ta, tb = (int(x) for x in rng.integers(min_sentences, max_sentences + 1, size=2))
        fa, fb = max(ta - ka, 0), max(tb - kb, 0)
        ua = ka + rng.normal(0, human_noise)
        ub = kb + rng.normal(0, human_noise)
        pref = "tie" if abs(ua - ub) < tie_margin else ("a" if ua > ub else "b")

        anchors.append(Anchor(
            id=f"anc-{i:04d}", prompt=q,
            response_a=_response(rng, ma, facts, ka, fa),
            response_b=_response(rng, mb, facts, kb, fb),
            model_a=ma, model_b=mb, human_pref=pref,
            human_score_a=_human_score(rng, ka, len(facts)), human_score_b=_human_score(rng, kb, len(facts)),
            meta={"facts_a": ka, "facts_b": kb, "filler_a": fa, "filler_b": fb},
        ))
    return anchors


def save_anchors(path: str, anchors: list[Anchor]) -> None:
    write_jsonl(path, anchors)


def load_anchors(path: str) -> list[Anchor]:
    anchors = [Anchor.from_dict(r) for r in read_jsonl(path)]
    for a in anchors:
        if a.human_pref not in ("a", "b", "tie"):
            raise ValueError(f"{a.id}: human_pref must be a/b/tie, got {a.human_pref!r}")
    return anchors
