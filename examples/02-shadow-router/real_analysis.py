"""Recompute the doc's extra real-run numbers from the saved pair logs (no API calls).

    uv run --no-sync python examples/02-shadow-router/real_analysis.py

Reads real_output/pairs.jsonl (final run) and real_output/pairs.run1_prefix.jsonl
(the first run: same 47 prompts, same models, temperature 0) and prints the
same-model noise floor, cross-model similarity and 120-word-limit compliance.
"""

import json
import statistics as st
from pathlib import Path

from evalkit.shadow_router.report import similarity

OUT = Path(__file__).parent / "real_output"


def load(name: str) -> dict[str, dict]:
    rows = [json.loads(line) for line in (OUT / name).read_text().splitlines() if line]
    return {r["request_id"]: r for r in rows}


final, run1 = load("pairs.jsonl"), load("pairs.run1_prefix.jsonl")
common = sorted(set(final) & set(run1))
assert all(final[k]["messages"] == run1[k]["messages"] for k in common)
print(f"pairs: final {len(final)}, run1 {len(run1)}, common prompts {len(common)}")
for side in ("primary", "shadow"):
    model = final[common[0]][side]["model"]
    same = sum(final[k][side]["text"] == run1[k][side]["text"] for k in common)
    sims = [similarity(final[k][side]["text"], run1[k][side]["text"]) for k in common]
    print(
        f"{model}: byte-identical across runs {same}/{len(common)}; "
        f"same-model similarity {st.mean(sims):.3f}"
    )
cross = st.mean(similarity(p["primary"]["text"], p["shadow"]["text"]) for p in final.values())
print(f"cross-model similarity (final run): {cross:.3f}")
for side in ("primary", "shadow"):
    words = [len(p[side]["text"].split()) for p in final.values()]
    print(
        f"{final[common[0]][side]['model']}: over 120 words {sum(w > 120 for w in words)}"
        f"/{len(words)}; mean words {st.mean(words):.1f}"
    )
