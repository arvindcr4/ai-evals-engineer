"""Coverage report over a generated golden set.

Answers "what did we actually test?": counts per axis, category and field,
which expected categories have no case yet, and how many of the spec's
field-level pairs appear together in at least one case (pairwise coverage).
"""

from __future__ import annotations

import itertools
from collections import Counter

from evalkit.edge_case_gen.cases import AXIS_CATEGORIES, EdgeCase, canonical
from evalkit.edge_case_gen.generators import field_levels
from evalkit.edge_case_gen.spec import TaskSpec
from evalkit.edge_case_gen.validate import jaccard, shingles


def level_universe(spec: TaskSpec) -> dict[str, list[str]]:
    return {f.name: list(field_levels(f, spec.example_value(f.name, 0))) for f in spec.fields}


def pairwise_stats(spec: TaskSpec, cases: list[EdgeCase]) -> dict:
    """Pairwise coverage of the spec's field levels, overall and from each axis alone."""
    uni = level_universe(spec)
    names = list(uni)
    total = {
        ((a, la), (b, lb))
        for a, b in itertools.combinations(names, 2)
        for la in uni[a]
        for lb in uni[b]
    }

    def hit(cs: list[EdgeCase]) -> set:
        out = set()
        for c in cs:
            lv = {k: v for k, v in c.levels.items() if k in uni and v in uni[k]}
            for a, b in itertools.combinations(names, 2):
                if a in lv and b in lv:
                    out.add(((a, lv[a]), (b, lv[b])))
        return out

    covered = hit(cases) & total
    by_axis = {ax: round(len(hit([c for c in cases if c.axis == ax]) & total) / len(total), 4)
               for ax in sorted({c.axis for c in cases})} if total else {}
    missing = sorted(total - covered)[:10]
    return {
        "pairs_total": len(total),
        "pairs_covered": len(covered),
        "pairwise_coverage": round(len(covered) / len(total), 4) if total else 1.0,
        "pairwise_coverage_by_axis": by_axis,
        "missing_pairs_sample": [f"{a}={la} × {b}={lb}" for (a, la), (b, lb) in missing],
    }


def llm_vs_rules(cases: list[EdgeCase]) -> dict | None:
    """What the model-written cases add over the rule-based ones.

    ``novelty_vs_rules`` is 1 − the highest shingle-Jaccard similarity between an
    LLM case's whole input and any rule case (the same measure the validator
    uses against seeds); ``new_categories`` are axis/category pairs no rule hit.
    """
    gen = [(c, str(c.provenance.get("generator", ""))) for c in cases]
    llm = [c for c, g in gen if g.startswith("llm:")]
    rules = [c for c, g in gen if g.startswith("rule:")]
    if not llm:
        return None
    rule_sh = [shingles(canonical(c.input)) for c in rules]
    nov = sorted(1.0 - max((jaccard(shingles(canonical(c.input)), r) for r in rule_sh),
                           default=0.0) for c in llm)
    rule_cats = {(c.axis, c.category) for c in rules}
    new = sorted({f"{c.axis}/{c.category}" for c in llm} - {f"{a}/{k}" for a, k in rule_cats})
    seed_nov = [c.novelty for c in llm if c.novelty is not None]
    return {
        "n_llm": len(llm),
        "n_rule": len(rules),
        "novelty_vs_rules_mean": round(sum(nov) / len(nov), 4),
        "novelty_vs_rules_median": round(nov[len(nov) // 2], 4),
        "novelty_vs_rules_min": round(nov[0], 4),
        "novelty_vs_seeds_mean_llm": round(sum(seed_nov) / len(seed_nov), 4) if seed_nov else None,
        "novelty_vs_seeds_mean_rule": round(
            sum(c.novelty for c in rules if c.novelty is not None)
            / max(1, sum(c.novelty is not None for c in rules)), 4) if rules else None,
        "new_categories": new,
        "llm_cases_in_new_categories": sum(
            1 for c in llm if f"{c.axis}/{c.category}" in set(new)),
    }


def coverage_report(spec: TaskSpec, cases: list[EdgeCase]) -> dict:
    axis = Counter(c.axis for c in cases)
    cat = Counter(f"{c.axis}/{c.category}" for c in cases)
    fld = Counter(c.field for c in cases if c.field)
    gens = Counter(c.provenance.get("generator", "?") for c in cases)
    present = {(c.axis, c.category) for c in cases}
    missing_cats = {
        ax: [k for k in cats if (ax, k) not in present]
        for ax, cats in AXIS_CATEGORIES.items()
    }
    expected = sum(len(v) for v in AXIS_CATEGORIES.values())
    hit_cats = sum(len(v) - len(missing_cats[ax]) for ax, v in AXIS_CATEGORIES.items())
    return {
        "spec": spec.name,
        "n_cases": len(cases),
        "by_axis": dict(sorted(axis.items())),
        "by_category": dict(sorted(cat.items())),
        "by_field": dict(sorted(fld.items())),
        "by_generator": dict(sorted(gens.items())),
        "fields_untouched": [f.name for f in spec.fields if f.name not in fld],
        "category_coverage": round(hit_cats / expected, 4),
        "missing_categories": {k: v for k, v in missing_cats.items() if v},
        "schema_valid": sum(1 for c in cases if c.schema_valid),
        "schema_invalid": sum(1 for c in cases if c.schema_valid is False),
        "labels": dict(Counter(str(c.label) for c in cases)),
        "needs_human_review": sum(1 for c in cases if c.needs_human_review),
        "llm_vs_rules": llm_vs_rules(cases),
        **pairwise_stats(spec, cases),
    }


def format_report(rep: dict) -> str:
    lines = [f"coverage for {rep['spec']}: {rep['n_cases']} cases"]
    lines.append("  by axis:     " + ", ".join(f"{k}={v}" for k, v in rep["by_axis"].items()))
    lines.append("  by field:    " + ", ".join(f"{k}={v}" for k, v in rep["by_field"].items()))
    lines.append("  generators:  " + ", ".join(f"{k}={v}" for k, v in rep["by_generator"].items()))
    lines.append(f"  schema:      valid={rep['schema_valid']} invalid={rep['schema_invalid']}")
    lines.append("  labels:      " + ", ".join(f"{k}={v}" for k, v in rep["labels"].items()))
    lines.append(f"  review:      {rep['needs_human_review']} flagged needs_human_review")
    lines.append(f"  categories:  {rep['category_coverage']:.0%} of expected axis categories hit")
    for ax, miss in rep["missing_categories"].items():
        lines.append(f"    missing {ax}: {', '.join(miss)}")
    lines.append(
        f"  pairwise:    {rep['pairs_covered']}/{rep['pairs_total']} level pairs "
        f"= {rep['pairwise_coverage']:.1%}"
    )
    for ax, v in rep["pairwise_coverage_by_axis"].items():
        lines.append(f"    {ax:<14} alone: {v:.1%}")
    lv = rep.get("llm_vs_rules")
    if lv:
        lines.append(
            f"  llm vs rules: {lv['n_llm']} llm cases; novelty vs rule cases mean "
            f"{lv['novelty_vs_rules_mean']:.2f} (median {lv['novelty_vs_rules_median']:.2f}, "
            f"min {lv['novelty_vs_rules_min']:.2f}); novelty vs seeds llm "
            f"{lv['novelty_vs_seeds_mean_llm']} vs rule {lv['novelty_vs_seeds_mean_rule']}")
        lines.append(f"    {lv['llm_cases_in_new_categories']} llm cases in "
                     f"{len(lv['new_categories'])} categories no rule produced: "
                     + ", ".join(lv["new_categories"][:12])
                     + (" ..." if len(lv["new_categories"]) > 12 else ""))
    if rep["fields_untouched"]:
        lines.append(f"  untouched fields: {', '.join(rep['fields_untouched'])}")
    return "\n".join(lines)
