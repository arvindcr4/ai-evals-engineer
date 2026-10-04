"""Coverage report over a generated golden set.

Answers "what did we actually test?": counts per axis, category and field,
which expected categories have no case yet, and how many of the spec's
field-level pairs appear together in at least one case (pairwise coverage).
"""

from __future__ import annotations

import itertools
from collections import Counter

from evalkit.edge_case_gen.cases import AXIS_CATEGORIES, EdgeCase
from evalkit.edge_case_gen.generators import field_levels
from evalkit.edge_case_gen.spec import TaskSpec


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
    if rep["fields_untouched"]:
        lines.append(f"  untouched fields: {', '.join(rep['fields_untouched'])}")
    return "\n".join(lines)
