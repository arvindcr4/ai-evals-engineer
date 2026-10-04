"""`evalkit <system> ...` — each system module exposes ``register(subparsers)``."""

from __future__ import annotations

import argparse
import importlib
import sys

SYSTEMS = [
    "trajectory_grader",      # 01
    "shadow_router",          # 02
    "calibrated_judge",       # 03
    "regression_gate",        # 04
    "rag_adversarial",        # 05
    "dpo_flywheel",           # 06
    "significance",           # 07
    "redteam_fuzzer",         # 08
    "drift_monitor",          # 09
    "pareto",                 # 10
    "replay_debugger",        # 11
    "edge_case_gen",          # 12
    "context_eviction",       # 13
    "contamination",          # 14
]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="evalkit", description=__doc__)
    sub = parser.add_subparsers(dest="system", required=True)
    for name in SYSTEMS:
        try:
            mod = importlib.import_module(f"evalkit.{name}.cli")
        except ModuleNotFoundError:
            continue
        mod.register(sub)
    args = parser.parse_args(argv)
    return int(args.func(args) or 0)


if __name__ == "__main__":
    sys.exit(main())
