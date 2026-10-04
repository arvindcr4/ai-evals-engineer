"""Synthetic edge-case generator (system 12).

Spec → rule-based + LLM generation across boundary, format, semantic,
adversarial and pairwise-combinatorial axes → validation (schema, dedupe,
novelty, labels with a human-review flag) → coverage report → probe a system.
"""

from evalkit.edge_case_gen.cases import AXES, AXIS_CATEGORIES, EdgeCase
from evalkit.edge_case_gen.coverage import coverage_report
from evalkit.edge_case_gen.generators import RuleGenerator, field_levels
from evalkit.edge_case_gen.llm_gen import LLMGenerator, mock_responder
from evalkit.edge_case_gen.pairwise import covering_array, pairwise_coverage
from evalkit.edge_case_gen.probe import probe, summarize
from evalkit.edge_case_gen.spec import TaskSpec, load_spec, spec_from_dict, validate_input
from evalkit.edge_case_gen.validate import Validator, propose_labels

__all__ = [
    "AXES", "AXIS_CATEGORIES", "EdgeCase", "LLMGenerator", "RuleGenerator", "TaskSpec",
    "Validator", "coverage_report", "covering_array", "field_levels", "load_spec",
    "mock_responder", "pairwise_coverage", "probe", "propose_labels", "spec_from_dict",
    "summarize", "validate_input",
]
