# 15 — Public Eval Methodology Teardown

> Senior evals engineers are hired for their methodology, not their dashboards.

Three deep-dives on exactly how evalkit grades, judges and does the statistics: the verbatim
rubrics, the judge prompts and the formulas. Each one includes the commands that reproduce its
numbers. Every number comes from the bundled synthetic datasets and seeded mock models.

1. [Grading the trajectory, not the answer](01-grading-rubrics.md): the trajectory DAG rubric,
   RAG answer/abstain/citation scoring, the CI gate policy, red-team oracles and replay bisection.
2. [Your LLM judge is an instrument; calibrate it](02-judge-prompts-and-calibration.md): every
   judge prompt verbatim, the anchor-set design, bias measurement and logistic recalibration.
3. [Stop reporting raw accuracy](03-statistical-framework.md): paired and clustered bootstrap,
   permutation and McNemar tests, power/MDE, drift statistics, Wilson intervals and Pareto
   frontiers.
