# Lifecycle ablation benchmark

`benchmark_lifecycle_ablation` trains one deterministic fixed-three relational
organism to the same 1,200-update clone point, then deep-copies all model, optimizer,
task-generator, evaluation-generator, slow-statistic, and topology state.

The five branches isolate:

- weight-learning drift with no lesion or lifecycle;
- default lifecycle pressure without a lesion;
- static recovery after a stronger radius-8 physical lesion;
- default lifecycle after the same lesion;
- a predeclared gentler lifecycle after that lesion (32-update cadence, at most four
  births and eight deaths per generation).

All radius-8 branches must kill exactly the same cells or the benchmark fails. Atomic
artifacts use profile `lifecycle_ablation_r8`, carry explicit lifecycle configuration,
and report the same accuracy, topology, turnover, and death-cause checkpoints as the
first recovery experiment. The gentle branch is an a priori pressure reduction, not a
post-outcome fit.
