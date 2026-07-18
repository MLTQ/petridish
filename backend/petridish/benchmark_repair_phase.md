# Finite repair-phase benchmark

`benchmark_repair_phase` is an exploratory follow-up to the radius-8 lifecycle
matrix. It trains the same deterministic competent base once, deep-copies it, applies
the same radius-8 physical lesion, and enables the default eight-update lifecycle for
exactly 60, 100, or 140 recovery updates.

At the predeclared boundary, lifecycle and topology mutation are disabled while
gradient learning continues through recovery update 240. Cumulative population and
topology measurements remain intact. These windows were chosen after observing that
continuous lifecycle repaired function rapidly and then destabilized it, so this is
hypothesis-generating rather than an independent confirmatory result.

Every atomic `repair_phase_r8` artifact records its repair window and whether mutation
has frozen. Branches fail if the matched lesions remove different cells, and every
branch restores the same CPU/CUDA global RNG stream before stochastic repair begins.
`--freeze-after` may be repeated to run a declared subset when a focused replication
is more informative than recomputing the full exploratory window sweep.
