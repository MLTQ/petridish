# Matched recovery benchmark

`benchmark_recovery` trains one deterministic separated-owner GRU on fixed
three-binding recall, evaluates competence, and deep-copies the complete experiment.
Model, optimizer, topology, slow statistics, task/evaluation generators, counters,
and RNG state are therefore identical before intervention.

Three branches are run sequentially:

- `control`: no lesion; gradient learning remains active, lifecycle/topology disabled.
- `lesion_static`: identical central lesion; gradient learning only.
- `lesion_lifecycle`: the same lesion; eight-update lifecycle and topology cycles.

Static branches use a recovery-external structural warm-up. This is necessary because
the cloned organism is already competent: a zero warm-up would immediately satisfy the
normal competence gate and silently re-enable topology mutation in the controls.

Every 20 recovery updates, each atomic artifact records held-out accuracy/loss by
query slot, distractor errors, living cells, edges, topology generation, cumulative
births/deaths, and death causes. The frontend compares the three artifacts as one
`matched_recovery` cohort using only these persisted measurements.

The benchmark is deterministic and fails if the cloned lesions remove different
cell counts. It does not reuse a previously selected result or tune a branch after
seeing recovery outcomes.
