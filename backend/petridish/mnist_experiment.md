# mnist_experiment.py

## Purpose

Runs one persistent organism through a balanced overfit curriculum, separating
readout, shared-rule, synapse, lifecycle, and structural learning while replaying
only measured forward, credit, and turnover phases.

## Components

### `MnistExperiment`
- **Does**: Owns data, shared-rule and synapse optimizers, persistent substrate,
  curriculum stage, phase gates, metrics, current digit, and trace playback.
- **Interacts with**: A `GraphLayout` that fixes semantic patch/class ports.
- **Rationale**: The output-bank parameter group uses its own learning rate so
  the initial separability probe is not limited by the slower recurrent-rule rate.
- **Rationale**: Optional trajectory loss is configuration-controlled; the
  default trains only the causally reachable final output state.

### `_train_trial`
- **Does**: Freezes topology, runs recurrent classification, backpropagates
  cross-entropy through the current graph, measures neuron/edge credit, updates
  weights and metabolism, then optionally runs one structural generation.
- **Rationale**: Discrete birth/death/growth never invalidates an active
  autograd graph.
- **Rationale**: Adam updates individual dendrites, and the reported mean
  update-to-weight ratio proves whether plasticity is numerically active.
- **Rationale**: Signed removal credit protects only neurons/edges whose removal
  would increase loss; cross-entropy improvement supplies continuous reward.
- **Rationale**: Metabolic pressure has its own warm-up; free-form topology
  plasticity still waits for accuracy competence or a measured plateau.
- **Rationale**: By default lifecycle begins after the fixed-reservoir readout
  probe, avoiding selection pressure on a representation that has not learned yet.
- **Rationale**: Death accounting distinguishes cumulative excitotoxic damage from
  transient stunned episodes; overload itself is no longer a death cause.
- **Rationale**: Gradients are still measured for scientific credit while only
  readout parameters, then the shared rule, then synapses are allowed to update.
- **Rationale**: The convex linear reservoir probe is not recurrent-gradient
  clipped; the shared recurrent rule and synapses retain configured clipping.

### `_optimization_phase` / `_should_activate_lifecycle` / `_should_unlock_structure`
- **Does**: Stages parameter updates, activates energy/turnover on a fixed
  schedule, and separately unlocks topology after competence or plateau.

### `_maybe_advance_curriculum`
- **Does**: Advances from a fixed balanced subset only after its rolling
  training accuracy meets the stage target.
- **Interacts with**: `build_curriculum` in `mnist_curriculum.py`.
- **Rationale**: The 20-example diagnostic stage is one complete batch so
  omitted samples cannot make the overfit measurement noisy or misleading.

### `step`
- **Does**: Advances input, forward, feedback, and structural evidence frames;
  one new optimizer update occurs only after a complete trace.

### `lifecycle_now`
- **Does**: Forces one genuine lifecycle plus topology cycle and replays the same digit.

### `_reset_mutation_optimizer_state`
- **Does**: Clears Adam moments for dendrite slots that change identity and
  genotype rows replaced through death/birth.

### `evaluate` / `lesion`
- **Does**: Evaluate without plasticity or remove occupied sites and incident
  dendrites before replaying the current digit.

## Contracts

| Dependent | Expects | Breaking changes |
|-----------|---------|------------------|
| Runtime | `step`, `lifecycle_now`, `evaluate`, and `lesion` remain synchronous | Method changes |
| Protocol | Current frame, persistent population metrics, and generation exist | Attribute changes |
| Tests | Synthetic datasets can avoid network downloads | Constructor injection changes |
| Protocol | Learning phase, curriculum size, and unlock reason are authoritative | Phase semantics |

## Notes

`tick` counts displayed phases. `training_step` counts gradient updates and
`substrate.generation` counts discrete structural cycles.
Manual lifecycle cycles remain explicit interventions and can bypass warm-up.
