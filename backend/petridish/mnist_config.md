# mnist_config.py

## Purpose

Defines physical dimensions, persistent neuronal capacity, curriculum phases,
optimizer rates, and lifecycle thresholds of the MNIST organism.

## Components

### `MnistModelConfig`
- **Does**: Configures the 64×64 substrate, recurrent rule, dendrite and
  axon/candidate budgets, structural cadence, and homeostatic death/birth rules.
- **Interacts with**: `SpatialSubstrate`, `CellularGraphClassifier`, and
  `MnistExperiment`.
- **Rationale**: Scientific hyperparameters remain explicit and immutable so
  experimental runs can be reproduced and compared.
- **Rationale**: An 8-cell discovery radius makes broadcasts more local; twelve
  recurrent steps preserve an end-to-end route across the smaller field.
- **Rationale**: Edge and neuron grace periods are expressed in training trials;
  structural mutation evaluates those accumulated ages at generation boundaries.
- **Rationale**: Positive task utility provides a bounded energy bonus without
  being counted as neural stimulation or traffic.
- **Rationale**: Signed initialization scale, message gain, gradient clipping,
  and structural warm-up are explicit experimental controls.
- **Rationale**: Readout and shared-rule learning rates are independent because
  the initial fixed-reservoir probe must converge on a much smaller parameter set.
- **Rationale**: Early-output trajectory supervision is optional and disabled by
  default because it is invalid when a local graph has not yet reached the outputs.
- **Rationale**: Persistent genotype width and query/key attention temperature
  determine cell specialization and content-addressed communication capacity.
- **Rationale**: Readout, rule, synapse, and structural unlock controls separate
  moving optimization targets; structure requires competence or a measured plateau.

## Contracts

| Dependent | Expects | Breaking changes |
|-----------|---------|------------------|
| `mnist_substrate.py` | Spatial and lifecycle fields use field-cell units | Renaming thresholds |
| `mnist_model.py` | Hidden/genotype widths, message steps, and edge slots are fixed per model | Shape changes |
| Tests | Tiny configurations can reduce field and population cost | New hard-coded defaults |
| `mnist_hyperparameters.py` | Every numeric field has a viewer control spec | Adding an undocumented field |
| `mnist_experiment.py` | Phase and curriculum thresholds are ordered and reproducible | Unlock semantics |
