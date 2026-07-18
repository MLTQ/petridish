# mnist_config.py

## Purpose

Defines physical dimensions, persistent neuronal capacity, curriculum phases,
optimizer rates, and independently gated lifecycle/topology thresholds.

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
- **Rationale**: Broadcast workspace slots, gain, and within-sequence decay
  configure the optional low-rank advertisement memory; zero gain is its ablation.
- **Rationale**: Fast-weight gain and decay configure a recurrent linear-attention
  matrix written and queried by sequence neurons; zero gain is its ablation.
- **Rationale**: Neuron-owned binding gain and address temperature configure an
  optional episodic relation memory whose storage locations are living physical
  cells; zero gain omits its parameters and preserves baseline checkpoints.
- **Rationale**: The token-value switch separates clean successor-symbol storage
  from the default mixed successor-neuron state as an explicit controlled ablation.
- **Rationale**: Optional address-separation regularization penalizes overlapping,
  diffuse token ownership; zero leaves baseline task optimization unchanged.
- **Rationale**: Readout and shared-rule learning rates are independent because
  the initial fixed-reservoir probe must converge on a much smaller parameter set.
- **Rationale**: Early-output trajectory supervision is optional and disabled by
  default because it is invalid when a local graph has not yet reached the outputs.
- **Rationale**: Persistent genotype width and query/key attention temperature
  determine cell specialization and content-addressed communication capacity.
- **Rationale**: Readout, rule, synapse, and structural unlock controls separate
  moving optimization targets; structure requires competence or a measured plateau.
- **Rationale**: Lifecycle enable/warm-up/cadence controls activate metabolic
  pressure independently from learned topology growth.
- **Rationale**: The default lifecycle warm-up ends with the readout-only probe,
  so metabolism begins when the shared cellular rule starts adapting.
- **Rationale**: Newborn energy and genotype mutation noise make replacement
  and inherited specialization explicit, reproducible variables.
- **Rationale**: Replacement-only births optionally cap neurogenesis to deaths in
  the same lifecycle cycle, preventing a healthy population from inflating merely
  because unused tensor sites remain.
- **Rationale**: Overload first produces a reversible stunned state. Recovery
  probability, refractory generations, damage accumulation/repair, and the
  repeated-episode death threshold are independent biological controls.
- **Rationale**: A local-density ceiling prevents birth from simply filling an
  already dense field before death or lesioning creates genuine vacancies.
- **Rationale**: `max_initial_neurons` caps initial occupancy independently from
  field extent, allowing 128–1024-cell fields without automatically creating a
  dense million-site organism.
- **Rationale**: `cell_architecture` records the shared sequence-cell family in
  checkpoints and manifests; it is categorical and therefore not a numeric slider.

## Contracts

| Dependent | Expects | Breaking changes |
|-----------|---------|------------------|
| `mnist_substrate.py` | Spatial and lifecycle fields use field-cell units | Renaming thresholds |
| `mnist_model.py` | Hidden/genotype widths, message steps, and edge slots are fixed per model | Shape changes |
| Tests | Tiny configurations can reduce field and population cost | New hard-coded defaults |
| `mnist_hyperparameters.py` | Every numeric field has a viewer control spec; categorical fields are excluded | Adding an undocumented numeric field |
| `mnist_experiment.py` | Phase and curriculum thresholds are ordered and reproducible | Unlock semantics |
