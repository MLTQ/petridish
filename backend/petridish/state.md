# state.py

## Purpose

Defines the complete mutable tensor state of one experiment without embedding
update logic or task policy.

## Components

### `PetriDishState`
- **Does**: Groups cell state and fixed-capacity directed-edge slot tensors.
- **Interacts with**: Constructed and mutated by `PetriDishSimulation`.

### `PetriDishState.clone`
- **Does**: Produces a detached deep copy for tests and future checkpoints.
- **Interacts with**: Determinism and intervention tests.

## Contracts

| Dependent | Expects | Breaking changes |
|-----------|---------|------------------|
| `simulation.py` | Cell shape `[N,C]`; edge tensors shape `[N,K]` | Shape or dtype changes |
| `protocol.py` | Destinations are flattened cell indices and gates are numeric | Index or gate semantics |
