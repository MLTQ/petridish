# graph_layout.py

## Purpose

Defines immutable semantic-to-physical port assignments for MNIST, associative
memory, and autoregressive language experiments.

## Components

### `GraphLayout`
- **Does**: Names a task, declares port counts/boundaries, specifies directed
  flow, and maps semantic token/class IDs to physical port positions.
- **Rationale**: Layout is task definition rather than a model hyperparameter,
  keeping transfer comparisons reproducible.

### `LAYOUTS`
- **Does**: Registers MNIST, permuted-port associative recall, and reversed-flow
  tiny-language layouts.
- **Rationale**: The sequence tasks test memory/autoregression while also
  removing the ordered left-to-right geometry as a hidden shortcut.

### `resolve_layout`
- **Does**: Resolves a task key to one immutable layout and rejects unknown keys.

### `sequence_layout`
- **Does**: Validates registered synthetic layouts and constructs a deterministic,
  vocabulary-sized Tiny Shakespeare layout at runtime.
- **Rationale**: Corpus vocabulary size comes from cached text, so its physical
  ports cannot be hard-coded in the registry.

## Contracts

| Dependent | Expects | Breaking changes |
|-----------|---------|------------------|
| `mnist_substrate.py` | Direction is `-1` or `1`; mappings are full permutations | Layout validation |
| Models | Input/output tensors are indexed by semantic ID | Mapping orientation |
| Runtime/frontend | Layout keys are stable experiment identifiers | Key renames |
| Benchmarks | Permutations do not vary with training seed | Permutation seeds |
