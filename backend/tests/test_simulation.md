# test_simulation.py

## Purpose

Protects core scientific invariants: bounded tensor shapes, finite updates,
deterministic replay, destructive lesion semantics, and reward-driven weight
change.

## Components

### `small_config`
- **Does**: Provides fast dimensions and cadence while retaining every mechanism.
- **Interacts with**: All simulation tests.

### State/determinism tests
- **Does**: Verify valid indices, finite tensors, and exact same-seed evolution.
- **Interacts with**: Simulation initialization, task RNG, and rewiring RNG.

### Intervention tests
- **Does**: Verify lesions cut incident edges and manual reward changes weights.
- **Interacts with**: `lesion`, `stimulate`, and `inject_reward`.

## Contracts

| Dependent | Expects | Breaking changes |
|-----------|---------|------------------|
| Simulator | Same-seed CPU runs stay reproducible and finite | RNG or update semantics |
| Lesion UI | Damage immediately zeros viability and gates | Lesion semantics |
