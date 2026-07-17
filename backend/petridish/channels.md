# channels.py

## Purpose

Names the stable cell-state channel layout shared by simulation, task setup, and
snapshot serialization.

## Components

### `Channel`
- **Does**: Maps semantic state names to tensor indices.
- **Interacts with**: `state.py`, `simulation.py`, and `protocol.py`.
- **Rationale**: An enum prevents hidden numeric coupling as experimental
  channels evolve.

## Contracts

| Dependent | Expects | Breaking changes |
|-----------|---------|------------------|
| Simulation update | Values match the 16-channel architecture document | Reordering values |
| Browser protocol | Serialized channel order remains stable | Reordering or removal |
