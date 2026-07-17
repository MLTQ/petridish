# config.py

## Purpose

Centralizes fixed dimensions and interpretable dynamics parameters. Device
selection lives here so simulation code does not contain platform branches.

## Components

### `SimulationConfig`
- **Does**: Defines field, graph, plasticity, pruning, and task cadence defaults.
- **Interacts with**: `PetriDishState`, `DelayedXorTask`, and `PetriDishSimulation`.
- **Rationale**: Shared parameters are centralized so a later outer optimizer can
  vary them without rewriting the tick function.

### `resolve_device`
- **Does**: Selects CUDA, then Apple MPS, then CPU unless explicitly overridden.
- **Interacts with**: `PetriDishSimulation.__init__`.

## Contracts

| Dependent | Expects | Breaking changes |
|-----------|---------|------------------|
| `state.py` | `cell_count`, `channels`, and `edge_slots` describe tensor shapes | Dimension semantics |
| `task.py` | Trial phase durations sum to `trial_ticks` | Task timing fields |
| `simulation.py` | Plasticity values are immutable during an episode | Field names and units |

## Notes

Learning and decay defaults are conservative enough for long live sessions;
manual reward should cause visible adaptation without driving every edge to the
weight clamp.
