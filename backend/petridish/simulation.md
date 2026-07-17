# simulation.py

## Purpose

Implements the coupled continuous cell field, sparse directed signaling,
three-factor synaptic plasticity, structural rewiring, and interventions.

## Components

### `PetriDishSimulation`
- **Does**: Owns deterministic generators, task, tensors, metrics, and events.
- **Interacts with**: `SimulationConfig`, `PetriDishState`, `DelayedXorTask`.

### `step`
- **Does**: Advances local perception, graph signaling, cell dynamics, plasticity,
  and scheduled rewiring without constructing autograd graphs.
- **Interacts with**: Called by `ExperimentRuntime` and tests.
- **Rationale**: Physics ticks are independent of snapshot/render cadence.
- **Rationale**: Reward-trace self-retention plus diffusion is contractive, so a
  reward pulse decays instead of becoming a permanent global learning signal.

### `_rewire`
- **Does**: Prunes dead or low-utility old edges and probabilistically assigns
  empty slots to locally compatible or exploratory remote targets.
- **Interacts with**: Edge state and the viewer's structural event stream.
- **Rationale**: Mutation runs less frequently than differentiable-shaped tensor
  updates, preserving a stable hot loop.

### `lesion`
- **Does**: Zeros cells inside a circular brush and cuts incident connections.
- **Interacts with**: WebSocket lesion commands.

### `stimulate` / `inject_reward`
- **Does**: Add controlled, short-lived experimental interventions.
- **Interacts with**: Viewer controls and the normal task update path.

## Contracts

| Dependent | Expects | Breaking changes |
|-----------|---------|------------------|
| `protocol.py` | `state`, `task`, metrics, events, and last observation remain readable | Attribute changes |
| `runtime.py` | `step`, `lesion`, `stimulate`, and `inject_reward` are synchronous | Method signatures |
| Tests | Same seed/device yields the same initialized tensors and tick sequence | RNG order or reset semantics |

## Notes

The local rule is hand-designed and intentionally interpretable. A later outer
optimizer should operate on centralized config/rule parameters rather than
silently changing state semantics.
