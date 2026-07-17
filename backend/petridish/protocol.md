# protocol.py

## Purpose

Routes active experiment state into the stable JSON snapshot contract consumed
by the browser.

## Components

### `CHANNEL_NAMES`
- **Does**: Publishes cell channel order alongside every field.
- **Interacts with**: Derived from `Channel`.

### `build_snapshot`
- **Does**: Dispatches MNIST experiments to their projector or copies XOR cells,
  edges, events, task status, and metrics to CPU-backed JSON-safe values.
- **Interacts with**: `ExperimentRuntime.broadcast` and frontend `protocol.ts`.
- **Rationale**: Snapshotting is the only deliberate accelerator-to-CPU boundary.
- **Does**: Leaves task-specific configuration payloads intact when dispatching
  to the MNIST projector.

## Contracts

| Dependent | Expects | Breaking changes |
|-----------|---------|------------------|
| Frontend protocol | `experiment` selects a task union inside a common envelope | Key, type, or union changes |
| Renderer | Edge arrays include measured `flow`/`credit`; sparse fields publish
  `indices` mapping rows to flattened site IDs | Edge or field semantics |
| Inspector | Cell rows follow advertised channel order | Channel omission or reorder |
| Common metrics | XOR publishes null synapse movement and unlocked structure | Removing shared keys |

## Notes

MNIST transmits only occupied rows and a bounded, importance-ranked subset of
real edges. This module remains the seam for a later binary transport.
