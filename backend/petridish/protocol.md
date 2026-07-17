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

## Contracts

| Dependent | Expects | Breaking changes |
|-----------|---------|------------------|
| Frontend protocol | `experiment` selects a task union inside a common envelope | Key, type, or union changes |
| Renderer | Edge arrays have equal length and flattened cell indices | Edge array semantics |
| Inspector | Cell rows follow advertised channel order | Channel omission or reorder |

## Notes

JSON is sufficient for the 32×32 MVP. This module is the seam for replacing it
with MessagePack or a custom binary frame when larger fields justify it.
