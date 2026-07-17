# renderer.ts

## Purpose

Renders the live field, weighted directed graph, activity pulses, structural
events, task regions, and selection into one GPU-accelerated PixiJS canvas.

## Components

### `FieldLayer`
- **Does**: Enumerates supported scalar and phase field views.
- **Interacts with**: Layer selector in `main.ts`.

### `DishRenderer`
- **Does**: Owns PixiJS lifecycle, draw layers, hit testing, and visual settings.
- **Interacts with**: Receives snapshots from `main.ts` and emits field pointer
  coordinates for selection or lesioning.
- **Rationale**: Requests WebGL explicitly because it is broadly available on
  CPU and GPU machines and falls back to Canvas2D when context startup fails or
  stalls, instead of blocking the experiment connection.

### `render`
- **Does**: Rebuilds cell, broadcast-halo, edge, event, and message-pulse geometry
  for an authoritative frame.
- **Interacts with**: Snapshot protocol.
- **Rationale**: Cells render beneath connections so growth, pruning, direction,
  and signal pulses remain inspectable at dense graph states.

### Broadcast and structural layers
- **Does**: Draws cyan axon advertisements, violet receptor advertisements,
  cyan growth flashes, red pruning flashes, and age-sensitive persistent edges.
- **Interacts with**: MNIST growth channels and structural events.
- **Rationale**: The viewer distinguishes asking to connect, forming a new edge,
  retaining it, transmitting on it, and pruning it.

### `setLayer` / `setEdgeThreshold` / `selectCell`
- **Does**: Applies presentation-only controls without changing simulation state.
- **Interacts with**: Viewer controls and inspector.

## Contracts

| Dependent | Expects | Breaking changes |
|-----------|---------|------------------|
| `main.ts` | Initialization is async; render/settings are synchronous | Public API changes |
| Protocol | Flattened index maps as `y * width + x` | Index semantics |
| Styling | Host provides bounded dimensions; canvas fills it | Host layout changes |
| CPU-only browsers | Canvas2D renders the same state and interventions | Fallback removal |

## Notes

Graphics rebuilding is intentionally simple for ≤4,096 edges and 1,024 cells.
A custom mesh or packed buffers should replace it before substantially larger
fields are attempted.
