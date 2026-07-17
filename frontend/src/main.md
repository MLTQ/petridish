# main.ts

## Purpose

Composes transport, renderer, chart, experiment switching, task-adaptive status
and controls, lesioning, digit preview, metrics, and cell inspection.

## Components

### Viewer initialization
- **Does**: Resolves required DOM contracts, initializes PixiJS, opens the stream,
  and attaches native control handlers.
- **Interacts with**: `DishRenderer`, `ExperimentSocket`, and `HistoryChart`.

### `receiveSnapshot`
- **Does**: Applies one authoritative frame, switches task presentation safely by
  discriminator, clears transient loading status, and updates common metrics.
- **Interacts with**: All snapshot-bound elements in `index.html`.

### `showCell`
- **Does**: Renders a concise state readout for a selected field coordinate.
- **Interacts with**: Renderer hit testing and advertised channel names.

### `updateXorTask` / `updateMnistTask`
- **Does**: Present task-specific labels, controls, metrics, and chart meaning.
- **Interacts with**: Discriminated protocol task union and experiment selector.
- **Rationale**: Each function accepts the already-narrowed task payload, keeping
  TypeScript's discriminator guarantee local and explicit.

### `mnistStageLabel`
- **Does**: Names seed, sensory patch-row, recurrent development, and readout
  frames independently from optimizer updates.
- **Interacts with**: MNIST assembly metadata from `protocol.ts`.

### `drawDigit`
- **Does**: Draws the current 28×28 training example without interpolation.
- **Interacts with**: External MNIST preview; pixels are not drawn into the dish.

## Contracts

| Dependent | Expects | Breaking changes |
|-----------|---------|------------------|
| `index.html` | Referenced IDs retain correct native element types | DOM IDs/types |
| Backend runtime | Commands match protocol and are safe to repeat | Command semantics |
| Task union | `kind` narrows before task-specific fields are read | Discriminator semantics |
| Renderer | Pointer coordinates are expressed in cell-space floats | Callback semantics |

## Notes

Lesion drag commands are throttled to prevent pointer sampling rate from flooding
the scientific runtime.
