# main.ts

## Purpose

Composes the MNIST transport, renderer, chart, lifecycle controls, lesioning,
digit preview, metrics, and cell inspection.

## Components

### Viewer initialization
- **Does**: Resolves required DOM contracts, initializes PixiJS, opens the stream,
  and attaches native control handlers.
- **Interacts with**: `DishRenderer`, `ExperimentSocket`, and `HistoryChart`.

### `receiveSnapshot`
- **Does**: Applies one authoritative MNIST frame, clears transient loading
  status, and updates learning, routing, metabolism, and turnover metrics.
- **Interacts with**: All snapshot-bound elements in `index.html`.

### `showCell`
- **Does**: Renders a concise state readout for a selected field coordinate.
- **Interacts with**: Renderer hit testing and advertised channel names.

### `updateMnistTask`
- **Does**: Presents digit, curriculum, learning phase, and structure state.
- **Interacts with**: The MNIST-only protocol payload.
- **Does**: Shows whether structure is locked for learning warm-up and the
  measured relative synapse update produced by the latest optimizer step.
- **Does**: Shows overfit-stage progress, active optimizer family, attention
  selectivity, effective capacity, directed hop distance, and temporal reachability.
- **Does**: Shows lifecycle activation, energy/stress, cumulative turnover, and
  the latest measured death-cause split.

### `mnistStageLabel`
- **Does**: Names input, actual forward traffic, backward gradient credit, and
  structural lifecycle frames independently from readout/rule/synapse phases.
- **Interacts with**: Persistent-lifetime metadata from `protocol.ts`.

### `drawDigit`
- **Does**: Draws the current 28×28 training example without interpolation.
- **Interacts with**: External MNIST preview; pixels are not drawn into the dish.

### `renderHyperparameters` / `hyperparameterControl`
- **Does**: Builds grouped sliders from the backend schema, stages edits without
  interrupting the run, and submits one atomic apply-and-restart command.
- **Rationale**: Dragging a slider never creates a sequence of incomparable
  partial organisms.

## Contracts

| Dependent | Expects | Breaking changes |
|-----------|---------|------------------|
| `index.html` | Referenced IDs retain correct native element types | DOM IDs/types |
| Backend runtime | Commands match protocol and are safe to repeat | Command semantics |
| Renderer | Pointer coordinates are expressed in cell-space floats; sparse row
  selection resolves through `field.indices` | Callback semantics |
| Hyperparameter schema | Every rendered slider is backend-defined and functional | Payload shape changes |

## Notes

Lesion drag commands are throttled to prevent pointer sampling rate from flooding
the scientific runtime.
