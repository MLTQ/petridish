# main.ts

## Purpose

Composes transport, renderer, charts, laboratory monitoring, lifecycle controls,
task previews, interactive corpus generation, metrics, and cell
inspection for every organism.

## Components

### Viewer initialization
- **Does**: Resolves required DOM contracts, initializes PixiJS, opens the stream,
  and attaches native control handlers.
- **Interacts with**: `DishRenderer`, `ExperimentSocket`, and `HistoryChart`.
- **Does**: Disables a stateful control only while its single requested state is
  pending, preventing repeated clicks from queuing contradictory commands.
- **Does**: Clears pending controls and revision history on reconnect so a
  restarted server can establish a fresh authoritative stream.
- **Does**: Submits saved-organism selections as pending operations until the
  backend confirms the loaded checkpoint identity.
- **Does**: Starts the independent `LaboratoryView`; run polling never blocks or
  mutates the high-frequency organism WebSocket.

### `receiveSnapshot`
- **Does**: Applies one authoritative MNIST frame, clears transient loading
  status, and updates learning, routing, metabolism, and turnover metrics.
- **Interacts with**: All snapshot-bound elements in `index.html`.
- **Does**: Suppresses renderer work during headless training while continuing
  to update loss, accuracy, optimizer and sequence throughput, and lifecycle diagnostics.
- **Does**: During visual sequence updates, renders each streamed authoritative
  snapshot and names forward, backward, optimizer, local-credit, lifecycle, and
  evaluation work without manufacturing animation.
- **Does**: Rejects snapshots older than the latest control revision and keeps
  optimistic playback/experiment choices stable until the backend confirms them.

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

### `updateSequenceTask`
- **Does**: Renders a bounded token window around the active context position,
  corpus/tokenizer metadata, perplexity, and sequence accuracy.
- **Does**: For corpus tasks, keeps prompt editing local while focused and shows
  the authoritative generated suffix plus next greedy-token diagnostic.
- **Rationale**: Whitespace tokens receive explicit visible labels so newlines
  and spaces cannot masquerade as missing data.
- **Does**: Makes leading-space wordpieces explicit and reports measured energy,
  stunned population, cumulative damage, and excitotoxic deaths.
- **Does**: Exposes headless training only for sequence organisms; stopping it asks
  the backend to rebuild a current trace before rendering resumes.
- **Rationale**: Mode switches remain pending and disable the action until the
  current indivisible optimizer update finishes and the backend confirms the state.

### `mnistStageLabel`
- **Does**: Names input, actual forward traffic, backward gradient credit, and
  structural lifecycle frames independently from readout/rule/synapse phases.
- **Interacts with**: Persistent-lifetime metadata from `protocol.ts`.

### `computePhaseLabel` / `updateCadenceLabels`
- **Does**: Names the currently measured sequence operation and its real progress.
- **Does**: During backward traversal, reports the measured token-credit position
  rather than a generic blocking status.
- **Does**: Describes sequence cadence as token-frame sampling rather than a
  misleading simulation speed multiplier; MNIST retains step-speed labels.
- **Rationale**: Rewrites native option labels only when the task kind changes;
  mutating options on every training snapshot prevents an open macOS menu from
  accepting or dismissing a selection.
- **Does**: Leaves the cadence select untouched while focused and preserves a
  selected speed until its revised backend snapshot confirms the value.

### `drawDigit`
- **Does**: Draws the current 28×28 training example without interpolation.
- **Interacts with**: External MNIST preview; pixels are not drawn into the dish.

### `renderHyperparameters` / `hyperparameterControl`
- **Does**: Builds grouped sliders from the backend schema, stages edits without
  interrupting the run, and submits one atomic apply-and-restart command.
- **Rationale**: Dragging a slider never creates a sequence of incomparable
  partial organisms.
- **Does**: Maps backend-provided discrete choices to slider indices while
  sending the selected scientific value, including power-of-two field sizes.

### `updateSavedOrganisms`
- **Does**: Reconciles the backend checkpoint catalog into a native selector,
  reports pending/loaded state, and disables loading when no checkpoints exist.
- **Rationale**: The browser sends an opaque checkpoint identifier; local paths
  and PyTorch deserialization remain backend-only concerns.

## Contracts

| Dependent | Expects | Breaking changes |
|-----------|---------|------------------|
| `index.html` | Referenced IDs retain correct native element types | DOM IDs/types |
| Backend runtime | Commands match protocol and are safe to repeat | Command semantics |
| Renderer | Pointer coordinates are expressed in cell-space floats; sparse row
  selection resolves through `field.indices` | Callback semantics |
| Hyperparameter schema | Every rendered slider is backend-defined and functional | Payload shape changes |

## Notes

Dish pointer input is inspection-only; the lesion recovery workflow has been removed.
- Experiment selection switches among persistent MNIST, associative-recall, and
  language organisms and resets only viewer-local chart/configuration state.
- Task rendering is discriminated: MNIST keeps its external image preview and
  curriculum, while sequences show ordered tokens, current position, predictions,
  held-out accuracy, and perplexity.
- Recall status names the active one-to-three-binding curriculum and its recent
  stage accuracy; language accuracy refers only to context-dependent positions.
