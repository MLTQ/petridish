# index.html

## Purpose

Defines the multi-experiment viewer shell: dominant dish canvas, training and lifecycle
controls, scientific diagnostics, history, and inspection.

## Components

### `#dish-host`
- **Does**: Hosts the PixiJS canvas and pointer interventions.
- **Interacts with**: `DishRenderer` in `renderer.ts`.

### Control elements
- **Does**: Expose experiment selection, playback, measured field layers, edge threshold, evaluation,
  forced lifecycle cycles, lesioning, and speed inputs.
- **Interacts with**: Event wiring in `main.ts`.

### Metric and task elements
- **Does**: Provide stable targets for image or token-sequence task diagnostics and history rendering.
- **Interacts with**: `main.ts` and `HistoryChart`.
- **Does**: Expose measured synapse update ratio and structural lock state.
- **Does**: Expose learning phase, hop distance, temporal reachability, local
  attention entropy, effective capacity, lifecycle state, turnover, energy,
  stress, and death causes.

### MNIST phase readout
- **Does**: Reuses the task heading/badge for input, forward traffic, backward
  credit, and structural stages while the digit remains external to the field.
- **Interacts with**: Persistent-lifetime metadata in `protocol.ts`.

### Sequence readout
- **Does**: Shows ordered tokens, current consumed position, aligned prediction,
  held-out accuracy, and perplexity for recall and language experiments.

### Corpus generation panel
- **Does**: Accepts a character prompt, installs it as the active context, shows
  the generated continuation, and requests exactly one next token per click.
- **Interacts with**: `prompt` and `generate` WebSocket commands.
- **Rationale**: Generation remains paused and deliberate so each new token can
  be inspected against the field and graph state.

### `#hyperparameter-panel`
- **Does**: Hosts grouped backend-defined sliders, pending-change status, and an
  explicit apply-and-restart action for the selected organism.
- **Interacts with**: Dynamic element construction in `main.ts`.

## Contracts

| Dependent | Expects | Breaking changes |
|-----------|---------|------------------|
| `main.ts` | Every referenced element ID exists with the expected native type | ID or element-type changes |
| `styles.css` | Structural class names remain stable | Class changes |
| Accessibility | Controls retain visible labels and native semantics | Removing labels or roles |
| Hyperparameters | Apply clearly states that a new organism is constructed | Silent in-place mutation |
