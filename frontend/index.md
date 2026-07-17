# index.html

## Purpose

Defines the semantic viewer shell: experiment selection, dominant dish canvas,
task-adaptive controls/status, scientific diagnostics, history, and inspection.

## Components

### `#dish-host`
- **Does**: Hosts the PixiJS canvas and pointer interventions.
- **Interacts with**: `DishRenderer` in `renderer.ts`.

### Control elements
- **Does**: Expose experiment switching, playback, layer, edge threshold,
  XOR stimuli/reward, MNIST evaluation/structural-cycle, lesion, and speed inputs.
- **Interacts with**: Event wiring in `main.ts`.

### Metric and task elements
- **Does**: Provide stable targets for XOR or MNIST snapshots, digit preview, and
  history rendering.
- **Interacts with**: `main.ts` and `HistoryChart`.
- **Does**: Expose measured synapse update ratio and structural lock state.
- **Does**: Expose learning phase, hop distance, temporal reachability, local
  attention entropy, and effective trainable capacity.

### MNIST phase readout
- **Does**: Reuses the task heading/badge for input, forward traffic, backward
  credit, and structural stages while the digit remains external to the field.
- **Interacts with**: Persistent-lifetime metadata in `protocol.ts`.

### `#hyperparameter-panel`
- **Does**: Hosts grouped backend-defined sliders, pending-change status, and an
  explicit apply-and-restart action for the MNIST organism.
- **Interacts with**: Dynamic element construction in `main.ts`.

## Contracts

| Dependent | Expects | Breaking changes |
|-----------|---------|------------------|
| `main.ts` | Every referenced element ID exists with the expected native type | ID or element-type changes |
| `styles.css` | Structural class names remain stable | Class changes |
| Accessibility | Controls retain visible labels and native semantics | Removing labels or roles |
| Hyperparameters | Apply clearly states that a new organism is constructed | Silent in-place mutation |
