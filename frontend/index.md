# index.html

## Purpose

Defines the semantic viewer shell: experiment selection, dominant dish canvas,
task-adaptive controls/status, interventions, history, and cell inspection.

## Components

### `#dish-host`
- **Does**: Hosts the PixiJS canvas and pointer interventions.
- **Interacts with**: `DishRenderer` in `renderer.ts`.

### Control elements
- **Does**: Expose experiment switching, playback, layer, edge threshold,
  XOR stimuli/reward, MNIST evaluation/new-assembly, lesion, and speed inputs.
- **Interacts with**: Event wiring in `main.ts`.

### Metric and task elements
- **Does**: Provide stable targets for XOR or MNIST snapshots, digit preview, and
  history rendering.
- **Interacts with**: `main.ts` and `HistoryChart`.

### MNIST phase readout
- **Does**: Reuses the task heading/badge for seed, sensory-token, development,
  and readout stages while the digit preview remains external to the field.
- **Interacts with**: Assembly metadata in `protocol.ts`.

## Contracts

| Dependent | Expects | Breaking changes |
|-----------|---------|------------------|
| `main.ts` | Every referenced element ID exists with the expected native type | ID or element-type changes |
| `styles.css` | Structural class names remain stable | Class changes |
| Accessibility | Controls retain visible labels and native semantics | Removing labels or roles |
