# index.html

## Purpose

Defines the multi-experiment viewer shell: dominant dish canvas, training and lifecycle
controls, scientific diagnostics, history, inspection, and a stable two-GPU
experiment laboratory above the live organism.

## Components

### `#dish-host`
- **Does**: Hosts the PixiJS canvas and pointer interventions.
- **Interacts with**: `DishRenderer` in `renderer.ts`.

### `.laboratory`
- **Does**: Hosts measured GPU lanes, persisted-run comparison, rolling-loss chart,
  and an explicitly enabled launch form without displacing live dish controls.
- **Does**: Aligns benchmark accuracy with a second fixed topology-retention plot;
  solid population and dashed active-edge histories share the unlesioned-base scale.
- **Interacts with**: `LaboratoryView` in `lab.ts` and `/api/lab`.
- **Rationale**: Run-level monitoring changes on a slower cadence than organism
  frames and therefore remains independent from the WebSocket renderer.

### `.visual-column` / `.task-panel`
- **Does**: Keeps the dynamic task/context readout directly below the network
  visualization instead of above the side-column controls.
- **Rationale**: Live token and phase changes cannot displace intervention,
  cadence, or hyperparameter controls while the user is editing them.
- **Does**: Uses a stable desktop height and wider two-column sequence/generation
  layout; narrow screens return to a natural single-column flow.

### Control elements
- **Does**: Expose experiment selection, playback, measured field layers, edge threshold, evaluation,
  forced lifecycle cycles, lesioning, and speed inputs.
- **Interacts with**: Event wiring in `main.ts`.
- **Does**: Defaults the field to a phase-aware signal layer: activation during
  forward computation and measured gradient credit during feedback.

### Headless training control
- **Does**: Lets sequence experiments run optimizer updates without token-frame
  capture and states plainly that traces are suspended.
- **Does**: Shows measured update latency and updates per second rather than an
  arbitrary speed multiplier.

### Cadence control
- **Does**: Selects simulation steps per MNIST frame or measured token-frame
  sampling stride for sequence organisms.

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
| `lab.ts` | Laboratory table, chart, status, and launch IDs are stable | ID changes |
| `styles.css` | Structural class names remain stable | Class changes |
| Accessibility | Controls retain visible labels and native semantics | Removing labels or roles |
| Hyperparameters | Apply clearly states that a new organism is constructed | Silent in-place mutation |
