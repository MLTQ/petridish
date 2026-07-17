# styles.css

## Purpose

Defines the responsive laboratory UI, restrained dark scientific palette,
control states, canvas sizing, metrics, task readouts, and chart encoding.

## Components

### Shell and top bar
- **Does**: Establishes identity, connection state, and always-visible MNIST metrics.
- **Interacts with**: `.topbar`, `.brand`, `.connection`, and `.top-metrics`.

### Dish layout
- **Does**: Gives the visualization dominant space and keeps controls/legend
  adjacent without overlaying scientific marks.
- **Interacts with**: `.dish-panel`, `.dish-toolbar`, and `#dish-host`.
- **Rationale**: Desktop dish height is bounded by both viewport height and the
  available left-column width, avoiding unused bars around the square field.

### Side panels
- **Does**: Styles task, MNIST digit preview, interventions, cadence, history,
  hyperparameter controls, and inspector as compact peers.
- **Interacts with**: Semantic sections in `index.html`.

### Trial phases
- **Does**: Gives input, forward, feedback, and structural badges stable encodings.
- **Interacts with**: `data-phase` written by `main.ts`.

### Responsive rules
- **Does**: Stacks the dish and panels below 980px and simplifies controls below
  620px.
- **Interacts with**: PixiJS `resizeTo` behavior.

### Hyperparameter controls
- **Does**: Keeps the full numeric schema compact through native nested details
  elements while displaying labels, current values, and full-width sliders.
- **Interacts with**: Generated markup in `main.ts`.

## Contracts

| Dependent | Expects | Breaking changes |
|-----------|---------|------------------|
| Renderer | `#dish-host` has non-zero bounded height and canvas fills it | Host sizing rules |
| `main.ts` | `.lesion-armed` and connection state classes provide feedback | State class removal |
| HTML | Class names correspond to structural groups | Selector renames |
