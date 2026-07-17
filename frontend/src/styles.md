# styles.css

## Purpose

Defines the responsive laboratory UI, restrained dark scientific palette,
control states, GPU/run monitoring, canvas sizing, metrics, task readouts, and
chart encoding.

## Components

### Shell and top bar
- **Does**: Establishes identity, connection state, and always-visible task metrics.
- **Interacts with**: `.topbar`, `.brand`, `.connection`, and `.top-metrics`.

### Remote laboratory
- **Does**: Gives two measured GPU lanes equal stable width, keeps run comparison
  tabular, and places the loss chart beside it without affecting dish geometry.
- **Interacts with**: `.laboratory`, `.gpu-lane`, `.run-table`, and `lab.ts`.
- **Does**: Places the controlled benchmark table beside its shared-scale held-out
  accuracy curve and uses point markers only for actual curriculum transitions.
- **Interacts with**: `.benchmark-grid`, `#benchmark-chart`, and persisted artifacts.
- **Does**: Reflows the launch form and GPU lanes at existing responsive breakpoints.

### Dish layout
- **Does**: Gives the visualization dominant space and keeps controls/legend
  adjacent without overlaying scientific marks.
- **Interacts with**: `.dish-panel`, `.dish-toolbar`, and `#dish-host`.
- **Rationale**: Desktop dish height is bounded by both viewport height and the
  available left-column width, avoiding unused bars around the square field.
- **Does**: Groups the dish and task readout in `.visual-column`; the desktop
  task panel has a fixed 390px block size and internal overflow so live content
  cannot reflow the side controls.

### Side panels
- **Does**: Styles interventions, cadence, saved-organism loading, history,
  hyperparameter controls, and inspector as compact peers independent of the
  dynamic task readout.
- **Interacts with**: Semantic sections in `index.html`.

### Trial phases
- **Does**: Gives input, forward, feedback, and structural badges stable encodings.
- **Interacts with**: `data-phase` written by `main.ts`.
- **Does**: Uses bordered token cells to distinguish future, consumed, and current
  sequence positions; the secondary line is the measured prediction.

### Responsive rules
- **Does**: Stacks the dish and panels below 980px and simplifies controls below
  620px.
- **Interacts with**: PixiJS `resizeTo` behavior.

### Hyperparameter controls
- **Does**: Keeps the full numeric schema compact through native nested details
  elements while displaying labels, current values, and full-width sliders.
- **Interacts with**: Generated markup in `main.ts`.

### Corpus generation
- **Does**: Keeps the prompt editor, explicit token-generation actions, generated
  text, and next-token diagnostic readable without decorative marks.
- **Interacts with**: `#generation-panel` and sequence updates in `main.ts`.

### Fast training status
- **Does**: Aligns the single training action with a terse authoritative mode
  status; no animation or decorative progress is implied while traces are disabled.

### Saved organism status
- **Does**: Aligns checkpoint selection and loading while keeping pending and
  loaded identity in a compact monospace status line.

## Contracts

| Dependent | Expects | Breaking changes |
|-----------|---------|------------------|
| Renderer | `#dish-host` has non-zero bounded height and canvas fills it | Host sizing rules |
| `main.ts` | `.lesion-armed` and connection state classes provide feedback | State class removal |
| HTML | Class names correspond to structural groups | Selector renames |
