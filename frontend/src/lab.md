# lab.ts

## Purpose

Owns the slower-cadence two-GPU experiment laboratory independently from the
high-frequency organism WebSocket. It polls measured hardware, corpus runs, and
stepping-stone benchmark artifacts; compares bounded histories; and submits
validated launch or stop requests.

## Components

### `LaboratoryView`
- **Does**: Coordinates polling, stable DOM updates, selection, and control actions.
- **Interacts with**: `/api/lab` routes and laboratory IDs in `index.html`.

### `renderGpus`
- **Does**: Shows actual utilization, power, memory, and compute-process count per GPU.

### `renderRuns`
- **Does**: Presents task, cell type, hardware, update, loss, throughput, and state.
- **Rationale**: Comparison is bounded to five curves so the complete lifecycle
  ablation matrix remains visible without unbounded chart density.

### `drawChart`
- **Does**: Draws comparable rolling loss against optimizer update using measured JSONL records.

### `renderRunDiagnostics`
- **Does**: Shows measured physical/conducting topology, token/context/graph reach,
  cell and edge turnover, pruning pressure, and the latest fixed-prompt generation
  for each selected corpus run.
- **Rationale**: Loss alone cannot establish that a cellular routing or lifecycle
  mechanism is functioning.

### `renderBenchmarks` / `drawBenchmarkChart`
- **Does**: Tabulates persisted architecture sweeps and plots the newest matched
  task/profile/update cohort on a shared 0–100% held-out accuracy scale.
- **Does**: Keeps replication seeds in the matched cohort; artifact IDs distinguish
  repeated runs directly.
- **Does**: Directly labels curves by artifact ID and states whether the newest cohort
  used deterministic or ordinary seeded execution.
- **Does**: Labels matched process-global branch RNG separately from deterministic
  kernels, because stochastic lifecycle mutation uses both guarantees.
- **Does**: Shows atomic live progress, completion state, elapsed time, and measured
  peak CUDA allocation when the benchmark publisher provides them.
- **Does**: Shows failed runs with their persisted exception type/message and keeps
  an empty pre-evaluation failure from producing invalid chart axes.
- **Does**: Shows final held-out accuracy for each queried binding slot so a
  one-memory solution is visible without inferring it from the aggregate curve.
- **Does**: Shows final accuracy at each supervised stream position so persistent
  context decay is visible instead of averaged away.
- **Does**: Shows presented-value coverage and distractor errors when available,
  separating value storage from correct key/value association.
- **Does**: Shows owner-address distinctness, entropy, and overlap only for artifacts
  that measured an owner map.
- **Does**: Shows recovery intervention identity, final living-cell/edge counts, and
  cumulative births/deaths when matched-recovery artifacts provide them.
- **Does**: Retains parsing of historical recovery artifacts while the active UI
  hides the lesion-oriented topology-recovery subchart.
- **Does**: Marks the measured checkpoint where each run changes curriculum
  difficulty; peak and final values remain separately visible in the table.
- **Rationale**: A high peak before a curriculum transition must not be mistaken
  for retention at the harder level.

### `launch` / `stop`
- **Does**: Submit bounded, task-aware Tiny Shakespeare or TinyStories commands and
  preserve authoritative server status.
- **Does**: Launch both corpus organisms at 68×68 so their 64- or 66-port banks
  remain one physical column per boundary.
- **Does**: Selects and displays named lifecycle interventions so baseline and
  balanced/replacement ablations remain distinguishable in manifests and diagnostics.
- **Does**: Launches an explicit microtick budget and flags runs whose budget is
  shorter than the measured minimum sensory-to-output route.
- **Does**: Launches and displays fixed/adaptive topology independently from the
  selected lifecycle profile, and accepts a measured `failed` run state.
- **Does**: Launches a bounded common learning-rate scale so stability controls are
  recorded in the immutable manifest rather than applied out of band.
- **Does**: Renders direct-routing controls with microticks, minimum hops, and
  one-token output reach instead of recall-only columns, with the measured task's
  chance baseline drawn directly on the accuracy plot.
- **Does**: Renders distributed context controls with sequence length, dependency
  horizon, and token/context/graph reach on the same measured accuracy surface.
- **Does**: Places delayed-copy memory between direct routing and XOR composition
  without changing the chart or topology semantics.
- **Does**: Places repeated copy/invert token streams after XOR composition and
  preserves their dependency horizon plus per-position accuracy.
- **Does**: Treats fixed-latency contextual pipelines as token controls, preserving
  their physical route budget and delayed per-position accuracy.
- **Does**: Treats masked context-settling clocks as a distinct token control so
  context-propagation time is not conflated with output latency.
- **Does**: Treats the decorrelated settled pipeline as the combined causal control,
  preserving its longer dependency horizon and delayed position metrics.
- **Does**: Keeps broadcast-on/off identity in the persisted intervention label so
  global-workspace shortcuts are not conflated with physical dendritic routing.

## Contracts

| Dependent | Expects | Breaking changes |
|-----------|---------|------------------|
| `main.ts` | `LaboratoryView.start()` is non-blocking | Constructor/start signature |
| `index.html` | Laboratory IDs use native controls and two measured SVG plots | ID or element changes |
| Backend | `/api/lab` payloads retain documented camelCase fields | API schema changes |

## Notes

- Polling is serialized so a slow request cannot accumulate stale UI work.
- Capability selectors advertise only architectures implemented by the backend.
- No synthetic progress is shown; missing values remain an em dash.
