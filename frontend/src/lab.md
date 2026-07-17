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
- **Rationale**: Selection is independent from run control and limited to four curves.

### `drawChart`
- **Does**: Draws comparable rolling loss against optimizer update using measured JSONL records.

### `renderBenchmarks` / `drawBenchmarkChart`
- **Does**: Tabulates persisted architecture sweeps and plots the newest matched
  task/profile/update cohort on a shared 0–100% held-out accuracy scale.
- **Does**: Keeps replication seeds in the matched cohort; artifact IDs distinguish
  repeated runs directly.
- **Does**: Directly labels curves by artifact ID and states whether the newest cohort
  used deterministic or ordinary seeded execution.
- **Does**: Shows atomic live progress, completion state, elapsed time, and measured
  peak CUDA allocation when the benchmark publisher provides them.
- **Does**: Shows final held-out accuracy for each queried binding slot so a
  one-memory solution is visible without inferring it from the aggregate curve.
- **Does**: Shows presented-value coverage and distractor errors when available,
  separating value storage from correct key/value association.
- **Does**: Shows owner-address distinctness, entropy, and overlap only for artifacts
  that measured an owner map.
- **Does**: Shows recovery intervention identity, final living-cell/edge counts, and
  cumulative births/deaths when matched-recovery artifacts provide them.
- **Does**: Plots living cells and active edges as fractions of the same unlesioned
  base directly below accuracy, preserving intervention color while using solid and
  dashed line style for the two physical measures.
- **Does**: Marks the measured checkpoint where each run changes curriculum
  difficulty; peak and final values remain separately visible in the table.
- **Rationale**: A high peak before a curriculum transition must not be mistaken
  for retention at the harder level.

### `launch` / `stop`
- **Does**: Submit bounded same-origin commands and preserve authoritative server status.

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
