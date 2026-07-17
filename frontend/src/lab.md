# lab.ts

## Purpose

Owns the slower-cadence two-GPU experiment laboratory independently from the
high-frequency organism WebSocket. It polls measured hardware/run state, compares
bounded loss histories, and submits validated launch or stop requests.

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

### `launch` / `stop`
- **Does**: Submit bounded same-origin commands and preserve authoritative server status.

## Contracts

| Dependent | Expects | Breaking changes |
|-----------|---------|------------------|
| `main.ts` | `LaboratoryView.start()` is non-blocking | Constructor/start signature |
| `index.html` | Laboratory IDs use native controls and one SVG | ID or element changes |
| Backend | `/api/lab` payloads retain documented camelCase fields | API schema changes |

## Notes

- Polling is serialized so a slow request cannot accumulate stale UI work.
- Capability selectors advertise only architectures implemented by the backend.
- No synthetic progress is shown; missing values remain an em dash.
