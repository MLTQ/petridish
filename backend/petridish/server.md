# server.py

## Purpose

Exposes health, the live command/snapshot WebSocket, the two-GPU laboratory API,
and an optional built frontend from one FastAPI application. The interactive
runtime device is configurable independently from trainer workers.

## Components

### `runtime`
- **Does**: Holds the process-wide interactive experiment.
- **Interacts with**: Application lifespan and every WebSocket client.

### `lifespan`
- **Does**: Starts and cancels the independent physics loop cleanly.
- **Interacts with**: FastAPI startup and shutdown.

### `health`
- **Does**: Reports readiness, active experiment, accelerator, and current tick.
- **Interacts with**: Smoke tests and operators.

### Laboratory routes
- **Does**: Report measured GPUs/runs, return bounded metric histories, and expose
  explicitly enabled launch and checkpoint-safe stop actions.
- **Interacts with**: `Laboratory` in `laboratory.py` and `lab.ts`.
- **Rationale**: Trainer processes remain independent from the interactive organism
  so opening the viewer cannot stall or acquire their GPU state.

### `LabLaunchRequest`
- **Does**: Bounds the selected corpus task and every process argument accepted from the browser.
- **Does**: Carries a named lifecycle intervention separately from the compatibility
  enable flag.
- **Does**: Carries adaptive/fixed topology independently from lifecycle.
- **Does**: Bounds a common learning-rate scale from 0.01 through 1.0 for stability
  controls.
- **Does**: Bounds corpus broadcast gain from 0 through 2 so physical-only and
  global-workspace runs are explicit API interventions.
- **Does**: Restricts token-vocabulary curriculum requests to 64–2,048 entries;
  the laboratory applies the exact supported power-of-two set.
- **Does**: Carries an explicit continuous or cold-window experience mode; the
  laboratory performs exact validation before process launch.
- **Does**: Bounds state retention from zero through one for recorded electrical
  relaxation experiments.
- **Does**: Bounds persistent trajectory lanes from one through sixteen separately
  from CUDA tensor batch size.
- **Rationale**: The API never accepts a shell command or arbitrary path.

### `websocket_endpoint`
- **Does**: Connects observers, receives JSON commands, and returns command errors.
- **Interacts with**: `ExperimentRuntime` and frontend `ExperimentSocket`.

### `frontend`
- **Does**: Serves the Vite production build when `frontend/dist` exists.
- **Interacts with**: `npm run build` output.

## Contracts

| Dependent | Expects | Breaking changes |
|-----------|---------|------------------|
| Frontend dev proxy | WebSocket at `/ws`, health at `/api/health` | Route changes |
| Laboratory frontend | Same-origin `/api/lab` routes use camelCase payloads | Schema changes |
| Deployment | Built assets live under `frontend/dist` | Directory or mount changes |
| Tests | Health returns status, experiment, device, and tick | Response schema |

## Notes

- Set `PETRIDISH_DEVICE=cpu` for a monitor-only server beside GPU trainers.
- Set `PETRIDISH_AUTOPLAY=0` to construct the viewer paused rather than spending
  CPU continuously, and `PETRIDISH_RUN_ROOT` when deployment uses a worktree
  separate from the trainer's shared run directory.
- Set `PETRIDISH_BENCHMARK_ROOT` to a shared directory of persisted sequence
  benchmark JSON artifacts when the server runs from a disposable worktree.
- Mutating laboratory routes require `PETRIDISH_LAB_CONTROL=1` at server startup.
