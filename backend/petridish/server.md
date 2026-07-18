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
explicitly enabled launch, same-lineage plasticity continuation, and checkpoint-safe
stop plus exact checkpoint-fork, failed-phase retry, read-only checkpoint evaluation,
and exact-phase resume after a deliberate stop.
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
- **Does**: Selects legacy wordpiece or byte-complete TinyStories tokenization; the
  laboratory enforces the byte profile's exact 256-token vocabulary.
- **Does**: Carries an explicit continuous or cold-window experience mode; the
  laboratory performs exact validation before process launch.
- **Does**: Bounds state retention from zero through one for recorded electrical
  relaxation experiments.
- **Does**: Bounds persistent trajectory lanes from one through 128 separately
  from CUDA tensor batch size.
- **Rationale**: The API never accepts a shell command or arbitrary path.

### `LabContinueRequest`
- **Does**: Bounds additional updates and selects only the next topology/lifecycle
  policy for an existing checkpointed organism.
- **Does**: Accepts an optional repeated-shard curriculum for a continuation phase;
  omission preserves the active stream and zero selects the full corpus.
- **Does**: Accepts an optional one-to-128 target lane count for append-only
  persistent experience expansion; the laboratory rejects any requested shrink.
- **Does**: Accepts fixed, adaptive, or prune-only topology; the laboratory performs
  categorical validation and records the resolved phase policy.
- **Rationale**: Architecture, geometry, weights, state, and existing corpus cursors
  come from the checkpoint. A shard changes which experiences repeat and a lane
  expansion adds cold trajectories; neither replaces organism-owned state.

### `LabEvaluateRequest`
- **Does**: Selects a measured GPU and optional state-horizon sweep for read-only
  evaluation of an existing checkpoint.
- **Does**: Validates an explicit `validation` or `training` split; the training
  choice audits the active repeated shard without invoking continuation or an
  optimizer step.
- **Does**: Accepts `trajectory` as the third explicit audit condition for the exact
  next checkpointed cursor/state lane.
- **Does**: Optionally bounds an explicit trajectory lane from zero through 127;
  the laboratory rejects it for every non-trajectory evaluation.

### `LabForkRequest`
- **Does**: Accepts only a bounded run ID for a new stopped-checkpoint branch; the
  laboratory preserves and fingerprints the complete parent organism state.

### `LabRetryRequest`
- **Does**: Selects only a measured GPU; the laboratory derives every training,
  lineage, phase, and checkpoint argument from the failed run's persisted manifest.
- **Rationale**: The browser cannot accidentally turn recovery into a new organism or
  a new experimental phase.

### `LabResumeRequest`
- **Does**: Selects only a measured GPU for resuming an incomplete deliberately
  stopped phase; target, lineage, topology, lifecycle, curriculum, and state remain
  server-derived checkpoint contracts.

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
