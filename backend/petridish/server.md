# server.py

## Purpose

Exposes health, the live command/snapshot WebSocket, and an optional built
frontend from one FastAPI application.

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
| Deployment | Built assets live under `frontend/dist` | Directory or mount changes |
| Tests | Health returns status, experiment, device, and tick | Response schema |
