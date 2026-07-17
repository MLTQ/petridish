# socket.ts

## Purpose

Owns the resilient same-origin WebSocket connection and keeps transport details
out of rendering and UI code.

## Components

### `ExperimentSocket`
- **Does**: Connects, decodes messages, sends typed commands, reports status, and
  reconnects after transient closure.
- **Interacts with**: Protocol types and callbacks supplied by `main.ts`.

### `connect` / `close`
- **Does**: Manage connection and retry lifecycle.
- **Interacts with**: Browser page lifecycle.

### `send`
- **Does**: Serializes one validated `ExperimentCommand`.
- **Interacts with**: FastAPI `/ws`.

## Contracts

| Dependent | Expects | Breaking changes |
|-----------|---------|------------------|
| `main.ts` | Constructor callbacks and `send` remain synchronous | Public API changes |
| Vite/server | Same-origin WebSocket is available at `/ws` | Route changes |
| Backend | Commands are JSON objects matching `ExperimentCommand` | Encoding changes |
