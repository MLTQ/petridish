# runtime.py

## Purpose

Decouples the selected organism cadence from WebSocket handling and serializes all
training and intervention mutations behind one lock.

## Components

### `ExperimentRuntime`
- **Does**: Lazily owns MNIST, associative-recall, synthetic tiny-language, and
  Tiny Shakespeare organisms;
  switching preserves each organism's learned state.
- **Interacts with**: FastAPI WebSockets, experiment classes, `build_snapshot`.

### `start` / `stop`
- **Does**: Manage the lifecycle of the independent simulation task.
- **Interacts with**: FastAPI application lifespan.

### `connect` / `disconnect` / `broadcast`
- **Does**: Maintain live observers and send sampled authoritative snapshots.
- **Interacts with**: `server.websocket_endpoint`.

### `handle_command`
- **Does**: Bounds and applies experiment switching, playback, reset, lesion,
  evaluation, forced lifecycle cycles, speed, and atomic hyperparameter restarts.
- **Interacts with**: Frontend `ExperimentSocket`.
- **Does**: Pauses corpus experiments while replacing the prompt or generating
  one sampled token, so visible state corresponds to the displayed completion.

## Contracts

| Dependent | Expects | Breaking changes |
|-----------|---------|------------------|
| `server.py` | Lifecycle and WebSocket methods are asynchronous | Method signatures |
| Frontend | Command `type` names and payload fields remain stable | Command schema |
| Scientific state | Commands never race a physics tick | Removing lock discipline |
| Hyperparameters | Values validate before a new organism replaces the old one | In-place config mutation |

## Notes

The MVP broadcasts sequentially because observer count is expected to be one.
Per-client queues are the next scaling step if multiple slow viewers are needed.
