# runtime.py

## Purpose

Decouples XOR/MNIST experiment cadence from WebSocket handling and serializes all
training, simulation, switching, and intervention mutations behind one lock.

## Components

### `ExperimentRuntime`
- **Does**: Lazily owns named experiments, active selection, playback state,
  observers, and the tick loop.
- **Interacts with**: FastAPI WebSockets, XOR/MNIST experiments, `build_snapshot`.

### `start` / `stop`
- **Does**: Manage the lifecycle of the independent simulation task.
- **Interacts with**: FastAPI application lifespan.

### `connect` / `disconnect` / `broadcast`
- **Does**: Maintain live observers and send sampled authoritative snapshots.
- **Interacts with**: `server.websocket_endpoint`.

### `handle_command`
- **Does**: Bounds and applies playback, experiment switching, reset, lesion,
  task-specific stimulus/reward/evaluation/new-assembly, and speed commands.
- **Interacts with**: Frontend `ExperimentSocket`.

## Contracts

| Dependent | Expects | Breaking changes |
|-----------|---------|------------------|
| `server.py` | Lifecycle and WebSocket methods are asynchronous | Method signatures |
| Frontend | Command `type` names and payload fields remain stable | Command schema |
| Scientific state | Commands never race a physics tick | Removing lock discipline |
| Experiment switching | Previously created experiment state is preserved until explicit reset | Cache semantics |

## Notes

The MVP broadcasts sequentially because observer count is expected to be one.
Per-client queues are the next scaling step if multiple slow viewers are needed.
