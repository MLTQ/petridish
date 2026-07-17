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
- **Rationale**: Heavy PyTorch calls run in worker threads so the async WebSocket
  loop remains responsive during multi-second accelerator updates.

### `connect` / `disconnect` / `broadcast`
- **Does**: Maintain live observers and send sampled authoritative snapshots.
- **Interacts with**: `server.websocket_endpoint`.
- **Does**: Reuses the latest immutable serialized state for immediate control
  acknowledgements; a dedicated send lock preserves snapshot order.

### `_snapshot`
- **Does**: Adds authoritative runtime mode, running state, compute latency, and
  measured optimizer updates/second and training sequences/second to the
  scientific snapshot.
- **Does**: Publishes the active compute phase plus exact progress for token
  forward passes and evaluation batches.
- **Does**: Includes a monotonic control revision so observers reject snapshots
  serialized before a newer Play, Pause, or cadence command.

### `_run_sequence_visual_update`
- **Does**: Runs one sequence optimizer update in a worker while relaying sampled
  snapshots produced from actual token, feedback, and structural frames.
- **Does**: Samples backward autograd-hook frames using the same selected token
  stride, producing visible reverse credit propagation without proxy particles.
- **Does**: Aborts a sequence update at a safe pre-optimizer boundary when a
  control command is waiting, discarding the partial batch rather than applying
  a half-observed update.
- **Rationale**: Optimizer phases do not invent field motion; they retain the
  last measured state and expose the operation name instead.
- **Rationale**: A two-item latest-value queue prevents a slow observer from
  accumulating stale frames. The cadence selector controls token sampling stride.

### `handle_command`
- **Does**: Bounds and applies experiment switching, playback, reset, lesion,
  evaluation, forced lifecycle cycles, speed, and atomic hyperparameter restarts.
- **Interacts with**: Frontend `ExperimentSocket`.
- **Does**: Pauses corpus experiments while replacing the prompt or generating
  one sampled token, so visible state corresponds to the displayed completion.
- **Does**: Enters headless trace-free sequence training or exits it by rebuilding one
  current visual trace.
- **Does**: Applies Play, Pause, and cadence commands without waiting for the
  scientific mutation lock. Other commands request an early safe interruption
  before acquiring that lock.

## Contracts

| Dependent | Expects | Breaking changes |
|-----------|---------|------------------|
| `server.py` | Lifecycle and WebSocket methods are asynchronous | Method signatures |
| Frontend | Command `type` names and payload fields remain stable | Command schema |
| Scientific state | Commands never race a physics tick or optimizer update | Removing lock discipline |
| Hyperparameters | Values validate before a new organism replaces the old one | In-place config mutation |

## Notes

Sequence visualization streams scientific states during the compute-bound update.
Headless training has no artificial delay, suppresses automatic validation, and
reports at most once per second. Broadcasts remain sequential because observer count
is expected to be one.
