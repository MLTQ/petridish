# laboratory.py

## Purpose

Provides the remote laboratory control plane independently of the interactive
organism runtime. It discovers measured NVIDIA state and persisted runs, reads
bounded metric histories, and supervises explicitly enabled trainer processes.

## Components

### `LaunchSpec`
- **Does**: Defines validated, reproducible arguments for one unattended corpus run.
- **Rationale**: Process commands are assembled from typed fields rather than shell text.

### `Laboratory.snapshot`
- **Does**: Reports GPU telemetry, compute processes, capabilities, and run summaries.
- **Interacts with**: `/api/lab` in `server.py` and `lab.ts`.

### `Laboratory.metrics`
- **Does**: Returns at most 2,000 valid records from the tail of a run's JSONL log.
- **Rationale**: Browser polling cannot force unbounded file reads or payloads.

### `Laboratory.launch`
- **Does**: Creates an immutable manifest and starts `train_shakespeare` with a GPU UUID.
- **Rationale**: UUID pinning avoids the host's conflicting CUDA and `nvidia-smi` indices.

### `Laboratory.stop_run`
- **Does**: Sends SIGTERM so the trainer completes its current update and checkpoints.

## Contracts

| Dependent | Expects | Breaking changes |
|-----------|---------|------------------|
| `server.py` | Snapshot, metrics, launch, and stop methods are synchronous | Method signatures |
| `lab.ts` | camelCase GPU/run fields and bounded histories | Payload field changes |
| Trainer | SIGTERM is checkpoint-safe; CLI arguments remain supported | Trainer CLI changes |
| Security | Run IDs cannot escape `runs/`; launch uses no shell | Path or subprocess handling |

## Notes

- Process control is disabled unless `PETRIDISH_LAB_CONTROL=1`.
- Server shutdown closes log descriptors but deliberately does not terminate trainers.
- GRU, LSTM, ESN, and temporal-transformer homogeneous controls share one trainer
  contract. Mixtures remain unavailable until their type/lifecycle semantics exist.
