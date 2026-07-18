# laboratory.py

## Purpose

Provides the remote laboratory control plane independently of the interactive
organism runtime. It discovers measured NVIDIA state, persisted corpus runs, and
bounded stepping-stone benchmark artifacts; reads metric histories; and supervises
explicitly enabled trainer processes.

## Components

### `LaunchSpec`
- **Does**: Defines validated, reproducible arguments for one unattended
  Tiny Shakespeare or distributed-token TinyStories run.
- **Does**: Requires the 68×68 geometry that gives both corpus tasks one linear
  input column and one linear output column.
- **Rationale**: Process commands are assembled from typed fields rather than shell text.

### `Laboratory.snapshot`
- **Does**: Reports GPU telemetry, compute processes, capabilities, and run summaries.
- **Interacts with**: `/api/lab` in `server.py` and `lab.ts`.

### `Laboratory.metrics`
- **Does**: Returns at most 2,000 valid records from the tail of a run's JSONL log.
- **Rationale**: Browser polling cannot force unbounded file reads or payloads.
- **Does**: Keeps optimizer, held-out/generation, and scientific diagnostic record
  types distinct so the frontend never infers biology from loss records.

### `Laboratory._discover_benchmarks`
- **Does**: Reads up to 100 recent JSON benchmark artifacts and returns at most
  2,000 valid held-out checkpoints per artifact.
- **Does**: Preserves atomic publisher status, completed updates, parameter counts,
  and measured CUDA allocation so in-progress experiments remain distinguishable
  from completed evidence.
- **Does**: Preserves optional neuron-owner address diagnostics for relational-memory
  experiments without synthesizing values for baseline artifacts.
- **Does**: Preserves the deterministic-execution flag from benchmark artifacts.
- **Does**: Preserves an explicit process-global branch-RNG match flag; deterministic
  kernels alone do not prove stochastic lifecycle branches received the same stream.
- **Does**: Preserves intervention, lesion, topology, and turnover fields for matched
  recovery artifacts.
- **Rationale**: Curriculum transitions and architecture comparisons should be
  based on persisted measurements while keeping the polling payload bounded.

### `Laboratory.launch`
- **Does**: Creates an immutable task-aware manifest and starts the corpus trainer with a GPU UUID.
- **Rationale**: UUID pinning avoids the host's conflicting CUDA and `nvidia-smi` indices.

### `Laboratory.stop_run`
- **Does**: Sends SIGTERM so the trainer completes its current update and checkpoints.

### `Laboratory._pid_alive`
- **Does**: Treats Linux zombie processes as stopped before falling back to a
  portable signal probe.
- **Rationale**: A failed CUDA worker can remain in `/proc` until reaped; its PID
  must not keep a failed experiment labeled as running.

## Contracts

| Dependent | Expects | Breaking changes |
|-----------|---------|------------------|
| `server.py` | Snapshot, metrics, launch, and stop methods are synchronous | Method signatures |
| `lab.ts` | camelCase GPU/run/benchmark fields and bounded histories | Payload field changes |
| Trainer | SIGTERM is checkpoint-safe; CLI arguments remain supported | Trainer CLI changes |
| Security | Run IDs cannot escape `runs/`; launch uses no shell | Path or subprocess handling |

## Notes

- Process control is disabled unless `PETRIDISH_LAB_CONTROL=1`.
- Benchmark discovery defaults to `benchmarks/lab` and may be redirected to a
  shared artifact directory independently of trainer runs.
- Server shutdown closes log descriptors but deliberately does not terminate trainers.
- GRU, LSTM, ESN, and temporal-transformer homogeneous controls share one trainer
  contract. Mixtures remain unavailable until their type/lifecycle semantics exist.
