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
- **Does**: Records `off`, original `baseline`, empirically `balanced`, or
  population-stable `replacement` lifecycle policy in the immutable run manifest.
- **Does**: Records adaptive versus fixed topology independently from lifecycle.
- **Does**: Records and bounds one common learning-rate scale for rule, readout,
  and synapse optimizer groups.
- **Does**: Records a bounded broadcast-workspace gain independently from the
  microtick budget; zero is the hard physical-routing ablation.
- **Does**: Records a 64–2,048 power-of-two token-vocabulary curriculum while
  preserving the same 64-port physical population code.
- **Does**: Records continuous organism experience versus the random cold-window
  control and passes that immutable mode to the trainer.
- **Does**: Records and bounds electrical retention independently from structure;
  the default 0.9 is a homeostatic-relaxation intervention, while 1.0 is the exact
  indefinite-state control.
- **Does**: Records one to sixteen round-robin persistent state lanes independently
  from tensor batch size.
- **Does**: Records fixed, adaptive, or prune-only topology independently from
  lifecycle. Prune-only retains the lineage and forbids replacement growth.
- **Does**: Derives the compatibility lifecycle boolean from the resolved profile so
  the manifest and trainer command cannot disagree.

### `Laboratory.snapshot`
- **Does**: Reports GPU telemetry, compute processes, capabilities, and run summaries.
- **Does**: Advertises checkpoint evaluation explicitly so a frontend loaded before
  the matching server restart cannot expose a nonfunctional mutation control.
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
- **Does**: Preserves failed artifacts even before their first evaluation checkpoint,
  including the bounded exception type/message published by the benchmark process.
- **Does**: Preserves optional neuron-owner address diagnostics for relational-memory
  experiments without synthesizing values for baseline artifacts.
- **Does**: Preserves message-microtick, output-count, and chance-baseline metadata
  for physical-routing and context-memory controls.
- **Does**: Preserves sequence length, dependency horizon, and token/context route
  coverage so temporal-memory results retain their physical communication budget.
- **Does**: Preserves the declared broadcast gain so global-workspace and purely
  dendritic routing controls remain distinguishable.
- **Does**: Preserves a benchmark's common learning-rate scale so non-finite
  stability retries remain visibly distinct from default-rate controls.
- **Does**: Preserves the deterministic-execution flag from benchmark artifacts.
- **Does**: Preserves an explicit process-global branch-RNG match flag; deterministic
  kernels alone do not prove stochastic lifecycle branches received the same stream.
- **Does**: Preserves intervention, lesion, topology, and turnover fields for matched
  recovery artifacts.
- **Rationale**: Curriculum transitions and architecture comparisons should be
  based on persisted measurements while keeping the polling payload bounded.

### `Laboratory.launch`
- **Does**: Creates an immutable task-aware manifest, organism lineage ID, and phase-zero
  record, then starts the corpus trainer with a GPU UUID.
- **Rationale**: UUID pinning avoids the host's conflicting CUDA and `nvidia-smi` indices.

### `ContinueSpec` / `Laboratory.continue_run`
- **Does**: Advances a stopped checkpoint into a new topology/lifecycle phase in the
  same run directory and on the same organism lineage.
- **Does**: Converts additional updates to an absolute target, appends a phase-boundary
  metric, and rejects continuation while the organism is already running.
- **Does**: Records the resolved topology profile in the manifest, phase history,
  append-only metric, and trainer command.
- **Rationale**: Structural warm-up, adaptive pruning, and lifecycle pressure must be
  phases of one organism rather than separately initialized comparison runs.

### `EvaluateSpec` / `Laboratory.evaluate_run`
- **Does**: Starts read-only held-out evaluation from a stopped checkpoint, optionally
  including the electrical-memory horizon, without appending a phase or optimizer step.
- **Rationale**: Causal state/topology diagnostics can be refreshed after code changes
  without altering the organism being measured.

### `Laboratory.stop_run`
- **Does**: Sends SIGTERM so the trainer completes its current update and checkpoints.

### Run status
- **Does**: Marks an ended run `failed` when its latest loss or rolling loss is
  non-finite instead of presenting the last checkpoint as ordinary completion.
- **Does**: Marks a run failed when the trainer persisted an explicit failure record,
  including failures before update one, and returns that bounded diagnostic.
- **Does**: Scopes terminal failure status to the current phase so an explicitly
  continued checkpoint is not permanently labeled by an earlier failed phase.

### `Laboratory._pid_alive`
- **Does**: Treats Linux zombie processes as stopped before falling back to a
  portable signal probe.
- **Rationale**: A failed CUDA worker can remain in `/proc` until reaped; its PID
  must not keep a failed experiment labeled as running.

## Contracts

| Dependent | Expects | Breaking changes |
|-----------|---------|------------------|
| `server.py` | Snapshot, metrics, launch, continuation, evaluation, and stop methods are synchronous | Method signatures |
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
