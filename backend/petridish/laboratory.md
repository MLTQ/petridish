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
- **Does**: Records the default global gradient ceiling so later phase-local
  interventions have an explicit manifest baseline.
- **Does**: Records a bounded broadcast-workspace gain independently from the
  microtick budget; zero is the hard physical-routing ablation.
- **Does**: Records a 64–2,048 power-of-two token-vocabulary curriculum while
  preserving the same 64-port physical population code.
- **Does**: Records legacy wordpiece versus byte-complete TinyStories tokenization;
  byte mode requires exactly 256 tokens and has no aggregate unknown class.
- **Does**: Records continuous organism experience versus the random cold-window
  control and passes that immutable mode to the trainer.
- **Does**: Records and bounds electrical retention independently from structure;
  the default 0.9 is a homeostatic-relaxation intervention, while 1.0 is the exact
  indefinite-state control.
- **Does**: Records one to 512 round-robin persistent state lanes independently
  from tensor batch size.
- **Does**: Records a zero-by-default random-offset auxiliary weight; a nonzero
  value adds disposable shared-rule training contexts without creating or replacing
  persistent organism lanes.
- **Does**: Records fixed, adaptive, or prune-only topology independently from
  lifecycle. Prune-only retains the lineage and forbids replacement growth.
- **Does**: Derives the compatibility lifecycle boolean from the resolved profile so
  the manifest and trainer command cannot disagree.

### `Laboratory.snapshot`
- **Does**: Reports GPU telemetry, compute processes, capabilities, and run summaries.
- **Does**: Advertises checkpoint evaluation explicitly so a frontend loaded before
  the matching server restart cannot expose a nonfunctional mutation control.
- **Does**: Advertises repeated-shard continuation explicitly so a frontend/backend
  version mismatch cannot silently discard a curriculum request.
- **Does**: Advertises append-only state-lane expansion explicitly so an older
  backend can never misinterpret phase-diversity controls as ordinary continuation.
- **Does**: Publishes the 512-lane bound and phase-balancing capability so older
  servers cannot be offered unsupported full-coverage expansions by a newer frontend.
- **Does**: Advertises checkpointed state-lane domains separately so mixed replay/new
  curricula are shown only when the server preserves their sampler semantics.
- **Does**: Advertises explicit trajectory-lane audits so the frontend never assumes
  a mixed-domain organism was measured through whichever lane happened to be next.
- **Does**: Advertises exact checkpoint forking explicitly so the frontend cannot
  present paired counterfactual controls against an older copy-only backend.
- **Does**: Advertises same-phase resume separately from failed-phase retry so a
  deliberate audit stop can restart without inventing a curriculum boundary.
- **Does**: Advertises phase-local gradient-clip control so a newer frontend never
  submits an optimizer intervention to a server that would ignore it.
- **Does**: Advertises the random-offset auxiliary separately so an older trainer
  cannot silently ignore a trajectory-generalization intervention.
- **Interacts with**: `/api/lab` in `server.py` and `lab.ts`.

### `Laboratory.metrics`
- **Does**: Returns at most 2,000 valid records from the tail of a run's JSONL log.
- **Rationale**: Browser polling cannot force unbounded file reads or payloads.
- **Does**: Keeps optimizer, held-out/generation, and scientific diagnostic record
  types distinct so the frontend never infers biology from loss records.
- **Does**: Repairs a lagging manifest's displayed current phase, shard, and lane
  metadata from authoritative training metrics in memory. Discovery never rewrites
  the checkpoint or manifest; the next controlled continuation persists the repair.

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
- **Does**: Optionally changes only the TinyStories experience distribution to a
  deterministic repeated token shard. The checkpoint organism, graph, optimizer,
  corpus cursor, RNG state, and electrical memory remain the same lineage.
- **Does**: Optionally appends up to 512 independently phased persistent
  experience lanes. Existing lane positions and electrical states may not shrink or
  move; only new cold lanes are allocated by the checkpoint's continuing RNG stream
  and balanced toward underrepresented cursor phases.
- **Does**: When shard breadth and lane count grow together, preserves each old
  lane's checkpointed stream domain and gives only appended lanes the new domain.
  Shrinking below any preserved domain is rejected.
- **Does**: Rejects a broader shard unless the same continuation appends lanes; a
  global shard change without new lanes would silently remap or fail to expose every
  saved trajectory. Full-stream domains likewise cannot contract to a prefix.
- **Does**: Recovers the authoritative phase index, repeated-shard size, current
  lane count, and gradient ceiling from the latest training metric when an older
  manifest lags a valid checkpoint, preventing stale orchestration metadata from
  relabeling continuation.
- **Does**: Captures cumulative cell and edge turnover at each phase boundary so
  later diagnostics can report phase-local change separately from lifetime totals.
- **Does**: Optionally changes only the restored configuration's global gradient
  norm ceiling, records the resolved value in manifest/phase/metrics, and omits the
  trainer flag when continuation must preserve the checkpoint-owned value.
- **Does**: Optionally changes a bounded random-offset auxiliary weight, recording
  the resolved value in manifest, phase history, phase metric, and trainer command.
  Blank preserves the checkpoint value; the transient auxiliary context contributes
  shared gradient but cannot replace any saved state lane or cursor.
- **Rationale**: Structural warm-up, adaptive pruning, and lifecycle pressure must be
  phases of one organism rather than separately initialized comparison runs.

### `ForkSpec` / `Laboratory.fork_run`
- **Does**: Copies a stopped atomic checkpoint and its metric history into a new,
  separately named counterfactual branch without constructing a model or organism.
- **Does**: Preserves the organism ID and records the parent run, checkpoint update,
  SHA-256 checkpoint identity, root run, and branch depth in the branch manifest.
- **Does**: Publishes the branch directory atomically and rejects running sources,
  missing lineage IDs, missing checkpoints, and existing destination names.
- **Rationale**: Two GPUs can compare fixed, pruning, or lifecycle interventions from
  byte-identical evolved structure without overwriting the sole original lineage.

### `RetrySpec` / `Laboratory.retry_run`
- **Does**: Restarts only a failed, stopped plasticity phase from the `latest.pt` in
  that same run directory, without appending a phase or constructing a model.
- **Does**: Fails closed unless the persisted command requires resume-plasticity,
  names the manifest's immutable organism ID, points to the same resolved checkpoint
  directory, and has no fresh-run or evaluation flags.
- **Does**: Fingerprints the checkpoint before launch and appends a retry boundary so
  the recovered execution supersedes the earlier failure without deleting its audit
  record.
- **Rationale**: A Python worker may restart; the cells, graph, electrical lanes,
  optimizer, sampler, RNG, curriculum, and phase remain checkpoint-owned organism
  state.

### `ResumeSpec` / `Laboratory.resume_run`
- **Does**: Restarts a deliberately stopped phase from its fingerprinted checkpoint
  and recorded remaining target without appending or modifying a phase.
- **Does**: Reconstructs the training command from immutable lineage and phase
  metadata even when a read-only audit was the most recent process command.
- **Does**: Rejects completed phases, active workers, missing lineage/checkpoints,
  unknown GPUs, and unrecovered failures; failures remain restricted to retry.
- **Rationale**: Interim causal audits must pause only the worker, never transform
  one persistent organism experiment into a nominally new phase.

### `EvaluateSpec` / `Laboratory.evaluate_run`
- **Does**: Starts read-only held-out evaluation from a stopped checkpoint, optionally
  including the electrical-memory horizon, without appending a phase or optimizer step.
- **Does**: Uses sixteen fixed-seed batches for a larger checkpoint-comparable causal
  audit than the four-batch scheduled training diagnostic.
- **Does**: Accepts explicit validation, warm active-training-shard, or independent
  cold-context splits and forwards them to the same read-only checkpoint evaluator.
- **Does**: Also exposes the exact next saved trajectory as a separately recorded
  audit, preserving the distinction between aligned competence and random-offset
  shard competence.
- **Does**: Can select one explicit saved trajectory lane and retains the latest audit
  for every measured lane in run discovery while keeping the legacy newest audit.
- **Does**: Discovers active-shard `training_audit` records separately from
  `held_out` validation records so higher memorization accuracy cannot replace the
  laboratory's generalization result.
- **Does**: Discovers `random_context_audit` separately as a cold topology/parameter
  probe; it never replaces the authoritative warm trajectory or validation records.
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
- **Does**: Treats a same-phase retry as an execution boundary, preserving the old
  failure in JSONL while reporting only failures that occurred after the latest retry.

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
