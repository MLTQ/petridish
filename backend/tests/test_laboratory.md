# test_laboratory.py

## Purpose

Protects the remote laboratory's filesystem, bounded telemetry, geometry, and
control-enable contracts without requiring NVIDIA hardware or launching trainers.

## Components

### Metrics and discovery tests
- **Does**: Verify partial JSONL records are ignored, payload size is bounded, and
  latest training/held-out measurements remain distinct.
- **Does**: Keeps the latest scientific diagnostic separate from optimizer and
  held-out/generation records.
- **Does**: Preserves physical-routing microticks, output count, and chance accuracy
  from benchmark artifacts.
- **Does**: Preserves sequence length, dependency horizon, and full-context route reach.
- **Does**: Preserves final graph-silencing and endpoint-rotation causal deltas.
- **Does**: Preserves held-out split provenance and free-running grammar metrics.
- **Does**: Preserves a zero broadcast gain rather than treating it as missing data.
- **Does**: Preserves the benchmark optimizer scale used by stability retries.
- **Does**: Preserves the benchmark's absolute-position signal intervention.
- **Does**: Preserves autoregressive feedback probability and warm-up provenance.
- **Does**: Preserves batch size, precision mode, and CUDA allocator provenance for
  memory-bounded benchmark cohorts.
- **Does**: Keeps a failed benchmark visible before its first checkpoint and
  preserves its bounded exception type/message.

### Safety tests
- **Does**: Reject path traversal and process launch when control is disabled.
- **Does**: Proves a checkpoint fork copies bytes rather than sharing a mutable
  inode, preserves lineage and phase history, fingerprints its exact parent, leaves
  the source manifest unchanged, marks the branch counterfactual, and refuses to
  branch a running organism.
- **Does**: Proves a failed-phase retry launches the exact persisted continuation
  command, preserves checkpoint bytes, organism ID, and phase history, fingerprints
  the checkpoint, and supersedes only the old failure in current status.
- **Does**: Rejects any retry command that can construct a fresh organism, including
  `--no-resume`, even when a checkpoint file happens to exist.
- **Does**: Requires the server snapshot to advertise checkpoint evaluation before
  the frontend exposes that action.
- **Does**: Proves a deliberately stopped phase resumes to its original target from
  a fingerprinted unchanged checkpoint, even after evaluation overwrote the last
  process command, without appending a phase or resupplying shard/lane mutations.
- **Does**: Verify a defunct Linux trainer is not reported as a live run merely
  because its PID still exists.

### Geometry test
- **Does**: Keeps 68×68 mandatory for both 66-port and 64-port single-column
  corpus layouts.

### Lifecycle profile tests
- **Does**: Preserve an explicit named policy in trainer commands and map the legacy
  enabled boolean to the original baseline policy.
- **Does**: Treat a named non-off profile as authoritative over the compatibility
  boolean so the command cannot silently disable the recorded intervention.
- **Rationale**: Commands and manifests must describe the homeostasis intervention
  that actually ran.

### Topology and failure status tests
- **Does**: Preserves an explicit fixed-connectome trainer flag independently from
  lifecycle and reports ended non-finite runs as failed.
- **Does**: Preserves and bounds the common learning-rate scale used by long-run
  stability controls.
- **Does**: Preserves a zero corpus broadcast gain in the trainer command and
  rejects gains above the bounded workspace range.
- **Does**: Preserves a power-of-two TinyStories vocabulary curriculum and rejects
  unsupported intermediate sizes.
- **Does**: Preserves byte-complete versus wordpiece tokenization and requires the
  byte profile's exact 256-token vocabulary.
- **Does**: Preserves continuous versus windowed experience in the trainer command
  and rejects unlabeled reset modes.
- **Does**: Preserves a bounded electrical-retention coefficient separately from
  experience mode and rejects values above one.
- **Does**: Preserves fixed, adaptive, and prune-only topology profiles in commands,
  manifests, and continuation phase records while rejecting unknown policies.
- **Does**: Preserves one to 512 round-robin persistent state lanes independently
  from tensor batch and rejects larger banks.
- **Does**: Reports a persisted pre-checkpoint trainer failure and its bounded detail
  as failed rather than stopped.
- **Does**: Proves continuation reuses one run/organism ID, advances from the measured
  checkpoint update, records a phase boundary, and invokes resume-plasticity without
  the fresh-run flag.
- **Does**: Proves every new continuation phase fingerprints the exact checkpoint it
  resumes and publishes the same SHA-256 identity in its result, manifest, phase
  history, and append-only phase record.
- **Does**: Proves canonical/counterfactual phase role is validated and persisted in
  responses, manifests, phase history, and append-only phase records.
- **Does**: Proves a repeated-shard curriculum is recorded in the same lineage's
  command, manifest, phase boundary, and append-only metric.
- **Does**: Proves append-only lane expansion reaches the command, manifest, phase
  boundary, and metric while shrink requests remain invalid.
- **Does**: Requires every corpus-breadth increase either to append lanes or explicitly
  expand carried domains, records that distinction through command/phase/manifest,
  and rejects bounded or full-stream shrinkage before launching a trainer.
- **Does**: Proves continuation repairs a lagging manifest from the latest checkpoint
  metric's phase, repeated-shard size, and lane count rather than relabeling the
  preserved organism with stale orchestration metadata.
- **Does**: Proves continuation snapshots edge and cell turnover counters for
  phase-local consolidation diagnostics.
- **Does**: Proves a bounded phase-local gradient ceiling reaches the trainer,
  manifest, phase history, and append-only metric while out-of-range values fail
  before any process launch.
- **Does**: Proves a bounded phase-local dendrite growth cap reaches the trainer,
  manifest, phase history, and append-only metric while out-of-range values fail
  before any process launch.
- **Does**: Proves bounded axon construction cost, endpoint reserve, and probation
  utility reach the trainer, manifest, phase history, and append-only metric while
  out-of-range values fail before process launch.
- **Does**: Proves a legacy nonzero disposable auxiliary setting is explicitly
  migrated to zero on continuation and recorded in command, manifest, phase history,
  and append-only metrics; any requested nonzero value fails before process launch.
- **Does**: Proves checkpoint evaluation invokes the read-only trainer path, retains
  lineage/phase metadata, and can request state horizons without a plasticity override.
- **Does**: Requires read-only audits to use sixteen fixed-seed batches rather than
  the smaller scheduled-training sample.
- **Does**: Verifies validation, warm training-trajectory, and cold independent-context
  splits reach the read-only trainer command and are advertised as separate
  capabilities.
- **Does**: Keeps active-shard causal audits distinct from held-out records so a high
  overfit result never overwrites validation accuracy in run discovery.
- **Does**: Keeps cold independent-context audits distinct from both warm active-shard
  trajectories and held-out records in run discovery.
- **Does**: Keeps full-corpus cold audits distinct from active-shard cold probes and
  advertises their TinyStories-only route independently.
- **Does**: Keeps exact-trajectory audits distinct from both validation and
  random-offset shard records.
- **Does**: Forwards an explicit saved trajectory lane to the read-only trainer,
  rejects it for validation, and retains the newest audit per measured lane.
- **Does**: Reconstructs a missing current phase for display when authoritative
  checkpoint training metrics are newer than an older manifest, without mutating
  either artifact during discovery.
- **Does**: Requires the server to advertise repeated-shard continuation before the
  frontend enables that phase control.
- **Does**: Requires the server to advertise state-lane expansion before the frontend
  enables the append-only lane control.
- **Does**: Requires the server to advertise persistent-state training while the two
  historical random-offset auxiliary capabilities remain disabled.
- **Does**: Requires the advertised 512-lane bound and phase-balanced expansion
  capability so newer phase-coverage controls remain safe against older servers.

## Contracts

| Dependent | Expects | Breaking changes |
|-----------|---------|------------------|
| `laboratory.py` | Safe run IDs, opt-in launch, bounded metrics | Relaxing validation |
| CI | Tests run without CUDA or `nvidia-smi` | Hardware-dependent setup |
