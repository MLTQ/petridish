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
- **Does**: Preserves a zero broadcast gain rather than treating it as missing data.
- **Does**: Preserves the benchmark optimizer scale used by stability retries.
- **Does**: Keeps a failed benchmark visible before its first checkpoint and
  preserves its bounded exception type/message.

### Safety tests
- **Does**: Reject path traversal and process launch when control is disabled.
- **Does**: Requires the server snapshot to advertise checkpoint evaluation before
  the frontend exposes that action.
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
- **Does**: Preserves one to sixteen round-robin persistent state lanes independently
  from tensor batch and rejects larger banks.
- **Does**: Reports a persisted pre-checkpoint trainer failure and its bounded detail
  as failed rather than stopped.
- **Does**: Proves continuation reuses one run/organism ID, advances from the measured
  checkpoint update, records a phase boundary, and invokes resume-plasticity without
  the fresh-run flag.
- **Does**: Proves a repeated-shard curriculum is recorded in the same lineage's
  command, manifest, phase boundary, and append-only metric.
- **Does**: Proves checkpoint evaluation invokes the read-only trainer path, retains
  lineage/phase metadata, and can request state horizons without a plasticity override.
- **Does**: Requires read-only audits to use sixteen fixed-seed batches rather than
  the smaller scheduled-training sample.

## Contracts

| Dependent | Expects | Breaking changes |
|-----------|---------|------------------|
| `laboratory.py` | Safe run IDs, opt-in launch, bounded metrics | Relaxing validation |
| CI | Tests run without CUDA or `nvidia-smi` | Hardware-dependent setup |
