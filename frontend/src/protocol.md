# protocol.ts

## Purpose

Types every server snapshot and viewer command at the Python/TypeScript boundary.

## Components

### `ExperimentSnapshot`
- **Does**: Describes aligned cell, edge, event, task, and metric payloads.
- **Interacts with**: Backend `build_snapshot`, renderer, charts, and main UI.
- **Does**: Carries runtime visualization/headless mode, running state, last
  optimizer latency, measured update/sequence throughput, and the active compute
  phase with numeric progress.
- **Does**: Carries a monotonic control revision used to reject pre-command
  snapshots that arrive after an immediate acknowledgement.
- **Does**: Carries the saved-organism catalog and current checkpoint identity.

### `MnistTaskSnapshot`
- **Does**: Provides the sole task status inside the field and graph envelope.
- **Interacts with**: Training and lifecycle panels in `main.ts`.
- **Rationale**: MNIST snapshots expose input, forward, real backward-credit,
  and structural phases independently from optimizer updates.
- **Does**: MNIST reports remaining structure warm-up updates; common metrics
  distinguish null versus measured synapse movement.
- **Does**: MNIST reports optimizer phase, curriculum progress, structure unlock
  reason, attention entropy, effective capacity, and real graph reachability.
- **Does**: Reports lifecycle activation, inherited turnover, and classified death causes.

### `SequenceTaskSnapshot`
- **Does**: Publishes token streams, aligned targets and predictions, current
  position, perplexity, accuracy, recall curriculum size, and shared lifecycle state.
- **Rationale**: Recall and language diagnostics stay explicit without weakening
  the MNIST image and curriculum contract.
- **Does**: Corpus tasks additionally publish dataset size/source, context length,
  tokenizer, prompt text, generated suffix, and the next greedy-token diagnostic.
- **Does**: Publishes stun/recovery counts and excitotoxic death separately from
  ordinary turnover.

### `HyperparameterSnapshot`
- **Does**: Types one authoritative numeric slider definition and current value.
- **Interacts with**: Dynamic controls in `main.ts` and MNIST snapshots.
- **Does**: Optionally carries discrete numeric choices for controls such as
  power-of-two field size.

### `ExperimentCommand`
- **Does**: Restricts outbound messages to experiment selection, supported
  lifecycle actions, and atomic configuration changes; lesion commands are absent.
- **Interacts with**: Backend `ExperimentRuntime.handle_command`.
- **Does**: Includes corpus prompt replacement and single-token generation.
- **Does**: Includes explicit entry to and exit from headless trace-free sequence training.
- **Does**: Includes loading a trusted saved organism by opaque identifier.

### `ServerMessage`
- **Does**: Discriminates snapshots from command errors.
- **Interacts with**: `ExperimentSocket`.

## Contracts

| Dependent | Expects | Breaking changes |
|-----------|---------|------------------|
| `socket.ts` | Every inbound payload has a discriminating `type` | Message envelope changes |
| `renderer.ts` | Sparse MNIST rows map through `field.indices`; edges expose
  measured flow and credit | Field or edge shape |
| `main.ts` | Task `kind` discriminates image and sequence payloads | Union changes |
| Hyperparameter UI | Configuration is present on live snapshots | Making it optional at runtime |
