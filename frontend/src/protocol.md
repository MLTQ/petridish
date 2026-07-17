# protocol.ts

## Purpose

Types every server snapshot and viewer command at the Python/TypeScript boundary.

## Components

### `ExperimentSnapshot`
- **Does**: Describes aligned cell, edge, event, task, and metric payloads.
- **Interacts with**: Backend `build_snapshot`, renderer, charts, and main UI.

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

### `HyperparameterSnapshot`
- **Does**: Types one authoritative numeric slider definition and current value.
- **Interacts with**: Dynamic controls in `main.ts` and MNIST snapshots.

### `ExperimentCommand`
- **Does**: Restricts outbound messages to supported interventions and atomic
  MNIST configuration changes.
- **Interacts with**: Backend `ExperimentRuntime.handle_command`.

### `ServerMessage`
- **Does**: Discriminates snapshots from command errors.
- **Interacts with**: `ExperimentSocket`.

## Contracts

| Dependent | Expects | Breaking changes |
|-----------|---------|------------------|
| `socket.ts` | Every inbound payload has a discriminating `type` | Message envelope changes |
| `renderer.ts` | Sparse MNIST rows map through `field.indices`; edges expose
  measured flow and credit | Field or edge shape |
| `main.ts` | Every snapshot is the MNIST experiment | Experiment discriminator changes |
| Hyperparameter UI | Configuration is present on live snapshots | Making it optional at runtime |
