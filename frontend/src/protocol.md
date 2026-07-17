# protocol.ts

## Purpose

Types every server snapshot and viewer command at the Python/TypeScript boundary.

## Components

### `ExperimentSnapshot`
- **Does**: Describes aligned cell, edge, event, task, and metric payloads.
- **Interacts with**: Backend `build_snapshot`, renderer, charts, and main UI.

### `XorTaskSnapshot` / `MnistTaskSnapshot`
- **Does**: Provide discriminated task-specific status inside one common field
  and graph envelope.
- **Interacts with**: Dynamic task panel in `main.ts`.
- **Rationale**: MNIST snapshots expose input, forward, real backward-credit,
  and structural phases independently from optimizer updates.
- **Does**: MNIST reports remaining structure warm-up updates; common metrics
  distinguish null versus measured synapse movement.
- **Does**: MNIST reports optimizer phase, curriculum progress, structure unlock
  reason, attention entropy, effective capacity, and real graph reachability.

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
| `renderer.ts` | Dense XOR rows use null indices; sparse MNIST rows map through
  `field.indices`; edges expose measured flow and credit | Field or edge shape |
| `main.ts` | `task.kind` safely narrows XOR versus MNIST fields | Union/discriminator changes |
| Hyperparameter UI | Configuration may be absent for non-MNIST experiments | Making it unconditional |
