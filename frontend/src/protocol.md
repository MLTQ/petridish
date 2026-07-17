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
- **Rationale**: MNIST snapshots expose seed, sensing, development, and readout
  micro-steps independently from optimizer updates.

### `ExperimentCommand`
- **Does**: Restricts outbound control messages to supported interventions.
- **Interacts with**: Backend `ExperimentRuntime.handle_command`.

### `ServerMessage`
- **Does**: Discriminates snapshots from command errors.
- **Interacts with**: `ExperimentSocket`.

## Contracts

| Dependent | Expects | Breaking changes |
|-----------|---------|------------------|
| `socket.ts` | Every inbound payload has a discriminating `type` | Message envelope changes |
| `renderer.ts` | Edge arrays are aligned and cells match width × height | Field or edge shape |
| `main.ts` | `task.kind` safely narrows XOR versus MNIST fields | Union/discriminator changes |
