# test_protocol.py

## Purpose

Guards the live browser boundary independently of server timing.

## Components

### `test_snapshot_arrays_are_aligned_and_json_safe`
- **Does**: Verifies field size, channel advertisement, aligned edge arrays,
  experiment discriminator, metrics, and XOR task phase.
- **Interacts with**: `build_snapshot` and frontend protocol types.

## Contracts

| Dependent | Expects | Breaking changes |
|-----------|---------|------------------|
| Frontend | Snapshot field/edge/task structure remains aligned | Protocol schema |
