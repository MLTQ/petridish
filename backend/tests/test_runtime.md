# test_runtime.py

## Purpose

Protects the asynchronous control plane independently of dataset loading and model
throughput.

## Components

### `test_pause_acknowledges_without_waiting_for_scientific_lock`
- **Does**: Verifies Pause updates authoritative control state and revision using
  the cached scientific snapshot without acquiring the long-lived mutation lock.
- **Does**: Ensures acknowledgement projection does not mutate the cached payload.

### `test_sequence_interruption_uses_only_safe_boundaries`
- **Does**: Allows interruption during forward, before backward, or before the
  optimizer, while requiring post-optimizer credit/lifecycle bookkeeping to finish.

## Contracts

| Dependent | Expects | Breaking changes |
|-----------|---------|------------------|
| Runtime control handlers | Pause acknowledgement is lock-independent | Moving Play/Pause behind `_lock` |
| Sequence optimizer | Partial batches never mutate weights | Interrupting after optimizer entry |
