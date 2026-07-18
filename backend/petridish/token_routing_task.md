# token_routing_task.py

## Purpose

Defines the smallest distributed-token task that can falsify physical credit
assignment. It asks the 68×68 organism to distinguish balanced batch rows after one
token, so position embeddings and class bias cannot solve the task.

## Components

### `token_routing_task`
- **Does**: Creates `mapping_size` input tokens with one unique target each.
- **Does**: Repeats a balanced deterministic batch for training and held-out checks.
- **Interacts with**: `SequenceExperiment` and `benchmark_sequences.py`.
- **Rationale**: If four microticks cannot traverse the graph but twelve can, this
  task measures that causal difference without corpus frequency or long-context noise.

## Contracts

| Dependent | Expects | Breaking changes |
|-----------|---------|------------------|
| `benchmark_sequences.py` | Task key selects distributed 64-port I/O | Changing `key` |
| Tests | Target `i` is `i + mapping_size` and batches are balanced | Mapping semantics |
| Scientific comparison | No row-specific positional or frequency shortcut | Unequal target frequency |
