# token_memory_task.py

## Purpose

Defines a distributed-token delayed-copy control between direct routing and
contextual composition. It isolates whether cellular state can preserve one bit
across a token boundary.

## Components

### `token_memory_task`
- **Does**: Alternates two equally frequent context tokens followed by one identical
  recall token and supervises the corresponding target only at the second position.
- **Interacts with**: `SequenceExperiment` and `benchmark_sequences.py`.
- **Rationale**: Position, current input, and target frequency are identical across
  classes; exceeding 50% requires memory of the first distributed token.

## Contracts

| Dependent | Expects | Breaking changes |
|-----------|---------|------------------|
| `benchmark_sequences.py` | Task key selects production distributed I/O | Changing `key` |
| Tests | Recall token is constant and targets are balanced | Generator semantics |
| Scientific ladder | Memory is required but XOR composition is not | Adding query variation |
