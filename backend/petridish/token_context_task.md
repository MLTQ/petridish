# token_context_task.py

## Purpose

Defines the first distributed-token stepping stone that requires persistent cellular
memory rather than direct sensory routing. Every supervised target is an XOR of a
context token and a later query token.

## Components

### `token_context_task`
- **Does**: Enumerates all four context/query bit combinations at identical positions.
- **Does**: Supervises only the second position with one of two balanced targets.
- **Interacts with**: `SequenceExperiment` and `benchmark_sequences.py`.
- **Rationale**: Neither position, class frequency, context alone, nor query alone
  exceeds 50%; success requires retaining and composing both distributed inputs.

## Contracts

| Dependent | Expects | Breaking changes |
|-----------|---------|------------------|
| `benchmark_sequences.py` | Task key selects production distributed I/O | Changing `key` |
| Tests | All four bit pairs and both targets are equally represented | Generator balance |
| Scientific comparison | Only the second position contributes loss | Loss-mask semantics |
