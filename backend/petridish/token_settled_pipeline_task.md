# token_settled_pipeline_task.py

## Purpose

Combines the two physical delays isolated by prior ESN controls: context must settle
before the stream, and each bit target is read two token clocks after input.

## Components

### `token_settled_pipeline_task`
- **Does**: Presents a rule, two settling clocks, four decorrelated bits, and two
  flush clocks.
- **Does**: Scores four rule-transformed bits at a fixed two-token output latency.
- **Interacts with**: `SequenceExperiment` and `benchmark_sequences.py`.
- **Rationale**: This is a causal spatial pipeline, not instantaneous global
  communication and not a period-two sequence shortcut.
- **Rationale**: The first two output positions exhaust every delayed/current-bit
  pair; the final two use constant clocks.

## Contracts

| Dependent | Expects | Breaking changes |
|-----------|---------|------------------|
| `benchmark_sequences.py` | Task key selects production distributed I/O | Changing `key` |
| Tests | Two settle clocks and two flush clocks surround four bits | Token layout |
| Scientific comparison | Every target is delayed exactly two token positions | Alignment |
