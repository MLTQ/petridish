# token_pipeline_task.py

## Purpose

Defines a latency-aligned token-prediction control for a physically extended
cellular network. Targets emerge two token clocks after their corresponding input.

## Components

### `token_pipeline_task`
- **Does**: Presents one copy/invert rule, four bit inputs, and two clock tokens.
- **Does**: Supervises four transformed outputs with a fixed two-token delay.
- **Interacts with**: `SequenceExperiment` and `benchmark_sequences.py`.
- **Rationale**: A left-to-right organism should not be forced to communicate
  instantaneously. Fixed latency permits a causal pipeline without global broadcast.
- **Rationale**: The four bit patterns exhaust every delayed/current-bit pair at
  the first two output clocks, preventing the contemporaneous input from replacing
  the delayed bit; the final clocks are constant.

## Contracts

| Dependent | Expects | Breaking changes |
|-----------|---------|------------------|
| `benchmark_sequences.py` | Task key selects production distributed I/O | Changing `key` |
| Tests | Targets are shifted exactly two token positions | Alignment semantics |
| Scientific comparison | Rule, current input, and target remain balanced | Pattern balance |
