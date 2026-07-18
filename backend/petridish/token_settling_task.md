# token_settling_task.py

## Purpose

Tests whether local stream failure comes from context-propagation latency rather
than forgetting or output latency. The rule receives two unsupervised clock periods
before same-clock token prediction begins.

## Components

### `token_settling_task`
- **Does**: Presents one copy/invert rule, two constant clocks, then four bit inputs.
- **Does**: Supervises each bit position immediately after its local microticks.
- **Interacts with**: `SequenceExperiment` and `benchmark_sequences.py`.
- **Rationale**: This separates time needed to establish a persistent field context
  from time needed to retain each current input for a delayed output pipeline.
- **Rationale**: Every supervised position balances rule, input, and target, so the
  clocks introduce computation time but no label information.

## Notes

The four bit patterns repeat with period two. This control can therefore expose a
two-token physical lag (perfect late positions) but cannot by itself prove that late
predictions use the contemporaneous bit. `token_settled_pipeline_task.py` removes
that ambiguity with decorrelated patterns and explicit delayed targets.

## Contracts

| Dependent | Expects | Breaking changes |
|-----------|---------|------------------|
| `benchmark_sequences.py` | Task key selects production distributed I/O | Changing `key` |
| Tests | Exactly two masked clock positions follow the rule | Clock placement |
| Scientific comparison | Targets remain aligned with current bit inputs | Target alignment |
