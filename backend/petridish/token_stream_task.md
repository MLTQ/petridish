# token_stream_task.py

## Purpose

Defines a repeated contextual-prediction stepping stone after delayed copy and
two-token XOR. One retained rule must govern four later token predictions.

## Components

### `token_stream_task`
- **Does**: Enumerates eight balanced copy/invert streams over constant, alternating,
  and inverse-alternating four-bit patterns.
- **Does**: Masks the rule position and supervises every later position with the
  rule/input XOR target.
- **Interacts with**: `SequenceExperiment` and `benchmark_sequences.py`.
- **Rationale**: At every prediction position, rule alone, current bit alone,
  position, and target frequency remain at 50%; success requires persistent local
  context plus repeated nonlinear composition.

## Contracts

| Dependent | Expects | Breaking changes |
|-----------|---------|------------------|
| `benchmark_sequences.py` | Task key selects production distributed I/O | Changing `key` |
| Tests | Every prediction position contains all rule/input/target cases | Pattern balance |
| Scientific ladder | Four targets reuse one retained rule | Mask or sequence semantics |
