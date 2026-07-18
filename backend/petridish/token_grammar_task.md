# token_grammar_task.py

## Purpose

Defines the first stepping stone that is literally autoregressive next-token
prediction rather than recall or transformed-label output.

## Components

### `token_grammar_task`
- **Does**: Enumerates two persistent rules and all 16 ordered pairs of four
  symbols, then generates a second-order modular token stream.
- **Does**: Predicts the actual next stream token at seven consecutive positions.
- **Interacts with**: `SequenceExperiment` and `benchmark_sequences.py`.
- **Rationale**: At every supervised position, target frequency is uniform and the
  rule, current token, or previous token alone remains at the 25% baseline.
- **Rationale**: The final prediction targets a token never presented as input,
  ruling out a shifted-copy interpretation.

### `_sample_rows`
- **Does**: Uses complete shuffled 32-state cycles so unit tests can inspect exact
  balance while memory-bounded GPU batches still cover the full language over time.

## Contracts

| Dependent | Expects | Breaking changes |
|-----------|---------|------------------|
| `benchmark_sequences.py` | Task key selects production distributed I/O | Changing `key` |
| Tests | Full batches contain every rule/seed state exactly once | Sampling semantics |
| Scientific ladder | Seven aligned targets are true next tokens | Target alignment |
