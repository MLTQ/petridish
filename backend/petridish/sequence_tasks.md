# sequence_tasks.py

## Purpose

Defines small generated sequence distributions that test capabilities on the
path from spatial classification to language modeling.

## Components

### `SequenceBatch`
- **Does**: Carries token inputs, aligned targets, and an explicit loss mask.

### `SequenceTask`
- **Does**: Names a distribution, vocabulary, sequence length, and deterministic
  batch generator.
- **Does**: Optionally supplies a distinct validation generator, text encoder and
  decoder, character/token counts, tokenizer metadata, source URL, and measured
  unigram/bigram validation baselines for cached corpus tasks.
- **Does**: Optionally exposes raw training/validation token streams and slices
  contiguous, wrapping windows from explicit per-lane positions.
- **Rationale**: Persistent organisms must receive adjacent experience without
  hiding mutable cursors inside the task or sampler.

### `associative_recall_batch`
- **Does**: Generates one to three random key/value bindings and a delayed query
  whose answer is supervised only at the final position; unused slots are neutral.
- **Rationale**: Success requires content-addressed memory rather than local
  next-token frequency.

### `tiny_language_batch`
- **Does**: Generates a compositional grammar and supervises every next token.
- **Rationale**: The verb is an XOR-like function of two earlier tokens, so a
  one-token Markov model cannot solve the deterministic portion.

### `resolve_sequence_task`
- **Does**: Resolves synthetic tasks immediately and lazily loads Tiny Shakespeare
  or token-level TinyStories only when selected, avoiding network or disk work for unrelated experiments.
- **Interacts with**: `corpus_task.py` and `SequenceExperiment`.

## Contracts

| Dependent | Expects | Breaking changes |
|-----------|---------|------------------|
| `sequence_experiment.py` | Masked targets use `-100` outside supervision | Mask semantics |
| Frontend | Vocabulary indices remain stable within a run | Token reordering |
| Benchmarks | Generators are deterministic for a supplied torch generator | RNG source |
| Continuous trainer | Returned next position begins at the previous window's target token | Stream stride or wrapping |
