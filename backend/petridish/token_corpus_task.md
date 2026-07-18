# token_corpus_task.py

## Purpose

Provides a bounded token-level TinyStories experiment without requiring a tokenizer
runtime or the multi-gigabyte training split.

## Components

### `load_tiny_stories_task`
- **Does**: Downloads/caches the 22.5 MB GPT-4 validation subset, learns a deterministic
  2,048-entry leading-space wordpiece vocabulary, and returns fixed train/validation
  token streams.
- **Rationale**: The subset is large enough to test token prediction while remaining
  practical for one-machine experiments and repository-local caching.

### `build_token_task`
- **Does**: Builds the same task from supplied text for tests and offline experiments.
- **Does**: Supports the legacy bounded leading-space wordpiece curriculum and a
  256-symbol UTF-8 byte profile with complete coverage and no aggregate unknown class.
- **Rationale**: A small wordpiece vocabulary can make `<unk>` the modal target;
  byte-complete experiments trade longer sequences for honest, always-decodable
  language targets.
- **Does**: Measures exact held-out unigram accuracy and a train-fitted bigram
  lookup baseline with vectorized counts, plus add-one-smoothed validation
  cross-entropy/perplexity for both distributions.
- **Rationale**: Leading spaces remain part of pieces, so concatenating decoded
  predictions produces readable text without a language-specific detokenizer.
- **Does**: Retains the split token tensors as explicit contiguous streams for
  state-carrying truncated-backpropagation experiments.
- **Does**: Can expose a deterministic prefix of the training stream as a repeated
  experience shard while retaining the complete held-out validation stream.
- **Rationale**: A persistent organism can be continued onto a learnable bounded
  curriculum without replacing its cells, connectome, weights, optimizer, or
  electrical state. Existing stream cursors are interpreted modulo the new shard;
  they are not re-randomized.

### `_next_token_baselines`
- **Does**: Fits the most-common global token and one-step transition table on the
  training split, then evaluates both on the untouched validation stream.
- **Rationale**: Uniform vocabulary chance is not a meaningful language baseline;
  learned models must be compared with frequency and one-token context. Accuracy
  catches modal collapse; smoothed loss detects useful probability mass below top one.

## Contracts

| Dependent | Expects | Breaking changes |
|-----------|---------|------------------|
| `sequence_tasks.py` | Lazy loading avoids network work for other tasks | Eager download |
| `sequence_model.py` | Vocabulary may be larger than physical port banks | Restoring one-port-per-token assumptions |
| Runtime | Cache survives restarts under `data/tinystories/` | Cache location |
| Checkpoints | Tokenizer profile, exact vocabulary, and optional training-shard size select the cached task | Silent tokenizer or curriculum substitution |
