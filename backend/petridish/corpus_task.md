# corpus_task.py

## Purpose

Provides the online-backed character corpus experiment without adding a heavyweight
dataset dependency. Tiny Shakespeare is downloaded from Karpathy's `char-rnn` source
once and cached under `data/tinyshakespeare/`.

## Components

### `load_tiny_shakespeare_task`
- **Does**: Downloads/caches the corpus, builds a deterministic character vocabulary,
  splits 90/10 by position, and returns train/validation chunk generators.
- **Does**: Supplies prompt encode/decode functions and corpus metadata.
- **Rationale**: Character tokenization keeps the vocabulary small enough to map onto
  physical ports while allowing arbitrary-length interactive prompting.

### `_chunk_generator`
- **Does**: Samples aligned context/next-character windows using the experiment's
  deterministic `torch.Generator`.

## Contracts

| Dependent | Expects | Breaking changes |
|-----------|---------|------------------|
| `sequence_tasks.py` | Lazy resolution avoids downloading until selected | Eager import/download |
| `sequence_experiment.py` | `encode`/`decode` support prompt generation | Token mapping changes |
| Runtime | Cache path is workspace-relative and survives restarts | Cache location |

The source corpus is a roughly 1 MB subset of Shakespeare used by `char-rnn`; the
works are public domain in the United States.
