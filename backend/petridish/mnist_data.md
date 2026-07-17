# mnist_data.py

## Purpose

Owns canonical MNIST acquisition and deterministic loader construction without
mixing dataset policy into cellular dynamics or training.

## Components

### `default_data_root`
- **Does**: Resolves the repository-local ignored `data/` directory.
- **Interacts with**: `load_mnist_datasets` and `.gitignore`.

### `load_mnist_datasets`
- **Does**: Downloads and returns the standard 60,000/10,000 MNIST splits.
- **Interacts with**: TorchVision `MNIST` and `MnistExperiment`.

### `make_loader`
- **Does**: Creates deterministic, single-process live-training loaders.
- **Interacts with**: Training and evaluation iterators in `mnist_experiment.py`.
- **Rationale**: Worker-free loading behaves consistently in desktop app and test
  processes; GPU transfers use pinned memory when useful.

## Contracts

| Dependent | Expects | Breaking changes |
|-----------|---------|------------------|
| `mnist_experiment.py` | Samples are `[1,28,28]` tensors with integer labels | Dataset transform or shape |
| Reproducibility | Same seed yields the same shuffled training order | Generator policy |
