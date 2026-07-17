# mnist_curriculum.py

## Purpose

Builds a deterministic class-balanced overfit ladder so transport and optimization
failures are exposed before the organism is asked to generalize across full MNIST.

## Components

### `CurriculumStage`
- **Does**: Couples a fixed subset size, advancement target, and dataset view.
- **Interacts with**: `MnistExperiment` in `mnist_experiment.py`.

### `build_curriculum`
- **Does**: Produces 20-example, 256-example, 1,000-example, and full-data stages
  when the source dataset is large enough.
- **Rationale**: The smallest balanced stage contains every available class; a
  nominal 16-example stage cannot represent ten classes evenly.

### `_balanced_order`
- **Does**: Interleaves seeded per-class sample buckets without duplicating data.

## Contracts

| Dependent | Expects | Breaking changes |
|-----------|---------|------------------|
| `mnist_experiment.py` | Final stage covers the complete training dataset | Omitting full stage |
| Reproducibility | Equal seeds produce equal subset membership | Sampling policy |
| Tests | TensorDataset labels are accepted without item-by-item loading | Label extraction |
