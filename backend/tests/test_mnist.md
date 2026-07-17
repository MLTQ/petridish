# test_mnist.py

## Purpose

Protects the recurrent self-assembling MNIST path without downloading data by
injecting deterministic synthetic 28×28 datasets.

## Components

### Empty-to-routed graph gradient test
- **Does**: Verifies the seed frame has no long-range edges, the first broadcast
  creates bounded directed slots, and gradients reach GRU, patch, and router
  parameters.
- **Interacts with**: Final and post-sensory trajectory logits.
- **Interacts with**: `CellularGraphClassifier` and `BroadcastRouter`.

### Experiment playback integration test
- **Does**: Starts a new training episode, advances one sensory frame, evaluates,
  and validates assembly metadata and aligned snapshot arrays.
- **Interacts with**: `MnistExperiment` and protocol.

### Lesion reassembly test
- **Does**: Damages cells, replays the same digit, and verifies no visible axon
  sends from or terminates on the lesion.
- **Interacts with**: Lesion mask, router, and snapshot projection.

## Contracts

| Dependent | Expects | Breaking changes |
|-----------|---------|------------------|
| MNIST model | Shared recurrent/router parameters remain differentiable | Gradient path changes |
| Viewer | Seed, sensing, and graph arrays retain documented semantics | Protocol changes |
| Lesion brush | Masked cells cannot participate in reassembled edges | Edge-mask semantics |
