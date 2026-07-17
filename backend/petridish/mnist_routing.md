# mnist_routing.py

## Purpose

Implements cell-advertised broadcast matching and the sparse graph that persists
within one MNIST developmental episode. No long-range connectome exists before
the first routing round.

## Components

### `EpisodeGraph`
- **Does**: Stores batch-specific destinations, weights, strengths, ages, and
  utilities for a bounded number of outgoing axons per cell.
- **Interacts with**: `CellularGraphClassifier` and `MnistFrame`.

### `RoutingUpdate`
- **Does**: Returns the updated graph, broadcast signals, and replaced slots.
- **Interacts with**: Trace/event capture in `mnist_model.py`.

### `BroadcastRouter`
- **Does**: Projects recurrent cell state into keys, receptor queries, growth
  requests, values, and signed synaptic weights; top-k compatibility creates
  axons.
- **Interacts with**: Shared recurrent cells in `mnist_model.py`.
- **Rationale**: Attention-like matching selects endpoints, while persistence
  bias, age, and utility turn transient attention into structural plasticity.

### `broadcast`
- **Does**: Applies differentiable all-cell key/query attention before top-k
  matches harden into persistent edges.
- **Interacts with**: The next shared GRU update.
- **Rationale**: The soft broadcast teaches a common addressing language; the
  sparse graph records which relationships persist.
- **Rationale**: A low softmax temperature keeps the broadcast selective enough
  that seven sensory cells are not diluted by the whole population.
- **Rationale**: Activity salience boosts cells whose recurrent state changed
  strongly, giving new sensory information an immediate way to advertise before
  the learned axon head has specialized.

### `messages`
- **Does**: Sends source hidden state along the current directed graph and
  normalizes accumulated input at each receiver.
- **Interacts with**: The next shared GRU update.

## Contracts

| Dependent | Expects | Breaking changes |
|-----------|---------|------------------|
| `mnist_model.py` | Graph tensors use `[batch, source, slot]` | Shape or endpoint semantics |
| `mnist_protocol.py` | Destination `-1` denotes an empty slot | Empty-edge sentinel |
| Tests | Router heads receive gradients through selected edge strengths | Detaching routing scores |
