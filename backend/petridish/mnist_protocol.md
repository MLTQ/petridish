# mnist_protocol.py

## Purpose

Projects the persistent configurable organism into a sparse JSON snapshot containing
only occupied sites and measured neural, metabolic, edge-flow, gradient-credit,
and structural state.

## Components

### `build_mnist_snapshot`
- **Does**: Serializes compact site rows, persistent dendrites, actual forward
  flow, actual backward credit, lifecycle events, and learning metrics.
- **Rationale**: Empty tensor positions are represented by absence rather than
  transmitting dense zero rows.
- **Rationale**: A configurable cap (4,000 by default) bounds rendered edges,
  selected deterministically by independently normalized measured flow, credit,
  weight, and utility; total edge count remains reported.
- **Does**: Publishes the backend-owned hyperparameter slider schema and current
  values with the authoritative scientific state.
- **Does**: Reports structural-lock state and measured synapse update-to-weight
  ratio alongside loss and accuracy.
- **Does**: Reports optimizer phase, overfit-curriculum progress, genotype and
  emission channels, local-attention entropy, effective parameter count, and
  cached sensory-to-output reachability.

## Contracts

| Dependent | Expects | Breaking changes |
|-----------|---------|------------------|
| Frontend | `field.indices[row]` maps compact rows to physical site IDs | Sparse mapping changes |
| Renderer | Edge flow and credit are measured values, never animation proxies | Field removal |
| Metrics | `edgeCount` is authoritative; `visibleEdgeCount` is presentation capacity | Count semantics |
| Hyperparameter panel | Parameter keys, ranges, steps, and values come from the backend | Schema removal |
| Diagnostics | Synapse update ratio is measured after the optimizer step | Synthetic progress values |
| Diagnostics UI | Hop counts and temporal reachability come from the real directed graph | Inferred browser paths |
