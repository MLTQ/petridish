# mnist_substrate.py

## Purpose

Owns the persistent physical population: occupied sites, trainable site
genotypes, advertised query/key memory, incoming dendrites, metabolism,
birth, death, content-aware growth, pruning, and lesions.

## Components

### `SpatialSubstrate`
- **Does**: Embeds a sparse neural population in a configurable 64×64 default
  address space and stores all state that survives across MNIST trials.
- **Interacts with**: The differentiable rule in `mnist_model.py` and lifecycle
  orchestration in `mnist_experiment.py`.
- **Rationale**: Initial dendrites prefer longer left-local sources so the
  input/output separation has a usable gradient path; later growth is
  still constrained to the same physical discovery radius.
- **Rationale**: Initial weights are signed and zero-centered so the graph does
  not begin as a saturating positive diffusion process.
- **Rationale**: Site genotypes specialize the shared recurrent rule; advertised
  query/key/gate EMAs let growth use content-addressing evidence.

### `GraphDiagnostics` / `graph_diagnostics`
- **Does**: Caches directed sensory-to-output hop distance and temporal
  reachability until topology changes.

### `GraphSnapshot`
- **Does**: Detaches dendrite sources, weights, utility, measured forward flow,
  and measured backward credit for one trace frame.
- **Interacts with**: Protocol serialization and renderer.

### `record_trial`
- **Does**: Applies actual stimulation, traffic, gradient credit, and correctness
  to slow edge/neuron statistics and metabolic energy.
- **Rationale**: Task credit and metabolic load remain separate signals.
- **Rationale**: Positive signed removal credit is percentile-normalized before
  utility accumulation; harmful influence is not protected by absolute value.
- **Rationale**: Activity and credit statistics accumulate during learning
  warm-up while energy penalties wait until structural plasticity unlocks.
- **Does**: Retains slow query, key, and emission statistics for connection proposals.

### `structural_step`
- **Does**: Prunes weak dendrites, accumulates content-compatible source
  candidates, forms thresholded connections, kills depleted neurons, and seeds neurons.
- **Rationale**: Topology mutates only between differentiable trials.
- **Rationale**: A connection requires a free target dendrite and unused source
  axon capacity; both incoming and outgoing maintenance drain energy.

## Contracts

| Dependent | Expects | Breaking changes |
|-----------|---------|------------------|
| `mnist_model.py` | Dendrite rows are targets and values are source site IDs | Edge orientation changes |
| `mnist_experiment.py` | `record_trial` receives real measured tensors | Statistic semantics |
| Protocol | Site IDs flatten as `y * width + x` | Geometry or ID changes |
| Diagnostics UI | Cached path metrics invalidate after every topology change | Missing invalidation |

## Notes

Input and output roles are environmental anchors during homeostasis, but a
manual lesion can still remove them. New neurons reuse empty site IDs only after
all incident dendrites have been removed. Input-role neurons never recruit
incoming dendrites; they are driven only by the environment.
