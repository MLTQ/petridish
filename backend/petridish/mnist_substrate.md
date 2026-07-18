# mnist_substrate.py

## Purpose

Owns the persistent physical population: occupied and temporarily stunned sites, trainable site
genotypes, advertised query/key memory, incoming dendrites, metabolism,
birth, inherited lineage, cause-classified death, content-aware growth, and pruning.

## Components

### `SpatialSubstrate`
- **Does**: Embeds a sparse neural population in a configurable address space,
  places task-defined semantic ports, and stores all state that survives trials.
- **Interacts with**: The differentiable rule in `mnist_model.py` and lifecycle
  orchestration in `mnist_experiment.py`.
- **Rationale**: Initial dendrites prefer local sources toward the configured
  input boundary, so left-to-right and right-to-left tasks both begin reachable;
  later growth remains constrained to the same physical radius.
- **Rationale**: Directional probe offsets and scores reverse together. This keeps
  boundary outputs supplied with upstream candidates in reversed-flow tasks.
- **Rationale**: Initial weights are signed and zero-centered so the graph does
  not begin as a saturating positive diffusion process.
- **Rationale**: Site genotypes specialize the shared recurrent rule; advertised
  query/key/gate EMAs let growth use content-addressing evidence.
- **Rationale**: Initial density is capped by `max_initial_neurons`, separating
  sparse population cost from physical address-space dimensions.
- **Does**: Places every semantic port in one ordered boundary column and rejects
  undersized fields instead of silently wrapping excess ports into another lane.
- **Does**: Applies the same linear boundary contract to MNIST's 49 vectorized
  patch features; image-grid position is semantic input order, not physical layout.
- **Rationale**: A second port column changes physical distance and connectivity,
  confounding experiments that are meant to share one linear sensory/output boundary.

### `GraphDiagnostics` / `graph_diagnostics`
- **Does**: Caches directed sensory-to-output hop distance and temporal
  reachability until topology changes.
- **Does**: Retains the bounded output-hop distribution so sequence experiments
  can distinguish one-token reach from reach accumulated across a context.

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
- **Does**: Measures normalized starvation/overload stress even while lifecycle
  mutation is waiting for its warm-up.
- **Does**: Converts sustained high traffic into a reversible, non-conducting
  stunned state; seeded recovery restores cells while repeated episodes accumulate
  repairable excitotoxic damage.
- **Rationale**: Indexed slow-state updates use explicit assignment because
  PyTorch advanced indexing returns temporary tensors for in-place operations.

### `structural_step`
- **Does**: Independently gates topology mutation and lifecycle turnover, reports
  death causes, and returns every changed edge and genotype site.
- **Does**: Separates pruning from growth so a prune-only consolidation phase can
  remove low-utility dendrites without immediately replacing them.
- **Does**: Ranks matured connection proposals by accumulated evidence and accepts
  no more than the configured growth budget per structural generation; deferred
  proposals retain their counters for later cycles.
- **Does**: Requires both source and target to retain the configured energy reserve,
  charges construction cost to both endpoints atomically, and records proposals
  blocked by energy, axon capacity, or the global safety cap.
- **Rationale**: Rejected proposals retain their evidence. Accepted axons begin at
  the configured utility, so zero creates a probationary edge that must earn task
  credit before the existing grace period ends.
- **Does**: Reports exact grown and pruned edge counts independently from the
  bounded event list used for visualization.
- **Rationale**: Topology mutates only between differentiable trials.
- **Rationale**: A connection requires a free target dendrite and unused source
  axon capacity; both incoming and outgoing maintenance drain energy.

### `_apply_birth`
- **Does**: Selects an active local parent with axon capacity, copies its genotype
  with configurable mutation noise, and creates one parent-to-child dendrite.
- **Does**: Accepts an optional cycle-local cap; replacement profiles set it to
  the number of deaths that just occurred.
- **Does**: Refuses birth where sampled local occupancy already exceeds the
  configured density ceiling.
- **Rationale**: A newborn enters an actual signal lineage instead of appearing
  disconnected and immediately starving.

### `_recover_stunned` / `_apply_death`
- **Does**: Gives refractory cells a seeded recovery chance without deleting their
  dendrites, then removes mature depleted cells or cells whose repeated excitotoxic
  episodes crossed the damage threshold.
- **Does**: Classifies death as starvation, excitotoxicity, or maintenance.

## Contracts

| Dependent | Expects | Breaking changes |
|-----------|---------|------------------|
| `mnist_model.py` | Dendrite rows are targets and values are source site IDs | Edge orientation changes |
| `graph_layout.py` | Semantic IDs index `input_sites` and `output_sites` | Mapping orientation |
| `mnist_experiment.py` | `record_trial` receives real measured tensors | Statistic semantics |
| Protocol | Site IDs flatten as `y * width + x` | Geometry or ID changes |
| Diagnostics UI | Cached path metrics invalidate after every topology change | Missing invalidation |
| Optimizers | Changed edge slots and inherited genotype rows are reported | Stale Adam moments |

## Notes

Input and output roles are environmental anchors during homeostasis. New neurons
reuse empty site IDs only after all incident dendrites have been removed. Input-role neurons never recruit
incoming dendrites; they are driven only by the environment.
