# test_mnist.py

## Purpose

Protects persistent spatial MNIST computation, exact gradient credit, sparse
serialization, lesions, and lifecycle mutation using a small deterministic
substrate and synthetic images.

The compact fixture is 24×52 rather than square so it remains inexpensive while
giving all 49 vectorized patch features one ordered physical boundary column.

## Components

### Persistent graph gradient test
- **Does**: Verifies topology does not reset during a trial while patch, GRU,
  genotype, local query/key, synapse, output-bank probe, attention-scale, and
  message-flow gradients remain nonzero.
- **Does**: Verifies all 49 input ports occupy rows 1–49 of one left column.

### Feedback/protocol integration test
- **Does**: Runs one optimizer update, advances to the feedback phase, and
  verifies real neuron credit and sparse site/edge alignment.
- **Does**: Verifies the first update is a readout-only reservoir probe and
  publishes real reachability/capacity diagnostics without moving synapses.

### Lesion test
- **Does**: Removes occupied sites and proves every serialized edge still has a
  living source and target.

### Lifecycle test
- **Does**: Forces depleted interior neurons to die while fixed input/output
  interface neurons survive ordinary homeostasis, and verifies starvation is
  reported as the dominant cause.

### Birth inheritance test
- **Does**: Requires a newborn to record a local parent, inherit its genotype,
  increment lineage depth, and enter the graph through one real parent dendrite.

### Replacement birth test
- **Does**: Proves a healthy replacement-only population cannot grow and any later
  births are capped by deaths in that exact lifecycle cycle.
- **Does**: Proves recovery-only can restore a depleted stunned neuron while cell
  occupancy and every dendrite endpoint remain bit-identical.

### Lifecycle staging test
- **Does**: Activates metabolic pressure and a turnover generation while
  competence-gated topology plasticity remains locked.

### Candidate-counter growth test
- **Does**: Supplies repeated local source evidence to a target with a free
  dendrite and verifies the stored source ID becomes a directed connection.
- **Does**: Verifies prune-only topology never invokes candidate discovery/growth.
- **Does**: Caps adaptive growth to the strongest configured number of proposals
  per generation and retains ready evidence when the budget is zero.
- **Does**: Refuses a matured proposal when either endpoint cannot retain its energy
  reserve, then verifies an affordable retry debits both endpoints, records the
  construction energy, and starts at the configured probationary utility.

### Hyperparameter schema tests
- **Does**: Locks the 64×64/8-cell-radius defaults, requires a control spec for
  every numeric configuration field, keeps categorical architecture out of the
  slider schema, verifies a one-cell direct-neighbor radius, and checks typed and
  cross-field validation.
- **Does**: Requires power-of-two square sizes from 16 through 1024 and a
  field-derived broadcast-radius ceiling.
- **Does**: Requires the MNIST snapshot to omit sequence-only broadcast and
  fast-weight/binding-memory controls so every displayed slider has an effect on
  that experiment.

### Fixed-connectome learning regression
- **Does**: Requires the organism to lower loss on an easy spatial digit task,
  produce measurable synapse updates, and preserve generation zero throughout
  structural warm-up.

### Curriculum balance test
- **Does**: Requires the first 20-example overfit stage to contain two examples
  of every MNIST class and the final stage to cover the source dataset.

## Contracts

| Dependent | Expects | Breaking changes |
|-----------|---------|------------------|
| Model | Topology is frozen within forward/backward | In-trial mutation |
| Protocol | Compact rows map through physical site IDs | Sparse contract changes |
| Substrate | Manual lesions remove all incident dendrites | Cleanup changes |
| Viewer controls | Snapshot publishes every task-relevant model field | Missing slider metadata |
| Optimizers | A fixed organism must demonstrate end-to-end supervised learning | Broken credit path |
