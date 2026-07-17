# mnist_model.py

## Purpose

Defines differentiable recurrent computation over living neurons, persistent
site genotypes, and dendrites. The image remains external and enters through 49
position-identified sensory neurons on the left of the default substrate.

## Components

### `CellularGraphClassifier`
- **Does**: Patchifies MNIST, combines fast state with persistent site genotype,
  performs query/key/value messaging over real dendrites, applies a genotype-
  modulated shared GRU, and reads ten right-side output neurons.
- **Interacts with**: `SpatialSubstrate` for population and topology.
- **Rationale**: Computation is compacted to occupied sites while site IDs retain
  their physical tensor positions.
- **Rationale**: The message projection begins as an identity, a learned
  positive gain gates it, and normalized recurrent state carries incoming
  information through one explicit residual path.
- **Rationale**: Zero-background hidden state prevents static positional context
  from drowning sensory differences; spatial/metabolic/type context remains a
  trainable input to every recurrent update.
- **Rationale**: Ten learned output identities condition recurrent context and
  query a shared attention pool over the ten physical output neurons. This
  breaks class symmetry without replacing the bank with a dense classifier.
- **Rationale**: A small linear probe over the complete physical output bank is
  retained in the learned classifier because output-bank separability is a
  required diagnostic; it cannot access non-output cells or the input image.
- **Rationale**: Identity-attention readout begins at zero residual scale so it
  cannot obscure the fixed-bank linear probe before learning proves it useful.
- **Rationale**: Each neuron keeps private recurrent state while advertising a
  key, value, and learned emit gate; each target attends only across its real dendrites.
- **Rationale**: A persistent genotype FiLM-modulates the shared rule, so
  specialization scales with occupied sites without a full model per cell.

### `MnistForward`
- **Does**: Returns logits, retained recurrent states, advertised query/key/gate
  statistics, attention entropy, measured traffic, and trace frames.
- **Interacts with**: `MnistExperiment._train_trial`.

### `make_frame`
- **Does**: Captures measured activation, stimulation, load, gradient credit,
  edge flow, and structural events without synthesizing visual activity.

### Stimulation versus load
- **Does**: Measures load as absolute transmitted traffic and stimulation as
  batch-varying information carried by that traffic.
- **Rationale**: Constant self-exciting loops remain metabolically expensive but
  cannot satisfy the organism merely by circulating an unchanging background.

### `shared_parameters`
- **Does**: Separates globally shared neural-rule parameters from persistent
  per-dendrite weights, allowing independent optimizers.

### `readout_parameters`
- **Does**: Identifies output-bank parameters assigned the faster optimizer rate.

### `probe_parameters`
- **Does**: Restricts the initial separability test to a linear output-bank probe
  and bias, keeping every feature-producing parameter frozen.

## Contracts

| Dependent | Expects | Breaking changes |
|-----------|---------|------------------|
| `mnist_experiment.py` | Retained states expose gradients after loss backward | Detaching training state |
| Protocol | Frames carry compact site IDs alongside rows | Removing `sites` |
| Tests | Every active synapse and shared rule remain differentiable | Message-path changes |
| Learning regression | Digit-dependent variance reaches outputs within configured message depth | Attenuating the sensory path |
| `mnist_substrate.py` | Query/key/gate statistics are compact-site aligned | Reordering advertised rows |

## Notes

Topology is frozen during each forward/backward trial. Discrete structural
mutation happens only after gradients have been consumed.
