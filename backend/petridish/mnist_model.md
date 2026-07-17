# mnist_model.py

## Purpose

Defines a 16×16 population of shared recurrent cells that receives MNIST through
seven sensory ports and assembles a new sparse directed graph for every input.
The image is an external token stream, never the cellular substrate.

## Components

### `MnistModelConfig`
- **Does**: Sets cell state, routing width, episode phases, optimizer, and
  evaluation cadence.
- **Interacts with**: `CellularGraphClassifier` and `MnistExperiment`.

### `MnistFrame`
- **Does**: Captures one detached micro-step for live playback, including cell
  broadcasts, sensory pulses, graph state, and structural events.
- **Interacts with**: `MnistExperiment` playback and MNIST protocol.

### `MnistForward`
- **Does**: Returns final logits/state/graph plus an optional visualization trace
  differentiable readout trajectory, and wiring cost.
- **Interacts with**: Training, evaluation, and protocol projection.

### `CellularGraphClassifier`
- **Does**: Patchifies a digit, presents seven patch tokens per sensory step,
  applies one shared GRU to every cell, mixes differentiable broadcasts with
  persistent sparse axons, and reads ten motor-interface cells.
- **Interacts with**: `BroadcastRouter` for persistent sparse communication.
- **Rationale**: Cells collectively implement recurrent sparse attention; no
  cell owns unique controller weights and no initial long-range graph exists.
- **Rationale**: Sensory-column and output-class one-hot roles define only the
  environment interface. They do not prescribe any internal connection.
- **Rationale**: Sensory, broadcast, and persistent-edge messages have residual
  paths around the GRU gates so early meta-training cannot silently discard all
  input before an addressing language emerges.

### `regularization`
- **Does**: Charges for strong and long episode edges.
- **Interacts with**: Cross-entropy in `MnistExperiment`.

### `_read_logits`
- **Does**: Applies one shared scalar head to the ten class-role interface cells.
- **Interacts with**: Final prediction and trajectory supervision.

### `lesion`
- **Does**: Masks a circular cell region for subsequent assembly episodes.
- **Interacts with**: The common lesion brush.

## Contracts

| Dependent | Expects | Breaking changes |
|-----------|---------|------------------|
| `mnist_experiment.py` | Trace starts with an empty seed frame and ends at readout | Frame ordering |
| `mnist_protocol.py` | Sensor/motor identities and current frame tensors are readable | Buffer names or frame shape |
| Viewer | Field is 16×16; seven left ports sense and ten right ports classify | Interface geometry |

## Notes

The outer loop is supervised meta-training. Within an episode, recurrent cells
receive only local averages, messages, token input at sensory ports, weak spatial
morphogens, role signals, and a clock; graph endpoints are not model parameters.
Optional episode noise is disabled by default so reset and lesion comparisons
remain deterministic.
