# mnist_protocol.py

## Purpose

Projects the currently replayed MNIST developmental frame into the common cell,
edge, event, task, and metric snapshot envelope.

## Components

### `build_mnist_snapshot`
- **Does**: Maps recurrent state, broadcast requests, sensory pulses, current
  episode axons, structural events, phase, and learning metrics to JSON.
- **Interacts with**: `MnistExperiment`, renderer, and task panel.
- **Rationale**: Trace tensors are already detached; this boundary performs only
  presentation mapping and serialization.

## Contracts

| Dependent | Expects | Breaking changes |
|-----------|---------|------------------|
| Frontend protocol | MNIST phase/assembly metadata and common graph envelope | Key or type changes |
| Renderer | Sixteen advertised channels retain standard ordering | Channel mapping |
| Inspector | Sensory pulses use `reward_trace`; broadcast requests use growth channels | Visualization semantics |

## Notes

Only strengths above 0.18 are serialized. Empty graph slots use destination -1
inside the backend but are never sent to the viewer.
