# mnist_experiment.py

## Purpose

Meta-trains the shared recurrent cell program and exposes every developmental
micro-step as live playback. Training episodes, held-out evaluation, lesions,
and deterministic data loading remain owned by the backend.

## Components

### `MnistExperiment`
- **Does**: Owns model, optimizer, data, training metrics, current input, and the
  detached assembly trace shown by the viewer.
- **Interacts with**: Runtime and MNIST protocol.

### `step`
- **Does**: Advances one trace frame; after readout, performs one gradient update
  on a new batch and returns to an empty seed graph.
- **Interacts with**: Runtime playback cadence.
- **Rationale**: Separating optimizer cadence from micro-step playback makes
  self-assembly observable instead of showing only its final result.

### `_train_episode`
- **Does**: Meta-trains the shared GRU, patch encoder, broadcast router, and
  readout using final/trajectory classification losses plus wiring cost.
- **Interacts with**: `CellularGraphClassifier`.

### `rewire_now`
- **Does**: Starts a newly trained assembly episode immediately.
- **Interacts with**: Viewer “New assembly” control.

### `evaluate`
- **Does**: Measures a bounded held-out slice without allocating traces.
- **Interacts with**: Viewer evaluation control.

### `lesion`
- **Does**: Masks cells, reports severed visible axons, and replays the same digit
  from an empty graph under the lesion.
- **Interacts with**: Shared lesion brush and renderer events.

## Contracts

| Dependent | Expects | Breaking changes |
|-----------|---------|------------------|
| Runtime | `step`, `lesion`, `evaluate`, `rewire_now`, `seed`, and `device` | Public method changes |
| MNIST protocol | `last_frame`, trace cursor, task metrics, and current image | Attribute names/shapes |
| Tests | Synthetic datasets can be injected without downloads | Constructor dataset arguments |

## Notes

`tick` counts displayed micro-steps; `training_step` counts optimizer updates.
Held-out evaluation and training schedules use the latter.
