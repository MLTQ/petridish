# task.py

## Purpose

Implements the first benchmark independently of cellular physics. Delayed XOR
requires nonlinear integration and memory while remaining cheap to replay.

## Components

### `TaskObservation`
- **Does**: Carries one tick of stimulus, reward, phase, and visible task status.
- **Interacts with**: Consumed by `PetriDishSimulation.step` and `protocol.py`.

### `DelayedXorTask`
- **Does**: Places sensory/motor regions, samples bits, schedules phases, and
  scores motor activation.
- **Interacts with**: Uses timing from `SimulationConfig` and cell activation from
  `PetriDishState`.
- **Rationale**: The environment owns evaluation; the automaton receives only
  local cues and a delayed scalar reward.

## Contracts

| Dependent | Expects | Breaking changes |
|-----------|---------|------------------|
| `simulation.py` | `observe` returns a length-`N` stimulus on the simulation device | Observation shape or timing |
| `protocol.py` | Phase, bits, target, prediction, and accuracy are serializable | Observation field changes |
| Viewer | Sensor A/B and motor 0/1 regions have stable meaning | Region semantics |
