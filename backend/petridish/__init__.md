# __init__.py

## Purpose

Defines the small public Python API for embedding the simulator in tests,
notebooks, or alternative servers.

## Components

### `SimulationConfig`
- **Does**: Configures field size, graph capacity, dynamics, and task cadence.
- **Interacts with**: `config.py`.

### `PetriDishSimulation`
- **Does**: Owns and advances one deterministic experiment.
- **Interacts with**: `simulation.py`.

## Contracts

| Dependent | Expects | Breaking changes |
|-----------|---------|------------------|
| External Python callers | Both names import from `petridish` | Export removal or rename |
