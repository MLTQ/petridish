# mnist_hyperparameters.py

## Purpose

Defines the authoritative numeric control schema for the MNIST organism and
validates user-selected configurations before a new organism is constructed.

## Components

### `HyperparameterSpec` / `SPECS`
- **Does**: Gives every `MnistModelConfig` field a label, group, slider bounds,
  and step size.
- **Interacts with**: `hyperparameter_payload` and the frontend control panel.
- **Rationale**: The backend owns scientific ranges so UI controls cannot drift
  from accepted values.
- **Rationale**: Substrate width/height describe the physical tensor; broadcast
  radius starts at one cell, meaning immediate-neighbor communication only.
- **Rationale**: Learning stability controls include signed weight scale,
  message gain, separate readout/rule rates, gradient clipping, and structure warm-up.
- **Rationale**: Early-output loss remains explicit and defaults to zero so
  unreachable pre-arrival steps cannot train the classifier toward uniform guesses.
- **Rationale**: Genotype capacity, local-attention temperature, staged optimizer
  unlocks, curriculum windows, and competence-gated structure remain viewer-tunable.
- **Rationale**: Lifecycle activation, cadence, local birth-density ceiling,
  newborn reserve, and inheritance noise are controls rather than hidden constants.

### `hyperparameter_payload`
- **Does**: Serializes ordered slider definitions with current values.
- **Interacts with**: `build_mnist_snapshot` in `mnist_protocol.py`.

### `configured`
- **Does**: Rejects unknown, non-finite, out-of-range, non-integral, or mutually
  inconsistent changes and returns a new immutable configuration.
- **Interacts with**: `ExperimentRuntime.handle_command`.

## Contracts

| Dependent | Expects | Breaking changes |
|-----------|---------|------------------|
| `mnist_config.py` | Every dataclass field has exactly one spec | Adding a field without a spec |
| Frontend | Payload keys, bounds, and integer flags are authoritative | Payload field renames |
| Runtime | Validation completes before replacing the live organism | In-place mutation |
