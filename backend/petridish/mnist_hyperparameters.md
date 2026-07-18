# mnist_hyperparameters.py

## Purpose

Defines the authoritative numeric control schema for the MNIST organism and
validates user-selected configurations before a new organism is constructed.
Categorical experiment identity such as sequence cell architecture is recorded
in configuration but selected through the laboratory, not represented as a slider.

## Components

### `HyperparameterSpec` / `SPECS`
- **Does**: Gives every `MnistModelConfig` field a label, group, slider bounds,
  and step size.
- **Interacts with**: `hyperparameter_payload` and the frontend control panel.
- **Rationale**: The backend owns scientific ranges so UI controls cannot drift
  from accepted values.
- **Rationale**: One square-field control exposes powers of two from 16 through
  1024. Both corpus tasks additionally expose their exact 68×68 single-column
  port geometry. Width and height remain internal config fields so every live experiment
  is square and a size change is atomic.
- **Rationale**: Broadcast radius starts at one cell, meaning immediate-neighbor
  communication only. Its emitted maximum is derived from the current field, so
  the viewer cannot offer a radius that wraps across half the tensor.
- **Rationale**: Learning stability controls include signed weight scale,
  message gain, separate readout/rule rates, gradient clipping, and structure warm-up.
- **Rationale**: Early-output loss remains explicit and defaults to zero so
  unreachable pre-arrival steps cannot train the classifier toward uniform guesses.
- **Rationale**: Genotype capacity, local-attention temperature, staged optimizer
  unlocks, curriculum windows, and competence-gated structure remain viewer-tunable.
- **Rationale**: Lifecycle activation, cadence, replacement-only birth policy,
  local birth-density ceiling, newborn reserve, and inheritance noise are controls
  rather than hidden constants.
- **Rationale**: Excitotoxicity controls distinguish the traffic threshold that
  causes a stun, probabilistic recovery, refractory time, accumulated damage,
  repair, and eventual death instead of treating one overload as fatal.
- **Rationale**: Sequence snapshots additionally expose low-rank broadcast workspace
  slots, gain, and memory decay; MNIST omits controls its model does not consume.
- **Rationale**: Sequence-only fast-weight gain/decay are also omitted from MNIST.

### `hyperparameter_payload`
- **Does**: Serializes ordered slider definitions with current values.
- **Interacts with**: `build_mnist_snapshot` in `mnist_protocol.py`.
- **Does**: Emits square field size as discrete choices and includes workspace
  controls only for sequence organisms.

### `configured`
- **Does**: Rejects unknown, non-finite, out-of-range, non-integral, or mutually
  inconsistent changes and returns a new immutable configuration.
- **Interacts with**: `ExperimentRuntime.handle_command`.
- **Does**: Expands `field_size` into equal width and height, clamps a formerly
  valid radius when shrinking, and rejects non-power-of-two field sizes except
  for the explicit corpus-task 68×68 geometry.

## Contracts

| Dependent | Expects | Breaking changes |
|-----------|---------|------------------|
| `mnist_config.py` | Every numeric dataclass field has exactly one spec | Adding a numeric field without a spec |
| Frontend | Payload keys, bounds, and integer flags are authoritative | Payload field renames |
| Runtime | Validation completes before replacing the live organism | In-place mutation |
