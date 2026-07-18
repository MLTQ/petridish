# lifecycle_profiles.py

## Purpose

Defines named homeostasis interventions for reproducible token-organism ablations.
Profiles are explicit experiment metadata rather than silent changes to global defaults.

## Components

### `LIFECYCLE_PROFILES`
- **Does**: Declares `off`, the original `baseline`, and the empirically motivated
  `balanced` intervention.

### `apply_lifecycle_profile`
- **Does**: Applies one profile to an immutable `MnistModelConfig`.
- **Interacts with**: Headless trainer launch and the two-GPU laboratory.
- **Rationale**: The original token baseline saturated its 64-death budget every
  cycle, lost 34 of 64 routes by update 1,000, and collapsed generation. Balanced
  keeps under/overstimulation death and stun/recovery while equalizing maximum
  births/deaths, lowering metabolic drain, and requiring repeated stronger overload
  for excitotoxic death.

### `resolve_lifecycle_profile`
- **Does**: Maps a legacy enabled boolean plus unspecified `off` profile to the
  original baseline while preserving explicit named profiles.
- **Rationale**: Old clients remain reproducible and manifests record the policy
  that actually ran.

## Contracts

| Dependent | Expects | Breaking changes |
|-----------|---------|------------------|
| `train_shakespeare.py` | Profile application preserves unrelated task settings | Profile names or semantics |
| `laboratory.py` | Every advertised profile is accepted by the trainer | Removing a profile |
| Experiment manifests | Profile names identify reproducible interventions | Renaming profiles |
