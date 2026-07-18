# lifecycle_profiles.py

## Purpose

Defines named homeostasis interventions for reproducible token-organism ablations.
Profiles are explicit experiment metadata rather than silent changes to global defaults.

## Components

### `LIFECYCLE_PROFILES`
- **Does**: Declares `off`, the original `baseline`, and the empirically motivated
  `balanced` intervention, and the population-stable `replacement` intervention.

### `apply_lifecycle_profile`
- **Does**: Applies one profile to an immutable `MnistModelConfig`.
- **Interacts with**: Headless trainer launch and the two-GPU laboratory.
- **Rationale**: The original token baseline saturated its 64-death budget every
  cycle, lost 34 of 64 routes by update 1,000, and collapsed generation. Balanced
  keeps under/overstimulation death and stun/recovery while equalizing maximum
  births/deaths, lowering metabolic drain, and requiring repeated stronger overload
  for excitotoxic death.

### `replacement`
- **Does**: Retains every balanced threshold but permits births only up to the
  number of deaths in the same lifecycle cycle.
- **Rationale**: The balanced run preserved all routes but added 224 cells and lost
  none by update 721; matching maximum budgets did not produce population balance.

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
