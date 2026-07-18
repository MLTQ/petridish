# topology_profiles.py

## Purpose

Names topology policies independently from lifecycle and converts legacy structure
booleans without changing existing checkpoints.

## Components

### `TOPOLOGY_PROFILES`

- **fixed**: routes through the current connectome but does not mutate endpoints.
- **adaptive**: permits credit-based pruning and activity/address-based growth.
- **prune_only**: permits credit-based pruning but forbids replacement growth.

### `resolve_topology_profile`

- **Does**: Converts an older `structure=true/false` phase to adaptive/fixed when no
  explicit profile exists and rejects unknown policies.

### `topology_mutates` / `topology_grows`

- **Does**: Provide the two independent gates needed by the experiment and substrate.

## Contracts

| Dependent | Expects | Breaking changes |
|-----------|---------|------------------|
| Trainer checkpoints | Missing profile resolves from saved structural enablement | Default mapping |
| Laboratory | Profile is recorded in manifests, phase records, and commands | Profile names |
| Substrate | Prune-only still uses the existing signed edge-utility rule | Gate semantics |
