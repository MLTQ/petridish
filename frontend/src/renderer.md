# renderer.ts

## Purpose

Renders sparse neurons for image and token tasks, persistent dendrites, measured forward traffic,
measured backward credit, and actual lifecycle events. It does
not synthesize signal particles or decorative neural activity.

## Components

### `DishRenderer`
- **Does**: Maps compact protocol rows back into configurable physical site positions,
  draws occupied neurons and real edges, handles lesions/selections, and falls
  back to Canvas2D if WebGL is unavailable.

### `drawEdges`
- **Does**: Uses measured edge flow during forward phases and measured gradient
  credit during feedback; weight sign determines color.
- **Rationale**: Line opacity and width encode real values. There are no animated
  packet proxies.

### `drawCells`
- **Does**: Colors occupied sites by a selected raw channel with per-frame scale;
  input/output outlines represent fixed environmental roles.
- **Does**: The default phase-signal layer resolves to activation during forward
  computation and measured neuron credit during feedback. Explicit activation
  and credit choices remain raw, phase-independent views.
- **Does**: Treats genotype magnitude and measured emit-gate EMA as selectable
  scientific layers using the same raw-channel contract.
- **Does**: Exposes measured homeostatic stress, neuron age, and lineage depth
  as additional raw layers.

### `drawEvents`
- **Does**: Marks only reported dendrite growth/pruning and neuron birth/death.

## Contracts

| Dependent | Expects | Breaking changes |
|-----------|---------|------------------|
| Protocol | Sparse fields map `cells[row]` to `indices[row]` | Mapping changes |
| Main | Pointer callbacks remain field-space coordinates | Callback changes |
| Scientific interpretation | Edge `flow` and `credit` are backend measurements | Synthetic values |
