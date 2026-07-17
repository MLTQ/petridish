# Petri Dish

Petri Dish is a live laboratory for neural populations embedded in physical
space. Neurons occupy sites in a positional tensor, retain metabolic and task
state across examples, communicate through persistent directed dendrites, and
undergo growth, pruning, birth, death, and lesions.

Two experiments share the viewer: a 32×32 delayed-XOR cellular organism and a
64×64 MNIST spatial neural organism.

## Quick start

Requirements: Python 3.11+, [uv](https://docs.astral.sh/uv/), and Node.js 20+.

```bash
uv sync --extra dev
cd frontend && npm install && cd ..
uv run uvicorn petridish.server:app --app-dir backend --reload
```

The backend serves the built viewer at <http://127.0.0.1:8000/>. For frontend
development, run `npm run dev` inside `frontend/` and use port 5173.

## MNIST experiment

- The physical address space defaults to 64×64; only occupied sites are computed and
  transmitted.
- A dense initial interior contains roughly 1,900 neurons and 7,000 local
  dendrites. Population and topology persist across digits.
- Forty-nine 4×4 patch vectors enter fixed sensory neurons on the left. Ten
  class neurons occupy the right.
- Every living neuron has a trainable persistent site genotype that modulates a
  shared recurrent rule, alongside distinct metabolic, activity, and task history.
- Dendrites own source-neuron references. Existing connections carry actual
  forward messages; nearby active sources accumulate in bounded candidate-ID
  counters before new dendrites can form.
- Neurons keep private recurrent state and advertise learned keys, values, and
  emit gates. Receivers apply local attention only across real dendrites.
- Cross-entropy backpropagates through the frozen connectome. Learning begins
  with an output-bank probe, then unlocks the shared rule and synaptic weights.
- Initial dendrites are signed and zero-centered. Gated normalized messages,
  persistent spatial/type context, and explicit class identities prevent the
  original saturation and output-symmetry collapse.
- Training follows balanced 20-, 256-, 1,000-, and full-MNIST stages; each fixed
  subset must meet its overfit target before advancing.
- Automatic pruning, death, birth, and growth wait for minimum warm-up plus
  accuracy competence or a measured plateau; manual cycles remain available.
- Actual retained-state gradients become the backward-credit phase. They are
  not synthetic animation.
- Stimulation and traffic load update homeostatic energy separately from task
  credit. Starved or persistently overloaded neurons can die; active regions
  can seed neurons into nearby empty sites.
- Structural mutation occurs between differentiable trials. **Structural
  cycle** forces one such update.

The field layers show raw measured state: activation, energy, stimulation,
traffic load, backward credit, task utility, genotype magnitude, emission, and
occupancy. Edge opacity/width uses measured forward flow or backward credit.
There are no decorative signal particles.

## Controls

- **Pause / Step**: inspect input, recurrent forward traffic, backward credit,
  and structural phases independently.
- **Evaluate test set**: evaluate without weight or topology mutation.
- **Structural cycle**: force pruning, candidate accumulation, growth, death,
  and birth once.
- **Lesion brush**: remove neurons and every incident dendrite at the selected
  physical positions.
- **Edge floor**: presentation-only minimum absolute synaptic weight.
- **Hyperparameters**: stage any numeric model, learning, growth, pruning, or
  homeostasis settings and apply them together by restarting the organism.
- **Synapse Δ / |w|**: measured relative optimizer movement, not an animation.
- **Structure**: reports the exact developmental unlock condition.
- **Diagnostics**: report learning phase, hop distance, outputs reachable within
  the recurrent time budget, attention entropy, and effective parameter count.

## Verification

```bash
uv run pytest
cd frontend && npm run check && npm run build
```

## Scientific status

This is a developmental substrate, not a competitive handwriting model. It has
no CNN feature extractor, predefined long-range connectome, or image painted
onto the field. Its small dense readout can access only the ten physical output
neurons and exists to prove whether the graph's representation is separable.
Input influence reaches all ten outputs within the default recurrent budget;
the overfit curriculum now makes failure explicit before generalization claims
are attempted. Displayed traffic, attention, credit, topology, and reachability
are backend measurements. Configuration changes start a new run so histories
from different settings are never mixed.

See `docs/architecture.md` for tensor, credit, and lifecycle contracts.
