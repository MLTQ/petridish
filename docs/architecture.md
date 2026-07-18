# Architecture

## Runtime boundary

Python owns learning, neural state, topology, lifecycle, data order, and
interventions. The browser receives sampled authoritative snapshots and never
simulates or invents scientific state.

## Physical substrate

MNIST defaults to a configurable 64×64 address space; synthetic sequence tasks
use a benchmarked 24×24 field, and Tiny Shakespeare defaults to 68×68. Every
experiment exposes a square power-of-two size from 16 through 1024, while Tiny
Shakespeare alone also exposes 68. A separate
initial-population cap prevents tensor extent from implying dense occupancy. A
flattened site ID is always `y * width + x`. Empty positions have no neuron row
in the sparse protocol.

Persistent per-site state includes:

| State | Meaning |
|-------|---------|
| Occupancy | Whether a neuron currently inhabits the physical position |
| Energy | Homeostatic viability reserve |
| Stimulation EMA | Recent task-varying external and incoming information |
| Load EMA | Recent incoming message traffic |
| Neuron credit | Magnitude of retained-state loss gradient |
| Task utility | Reward/credit-weighted slow value |
| Age | Trials survived since birth |
| Homeostatic stress | Normalized starvation or overload pressure |
| Parent / lineage | Reproductive source site and inherited generation depth |
| Candidate IDs/counters | Nearby emitting sources being considered |
| Genotype | Trainable site-specific identity that FiLM-modulates the shared rule |
| Query/key/emission EMA | Slow memory of what a neuron requests and advertises |

Forty-nine immortal-under-homeostasis input roles form a 7×7 patch bank near
the left boundary. Ten output roles lie near the right boundary. A lesion can
still physically remove interface neurons.

Sequence layouts expose vocabulary-sized token and output ports. Tiny Shakespeare's
66 ports fit exactly in one column per boundary because the first and last rows are
reserved. Associative
recall permutes both boundaries. Synthetic tiny language and Tiny Shakespeare
reverse the physical direction so tokens enter on the right and predictions
leave on the left. Boundary ports pack into non-overlapping stripes when one
column is too short for the vocabulary. Probe offsets and scores reverse
together, preserving upstream dendrites in either orientation.

## Dendrites and axons

Each target neuron owns a fixed number of dendrite slots. A slot stores a source
site ID, signed weight, age, task utility, measured flow, and measured backward
credit. The same directed relation is the source's axon and the target's
dendrite; there is only one edge state. Targets have four dendrite slots and
sources support at most eight axons by default.

Initial dendrites are local and preferentially select farther-left sources,
providing a physically traversable input-to-output bridge without defining a
task-specific connectome. The default 8-cell discovery radius keeps broadcasts
local; twenty recurrent steps provide enough depth to cross the field.

For new growth, a target samples a bounded set of physically local sources.
Repeated emission by the same source increments a target-local candidate
counter. A free dendrite forms only after the source ID's counter crosses the
configured threshold, both endpoints can pay construction cost while retaining
their energy reserve, and the per-generation safety budget has capacity. A new
axon may start with zero utility and must earn task credit during its grace period
or become eligible for pruning.

## Differentiable trial

Topology is frozen for one trial:

1. A digit becomes 49 vectors of 16 pixels outside the substrate.
2. Patch vectors stimulate the 49 left-side input neurons.
3. Twenty recurrent message-passing steps run over occupied neurons and active
   dendrites. Each neuron keeps private fast state, advertises query/key/value
   projections plus an emit gate, and attends only across its real dendrites.
   A persistent site genotype FiLM-modulates the shared normalized GRU rule.
4. Ten right-side neurons receive persistent learned class identities. Those
   identities query a shared transformer-like attention pool, while a small
   linear probe over the complete physical output bank verifies separability.
5. Cross-entropy backpropagates through the exact recurrent graph.
6. Adam updates shared patch/rule/readout parameters and individual dendrite
   weights with separately configurable rates.
7. Retained-state gradients become per-neuron backward credit; synapse
   gradients times weights become per-edge credit.

Discrete endpoints do not receive gradients and never mutate during these
steps. This preserves a valid reverse-mode graph while allowing topology to be
governed by local structural rules afterward.

## Token-stream trial

Sequence examples stimulate exactly one semantic token port at a time; token IDs are
not painted over the field. Neuron state persists across token boundaries and the
shared genotype-modulated GRU performs six graph updates per token. Output ports are
read after each token for either final delayed recall or autoregressive loss.

Alongside persistent axons, sequence organisms can use a low-rank advertisement
workspace. Neurons content-select values into shared slots and query those slots on
later updates. Slots persist only within an example and decay explicitly. An optional
fast-weight matrix accumulates neuron-advertised key/value outer products and supports
linear-attention reads; it is disabled by default after failing the two-binding recall
ablation. Neither mechanism appears as a persistent edge in the viewer. Cell load
includes its real contribution, while edge marks continue to encode only actual
axonal flow or credit.

Recall uses a one-to-three-binding curriculum. Language loss covers every next token,
but reported accuracy includes only the context-dependent verb/object positions. This
prevents frequency and EOS learning from masquerading as sequence understanding.

Tiny Shakespeare uses a 90/10 sequential train/validation split, a 64-character
context, and a dynamically sized character vocabulary. The source text is
downloaded once and then read from a local cache. Interactive prompts reuse the
same encoder and recurrent inference path as validation. Each explicit generate
command samples one character at temperature 0.85 and then recomputes the visible
trace for the extended context; it does not mutate weights or topology.

## Homeostasis and lifecycle

Forward traffic and backward task credit are deliberately separate. Traffic
updates stimulation/load EMAs and metabolic energy. Credit updates task
utility. Positive task utility supplies a small bounded energy bonus without
counting as stimulation. Healthy stimulation recovers energy; starvation,
overload, and dendrite maintenance consume it.

Load is absolute traffic. Stimulation is the batch-varying component of incoming
messages and external input. Constant self-exciting loops therefore incur load
without satisfying the information-stimulation requirement.

At a slower structural cadence:

1. Dead-source and dead-target dendrites are removed.
2. Old low-utility dendrites prune within a bounded budget.
3. Depleted non-interface neurons past their juvenile grace period die.
4. Active neighborhoods may seed neurons into empty sites.
5. Candidate source counters decay, accumulate new evidence, and form
   thresholded dendrites.

This order lets overload first create local growth demand and alternate paths;
death is a sustained outcome rather than an instantaneous response.

Metabolic pressure and population turnover have their own activation warm-up.
Training first fits the fixed output-bank probe, then unlocks the shared cellular
rule and synapses. Free-form pruning and candidate-based growth additionally
require accuracy competence or a measured learning plateau. Signed removal
credit is percentile-normalized, so harmful influence receives no protection.

Newborn sites select a locally active parent with available axon capacity,
inherit its genotype with configurable Gaussian mutation, and begin with one
parent-to-child dendrite. Adam moments for replaced genotype rows and changed
dendrite slots are cleared so inheritance is not contaminated by a prior occupant.
Deaths are classified by dominant starvation, overload, or maintenance pressure.

## Overfit curriculum and diagnostics

Training begins on a deterministic class-balanced 20-example subset, then moves
to 256 examples, 1,000 examples, and full MNIST only after meeting each stage's
rolling training-accuracy target. The first stage is a complete batch so failure
to overfit cannot be blamed on sample omission.

The backend caches directed sensory-to-output hop distances until topology
changes. Snapshots report minimum/median hops, outputs reachable within the
message-step budget, local-attention entropy, active parameter count, and
parameters per living neuron. These metrics distinguish capacity, routing, and
optimization failures without inferring state in the browser.

## Sparse viewer protocol

MNIST snapshots transmit `field.indices[row]` beside `field.cells[row]` instead
of dense empty positions. Edge count is authoritative, while rendering defaults
to the 4,000 edges with greatest measured flow, credit, weight, or utility.

The snapshot also carries one backend-owned numeric control specification for
every configuration field. The square-size slider carries discrete powers of
two rather than pretending the values are continuous, and broadcast radius is
bounded by the current field. Slider edits remain local until Apply; the runtime
validates the complete change set and constructs a fresh organism.

Viewer marks are evidentiary:

- Cell color: selected measured channel, autoscaled per frame with raw inspector
  values.
- Edge sign: excitatory or inhibitory weight.
- Edge opacity/width: measured forward flow, or measured gradient credit during
  feedback.
- Cyan/red event marks: actual growth/birth or pruning/death reports.

There are no synthetic signal dots, arbitrary moving packets, or inferred
activity.

## Training and visualization cadence

Visualization cadence and optimizer cadence are separate. Ordinary playback walks
through stored token, feedback, and structural frames, then performs one optimizer
update when that trace is exhausted. Fast training calls the same differentiable
update directly with trace capture disabled, removes the 15 Hz presentation delay,
and suppresses automatic validation sweeps. Aggregate flow, stimulation, credit,
homeostasis, and topology still update; only token-local graph copies are omitted.

Heavy PyTorch work runs in a worker thread so it cannot block WebSocket processing.
Fast mode reports at most once per second and the browser deliberately leaves the
field unchanged while continuing to update metrics. Exiting the mode replays the
latest sequence with current weights and restores an authoritative visual trace.

## Reproducibility

Reset reconstructs model parameters, initial occupancy, local probes,
connectome, and deterministic data order from one seed. Trace frames detach
only after a differentiable trial has retained the tensors needed for credit.
