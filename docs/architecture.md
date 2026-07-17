# Architecture

## Runtime boundary

The Python process lazily owns named XOR and MNIST experiments, their task state,
mutation, optimizer state, and history. Switching preserves each experiment
until explicit reset. The browser is an observer/controller and never mutates a
local copy of scientific state.

Snapshots are sampled independently of simulation ticks. This lets a GPU run
many physics steps per rendered frame and keeps a slow browser from changing the
experiment.

## Cell state

The field has shape `[height, width, 16]` and stable channel meanings:

| Channel | Meaning |
|---------|---------|
| 0 | viability/alive amount |
| 1 | signed activation |
| 2–3 | phase sine and cosine |
| 4–7 | private recurrent memory |
| 8 | metabolic energy |
| 9 | axon growth signal |
| 10 | dendrite/receptor signal |
| 11 | XOR reward trace or current MNIST sensory-port pulse |
| 12–13 | normalized spatial identity |
| 14 | sensory-cell identity |
| 15 | motor-cell identity |

## Edge state

Every source cell owns a small fixed number of edge slots. XOR slots store a
destination, signed weight, existence gate, eligibility, age, and utility.
MNIST episodes begin with empty slots. Recurrent cells broadcast axon keys,
dendritic queries, and growth strengths; top-k matches create destinations,
signed weights, strengths, ages, and utilities that persist only for that input
episode. Fixed slot count preserves bounded tensor shapes without prescribing a
connectome.

## XOR tick order

1. The delayed-XOR task produces sensory input and an optional reward pulse.
2. Local 3×3 perception and sparse axonal messages are computed.
3. Cell activation, memory, phase, energy, growth signals, and viability update.
4. Edge eligibility and reward-modulated weights update.
5. At a slower cadence, old low-utility edges prune and empty slots grow.
6. A sampled, CPU-safe snapshot is sent to connected observers.

## MNIST developmental episode

1. A 28×28 digit becomes a 7×7 sequence of 4×4 patch vectors outside the field.
2. The 16×16 field starts from shared-rule seed state with no long-range edges.
3. For seven sensing steps, one patch row enters seven left-border sensory ports.
4. Every cell applies the same GRU to local 3×3 perception, an all-cell soft
   broadcast, persistent graph messages, sensory input, weak morphogens,
   interface role, and episode clock.
5. Cells advertise keys, receptor queries, values, and growth requests. Soft
   attention teaches the broadcast language; top-k matches create or retain up
   to four directed axons per source. Distance and wiring costs discourage
   gratuitous connections.
6. Additional development and readout steps run without sensory input. Ten
   right-border output cells carry one-hot class roles and use one shared scalar
   readout head. Sensor-column/output-class roles define the external interface,
   not internal wiring.
7. Cross-entropy meta-trains the patch projection, shared GRU, broadcast
   language, synaptic rule, and output scale. Auxiliary loss on the post-sensory
   readout trajectory shortens the training signal; graph endpoints are not
   parameters.

The viewer replays the empty seed and every micro-step before training the next
episode, so optimizer cadence is deliberately slower than visualization cadence.

## Intervention semantics

A lesion zeros XOR viability/energy or masks MNIST cells. XOR viability can
regrow through local field dynamics. MNIST immediately replays the same digit
from an empty graph under the persistent lesion, exposing whether broadcast
routing can assemble an alternate path.

Manual stimuli are short-lived additions to the task signal. Manual reward uses
the same scalar third factor as task reward; it does not directly edit weights.

## Reproducibility

Resetting rebuilds the selected experiment, data order, shared model, and task
generators from one integer seed. There is no MNIST topology to restore: every
episode assembles its own. Trace frames detach learned state and never retain an
autograd graph.
