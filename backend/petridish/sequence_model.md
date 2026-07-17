# Cellular sequence model

The sequence model embeds a persistent recurrent computation in the same physical
neuron field used by the MNIST experiment. Each vocabulary item owns one semantic
input port, but `GraphLayout` permutes those ports in space. A token is supplied by
stimulating its port; the token ID is not broadcast to every cell.

For each token the selected shared genotype-modulated recurrent rule runs several
local graph updates. GRU is the preserved baseline; controlled homogeneous LSTM,
ESN, and temporal-transformer rules share the same physical graph and readout.
Queries, keys, values, emission gates, weighted directed dendrites, measured traffic,
and local attention are identical in spirit to the classifier. Crucially, neuron state
is retained across token boundaries. The vocabulary-sized output bank is read after every token,
which supports both a final delayed-recall target and autoregressive next-token loss.

A small low-rank broadcast workspace implements the advertisement channel without an
all-pairs attention matrix. Neurons softly write content-selected values into shared
slots and selectively read them during the recurrent update. This path is ephemeral
at the edge level but its slots persist within a sequence using configurable decay.
It does not pretend to be an axon; persistent weighted dendrites still own the
structural graph. `broadcast_gain = 0` is the controlled no-workspace ablation.

In parallel, neuron-advertised keys and values update a persistent fast-weight matrix;
neuron queries read it as recurrent linear attention. This explicitly tests
key-addressed binding without creating fictitious persistent edges. Fast weights reset
between examples, while axons, genotypes, rule parameters, and lifecycle state persist.
`fast_weight_gain = 0` removes this path independently.

The optional neuron-owned binding memory is a narrower relational intervention.
Every living neuron's genotype defines a content address. After adjacent tokens are
processed, the successor state is softly written into physical owner neurons selected
by the predecessor token's key. A later occurrence of that token reads the owners and
injects the retrieved state only at the stimulated input port; ordinary dendrites must
still transport it to outputs. This tests learned token-conditioned ownership without
granting the readout direct access to a global key/value table. The memory resets per
example and is omitted entirely when `binding_memory_gain = 0`, preserving legacy and
corpus checkpoints.
The `binding_token_values` ablation writes the successor token embedding instead of
the successor input neuron's mixed hidden state while keeping addresses, owners,
retrieval, injection site, topology, and optimizer identical.

Frames contain one measured state per token rather than every microstep. This makes the
live sequence causally readable without inventing presentation-only traffic.
An optional frame callback fires immediately after each such frame and its aligned
logits exist. The callback cannot be used when trace capture is disabled; this keeps
the live stream tied to measured model state rather than synthesized progress.

When trace capture is disabled, the model omits token-local frame accumulators and
full graph copies while retaining aggregate stimulation, load, edge flow, and gradient
state required by learning and lifecycle rules. Attention entropy remains on the
accelerator throughout the recurrent loop and crosses to the CPU only once, avoiding
one synchronization stall per microstep.

Topology-derived compact source/target mappings, input/output compact ports,
indegree normalization, stable edge-weight views, normalized broadcast slot keys,
batch row indices, and scalar gains are computed once per forward pass. They are not
cached across forwards, so a lifecycle topology mutation cannot leave stale indices.

CUDA autocast constructs differentiable recurrent tensors in the active compute
dtype, including synaptic weights before scatter/index-add operations. Detached slow
statistics remain FP32, and edge-attention denominators and entropy reductions use
FP32 before messages return to the recurrent dtype.

The documented `fast_weight_gain = 0` ablation bypasses fast-key/value projections,
outer-product memory updates, and memory reads entirely. Positive configured gains
retain the original differentiable path; the primary Tiny Shakespeare baseline keeps
it disabled.

Cell implementations live in `sequence_cells.py`. LSTM neurons retain a protected
cell state, ESN neurons use a fixed orthogonal reservoir, and transformer neurons
attend over four private temporal message slots. Those private slots are distinct
from the field's dendritic and broadcast attention.
