# Cellular sequence model

The sequence model embeds a persistent recurrent computation in the same physical
neuron field used by the MNIST experiment. Each vocabulary item owns one semantic
input port, but `GraphLayout` permutes those ports in space. A token is supplied by
stimulating its port; the token ID is not broadcast to every cell.

The token cellular-language variant intentionally replaces that legacy interface.
A learned token vector writes a distributed code across 64 sensory neurons. Sixty-four
output neurons are decoded to one hidden-width population code and compared with the
tied vocabulary codebook, so adding vocabulary entries does not add physical neurons.

For each token the selected shared genotype-modulated recurrent rule runs several
local graph updates. GRU is the preserved baseline; controlled homogeneous LSTM,
ESN, and temporal-transformer rules share the same physical graph and readout.
Queries, keys, values, emission gates, weighted directed dendrites, measured traffic,
and local attention are identical in spirit to the classifier. Crucially, neuron state
is retained across token boundaries. The vocabulary-sized output bank is read after every token,
which supports both a final delayed-recall target and autoregressive next-token loss.

`SequenceRuntimeState` carries hidden state, architecture-private memory, broadcast
workspace, optional fast/binding memory, and absolute token position between calls.
Interactive generation therefore consumes only the newly sampled token after the
prompt has initialized the organism; it does not replay the context window.
`SequenceRuntimeState.detached` preserves that complete electrical state while
cutting autograd history at a truncated-backpropagation boundary. If lifecycle
changes the population, `reconcile_runtime_state` maps every survivor by physical
site, initializes only newborn state, and retains workspace and address memories;
cell death therefore never resets the surviving organism.
`SequenceRuntimeState.cloned_detached` additionally copies every tensor for held-out
counterfactuals. Evaluation branches can therefore start from identical checkpoint
electricity without sharing mutable storage with the living training state.
`relax_runtime_state` optionally mixes electrical state toward each same physical
neuron's genotype/role/homeostasis-defined resting state and decays private/workspace
memories by the same coefficient. It preserves sites, structure, learned parameters,
and absolute electrical age; retention one is an exact no-relaxation control.

Stunned neurons retain their private state and physical dendrites but are gated out
of external drive, sending, receiving, recurrent updates, and readout until recovery.
Their incident edges are not mistaken for deleted structure or pruning candidates.

A small low-rank broadcast workspace implements the advertisement channel without an
all-pairs attention matrix. Neurons softly write content-selected values into shared
slots and selectively read them during the recurrent update. This path is ephemeral
at the edge level but its slots persist within a sequence using configurable decay.
It does not pretend to be an axon; persistent weighted dendrites still own the
structural graph. `broadcast_gain = 0` is the controlled no-workspace ablation.
That zero configuration hard-bypasses workspace writes, reads, and gradients; the
trainable gain cannot silently reactivate an experiment labeled broadcast-off.

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

`binding_memory_diagnostics` measures the learned address map without running a task
batch: distinct winning physical owners across the vocabulary, normalized address
entropy, cross-token attention overlap, and mean peak ownership. These distinguish
address collision/diffusion from downstream failure to use a clean retrieval.
Positive `binding_address_regularization` penalizes squared cross-token attention
overlap plus a smaller normalized-entropy term. It encourages differentiated, sparse
physical owners but never specifies which neuron owns a token or what value it stores.

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
