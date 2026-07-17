# Cellular sequence model

The sequence model embeds a persistent recurrent computation in the same physical
neuron field used by the MNIST experiment. Each vocabulary item owns one semantic
input port, but `GraphLayout` permutes those ports in space. A token is supplied by
stimulating its port; the token ID is not broadcast to every cell.

For each token the shared genotype-modulated GRU rule runs several local graph updates.
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
