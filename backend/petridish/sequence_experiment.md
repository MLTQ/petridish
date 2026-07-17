# Sequence experiment lifecycle

`SequenceExperiment` trains one persistent spatial organism on either associative
recall or tiny autoregressive language. It uses masked cross-entropy so recall is
judged only at the query response, while language is judged at every next-token
position. Accuracy, loss, perplexity inputs, and held-out evaluation use freshly
generated sequences rather than a memorized static dataset.

Task gradients update the shared neuron rule, token/output embeddings, per-site
genotypes, readout, and active synaptic weights jointly. Gradient×state and
gradient×weight credit are also recorded as slow local utility. After warm-up, the
same energy homeostasis, birth/death, dendrite growth/pruning, forced lifecycle, and
lesion pathways used by MNIST become active.

The displayed trial advances through token frames, feedback, then structural change.
All edge flow and neuron stimulation in those frames comes from the actual forward
pass.
