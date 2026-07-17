# Sequence experiment lifecycle

`SequenceExperiment` trains one persistent spatial organism on associative recall,
synthetic autoregressive language, or cached character-level Tiny Shakespeare. It
uses masked cross-entropy so recall is judged only at the query response, while
language is judged at every next-token position. Synthetic held-out evaluation uses
fresh generator streams; corpus evaluation uses the fixed validation split.

Recall begins with one binding and advances to two and three only when the most
recent 24 training batches exceed 90% accuracy. Tiny-language accuracy is reported
on the context-dependent verb and object positions, while loss still trains every
next token. This prevents easy EOS/frequency predictions from masquerading as
context learning.

Task gradients update the shared neuron rule, token/output embeddings, per-site
genotypes, readout, and active synaptic weights jointly. Gradient×state and
gradient×weight credit are also recorded as slow local utility. After warm-up, the
same energy homeostasis, birth/death, dendrite growth/pruning, forced lifecycle, and
lesion pathways used by MNIST become active.

Reward measures improvement over the uniform-loss baseline for the active vocabulary,
so a 66-character corpus and a ten-token synthetic task use comparable signed credit.

The displayed trial advances through token frames, feedback, then structural change.
All edge flow and neuron stimulation in those frames comes from the actual forward
pass.

Corpus tasks expose `set_prompt` and `generate_token`. Prompt installation encodes
only the trailing context window and replays it without training. Each generate call
samples one character at temperature 0.85, appends it, and recomputes the visible
trace. Generation never performs an optimizer or structural update. The snapshot
also reports the greedy next-token prediction so sampling can be distinguished from
the model's modal choice.
