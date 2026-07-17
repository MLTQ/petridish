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

`train_visual_update` performs one complete optimizer update and exposes progress as
it happens. Token callbacks install the current batch, prediction, confidence, and
measured frame. Autograd hooks on retained token states expose gradient×state neuron
credit as differentiation reaches each token in reverse traversal order. These are
real backward measurements; edge credit remains unavailable until the leaf synaptic
gradient is complete. Optimizer status therefore retains the last measured field,
while credit and lifecycle callbacks install their actual terminal frames. Completed
streamed updates remain on the structural frame instead of replaying already-seen
token states.

`train_updates` is the headless path: it bypasses frame playback and asks the model for no token-local graph
copies. It still performs the full forward/backward optimizer, local credit,
homeostasis, and scheduled structural work, but deliberately suppresses automatic
validation so throughput runs are interrupted only by explicit evaluation. It also
leaves viewer-only tokens, predictions, confidences, and next-token text untouched,
avoiding accelerator-to-CPU projection during optimizer updates. The last visible
trace remains stable until `refresh_visual_trace` replays the most recent training
sequence with current weights.

`evaluate_metrics` reports held-out loss and accuracy together while retaining the
historical accuracy-only `evaluate` API used by the live viewer. Both operate outside
benchmark timing and never mutate optimizer or topology state.

An optional CUDA bfloat16 autocast mode wraps the shared model forward path. Task
loss and held-out loss are reduced in FP32; the default live experiment remains FP32.

`enable_compile` wraps only the model forward callable. The original module remains
authoritative for optimizer parameters, state dictionaries, mutable substrate state,
and topology mutation; compilation is opt-in for controlled headless measurements.

Corpus tasks expose `set_prompt` and `generate_token`. Prompt installation encodes
only the trailing context window and replays it without training. Each generate call
samples one character at temperature 0.85, appends it, and recomputes the visible
trace. Generation never performs an optimizer or structural update. The snapshot
also reports the greedy next-token prediction so sampling can be distinguished from
the model's modal choice.
