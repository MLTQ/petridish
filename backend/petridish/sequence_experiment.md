# Sequence experiment lifecycle

`SequenceExperiment` trains one persistent spatial organism on associative recall,
synthetic autoregressive language, cached character-level Tiny Shakespeare, or
token-level TinyStories. It
uses masked cross-entropy so recall is judged only at the query response, while
language is judged at every next-token position. Synthetic held-out evaluation uses
fresh generator streams; corpus evaluation uses the fixed validation split.

Corpus training supports two explicit experience modes. `continuous` advances each
batch lane through adjacent windows and carries the complete detached neuron/runtime
state between optimizer updates; detachment bounds gradient memory but does not clear
activation, private cell memory, workspace, or fast/binding state. `windowed` samples
unrelated contexts and reinitializes only fast electrical state as a cold-start
control. Learned rules, genotypes, synapses, topology, and lifecycle state persist in
both modes. Continuous validation similarly carries a fresh organism state across
held-out windows, and metrics record the selected mode. `evaluate_state_ablation`
replays identical contiguous held-out tokens with state carry disabled, restores the
evaluation RNG to a single-evaluation advance, and reports the causal accuracy
difference without changing training state.
`evaluate_graph_ablation` similarly replays one matched held-out stream with active
synapses silenced and with conducting source endpoints deterministically rotated.
It restores every source, weight, diagnostic cache, and RNG state, measuring whether
the actual learned connectome causally improves prediction rather than merely existing.
Continuous training may set a bounded `state_retention` at truncation boundaries.
This models homeostatic electrical relaxation, not organism reset: current state is
mixed with the same physical cells' resting field while topology, synapses, genotype,
parameters, stream position, and survivor identity remain intact. Evaluation applies
the identical retention schedule.
`state_lanes` adds round-robin persistent electrical trajectories without increasing
the tensor batch. Every lane advances through its own contiguous corpus position and
retains its own complete runtime state; all lanes share one physical substrate,
parameters, synapses, genotype, optimizer, and lifecycle. This tests trajectory
diversity on memory-limited GPUs without creating an ensemble or resetting a lane.
Topology policy is phase-local and checkpointed: fixed continues routing through the
existing graph, adaptive permits pruning plus growth, and prune-only permits signed-
utility pruning while forbidding replacement growth. None of these policies reset
electrical state, weights, optimizer moments, cells, or corpus position.
Held-out checkpoint-state evaluation begins with a tensor-cloned copy of the
organism's actual saved hidden/private/workspace state and absolute token position.
The matched cold branch begins without that electricity; both receive identical
tokens and neither can alter training state. Graph reference, silence, and endpoint
rotation branches likewise begin from identical checkpoint-state clones.
`evaluate_state_horizons` replays one held-out stream from that checkpoint state while
bounding additional electrical carry to 1, 2, 4, 8, or 16 context windows. Every
horizon receives identical tokens and the sampler advances only once, producing a
measured memory-lifetime response curve.

Recall begins with one binding and advances to two and three only when the most
recent 24 training batches exceed 90% accuracy. Tiny-language accuracy is reported
on the context-dependent verb and object positions, while loss still trains every
next token. This prevents easy EOS/frequency predictions from masquerading as
context learning.

Controlled benchmark construction may set both an initial binding count and a
curriculum maximum. Setting both to two creates a fixed two-binding control without
changing the default adaptive live experiment.

Task gradients update the shared neuron rule, token/output embeddings, per-site
genotypes, readout, and active synaptic weights jointly. Gradient×state and
gradient×weight credit are also recorded as slow local utility. After warm-up, the
same energy homeostasis, birth/death, dendrite growth/pruning, forced lifecycle, and
lesion pathways used by MNIST become active.
Non-finite loss is rejected before backward, and a non-finite total gradient norm
is rejected before the optimizer or homeostatic state can mutate. Failed long runs
therefore preserve their last finite checkpoint instead of applying NaN gradients
for several progress intervals.

Adaptive topology and lifecycle are independently gated. A fixed-connectome control
continues optimizing synaptic weights and neuron rules but can never unlock edge
growth or pruning, even after a learning plateau.

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
It also reports accuracy for every supervised sequence position. Persistent-stream
controls use this to expose context decay across repeated predictions instead of
hiding a weak final position inside aggregate accuracy.
For associative recall, the same evaluation forward passes also report accuracy by
queried binding slot. This distinguishes first-pair memory, recency, and mixed failure
modes that aggregate accuracy cannot identify.
They also separate predictions into the correct value, another value that appeared in
the sequence, and an absent value. High presented-value coverage with high distractor
errors means the organism retains the value set but loses key/value binding identity.

An optional CUDA bfloat16 autocast mode wraps the shared model forward path. Task
loss and held-out loss are reduced in FP32; the default live experiment remains FP32.

`enable_compile` wraps only the model forward callable. The original module remains
authoritative for optimizer parameters, state dictionaries, mutable substrate state,
and topology mutation; compilation is opt-in for controlled headless measurements.

Corpus tasks expose `set_prompt` and `generate_token`. Prompt installation encodes
only the trailing context window once. Each generate call samples one character or
wordpiece at temperature 0.85, advances the saved organism runtime state by that one
token, and records its measured frame. Generation never performs an optimizer or structural update. The snapshot
also reports the greedy next-token prediction so sampling can be distinguished from
the model's modal choice.

`greedy_completion` runs a deterministic incremental continuation for unattended
checkpoint diagnostics. It preserves the training sampler, interactive prompt state,
and model train/eval mode, so measurement cannot alter the organism's learning path.

Training records stun and recovery events separately from births/deaths. Excitotoxic
death counts only cumulative damage; transient overload is not classified as death.
Exact edge-growth and pruning totals are retained independently from the bounded
visual event stream and survive checkpoints.
Population changes reconcile continuous runtime state by physical site, retaining
survivors and initializing only births before the next token window.

Corpus construction primes the viewer with only four measured tokens; this avoids a
full training-context forward pass while the runtime lock is switching experiments.
Training batches and explicit prompts retain their configured context length.
