# test_sequence.py

These tests protect the sequence ladder's scientific assumptions: semantic ports are
permuted or direction-reversed, generated recall targets match the queried binding at
all curriculum sizes, recurrent token frames retain a differentiable path to synaptic
weights, and live snapshots align sparse cells/edges with readable sequence metadata.

The compact 20×20 fixture checks contracts rather than learning quality. Reproducible
learning curves belong to the benchmark command because convergence tests would make
the normal unit suite slow and hardware-sensitive.

The corpus fixture avoids network access while protecting dynamic vocabulary-sized
ports, prompt installation, single-token generation, and interactive task
serialization. Explicit geometry tests prove Tiny Shakespeare's 66 input and 66
output ports each occupy one unique boundary column on 68×68, preserve graph-layout
semantic order, and keep 68 unavailable to other tasks.

The trace-free regression requires optimizer updates to advance metrics and examples
without replacing the visible frame buffer, then verifies an explicit refresh rebuilds
one current token/feedback/structural trace.

Streaming regressions require one callback per genuinely computed token with aligned
logit shape, then verify a visual optimizer update reports forward, backward,
optimizer, local-credit, and lifecycle work and finishes on its measured structural
frame.
The backward contract requires an initial phase boundary plus one actual autograd
hook callback for every retained token state, ending at complete backward progress.

The homogeneous architecture regression runs GRU, LSTM, ESN, and temporal-transformer
rules through the same physical graph and verifies finite logits plus synaptic gradients.
Checkpoint migration protects pre-architecture GRU model keys while leaving unrelated
state untouched.

The distributed-token regression builds a small wordpiece corpus, verifies that its
vocabulary is encoded through fixed 64-cell population banks, and proves that a
sequence split across incremental calls produces the same logits as one uninterrupted
forward pass. The excitotoxicity regression verifies stun, seeded recovery without
edge deletion, and lethal classification only after accumulated damage crosses the
configured threshold.

Benchmark artifact replacement is required to be atomic so the polling laboratory
sees either the previous complete JSON document or the next one, never a partial write.
Recall evaluation also reports one accuracy per queried binding slot so aggregate
plateaus can be assigned to primacy, recency, or mixed retrieval failure.
Presented-value, distractor, and absent-value rates further distinguish lost content
from lost key/value association.
The neuron-owned binding-memory regression verifies that owner addresses receive task
gradients when enabled, exercises the clean token-value ablation, and verifies that
the default model omits the intervention entirely.
Owner diagnostics are bounded and baseline models report no fictitious address map.
Address-separation regularization must backpropagate into physical owner addresses.
Matched recovery branches must deep-copy independent topology and replace the immutable
configuration consistently across experiment, model, and substrate before lifecycle.
The static-branch regression also proves that a competent cloned organism cannot
re-unlock topology during the recovery window.
Lifecycle branch configuration tests cover explicit cadence and per-generation birth
and death budgets used by the severe-lesion ablation matrix.
The global-RNG regression proves that clone branches can replay the same stochastic
mutation stream; owned experiment generators alone do not cover birth genotype noise.
