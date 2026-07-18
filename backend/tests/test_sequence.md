# test_sequence.py

These tests protect the sequence ladder's scientific assumptions: semantic ports are
permuted or direction-reversed, generated recall targets match the queried binding at
all curriculum sizes, recurrent token frames retain a differentiable path to synaptic
weights, and live snapshots align sparse cells/edges with readable sequence metadata.

The compact 20×20 fixture checks contracts rather than learning quality. Corpus
fixtures use a deliberately sparse 68×68 field because their distributed 64-port
bank must retain the same one-column geometry as production. Reproducible learning
curves belong to the benchmark command because convergence tests would make the
normal unit suite slow and hardware-sensitive.

The corpus fixture avoids network access while protecting dynamic vocabulary-sized
ports, prompt installation, single-token generation, and interactive task
serialization. Explicit geometry tests prove Tiny Shakespeare's 66 ports and the
distributed token organism's 64 ports each occupy one unique boundary column on
68×68, preserve graph-layout semantic order, reject a 66×66 field rather than
wrapping its last two ports, and keep 68 unavailable to non-corpus tasks.
The headless-launch regression also prevents CLI overrides from silently replacing
the token task's lifecycle and pruning warm-up schedule with Shakespeare defaults.
It also requires the stability intervention to scale rule, readout, and synapse
learning rates together without changing their relative schedule.
The same launch fixture requires an explicit zero broadcast gain to survive into
configuration as a hard physical-workspace ablation.
The benchmark-scale helper independently proves that all three optimizer groups
receive the same bounded multiplier.
Named lifecycle-profile tests preserve the original baseline while proving the
balanced intervention keeps stun, recovery, starvation, and eventual death but
equalizes maximum births/deaths and lowers the empirically saturated pressures.
The replacement profile additionally couples each cycle's birth cap to its measured
deaths, preventing unused spatial capacity from causing unconditional population growth.
The headless configuration regression independently disables adaptive topology while
leaving lifecycle selection unchanged.
The continuation CLI regression additionally requires `--resume-plasticity` to fail
before task or model construction when no checkpoint exists, preventing a fresh
organism from being mislabeled as another phase of an established lineage.

The trace-free regression requires optimizer updates to advance metrics and examples
without replacing the visible frame buffer, then verifies an explicit refresh rebuilds
one current token/feedback/structural trace.
The finite-update regression injects a NaN loss and requires rejection before
backward, optimizer mutation, or training-step advancement.

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
forward pass. It also requires exact unigram and train-fitted bigram validation
baselines, with the contextual baseline outperforming global frequency. The
same fixture requires smoothed bigram loss to beat smoothed unigram loss. The
byte-token regression requires all 256 byte values, UTF-8 prompt round trips, no
special or unknown class, zero validation unknown rate, and rejection of a truncated
byte vocabulary. The
repeated-shard regression keeps the complete validation split, interprets an existing
corpus cursor modulo a deterministic training prefix, and rejects a shard shorter than
one context. The checkpoint regression records that curriculum alongside the same
organism-owned electrical and structural state. The
continuous-experience regressions prove adjacent windows share the boundary token,
optimizer updates carry detached neuron state, checkpoints resume the exact lane and
runtime state, and cell death preserves each surviving neuron's state by physical
site. A multi-lane turnover regression additionally replaces one physical site,
requires every lane to retain survivor values and token age while initializing the
birth, preserves the active-lane pointer, and advances again without a stale-site
failure. They also require finite non-negative pre-clipping gradient norms for bias,
readout, token encoding, recurrent rules, and synapses, plus a bounded positive global
clip scale. The same checkpoint regression
enables structural plasticity as a new phase
without losing the population mask, dendrite sources, synaptic weights, edge ages,
edge utilities, genotypes, generation, competence history, turnover counters, lineage
metadata, or continuous runtime state. Scientific diagnostics report zero electrical age before the first carried
window. The paired held-out ablation must reuse the same contiguous tokens for carried
and cold state, label both conditions, publish signed accuracy and loss value, and
advance the evaluation RNG only once. The
graph-causality regression requires intact, silenced, and source-rotated evaluations
to use the same tokens, advance the sampler once, and restore all physical edges and
weights exactly. It also reassigns weights only among each destination's existing
sources and silences configured broadcast gain; zero-broadcast runs must report an
identical no-op branch. The
headless diagnostic helper must publish the signed accuracy difference used by both
scheduled and evaluate-only records. Electrical relaxation must preserve physical
sites and absolute age, leave retention-one state bit-identical, change state at 0.9,
remain finite, and survive checkpoints. The
state-horizon curve must report ordered window/token spans from identical validation
tokens while advancing the sampler exactly once. The
checkpoint-state ablation must seed every held-out counterfactual from a storage-
independent clone of the real saved hidden/private/workspace state, report its
absolute seed age, and leave the live runtime tensors unchanged. The
fixed-audit regression requires repeated checkpoint evaluations to use the same
validation slice, publish seed/batch/token counts, and restore the checkpoint sampler
bit-for-bit. It also requires the headline carried-state result and intact-graph
reference to use that same slice and agree exactly, preventing random-offset variance
from masquerading as a graph effect. The
active-training-shard audit must use the independent evaluation RNG, report its split
explicitly, and restore graph sources and weights after every counterfactual. The
disposable-auxiliary regression requires a nonzero historical setting to fail before
model, optimizer, cursor, or electrical-state mutation. Auxiliary-domain regressions
retain historical sampling/provenance compatibility, and weight/scope still round-trip
through checkpoint metadata for exact audit loading. The
trajectory audit must clone the exact next stream position and matching state lane,
report that lane, preserve the cursor, and restore every graph counterfactual. An
explicit lane audit must select its matching recurrent state and checkpointed stream
domain, remain sampler-read-only, and reject an out-of-range or non-trajectory lane. The
phase-continuation regression must allow prune-only topology without altering any
checkpoint-owned electrical, graph, optimizer, or RNG state, while allowing the
restored configuration's global gradient ceiling and structural growth budget to
change independently. The
state-lane regression must alternate two independent persistent trajectories at
batch one, retain both states, report their exact age range, and expose authoritative
structure/lifecycle gate reasons. Append-only expansion from both one and two lanes
must preserve every old cursor and runtime state, consume rather than reseed the
saved training RNG for new positions, co-locate new cursors with a CUDA-restored
checkpoint, reject shrinking, and survive a checkpoint round trip with added lanes
still cold. Diagnostics must separately measure active/cold lanes, tensor trajectory
count, unique cursor phases, and per-domain lane counts. Each domain diagnostic also
identifies its first representative lane and its own unique/minimum/maximum phase
coverage. Large append-only expansion must fill every missing cursor phase within the
destination stream domain before duplicating one, ignore unrelated replay domains
when choosing those new phases, report global and per-domain phase occupancy, and
leave every old cursor exact. Per-row stream tests require
independent wrap lengths, and legacy checkpoint expansion must infer the old shard,
retain every old cursor/state/domain exactly, assign the broader domain only to cold
new lanes, train across the mixed domains, and reject a curriculum smaller than any
preserved domain. Explicit carried-domain expansion must change only domain lengths,
retain model/optimizer/RNG/cursors/runtime objects exactly, and demonstrate that new
tokens appear at the old wrap boundary rather than through cursor remapping. Its
checkpointed origin lengths must produce exact novel-token fraction/row metrics. Pre-update process
failures must persist a newline-free, bounded record. The
excitotoxicity regression verifies stun, seeded recovery without
edge deletion, and lethal classification only after accumulated damage crosses the
configured threshold.
The structural-transaction cadence regression proves the trainer predicts every due
lifecycle or mutable-topology update before entering it, including the first warm-up
boundary and later intervals, while fixed topology does not trigger graph checkpoints.
The direct-routing fixture supplies eight different tokens at the same single position
with eight equally frequent targets, preventing position embeddings or class bias from
masquerading as sensory-to-output learning.
The context fixture enumerates every two-token XOR pair, masks the first position,
and verifies that holding either context or query constant still produces both targets.
The delayed-copy fixture holds the recall token and position constant while alternating
balanced targets, isolating persistence before contextual composition.
The persistent-stream fixture reuses one copy/invert rule across four supervised
queries and requires every position to remain balanced against rule-only, input-only,
position, and frequency shortcuts.
The latency-pipeline fixture shifts all four targets by exactly two token clocks and
balances delayed bits against contemporaneous inputs, proving that pipeline success
cannot come from reading the current bit instead.
The context-settling fixture instead masks two constant clocks after the rule and
keeps all targets aligned with current inputs, separating context propagation from
output latency.
The settled-pipeline fixture combines context clocks with delayed outputs and
decorrelated bits, eliminating the period-two shortcut identified in the settling
control.
The autoregressive-grammar fixture enumerates all rule/seed states, aligns seven
targets to the actual next stream symbol, and proves that rule-only, current-only,
previous-only, position, and class-frequency baselines remain at 25%.
The zero-broadcast regression requires a hard workspace bypass with no gradients into
its gain or projections, preventing a nominal ablation from learning to reactivate.
Headless diagnostics separately verify physical/conducting edges, one-token/context/
graph reachability ordering, and exact pruning counts. Fixed-prompt greedy generation
must be repeatable without consuming the training sampler or mutating interactive state.
The independent-context audit regression draws each context separately with the
evaluation RNG, starts only disposable probe activations cold, reproduces the exact
seeded sample, and restores both training/evaluation RNG plus the living recurrent
state and learned graph after every matched counterfactual.
The full-corpus cold-audit regression uses the same sampler/RNG/graph restoration
contract while drawing from the unsharded training tensor and publishing a distinct
split identity.
Same-phase metric-history reconstruction retains only the last bounded train records
matching both phase identity fields, ignores other record types/phases and a partial
trailing line, and preserves one phase rolling curve across a worker/GPU handoff.
The same reconstruction folds legacy per-window novelty measurements into exact
phase token/window totals and resumes directly from newer cumulative records.

Benchmark artifact replacement is required to be atomic so the polling laboratory
sees either the previous complete JSON document or the next one, never a partial write.
Recall evaluation also reports one accuracy per queried binding slot so aggregate
plateaus can be assigned to primacy, recency, or mixed retrieval failure.
All sequence evaluation reports supervised-position accuracy so repeated contextual
prediction can expose temporal decay directly.
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
