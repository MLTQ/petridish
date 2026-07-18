# Experiment results

This notebook records controlled results produced by the two-GPU laboratory.
Raw JSON artifacts remain under the configured `PETRIDISH_BENCHMARK_ROOT`; the
frontend renders those measurements directly.

## Associative-recall architecture pilot — 2026-07-17

All four runs used `compact24`, seed 11, lifecycle disabled, the same initialized
substrate contract, and 160 optimizer updates on the RTX 2070 Super. The adaptive
curriculum began with one key/value binding and advanced to two bindings after
mastery.

| Cell rule | Time (s) | One-pair peak held out | Final pairs | Final held out | Final loss |
|-----------|---------:|-----------------------:|------------:|---------------:|-----------:|
| GRU | 35.46 | 100.0% | 2 | 47.92% | 0.94635 |
| LSTM | 36.32 | 100.0% | 2 | 51.56% | 1.15704 |
| ESN | 36.14 | 72.92% | 2 | 47.40% | 1.18472 |
| temporal transformer | 47.20 | 100.0% | 2 | 43.75% | 0.87524 |

### Interpretation

The pilot does not identify a winning cell rule. GRU, LSTM, and the temporal
transformer mastered the one-binding curriculum. After advancing to two bindings,
all four retained accuracy above the 25% four-value chance baseline but remained
far from reliable. The common degradation at the same transition points first to
retention, routing, training budget, or curriculum stability in the shared substrate
rather than to the recurrent cell family.

The next controlled sweep should keep the same seed and topology while extending
the two-binding training budget. A fixed two-binding control is also needed to
separate catastrophic curriculum transition from ordinary convergence speed.

## Legacy checkpoint migration — 2026-07-17

The live 4090 checkpoint at update 15,300 was loaded read-only on CPU through the
new architecture wrapper. Model state, all 31 populated optimizer states,
experiment state, and random-generator state restored successfully as a GRU with
64 model-state entries and last loss 2.821059. The migration did not write to or
pause the live trainer. The separate new-wrapper continuation test below verifies
the full save/restart boundary without mutating the legacy run.

## Fixed two-binding recall — 2026-07-17

All four architectures were initialized from scratch with `compact24`, seed 11,
lifecycle and structural mutation disabled, exactly two key/value bindings, and
800 optimizer updates on the RTX 2070 Super. Chance remains 25% because the response
is one of four value tokens.

| Cell rule | Time (s) | Held out @ 20 | Held out @ 400 | Final held out | Peak held out | Final loss |
|-----------|---------:|--------------:|---------------:|---------------:|--------------:|-----------:|
| GRU | 172.04 | 29.69% | 48.44% | 48.44% | 54.69% | 0.73972 |
| LSTM | 215.79 | 26.56% | 50.00% | 45.83% | 54.17% | 0.75380 |
| ESN | 235.69 | 23.44% | 53.65% | 53.65% | 54.17% | 0.73689 |
| temporal transformer | 310.06 | 29.69% | 54.17% | 48.44% | 54.17% | 0.72139 |

### Interpretation

Every architecture reaches the same roughly 50% attractor by update 400 and fails
to improve over the next 400 updates. The small peak spread and shared plateau rule
out ordinary budget shortage and provide no evidence that recurrent gating, a fixed
reservoir, or four private temporal-attention slots changes retrieval capacity in
the present substrate. The transformer's lower loss does not translate into higher
discrete retrieval accuracy and costs 80% more wall time than GRU.

The 50% attractor is consistent with reliably retaining one of two bindings. The
next instrumented sweep records held-out accuracy by queried binding position to test
that explanation directly. If confirmed, the next architectural intervention should
be sparse token-conditioned ownership of memory by physical neurons, not another
homogeneous recurrent cell replacement.

## New-wrapper checkpoint/resume — 2026-07-17

A fresh CPU GRU corpus organism trained update 0→1, atomically checkpointed, exited,
then restored in a second Python process and trained update 1→2. The append-only
metric log contains exactly updates 1 and 2, and the second process reported starting
at update 1. This verifies the new architecture wrapper's save/restart boundary in
addition to the read-only legacy-checkpoint migration above.

## Query-slot decomposition — 2026-07-17

GRU and temporal transformer were rerun for 400 fixed-two updates with atomic live
artifacts and held-out accuracy separated by which binding position was queried.

| Cell rule | Parameters | Peak CUDA | Time (s) | Overall | Query slot 1 | Query slot 2 |
|-----------|-----------:|----------:|---------:|--------:|-------------:|-------------:|
| GRU | 11,725 | 1.2868 GiB | 82.84 | 47.92% | 47.92% | 47.92% |
| temporal transformer | 12,337 | 1.6450 GiB | 109.88 | 47.92% | 47.92% | 47.92% |

The exact symmetry falsifies fixed primacy and recency explanations: neither model
always preserves a particular pair position. The transformer uses 5.2% more parameters,
27.8% more peak allocated memory, and 32.6% more wall time without changing retrieval.
The next diagnostic measures whether predictions remain inside the set of presented
values; high set coverage with 50% correctness would isolate lost key/value association.

## Presented-value decomposition — 2026-07-17

The fixed-two GRU control was repeated for 400 updates with each held-out prediction
classified as correct, another value presented in the same episode, or a value absent
from the episode. Final overall and per-slot accuracy were again 47.92%.

| Presented-value coverage | Correct | Presented distractor | Absent value |
|-------------------------:|--------:|---------------------:|-------------:|
| 100.00% | 47.92% | 52.08% | 0.00% |

This isolates the failure: the organism transports and retains the complete value set
but loses which key owns which value. Additional generic memory capacity, training
budget, or homogeneous cell complexity is therefore the wrong next variable. The next
intervention stores successor state in content-addressed physical owner neurons and
reads it back only through the queried token's input port.

### Mixed-state owner-memory result

The first neuron-owner intervention stored each successor input neuron's processed
hidden state. At 400 updates it again produced 47.92% correct, 52.08% presented-value
distractors, and 0% absent values (88.78 s, 12,254 parameters, 1.2933 GiB peak CUDA).
The organism ignored the added pathway and recovered the existing value-set shortcut.
A clean-token-value ablation is required before rejecting content-addressed physical
ownership itself.

### Clean-token owner address map

The clean-token variant also returned to the shortcut (54.17% correct at update 400,
100% presented-value coverage). Its initially collision-free map had 10/10 distinct
winning owners, 0.098 cross-token overlap, and 15.3% mean peak ownership. Training
sharpened but collapsed it to 7/10 owners, 0.160 overlap, and 42.1% peak ownership.
This motivates a separated-address control before judging the downstream pathway.

### Separated physical owners

The separated-address profile changed only address temperature (0.05) and added a
0.02 overlap/entropy penalty. At update 400 it retained 10/10 distinct physical
owners, 99.77% mean peak ownership, 0.000041 cross-token overlap, and 0.00395
normalized entropy.

| Overall held out | Query slot 1 | Query slot 2 | Distractor | Absent | Final loss |
|-----------------:|-------------:|-------------:|-----------:|-------:|-----------:|
| 63.54% | 64.58% | 62.50% | 36.46% | 0.00% | 0.67853 |

This is the first intervention to exceed every homogeneous baseline's roughly 55%
peak and it improves both query positions. Stable differentiated ownership therefore
contributes useful relational capacity; replication across a second seed and a longer
training budget are running before treating the effect as robust.

### Replication and long-run consolidation

Seed 12 did not replicate the 400-update gain: it ended at 44.79% despite 10/10
distinct owners, 99.22% peak ownership, and negligible overlap. The seed-11 run was
therefore extended rather than treating its earlier result as generally robust.

Seed 11 continued from 71.35% at update 440 to 100% held-out accuracy at update 820,
then remained perfect through update 1,200:

| Update | Training accuracy | Held out | Slot 1 | Slot 2 | Distractor | Loss |
|-------:|------------------:|---------:|-------:|-------:|-----------:|-----:|
| 440 | 57.67% | 71.35% | 75.86% | 67.62% | 28.65% | 0.67706 |
| 820 | 99.60% | 100.00% | 100.00% | 100.00% | 0.00% | 0.02359 |
| 1,200 | 100.00% | 100.00% | 100.00% | 100.00% | 0.00% | 0.00072 |

The final address map has 9/10 distinct argmax owners, 99.35% peak ownership, 0.02235
cross-token overlap, and 0.00929 entropy. Perfect binding therefore does not require
one unique argmax neuron per symbol; the full sparse ownership distribution retains
enough identity. A 1,200-update seed-12 replication is running to distinguish variable
convergence speed from lucky initialization.

The second long execution also solved the task: it reached 100% held out by update
760 and remained perfect through update 1,200 with loss 0.00033. Its final owner map
has 9/10 distinct argmax owners, 99.78% peak ownership, 0.02225 overlap, and 0.00381
entropy. Two independent long executions therefore converge to reliable relational
binding, while 400 updates are insufficient for a robust conclusion.

### Reproducibility caveat

Two nominally identical seed-12 CUDA executions diverged before update 400: the short
run ended at 44.79%, while the long execution had reached 59.38% by update 320. Python,
Torch, task, and evaluation generators are seeded, but CUDA indexed/scatter reductions
are not forced deterministic. The two successful long executions are independent
behavioral replications, not bitwise deterministic reproductions. A deterministic-mode
audit remains required before tight seed-to-seed variance claims.

### Deterministic-mode audit

PyTorch deterministic algorithms plus `CUBLAS_WORKSPACE_CONFIG=:4096:8` executed the
complete CUDA forward/backward path without unsupported-operation errors. Two fresh
100-update seed-11 runs produced identical canonical result JSON after removing only
wall-clock seconds; both SHA-256 hashes were
`fc29c30138c1b9780696ce1f1ab09b743072d56855405c9f76f69de94e691783`.
Future variance studies should use `--deterministic`; older artifacts remain valid
learning observations but are not bitwise reproduction controls.

## Deterministic three-binding recall — 2026-07-17

The separated-owner GRU was trained from scratch with three simultaneous key/value
relations, fixed difficulty, seed 11, and deterministic CUDA execution. No vocabulary,
field, topology, or cell-rule capacity was added relative to the solved two-binding run.

| Update | Training accuracy | Held out | Slot 1 | Slot 2 | Slot 3 | Distractor | Loss |
|-------:|------------------:|---------:|-------:|-------:|-------:|-----------:|-----:|
| 580 | 53.61% | 60.42% | 45.71% | 67.69% | 70.18% | 39.58% | 0.85464 |
| 920 | 90.65% | 95.31% | 90.32% | 96.67% | 98.57% | 4.69% | 0.25753 |
| 1,280 | 98.23% | 98.96% | 96.88% | 100.00% | 100.00% | 1.04% | 0.06772 |
| 1,600 | 99.84% | 100.00% | 100.00% | 100.00% | 100.00% | 0.00% | 0.00906 |

The final address map uses 8/10 distinct argmax owners, 94.97% mean peak ownership,
0.06678 cross-token overlap, and 0.01740 entropy. Perfect three-relation recall does
not require a rigid one-symbol/one-neuron lookup table; the learned sparse ownership
distributions retain identity. The next experiment lesions a competent relational
substrate and compares gradient-only recovery with lifecycle/topology repair from an
identical pre-lesion state.

## Matched radius-4 lesion and lifecycle recovery — 2026-07-17

A deterministic seed-11 fixed-three organism was trained for 1,200 updates, then
deep-copied with optimizer, graph, slow statistics, and RNG state intact. The first
attempt was discarded after the control exposed an isolation bug: zero structural
warm-up let the already-competent clone silently re-enable topology. The corrected
run holds control/static topology at generation zero; a unit regression protects
that condition.

The physical lesion removed 27/250 living cells and 204/867 active edges. It was not
functionally severe: immediate held-out accuracy was 98.44%, versus 99.22% in the
unlesioned control.

| Branch | Immediate | Minimum | Final | Final cells | Final edges | Births | Deaths |
|--------|----------:|--------:|------:|------------:|------------:|-------:|-------:|
| control | 99.22% | 94.53% | 99.48% | 250 | 867 | 0 | 0 |
| lesion, static topology | 98.44% | 97.40% | 100.00% | 223 | 663 | 0 | 0 |
| lesion, default lifecycle | 98.44% | 24.48% | 68.49% | 289 | 808 | 209 | 143 |

The lifecycle branch collapsed to 26.56% by update 20 after 24 births and 45 deaths,
and reached its 24.48% minimum at update 80. It partially recovered to 84.90% at
update 220 but ended at 68.49%. Deaths were overwhelmingly overload-classified:
127 overload, 12 starvation, and 4 maintenance. Thus the current lifecycle repairs
population count but destroys learned computation faster than gradient learning can
stabilize it. The next matrix separates lifecycle-without-lesion from lesion effects,
uses a stronger radius-8 physical intervention, and tests a lower-cadence bounded-
turnover lifecycle without tuning this recorded baseline.

## Discarded radius-8 lifecycle ablation matrix — 2026-07-17

These values are retained only as a diagnostic that motivated a finite repair phase;
they are not accepted comparative evidence. Five deterministic branches were cloned
from one 1,200-update fixed-three base. The
radius-8 intervention was matched across branches and removed 106/250 neurons plus
624/867 active edges, reducing immediate held-out accuracy to 23.18%.

| Branch | Initial | Minimum | Peak | Final | Cells | Edges | Births | Deaths |
|--------|--------:|--------:|-----:|------:|------:|------:|-------:|-------:|
| static control | 99.22% | 94.53% | 100.00% | 99.48% | 250 | 867 | 0 | 0 |
| lifecycle control, interval 8 | 99.22% | 29.17% | 99.22% | 49.74% | 301 | 886 | 222 | 171 |
| radius-8, static | 23.18% | 23.18% | 88.02% | 85.94% | 144 | 243 | 0 | 0 |
| radius-8, lifecycle interval 8 | 23.18% | 23.18% | 100.00% | 77.86% | 296 | 804 | 251 | 99 |
| radius-8, interval 32 / birth 4 / death 8 | 23.18% | 13.02% | 65.62% | 13.02% | 115 | 306 | 32 | 61 |

Default lifecycle is independently destructive: without a lesion it ends at 49.74%
versus 99.48% for the topology-frozen clone, with 161/171 deaths classified as
overload. After severe damage, however, it accelerates functional recovery: the
radius-8 lifecycle branch reaches 100% at update 100, while the static branch peaks at
88.02% and ends at 85.94%. Continued turnover then destabilizes the repaired branch.

Lower cadence with asymmetric caps is worse, not gentler: only 32 births replace 61
deaths, population falls to 115, and final accuracy is 13.02%. Mutation frequency is
therefore not the sole control variable. The data support a finite repair phase:
activate birth/death/topology after damage, then freeze mutation when competence
returns and let gradients consolidate the repaired substrate.

Audit of the subsequent repair-window run exposed a branch-order confound: newborn
genotype inheritance uses PyTorch's device-global RNG, which is not part of a deep-
copied experiment. Later lifecycle branches therefore received different mutation-
noise streams. All ablation and partial repair artifacts were moved out of the live
laboratory, explicit CPU/CUDA RNG restoration was added at each branch boundary, and
the matrix must be rerun before any numerical comparison above is accepted.

## Corrected radius-8 lifecycle matrix — 2026-07-17

The full five-branch matrix was rerun after restoring the same CPU/CUDA global RNG at
each clone boundary. The corrected severe-lifecycle prefix exactly matches the first
repair-window branch through update 60 across accuracy, cells, edges, births, and
deaths, validating branch-order independence.

| Branch | Initial | Minimum | Peak | Final | Cells | Edges | Births | Deaths |
|--------|--------:|--------:|-----:|------:|------:|------:|-------:|-------:|
| static control | 99.22% | 94.53% | 100.00% | 99.48% | 250 | 867 | 0 | 0 |
| lifecycle control, interval 8 | 99.22% | 29.17% | 99.22% | 49.74% | 301 | 886 | 222 | 171 |
| radius-8, static | 23.18% | 23.18% | 88.02% | 85.94% | 144 | 243 | 0 | 0 |
| radius-8, lifecycle interval 8 | 23.18% | 23.18% | 79.43% | 78.39% | 294 | 851 | 284 | 134 |
| radius-8, interval 32 / birth 4 / death 8 | 23.18% | 20.05% | 77.34% | 61.20% | 113 | 308 | 32 | 63 |

The accepted result is narrower than the discarded run: continuous lifecycle restores
physical population and connectivity but does not outperform gradient-only functional
recovery. It is also independently destructive without a lesion. Lower cadence plus
asymmetric caps reduces both repair and final accuracy. A focused corrected replication
of the 60-update repair-then-freeze branch is next; its pre-fix first-branch trace is
only hypothesis-generating even though its prefix matches the corrected matrix exactly.

## Corrected 60-update repair-then-freeze replication — 2026-07-17

The focused branch was rerun from scratch with deterministic kernels and explicit
CPU/CUDA branch-RNG restoration. Its complete prefix through the freeze boundary
matches the earlier first-branch trace exactly. Lifecycle and topology mutation were
disabled after recovery update 60; gradient learning continued to update 240.

| Update | Held out | Loss | Cells | Edges | Births | Deaths | Generation |
|-------:|---------:|-----:|------:|------:|-------:|-------:|-----------:|
| 0 | 23.18% | 2.32709 | 144 | 243 | 0 | 0 | 0 |
| 20 | 60.42% | 1.07865 | 149 | 348 | 24 | 19 | 2 |
| 40 | 54.17% | 1.36879 | 172 | 478 | 60 | 32 | 5 |
| 60, freeze | 66.41% | 0.82708 | 195 | 563 | 84 | 33 | 7 |
| 100 | 88.80% | 0.39848 | 195 | 563 | 84 | 33 | 7 |
| 140 | 99.74% | 0.09331 | 195 | 563 | 84 | 33 | 7 |
| 160 | 100.00% | 0.02515 | 195 | 563 | 84 | 33 | 7 |
| 240 | 100.00% | 0.00197 | 195 | 563 | 84 | 33 | 7 |

This is the first lifecycle policy to outperform both matched alternatives after the
severe lesion: static recovery ends at 85.94%, continuous default lifecycle at 78.39%,
and repair-then-freeze at 100%. Physical regrowth is useful, but only as a bounded
developmental phase; continued birth/death/topology mutation erodes the computation it
helps reconstruct. The result is one deterministic seed and one task, so future work
should replicate seeds and replace the fixed 60-update schedule with a training-only
competence/stability gate rather than held-out feedback.

## Interactive Shakespeare checkpoint sample — 2026-07-17

The running 4090 organism's update-33,000 checkpoint was loaded into the paused CPU
viewer without interrupting training. The shared run-root bug was fixed first so the
viewer and laboratory discover the same atomic `latest.pt`. Three temperature-0.85
samples of 96 characters each took 132–143 seconds (about 0.7 characters/second):

```text
ROMEO:
Whome so ttoors
Ih for reet tetler, an itnerd-edeson ou Yhu duain.

Second Cerimen:
I uiv, mysal

KING LEAR:
Tat, who cuapyild be will comesit ag anlen.

CORIOLAN:
So htmel that, ord to lees,
Tteain tine o

What is love?

MENENIUS:
Now lhabalt, Which Jome to my ace trutked my lord.

KING RICHARD II:
My dood foow'd c
```

The organism has learned speaker/line structure, punctuation, and word-like local
statistics, but not dependable spelling, long-range syntax, meaning, or dialogue.
Generation currently replays the full 64-character context for every character;
persistent incremental inference is required for conversational latency.

## Persistent TinyStories graph/state audit — 2026-07-18

Two ESN token organisms were trained as continuous checkpoint lineages on the
128-token TinyStories corpus. The 2070 lineage used four microticks and a broadcast
workspace; the 4090 lineage used sixteen local-only microticks. Neither used lifecycle.

The first state-horizon evaluator began held-out evaluation from a fresh field and
only carried electricity between evaluation windows. Its “carried” label therefore
did not measure the checkpoint's actual persistent state. The evaluator was corrected
to clone every saved hidden/private/workspace tensor plus absolute position for each
counterfactual. The cold branch receives the same tokens without that state. Graph
reference, silence, and endpoint-rotation branches now begin from identical checkpoint
state clones as well. SHA-256 hashes of both checkpoints were identical before and
after the corrected evaluations.

| Lineage | Update | State age | Checkpoint | Cold | State delta | Graph ref | Silence delta | Rotate delta |
|---------|-------:|----------:|-----------:|-----:|------------:|----------:|--------------:|-------------:|
| 2070, broadcast | 2,500 | 80,000 | 36.33% | 36.33% | 0.00 pp | 29.69% | 0.00 pp | 0.00 pp |
| 4090, local | 1,500 | 96,000 | 28.71% | 22.46% | +6.25 pp | 41.80% | 0.00 pp | −1.37 pp |

The matched graph batches are distinct from the state-ablation batches, so graph
reference accuracy must be compared only with its silence/rotation counterfactuals.
On the 4090 graph, silencing preserved accuracy but improved loss by 0.06099; rotating
sources reduced accuracy by 1.37 points while leaving loss essentially unchanged.
Endpoint organization therefore has a small causal effect, but weighted graph traffic
does not yet improve discrete prediction. The graph is weak, not catastrophically
harmful under the organism's real electrical state.

The corrected 4090 horizon curve improves with longer continuation from the actual
checkpoint: h1 19.43%, h2 19.14%, h4 23.73%, h8 25.29%, and h16 29.39%. The 2070
curve remains flat at 32.23%, consistent with its broadcast shortcut and zero graph
ablation effect. The next same-lineage intervention is prune-only topology: allow
signed-utility pruning of the 4090 organism's 5,327 eligible edges while forbidding
replacement growth, then remeasure state and graph causality before enabling cell
death or renewed axon growth.

## Same-organism prune-only consolidation — 2026-07-18

Each live organism continued for exactly 100 updates with its identity, cells,
positions, weights, optimizer, RNG, corpus position, and electrical state intact.
Topology used the `prune_only` profile: signed-utility pruning remained active while
all replacement growth and lifecycle events were disabled. The post-phase evaluations
were read-only counterfactuals cloned from the resulting checkpoint state.

| Lineage | Updates | State age | Edges before → after | Grown | Pruned before → after | Reachable outputs |
|---------|--------:|----------:|---------------------:|------:|----------------------:|------------------:|
| 2070, broadcast | 2,500 → 2,600 | 83,200 | 16,676 → 14,884 | 7,764 → 7,764 | 8,192 → 9,984 | 64 / 64 |
| 4090, local | 1,500 → 1,600 | 102,400 | 16,977 → 15,441 | 3,201 → 3,201 | 3,328 → 4,864 | 64 / 64 |

The continuity counters are important: state age increased monotonically, organism IDs
did not change, cumulative growth was exactly constant, and no cells were born, died,
stunned, or recovered. This was consolidation of an existing organism, not a reset or
reinitialization.

| Lineage | Checkpoint | Cold | State delta | Graph ref | Silence accuracy delta | Silence loss delta before → after | Rotate accuracy delta before → after |
|---------|-----------:|-----:|------------:|----------:|-----------------------:|----------------------------------:|-------------------------------------:|
| 2070, broadcast | 36.33% | 36.33% | 0.00 pp | 29.69% | 0.00 pp | +0.00470 → +0.00502 | 0.00 → 0.00 pp |
| 4090, local | 28.71% | 21.88% | +6.84 pp | 41.80% | 0.00 pp | −0.06099 → +0.01696 | −1.37 → 0.00 pp |

The 2070 broadcast organism remains graph-independent: pruning removed 1,792 edges
without making endpoint identity affect accuracy. Its surviving graph has only a tiny
positive loss contribution, consistent with computation continuing to bypass it
through the broadcast workspace.

The 4090 local organism improved in the intended direction. Before pruning, silencing
the graph *improved* loss by 0.06099; afterward, silencing it *worsened* loss by
0.01696. The 1,536 removed edges therefore changed weighted graph traffic from mildly
harmful to mildly useful while preserving full output reachability. Its accumulated
electrical state also retained a +6.84-point held-out advantage over the same-token
cold branch. Accuracy is still low and graph ablations do not yet flip enough argmaxes
to change discrete accuracy, so this is evidence for selective consolidation rather
than a solved routing mechanism. The next intervention should preserve this lineage
and test whether a fixed-topology learning phase strengthens the small loss-level
causal effect before any cell lifecycle or renewed growth is enabled.

## Same-organism fixed-topology consolidation — 2026-07-18

Both post-prune organisms continued for another 100 updates with topology and
lifecycle mutation disabled. The saved connectome still conducted messages and all
ordinary weights/dynamics remained trainable. Organism IDs, cells, edges, cumulative
turnover counters, optimizer state, corpus position, and electrical state continued
from the prune-only checkpoints.

| Lineage | Update | State age | Edges | Cells | Checkpoint | Cold | State delta | Silence loss delta | Rotate loss delta |
|---------|-------:|----------:|------:|------:|-----------:|-----:|------------:|-------------------:|------------------:|
| 2070, broadcast | 2,700 | 86,400 | 14,884 | 2,237 | 36.33% | 36.33% | 0.00 pp | +0.00123 | +0.00361 |
| 4090, local | 1,700 | 108,800 | 15,441 | 2,237 | 28.71% | 28.32% | +0.39 pp | −0.02139 | +0.03817 |

The fixed phase did not strengthen the 4090 graph's small post-prune benefit. Its
saved-state accuracy advantage contracted from +6.84 to +0.39 points and graph
silencing again improved loss. Rotating graph sources still worsened loss, suggesting
endpoint organization carries information even though the aggregate weighted traffic
is not reliably helpful. The graph batches contain only 256 predictions and advance
with checkpoint sampler state, so before/after sign changes are not yet a robust
estimate. Future causal audits require a larger, checkpoint-independent fixed
validation slice.

## 128-wordpiece unknown-token confound — 2026-07-18

The repeated replacement-character generations were traced to the tokenizer rather
than the renderer. Vocabulary index 1 is `<unk>`, and the decoder intentionally renders
it as `�`. With only 128 wordpieces, every out-of-vocabulary piece collapses into that
single class. Direct measurement found `<unk>` at 30.43% of the training stream and
30.64% of validation; it is the modal token in both. The reported 30.64% unigram
baseline is therefore unknown-token prediction, not language competence.

These lineages remain informative for persistent-state and routing experiments, but
their 29–36% top-one accuracy cannot be presented as meaningful token prediction.
The next corpus lineage must use a complete byte vocabulary or byte-fallback tokenizer
with no aggregate unknown class. Generation diagnostics must also report special-token
rates so future modal collapse is explicit rather than hidden behind decoded text.

## Byte-complete TinyStories lineages — 2026-07-18

A new UTF-8 byte task replaces the aggregate unknown class with exactly 256 complete
byte targets while retaining the same distributed 64-cell input/output population
code. Its validation unknown rate is exactly zero. On 2,250,261 validation bytes, the
space-only unigram baseline is 19.09% accuracy / 3.08455 loss and the train-fitted byte
bigram is 31.86% / 2.28177. These, rather than uniform 1/256 chance, are the required
language controls.

Two new organisms were launched from scratch because their token semantics differ
from the preserved wordpiece lineages:

| Lineage | GPU | Microticks | Broadcast | State lanes | AMP | Initial topology | Target |
|---------|-----|-----------:|----------:|------------:|-----|------------------|-------:|
| `byte-nca-2070-esn4-broadcast-r090-lanes2-lr025` | 2070 | 4 | 0.35 | 2 | off | adaptive after warm-up | 1,000 |
| `byte-nca-4090-esn16-local-r090-lr025` | 4090 | 16 | 0.00 | 1 | bfloat16 | adaptive after warm-up | 1,000 |

Both use ESN cells, batch one, 64-byte contexts, retention 0.9, learning-rate scale
0.25, no lifecycle, and seed one. The first atomic checkpoints independently record
`tokenizer_profile=byte` and all 256 vocabulary entries. At update 287 the 2070
control's rolling accuracy was 18.90%; at update 104 the 4090 local organism was
17.07%. Both remain below the unigram baseline, so this early modal learning is not
credited as contextual prediction. Fixed-seed held-out, state, and graph audits begin
at update 500 without resetting or pausing either organism.

The first byte checkpoints did not yet learn language. The 2070 lineage completed
1,000 updates with 32,000 tokens of persistent electrical age per lane. Its larger
1,024-byte read-only audit measured 19.82% accuracy / 3.08944 loss: only +0.74 points
above the unigram top-one baseline and far below the 31.86% bigram control. Generation
collapsed to spaces. Silencing the graph worsened loss by 0.00343, rotating sources by
0.02057, and reassigning weights among each neuron's unchanged dendrite sources by
0.02981. These are small below-argmax graph effects, not useful text prediction. The
broadcast ablation improved loss by 0.00158, so the configured shortcut was slightly
harmful on this slice.

At update 500, the 4090 local lineage measured 19.53% / 3.21387 on its scheduled
256-byte audit. Its saved electrical state beat the zero-state ablation copy by
1.56 points and 0.03182 loss. Graph silence worsened loss by 0.11750; globally rotating
sources and reassigning weights within fixed incoming source sets produced much larger
loss changes, 4.36250 and 2.35949 respectively. The slice is too small to treat those
magnitudes as stable estimates, but it directly falsifies the claim that the saved
connectome is irrelevant to propagation. The organism remained on the same lineage,
with all 17,130 edges and 32,000 tokens of electrical age represented in its checkpoint.

A supervisor deployment inadvertently signaled both trainers while they were active.
Their signal handlers completed the current indivisible update and atomically saved
the exact lineages at updates 678 and 249; continuation restored the same organism IDs,
graphs, optimizer moments, RNG streams, corpus cursors, and electrical tensors. No
checkpoint rollback occurred. Deployment now refuses to restart the service while any
persistent trainer is active.

The next same-lineage curriculum is a deterministic repeated byte prefix. It changes
only which existing corpus experiences recur: checkpoint-owned cells, positions,
dendrites, weights, optimizer, RNG, cursor, and electrical memory remain continuous.
Validation stays the complete 2.25-million-byte split, and shard-fitted unigram/bigram
controls remain visible. This tests whether the organism can first overfit a bounded
language distribution before scaling experience, without discarding the structure it
has already grown.

### Persistent 8,192-byte curriculum phases

The 2070 broadcast lineage continued from update 1,000 to 1,500 on an 8,192-byte
repeated prefix with adaptive topology. Its organism ID remained
`organism-b2505376398a491e8cf4150a5daf3fab`; both electrical lanes advanced from age
32,000 to 48,000 tokens and their saved stream cursors ended at 453 and 5,647 within
the shard. The checkpoint retained all optimizer and graph state.

This adaptive phase did not learn the shard or transfer to full validation. The
scheduled 256-byte audit measured 19.53% accuracy / 3.12801 loss versus the shard-fitted
unigram 19.09% / 3.11355 and bigram 31.42% / 2.91118. Generation remained sixteen
spaces. Meanwhile, cumulative topology turnover reached 7,627 grown and 8,192 pruned
edges, leaving 16,565 active edges and all 64 outputs reachable. Graph silence,
source rotation, within-destination weight reassignment, and broadcast silence each
slightly *improved* loss. The actively churning graph was therefore harmful on this
slice rather than a useful route to language. The same checkpoint is now continuing
with that resulting graph fixed, isolating consolidation from further endpoint churn.

The 4090 local lineage reached update 1,000 on the full corpus with the same organism
ID `organism-3b2405d533c548e0ab2666a8d8ff9987`, electrical age 64,000, and all 17,130
edges intact. Accuracy remained modal at 19.53% and generation collapsed to spaces.
However, causal graph organization was reproducible below the argmax boundary:
silence worsened loss by 0.04332, global source rotation by 1.20878, and weight
reassignment within fixed incoming source sets by 0.52419. This distinguishes
meaningful route/weight organization from language competence. The exact organism is
now continuing on the same 8,192-byte shard with topology fixed; its first continued
update advanced electrical age from 64,000 to 64,064 instead of reinitializing it.
