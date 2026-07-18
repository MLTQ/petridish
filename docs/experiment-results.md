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

### Progressive same-organism overfit control

After both 8,192-byte phases failed, the 2070 organism continued again with the same
cells, 16,565-edge fixed graph, optimizer, and electrical state but only 128 repeating
training bytes. This is two 64-byte windows; the in-shard unigram and fitted bigram
baselines are 21.26% and 43.31% respectively. The first hundred updates remained near
frequency prediction at 21.09%, but the final hundred reached 79.83% accuracy / 1.20474
loss. The final 160-update average was 68.23% / 1.47332. This exceeds the one-byte
bigram ceiling and proves the persistent organism can learn longer conditional
structure when the curriculum is sufficiently bounded.

The result is genuine overfit rather than language generalization. Full-validation
accuracy fell to 8.98% / 4.25289 loss, while generation changed from sixteen spaces to
` f ll  frll  oal`. Broadcast silence worsened loss by 1.96877, whereas graph silence
improved it by 0.82746. The conditional solution therefore lives primarily in the
broadcast workspace; this particular physical connectome remains an interfering
pathway.

The exact lineage next expanded to a 512-byte fixed-graph curriculum at electrical
age 80,000. Its in-shard unigram/bigram baselines are 20.16% / 35.03%. Across the first
14 updates it retained 30.47% phase-local accuracy, showing partial transfer rather
than a return to the frequency-only starting point. Pre-clipping gradient norms at
update 2,514 were bias 0.151, decoder 0.353, token encoder 0.906, cell rule 1.512, and
synapses 2.606. Conditional machinery is receiving substantial credit; the earlier
collapse cannot be explained by a missing gradient path alone.

The 512-byte phase subsequently converged. Average training accuracy rose from 22.13%
in updates 2,501–2,600 to 37.98% in 2,701–2,800 and 60.73% in 2,901–3,000; the final
160-update average was 58.32%. This is well above the 512-byte shard's 20.16% unigram
and 35.03% fitted-bigram baselines, demonstrating retained longer-context learning as
the curriculum expanded fourfold. Generation changed again to `tue l utddlmolla`.
Full-validation accuracy remained only 8.59%, so the result is still curriculum
memorization rather than transferable language.

The causal split stayed consistent: broadcast silence worsened loss by 1.39569 while
graph silence improved it by 0.58693. The final pre-clipping gradients were bias 0.084,
decoder 0.611, token encoder 1.467, cell rule 9.108, and synapses 5.761. Conditional
credit dominates the unigram-bias gradient, but the physical graph receives large
gradients despite contributing harmful traffic. The same lineage is now expanding to
2,048 bytes with its fixed 16,565-edge graph and 96,000-token electrical age intact.

The 4090 local organism's fixed-graph 8,192-byte phase was stopped checkpoint-safely
at update 1,301 after 301 updates because phase-local accuracy remained 18.99%. A
larger 1,024-byte read-only audit measured 19.82% validation accuracy and space-only
generation. Yet topology causality remained strong: graph silence worsened loss by
0.02182, global source rotation by 6.33700, and within-destination weight reassignment
by 0.63901. This is robust evidence that endpoint identity and synaptic pairing affect
computation, but not evidence of useful byte prediction. The same local-only lineage
is now on the 128-byte overfit control with its cells, 17,130-edge graph, optimizer,
and 83,264-token electrical history preserved.

### The 2,048-byte expansion exceeded the learned broadcast curriculum

The 2070 organism completed its next 500-update phase at update 3,500 without any
lineage or substrate replacement. Its organism ID remained
`organism-b2505376398a491e8cf4150a5daf3fab`, its two electrical lanes advanced from
96,064 to 112,000 tokens, and its fixed graph remained at 16,565 edges. Accuracy by
100-update bin was 23.64%, 19.61%, 20.02%, 19.94%, and 21.08%; the final 160-update
average was 20.71% / 2.92126 loss. Unlike the 512-byte phase, it never exceeded the
active shard's 30.68% fitted-bigram accuracy baseline. The initial high bin was
transient retention from the smaller curriculum rather than convergence on 2,048
bytes.

A corrected fixed-seed 16-batch/1,024-byte evaluation copy measured 14.84% accuracy /
3.28422 loss on the unchanged full validation split. Generation regressed to
`         e  o   `. Saved electrical state was slightly harmful relative to its
zero-state ablation copy. Graph silence improved loss by 0.09109 and accuracy by 3.71
points, whereas broadcast silence worsened loss by 0.45721 and accuracy by 6.64
points. This confirms that the broadcast workspace still carries the useful learned
conditional computation and the physical connectome still interferes. Endpoint
rotation was nearly neutral; within-destination weight reassignment worsened loss by
0.03018, a much smaller effect than broadcast removal.

Recent clipping was persistent but not dominated by catastrophic spikes: over the
final 160 updates the median global clip scale was 0.207, its tenth percentile was
0.138, and 2.5% of updates were scaled to 0.1 or below. The failed expansion is
therefore not explained solely by rare exploding gradients. The same organism is now
continuing on an intermediate 1,024-byte shard through update 4,250. This is a
curriculum-granularity recovery test: it preserves the 112,000-token recurrent state,
all cells and edges, learned parameters, optimizer moments, and RNG streams produced
by the failed 2,048-byte phase.

### Physical-graph sequence learning and its cursor-phase limitation

The intermediate 1,024-byte recovery succeeded on the unchanged 2070 lineage. Across
updates 3,501–4,250, 100-update-bin accuracy rose from 23.61% through 24.28%, 33.80%,
34.81%, 39.34%, 45.56%, and 51.58%; the final 50 updates averaged 50.44%. The final
160-update rolling result was 51.32% / 1.76821, above the shard's 20.53% unigram and
37.24% fitted-bigram training ceilings. The same organism ID, 16,565-edge graph, and
all cells persisted while its electrical lanes advanced from 112,064 to 136,000
tokens. Severe global clipping remained common: the final-160 median scale was 0.073,
its tenth percentile 0.048, and 88.75% of updates were scaled to 0.1 or below.

Three distinct read-only audits exposed what that competence means. On the exact next
saved trajectory, accuracy was 48.54% / 1.75134; graph silence reduced it to 9.47%,
broadcast silence to 18.65%, source rotation to 12.89%, and weight reassignment to
11.33%. Both the physical graph and broadcast workspace have therefore become useful
on the learned trajectory, reversing the graph's harmful effect in the earlier
512/2,048-byte phases. Yet random offsets within the same 1,024-byte shard achieved
only 19.14% graph-reference accuracy, and full validation achieved 13.96%. The model
learned an aligned recurrent trajectory rather than a phase-shift-invariant byte
distribution.

The 4090 local-only lineage produced the stronger mechanistic result. It continued
from update 1,301 to 1,801 on the 128-byte shard with zero broadcast, the same organism
ID `organism-3b2405d533c548e0ab2666a8d8ff9987`, its original fixed 17,130-edge graph,
and electrical age increasing from 83,328 to 115,264 tokens. Accuracy by 100-update
bin was 21.09%, 24.91%, 68.56%, 99.53%, and 100.00%; the final 160 updates were exactly
100% / 0.26184. This exceeds the shard's 21.26% unigram and 43.31% fitted-bigram
training ceilings without a global workspace.

The exact-trajectory causal audit was decisive across 1,024 predicted bytes: intact
graph plus saved electrical state achieved 100% at every position. Zeroing graph
weights reduced accuracy to 14.26%; rotating endpoints to 4.69%; reassigning each
destination's learned source/weight pairing to 4.00%; and retaining the graph but
starting from cold electrical state to 28.91%. Broadcast silence was a no-op because
the run's gain is exactly zero. The learned computation therefore resides jointly in
the physical dendritic graph, its learned synaptic assignment, and persistent local
electrical state.

This competence is also cursor-phase specific. A random-offset audit on the same
128-byte shard reached 26.76% with the intact graph versus 12.30% silenced, 5.08%
rotated, and 2.93% reassigned. The graph causally improves some shifted predictions,
but far less than the aligned 100%. Full validation was 9.08% / 4.90073 and cold state
was less harmful there, confirming genuine overfit. The next intervention should add
independently phased persistent experience lanes while preserving every existing
lane, rather than scaling the corpus immediately. This directly tests whether one
organism can consolidate the same local rule and graph across many cursor phases.

### Persistent lanes converted trajectory memorization into phase-robust graph learning

Both organisms then expanded their experience banks append-only to sixteen lanes.
The 2070 organism retained its two 136,000-token lanes and appended fourteen cold
lanes; the 4090 organism retained its 115,264-token lane and appended fifteen. Cells,
positions, dendrites, synaptic weights, model parameters, optimizer moments, random
streams, and existing runtime tensors were preserved. A device-placement fault in the
first launch attempt stopped before an optimizer update or checkpoint write; the
device-safe retry resumed the same checkpoint and was verified by exact cursor and
age advancement rather than reseeding.

On the unchanged 2070 organism, 1,000 phase updates over the same 1,024-byte shard
rose monotonically by 100-update bin from 25.17%, 30.38%, 35.05%, 43.94%, 47.86%,
53.88%, 60.13%, 64.50%, and 68.91% to 71.33%. The final-160 result was 70.95% /
1.18772, and the final four visits to every lane ranged from 61.72% to 85.55% with a
71.88% mean. This improvement was therefore distributed across newly appended lanes,
not confined to the two inherited electrical trajectories. The final-160 gradient
clip scale had median 0.103, tenth percentile 0.083, and 40.63% of updates at or below
0.1.

Read-only 1,024-token audits show that multi-lane competence remains genuinely
physical. The exact next trajectory reached 72.85% / 1.06254; graph silence collapsed
to 0.20%, global endpoint rotation to 9.96%, within-destination source/weight
reassignment to 9.96%, broadcast silence to 16.80%, and cold electrical state to
57.03%. More importantly, random offsets within the active shard reached 60.25% with
the intact graph, versus 0.10% silenced, 8.59% rotated, 9.67% reassigned, 16.80% with
broadcast silenced, and 46.39% from cold state. The earlier random-offset result was
only 19.14%; sixteen persistent lanes therefore converted an aligned replay solution
into broad cursor-phase competence while retaining strong dependence on the emerged
connectome.

This is not yet language generalization. The fixed-seed validation audit measured
12.21% aggregate accuracy / 4.07420 loss, with the matched graph-reference branch at
10.55%, below the 19.09% unigram baseline. Graph silence still fell to 0.10%, but
rotating or reassigning an already out-of-distribution graph changed little. The
generation sample was `oe ooee ao  lie `. Phase coverage solved an experience-alignment
problem, not the corpus-breadth problem.

The 4090 local-only organism independently produced an even cleaner result. Its
100-update bins rose from 21.64% and 28.34% through 63.97% and 96.86% to 99.98% over
the final 88 updates. It was checkpoint-stopped at update 2,289 after all sixteen
lanes achieved 100% over their last four visits; its final-160 rolling result was
99.36% / 0.29223. No broadcast workspace exists in this lineage.

The exact-trajectory audit remained 100% at every one of 1,024 positions. Graph
silence reduced accuracy to 7.91%, endpoint rotation to 6.15%, source/weight
reassignment to 9.18%, and cold state to 82.03%. The random-offset active-shard audit
reached 99.32%, versus 7.81% silenced, 6.84% rotated, 9.28% reassigned, and 80.47%
from cold state. Before the lane intervention, random-offset accuracy was 26.76% and
cold exact-trajectory accuracy 28.91%. A shared local rule and fixed dendritic graph
therefore learned essentially the complete 128-byte conditional distribution across
cursor phases without a global information shortcut. Validation remained 9.57% /
4.65768 with a 10.94% graph-reference branch, again separating physical computation
from transferable language.

All six final audits ran on disposable model/state copies. The 2070 checkpoint stayed
byte-identical at SHA-256 `a2d4552c35b145362a8d08b82e847e741216b079d09c24d23c31bd8222c01991`;
the 4090 checkpoint stayed byte-identical at
`818a9c58f84c9454ffde9bb21e6d423a7ecc8b68044ceacad0917a6736365c39`.

The next experiment branches the exact 2070 update-5,250 checkpoint into two separate
files with the same organism ID and SHA-256. Both preserve all sixteen lanes and move
to a 2,048-byte shard for 1,000 updates. `nca2070-u5250-fixed-2k` keeps the connectome
fixed; `nca2070-u5250-prune-2k` permits pruning but forbids replacement growth.
Lifecycle remains off in both so cell turnover cannot confound the pruning contrast.
Their first updates matched in loss, accuracy, gradients, population, graph, cursor,
and lane age. Pruning remains governed by the checkpointed competence/plateau gate,
so the two descendants stay matched until the organism's own structural policy
activates.

### Paired pruning removed graph redundancy without improving language

The exact update-5,250 organism checkpoint was copied into distinct inodes for the
fixed and prune-only descendants. Both copies matched the parent SHA-256
`a2d4552c35b145362a8d08b82e847e741216b079d09c24d23c31bd8222c01991`,
retained organism ID `organism-b2505376398a491e8cf4150a5daf3fab`, and produced
numerically matching losses, accuracies, gradient norms, state ages, cells, and edges
before structural activation. Both expanded to the same 2,048-byte shard for 1,000
updates; lifecycle stayed off.

The expansion initially transferred strongly, then exposed interference as new shard
positions arrived. Both descendants averaged 62.28%, 40.56%, and about 29.7% across
their first three 100-update bins, recovered through about 37%, 59%, and 67%, dipped
again with lane/shard phase, and finished at 63.81% fixed versus 63.96% prune-only over
the final 160 updates. The fitted 2,048-byte bigram baseline is 30.68%, so both still
learned longer conditional structure.

Pruning obeyed the checkpointed 82% competence / 500-update plateau policy rather
than beginning out of band. The plateau gate opened around update 5,750, after which
the branch removed up to 256 eligible dendrites every sixteen updates. It stabilized
at 13,737 edges from the parent's 16,565: 2,828 edges or 17.07% removed, three still
eligible, no replacement growth, no cell turnover, and all 64 outputs plus complete
context reach preserved.

The final causal audits show a small capacity cost rather than a benefit. Exact-next
trajectory accuracy was 75.39% fixed and 74.41% pruned; random-offset active-shard
accuracy was 62.30% and 61.43%. Cold-state trajectory accuracy was 59.77% and 59.86%.
Both remained strongly physical: trajectory graph silence reduced them to 3.22% and
6.25%, endpoint rotation to 6.54% and 7.13%, source/weight reassignment to 7.52% and
6.54%, and broadcast silence to 15.33% and 15.23%.

Validation remained unsolved and essentially tied. Fixed measured 9.18% / 4.22841;
pruned measured 9.67% / 4.17402. Their matched graph-reference branches were 11.33%
and 11.52%. The half-point aggregate accuracy and 0.054 loss differences are too
small for a generalization claim, but removing one sixth of the evolved connectome
without meaningful performance loss demonstrates substantial route redundancy.
Checkpoint audits were again non-mutating: the terminal fixed hash stayed
`55e39a49ce0751902af669097811b37594083b6bbd28e30d2349062a8683049e`,
and pruned stayed
`d615377f6c5ba0c548d07074048342365d0fc3f7b5e397dd966301028c6601dd`.

The next matched branch isolates lifecycle on the smaller pruned descendant. Both
copies freeze topology and retain the 2,048-byte curriculum and all sixteen lanes.
One keeps lifecycle off; the other uses replacement-balanced homeostasis, which
retains stun, probabilistic recovery, starvation/maintenance/excitotoxic death, and
birth but caps births to measured deaths so population cannot grow unconditionally.

### Replacement lifecycle preserved lineage but destroyed too much competence

The lifecycle control and intervention began from byte-identical copies of the
pruned update-6,250 checkpoint, SHA-256
`d615377f6c5ba0c548d07074048342365d0fc3f7b5e397dd966301028c6601dd`, with organism
ID `organism-b2505376398a491e8cf4150a5daf3fab`, 2,224 neurons, 13,737 directed edges,
and all sixteen electrical-state lanes. Both kept the topology policy fixed and the
same 2,048-byte experience shard; only lifecycle differed.

The first actual population replacement at update 6,532 exposed a persistence bug:
the currently active lane was reconciled to the changed site set, but the other
fifteen persistent lanes retained the previous population and failed on their next
forward pass. The worker stopped before saving this inconsistent in-memory mutation.
Its last atomic checkpoint remained the same complete organism at update 6,500,
SHA-256 `93ba22db34f1088828d277e9f168ab22b8438f00a9d1dfb701026e7d31febc95`.

The repair remaps every persistent lane after a population change, preserving each
survivor's lane-specific hidden/workspace state and age while initializing only true
newborns. Lifecycle/topology intervals are now storage transactions: the trainer
atomically saves immediately before entering the mutation and again after successful
completion. A fail-closed retry route fingerprinted and reloaded the update-6,500
checkpoint without adding a phase, changing policy, or constructing an organism. It
replayed the lost RAM-only updates, crossed the former update-6,532 fault with a
committed `populationChanged` checkpoint, and completed update 6,750 under the same
organism ID and phase 11.

Replacement pressure was scientifically negative at this intensity. The no-lifecycle
control finished its final 160 updates at 72.18% / 1.11706; replacement finished at
40.62% / 2.02411. Random-offset active-shard accuracy was 67.38% control versus
36.91% lifecycle, and exact-next trajectory accuracy was 66.99% versus 36.82%.
Electrical-state carry remained useful in-distribution, but its trajectory advantage
fell from 13.38 points to 5.66 points.

The mechanism was substantial rather than cosmetic. Replacement produced 164 deaths
and 164 births, 86 stun episodes and 80 recoveries, with five cells still stunned at
the final checkpoint. Population stayed at 2,224, while directed edges fell from
13,737 to 11,283 (17.86%) because death removes incident dendrites even under a fixed
topology policy. Mean energy fell to 0.577. The surviving graph still carried learned
computation: lifecycle trajectory accuracy fell from 36.82% to 3.22% when graph
weights were silenced, 6.35% under endpoint rotation, and 9.47% after within-target
source/weight reassignment. The control corresponding values were 66.99%, 6.74%,
6.45%, and 4.79%.

Validation did not justify the capacity loss. Aggregate accuracy changed from 8.98%
control to 10.06% lifecycle with losses 4.48813 and 4.12421; matched graph-reference
accuracy was 11.13% and 10.35%. This roughly one-point movement is too small and
internally inconsistent to claim generalization. Both fixed-prompt samples remained
unreadable (`s  oooe imm onn ` and `a r    eaao lwe `). All six causal audits were
read-only: terminal checkpoint hashes remained exactly
`8dc60e3e4521c22295a4c03626761c91d6e385f289c1853745e453476604bedc` for control and
`fdfd176a7bc2a8fed3f3db8ee59d3c37bcbf36c12ca8c05ee970fcfa6b3408ee` for lifecycle.

The result rejects aggressive replacement as the next route toward language. The
next paired intervention returns to the competent no-lifecycle checkpoint and tests
experience breadth directly: one exact descendant remains on the 2,048-byte shard,
while the other expands to 4,096 bytes. Both retain the emerged cells, pruned graph,
sixteen lanes, optimizer, RNG, and electrical histories with fixed topology and no
lifecycle. This asks whether incremental corpus breadth transfers conditional
computation before reintroducing gentler homeostasis.

### A hard breadth switch remapped trajectories instead of preserving replay

Two exact descendants of the no-lifecycle update-6,750 checkpoint began from SHA-256
`8dc60e3e4521c22295a4c03626761c91d6e385f289c1853745e453476604bedc`, organism ID
`organism-b2505376398a491e8cf4150a5daf3fab`, 2,224 cells, 13,737 fixed directed edges,
and sixteen active recurrent lanes. The control retained the 2,048-byte stream; the
treatment doubled it to 4,096 bytes. No topology or lifecycle mutation occurred.

The retained curriculum remained competent throughout and finished its final 160
updates at 81.75% / 0.75305. The doubled stream initially tracked it, then crossed a
large novelty band: phase rolling accuracy moved from 64.6% at update 6,906 through
58.4%, 55.8%, 44.0%, and 31.4%, before recovering to 46.83% / 1.92284 at update
7,750. The final individual window reached 87.5%, showing active learning rather than
optimizer failure, but consolidation was incomplete after 1,000 updates.

Causal audits separated retained knowledge from phase alignment. The control reached
80.47% on its exact next trajectory and 72.85% at random active-shard offsets. The
expanded organism reached 54.98% on its exact trajectory but only 20.41% at random
offsets, below its fitted 31.01% bigram ceiling. Cold-state trajectory accuracy was
58.50% control and 43.75% expanded, so accumulated electrical state contributed 21.97
and 11.23 points respectively.

Both computations still depended on the unchanged physical graph. Exact-trajectory
graph silence reduced control from 80.47% to 6.25% and expanded from 54.98% to 4.88%;
endpoint rotation produced 6.54% and 6.15%, while within-target source/weight
reassignment produced 4.59% and 5.57%. Broadcast silence left 24.22% and 20.31%.
The treatment therefore did not bypass the connectome; it learned a narrower physical
trajectory through it.

Validation improved modestly but remained below trivial language statistics. Control
measured 9.28% / 4.77879 and expanded measured 11.13% / 4.20356, versus the shared
19.09% unigram baseline. Their fixed-prompt generations remained unreadable:
` bom  ewnm  nnee` and `o  T heenom  nd `. All audits were non-mutating. Control
checkpoint SHA-256 stayed
`d853dccf9cd81e3827f12fd3bad43fe8f5a8d8e33d0d9db476d7a69d9d902d33`; expanded
stayed `619aac02e8692a3fc9722f2ba2432a326f93ab7196942f0f5e0dbee9a0fd4c86`.

The failure mode is now concrete: changing the task's single stream length changes
the modulo interpretation of every saved absolute cursor. Existing electrical lanes
are preserved as tensors, but they no longer receive the old continuation they had
learned. The next implementation makes stream domain checkpoint-owned per lane.
Existing lanes will retain their exact 2,048-byte domain and cursor interpretation;
newly appended cold lanes alone will receive the 4,096-byte domain. This produces
interleaved replay and novelty inside one persistent organism rather than a hard
global remapping.

### Per-lane replay preserved the old organism while new lanes learned breadth

Both replay-preserving descendants began from the exact update-7,750 control
checkpoint, SHA-256
`d853dccf9cd81e3827f12fd3bad43fe8f5a8d8e33d0d9db476d7a69d9d902d33`,
with organism ID `organism-b2505376398a491e8cf4150a5daf3fab`, 2,224 cells, and
13,737 directed edges. The original sixteen lanes, including their positions,
2,048-byte domains, and complete electrical/private/workspace state, were retained.
Both descendants appended sixteen cold lanes without replacing organism state. The
control assigned the new lanes the same 2,048-byte domain; the treatment alone
assigned its new lanes the 4,096-byte domain. Topology and lifecycle stayed fixed so
the intervention isolated experience replay.

After 1,000 additional updates, the all-2K control reached 83.13% / 0.67747 over its
final 160 updates. The mixed organism reached 66.58% / 1.46337 overall, but that
aggregate separates cleanly into 84.49% on the sixteen inherited 2K lanes and 48.67%
on the sixteen appended 4K lanes. Old-domain performance therefore matched and
slightly exceeded the control while new-domain performance rose from 33.1% over its
first 100 updates to 48.7%. Unlike the hard global switch, breadth did not destroy
the replay trajectories.

Explicit lane-selectable read-only audits established what each half learned. The
old lane zero had accumulated 152,000 electrical-state tokens and reached 90.82% in
the control versus 87.30% in the mixed organism. Silencing the fixed connectome
removed 83.30 and 80.66 percentage points; endpoint rotation removed 85.64 and
80.96 points; within-target source/weight reassignment removed 85.16 and 82.71
points. Saved electrical state contributed 26.66 and 23.63 points. Replay therefore
remained both persistent and physically routed.

Lane sixteen is an age-matched comparison between newly appended trajectories: both
had accumulated 1,984 state tokens, but the control repeated 2K while the treatment
covered 4K. It reached 75.98% control versus 57.71% treatment. Graph silence removed
69.14 and 51.66 points; endpoint rotation removed 69.14 and 51.46 points; reassignment
removed 69.63 and 52.25 points. Saved electrical state contributed 17.19 and 15.63
points. The broader new lane is weaker, but it is learning through the same emerged
connectome rather than through a replacement network or global bypass.

Random-offset competence did not follow automatically. The 2K control measured
81.05% aggregate accuracy on its active shard, while the mixed organism measured
15.92% on 4K; the mixed graph-reference slice was 28.42%, still below its 31.01%
bigram baseline. Its graph remained causal on that slice—silencing removed 22.36
points—but the learned solution was trajectory-specific. The earlier hard-switch
organism reached 20.41% random-offset accuracy, so replay preservation solved
catastrophic interference, not phase-general conditional prediction.

Validation remained below trivial corpus statistics. The control measured 9.18% /
5.19218 and the mixed organism 10.16% / 4.68606, versus 19.09% unigram and roughly
31% bigram accuracy. Fixed-prompt samples were `llbar  wam  nn  ` and
`llbbe   am  an o`. Both checkpoints remained byte-identical through all trajectory,
active-shard, graph/state-counterfactual, validation, and generation audits:
`39e8d890073a61178a93fd51b3f7877cf4e66f4e8c4636a6a957ef566b4526ce`
for the 2K control and
`02adde1864341cbebd2a16a801e3314b797e63548f1b2559e8095d1398048d8a`
for the mixed organism.

The remaining phase failure is measurable rather than speculative. Both 32-lane
organisms cover only 26 of 64 possible cursor phases. Because each contiguous window
advances by exactly 64 tokens, a lane never changes its phase modulo context length.
The next same-lineage intervention should append cold persistent lanes at uncovered
phases while leaving all 32 existing lanes, cells, graph, weights, optimizer moments,
RNG state, and electrical histories untouched. This tests whether phase-complete
experience converts the new 4K trajectory solution into random-offset competence
before changing cellular structure again.

### Phase-diverse replay produced graph-routed random-offset competence

Phase 14 appended 64 cold trajectories to each stopped update-8,750 organism without
moving or replacing any of its 32 existing lanes. The control therefore carried 96
2K-domain lanes. The treatment carried its original sixteen 2K replay lanes, sixteen
older 4K lanes, and 64 new 4K lanes. Both retained organism ID
`organism-b2505376398a491e8cf4150a5daf3fab`, 2,224 cells, 13,737 directed edges,
fixed topology, disabled lifecycle, learned weights and rules, optimizer moments,
random streams, and every saved electrical/private/workspace tensor.

The long phase survived two deliberate worker boundaries without an organism
boundary. Read-only interim auditing stopped and resumed control at update 10,044
from SHA-256
`612aa630aecfc595a396d22e0c32a77629fc66652fdc77f1d32d14b3347bd629`
and treatment at 10,020 from
`524ab43d1762135e1d3015fd691b13681d8f70ab2d628e4f126d61d36c7f4d7c`.
A later laboratory-service restart terminated only the worker processes; their signal
handlers atomically saved control update 10,558 as
`f5a873468f9b219c2d85e54dd9f2798f49a5730eb544f83be146ce816196a763`
and treatment update 10,523 as
`1afa2a340cacfe515c73bc64931bd64aabb981e2c256ccf27e186d4151cb86d7`.
The fail-closed same-phase route fingerprinted and resumed those exact checkpoints to
the original target. Neither event added a phase, reconstructed an organism, remapped
a cursor, or reset state.

At update 11,750, the 2K control's final 160 updates reached 90.53% / 0.36694.
The mixed organism reached 70.90% / 1.05804 overall, separating into 84.57% /
0.61402 on 2K replay and 67.48% / 1.16904 on 4K experience. All 96 lanes were active;
cells and edges remained exactly 2,224 and 13,737 with no phase-local birth, death,
growth, or pruning.

The phase-balancing diagnostic exposed an important experimental correction. The
control covered all 64 corpus cursor phases with one or two lanes per phase. The
mixed organism also covered 64/64 globally, but its domains did not: sixteen 2K lanes
covered 14 phases and eighty 4K lanes covered only 55. The original append algorithm
balanced against every preserved cursor, so unrelated 2K offsets could make the 4K
allocation appear complete. Future expansion now balances only against preserved
lanes in the destination domain, and the laboratory reports phase coverage inside
each domain. No existing lane was changed by that implementation repair.

Despite that nine-phase deficit, phase diversity changed the scientific result. On
the fixed random-offset 4K audit, treatment accuracy rose from 15.92% before the
64-lane append and 30.18% near update 10,000 to 61.13% terminally. Its separately
matched intact-graph slice reached 63.38%, more than twice the 31.01% bigram accuracy
baseline. Graph silence removed 61.62 points, endpoint rotation 56.05, and within-
target source/weight reassignment 57.62; saved electrical state added 16.31 points.
The control reached 90.63% random-offset accuracy and 91.80% on its graph-reference
slice, losing 86.91/88.18/85.94 points under those graph counterfactuals while state
carry added 25.59 points. The broad result is therefore learned through the physical
connectome rather than a frequency-only output bypass.

Exact lane audits separated replay, old breadth, and new breadth. Inherited lane zero
had 153,984 state tokens and reached 93.07% control versus 87.01% mixed; graph silence
removed 87.70 and 85.25 points, while state carry added 28.52 and 25.29. Lane sixteen
had 4,032 tokens and reached 87.50% on control 2K versus 58.50% on treatment 4K;
silence removed 82.13 and 55.76 points, while state added 24.41 and 13.38. Newly
appended lane 64 had 1,984 tokens and reached 91.80% control versus 70.31% treatment;
silence removed 86.43 and 68.36 points, while state added 28.13 and 18.26. The older
single 4K trajectory barely changed while random-offset competence doubled, evidence
that the appended bank taught a shared conditional rule rather than merely refining
one path.

Held-out language remains unsolved. The fixed 16-batch validation audit measured
6.93% / 5.93527 (perplexity 378.1) for control and 9.57% / 4.88900 (perplexity 132.8)
for mixed, below 19.09% unigram and roughly 31% bigram accuracy. Fixed-prompt samples
remained unreadable (` ndl a parm and ` and `  oher pnmm andt`). Final checkpoint
SHA-256 values remained exactly
`8537ae4cf95271237976e67d320b59f45ce36dbfa6a24bc246e56742629047d6`
for control and
`dc030915cd14734f49191a1cd3d91cf15ce7a5d9ebf9de9d4fd068b8c0e7a6f6`
for mixed through every random-offset, trajectory, graph/state-counterfactual,
validation, and generation audit.

The next same-lineage intervention appends exactly nine cold lanes to both organisms.
The treatment's new 4K lanes will occupy its nine missing phases; the control receives
the same lane count and update budget on 2K. This completes the domain-specific phase
test before appending a full 64-phase 8K bank or altering gradient clipping. Topology
and lifecycle remain fixed because the present bottleneck is experience breadth, not
excess capacity.

### Completing 4K cursor phases transferred immediately but did not improve random offsets

Phase 15 continued the exact update-11,750 checkpoints above. It did not initialize,
replace, or reconstruct either organism. Each descendant retained organism ID
`organism-b2505376398a491e8cf4150a5daf3fab`, all 2,224 cells, all 13,737 learned
directed edges, learned parameters, optimizer moments, random streams, and the full
electrical/private/workspace state of its existing 96 lanes. Exactly nine cold lanes
were appended. The control assigned them to its 2K domain; the treatment assigned
them to the nine missing 4K phases, `[9, 7, 1, 16, 8, 15, 5, 0, 6]`. This produced
105 2K lanes covering 64/64 phases in control, and sixteen retained 2K lanes covering
14/64 plus 89 4K lanes covering 64/64 in treatment. Topology and lifecycle remained
fixed, with no phase-local birth, death, growth, or pruning.

The new treatment lanes did not have to relearn the task independently. On their
first 64-token visit they averaged 50.87% accuracy; on their second visit they
averaged about 67.9%. Individual first-to-second visits were 43.8→51.6, 46.9→56.2,
57.8→87.5, 51.6→57.8, 59.4→89.1, 45.3→48.4, 54.7→67.2, 51.6→75.0,
and 46.9→78.1%. This is direct transfer into newly allocated electrical histories
through the already-emerged organism rather than evidence of a reset network.

After 1,000 additional updates, the 2K control's final 160 updates reached 91.94%
accuracy / 0.30508 loss. Treatment reached 77.11% / 0.84625 overall, separating into
87.60% / 0.50790 on retained 2K replay and 74.49% / 0.93084 on 4K experience. The
new lane-96 trajectory reached 89.84% control and 85.45% treatment. Silencing its
graph removed 82.23 and 82.32 percentage points; rotating edge sources removed 85.35
and 78.13; within-target source/weight reassignment removed 81.15 and 80.47. Saved
electrical state contributed 25.78 points in both descendants. Phase completion
therefore produced rapid, stateful, connectome-routed transfer.

It did not improve aggregate random-offset competence. The treatment's fixed 4K
audit declined from 61.13% at update 11,750 to 58.50% at update 12,750 even though
its rolling 4K training accuracy rose from 67.48% to 74.49%. Its matched intact-graph
slice measured 56.35%; graph silence removed 53.03 points, endpoint rotation 50.29,
and weight reassignment 51.66, while state carry added 16.02. Control measured 88.77%
aggregate and 90.72% intact-graph accuracy, losing 83.20/84.28/82.81 points under
the same interventions while state added 25.49. The new phases were useful, but the
remaining limitation is consolidation/interference rather than missing cursor-phase
exposure.

Held-out language remained below trivial statistics. The terminal 16-batch validation
audits measured 9.38% / 6.28098 for control and 11.13% / 5.26256 for treatment,
versus the 19.09% unigram and roughly 31% bigram accuracy baselines. Fixed-prompt
samples were ` nd ba nam  nndd` and `buchey pad  end `. The learned connectome is
causal on trained trajectories but has not yet become a general language model.

One worker handoff preserved the same treatment organism while the unrelated 4090
workload was saturated. Its signal handler atomically stopped at update 12,278 with
checkpoint SHA-256
`175c8196bea9c90708c72e2aecb47a6578ce2e55f8a3a17789ad697f903ba941`;
the fail-closed same-phase route resumed that exact checkpoint on the 2070. It did not
create a phase, cold-start a lane, or alter topology. Terminal checkpoints remained
byte-identical through every read-only trajectory, random-offset, graph/state,
validation, generation, and laboratory-deployment audit:
`f39e7b14dd7f3e916db2c341c159af7c39301eafbf476e2005b1b38987d77382`
for control and
`fd96e449c936ca6b590833404e334c183c19e4119a4916f1798611d8a2d58a1d`
for treatment.

Optimization pressure is now the cleaner next variable. Every phase-15 update was
gradient-clipped. Median clip scale was 0.0894 for control and 0.0606 for treatment,
corresponding to median pre-clip norms of 11.19 and 16.50 against the fixed norm-one
ceiling. A matched same-lineage continuation should vary only the clipping ceiling,
without appending lanes or changing topology, lifecycle, corpus domains, or any
organism state. This tests whether breadth gradients are being compressed enough to
cause the observed consolidation tradeoff before exposing the organism to 8K text.

### Larger unclipped updates damaged the accumulated organism

The phase-16 counterfactual began from two byte-identical copies of the stopped
phase-15 mixed checkpoint. Source, clip-one control, and clip-five treatment all had
SHA-256
`fd96e449c936ca6b590833404e334c183c19e4119a4916f1798611d8a2d58a1d`
before either worker started. The source remained frozen. Both branches retained the
same organism ID, 2,224 cells, 13,737 fixed directed edges, 105 electrical lanes and
their 2K/4K domains, optimizer moments, sampler/RNG state, and all private/workspace
tensors. Their only difference was the global gradient-norm ceiling: 1.0 control
versus 5.0 treatment. Both ran exactly 500 updates to 13,250.

The intervention changed the applied update magnitude as intended. Control's median
pre-clip norm was 16.46 and median clip scale 0.0608; 93.6% of its steps had scale at
or below 0.1. Treatment's median norm was 18.68 and median scale 0.2677, with no step
at or below 0.1. The larger ceiling stayed finite but initially disrupted the learned
solution. Over the final 160 windows, control reached 78.20% / 0.80322 overall,
separating into 87.84% / 0.49967 on retained 2K lanes and 75.79% / 0.87910 on 4K.
Treatment recovered only to 68.29% / 1.08433 overall, 79.15% / 0.72684 on 2K, and
65.58% / 1.17370 on 4K.

The fixed random-offset 4K audit rejects gradient starvation as the earlier breadth
bottleneck. Control improved modestly from its phase-15 source's 58.50% to 61.62%;
treatment reached only 54.98%. Their separately matched intact-graph slices measured
62.89% and 57.81%. Graph silence removed 60.45 and 52.93 percentage points, endpoint
rotation 56.05 and 51.66, and within-target source/weight reassignment 57.23 and
52.25. Saved electrical state added 16.02 and 14.55 points. Passing roughly four
times more gradient therefore weakened the same physically routed conditional rule;
it did not uncover a stronger one.

Lane 96 showed the same damage on an exact persistent trajectory. Control reached
89.55% / 0.43124 versus treatment's 83.98% / 0.57421. Graph silence removed 87.40
and 80.37 points, rotation 82.91 and 77.64, reassignment 85.25 and 79.79, while saved
state contributed 29.39 and 24.80. The emerged connectome and its electrical history
remained causal in both descendants, but larger steps reduced their contribution.

Held-out language remained noisy and unsolved. Treatment measured 11.62% / 5.18860
versus control 10.55% / 5.39949, a small reversal relative to the decisive training
and trajectory effects but still far below the 19.09% unigram baseline. Samples
remained fragmentary: `o chey padm and ` control and `a Thee andm anlt` treatment.
This may reflect mild distributional smoothing from destructive updates, not useful
conditional generalization.

The treatment worker was checkpointed at update 13,023 to release a compute-contended
4090. The stop checkpoint SHA-256 was
`bed5cdff887c49685c6f53a73b2bd4707e8fe827b364492493e643d2685a1ecd`,
and the same-phase resume endpoint returned that exact fingerprint before continuing
on the free 2070. No phase or organism state was reconstructed. Terminal checkpoints
remained byte-identical through every read-only audit:
`cff9b5b133a4cc5888686343fec49ecdb94214df4e6d0c9f99c0577112d187f1`
control and
`935e1f0605d5bd6105763bff0833e9d14645b553e5f3541c9ab3316f0cb6e39c`
treatment.

The norm-one ceiling should therefore remain unchanged. The next breadth experiment
should fork the stronger clip-one endpoint into matched descendants, append 64 cold
lanes to each, and compare a 4K replay bank against a new 8K bank. Existing 105 lanes,
cells, graph, optimizer, RNG, and electrical state remain exact; only the new lanes
receive their assigned domain. This tests genuine corpus expansion against an
age- and capacity-matched replay control without reintroducing the rejected clipping
intervention.

### An 8K bank learned its own trajectory but not a reusable broader rule

Phase 17 began from two byte-identical copies of the stronger phase-16 clip-one
checkpoint. Source, 4K replay control, and 8K breadth treatment all had SHA-256
`cff9b5b133a4cc5888686343fec49ecdb94214df4e6d0c9f99c0577112d187f1`
before either descendant ran. The source remained frozen. Both descendants retained
organism ID `organism-b2505376398a491e8cf4150a5daf3fab`, all 2,224 cells and their
positions, all 13,737 learned directed edges, learned parameters, optimizer moments,
random streams, and every one of the 105 existing electrical/private/workspace state
lanes. Exactly 64 new lanes were appended to each. Control assigned its new bank to
the existing 4K domain; treatment assigned its new bank to an 8K domain with exactly
one lane in each of the 64 cursor phases. Topology, lifecycle, and the norm-one
gradient ceiling were unchanged. Both ran 1,000 updates to 14,250.

The matched new-bank curves separated cold-lane transfer from genuine corpus breadth.
Control's new 4K lanes progressed from 55.47% on their first visit to 76.71%, 77.66%,
76.44%, 77.88%, and 78.20% over visits two through six. Treatment's new 8K lanes
progressed from 26.88% to 36.74%, 37.84%, 37.30%, 37.38%, and 35.69%. Its inherited
lanes remained stable near 74–76%, so the failure was not wholesale erasure of the
accumulated organism. The last 160 training windows reached 78.86% / 0.77402 in
control, separating into 85.94% / 0.50734 on 2K and 78.07% / 0.80365 on 4K. Treatment
reached 59.44% / 1.69968 overall: 79.79% / 0.90436 on 2K, 74.38% / 0.94326 on retained
4K, and only 35.69% / 2.84403 on the new 8K bank. Median pre-clip norms remained
16.64 control and 16.55 treatment, with median scales 0.0601 and 0.0605, so the result
does not reopen the rejected clipping intervention.

Exact trajectories show that the broader bank nevertheless acquired real stateful,
physically routed computation. Inherited lane 16 reached 84.77% control and 83.11%
treatment. Newly appended lane 105 reached 83.40% on control 4K and 71.19% on treatment
8K. On treatment lane 105, cold state reduced accuracy to 51.17%, graph silence to
4.10%, endpoint rotation to 7.03%, within-target source/weight reassignment to 3.32%,
and broadcast silence to 12.70%. The organism therefore learned a causal continuation
along that saved 8K experience rather than bypassing its connectome.

That solution did not transfer to random offsets. The corrected fixed random audit
measured 68.55% / 1.16260 for control 4K but only 12.11% / 4.58095 for treatment 8K.
Control's graph silence/rotation/reassignment accuracies were 3.61/6.64/5.86%; the
treatment's were 2.54/7.91/3.52%. Saved electrical state added 16.80 points to control
but only 0.39 points to the random 8K audit. The treatment can continue one learned
trajectory with its saved state, but its shared cellular rule does not yet predict an
arbitrary context in the enlarged domain.

Held-out language likewise remains unsolved. Control measured 11.33% / 5.58403 and
treatment 10.55% / 4.94956, both below the 19.09% unigram accuracy baseline; fixed
samples were `t ahey pnr  and ` and `  hrry pat  anr `. The treatment's lower loss
suggests slightly less concentrated error, not useful language.

This audit uncovered and repaired a measurement defect without changing either
organism. The headline carried-state evaluation and its intact-graph reference had
previously consumed successive random slices. The evaluator now rewinds its dedicated
evaluation RNG so both replay identical tokens; a regression requires their accuracy
and loss to agree. All phase-17 audits were rerun under that corrected contract. The
terminal checkpoints remained byte-identical through the service deployment and all
corrected state/graph/validation/trajectory audits:
`78e3e71d1071a0b7afb240249dc9c462783dfc913dfa070366df2722860252c8`
for control and
`840501028a30d8748ff72086ecfce5ac8f88ac65f31af29b3f683660b466a060`
for treatment.

The next intervention should target trajectory dependence directly. From exact copies
of the breadth endpoint, a matched control can continue ordinary persistent-lane
training while a treatment adds a disposable random-offset auxiliary batch to each
optimizer update. The primary lanes and all organism-owned state continue normally;
the auxiliary context runs through the same cells and graph, contributes gradient to
the shared rule and synapses, and is discarded rather than replacing any saved lane.
This tests whether explicit pressure for state-independent context prediction can turn
the physically routed trajectory solution into a reusable conditional rule before
changing the organism's topology or lifecycle again.
