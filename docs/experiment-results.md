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

## Radius-8 lifecycle ablation matrix — 2026-07-17

Five deterministic branches were cloned from one 1,200-update fixed-three base. The
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
