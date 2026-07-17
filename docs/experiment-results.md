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
