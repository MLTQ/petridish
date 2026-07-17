# Sequence stepping-stone benchmark

## Question

Can the current physical neural graph learn capabilities closer to language modeling
than spatial classification, and which small substrate provides enough transport?

## Protocol

All runs used seed 1 on Apple MPS for 60 gradient updates. Lifecycle and structural
mutation were disabled to isolate the differentiable recurrent substrate. Held-out
batches were newly generated. Recall begins with one key/value binding and may advance
to two or three after 24 recent batches exceed 90%. Language accuracy counts only the
context-dependent verb and object predictions; its chance baseline is 50%. Recall's
value-class baseline is 25%.

| Profile | Field | Updates/token | Reachable in one token | Recall held-out | Recall stage | Context language |
|---|---:|---:|---:|---:|---:|---:|
| shallow32 | 32×32 | 2 | 0/10 | 45.3% | 1 pair | 50.3% |
| default32 | 32×32 | 4 | 0/10 | 100.0% | 1 pair | 50.3% |
| compact24 | 24×24 | 6 | 10/10 | 51.0% | 2 pairs | 74.0% |

The 100% recall number for `default32` is the solved one-pair stage, not the full
three-pair task. The compact run had already promoted itself to the harder two-pair
stage before its final measurement, so its lower final number represents increased
difficulty rather than regression on the original task.

## Findings

- Transport depth, not neuron count, was the binding constraint. The 24×24 field has
  about 242 living cells versus 412, but radius eight and six microsteps make all
  outputs temporally reachable.
- Full next-token loss can hide failure: the larger fields lowered language loss while
  context-only accuracy stayed at chance. The live page now reports context accuracy.
- Starting recall directly with three bindings stayed near 25% in preliminary runs.
  A one-to-three curriculum produced a solved first stage and promotion to two pairs.
- A regression test exposed a reversed-flow bug: language output neurons sampled
  neighborhoods in the original left-to-right direction, yielding zero synaptic
  gradient. Directional offsets and scores now reverse together.

## Selected default and next ablations

Sequence experiments now default to the compact24 profile. The next scientific tests
are (1) sustained two/three-binding recall, (2) lesion and recovery at matched
competence, (3) lifecycle off versus on from identical checkpoints, and (4) replacing
the generated grammar with a byte/token corpus only after contextual prediction is
reliable. Use `python -m petridish.benchmark_sequences` to reproduce or extend the
sweep; these are local comparisons, not claims of globally optimal hyperparameters.

## Content-addressing follow-up

Two additional mechanisms were tested after the transport sweep. A four-slot broadcast
workspace with within-sequence persistence solved one binding but measured 42.7%
held-out accuracy after 140 updates on the two-binding stage. Adding a neuron-written
fast-weight linear-attention matrix measured 50.0% at the same point. Neither exceeded
the 50% strategy of remembering one of two values, so neither is evidence of genuine
key-addressed retrieval. Fast weights are therefore disabled by default and retained
as an explicit experimental gain/profile.

This negative result narrows the next design: writes must have sparse, token-conditioned
ownership. A promising experiment is to let the currently stimulated token neuron
advertise a write proposal, let a small set of recipient neurons compete to own that
key/value trace, and reward the winning chain when the queried output is correct. That
tests self-assembled binding directly instead of adding another undifferentiated global
average.
