# test_sequence.py

These tests protect the sequence ladder's scientific assumptions: semantic ports are
permuted or direction-reversed, generated recall targets match the queried binding at
all curriculum sizes, recurrent token frames retain a differentiable path to synaptic
weights, and live snapshots align sparse cells/edges with readable sequence metadata.

The compact 20×20 fixture checks contracts rather than learning quality. Reproducible
learning curves belong to the benchmark command because convergence tests would make
the normal unit suite slow and hardware-sensitive.

The corpus fixture avoids network access while protecting dynamic vocabulary-sized
ports, square-field choices from 16 through 1024, prompt installation, single-token
generation, and interactive task serialization.

The trace-free regression requires optimizer updates to advance metrics and examples
without replacing the visible frame buffer, then verifies an explicit refresh rebuilds
one current token/feedback/structural trace.

Streaming regressions require one callback per genuinely computed token with aligned
logit shape, then verify a visual optimizer update reports forward, backward,
optimizer, local-credit, and lifecycle work and finishes on its measured structural
frame.
The backward contract requires an initial phase boundary plus one actual autograd
hook callback for every retained token state, ending at complete backward progress.
