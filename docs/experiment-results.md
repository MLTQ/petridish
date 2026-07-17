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
