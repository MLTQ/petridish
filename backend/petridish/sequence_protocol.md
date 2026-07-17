# Sequence protocol

Sequence snapshots reuse the occupied-cell, directed-edge, measured-flow, credit,
lifecycle, and reachability schema of MNIST. The task payload adds the readable
vocabulary, complete input sequence, aligned targets and predictions, current token
position, held-out accuracy, and perplexity.

The frontend can therefore compare tasks with one renderer while presenting the
diagnostic that matters for sequences: what has been consumed, what prediction is
being judged, and whether delayed or autoregressive accuracy improves.
