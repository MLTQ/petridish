# Sequence protocol

Sequence snapshots reuse the occupied-cell, directed-edge, measured-flow, credit,
lifecycle, and reachability schema of MNIST. The task payload adds the readable
vocabulary, complete input sequence, aligned targets and predictions, current token
position, held-out accuracy, perplexity, and recall curriculum size.

Corpus payloads also expose dataset name, character/token count, tokenizer, context length, source
URL, prompt, generated suffix, and the next greedy-token diagnostic. Visible trial
length follows the current encoded context instead of assuming a fixed synthetic
sequence length.

The frontend can therefore compare tasks with one renderer while presenting the
diagnostic that matters for sequences: what has been consumed, what prediction is
being judged, and whether delayed or autoregressive accuracy improves.
During a streamed forward pass, not-yet-computed predictions are represented by an
explicit em dash and zero confidence. They are never populated from stale or guessed
values.

Cell channels append measured stunned state and cumulative excitotoxic damage.
Task and metric payloads report current and cumulative stun/recovery counts so
temporary silencing remains distinct from pruning and death.
They also report exact edge growth/pruning totals and distinguish output reach in
one token from reach accumulated across the full persistent context.

Both corpus configuration payloads include the 68×68 single-column geometry among
their discrete field choices; non-corpus tasks retain power-of-two choices.
