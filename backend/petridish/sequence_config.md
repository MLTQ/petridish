# Sequence configuration

`sequence_config()` specializes the shared substrate configuration for token streams.
Synthetic recall and grammar use the benchmarked 24×24 field. Tiny Shakespeare uses
a 128×128 address space, 64-character contexts, and at most 4,096 initial occupants,
separating physical extent from population and compute.

The default six recurrent updates and radius-eight neighborhood were selected by a
controlled three-profile sweep: this was the smallest tested profile where all ten
outputs were reachable within a token and context learning exceeded chance.

`message_steps` means recurrent graph updates per token for sequence tasks. Lifecycle
and structural mutation begin only after the differentiable rule has had time to learn.
Keyword overrides make benchmark sweeps use the same source of defaults as the live UI.

Corpus construction uses fewer batch items and recurrent updates per token than the
short synthetic tasks, keeping the first 128×128 experiment practical on CPU or a
modern GPU. Those defaults are not a claim of language-model competitiveness.
