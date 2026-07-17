# Sequence configuration

`sequence_config()` specializes the shared substrate configuration for short token
streams. The 32×32 field and twelve-channel neuron state keep both experiments
interactive on CPU while retaining hundreds of living neurons and mutable local
dendrites.

`message_steps` means recurrent graph updates per token for sequence tasks. Lifecycle
and structural mutation begin only after the differentiable rule has had time to learn.
Keyword overrides make benchmark sweeps use the same source of defaults as the live UI.
