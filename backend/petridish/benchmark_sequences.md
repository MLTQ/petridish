# benchmark_sequences.py

This command runs reproducible, hardware-bounded learning sweeps for associative
recall and tiny language. Profiles vary field size and recurrent microsteps while
holding the task, seed, optimizer, and lifecycle state constant. Lifecycle and
structural mutation are disabled so a run measures the differentiable substrate
before testing turnover as a separate intervention.

`--architecture` selects a homogeneous GRU, LSTM, ESN, or temporal-transformer
population under the same graph, seed, task, and profile. These controls identify
cell-rule effects before testing heterogeneous mixtures.

Checkpoints report held-out accuracy on freshly generated sequences, rolling loss,
recall curriculum size, graph reachability, living cells, edge count, device, and
wall time. Language accuracy is the context-dependent verb/object metric; recall's
chance baseline is 25%. Run it with:

`compact24_no_broadcast` removes slot broadcasting, while
`compact24_no_global_memory` removes both slot and fast-weight memory.
`compact24_fast_weights` enables recurrent linear-attention memory at gain 0.5.

```bash
python -m petridish.benchmark_sequences --task associative_recall \
  --profile compact24 --architecture transformer --steps 80
```

These short sweeps compare local choices; they do not claim globally optimal
hyperparameters.
