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
wall time. Artifacts also record parameter counts, initial CUDA allocation, and peak
training allocation. Language accuracy is the context-dependent verb/object metric;
recall's chance baseline is 25%. Run it with:

Associative-recall checkpoints include held-out accuracy for each queried binding
slot. A near-perfect slot beside a chance slot is direct evidence of single-binding
retention rather than a generic optimization plateau.
They also report presented-value, distractor, and absent-value rates. These distinguish
failure to store values from failure to bind stored values to their keys.

`--fixed-recall-pairs 2` begins and remains at two bindings. This control separates
ordinary convergence at the harder task from interference caused by advancing an
already-trained one-binding organism.

`--output benchmarks/lab/run.json` atomically publishes a `running` artifact at
every evaluation checkpoint and replaces it with `complete` at the end. This lets
the laboratory render real progress without parsing process output or accepting
partially written JSON.

Optimizer updates use the public trace-free training path. Viewer frame capture and
implicit validation are excluded; only the explicit checkpoint evaluations consume
the held-out generator, so timing and learning curves are directly attributable to
the declared protocol.

`compact24_no_broadcast` removes slot broadcasting, while
`compact24_no_global_memory` removes both slot and fast-weight memory.
`compact24_fast_weights` enables recurrent linear-attention memory at gain 0.5.
`compact24_binding_owners` enables successor values stored at genotype-addressed
living neurons and read back through the queried token's physical input port.
`compact24_binding_tokens` changes only the stored value from the successor neuron's
mixed state to the clean successor token representation.
Owner-memory artifacts include address distinctness, entropy, overlap, and peak
ownership at every published checkpoint.
`compact24_binding_sparse` lowers address temperature and adds a small separation
penalty to test stable differentiated owner neurons.

```bash
python -m petridish.benchmark_sequences --task associative_recall \
  --profile compact24 --architecture transformer --steps 80 \
  --fixed-recall-pairs 2 --output benchmarks/lab/recall-transformer.json
```

These short sweeps compare local choices; they do not claim globally optimal
hyperparameters.
