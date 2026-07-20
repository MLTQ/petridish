# benchmark_sequences.py

This command runs reproducible, hardware-bounded learning sweeps for associative
recall, tiny language, direct physical routing, delayed-copy memory, distributed
context composition, persistent contextual streams, exact next-token grammar, and
held-out rule composition.
Profiles vary field size and recurrent microsteps while
holding the task, seed, optimizer, and lifecycle state constant. Lifecycle and
structural mutation are disabled so a run measures the differentiable substrate
before testing turnover as a separate intervention.

`--architecture` selects a homogeneous GRU, LSTM, ESN, or temporal-transformer
population under the same graph, seed, task, and profile. These controls identify
cell-rule effects before testing heterogeneous mixtures.

Checkpoints report held-out accuracy on freshly generated sequences, rolling loss,
recall curriculum size, graph reachability, living cells, edge count, device, and
wall time. Artifacts also record parameter counts, initial CUDA allocation, and peak
training allocation. Completed distributed-token artifacts additionally run a
matched read-only graph audit on identical held-out rows. They report the accuracy
lost and loss added by silencing all learned synapses, rotating source endpoints,
reassigning weights within each receiving neuron, and silencing broadcast. This
distinguishes physical causal routing from mere reachability. Language accuracy is the context-dependent verb/object metric;
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
If training raises, the same path is atomically replaced with `failed`, the last
fully completed optimizer update, and a bounded exception type/message before the
process exits nonzero.

Optimizer updates use the public trace-free training path. Viewer frame capture and
implicit validation are excluded; only the explicit checkpoint evaluations consume
the held-out generator, so timing and learning curves are directly attributable to
the declared protocol.

`--deterministic` enables PyTorch deterministic algorithms and the required cuBLAS
workspace configuration before model construction. Artifacts record the flag so
bitwise reproductions cannot be confused with ordinary seeded CUDA executions.

`--learning-rate-scale 0.01..1` scales the cell-rule, output-readout, and physical
synapse optimizer groups together. The exact scale is embedded in both the artifact
and intervention label so a stability retry cannot be mistaken for a matched default-rate run.

`--batch-size 1..64` is an explicit memory-bound override. Artifacts always record
the effective batch, so reduced-memory replications cannot be mistaken for the
eight-example production control. `--amp bfloat16` uses the same autocast path as
the corpus trainer and records the precision mode alongside peak CUDA allocation.
These controls make seed replications possible on smaller GPUs while keeping every
other organism and task parameter declared; comparisons across batch or precision
remain hardware controls rather than exact optimizer replications.
The artifact also records `PYTORCH_ALLOC_CONF` or its legacy
`PYTORCH_CUDA_ALLOC_CONF` spelling when present. This makes the expandable-segment
allocator needed at the 2070 memory limit reproducible without treating it as a
learning intervention.

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

`token_routing` with `token_route68` repeats eight balanced one-token mappings on
the production 68×68 distributed-I/O substrate. `--message-steps` is the sole
intervention, allowing four-microtick and route-covering controls to share initial
weights, topology, optimizer, seed, and target distribution. Checkpoints retain
gradient norms for the token input, input projection, cell rule, physical synapses,
broadcast path, and output readout.
`--broadcast-gain 0` is the matched hard ablation: workspace writes, reads, and
gradients are bypassed, so four ticks test an under-length physical graph while
twelve ticks test the same graph after every output becomes temporally reachable.

```bash
python -m petridish.benchmark_sequences --task token_routing \
  --profile token_route68 --message-steps 12 --steps 200 \
  --broadcast-gain 0 \
  --output benchmarks/lab/token-route-12.json
```

`token_context` with `token_context68` enumerates all four two-token bit pairs and
supervises their XOR only after the second token. Its 50% restricted-target baseline
cannot be beaten using position, class frequency, the context token alone, or the
query token alone. It reuses the same 68×68 population, 64-port distributed I/O,
microtick override, and hard broadcast ablation as the routing control.

```bash
python -m petridish.benchmark_sequences --task token_context \
  --profile token_context68 --message-steps 12 --broadcast-gain 0 \
  --steps 1000 --output benchmarks/lab/token-context-local.json
```

`token_memory` with `token_memory68` is the intervening delayed-copy control. Both
rows receive the same second-position recall token, while the balanced target repeats
the first context bit. It tests persistent state without requiring XOR composition.

```bash
python -m petridish.benchmark_sequences --task token_memory \
  --profile token_memory68 --message-steps 12 --broadcast-gain 0 \
  --steps 800 --output benchmarks/lab/token-memory-local.json
```

`token_stream` with `token_stream68` retains one copy/invert rule while four later
bit tokens each require a supervised transformed-token prediction. Every position is
balanced across rules, inputs, and targets, so persistent context and repeated
composition are both necessary. Checkpoints report accuracy at each of the four
prediction positions to reveal temporal decay hidden by aggregate accuracy.

```bash
python -m petridish.benchmark_sequences --task token_stream \
  --profile token_stream68 --message-steps 16 --broadcast-gain 0 \
  --steps 1200 --output benchmarks/lab/token-stream-local.json
```

`token_pipeline` with `token_pipeline68` shifts each stream target by two token
clocks. This recognizes that a signal traversing a spatial organism has latency and
tests a causal local pipeline rather than demanding same-clock global communication.
The input patterns balance every delayed/current-bit pair at the first two output
positions, while the final two positions receive only a constant clock token.

```bash
python -m petridish.benchmark_sequences --task token_pipeline \
  --profile token_pipeline68 --architecture esn --message-steps 16 \
  --broadcast-gain 0 --steps 1200 \
  --output benchmarks/lab/token-pipeline-esn-local.json
```

`token_settling` with `token_settling68` keeps targets aligned with their current
bit input but inserts two masked, constant clock tokens after the rule. It tests
whether local failure reflects time needed to establish a distributed context rather
than delayed output or memory decay. The clocks carry no target information.

```bash
python -m petridish.benchmark_sequences --task token_settling \
  --profile token_settling68 --architecture esn --message-steps 16 \
  --broadcast-gain 0 --steps 1200 \
  --output benchmarks/lab/token-settling-esn-local.json
```

`token_settled_pipeline` with `token_settled_pipeline68` is the combined physical
protocol: two rule-settling clocks, four decorrelated input bits, two flush clocks,
and targets delayed by two token positions. It removes the period-two shortcut in
the settling-only control while preserving a purely local causal path.

```bash
python -m petridish.benchmark_sequences --task token_settled_pipeline \
  --profile token_settled_pipeline68 --architecture esn --message-steps 16 \
  --broadcast-gain 0 --steps 800 \
  --output benchmarks/lab/token-settled-pipeline-esn-local.json
```

`token_grammar` with `token_grammar68` is the first literal autoregressive control.
After a rule token and two initial symbols, every position predicts the actual next
symbol in a four-symbol second-order modular language. All 32 rule/seed states are
sampled in shuffled balanced cycles. At every position, the target is uniform and
cannot be inferred from the rule, current symbol, previous symbol, or position alone.
The last target is never presented as an input. This separates genuine next-token
prediction from transformed labels and delayed copying.

```bash
python -m petridish.benchmark_sequences --task token_grammar \
  --profile token_grammar68 --architecture esn --message-steps 16 \
  --broadcast-gain 0 --learning-rate-scale 0.25 --steps 1200 \
  --output benchmarks/lab/token-grammar-esn-local.json
```

For an 8 GiB replication, add `--batch-size 4 --amp bfloat16` and encode both
interventions in the artifact filename. First reproduce the known seed under that
same regime before comparing independent seeds.

`token_compositional_grammar` with `token_compositional_grammar68` is the first
anti-memorization gate. Two separately encoded rule tokens select an operator and
offset for a second-order four-symbol recurrence. Six operator/offset pairs and all
their seeds are training-only; `sum+offset-1` and `difference+offset-2` are
evaluation-only. Every operator and offset token occurs during training, every
supervised target remains uniformly distributed when conditioned on either rule,
the current token, or the previous token, and train/evaluation state overlap is zero.

Completed artifacts persist the exact split and free-run all 32 withheld prompts for
nine generated tokens. Token accuracy, exact-sequence accuracy, invalid-token rate,
per-position decay, one readable example, and the matched physical-graph audit must
all be considered together. Teacher-forced held-out accuracy alone is not success.

```bash
python -m petridish.benchmark_sequences --task token_compositional_grammar \
  --profile token_compositional_grammar68 --architecture esn --message-steps 16 \
  --broadcast-gain 0 --learning-rate-scale 0.25 --batch-size 4 --amp bfloat16 \
  --steps 1200 --output benchmarks/lab/token-composition-esn-local.json
```
