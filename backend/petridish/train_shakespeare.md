# Resumable corpus trainer

`train_shakespeare.py` runs the same trace-free `SequenceExperiment.train_updates`
path used by live fast training and the CUDA benchmark. It defaults to the required
68×68, batch-16, context-64, two-message-step baseline with lifecycle disabled and
both lifecycle and structural warm-ups set to 5,000 updates. Measured batch/AMP
choices can be supplied explicitly.

`--task tiny_stories` selects the 68×68 distributed-token organism and its cached
2,048-piece TinyStories task. The historical module name remains stable for existing
service files and checkpoints. Metrics report token throughput for both task types.
`--vocabulary-size` selects a 64–2,048 power-of-two lexical curriculum without
changing the 64-cell input/output population banks. Resume derives the saved size
from checkpoint vocabulary metadata rather than silently restoring 2,048 pieces.
`--stream-mode continuous` is the corpus default: adjacent windows carry the full
detached cellular runtime state across optimizer steps. `windowed` retains the old
random-context/cold-electrical-state behavior as a recorded control. Checkpoints save
the stream lane positions and cellular runtime state, so resume continues the same
experience rather than beginning a new one.
The token profile retains its task-specific 500-update lifecycle and 1,000-update
pruning warm-ups; the Shakespeare profile retains its conservative 5,000-update
warm-ups. CLI lifecycle selection changes activation, not those task definitions.
`--lifecycle-profile off|baseline|balanced|replacement` records an explicit intervention.
The legacy `--lifecycle` flag maps to `baseline` when no profile is supplied.
`--no-structure` independently fixes the connectome while leaving differentiable
synaptic weights and cell rules trainable.

### `_fresh_config`

Applies field, batch, microtick, broadcast-workspace gain, architecture, named
lifecycle, topology, and a bounded common learning-rate scale while preserving all
other task-specific defaults, including structural timing. A zero broadcast gain is
a hard workspace bypass, allowing corpus runs to isolate dendritic routing. The rate
scale changes rule, readout, and synapse optimizer rates together so long-run
stability controls do not alter their relative schedule.

`--architecture` selects a checkpointed homogeneous GRU, LSTM, ESN, or temporal
transformer population. GRU remains the default and preserves existing checkpoints.
Version-one GRU checkpoints written before the architecture wrapper are migrated
from `cell_rule.*` to `cell_rule.rule.*` keys during restore; optimizer ordering is unchanged.

The trainer writes one append-only JSONL record per optimizer update, a measured
topology/routing/lifecycle diagnostic at the progress interval, and separate
held-out records at an infrequent configurable interval. Diagnostics distinguish
physical from conducting edges, report pruning pressure and exact cumulative edge
turnover, and compare output reach within one token, one context, and the complete
graph. Held-out records include a fixed-prompt greedy continuation and diversity ratio;
token-corpus records also carry exact unigram and bigram validation baselines;
those baselines include add-one-smoothed loss so model perplexity is calibrated as
well as top-one accuracy;
generation preserves the training sampler and organism state. `latest.pt` is replaced
atomically and contains model parameters, optimizer moments, all substrate/topology
buffers, generation and update counters, configuration, vocabulary, rolling metrics,
and Python, NumPy, Torch, CUDA, sampler, evaluation, and substrate lifecycle random states.
The payload also contains continuous corpus positions and detached neuron, private
memory, workspace, fast-memory, and binding-memory state when active.
Training and scientific records expose `electricalStateTokens`, the exact absolute
token position carried by that runtime state; a continuous run therefore proves its
electrical age directly instead of inferring it from update count.
Continuous held-out records also include carried-state accuracy/loss, an identical
contiguous-token cold-state ablation, and their accuracy delta.
`--evaluate-only` loads an existing checkpoint, appends that complete held-out record
without an optimizer update, and exits. This lets older or interrupted trainers gain
new diagnostics without altering the organism being measured.
`--state-retention 0..1` records the fraction of electrical/private/workspace state
retained at each context boundary. One reproduces indefinite persistence; the lab
defaults new controlled launches to 0.9 after the no-relaxation trajectory ablation
showed harmful accumulated state.
`--evaluate-only --state-horizon-eval` appends the five-point identical-token state
horizon curve without training. Scheduled validation remains bounded to the cheaper
carried-versus-cold pair.
`--state-lanes 1..16` alternates independent persistent corpus trajectories at the
same tensor batch size and records their minimum/maximum electrical ages. It is the
memory-constant alternative to increasing CUDA batch size.
Unhandled trainer exceptions append a bounded `failure` metric before propagating,
so OOM and pre-checkpoint failures remain visible to the laboratory.

By default a fresh invocation resumes `latest.pt` when present. Resume restores the
saved configuration, context, seed, vocabulary contract, AMP mode, organism, and
optimizer before continuing from the saved update count. `SIGINT` and `SIGTERM` set a
stop flag; the current indivisible update finishes, then a final atomic checkpoint is
written. Progress reports loss, accuracy, update and target-token throughput, GPU
memory, and a finite-loss/gradient check.

Compilation remains opt-in because the measured stable-forward attempt currently has
dynamic topology graph breaks; production runs should use `--compile off`.

### `_baseline_diagnostics`

Publishes task-owned frequency and one-token-context baselines only when the corpus
measured them, keeping synthetic controls free of invented comparison values.

Example:

```bash
CUDA_VISIBLE_DEVICES=0 uv run python -m petridish.train_shakespeare \
  --device cuda --field-size 68 --batch-size 64 --context-length 64 \
  --message-steps 12 --broadcast-gain 0 --amp bfloat16 --compile off --updates 100000 \
  --checkpoint-dir runs/shakespeare-4090 --checkpoint-interval 100 \
  --eval-interval 500 --eval-batches 4 --progress-interval 10
```
