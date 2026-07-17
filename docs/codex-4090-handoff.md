# RTX 4090 training handoff

This document is the execution brief for a fresh Codex CLI session running on
the RTX 4090 machine. Read the repository's existing companion documentation
before changing code, preserve unrelated work, and use `apply_patch` for edits.

## Objective

Make Tiny Shakespeare training substantially faster while preserving the
developmental neural substrate experiment. Establish a clean, reproducible
CUDA baseline, implement a 68×68 corpus field with batch 16, optimize the hot
training path, and leave a resumable headless trainer that can run unattended.

This is a technical investigation. Do not add decorative animation or replace
measured state with synthetic visual effects.

## Required scientific configuration

- Task: `tiny_shakespeare`
- Field: exactly 68×68
- Vocabulary: 66 Tiny Shakespeare characters
- Context length: 64 characters
- Batch size: initially 16
- Recurrent message steps per token: 2
- Lifecycle: disabled for the initial learning baseline
- Structural warmup: at least 5,000 optimizer updates
- Initial population cap: retain 4,096 unless a measured result justifies a change
- Evaluation: held-out corpus chunks, performed infrequently and outside timed
  training intervals

The 68-cell height is intentional. `SpatialSubstrate._boundary_sites()` reserves
the first and last rows, leaving exactly 66 usable rows. Enforce these invariants:

1. All 66 input character neurons occupy one linear boundary column.
2. All 66 output character neurons occupy one linear boundary column.
3. Every port site is unique.
4. No character ports wrap into a second column.
5. Input/output semantic order remains the task vocabulary order defined by the
   graph layout.

Add explicit tests for these properties. Do not silently substitute 64, 66, or
a power-of-two field.

## Repository and environment safety

1. Run `git status --short` before editing. The source machine had legitimate
   uncommitted work; never reset, discard, or overwrite unrelated changes.
2. Confirm the current branch and remote before pulling or creating a branch.
3. Read the companion `.md` file before modifying any `.py` or `.ts` file, and
   update that companion after the code change.
4. Keep the existing scientific behavior unless this brief explicitly changes it.

## CUDA setup

From the repository root:

```bash
nvidia-smi
uv sync --extra dev
uv run python - <<'PY'
import torch
print("torch:", torch.__version__)
print("cuda build:", torch.version.cuda)
print("cuda available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("device:", torch.cuda.get_device_name(0))
    print("capability:", torch.cuda.get_device_capability(0))
PY
```

Do not proceed with performance conclusions unless PyTorch reports CUDA
available and identifies the RTX 4090. If the lock resolves a CPU-only build,
install the official PyTorch CUDA build compatible with the installed NVIDIA
driver, then repeat the check.

Run the existing verification suite before modifications:

```bash
uv run pytest
cd frontend && npm ci && npm run check && npm run build
```

## First implementation: 68×68 and a real benchmark

The current general field-size control accepts only powers of two. Add 68 as an
explicit Tiny Shakespeare geometry rather than weakening validation for every
task. The Tiny Shakespeare default and the live parameter control must both be
able to select 68. Preserve the existing power-of-two options for other tasks.

Extend or add a headless Tiny Shakespeare benchmark. It must use the same model,
optimizer, data sampler, and update function as the live application, without
building visualization traces. It should accept at least:

- device
- field size
- batch size
- context length if the task loader permits it
- message steps
- warmup updates
- measured updates
- seed
- AMP mode
- optional `torch.compile` mode

Report machine-readable JSON containing:

- full configuration and seed
- device name, PyTorch version, and CUDA version
- living neurons and active edges
- mean/median update latency
- updates/second
- sequences/second
- target characters/second (`batch × context / seconds`)
- peak allocated and reserved CUDA memory
- rolling training loss and accuracy
- held-out loss and accuracy at the end, outside the timed region

CUDA timings must synchronize before and after measured regions. Include warmup
updates so compilation and allocator startup do not pollute the result.

## Baselines to run

Benchmark these first, with lifecycle and topology changes disabled:

1. Historical reference: 128×128, batch 4, context 64, two message steps.
2. Requested profile: 68×68, batch 16, context 64, two message steps.
3. Batch sweep on 68×68: 8, 16, 32, and 64, stopping before out-of-memory.

Select primarily by target characters/second, not updates/second. A larger batch
does more training work per update. Record loss curves as well as throughput so
an optimization that breaks learning is rejected.

For context, the source Mac/MPS measurements were:

| Configuration | Seconds/update | Target characters/second |
|---|---:|---:|
| 128², batch 4, context 64, steps 2 | 4.212 | 60.8 |
| 64², batch 16, context 64, steps 2 | 3.663 | 279.5 |

These are orientation only, not CUDA acceptance thresholds.

## Optimization sequence

Make one change at a time and retain before/after benchmark JSON. Prefer changes
that preserve exact numerical semantics before trying approximations.

### 1. Remove invariant work from recurrent loops

Inspect `sequence_model.py` and its companion documentation. In particular:

- normalize learned broadcast slot keys once per forward pass, not once per
  token microstep;
- compute graph indegree and other topology-only tensors once per forward pass;
- precompute stable compact-site mappings, source/target indices, and masks;
- reuse safe source indices and avoid repeated `arange`, `zeros`, and temporary
  tensor construction inside the 64×2 recurrent loop;
- use `optimizer.zero_grad(set_to_none=True)` where compatible.

Do not cache across topology mutations without a clear invalidation mechanism.

### 2. Make headless mode genuinely trace-free

The training-only path should not build frames, copy tensors to CPU, serialize
snapshots, run evaluation, or register visualization-only gradient hooks. The
live server must remain responsive and may consume occasional detached snapshots,
but rendering cadence must not gate optimizer cadence.

Measure the trace-free path independently from WebSocket and browser rendering.

### 3. CUDA automatic mixed precision

Add an optional CUDA AMP path using a supported dtype on the 4090. The current
code has failed autocast around `index_add_` because destination and source
dtypes differed. Fix dtype construction at the source; do not scatter ad hoc
casts throughout the model.

Keep numerically sensitive normalization, loss, and metric reductions in FP32
where needed. Verify finite loss and gradients, compare a short deterministic
loss curve with FP32, and report actual speed and memory differences. AMP must
remain optional until it passes those checks.

### 4. Compilation and kernel-launch overhead

Attempt `torch.compile` only around the stable tensor compute path. Mutable
topology, Python event construction, hooks, and visualization are likely graph
breaks, so separate them from the compiled recurrent kernel first. Record graph
breaks and compilation time. Keep compilation only if steady-state throughput
improves without changing learning results.

Use `torch.profiler` after the basic cleanup to identify remaining hotspots.
Expect many small scatter/index-add and recurrent kernel launches; optimize from
measured traces rather than guessing.

### 5. Optional scientific compromises

Do not enable these in the primary comparison, but benchmark separately if the
exact path is still too slow:

- truncated backpropagation through time;
- context curriculum such as 16 → 32 → 64;
- one message step per token;
- less frequent lifecycle/statistics accumulation.

Label these clearly because they alter temporal credit assignment or local
information propagation. Do not use gradient checkpointing as a speed feature;
it generally trades additional computation for lower memory.

## Resumable unattended training

Add or verify a headless training command suitable for a long 4090 run. It must:

- save model, optimizer, topology/substrate state, update count, configuration,
  vocabulary, and random-number state;
- resume from the latest checkpoint without restarting the organism;
- write append-only JSONL or CSV metrics;
- checkpoint atomically at a configurable interval;
- handle `SIGINT`/`SIGTERM` by finishing or safely aborting the current update and
  writing a final checkpoint;
- evaluate infrequently on held-out data;
- print enough progress to monitor throughput, loss, accuracy, and GPU memory.

Start with lifecycle disabled. Once the differentiable baseline demonstrably
learns, run lifecycle as a controlled intervention from a known checkpoint.

## Learning and performance acceptance criteria

Before calling the work complete:

1. All existing backend and frontend checks pass.
2. New port-layout tests prove a single 66-neuron input column and a single
   66-neuron output column on the 68×68 field.
3. The headless trainer and live application use the same training semantics.
4. A short fixed-seed run shows decreasing training loss and beats the uniform
   character baseline; no NaNs or infinite gradients occur.
5. Held-out evaluation is reported separately from rolling training metrics.
6. The optimized 68×68 configuration is benchmarked against its unoptimized
   FP32 equivalent and against the historical 128×128/batch-4 reference.
7. Each claimed speedup states whether it is updates/second or target
   characters/second and includes the exact configuration.
8. A checkpoint can be loaded in a fresh process and continue from the same
   update count.

## Final handoff report

Return:

- code and documentation changed;
- exact commands used;
- benchmark table for every accepted/rejected optimization;
- chosen batch size and AMP/compile modes;
- learning-curve evidence;
- remaining bottlenecks from the profiler;
- the exact command the user should run for unattended training and for the live
  viewer.

Do not claim that throughput alone demonstrates learning or that Tiny Shakespeare
success demonstrates LLM-like behavior. This remains a stepping-stone experiment
in spatial self-organizing recurrent computation.
