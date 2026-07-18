# lab.ts

## Purpose

Owns the slower-cadence two-GPU experiment laboratory independently from the
high-frequency organism WebSocket. It polls measured hardware, corpus runs, and
stepping-stone benchmark artifacts; compares bounded histories; and submits
validated launch or stop requests.

## Components

### `LaboratoryView`
- **Does**: Coordinates polling, stable DOM updates, selection, and control actions.
- **Interacts with**: `/api/lab` routes and laboratory IDs in `index.html`.

### `renderGpus`
- **Does**: Shows actual utilization, power, memory, and compute-process count per GPU.

### `renderRuns`
- **Does**: Presents task, cell type, hardware, update, loss, throughput, and state.
- **Rationale**: Comparison is bounded to five curves so the complete lifecycle
  ablation matrix remains visible without unbounded chart density.

### `drawChart`
- **Does**: Draws comparable rolling loss against optimizer update using measured JSONL records.

### `renderRunDiagnostics`
- **Does**: Shows measured physical/conducting topology, token/context/graph reach,
  cell and edge turnover, pruning pressure, and the latest fixed-prompt generation
  for each selected corpus run.
- **Does**: Shows authoritative topology/lifecycle gate reasons and remaining warm-up
  or plateau budget so zero turnover is not mistaken for a reset or broken mutation.
- **Does**: Shows matched graph-silencing and source-rotation accuracy/loss deltas so
  an evolved connectome is credited only when its actual routing improves prediction.
- **Does**: Separates global topology rotation from within-neuron weight/source
  reassignment and broadcast-workspace silence, distinguishing edge existence,
  synaptic assignment, and advertisement shortcuts on identical validation bytes.
- **Does**: Labels validation and active-shard causal audits separately. A shard
  silence/rotation/reassignment result can prove where memorized computation lives
  without being presented as held-out language generalization.
- **Does**: Aggregates held-out position accuracy into `p0–3`, `p4–15`, and `p16+`
  bands, separating organism startup/settling from steady-state token prediction.
- **Does**: Places exact corpus unigram and bigram accuracy beside those bands so
  improvements are judged against real frequency/context baselines, not 1/vocabulary.
- **Does**: Shows signed held-out model-minus-unigram and model-minus-bigram accuracy
  directly, preventing modal-token accuracy from being read as contextual learning.
- **Does**: Places smoothed unigram/bigram perplexity beside accuracy so distributed
  learning below the modal token is not mistaken for total failure.
- **Rationale**: Loss alone cannot establish that a cellular routing or lifecycle
  mechanism is functioning.

### `renderBenchmarks` / `drawBenchmarkChart`
- **Does**: Tabulates persisted architecture sweeps and plots the newest matched
  task/profile/update cohort on a shared 0–100% held-out accuracy scale.
- **Does**: Keeps replication seeds in the matched cohort; artifact IDs distinguish
  repeated runs directly.
- **Does**: Directly labels curves by artifact ID and states whether the newest cohort
  used deterministic or ordinary seeded execution.
- **Does**: Labels matched process-global branch RNG separately from deterministic
  kernels, because stochastic lifecycle mutation uses both guarantees.
- **Does**: Shows atomic live progress, completion state, elapsed time, and measured
  peak CUDA allocation when the benchmark publisher provides them.
- **Does**: Keeps the persisted benchmark learning-rate scale in its intervention
  identity rather than merging stability retries into a default-rate cohort.
- **Does**: Shows failed runs with their persisted exception type/message and keeps
  an empty pre-evaluation failure from producing invalid chart axes.
- **Does**: Shows final held-out accuracy for each queried binding slot so a
  one-memory solution is visible without inferring it from the aggregate curve.
- **Does**: Shows final accuracy at each supervised stream position so persistent
  context decay is visible instead of averaged away.
- **Does**: Shows presented-value coverage and distractor errors when available,
  separating value storage from correct key/value association.
- **Does**: Shows owner-address distinctness, entropy, and overlap only for artifacts
  that measured an owner map.
- **Does**: Shows recovery intervention identity, final living-cell/edge counts, and
  cumulative births/deaths when matched-recovery artifacts provide them.
- **Does**: Retains parsing of historical recovery artifacts while the active UI
  hides the lesion-oriented topology-recovery subchart.
- **Does**: Marks the measured checkpoint where each run changes curriculum
  difficulty; peak and final values remain separately visible in the table.
- **Rationale**: A high peak before a curriculum transition must not be mistaken
  for retention at the harder level.

### `launch` / `stop` / `evaluateCheckpoint`
- **Does**: Submit bounded, task-aware Tiny Shakespeare or TinyStories commands and
  preserve authoritative server status.
- **Does**: Launch both corpus organisms at 68×68 so their 64- or 66-port banks
  remain one physical column per boundary.
- **Does**: Selects and displays named lifecycle interventions so baseline and
  balanced/replacement ablations remain distinguishable in manifests and diagnostics.
- **Does**: Launches an explicit microtick budget and flags runs whose budget is
  shorter than the measured minimum sensory-to-output route.
- **Does**: Launches, persists, and displays exact broadcast-workspace gain so
  local-only corpus routing is distinguishable from legacy/global runs.
- **Does**: Launches a recorded power-of-two lexical curriculum independently from
  the fixed distributed I/O population size.
- **Does**: Selects byte-complete TinyStories tokenization by default, locks its
  vocabulary to 256, and labels legacy wordpieces as capable of producing `<unk>`.
- **Does**: Launches and displays continuous state-carrying experience separately
  from the cold-window control; legacy runs are labeled windowed rather than guessed
  to be continuous.
- **Does**: Displays measured electrical-state age in tokens beside routing so a
  nominal continuous run cannot masquerade as one while repeatedly reinitializing.
- **Does**: Displays a saved-state evaluation copy versus an identical-token,
  disposable zero-state ablation copy and their signed accuracy/loss state-value
  deltas. Neither copy overwrites or resets the checkpointed organism. The checkpoint
  electrical age is shown explicitly.
- **Does**: Derives the loss delta from saved/cold loss for historical records written
  before the backend published that signed field explicitly.
- **Does**: Launches and displays the bounded electrical-retention intervention so
  indefinite persistence and homeostatic relaxation remain distinguishable.
- **Does**: Shows optional h1/h2/h4/h8/h16 held-out accuracy from the identical-token
  electrical-memory horizon evaluation.
- **Does**: Launches and displays persistent state-lane count and their measured age
  range independently from tensor batch size.
- **Does**: Displays measured active/cold lane count and unique corpus-cursor phases,
  so allocating sixteen lanes cannot masquerade as sixteen distinct experiences.
- **Does**: Summarizes the newest four visits to every lane as mean and min–max
  accuracy, distinguishing broad consolidation from one inherited trajectory.
- **Does**: Shows bounded explicit trainer failures in the authoritative run-status
  cell instead of labeling pre-checkpoint OOM as an ordinary stop.
- **Does**: Launches and displays fixed/adaptive topology independently from the
  selected lifecycle profile, and accepts a measured `failed` run state.
- **Does**: Launches and displays prune-only phases separately from adaptive
  prune-plus-grow and fixed conducting controls.
- **Does**: Launches a bounded common learning-rate scale so stability controls are
  recorded in the immutable manifest rather than applied out of band.
- **Does**: Continues a checkpointed organism into a named topology/lifecycle phase
  without accepting architecture, geometry, connectome, weight, or state replacements.
- **Does**: Can change a continued TinyStories organism to a deterministic repeated
  experience shard without resetting or replacing any organism-owned state.
- **Does**: Can expand a continued organism's persistent experience lanes without
  exposing a shrink operation; existing lane positions and electrical histories stay
  organism-owned while newly allocated lanes begin cold at independent phases.
- **Does**: Enables that control only when the serving backend advertises support,
  preventing an older API from silently ignoring the requested curriculum.
- **Does**: States the continuation invariant directly in the controls: cells,
  connectome, weights, optimizer, stream position, and electrical memory persist,
  while checkpoint evaluations operate only on disposable copies.
- **Does**: Displays the immutable lineage ID and current phase, and marks measured
  phase boundaries directly on the existing rolling-loss trace.
- **Does**: Shows the current repeated-shard size or full-stream curriculum directly
  beside the lineage phase before the next held-out audit is available.
- **Does**: Displays rolling training and held-out accuracy separately in the run
  table, which makes in-shard memorization versus validation transfer explicit.
- **Does**: Prefers phase-local rolling loss/accuracy when the trainer publishes it;
  this segments curriculum statistics without resetting organism-owned history.
- **Does**: Shows pre-clipping bias/readout/token/rule/synapse gradient norms beside
  routing diagnostics, together with total norm and clip scale, to localize
  frequency-only learning failures and clipped instability.
- **Does**: Summarizes up to 160 current-phase clip scales as the median, tenth
  percentile, and fraction at or below 0.1. This distinguishes sustained optimizer
  starvation from an isolated recurrent-gradient spike without mixing curriculum
  phases or implying that the organism itself restarted.
- **Does**: Separates phase-local grown/pruned edge counts from lifetime totals when
  the matching phase boundary recorded its starting counters.
- **Does**: Offers read-only evaluation on stopped checkpoints so new causal metrics
  can be collected without continuing training or changing plasticity phase; the
  button appears only when the serving backend advertises the matching route.
- **Does**: Offers a separately labeled active-shard audit only when the current phase
  has a bounded shard and the backend advertises support; validation stays a separate
  action.
- **Does**: Offers and displays an exact next-trajectory audit separately from the
  random-offset shard audit, including the matched recurrent lane and graph
  counterfactuals.
- **Does**: Displays fixed audit seed/sample size plus generation special/unknown
  rates and the validation unknown rate, so modal collapse cannot masquerade as
  readable token prediction.
- **Does**: Renders direct-routing controls with microticks, minimum hops, and
  one-token output reach instead of recall-only columns, with the measured task's
  chance baseline drawn directly on the accuracy plot.
- **Does**: Renders distributed context controls with sequence length, dependency
  horizon, and token/context/graph reach on the same measured accuracy surface.
- **Does**: Places delayed-copy memory between direct routing and XOR composition
  without changing the chart or topology semantics.
- **Does**: Places repeated copy/invert token streams after XOR composition and
  preserves their dependency horizon plus per-position accuracy.
- **Does**: Treats fixed-latency contextual pipelines as token controls, preserving
  their physical route budget and delayed per-position accuracy.
- **Does**: Treats masked context-settling clocks as a distinct token control so
  context-propagation time is not conflated with output latency.
- **Does**: Treats the decorrelated settled pipeline as the combined causal control,
  preserving its longer dependency horizon and delayed position metrics.
- **Does**: Treats the balanced second-order grammar as an autoregressive token
  control with a measured 25% baseline and seven true next-token positions.
- **Does**: Keeps broadcast-on/off identity in the persisted intervention label so
  global-workspace shortcuts are not conflated with physical dendritic routing.

## Contracts

| Dependent | Expects | Breaking changes |
|-----------|---------|------------------|
| `main.ts` | `LaboratoryView.start()` is non-blocking | Constructor/start signature |
| `index.html` | Laboratory IDs use native controls and two measured SVG plots | ID or element changes |
| Backend | `/api/lab` payloads retain documented camelCase fields | API schema changes |

## Notes

- Polling is serialized so a slow request cannot accumulate stale UI work.
- Capability selectors advertise only architectures implemented by the backend.
- No synthetic progress is shown; missing values remain an em dash.
