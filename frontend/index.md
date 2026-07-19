# index.html

## Purpose

Defines the multi-experiment viewer shell: dominant dish canvas, training and lifecycle
controls, scientific diagnostics, history, inspection, and a stable two-GPU
experiment laboratory above the live organism.

## Components

### `#dish-host`
- **Does**: Hosts the PixiJS canvas and pointer interventions.
- **Interacts with**: `DishRenderer` in `renderer.ts`.

### `.laboratory`
- **Does**: Hosts measured GPU lanes, persisted-run comparison, rolling-loss chart,
  and an explicitly enabled launch form without displacing live dish controls.
- **Does**: Places selected-run topology, routing, lifecycle, pruning, and fixed-prompt
  generation evidence directly below the corresponding loss histories.
- **Does**: Includes startup, settling, and steady-state held-out position bands so
  delayed cellular competence is not hidden inside aggregate corpus accuracy.
- **Does**: Keeps historical topology-recovery data available to the laboratory
  parser but removes that lesion-oriented plot from the active investigation surface.
- **Interacts with**: `LaboratoryView` in `lab.ts` and `/api/lab`.
- **Rationale**: Run-level monitoring changes on a slower cadence than organism
  frames and therefore remains independent from the WebSocket renderer.
- **Does**: Launches named off/recovery-only/baseline/balanced/replacement lifecycle interventions without
  hiding their scientific identity behind a boolean checkbox.
- **Does**: Exposes cellular microticks per token beside corpus launch controls so
  one-token routing budget is an explicit intervention.
- **Does**: Exposes exact broadcast-workspace gain beside microticks; zero launches
  a hard local-only corpus run rather than an undocumented global shortcut.
- **Does**: Exposes a 64–2,048 power-of-two TinyStories vocabulary curriculum while
  leaving the organism's 64 physical boundary ports unchanged.
- **Does**: Makes continuous organism experience the default and labels the old
  random-window electrical reset as a cold-window control.
- **Does**: Exposes electrical retention separately from experience mode; 0.9 means
  gradual relaxation toward each cell's physical resting state, not a reset.
- **Does**: Exposes persistent state lanes separately from tensor batch size so
  trajectory diversity does not imply multiple organisms or larger CUDA batches.
- **Does**: Selects fixed or adaptive topology independently from lifecycle.
- **Does**: Adds an explicit prune-only topology phase that can remove low-utility
  dendrites but cannot replace them; fixed still conducts through the saved graph.
- **Does**: Provides a separate continuation form for moving a stopped checkpoint
  into adaptive topology or lifecycle while explicitly retaining the same organism.
- **Does**: Accepts an optional counterfactual branch ID so continuation can first
  copy and fingerprint the exact stopped checkpoint, enabling paired GPU interventions
  without replacing or overwriting the source organism.
- **Does**: Requires every continuation to be visibly designated as either the
  canonical living line or a counterfactual measurement that will never be merged.
- **Does**: Lets that same-organism continuation optionally repeat a deterministic
  TinyStories shard; the default preserves the current stream and zero explicitly
  returns to the full corpus.
- **Does**: Offers a separate explicit checkbox to expand carried streams in place.
  Their saved state and next-token cursor persist; only the future wrap boundary moves
  to the selected broader corpus domain.
- **Does**: Includes 4,096- and 16,384-token intermediate shard choices so corpus
  breadth can double without forcing a fourfold curriculum jump. Selecting one still
  changes only the experience stream of the checkpointed organism.
- **Does**: Lets continuation append any exact target up to 512 persistent experience
  lanes while rejecting values below the organism's current count. Blank preserves
  the current bank, so measured gaps such as nine missing domain phases do not require
  arbitrary preset totals.
- **Does**: Treats larger targets as append-only unless the carried-domain checkbox is
  explicit: existing domains then widen while cursors and runtime state stay fixed;
  otherwise new cold lanes alone receive the selected broader corpus
  domain and are balanced within it before any phase duplicate.
- **Does**: Calls held-out zero-state comparisons disposable ablation copies so the
  interface never implies that the live organism was reset.
- **Does**: Labels the independent-context action as a cold, read-only probe and the
  active-shard action as a warm trajectory audit. Both preserve the organism's cells,
  graph, parameters, recurrent lane bank, optimizer, RNG, and live checkpoint.
- **Does**: Offers a separate full-corpus cold probe for TinyStories when advertised,
  so active-shard reuse cannot masquerade as corpus-wide transfer.
- **Does**: Shows lineage and phase beside run status so a continuation cannot be
  mistaken for a newly initialized comparison organism.
- **Does**: Exposes a bounded common learning-rate scale for controlled long-run
  stability experiments.
- **Does**: Exposes an optional bounded gradient-clip ceiling only on persistent
  continuation; blank explicitly preserves the checkpoint's current ceiling.
- **Does**: Exposes a separate per-generation dendrite-growth budget; blank preserves
  the checkpoint cap and zero defers every ready proposal without clearing evidence.
- **Does**: Exposes construction cost per axon endpoint, post-growth energy reserve,
  and probationary starting utility; every blank preserves the living checkpoint's
  current value.
- **Does**: Shows disposable cold-gradient training as disabled. Legacy auxiliary
  metadata remains diagnostic history, while continuation migrates its weight to zero
  and fresh-state contexts remain read-only audits.
- **Does**: Uses a task-neutral stepping-stone table and chart for recall and direct
  physical-routing, memory, composition, and persistent-stream overfit controls.
- **Does**: Labels detailed held-out accuracy as binding slot or supervised stream
  position rather than conflating the two diagnostics.
- **Does**: Shows rolling training accuracy beside held-out accuracy in the main run
  table so repeated-shard overfit cannot be inferred from loss alone.

### `.visual-column` / `.task-panel`
- **Does**: Keeps the dynamic task/context readout directly below the network
  visualization instead of above the side-column controls.
- **Rationale**: Live token and phase changes cannot displace intervention,
  cadence, or hyperparameter controls while the user is editing them.
- **Does**: Uses a stable desktop height and wider two-column sequence/generation
  layout; narrow screens return to a natural single-column flow.

### Control elements
- **Does**: Expose experiment selection, playback, measured field layers, edge threshold,
  evaluation, forced lifecycle cycles, and speed inputs.
- **Does**: Exposes stunned state and cumulative excitotoxic damage as measured layers;
  lesion intervention controls are intentionally absent.
- **Interacts with**: Event wiring in `main.ts`.
- **Does**: Defaults the field to a phase-aware signal layer: activation during
  forward computation and measured gradient credit during feedback.

### Headless training control
- **Does**: Lets sequence experiments run optimizer updates without token-frame
  capture and states plainly that traces are suspended.
- **Does**: Shows measured update latency and updates per second rather than an
  arbitrary speed multiplier.

### Cadence control
- **Does**: Selects simulation steps per MNIST frame or measured token-frame
  sampling stride for sequence organisms.

### Metric and task elements
- **Does**: Provide stable targets for image or token-sequence task diagnostics and history rendering.
- **Interacts with**: `main.ts` and `HistoryChart`.
- **Does**: Expose measured synapse update ratio and structural lock state.
- **Does**: Expose learning phase, hop distance, temporal reachability, local
  attention entropy, effective capacity, lifecycle state, turnover, energy,
  stress, and death causes.
- **Does**: Distinguishes cell turnover from edge growth/pruning and labels
  sequence reach as token/context/complete-graph counts.

### MNIST phase readout
- **Does**: Reuses the task heading/badge for input, forward traffic, backward
  credit, and structural stages while the digit remains external to the field.
- **Interacts with**: Persistent-lifetime metadata in `protocol.ts`.

### Sequence readout
- **Does**: Shows ordered tokens, current consumed position, aligned prediction,
  held-out accuracy, and perplexity for recall and language experiments.

### Corpus generation panel
- **Does**: Accepts a character prompt, installs it as the active context, shows
  the generated continuation, and requests exactly one next token per click.
- **Interacts with**: `prompt` and `generate` WebSocket commands.
- **Rationale**: Generation remains paused and deliberate so each new token can
  be inspected against the field and graph state.
- **Does**: Supports character and wordpiece prompts while the token organism advances
  its persistent state once per generated token.

### `#hyperparameter-panel`
- **Does**: Hosts grouped backend-defined sliders, pending-change status, and an
  explicit apply-and-restart action for the selected organism.
- **Interacts with**: Dynamic element construction in `main.ts`.

## Contracts

| Dependent | Expects | Breaking changes |
|-----------|---------|------------------|
| `main.ts` | Every referenced element ID exists with the expected native type | ID or element-type changes |
| `lab.ts` | Laboratory table, chart, status, and launch IDs are stable | ID changes |
| `styles.css` | Structural class names remain stable | Class changes |
| Accessibility | Controls retain visible labels and native semantics | Removing labels or roles |
| Hyperparameters | Apply clearly states that a new organism is constructed | Silent in-place mutation |
