# Petri Dish

Petri Dish is a live laboratory for neural populations embedded in physical
space. Neurons occupy sites in a positional tensor, retain metabolic and task
state across examples, communicate through persistent directed dendrites, and
undergo growth, pruning, birth, death, and lesions.

The viewer switches among persistent MNIST, associative-recall, synthetic
tiny-language, and Tiny Shakespeare organisms. They share the same renderer,
lifecycle, lesions, directed graph, and diagnostics so classification, memory,
and autoregression remain comparable.

## Quick start

Requirements: Python 3.11+, [uv](https://docs.astral.sh/uv/), and Node.js 20+.

```bash
uv sync --extra dev
cd frontend && npm install && cd ..
uv run uvicorn petridish.server:app --app-dir backend --reload
```

The backend serves the built viewer at <http://127.0.0.1:8000/>. For frontend
development, run `npm run dev` inside `frontend/` and use port 5173.

## MNIST experiment

- The physical address space defaults to 64×64; only occupied sites are computed and
  transmitted.
- A dense initial interior contains roughly 1,900 neurons and 7,000 local
  dendrites. Population and topology persist across digits.
- Forty-nine 4×4 patch vectors enter fixed sensory neurons on the left. Ten
  class neurons occupy the right.
- Every living neuron has a trainable persistent site genotype that modulates a
  shared recurrent rule, alongside distinct metabolic, activity, and task history.
- Dendrites own source-neuron references. Existing connections carry actual
  forward messages; nearby active sources accumulate in bounded candidate-ID
  counters before new dendrites can form.
- Neurons keep private recurrent state and advertise learned keys, values, and
  emit gates. Receivers apply local attention only across real dendrites.
- Cross-entropy backpropagates through the frozen connectome. Learning begins
  with an output-bank probe, then unlocks the shared rule and synaptic weights.
- Initial dendrites are signed and zero-centered. Gated normalized messages,
  persistent spatial/type context, and explicit class identities prevent the
  original saturation and output-symmetry collapse.
- Training follows balanced 20-, 256-, 1,000-, and full-MNIST stages; each fixed
  subset must meet its overfit target before advancing.
- Metabolic pressure and population turnover activate after their own warm-up;
  free-form pruning/growth still wait for competence or a measured plateau.
- Newborn neurons inherit a local parent's genotype with configurable mutation
  noise and receive one real parent-to-child dendrite.
- Actual retained-state gradients become the backward-credit phase. They are
  not synthetic animation.
- Stimulation and traffic load update homeostatic energy separately from task
  credit. Starved or persistently overloaded neurons can die; active regions
  can seed neurons into nearby empty sites.
- Structural mutation occurs between differentiable trials. **Structural
  cycle** forces one such update.

The field layers show raw measured state: activation, energy, stimulation,
traffic load, backward credit, task utility, genotype magnitude, emission, and
occupancy. Edge opacity/width uses measured forward flow or backward credit.
There are no decorative signal particles.

## Sequence stepping stones

- **Associative recall** streams one to three random key/value bindings through
  permuted semantic ports, then asks for the value associated with a query key.
  It begins with one binding and increases difficulty only after recent accuracy
  exceeds 90%.
- **Tiny language** predicts a generated compositional sentence one token at a
  time. Visible accuracy counts only verb/object positions that require context;
  easy EOS and unigram predictions cannot inflate it.
- **Tiny Shakespeare** trains character by character on Karpathy's 1.1-million-
  character corpus. It defaults to a 64-character context on a 68×68 address
  space with batch 16 and at most 4,096 initially occupied sites. The corpus is downloaded
  once from the canonical `char-rnn` source and cached under `data/`.
- Sequence neurons retain recurrent state across tokens. Six local graph updates
  run per token on a benchmark-selected 24×24 field.
- A low-rank broadcast workspace lets neurons advertise and selectively read
  transient shared state. Experimental fast-weight linear attention is available
  but disabled by default because it did not solve two-binding retrieval.
- Ports are permuted or direction-reversed, so success cannot be explained as
  copying ordered values across the field.

Run a controlled sweep with:

```bash
uv run python -m petridish.benchmark_sequences \
  --task associative_recall --profile compact24 --steps 80
```

See [`docs/sequence-benchmark-results.md`](docs/sequence-benchmark-results.md)
for measured results and negative findings.

## Controls

- **Pause / Step**: inspect input, recurrent forward traffic, backward credit,
  and structural phases independently.
- **Train fast**: sequence organisms run full optimizer, local-credit,
  homeostasis, and structural updates without building or replaying token traces.
  The viewer remains responsive and reports measured updates/second; stopping
  fast training rebuilds one current trace before visualization resumes.
- **Evaluate test set**: evaluate without weight or topology mutation.
- **Lifecycle cycle**: force pruning, candidate accumulation, growth, death,
  inheritance, and birth once.
- **Lesion brush**: remove neurons and every incident dendrite at the selected
  physical positions.
- **Edge floor**: presentation-only minimum absolute synaptic weight.
- **Hyperparameters**: stage any numeric model, learning, growth, pruning, or
  homeostasis settings and apply them together by restarting the organism.
- **Saved organism**: select a trusted `runs/*/latest.pt` trainer checkpoint and
  load its learned graph, model, optimizer, histories, and random state paused for
  held-out evaluation or interactive generation.
- **Square field size**: choose 16, 32, 64, 128, 256, 512, or 1024; Tiny Shakespeare
  additionally offers its exact 68×68 single-column port geometry. Tensor extent
  and initial population are separate controls, so a large address space need
  not begin densely occupied.
- **Corpus generation**: enter a character prompt, install it as the active
  context, and request one sampled next character at a time while inspecting the
  corresponding organism state. The displayed next-token prediction is the
  greedy diagnostic; generation samples at temperature 0.85.
- **Synapse Δ / |w|**: measured relative optimizer movement, not an animation.
- **Structure**: reports the exact developmental unlock condition.
- **Lifecycle**: reports activation, energy/stress, turnover, neuron age/lineage,
  and starvation/overload/maintenance death causes.
- **Diagnostics**: report learning phase, hop distance, outputs reachable within
  the recurrent time budget, attention entropy, and effective parameter count.
- **Performance diagnostics**: report actual update latency and trace-free
  optimizer throughput instead of treating visualization-frame speed as training speed.

## Verification

```bash
uv run pytest
cd frontend && npm run check && npm run build
```

## Scientific status

This is a developmental substrate, not a competitive handwriting or language model. It has
no CNN feature extractor, predefined long-range connectome, or image painted
onto the field. Its small dense readout can access only the ten physical output
neurons and exists to prove whether the graph's representation is separable.
Input influence reaches all ten outputs within the default recurrent budget;
the overfit curriculum now makes failure explicit before generalization claims
are attempted. Displayed traffic, attention, credit, topology, and reachability
are backend measurements. Configuration changes start a new run so histories
from different settings are never mixed. Current sequence results solve one-binding
recall and exceed chance on context-dependent grammar predictions, but do not yet
demonstrate genuine two-binding content-addressed retrieval; that failure is the
target for the next sparse write-ownership experiment.

The Tiny Shakespeare corpus is sourced from
[Karpathy's `char-rnn` repository](https://github.com/karpathy/char-rnn/tree/master/data/tinyshakespeare);
Shakespeare's works are also available through
[Project Gutenberg](https://www.gutenberg.org/ebooks/100).

See `docs/architecture.md` for tensor, credit, and lifecycle contracts.
