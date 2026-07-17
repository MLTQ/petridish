# Petri Dish

Petri Dish is a live neural cellular-automata laboratory. A continuous 2D cell
field communicates through mutable local state and a sparse directed graph whose
weighted edges grow, learn, and prune while the experiment runs.

Two experiments share the live viewer: a 32×32 delayed-XOR organism with
reward-modulated local plasticity, and a 16×16 MNIST population whose shared
recurrent cells assemble a fresh sparse connectome for every digit.

## Quick start

Requirements: Python 3.11+, [uv](https://docs.astral.sh/uv/), and Node.js 20+.

```bash
uv sync --extra dev
cd frontend && npm install && cd ..
```

Run the backend in one terminal:

```bash
uv run uvicorn petridish.server:app --app-dir backend --reload
```

Run the viewer in another:

```bash
cd frontend
npm run dev
```

Open <http://localhost:5173>. The Vite development server proxies `/ws` and
`/api` to the simulation server on port 8000.

## What to try

- Switch **Experiment** to **MNIST self-assembler**. The first switch downloads the
  canonical 60,000/10,000 train/test splits into ignored `data/` storage.
- Watch the seed frame begin with no long-range edges. Seven sensory ports then
  receive one row of 4×4 patch tokens at a time while cells advertise axon keys
  and dendritic receptors.
- Pause and step through sensing, recurrent development, and readout. **New
  assembly** skips to another trained episode and returns to an empty graph.
- Lesion the MNIST population: the same digit is replayed from the seed under the
  damage so the graph must route around dead cells.
- Switch the field layer between phase, activation, energy, growth, and reward.
- Use **Stim A**, **Stim B**, and **Reward** to inject signals.
- Arm the lesion brush, then drag across the field to kill cells and cut their
  incident axons.
- Reset with the same seed to reproduce an episode exactly.
- Watch young axons brighten, weighted signals pulse toward their destinations,
  and low-utility connections disappear.

## Architecture

```text
XOR local plasticity ─┐
                     ├→ FastAPI snapshot stream → PixiJS/Canvas field renderer
MNIST recurrent assembly ┘        ↑                         ↓
                         WebSocket controls          charts/inspection
```

The simulation is authoritative. Rendering runs at a lower rate than physics,
so viewer load never determines the learning timestep. See `docs/architecture.md`
for state and update contracts.

## Verification

```bash
uv run pytest
cd frontend && npm run check && npm run build
```

## Current scientific status

This is an experimental substrate, not a competitive handwriting model. The
MNIST experiment deliberately has no CNN feature extractor, dense classifier,
predefined long-range connectome, or image painted onto the field. An image is
split into 49 patch vectors and streamed through seven sensory interface cells.
All 256 cells run one shared GRU; key/query broadcasts assemble at most four
persistent outgoing axons per cell, and ten output interface cells expose the
class logits.

Supervised gradients currently meta-train the shared cell program, patch
projection, broadcast language, and readout. Endpoint choices, cell states, and
episode synapses are constructed anew at runtime rather than stored as global
per-cell parameters. Reward-modulated lifetime plasticity remains the next
scientific layer. A short seed-31 smoke run reached about 19% on a bounded
256-image held-out slice after 150 updates, confirming learning above chance but
not yet constituting a serious MNIST benchmark.
