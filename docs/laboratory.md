# Two-GPU experiment laboratory

## Purpose

The laboratory separates unattended trainer processes from the interactive
organism viewer. A CPU-hosted FastAPI process reads append-only metrics and
NVIDIA telemetry, while each experiment runs in its own process pinned to a GPU
UUID. Opening or refreshing the browser therefore cannot acquire trainer VRAM.

## Topology

```text
Mac browser -- SSH tunnel --> FastAPI on Aine (CPU)
                              |-- read nvidia-smi + runs/*/metrics.jsonl
                              |-- 4090 trainer process
                              `-- 2070 trainer process
```

The host's `nvidia-smi` and CUDA indices are not aligned. Jobs are pinned by the
stable UUIDs advertised by `/api/lab`, never by an integer index.

## Run contract

Every server-launched run receives its own `runs/<run-id>/` directory containing:

- `manifest.json`: immutable requested configuration, architecture, GPU UUID,
  source commit, command, creation time, and PID;
- `metrics.jsonl`: append-only training and held-out measurements;
- `latest.pt`: atomically replaced resumable checkpoint;
- `trainer.log`: process output for operator diagnosis.

The UI displays only measured values. Missing measurements remain absent rather
than being estimated. Up to four selected runs are compared using rolling loss
against optimizer update, not wall time.

## Safety

- Launch and stop routes are disabled unless `PETRIDISH_LAB_CONTROL=1`.
- The production server should bind to `127.0.0.1`; access it through SSH.
- Run IDs are lowercase slugs and cannot address paths outside `runs/`.
- Commands are constructed as argument arrays with no shell interpolation.
- Stop sends SIGTERM to the trainer's checkpoint-safe shutdown path.
- Server shutdown never terminates independent trainers.

## Remote launch

From the deployed worktree on Aine:

```bash
cd ~/Code/petridish-lab
uv sync --extra dev
cd frontend && npm ci && npm run build && cd ..
PETRIDISH_DEVICE=cpu PETRIDISH_AUTOPLAY=0 PETRIDISH_LAB_CONTROL=1 \
PETRIDISH_RUN_ROOT=/home/m/Code/petridish/runs \
  uv run uvicorn petridish.server:app --app-dir backend \
  --host 127.0.0.1 --port 8000
```

From the Mac:

```bash
ssh -N -L 8000:127.0.0.1:8000 m@192.168.0.203
```

Then open <http://127.0.0.1:8000/>. The tunnel is the access boundary; do not
bind the control service to the LAN without adding authentication.

## Extension contract

An architecture may appear in the launch selector only when all of the following
exist:

1. a trainer/model configuration path;
2. checkpoint metadata sufficient to reconstruct it;
3. fixed-seed tests and a controlled benchmark;
4. metrics that distinguish its scientific behavior;
5. laboratory capability advertisement.

The first homogeneous capabilities are `gru`, `lstm`, `esn`, and `transformer`.
The transformer gives every neuron four private temporal memory slots; it does not
replace the physical field's sparse dendritic attention. Heterogeneous mixtures
remain unavailable until type inheritance, energy cost, and mutation are explicit.
