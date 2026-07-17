# RTX 4090 Tiny Shakespeare results

## Outcome

The required Tiny Shakespeare organism now uses an exact 68×68 field, batch 16 by
default, context 64, two recurrent message steps, all 66 vocabulary ports in one
unique input column and one unique output column, lifecycle disabled, a 5,000-update
structural warm-up, and the existing 4,096 initial-population cap. The live control
offers 68 only for Tiny Shakespeare.

The selected unattended profile is batch 64 with CUDA bfloat16 and compile off. It
measured 6,147 target characters/s in the short sweep at 18.34 GB peak allocation.
FP32 batch 32 is the lower-memory alternative at 3,584 target characters/s and
12.31 GB. FP32 batch 64 was stopped before OOM based on the measured batch-32
footprint.

## Benchmarks

All rows use seed 1, context 64, two message steps, lifecycle/topology mutation off,
synchronized CUDA timing, and held-out evaluation outside the timed interval. Memory
is peak allocated GB (decimal). “Chars/s” means supervised target characters/s, not
optimizer updates/s.

| Stage | Field / batch | AMP | Mean s/update | Chars/s | Peak GB | Decision |
|---|---:|---|---:|---:|---:|---|
| Historical eager baseline | 128² / 4 | FP32 | 0.852 | 301 | 6.12 | Reference |
| Eager batch sweep | 68² / 8 | FP32 | 0.846 | 605 | 3.60 | Baseline |
| Requested eager baseline | 68² / 16 | FP32 | 0.869 | 1,178 | 7.11 | Baseline |
| Eager batch sweep | 68² / 32 | FP32 | 0.861 | 2,379 | 14.17 | Baseline |
| Eager batch sweep | 68² / 64 | FP32 | — | — | projected >24 GB | Skipped before OOM |
| Loop invariants hoisted | 68² / 16 | FP32 | 0.784 | 1,306 | 7.11 | Accepted (+10.8%) |
| Viewer projection removed | 68² / 16 | FP32 | 0.777 | 1,319 | 7.11 | Accepted (+1.0%) |
| AMP control | 68² / 16 | FP32 | 0.773 | 1,325 | 7.11 | Control |
| CUDA AMP | 68² / 16 | BF16 | 0.884 | 1,158 | 5.36 | Optional; rejected for speed |
| `torch.compile` | 68² / 16 | FP32 | — | — | — | Rejected; graph breaks/backward failure |
| Skip disabled fast weights | 68² / 16 | FP32 | 0.586 | 1,749 | 6.18 | Accepted (+32.0%) |
| Optimized batch sweep | 68² / 32 | FP32 | 0.571 | 3,584 | 12.31 | Accepted alternative |
| Optimized batch sweep | 68² / 64 | BF16 | 0.666 | 6,147 | 18.34 | Selected |

The 40-update selected-profile run averaged 6,012 chars/s including first-update
allocator startup. Loss moved from 4.424 at update 1 to a 3.614 rolling value;
held-out loss was 3.542 versus `ln(66)=4.190`, and held-out accuracy was 15.23%
versus the 1.52% uniform baseline. Losses and gradients were finite. This demonstrates
short-run learning, not language-model or LLM-like capability.

## Profiler

Before the final inactive-path cleanup, `aten::bmm` accounted for 34.97% of CUDA
self-time across 2,304 calls. Matrix multiplies accounted for another 11.12%, indexed
writes 8.93%, and elementwise multiplication 7.64%. The zero-configured fast-weight
path was still performing outer-product writes and reads; bypassing that documented
ablation produced the largest accepted speedup. Remaining costs are the active
broadcast/message BMMs, small GEMMs, indexed scatter/write kernels, layer-norm
backward, and thousands of small elementwise launches.

## Verification and operational commands

Environment and baseline:

```bash
nvidia-smi
uv sync --extra dev
CUDA_VISIBLE_DEVICES=0 uv run python -c 'import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available(), torch.cuda.get_device_name(0), torch.cuda.get_device_capability(0))'
CUDA_VISIBLE_DEVICES=0 uv run pytest
cd frontend && npm ci && npm run check && npm run build
```

Representative benchmark (all JSON artifacts are under `benchmarks/4090/`):

```bash
CUDA_VISIBLE_DEVICES=0 uv run python -m petridish.benchmark_shakespeare \
  --device cuda --field-size 68 --batch-size 64 --context-length 64 \
  --message-steps 2 --warmup-updates 1 --measured-updates 2 \
  --evaluation-batches 1 --seed 1 --amp bfloat16 --compile off \
  --output benchmarks/4090/07_optimized_68_b64_bfloat16.json
```

Unattended trainer:

```bash
CUDA_VISIBLE_DEVICES=0 uv run python -m petridish.train_shakespeare \
  --device cuda --field-size 68 --batch-size 64 --context-length 64 \
  --message-steps 2 --amp bfloat16 --compile off --updates 100000 \
  --checkpoint-dir runs/shakespeare-4090 --checkpoint-interval 100 \
  --eval-interval 500 --eval-batches 4 --progress-interval 10
```

Live viewer:

```bash
CUDA_VISIBLE_DEVICES=0 uv run uvicorn petridish.server:app --app-dir backend \
  --host 127.0.0.1 --port 8000
```

Checkpoint/resume was exercised across two fresh Python processes: the first stopped
at update 2 and the second loaded the same organism/optimizer/RNG state and continued
from update 2 to update 3. `SIGINT`/`SIGTERM` finish the current indivisible update and
atomically replace `latest.pt`.
