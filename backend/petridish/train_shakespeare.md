# Resumable Tiny Shakespeare trainer

`train_shakespeare.py` runs the same trace-free `SequenceExperiment.train_updates`
path used by live fast training and the CUDA benchmark. It defaults to the required
68×68, batch-16, context-64, two-message-step baseline with lifecycle disabled and
both lifecycle and structural warm-ups set to 5,000 updates. Measured batch/AMP
choices can be supplied explicitly.

The trainer writes one append-only JSONL record per optimizer update plus separate
held-out records at an infrequent configurable interval. `latest.pt` is replaced
atomically and contains model parameters, optimizer moments, all substrate/topology
buffers, generation and update counters, configuration, vocabulary, rolling metrics,
and Python, NumPy, Torch, CUDA, sampler, and evaluation random states.

By default a fresh invocation resumes `latest.pt` when present. Resume restores the
saved configuration, context, seed, vocabulary contract, AMP mode, organism, and
optimizer before continuing from the saved update count. `SIGINT` and `SIGTERM` set a
stop flag; the current indivisible update finishes, then a final atomic checkpoint is
written. Progress reports loss, accuracy, update and character throughput, GPU memory,
and a finite-loss/gradient check.

Compilation remains opt-in because the measured stable-forward attempt currently has
dynamic topology graph breaks; production runs should use `--compile off`.

Example:

```bash
CUDA_VISIBLE_DEVICES=0 uv run python -m petridish.train_shakespeare \
  --device cuda --field-size 68 --batch-size 64 --context-length 64 \
  --message-steps 2 --amp bfloat16 --compile off --updates 100000 \
  --checkpoint-dir runs/shakespeare-4090 --checkpoint-interval 100 \
  --eval-interval 500 --eval-batches 4 --progress-interval 10
```
