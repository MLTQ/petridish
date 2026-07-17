# Tiny Shakespeare CUDA benchmark

`benchmark_shakespeare.py` measures trace-free Tiny Shakespeare optimizer updates
using the same `SequenceExperiment`, model, sampler, optimizer, and update method as
the live application. Warm-up updates precede synchronized CUDA timing, and held-out
evaluation follows it, so neither allocator startup nor validation contaminates the
training interval.

The CLI controls device, square field size, batch size, corpus context, recurrent
message steps, warm-up and measured update counts, seed, AMP, and optional compile
mode. JSON output includes the complete scientific configuration, hardware/software
identity, population and edge counts, mean/median latency, work-normalized throughput,
CUDA memory peaks, the measured loss curve, and final held-out loss and accuracy.

CUDA bfloat16 autocast is optional; FP32 remains the default comparison. Compile
modes wrap only the stable model-forward callable and report setup plus warm-up time
separately from steady-state updates. Benchmark artifacts can be written for later
comparisons with `--output`.

`--profile-output` runs one additional untimed optimizer update under
`torch.profiler`, exports a Chrome trace, and embeds the twenty highest self-time
operators in the JSON report.

Example:

```bash
CUDA_VISIBLE_DEVICES=0 uv run python -m petridish.benchmark_shakespeare \
  --field-size 68 --batch-size 16 --context-length 64 --message-steps 2 \
  --warmup-updates 3 --measured-updates 5 --output benchmarks/4090/baseline.json
```
