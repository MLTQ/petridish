"""Machine-readable RTX benchmark for trace-free Tiny Shakespeare training."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
import math
from pathlib import Path
import statistics
import time
from typing import Any

import torch

from .corpus_task import load_tiny_shakespeare_task
from .sequence_config import sequence_config
from .sequence_experiment import SequenceExperiment


def _synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def run_benchmark(
    *,
    device: str,
    field_size: int,
    batch_size: int,
    context_length: int,
    message_steps: int,
    warmup_updates: int,
    measured_updates: int,
    seed: int,
    amp: str = "off",
    compile_mode: str = "off",
    evaluation_batches: int = 2,
    profile_output: Path | None = None,
) -> dict[str, Any]:
    """Run the live training semantics without traces or timed evaluation."""

    task = load_tiny_shakespeare_task(context_length)
    config = sequence_config(
        "tiny_shakespeare",
        width=field_size,
        height=field_size,
        batch_size=batch_size,
        message_steps=message_steps,
        lifecycle_enabled=0,
        structural_warmup_trials=max(5_000, warmup_updates + measured_updates + 1),
    )
    experiment = SequenceExperiment(
        task, config, seed=seed, device=device, amp_mode=amp
    )
    if len(task.vocabulary) != 66:
        raise RuntimeError(f"expected 66 Tiny Shakespeare characters, got {len(task.vocabulary)}")

    compile_setup_seconds = 0.0
    if compile_mode != "off":
        compile_started = time.perf_counter()
        experiment.enable_compile(compile_mode)
        compile_setup_seconds = time.perf_counter() - compile_started
    warmup_started = time.perf_counter()
    for _ in range(max(0, warmup_updates)):
        experiment.train_updates(1)
    _synchronize(experiment.device)
    warmup_seconds = time.perf_counter() - warmup_started
    if experiment.device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(experiment.device)

    latencies: list[float] = []
    curve: list[dict[str, float | int]] = []
    measured_started = time.perf_counter()
    for _ in range(max(1, measured_updates)):
        _synchronize(experiment.device)
        started = time.perf_counter()
        experiment.train_updates(1)
        _synchronize(experiment.device)
        latency = time.perf_counter() - started
        latencies.append(latency)
        curve.append(
            {
                "update": experiment.training_step,
                "loss": experiment.last_loss,
                "accuracy": experiment.last_batch_accuracy,
                "latencySeconds": latency,
            }
        )
    _synchronize(experiment.device)
    measured_seconds = time.perf_counter() - measured_started

    held_out = experiment.evaluate_metrics(evaluation_batches)
    finite_loss = math.isfinite(experiment.last_loss) and math.isfinite(held_out["loss"])
    finite_gradients = all(
        parameter.grad is None or bool(torch.isfinite(parameter.grad).all())
        for parameter in experiment.model.parameters()
    )
    profiler_table: str | None = None
    if profile_output is not None:
        profile_output.parent.mkdir(parents=True, exist_ok=True)
        activities = [torch.profiler.ProfilerActivity.CPU]
        if experiment.device.type == "cuda":
            activities.append(torch.profiler.ProfilerActivity.CUDA)
        with torch.profiler.profile(
            activities=activities,
            record_shapes=True,
            profile_memory=True,
        ) as profiler:
            experiment.train_updates(1)
        profiler.export_chrome_trace(str(profile_output))
        profiler_table = profiler.key_averages().table(
            sort_by=(
                "self_cuda_time_total"
                if experiment.device.type == "cuda"
                else "self_cpu_time_total"
            ),
            row_limit=20,
        )
    substrate = experiment.model.substrate
    device_name = (
        torch.cuda.get_device_name(experiment.device)
        if experiment.device.type == "cuda"
        else str(experiment.device)
    )
    peak_allocated = (
        torch.cuda.max_memory_allocated(experiment.device)
        if experiment.device.type == "cuda"
        else 0
    )
    peak_reserved = (
        torch.cuda.max_memory_reserved(experiment.device)
        if experiment.device.type == "cuda"
        else 0
    )
    count = len(latencies)
    return {
        "schemaVersion": 1,
        "configuration": {
            **asdict(config),
            "task": task.key,
            "contextLength": context_length,
            "warmupUpdates": warmup_updates,
            "measuredUpdates": count,
            "seed": seed,
            "amp": amp,
            "compile": compile_mode,
        },
        "environment": {
            "device": str(experiment.device),
            "deviceName": device_name,
            "pytorchVersion": torch.__version__,
            "cudaVersion": torch.version.cuda,
        },
        "population": {
            "livingNeurons": int(substrate.occupied.sum()),
            "activeEdges": int(substrate.active_edge_mask.sum()),
        },
        "timing": {
            "compileSetupSeconds": compile_setup_seconds,
            "warmupSeconds": warmup_seconds,
            "measuredSeconds": measured_seconds,
            "meanUpdateSeconds": statistics.fmean(latencies),
            "medianUpdateSeconds": statistics.median(latencies),
            "updatesPerSecond": count / measured_seconds,
            "sequencesPerSecond": count * batch_size / measured_seconds,
            "targetCharactersPerSecond": (
                count * batch_size * context_length / measured_seconds
            ),
        },
        "memory": {
            "peakAllocatedBytes": peak_allocated,
            "peakReservedBytes": peak_reserved,
        },
        "training": {
            "rollingLoss": experiment.rolling_loss,
            "rollingAccuracy": experiment.rolling_accuracy,
            "curve": curve,
        },
        "heldOut": held_out,
        "numerics": {
            "finiteLoss": finite_loss,
            "finiteGradients": finite_gradients,
        },
        "profiler": {
            "trace": str(profile_output) if profile_output is not None else None,
            "topOperations": profiler_table,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--field-size", type=int, default=68)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--context-length", type=int, default=64)
    parser.add_argument("--message-steps", type=int, default=2)
    parser.add_argument("--warmup-updates", type=int, default=3)
    parser.add_argument("--measured-updates", type=int, default=5)
    parser.add_argument("--evaluation-batches", type=int, default=2)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--amp", choices=("off", "bfloat16"), default="off")
    parser.add_argument(
        "--compile", dest="compile_mode",
        choices=("off", "default", "reduce-overhead", "max-autotune"), default="off",
    )
    parser.add_argument("--output", type=Path)
    parser.add_argument("--profile-output", type=Path)
    args = parser.parse_args()
    result = run_benchmark(
        device=args.device,
        field_size=args.field_size,
        batch_size=args.batch_size,
        context_length=args.context_length,
        message_steps=args.message_steps,
        warmup_updates=max(0, args.warmup_updates),
        measured_updates=max(1, args.measured_updates),
        seed=args.seed,
        amp=args.amp,
        compile_mode=args.compile_mode,
        evaluation_batches=max(1, args.evaluation_batches),
        profile_output=args.profile_output,
    )
    payload = json.dumps(result, indent=2)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload + "\n", encoding="utf-8")
    print(payload)


if __name__ == "__main__":
    main()
