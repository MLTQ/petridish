"""Reproducible short learning sweeps for sequence stepping stones."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import time
from typing import Any

import torch

from .sequence_config import sequence_config
from .sequence_cells import CELL_ARCHITECTURES
from .sequence_experiment import SequenceExperiment


PROFILES: dict[str, dict[str, Any]] = {
    "shallow32": {"message_steps": 2},
    "default32": {},
    "compact24": {
        "width": 24, "height": 24, "message_steps": 6,
        "local_radius": 8, "candidate_probes": 28, "initial_density": 0.48,
        "message_gain": 2.0,
    },
    "compact24_no_broadcast": {
        "width": 24, "height": 24, "message_steps": 6,
        "local_radius": 8, "candidate_probes": 28, "initial_density": 0.48,
        "message_gain": 2.0, "broadcast_gain": 0.0,
    },
    "compact24_no_global_memory": {
        "width": 24, "height": 24, "message_steps": 6,
        "local_radius": 8, "candidate_probes": 28, "initial_density": 0.48,
        "message_gain": 2.0, "broadcast_gain": 0.0, "fast_weight_gain": 0.0,
    },
    "compact24_fast_weights": {
        "width": 24, "height": 24, "message_steps": 6,
        "local_radius": 8, "candidate_probes": 28, "initial_density": 0.48,
        "message_gain": 2.0, "fast_weight_gain": 0.5,
    },
}


def run_benchmark(
    task: str, profile: str, *, steps: int, seed: int, device: str,
    architecture: str = "gru",
    fixed_recall_pairs: int | None = None,
    output_path: Path | None = None,
) -> dict[str, Any]:
    """Train one controlled run and return checkpointed held-out metrics."""

    if profile not in PROFILES:
        raise ValueError(f"unknown profile: {profile}")
    config = sequence_config(
        cell_architecture=architecture,
        lifecycle_enabled=0,
        structural_warmup_trials=max(steps + 1, 10_000),
        **PROFILES[profile],
    )
    if fixed_recall_pairs is not None and task != "associative_recall":
        raise ValueError("fixed recall pairs apply only to associative recall")
    experiment = SequenceExperiment(
        task, config, seed=seed, device=device,
        recall_pair_count=fixed_recall_pairs,
        recall_pair_max=fixed_recall_pairs or 3,
    )
    started = time.perf_counter()
    checkpoints: list[dict[str, Any]] = []
    parameter_count = sum(parameter.numel() for parameter in experiment.model.parameters())
    trainable_parameter_count = sum(
        parameter.numel() for parameter in experiment.model.parameters()
        if parameter.requires_grad
    )
    initial_cuda_gib = (
        torch.cuda.memory_allocated(experiment.device) / 2**30
        if experiment.device.type == "cuda" else 0.0
    )
    if experiment.device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(experiment.device)

    def result(status: str, completed_steps: int) -> dict[str, Any]:
        diagnostics = experiment.model.substrate.graph_diagnostics()
        return {
            "task": task, "profile": profile, "architecture": architecture,
            "recallMode": (
                f"fixed_{fixed_recall_pairs}"
                if fixed_recall_pairs is not None else "adaptive"
            ),
            "seed": seed, "device": str(experiment.device), "steps": steps,
            "completedSteps": completed_steps, "status": status,
            "seconds": round(time.perf_counter() - started, 2),
            "parameterCount": parameter_count,
            "trainableParameterCount": trainable_parameter_count,
            "cudaAllocatedGiB": round(initial_cuda_gib, 4),
            "peakCudaAllocatedGiB": round(
                torch.cuda.max_memory_allocated(experiment.device) / 2**30, 4
            ) if experiment.device.type == "cuda" else 0.0,
            "livingCells": int(experiment.model.substrate.occupied.sum()),
            "edgeCount": int(experiment.model.substrate.active_edge_mask.sum()),
            "minimumOutputHops": diagnostics.minimum_output_hops,
            "temporallyReachableOutputs": diagnostics.temporally_reachable_outputs,
            "checkpoints": list(checkpoints),
        }

    if output_path is not None:
        _write_result(output_path, result("running", 0))
    interval = max(1, min(20, steps))
    for update in range(1, steps + 1):
        experiment.train_updates(1)
        if update % interval == 0 or update == steps:
            held_out = experiment.evaluate_metrics(4)
            checkpoints.append(
                {
                    "update": update,
                    "trainingAccuracy": round(experiment.rolling_accuracy, 4),
                    "heldOutAccuracy": round(float(held_out["accuracy"]), 4),
                    "heldOutSlotAccuracy": [
                        round(float(accuracy), 4)
                        for accuracy in held_out.get("slotAccuracy", [])
                    ],
                    "loss": round(experiment.rolling_loss, 5),
                    "recallPairs": experiment.recall_pair_count,
                }
            )
            if output_path is not None:
                _write_result(output_path, result("running", update))
    final = result("complete", steps)
    if output_path is not None:
        _write_result(output_path, final)
    return final


def _write_result(path: Path, payload: dict[str, Any]) -> None:
    """Atomically replace one live benchmark artifact."""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", choices=("associative_recall", "tiny_language"), required=True)
    parser.add_argument("--profile", choices=tuple(PROFILES), default="default32")
    parser.add_argument("--steps", type=int, default=80)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--architecture", choices=CELL_ARCHITECTURES, default="gru")
    parser.add_argument("--fixed-recall-pairs", type=int, choices=(1, 2, 3))
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = run_benchmark(
        args.task, args.profile, steps=max(1, args.steps), seed=args.seed,
        device=args.device, architecture=args.architecture,
        fixed_recall_pairs=args.fixed_recall_pairs,
        output_path=args.output,
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
