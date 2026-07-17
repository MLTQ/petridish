"""Reproducible short learning sweeps for sequence stepping stones."""

from __future__ import annotations

import argparse
import json
import time
from typing import Any

from .sequence_config import sequence_config
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
    task: str, profile: str, *, steps: int, seed: int, device: str
) -> dict[str, Any]:
    """Train one controlled run and return checkpointed held-out metrics."""

    if profile not in PROFILES:
        raise ValueError(f"unknown profile: {profile}")
    config = sequence_config(
        lifecycle_enabled=0,
        structural_warmup_trials=max(steps + 1, 10_000),
        **PROFILES[profile],
    )
    experiment = SequenceExperiment(task, config, seed=seed, device=device)
    started = time.perf_counter()
    checkpoints: list[dict[str, Any]] = []
    interval = max(1, min(20, steps))
    for update in range(1, steps + 1):
        experiment._train_trial()
        if update % interval == 0 or update == steps:
            checkpoints.append(
                {
                    "update": update,
                    "trainingAccuracy": round(experiment.rolling_accuracy, 4),
                    "heldOutAccuracy": round(experiment.evaluate(4), 4),
                    "loss": round(experiment.rolling_loss, 5),
                    "recallPairs": experiment.recall_pair_count,
                }
            )
    diagnostics = experiment.model.substrate.graph_diagnostics()
    return {
        "task": task, "profile": profile, "seed": seed, "device": str(experiment.device),
        "steps": steps, "seconds": round(time.perf_counter() - started, 2),
        "livingCells": int(experiment.model.substrate.occupied.sum()),
        "edgeCount": int(experiment.model.substrate.active_edge_mask.sum()),
        "minimumOutputHops": diagnostics.minimum_output_hops,
        "temporallyReachableOutputs": diagnostics.temporally_reachable_outputs,
        "checkpoints": checkpoints,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", choices=("associative_recall", "tiny_language"), required=True)
    parser.add_argument("--profile", choices=tuple(PROFILES), default="default32")
    parser.add_argument("--steps", type=int, default=80)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()
    result = run_benchmark(
        args.task, args.profile, steps=max(1, args.steps), seed=args.seed, device=args.device
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
