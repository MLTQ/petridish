"""Causal lifecycle and severe-lesion ablations from one competent clone point."""

from __future__ import annotations

import argparse
import copy
from dataclasses import dataclass
import os
from pathlib import Path
import time
from typing import Any

import torch

from .benchmark_recovery import _apply_branch_config, _artifact, _checkpoint
from .benchmark_sequences import PROFILES, _write_result
from .sequence_config import sequence_config
from .sequence_experiment import SequenceExperiment


@dataclass(frozen=True)
class BranchSpec:
    name: str
    lesion_radius: float
    lifecycle: bool
    interval: int = 8
    births_per_generation: int | None = None
    max_deaths_per_generation: int | None = None


BRANCH_SPECS = (
    BranchSpec("control_static", 0, False),
    BranchSpec("control_lifecycle_i8_b12_d48", 0, True),
    BranchSpec("lesion_r8_static", 8, False),
    BranchSpec("lesion_r8_lifecycle_i8_b12_d48", 8, True),
    BranchSpec(
        "lesion_r8_lifecycle_i32_b4_d8", 8, True, interval=32,
        births_per_generation=4, max_deaths_per_generation=8,
    ),
)


def run_ablation(
    output_dir: Path, *, base_steps: int = 1_200, recovery_steps: int = 240,
    seed: int = 11, device: str = "cuda",
) -> dict[str, Any]:
    """Train once, then isolate lesion severity and lifecycle mutation pressure."""

    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.use_deterministic_algorithms(True)
    config = sequence_config(
        cell_architecture="gru", lifecycle_enabled=0,
        structural_warmup_trials=max(base_steps + recovery_steps + 1, 10_000),
        **PROFILES["compact24_binding_sparse"],
    )
    base = SequenceExperiment(
        "associative_recall", config, seed=seed, device=device,
        recall_pair_count=3, recall_pair_max=3,
    )
    for _ in range(base_steps):
        base.train_updates(1)
    base_metrics = base.evaluate_metrics(12)
    branches = {spec.name: copy.deepcopy(base) for spec in BRANCH_SPECS}
    center_x = (config.width - 1) / 2
    center_y = (config.height - 1) / 2
    lesion_counts: dict[str, int] = {}
    for spec in BRANCH_SPECS:
        lesion_counts[spec.name] = (
            branches[spec.name].model.substrate.lesion(
                center_x, center_y, spec.lesion_radius
            )
            if spec.lesion_radius > 0 else 0
        )
    severe_counts = {
        lesion_counts[spec.name] for spec in BRANCH_SPECS if spec.lesion_radius == 8
    }
    if len(severe_counts) != 1:
        raise RuntimeError("matched radius-8 branches removed different cells")

    summaries: dict[str, Any] = {}
    for spec in BRANCH_SPECS:
        experiment = branches[spec.name]
        _apply_branch_config(
            experiment, lifecycle=spec.lifecycle, interval=spec.interval,
            births_per_generation=spec.births_per_generation,
            max_deaths_per_generation=spec.max_deaths_per_generation,
        )
        checkpoints = [_checkpoint(experiment, 0, evaluation_batches=8)]
        started = time.perf_counter()
        path = output_dir / f"ablation-{spec.name}-seed{seed}.json"

        def publish(status: str) -> dict[str, Any]:
            payload = _artifact(
                experiment, branch=spec.name, seed=seed, steps=recovery_steps,
                base_steps=base_steps, lesion_count=lesion_counts[spec.name],
                lesion_radius=spec.lesion_radius, checkpoints=checkpoints,
                status=status, started=started, profile="lifecycle_ablation_r8",
            )
            payload["lifecycleConfiguration"] = {
                "enabled": spec.lifecycle,
                "interval": spec.interval,
                "birthsPerGeneration": experiment.config.births_per_generation,
                "maxDeathsPerGeneration": experiment.config.max_deaths_per_generation,
            }
            _write_result(path, payload)
            return payload

        publish("running")
        for update in range(1, recovery_steps + 1):
            experiment.train_updates(1)
            if update % 20 == 0 or update == recovery_steps:
                checkpoints.append(_checkpoint(experiment, update, evaluation_batches=8))
                publish("running")
        summaries[spec.name] = publish("complete")
    return {"baseAccuracy": float(base_metrics["accuracy"]), "branches": summaries}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--base-steps", type=int, default=1_200)
    parser.add_argument("--recovery-steps", type=int, default=240)
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    result = run_ablation(
        args.output_dir, base_steps=max(1, args.base_steps),
        recovery_steps=max(1, args.recovery_steps), seed=args.seed,
        device=args.device,
    )
    print({
        "baseAccuracy": result["baseAccuracy"],
        "branches": {
            name: payload["checkpoints"][-1]
            for name, payload in result["branches"].items()
        },
    })


if __name__ == "__main__":
    main()
