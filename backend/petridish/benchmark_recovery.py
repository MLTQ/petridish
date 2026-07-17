"""Matched lesion and lifecycle recovery experiments for competent sequence organisms."""

from __future__ import annotations

import argparse
import copy
from dataclasses import replace
import os
from pathlib import Path
import time
from typing import Any

import torch

from .benchmark_sequences import PROFILES, _write_result
from .sequence_config import sequence_config
from .sequence_experiment import SequenceExperiment


BRANCHES = ("control", "lesion_static", "lesion_lifecycle")


def _apply_branch_config(
    experiment: SequenceExperiment, *, lifecycle: bool
) -> None:
    """Replace the shared immutable config consistently across one cloned branch."""

    config = replace(
        experiment.config,
        lifecycle_enabled=int(lifecycle),
        lifecycle_warmup_trials=0,
        lifecycle_interval=8,
        structural_warmup_trials=0,
        structural_interval=8,
    )
    experiment.config = config
    experiment.model.config = config
    experiment.model.substrate.config = config
    experiment.lifecycle_active = lifecycle
    experiment.lifecycle_reason = (
        "energy pressure and turnover active"
        if lifecycle else "disabled by matched recovery branch"
    )
    experiment.structure_unlocked = lifecycle
    experiment.structure_unlock_reason = (
        "matched repair branch" if lifecycle else "disabled by matched recovery branch"
    )


def _checkpoint(
    experiment: SequenceExperiment, recovery_update: int, *, evaluation_batches: int
) -> dict[str, Any]:
    metrics = experiment.evaluate_metrics(evaluation_batches)
    substrate = experiment.model.substrate
    return {
        "update": recovery_update,
        "trainingUpdate": experiment.training_step,
        "heldOutAccuracy": round(float(metrics["accuracy"]), 4),
        "heldOutSlotAccuracy": [
            round(float(value), 4) for value in metrics.get("slotAccuracy", [])
        ],
        "heldOutPresentedValueRate": round(
            float(metrics.get("presentedValueRate", 0.0)), 4
        ),
        "heldOutDistractorRate": round(float(metrics.get("distractorRate", 0.0)), 4),
        "heldOutAbsentValueRate": round(float(metrics.get("absentValueRate", 0.0)), 4),
        "loss": round(float(metrics["loss"]), 5),
        "recallPairs": experiment.recall_pair_count,
        "livingCells": int(substrate.occupied.sum()),
        "edgeCount": int(substrate.active_edge_mask.sum()),
        "generation": substrate.generation,
        "cumulativeBirths": experiment.cumulative_births,
        "cumulativeDeaths": experiment.cumulative_deaths,
        "deathCauses": dict(experiment.cumulative_death_causes),
    }


def _artifact(
    experiment: SequenceExperiment,
    *,
    branch: str,
    seed: int,
    steps: int,
    base_steps: int,
    lesion_count: int,
    lesion_radius: float,
    checkpoints: list[dict[str, Any]],
    status: str,
    started: float,
) -> dict[str, Any]:
    substrate = experiment.model.substrate
    return {
        "task": "associative_recall",
        "profile": "matched_recovery",
        "architecture": experiment.config.cell_architecture,
        "intervention": branch,
        "recallMode": "fixed_3",
        "seed": seed,
        "device": str(experiment.device),
        "deterministic": True,
        "steps": steps,
        "completedSteps": checkpoints[-1]["update"] if checkpoints else 0,
        "baseSteps": base_steps,
        "status": status,
        "seconds": round(time.perf_counter() - started, 2),
        "lesionCount": lesion_count,
        "lesionRadius": lesion_radius,
        "livingCells": int(substrate.occupied.sum()),
        "edgeCount": int(substrate.active_edge_mask.sum()),
        "parameterCount": sum(parameter.numel() for parameter in experiment.model.parameters()),
        "trainableParameterCount": sum(
            parameter.numel() for parameter in experiment.model.parameters()
            if parameter.requires_grad
        ),
        "bindingDiagnostics": experiment.model.binding_memory_diagnostics(),
        "checkpoints": list(checkpoints),
    }


def run_recovery(
    output_dir: Path,
    *,
    base_steps: int = 1_200,
    recovery_steps: int = 240,
    seed: int = 11,
    device: str = "cuda",
    lesion_radius: float = 4.0,
) -> dict[str, Any]:
    """Train one base, clone it exactly, and run three matched recovery branches."""

    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.use_deterministic_algorithms(True)
    config = sequence_config(
        cell_architecture="gru",
        lifecycle_enabled=0,
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
    branches = {name: copy.deepcopy(base) for name in BRANCHES}
    center_x = (config.width - 1) / 2
    center_y = (config.height - 1) / 2
    lesion_counts = {"control": 0}
    lesion_counts["lesion_static"] = branches["lesion_static"].model.substrate.lesion(
        center_x, center_y, lesion_radius
    )
    lesion_counts["lesion_lifecycle"] = branches[
        "lesion_lifecycle"
    ].model.substrate.lesion(center_x, center_y, lesion_radius)
    if lesion_counts["lesion_static"] != lesion_counts["lesion_lifecycle"]:
        raise RuntimeError("matched lesion branches removed different cells")

    summaries: dict[str, Any] = {}
    for branch, experiment in branches.items():
        _apply_branch_config(experiment, lifecycle=branch == "lesion_lifecycle")
        checkpoints = [_checkpoint(experiment, 0, evaluation_batches=8)]
        started = time.perf_counter()
        path = output_dir / f"recovery-{branch}-seed{seed}.json"
        _write_result(
            path,
            _artifact(
                experiment, branch=branch, seed=seed, steps=recovery_steps,
                base_steps=base_steps, lesion_count=lesion_counts[branch],
                lesion_radius=lesion_radius, checkpoints=checkpoints,
                status="running", started=started,
            ),
        )
        for update in range(1, recovery_steps + 1):
            experiment.train_updates(1)
            if update % 20 == 0 or update == recovery_steps:
                checkpoints.append(_checkpoint(experiment, update, evaluation_batches=8))
                _write_result(
                    path,
                    _artifact(
                        experiment, branch=branch, seed=seed, steps=recovery_steps,
                        base_steps=base_steps, lesion_count=lesion_counts[branch],
                        lesion_radius=lesion_radius, checkpoints=checkpoints,
                        status="running", started=started,
                    ),
                )
        final = _artifact(
            experiment, branch=branch, seed=seed, steps=recovery_steps,
            base_steps=base_steps, lesion_count=lesion_counts[branch],
            lesion_radius=lesion_radius, checkpoints=checkpoints,
            status="complete", started=started,
        )
        _write_result(path, final)
        summaries[branch] = final
    return {"baseAccuracy": float(base_metrics["accuracy"]), "branches": summaries}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--base-steps", type=int, default=1_200)
    parser.add_argument("--recovery-steps", type=int, default=240)
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--lesion-radius", type=float, default=4.0)
    args = parser.parse_args()
    result = run_recovery(
        args.output_dir, base_steps=max(1, args.base_steps),
        recovery_steps=max(1, args.recovery_steps), seed=args.seed,
        device=args.device, lesion_radius=max(0.5, args.lesion_radius),
    )
    print({
        "baseAccuracy": result["baseAccuracy"],
        "branches": {
            name: payload["checkpoints"][-1] for name, payload in result["branches"].items()
        },
    })


if __name__ == "__main__":
    main()
