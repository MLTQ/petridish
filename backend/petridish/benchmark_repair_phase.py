"""Exploratory finite lifecycle repair windows after a severe physical lesion."""

from __future__ import annotations

import argparse
import copy
from dataclasses import dataclass
import os
from pathlib import Path
import time
from typing import Any

import torch

from .benchmark_recovery import (
    _apply_branch_config,
    _artifact,
    _capture_global_rng,
    _checkpoint,
    _restore_global_rng,
)
from .benchmark_sequences import PROFILES, _write_result
from .sequence_config import sequence_config
from .sequence_experiment import SequenceExperiment


@dataclass(frozen=True)
class RepairWindow:
    name: str
    freeze_after: int


REPAIR_WINDOWS = (
    RepairWindow("lesion_r8_repair60_freeze", 60),
    RepairWindow("lesion_r8_repair100_freeze", 100),
    RepairWindow("lesion_r8_repair140_freeze", 140),
)


def run_repair_windows(
    output_dir: Path, *, base_steps: int = 1_200, recovery_steps: int = 240,
    seed: int = 11, device: str = "cuda", lesion_radius: float = 8,
    freeze_afters: tuple[int, ...] | None = None,
) -> dict[str, Any]:
    """Compare fixed lifecycle windows, then consolidate with mutation frozen."""

    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.use_deterministic_algorithms(True)
    config = sequence_config(
        cell_architecture="gru", lifecycle_enabled=0,
        structural_warmup_trials=max(base_steps + recovery_steps + 1, 10_000),
        **PROFILES["compact24_binding_sparse"],
    )
    windows = REPAIR_WINDOWS if freeze_afters is None else tuple(
        RepairWindow(f"lesion_r8_repair{value}_freeze", value)
        for value in freeze_afters
    )
    base = SequenceExperiment(
        "associative_recall", config, seed=seed, device=device,
        recall_pair_count=3, recall_pair_max=3,
    )
    for _ in range(base_steps):
        base.train_updates(1)
    base_metrics = base.evaluate_metrics(12)
    branches = {window.name: copy.deepcopy(base) for window in windows}
    center_x = (config.width - 1) / 2
    center_y = (config.height - 1) / 2
    lesion_counts = {
        window.name: branches[window.name].model.substrate.lesion(
            center_x, center_y, lesion_radius
        )
        for window in windows
    }
    if len(set(lesion_counts.values())) != 1:
        raise RuntimeError("repair-window branches removed different cells")
    branch_rng = _capture_global_rng(base.device)

    summaries: dict[str, Any] = {}
    for window in windows:
        experiment = branches[window.name]
        _restore_global_rng(branch_rng, experiment.device)
        _apply_branch_config(experiment, lifecycle=True, interval=8)
        checkpoints = [_checkpoint(experiment, 0, evaluation_batches=8)]
        started = time.perf_counter()
        path = output_dir / f"repair-{window.name}-seed{seed}.json"

        def publish(status: str) -> dict[str, Any]:
            payload = _artifact(
                experiment, branch=window.name, seed=seed,
                steps=recovery_steps, base_steps=base_steps,
                lesion_count=lesion_counts[window.name],
                lesion_radius=lesion_radius, checkpoints=checkpoints,
                status=status, started=started, profile="repair_phase_r8",
            )
            payload["repairWindow"] = window.freeze_after
            payload["mutationFrozen"] = experiment.training_step >= (
                base_steps + window.freeze_after
            )
            _write_result(path, payload)
            return payload

        publish("running")
        for update in range(1, recovery_steps + 1):
            experiment.train_updates(1)
            if update == window.freeze_after:
                _apply_branch_config(experiment, lifecycle=False)
            if update % 20 == 0 or update == recovery_steps:
                checkpoints.append(_checkpoint(experiment, update, evaluation_batches=8))
                publish("running")
        summaries[window.name] = publish("complete")
    return {"baseAccuracy": float(base_metrics["accuracy"]), "branches": summaries}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--base-steps", type=int, default=1_200)
    parser.add_argument("--recovery-steps", type=int, default=240)
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--lesion-radius", type=float, default=8)
    parser.add_argument(
        "--freeze-after", type=int, action="append",
        help="run only the specified repair window; may be repeated",
    )
    args = parser.parse_args()
    result = run_repair_windows(
        args.output_dir, base_steps=max(1, args.base_steps),
        recovery_steps=max(1, args.recovery_steps), seed=args.seed,
        device=args.device, lesion_radius=max(0.5, args.lesion_radius),
        freeze_afters=(
            tuple(max(1, value) for value in args.freeze_after)
            if args.freeze_after else None
        ),
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
