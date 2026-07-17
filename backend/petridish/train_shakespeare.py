"""Resumable, signal-safe headless Tiny Shakespeare trainer."""

from __future__ import annotations

import argparse
from collections import deque
from dataclasses import asdict
import json
import math
import os
from pathlib import Path
import random
import signal
import time
from typing import Any

import numpy as np
import torch

from .corpus_task import load_tiny_shakespeare_task
from .mnist_config import MnistModelConfig
from .sequence_config import sequence_config
from .sequence_experiment import SequenceExperiment


CHECKPOINT_VERSION = 1


def _experiment_state(experiment: SequenceExperiment) -> dict[str, Any]:
    names = (
        "tick", "training_step", "seen_examples", "last_loss", "last_reward",
        "last_batch_accuracy", "test_accuracy", "recall_pair_count",
        "last_synapse_update_ratio", "last_mean_attention_entropy",
        "lifecycle_active", "lifecycle_reason", "structure_unlocked",
        "structure_unlock_reason", "best_rolling_accuracy",
        "last_accuracy_improvement_step", "last_births", "last_deaths",
        "last_death_causes", "cumulative_births", "cumulative_deaths",
        "cumulative_death_causes",
    )
    state = {name: getattr(experiment, name) for name in names}
    state.update(
        {
            "accuracy_history": list(experiment.accuracy_history),
            "loss_history": list(experiment.loss_history),
            "reward_history": list(experiment.reward_history),
            "stage_accuracy_history": list(experiment.stage_accuracy_history),
            "substrate_generation": experiment.model.substrate.generation,
        }
    )
    return state


def _restore_experiment_state(
    experiment: SequenceExperiment, state: dict[str, Any]
) -> None:
    history_names = {
        "accuracy_history": 160,
        "loss_history": 160,
        "reward_history": 160,
        "stage_accuracy_history": 24,
    }
    for name, maximum in history_names.items():
        setattr(experiment, name, deque(state.pop(name, []), maxlen=maximum))
    experiment.model.substrate.generation = int(state.pop("substrate_generation", 0))
    for name, value in state.items():
        setattr(experiment, name, value)


def save_checkpoint(
    path: Path,
    experiment: SequenceExperiment,
    *,
    context_length: int,
    amp_mode: str,
) -> None:
    """Atomically save every state needed to continue the same organism."""

    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "version": CHECKPOINT_VERSION,
        "configuration": asdict(experiment.config),
        "task": {
            "key": experiment.task.key,
            "context_length": context_length,
            "vocabulary": experiment.task.vocabulary,
            "amp_mode": amp_mode,
            "seed": experiment.seed,
        },
        "model": experiment.model.state_dict(),
        "optimizer": experiment.optimizer.state_dict(),
        "experiment": _experiment_state(experiment),
        "random": {
            "training_generator": experiment.generator.get_state(),
            "evaluation_generator": experiment.eval_generator.get_state(),
            "torch_cpu": torch.get_rng_state(),
            "torch_cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
            "python": random.getstate(),
            "numpy": np.random.get_state(),
        },
    }
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)


def load_checkpoint(path: Path, device: torch.device) -> dict[str, Any]:
    """Load a trusted local trainer checkpoint in a fresh process."""

    payload = torch.load(path, map_location=device, weights_only=False)
    if payload.get("version") != CHECKPOINT_VERSION:
        raise ValueError(f"unsupported checkpoint version: {payload.get('version')}")
    return payload


def restore_checkpoint(
    experiment: SequenceExperiment, payload: dict[str, Any]
) -> None:
    experiment.model.load_state_dict(payload["model"])
    experiment.optimizer.load_state_dict(payload["optimizer"])
    _restore_experiment_state(experiment, dict(payload["experiment"]))
    rng = payload["random"]
    experiment.generator.set_state(rng["training_generator"].cpu())
    experiment.eval_generator.set_state(rng["evaluation_generator"].cpu())
    torch.set_rng_state(rng["torch_cpu"].cpu())
    if torch.cuda.is_available() and rng["torch_cuda"]:
        torch.cuda.set_rng_state_all([state.cpu() for state in rng["torch_cuda"]])
    random.setstate(rng["python"])
    np.random.set_state(rng["numpy"])


def _gradients_finite(experiment: SequenceExperiment) -> bool:
    return math.isfinite(experiment.last_loss) and all(
        parameter.grad is None or bool(torch.isfinite(parameter.grad).all())
        for parameter in experiment.model.parameters()
    )


def _append_metric(path: Path, record: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(record, separators=(",", ":")) + "\n")
        stream.flush()
        os.fsync(stream.fileno())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--field-size", type=int, default=68)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--context-length", type=int, default=64)
    parser.add_argument("--message-steps", type=int, default=2)
    parser.add_argument("--updates", type=int, default=100_000)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--amp", choices=("off", "bfloat16"), default="off")
    parser.add_argument(
        "--compile", dest="compile_mode",
        choices=("off", "default", "reduce-overhead", "max-autotune"), default="off",
    )
    parser.add_argument("--checkpoint-dir", type=Path, default=Path("runs/shakespeare"))
    parser.add_argument("--checkpoint-interval", type=int, default=100)
    parser.add_argument("--eval-interval", type=int, default=500)
    parser.add_argument("--eval-batches", type=int, default=4)
    parser.add_argument("--progress-interval", type=int, default=10)
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--lifecycle", action=argparse.BooleanOptionalAction, default=False)
    args = parser.parse_args()

    latest = args.checkpoint_dir / "latest.pt"
    payload: dict[str, Any] | None = None
    requested_device = torch.device(args.device)
    if args.resume and latest.exists():
        payload = load_checkpoint(latest, requested_device)
        saved_task = payload["task"]
        args.context_length = int(saved_task["context_length"])
        args.seed = int(saved_task["seed"])
        args.amp = str(saved_task["amp_mode"])
        config = MnistModelConfig(**payload["configuration"])
    else:
        config = sequence_config(
            "tiny_shakespeare",
            width=args.field_size,
            height=args.field_size,
            batch_size=args.batch_size,
            message_steps=args.message_steps,
            lifecycle_enabled=int(args.lifecycle),
            lifecycle_warmup_trials=5_000,
            structural_warmup_trials=5_000,
        )
    task = load_tiny_shakespeare_task(args.context_length)
    if len(task.vocabulary) != 66:
        raise RuntimeError(f"expected 66 Tiny Shakespeare characters, got {len(task.vocabulary)}")
    if payload is not None and tuple(payload["task"]["vocabulary"]) != task.vocabulary:
        raise ValueError("checkpoint vocabulary does not match the cached corpus")

    experiment = SequenceExperiment(
        task, config, seed=args.seed, device=args.device, amp_mode=args.amp
    )
    if payload is not None:
        restore_checkpoint(experiment, payload)
    if args.compile_mode != "off":
        experiment.enable_compile(args.compile_mode)

    stop_requested = False

    def request_stop(signum: int, _frame: object) -> None:
        nonlocal stop_requested
        stop_requested = True
        print(f"received signal {signum}; checkpointing after the current update", flush=True)

    signal.signal(signal.SIGINT, request_stop)
    signal.signal(signal.SIGTERM, request_stop)
    metrics_path = args.checkpoint_dir / "metrics.jsonl"
    started = time.perf_counter()
    interval_started = started
    interval_updates = 0
    print(
        f"starting at update {experiment.training_step} on {experiment.device}; "
        f"batch={config.batch_size} amp={args.amp} compile={args.compile_mode}",
        flush=True,
    )

    while experiment.training_step < max(0, args.updates) and not stop_requested:
        if experiment.device.type == "cuda":
            torch.cuda.synchronize(experiment.device)
        update_started = time.perf_counter()
        experiment.train_updates(1)
        if experiment.device.type == "cuda":
            torch.cuda.synchronize(experiment.device)
        update_seconds = time.perf_counter() - update_started
        interval_updates += 1
        record: dict[str, Any] = {
            "type": "train",
            "update": experiment.training_step,
            "loss": experiment.last_loss,
            "accuracy": experiment.last_batch_accuracy,
            "rollingLoss": experiment.rolling_loss,
            "rollingAccuracy": experiment.rolling_accuracy,
            "updateSeconds": update_seconds,
            "targetCharactersPerSecond": config.batch_size * args.context_length / update_seconds,
            "timestamp": time.time(),
        }
        _append_metric(metrics_path, record)

        if experiment.training_step % max(1, args.eval_interval) == 0:
            held_out = experiment.evaluate_metrics(max(1, args.eval_batches))
            _append_metric(
                metrics_path,
                {"type": "held_out", "update": experiment.training_step, **held_out},
            )
        if experiment.training_step % max(1, args.progress_interval) == 0:
            elapsed = time.perf_counter() - interval_started
            finite = _gradients_finite(experiment)
            allocated = (
                torch.cuda.memory_allocated(experiment.device) / 2**30
                if experiment.device.type == "cuda" else 0.0
            )
            reserved = (
                torch.cuda.memory_reserved(experiment.device) / 2**30
                if experiment.device.type == "cuda" else 0.0
            )
            print(
                f"update={experiment.training_step} loss={experiment.rolling_loss:.4f} "
                f"accuracy={experiment.rolling_accuracy:.4f} "
                f"updates/s={interval_updates / elapsed:.3f} "
                f"chars/s={interval_updates * config.batch_size * args.context_length / elapsed:.1f} "
                f"gpu={allocated:.2f}/{reserved:.2f}GiB finite={finite}",
                flush=True,
            )
            if not finite:
                raise FloatingPointError("non-finite loss or gradient")
            interval_started = time.perf_counter()
            interval_updates = 0
        if experiment.training_step % max(1, args.checkpoint_interval) == 0:
            save_checkpoint(
                latest, experiment, context_length=args.context_length, amp_mode=args.amp
            )

    save_checkpoint(
        latest, experiment, context_length=args.context_length, amp_mode=args.amp
    )
    print(
        f"stopped at update {experiment.training_step} after "
        f"{time.perf_counter() - started:.1f}s; checkpoint={latest}",
        flush=True,
    )


if __name__ == "__main__":
    main()
