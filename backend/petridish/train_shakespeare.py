"""Resumable, signal-safe headless Tiny Shakespeare trainer."""

from __future__ import annotations

import argparse
from collections import deque
from dataclasses import asdict, replace
import json
import math
import os
from pathlib import Path
import random
import signal
import sys
import time
from typing import Any
import uuid

import numpy as np
import torch

from .corpus_task import load_tiny_shakespeare_task
from .lifecycle_profiles import (
    LIFECYCLE_PROFILES, apply_lifecycle_profile, resolve_lifecycle_profile,
)
from .token_corpus_task import TOKENIZER_PROFILES, load_tiny_stories_task
from .mnist_config import MnistModelConfig
from .sequence_config import sequence_config
from .sequence_experiment import SequenceExperiment
from .sequence_cells import CELL_ARCHITECTURES
from .sequence_tasks import STREAM_MODES
from .topology_profiles import (
    TOPOLOGY_PROFILES,
    resolve_topology_profile,
    topology_mutates,
)


CHECKPOINT_VERSION = 1


def _experiment_state(experiment: SequenceExperiment) -> dict[str, Any]:
    names = (
        "tick", "training_step", "seen_examples", "last_loss", "last_reward",
        "last_batch_accuracy", "test_accuracy", "recall_pair_count",
        "last_synapse_update_ratio", "last_mean_attention_entropy",
        "last_gradient_norms",
        "lifecycle_active", "lifecycle_reason", "structure_unlocked",
        "structure_unlock_reason", "best_rolling_accuracy",
        "last_accuracy_improvement_step", "last_births", "last_deaths",
        "last_death_causes", "cumulative_births", "cumulative_deaths",
        "cumulative_death_causes", "last_stuns", "last_recoveries",
        "cumulative_stuns", "cumulative_recoveries",
        "last_grown_edges", "last_pruned_edges", "cumulative_grown_edges",
        "cumulative_pruned_edges",
    )
    state = {name: getattr(experiment, name) for name in names}
    state.update(
        {
            "accuracy_history": list(experiment.accuracy_history),
            "loss_history": list(experiment.loss_history),
            "reward_history": list(experiment.reward_history),
            "stage_accuracy_history": list(experiment.stage_accuracy_history),
            "substrate_generation": experiment.model.substrate.generation,
            "substrate_lifecycle_rng": experiment.model.substrate._lifecycle_generator.get_state(),
            "stream_mode": experiment.stream_mode,
            "state_retention": experiment.state_retention,
            "state_lanes": experiment.state_lanes,
            "topology_profile": experiment.topology_profile,
            "_training_stream_positions": experiment._training_stream_positions,
            "_training_runtime_state": experiment._training_runtime_state,
            "_training_runtime_bank": experiment._training_runtime_bank,
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
    lifecycle_rng = state.pop("substrate_lifecycle_rng", None)
    if lifecycle_rng is not None:
        experiment.model.substrate._lifecycle_generator.set_state(lifecycle_rng.cpu())
    for name, value in state.items():
        setattr(experiment, name, value)


def save_checkpoint(
    path: Path,
    experiment: SequenceExperiment,
    *,
    context_length: int,
    amp_mode: str,
    organism_id: str = "untracked",
    phase_index: int = 0,
    phase_name: str = "training",
) -> None:
    """Atomically save every state needed to continue the same organism."""

    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "version": CHECKPOINT_VERSION,
        "lineage": {
            "organism_id": organism_id,
            "phase_index": phase_index,
            "phase_name": phase_name,
        },
        "configuration": asdict(experiment.config),
        "task": {
            "key": experiment.task.key,
            "context_length": context_length,
            "vocabulary": experiment.task.vocabulary,
            "amp_mode": amp_mode,
            "seed": experiment.seed,
            "stream_mode": experiment.stream_mode,
            "state_retention": experiment.state_retention,
            "state_lanes": experiment.state_lanes,
            "topology_profile": experiment.topology_profile,
            "tokenizer_profile": experiment.task.tokenizer_profile,
            "training_shard_tokens": experiment.task.training_shard_tokens,
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
    incompatible = experiment.model.load_state_dict(
        _migrate_model_state(payload["model"]), strict=False
    )
    unexpected = list(incompatible.unexpected_keys)
    if unexpected:
        raise ValueError(f"checkpoint has unexpected model keys: {unexpected[:3]}")
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


def plasticity_phase_config(
    config: MnistModelConfig,
    *,
    structure: bool,
    lifecycle: bool,
    lifecycle_profile: str,
    topology_profile: str | None = None,
) -> MnistModelConfig:
    """Change only plasticity policy while preserving organism dimensions and rules."""

    profile = resolve_lifecycle_profile(lifecycle_profile, enabled=lifecycle)
    topology = resolve_topology_profile(topology_profile, structure=structure)
    return apply_lifecycle_profile(
        replace(config, structural_enabled=int(topology_mutates(topology))), profile
    )


def reconcile_plasticity_phase_status(experiment: SequenceExperiment) -> None:
    """Derive policy status from preserved history without resetting the organism."""

    experiment._should_activate_lifecycle(experiment.training_step)
    experiment._should_unlock_structure(
        experiment.training_step, experiment.last_batch_accuracy
    )


def _migrate_model_state(state: dict[str, Any]) -> dict[str, Any]:
    """Map pre-architecture GRU keys into the shared cell-rule wrapper."""

    migrated = dict(state)
    for suffix in ("weight_ih", "weight_hh", "bias_ih", "bias_hh"):
        legacy = f"cell_rule.{suffix}"
        current = f"cell_rule.rule.{suffix}"
        if legacy in migrated and current not in migrated:
            migrated[current] = migrated.pop(legacy)
    return migrated


def _gradients_finite(experiment: SequenceExperiment) -> bool:
    return math.isfinite(experiment.last_loss) and all(
        parameter.grad is None or bool(torch.isfinite(parameter.grad).all())
        for parameter in experiment.model.parameters()
    )


@torch.no_grad()
def _scientific_metrics(experiment: SequenceExperiment) -> dict[str, Any]:
    """Measure topology, routing, pruning pressure, and cellular lifecycle state."""

    substrate = experiment.model.substrate
    config = experiment.config
    living = substrate.living_sites
    active = substrate.active_edge_mask
    safe_sources = substrate.dendrite_source.clamp_min(0)
    responsive = active & ~substrate.stunned.unsqueeze(1) & ~substrate.stunned[safe_sources]
    prune_eligible = (
        responsive
        & (substrate.edge_age >= config.edge_grace_trials)
        & (substrate.edge_utility < config.prune_utility)
    )
    diagnostics = substrate.graph_diagnostics()
    context_budget = config.message_steps * experiment.task.sequence_length
    lane_states = (
        experiment._training_runtime_bank
        if experiment.state_lanes > 1 else [experiment._training_runtime_state]
    )
    lane_ages = [state.position for state in lane_states if state is not None]
    plateau_age = experiment.training_step - experiment.last_accuracy_improvement_step
    return {
        "electricalStateTokens": (
            experiment._training_runtime_state.position
            if experiment._training_runtime_state is not None else 0
        ),
        "stateRetention": experiment.state_retention,
        "stateLanes": experiment.state_lanes,
        "topologyProfile": experiment.topology_profile,
        "minimumElectricalStateTokens": min(lane_ages, default=0),
        "maximumElectricalStateTokens": max(lane_ages, default=0),
        "generation": substrate.generation,
        "livingCells": int(living.numel()),
        "stunnedCells": int(substrate.stunned[living].sum()),
        "meanEnergy": float(substrate.energy[living].mean()),
        "meanExcitotoxicDamage": float(substrate.excitotoxic_damage[living].mean()),
        "edgeCount": int(active.sum()),
        "conductingEdgeCount": int(responsive.sum()),
        "pruneEligibleEdges": int(prune_eligible.sum()),
        "minimumOutputHops": diagnostics.minimum_output_hops,
        "medianOutputHops": diagnostics.median_output_hops,
        "reachableOutputs": diagnostics.reachable_outputs,
        "tokenReachableOutputs": diagnostics.temporally_reachable_outputs,
        "contextReachableOutputs": sum(
            hops <= context_budget for hops in diagnostics.output_hops
        ),
        "outputCount": substrate.output_count,
        "lifecycleActive": experiment.lifecycle_active,
        "lifecycleReason": experiment.lifecycle_reason,
        "lifecycleWarmupRemaining": experiment.lifecycle_warmup_remaining,
        "structureUnlocked": experiment.structure_unlocked,
        "structureUnlockReason": experiment.structure_unlock_reason,
        "structuralWarmupRemaining": experiment.structural_warmup_remaining,
        "structurePlateauRemaining": max(
            0, config.structure_plateau_trials - plateau_age
        ),
        "structuralInterval": config.structural_interval,
        "lastBirths": experiment.last_births,
        "lastDeaths": experiment.last_deaths,
        "cumulativeBirths": experiment.cumulative_births,
        "cumulativeDeaths": experiment.cumulative_deaths,
        "lastStuns": experiment.last_stuns,
        "lastRecoveries": experiment.last_recoveries,
        "cumulativeStuns": experiment.cumulative_stuns,
        "cumulativeRecoveries": experiment.cumulative_recoveries,
        "lastGrownEdges": experiment.last_grown_edges,
        "lastPrunedEdges": experiment.last_pruned_edges,
        "cumulativeGrownEdges": experiment.cumulative_grown_edges,
        "cumulativePrunedEdges": experiment.cumulative_pruned_edges,
    }


@torch.no_grad()
def _generation_diagnostics(experiment: SequenceExperiment) -> dict[str, Any]:
    """Return one fixed-prompt greedy continuation without consuming sampler state."""

    if experiment.task.encode is None or experiment.task.decode is None:
        return {}
    prompt = "Once upon a time" if experiment.task.key == "tiny_stories" else "ROMEO:"
    sample, token_ids = experiment.greedy_completion(prompt, max_tokens=16)
    special = set(experiment.task.special_token_ids)
    unknown = experiment.task.unknown_token_id
    return {
        "generationPrompt": prompt,
        "generationSample": sample,
        "generationTokenCount": len(token_ids),
        "generationUniqueTokenRatio": len(set(token_ids)) / max(1, len(token_ids)),
        "generationSpecialTokenRatio": (
            sum(token in special for token in token_ids) / max(1, len(token_ids))
        ),
        "generationUnknownTokenRatio": (
            sum(token == unknown for token in token_ids) / max(1, len(token_ids))
            if unknown is not None else 0.0
        ),
        "generationTokenIds": token_ids,
    }


def _baseline_diagnostics(experiment: SequenceExperiment) -> dict[str, Any]:
    """Return corpus baselines only when the task measured them."""

    task = experiment.task
    return {
        key: value
        for key, value in (
            ("unigramBaselineAccuracy", task.unigram_baseline_accuracy),
            ("bigramBaselineAccuracy", task.bigram_baseline_accuracy),
            ("unigramBaselineLoss", task.unigram_baseline_loss),
            ("bigramBaselineLoss", task.bigram_baseline_loss),
            ("validationUnknownTokenRate", task.validation_unknown_token_rate),
            ("trainingStreamTokens", task.training_stream_tokens),
            ("fullTrainingStreamTokens", task.full_training_stream_tokens),
            ("trainingShardTokens", task.training_shard_tokens),
        )
        if value is not None
    }


@torch.no_grad()
def _held_out_diagnostics_from_current_sampler(
    experiment: SequenceExperiment,
    batches: int,
    *,
    include_state_horizons: bool = False,
) -> dict[str, Any]:
    """Evaluate one checkpoint, including the matched electrical-state ablation."""

    if experiment.stream_mode == "continuous":
        held_out, cold_state = experiment.evaluate_state_ablation(max(1, batches))
        held_out.update(
            {
                "coldStateLoss": cold_state["loss"],
                "coldStateAccuracy": cold_state["accuracy"],
                "stateCarryAccuracyDelta": (
                    held_out["accuracy"] - cold_state["accuracy"]
                ),
                "stateCarryLossDelta": (
                    cold_state["loss"] - held_out["loss"]
                ),
            }
        )
    else:
        held_out = experiment.evaluate_metrics(max(1, batches))
    (
        graph_reference, graph_silenced, source_rotated, weight_reassigned,
        broadcast_silenced,
    ) = (
        experiment.evaluate_graph_ablation(max(1, batches))
    )
    held_out.update(
        {
            "graphReferenceLoss": graph_reference["loss"],
            "graphReferenceAccuracy": graph_reference["accuracy"],
            "graphSilencedLoss": graph_silenced["loss"],
            "graphSilencedAccuracy": graph_silenced["accuracy"],
            "graphSilencedLossDelta": (
                graph_silenced["loss"] - graph_reference["loss"]
            ),
            "graphSilencedAccuracyDelta": (
                graph_reference["accuracy"] - graph_silenced["accuracy"]
            ),
            "sourceRotatedLoss": source_rotated["loss"],
            "sourceRotatedAccuracy": source_rotated["accuracy"],
            "sourceRotatedLossDelta": (
                source_rotated["loss"] - graph_reference["loss"]
            ),
            "sourceRotatedAccuracyDelta": (
                graph_reference["accuracy"] - source_rotated["accuracy"]
            ),
            "weightReassignedLoss": weight_reassigned["loss"],
            "weightReassignedAccuracy": weight_reassigned["accuracy"],
            "weightReassignedLossDelta": (
                weight_reassigned["loss"] - graph_reference["loss"]
            ),
            "weightReassignedAccuracyDelta": (
                graph_reference["accuracy"] - weight_reassigned["accuracy"]
            ),
            "broadcastSilencedLoss": broadcast_silenced["loss"],
            "broadcastSilencedAccuracy": broadcast_silenced["accuracy"],
            "broadcastSilencedLossDelta": (
                broadcast_silenced["loss"] - graph_reference["loss"]
            ),
            "broadcastSilencedAccuracyDelta": (
                graph_reference["accuracy"] - broadcast_silenced["accuracy"]
            ),
            "broadcastAblationApplicable": experiment.config.broadcast_gain > 0,
        }
    )
    diagnostics = {
        **held_out,
        **_scientific_metrics(experiment),
        **_baseline_diagnostics(experiment),
        **_generation_diagnostics(experiment),
    }
    if include_state_horizons and experiment.stream_mode == "continuous":
        diagnostics["stateHorizon"] = experiment.evaluate_state_horizons(
            max(16, batches)
        )
    return diagnostics


@torch.no_grad()
def _held_out_diagnostics(
    experiment: SequenceExperiment,
    batches: int,
    *,
    include_state_horizons: bool = False,
    evaluation_seed: int | None = None,
) -> dict[str, Any]:
    """Evaluate a checkpoint on an optional fixed slice and restore sampler state."""

    before = experiment.eval_generator.get_state().clone()
    try:
        if evaluation_seed is not None:
            experiment.eval_generator.manual_seed(evaluation_seed)
        diagnostics = _held_out_diagnostics_from_current_sampler(
            experiment, batches, include_state_horizons=include_state_horizons
        )
    finally:
        if evaluation_seed is not None:
            experiment.eval_generator.set_state(before)
    diagnostics["evaluationSeed"] = evaluation_seed
    diagnostics["evaluationBatches"] = max(1, min(50, batches))
    diagnostics["evaluatedTokens"] = (
        max(1, min(50, batches))
        * experiment.config.batch_size
        * experiment.task.sequence_length
    )
    return diagnostics


def _fresh_config(
    task: str,
    *,
    field_size: int | None,
    batch_size: int | None,
    message_steps: int | None,
    architecture: str,
    lifecycle: bool,
    broadcast_gain: float | None = None,
    learning_rate_scale: float = 1.0,
    lifecycle_profile: str = "off",
    structure: bool = True,
    topology_profile: str | None = None,
) -> MnistModelConfig:
    """Apply launch overrides without erasing task-specific warm-up policy."""

    defaults = sequence_config(task)
    size = field_size or defaults.width
    topology = resolve_topology_profile(topology_profile, structure=structure)
    config = sequence_config(
        task,
        width=size,
        height=size,
        batch_size=batch_size or defaults.batch_size,
        message_steps=message_steps or defaults.message_steps,
        broadcast_gain=(
            defaults.broadcast_gain if broadcast_gain is None else broadcast_gain
        ),
        cell_architecture=architecture,
        lifecycle_enabled=int(lifecycle),
        structural_enabled=int(topology_mutates(topology)),
        learning_rate=defaults.learning_rate * learning_rate_scale,
        readout_learning_rate=defaults.readout_learning_rate * learning_rate_scale,
        synapse_learning_rate=defaults.synapse_learning_rate * learning_rate_scale,
    )
    profile = resolve_lifecycle_profile(lifecycle_profile, enabled=lifecycle)
    return apply_lifecycle_profile(config, profile)


def _append_metric(path: Path, record: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(record, separators=(",", ":")) + "\n")
        stream.flush()
        os.fsync(stream.fileno())


def _record_process_failure(arguments: list[str], error: BaseException) -> None:
    """Persist a bounded terminal failure even when training never reaches update one."""

    checkpoint_dir = Path("runs/shakespeare")
    if "--checkpoint-dir" in arguments:
        index = arguments.index("--checkpoint-dir") + 1
        if index < len(arguments):
            checkpoint_dir = Path(arguments[index])
    message = str(error).replace("\n", " ").strip()[:1_000]
    _append_metric(
        checkpoint_dir / "metrics.jsonl",
        {
            "type": "failure",
            "failureType": type(error).__name__,
            "failureMessage": message,
            "timestamp": time.time(),
        },
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--task", choices=("tiny_shakespeare", "tiny_stories"),
        default="tiny_shakespeare",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--field-size", type=int)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--context-length", type=int, default=64)
    parser.add_argument(
        "--vocabulary-size", type=int,
        choices=(64, 128, 256, 512, 1_024, 2_048), default=2_048,
    )
    parser.add_argument("--message-steps", type=int)
    parser.add_argument("--stream-mode", choices=STREAM_MODES, default="continuous")
    parser.add_argument("--state-retention", type=float, default=1.0)
    parser.add_argument("--state-lanes", type=int, default=1)
    parser.add_argument("--broadcast-gain", type=float)
    parser.add_argument("--architecture", choices=CELL_ARCHITECTURES, default="gru")
    parser.add_argument("--updates", type=int, default=100_000)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--learning-rate-scale", type=float, default=1.0)
    parser.add_argument("--amp", choices=("off", "bfloat16"), default="off")
    parser.add_argument(
        "--compile", dest="compile_mode",
        choices=("off", "default", "reduce-overhead", "max-autotune"), default="off",
    )
    parser.add_argument("--checkpoint-dir", type=Path, default=Path("runs/shakespeare"))
    parser.add_argument("--checkpoint-interval", type=int, default=100)
    parser.add_argument("--eval-interval", type=int, default=500)
    parser.add_argument("--eval-batches", type=int, default=4)
    parser.add_argument("--evaluation-seed", type=int, default=10_001)
    parser.add_argument("--progress-interval", type=int, default=10)
    parser.add_argument("--evaluate-only", action="store_true")
    parser.add_argument("--state-horizon-eval", action="store_true")
    parser.add_argument(
        "--tokenizer-profile", choices=TOKENIZER_PROFILES, default="wordpiece"
    )
    parser.add_argument(
        "--training-shard-tokens", type=int,
        help=(
            "repeat a deterministic training prefix without replacing the organism; "
            "zero selects the full stream and omission preserves a resumed checkpoint"
        ),
    )
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--resume-plasticity", action="store_true")
    parser.add_argument("--organism-id")
    parser.add_argument("--phase-index", type=int)
    parser.add_argument("--phase-name")
    parser.add_argument("--lifecycle", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--structure", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--topology-profile", choices=TOPOLOGY_PROFILES)
    parser.add_argument(
        "--lifecycle-profile", choices=LIFECYCLE_PROFILES, default="off"
    )
    args = parser.parse_args()
    if not 0.01 <= args.learning_rate_scale <= 1.0:
        parser.error("--learning-rate-scale must be between 0.01 and 1.0")
    if args.broadcast_gain is not None and not 0 <= args.broadcast_gain <= 2.0:
        parser.error("--broadcast-gain must be between 0 and 2")
    if not 0 <= args.state_retention <= 1:
        parser.error("--state-retention must be between 0 and 1")
    if not 1 <= args.state_lanes <= 16:
        parser.error("--state-lanes must be between 1 and 16")
    if args.phase_index is not None and args.phase_index < 0:
        parser.error("--phase-index must be non-negative")

    latest = args.checkpoint_dir / "latest.pt"
    payload: dict[str, Any] | None = None
    requested_device = torch.device(args.device)
    requested_training_shard_tokens = args.training_shard_tokens
    if args.resume and latest.exists():
        payload = load_checkpoint(latest, requested_device)
        saved_lineage = dict(payload.get("lineage", {}))
        saved_organism_id = str(saved_lineage.get("organism_id", ""))
        if args.organism_id and saved_organism_id and args.organism_id != saved_organism_id:
            raise ValueError("requested organism ID does not match checkpoint lineage")
        organism_id = args.organism_id or saved_organism_id or f"organism-{uuid.uuid4().hex}"
        phase_index = (
            args.phase_index
            if args.phase_index is not None else int(saved_lineage.get("phase_index", 0))
        )
        phase_name = args.phase_name or str(saved_lineage.get("phase_name", "training"))
        saved_task = payload["task"]
        args.context_length = int(saved_task["context_length"])
        args.seed = int(saved_task["seed"])
        args.amp = str(saved_task["amp_mode"])
        args.task = str(saved_task.get("key", "tiny_shakespeare"))
        args.stream_mode = str(saved_task.get("stream_mode", "windowed"))
        args.state_retention = float(saved_task.get("state_retention", 1.0))
        args.state_lanes = int(saved_task.get("state_lanes", 1))
        args.vocabulary_size = len(tuple(saved_task.get("vocabulary", ())))
        args.tokenizer_profile = str(
            saved_task.get("tokenizer_profile") or "wordpiece"
        )
        saved_training_shard_tokens = saved_task.get("training_shard_tokens")
        if args.resume_plasticity and requested_training_shard_tokens is not None:
            args.training_shard_tokens = (
                None
                if requested_training_shard_tokens == 0
                else requested_training_shard_tokens
            )
        else:
            args.training_shard_tokens = saved_training_shard_tokens
        config = MnistModelConfig(**payload["configuration"])
        topology_profile = resolve_topology_profile(
            saved_task.get("topology_profile"),
            structure=bool(config.structural_enabled),
        )
        if args.resume_plasticity:
            topology_profile = resolve_topology_profile(
                args.topology_profile, structure=args.structure
            )
            config = plasticity_phase_config(
                config,
                structure=args.structure,
                lifecycle=args.lifecycle,
                lifecycle_profile=args.lifecycle_profile,
                topology_profile=topology_profile,
            )
    else:
        organism_id = args.organism_id or f"organism-{uuid.uuid4().hex}"
        phase_index = args.phase_index or 0
        phase_name = args.phase_name or "initial training"
        topology_profile = resolve_topology_profile(
            args.topology_profile, structure=args.structure
        )
        config = _fresh_config(
            args.task,
            field_size=args.field_size,
            batch_size=args.batch_size,
            message_steps=args.message_steps,
            architecture=args.architecture,
            lifecycle=args.lifecycle,
            broadcast_gain=args.broadcast_gain,
            learning_rate_scale=args.learning_rate_scale,
            lifecycle_profile=args.lifecycle_profile,
            structure=args.structure,
            topology_profile=topology_profile,
        )
    task = (
        load_tiny_stories_task(
            args.context_length, args.vocabulary_size,
            tokenizer_profile=args.tokenizer_profile,
            training_shard_tokens=args.training_shard_tokens,
        )
        if args.task == "tiny_stories"
        else load_tiny_shakespeare_task(args.context_length)
    )
    if args.task == "tiny_shakespeare" and len(task.vocabulary) != 66:
        raise RuntimeError(f"expected 66 Tiny Shakespeare characters, got {len(task.vocabulary)}")
    if payload is not None and tuple(payload["task"]["vocabulary"]) != task.vocabulary:
        raise ValueError("checkpoint vocabulary does not match the cached corpus")

    experiment = SequenceExperiment(
        task, config, seed=args.seed, device=args.device, amp_mode=args.amp,
        stream_mode=args.stream_mode, state_retention=args.state_retention,
        state_lanes=args.state_lanes,
        topology_profile=topology_profile,
    )
    if payload is not None:
        restore_checkpoint(experiment, payload)
        if args.resume_plasticity:
            experiment.topology_profile = topology_profile
            reconcile_plasticity_phase_status(experiment)
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
    phase_accuracy_history: deque[float] = deque(maxlen=160)
    phase_loss_history: deque[float] = deque(maxlen=160)
    print(
        f"starting at update {experiment.training_step} on {experiment.device}; "
        f"architecture={config.cell_architecture} batch={config.batch_size} "
        f"stream={args.stream_mode} retention={args.state_retention:.3f} "
        f"lanes={args.state_lanes} topology={experiment.topology_profile} "
        f"tokenizer={experiment.task.tokenizer_profile or 'character'} "
        f"training_tokens={experiment.task.training_stream_tokens or 'full'} "
        f"organism={organism_id} phase={phase_index}:{phase_name} "
        f"amp={args.amp} compile={args.compile_mode}",
        flush=True,
    )
    if args.evaluate_only:
        if payload is None:
            parser.error("--evaluate-only requires a resumable checkpoint")
        record = {
            "type": "held_out", "update": experiment.training_step,
            "organismId": organism_id, "phaseIndex": phase_index,
            "phaseName": phase_name,
            **_held_out_diagnostics(
                experiment, args.eval_batches,
                include_state_horizons=args.state_horizon_eval,
                evaluation_seed=args.evaluation_seed,
            ),
        }
        _append_metric(metrics_path, record)
        print(json.dumps(record, separators=(",", ":")), flush=True)
        return

    while experiment.training_step < max(0, args.updates) and not stop_requested:
        if experiment.device.type == "cuda":
            torch.cuda.synchronize(experiment.device)
        update_started = time.perf_counter()
        experiment.train_updates(1)
        phase_accuracy_history.append(experiment.last_batch_accuracy)
        phase_loss_history.append(experiment.last_loss)
        if experiment.device.type == "cuda":
            torch.cuda.synchronize(experiment.device)
        update_seconds = time.perf_counter() - update_started
        interval_updates += 1
        record: dict[str, Any] = {
            "type": "train",
            "update": experiment.training_step,
            "organismId": organism_id,
            "phaseIndex": phase_index,
            "phaseName": phase_name,
            "loss": experiment.last_loss,
            "accuracy": experiment.last_batch_accuracy,
            "rollingLoss": experiment.rolling_loss,
            "rollingAccuracy": experiment.rolling_accuracy,
            "phaseRollingLoss": sum(phase_loss_history) / len(phase_loss_history),
            "phaseRollingAccuracy": (
                sum(phase_accuracy_history) / len(phase_accuracy_history)
            ),
            "streamMode": experiment.stream_mode,
            "stateRetention": experiment.state_retention,
            "stateLanes": experiment.state_lanes,
            "topologyProfile": experiment.topology_profile,
            "trainingStreamTokens": experiment.task.training_stream_tokens,
            "fullTrainingStreamTokens": experiment.task.full_training_stream_tokens,
            "trainingShardTokens": experiment.task.training_shard_tokens,
            **experiment.last_gradient_norms,
            "electricalStateTokens": (
                experiment._training_runtime_state.position
                if experiment._training_runtime_state is not None else 0
            ),
            "updateSeconds": update_seconds,
            "targetCharactersPerSecond": config.batch_size * args.context_length / update_seconds,
            "targetTokensPerSecond": config.batch_size * args.context_length / update_seconds,
            "timestamp": time.time(),
        }
        _append_metric(metrics_path, record)

        if experiment.training_step % max(1, args.eval_interval) == 0:
            _append_metric(
                metrics_path,
                {
                    "type": "held_out", "update": experiment.training_step,
                    "organismId": organism_id, "phaseIndex": phase_index,
                    "phaseName": phase_name,
                    **_held_out_diagnostics(
                        experiment, args.eval_batches,
                        evaluation_seed=args.evaluation_seed,
                    ),
                },
            )
        if experiment.training_step % max(1, args.progress_interval) == 0:
            _append_metric(
                metrics_path,
                {
                    "type": "diagnostic", "update": experiment.training_step,
                    "organismId": organism_id, "phaseIndex": phase_index,
                    "phaseName": phase_name,
                    **_scientific_metrics(experiment),
                },
            )
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
                f"tokens/s={interval_updates * config.batch_size * args.context_length / elapsed:.1f} "
                f"gpu={allocated:.2f}/{reserved:.2f}GiB finite={finite}",
                flush=True,
            )
            if not finite:
                raise FloatingPointError("non-finite loss or gradient")
            interval_started = time.perf_counter()
            interval_updates = 0
        if experiment.training_step % max(1, args.checkpoint_interval) == 0:
            save_checkpoint(
                latest, experiment, context_length=args.context_length, amp_mode=args.amp,
                organism_id=organism_id, phase_index=phase_index, phase_name=phase_name,
            )

    save_checkpoint(
        latest, experiment, context_length=args.context_length, amp_mode=args.amp,
        organism_id=organism_id, phase_index=phase_index, phase_name=phase_name,
    )
    print(
        f"stopped at update {experiment.training_step} after "
        f"{time.perf_counter() - started:.1f}s; checkpoint={latest}",
        flush=True,
    )


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        _record_process_failure(sys.argv[1:], error)
        raise
